from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from transformers import pipeline
import logging
import os
logging.getLogger("transformers.modeling_utils").setLevel((logging.ERROR))
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# VALID_OPERATION_LEVEL = ["visit", "event"]


class RETAINLayer(nn.Module):
    """RETAIN layer.

    Paper: Edward Choi et al. RETAIN: An Interpretable Predictive Model for
    Healthcare using Reverse Time Attention Mechanism. NIPS 2016.

    This layer is used in the RETAIN model. But it can also be used as a
    standalone layer.

    Args:
        feature_size: the hidden feature size.
        dropout: dropout rate. Default is 0.5.

    Examples:
        >>> from pyhealth.models import RETAINLayer
        >>> input = torch.randn(3, 128, 64)  # [batch size, sequence len, feature_size]
        >>> layer = RETAINLayer(64)
        >>> c = layer(input)
        >>> c.shape
        torch.Size([3, 64])
    """

    def __init__(
        self,
        feature_size: int,
        dropout: float = 0.5,
    ):
        super(RETAINLayer, self).__init__()
        self.feature_size = feature_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        self.alpha_gru = nn.GRU(feature_size, feature_size, batch_first=True)
        self.beta_gru = nn.GRU(feature_size, feature_size, batch_first=True)

        self.alpha_li = nn.Linear(feature_size, 1)
        self.beta_li = nn.Linear(feature_size, feature_size)

    @staticmethod
    def reverse_x(input, lengths):
        """Reverses the input."""
        reversed_input = input.new(input.size())
        for i, length in enumerate(lengths):
            reversed_input[i, :length] = input[i, :length].flip(dims=[0])
        return reversed_input

    def compute_alpha(self, rx, lengths):
        """Computes alpha attention."""
        rx = rnn_utils.pack_padded_sequence(
            rx, lengths, batch_first=True, enforce_sorted=False
        )
        g, _ = self.alpha_gru(rx)
        g, _ = rnn_utils.pad_packed_sequence(g, batch_first=True)
        attn_alpha = torch.softmax(self.alpha_li(g), dim=1)
        return attn_alpha

    def compute_beta(self, rx, lengths):
        """Computes beta attention."""
        rx = rnn_utils.pack_padded_sequence(
            rx, lengths, batch_first=True, enforce_sorted=False
        )
        h, _ = self.beta_gru(rx)
        h, _ = rnn_utils.pad_packed_sequence(h, batch_first=True)
        attn_beta = torch.tanh(self.beta_li(h))
        return attn_beta

    def forward(
        self,
        x: torch.tensor,
        mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, feature_size].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.

        Returns:
            c: a tensor of shape [batch size, feature_size] representing the
                context vector.
        """
        # rnn will only apply dropout between layers
        x = self.dropout_layer(x)
        batch_size = x.size(0)
        if mask is None:
            lengths = torch.full(
                size=(batch_size,), fill_value=x.size(1), dtype=torch.int64
            )
        else:
            lengths = torch.sum(mask.int(), dim=-1).cpu()
        rx = self.reverse_x(x, lengths)
        attn_alpha = self.compute_alpha(rx, lengths)
        attn_beta = self.compute_beta(rx, lengths)
        c = attn_alpha * attn_beta * x  # (patient, sequence len, feature_size)
        c = torch.sum(c, dim=1)  # (patient, feature_size)
        return c


class RETAIN(BaseModel):
    """RETAIN model.

    Paper: Edward Choi et al. RETAIN: An Interpretable Predictive Model for
    Healthcare using Reverse Time Attention Mechanism. NIPS 2016.

    Note:
        We use separate Retain layers for different feature_keys.
        Currentluy, we automatically support different input formats:
            - code based input (need to use the embedding table later)
            - float/int based value input
        We follow the current convention for the Retain model:
            - case 1. [code1, code2, code3, ...]
                - we will assume the code follows the order; our model will encode
                each code into a vector and apply Retain on the code level
            - case 2. [[code1, code2]] or [[code1, code2], [code3, code4, code5], ...]
                - we will assume the inner bracket follows the order; our model first
                use the embedding table to encode each code into a vector and then use
                average/mean pooling to get one vector for one inner bracket; then use
                Retain one the braket level
            - case 3. [[1.5, 2.0, 0.0]] or [[1.5, 2.0, 0.0], [8, 1.2, 4.5], ...]
                - this case only makes sense when each inner bracket has the same length;
                we assume each dimension has the same meaning; we run Retain directly
                on the inner bracket level, similar to case 1 after embedding table
            - case 4. [[[1.5, 2.0, 0.0]]] or [[[1.5, 2.0, 0.0], [8, 1.2, 4.5]], ...]
                - this case only makes sense when each inner bracket has the same length;
                we assume each dimension has the same meaning; we run Retain directly
                on the inner bracket level, similar to case 2 after embedding table

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        feature_keys:  list of keys in samples to use as features,
            e.g. ["conditions", "procedures"].
        label_key: key in samples to use as label (e.g., "drugs").
        mode: one of "binary", "multiclass", or "multilabel".
        embedding_dim: the embedding dimension. Default is 128.
        **kwargs: other parameters for the RETAIN layer.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        embedding_dim: int = 128,
        **kwargs,
    ):
        super(RETAIN, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        self.embedding_dim = embedding_dim

        # validate kwargs for RETAIN layer
        if "feature_size" in kwargs:
            raise ValueError("feature_size is determined by embedding_dim")

        # the key of self.feat_tokenizers only contains the code based inputs
        self.feat_tokenizers = {}
        self.label_tokenizer = self.get_label_tokenizer()
        # the key of self.embeddings only contains the code based inputs
        self.embeddings = nn.ModuleDict()
        # the key of self.linear_layers only contains the float/int based inputs
        self.linear_layers = nn.ModuleDict()

        # add feature RETAIN layers
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            # sanity check
            if input_info["type"] not in [str, float, int]:
                raise ValueError(
                    "RETAIN only supports str code, float and int as input types"
                )
            elif (input_info["type"] == str) and (input_info["dim"] not in [2, 3]):
                raise ValueError(
                    "RETAIN only supports 1-level or 2-level str code as input types"
                )
            elif (input_info["type"] in [float, int]) and (
                input_info["dim"] not in [2, 3]
            ):
                raise ValueError(
                    "RETAIN only supports 2-level or 3-level float and int as input types"
                )
            # for code based input, we need Type
            # for float/int based input, we need Type, input_dim
            self.add_feature_transform_layer(feature_key, input_info)

        self.retain = nn.ModuleDict()
        for feature_key in feature_keys:
            self.retain[feature_key] = RETAINLayer(feature_size=embedding_dim, **kwargs)

        output_size = self.get_output_size(self.label_tokenizer)
        self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)
        self.feature_extract = pipeline("feature-extraction",
                                        model="emilyalsentzer/Bio_ClinicalBERT",
                                        tokenizer="emilyalsentzer/Bio_ClinicalBERT",
                                        return_unused_kwargs=False)
    def search_icd(self,
                   data,
                   keywords,
                   result,
                   search_feature = 'procedures'):
        """
        This is the first stage searching, the aim is to mattching the patients with same procedures_ICD
        Args:
            data: all dataset
            keywords: each ICD from procedures
            result: the list which contain visit, durg, durg_onehot and icu days
        Returns: result
        """

        tmp_list = []
        for i in range(len(data)):
            if len(data[i][search_feature])>1 and len(list(set(data[i][search_feature][-1]).difference(set(data[i][search_feature][-2])))) !=0:
                if len(list(set(data[i][search_feature][-1]).difference(set(data[i][search_feature][-2])))) == len(keywords):
                    if set(keywords).issubset(set(list(set(data[i][search_feature][-1]).difference(set(data[i][search_feature][-2]))))):
                        tmp_list.append({'visit_id': data[i]["visit_id"],
                                         'drug': data[i]["drugs"],
                                         'drug_hot': self.prepare_labels([data[i]["drugs"]], self.label_tokenizer),
                                         'icu_days': data[i]["readmission"],
                                         'basic_info': data[i]["multimodal"]}
                                        )
            else:
                if len(data[i][search_feature][-1]) == len(keywords):
                    if set(keywords).issubset(set(data[i][search_feature][-1])):
                        tmp_list.append({'visit_id': data[i]["visit_id"],
                                         'drug': data[i]["drugs"],
                                         'drug_hot': self.prepare_labels([data[i]["drugs"]], self.label_tokenizer),
                                         'icu_days': data[i]["readmission"],
                                         'basic_info': data[i]["multimodal"]}
                                        )
        result.append(tmp_list)

        return result

    def search_1st_step(self,
                        data,
                        keywords_procedures,
                        keywords_conditions,
                        result):
        """
        This is the first stage searching, the aim is to mattching the patients with same procedures_ICD
        Args:
            data: all dataset
            keywords: each ICD from procedures and conditions
            result: the list which contain visit, durg, durg_onehot and icu days
        Returns: result
        """
        tmp_list = []
        for i in range(len(data)):
            if len(data[i]['conditions'][-1]) == len(keywords_conditions):
                if len(data[i]['procedures'][-1]) == len(keywords_procedures):
                    if set(keywords_conditions).issubset(set(data[i]['conditions'][-1])):
                        if set(keywords_procedures).issubset(set(data[i]['procedures'][-1])):
                            tmp_list.append({'visit_id': data[i]["visit_id"],
                                             # 'drug': data[i]["drugs"],
                                             # 'drug_hot': self.prepare_labels([data[i]["drugs"]], self.label_tokenizer),
                                             'icu_days': data[i]["readmission"],
                                             'basic_info': data[i]["multimodal"]}
                                            )
        result.append(tmp_list)

        return result

    def search_drug(self,
                    seq,
                    icu_days_list,
                    prediction_list,
                    mattching_list):
        """
        This is the 2nd stage search, it is mattching from the list of 1st stage.
        Args:
            seq: the seqence number of the drug result
            icu_days_list: the list contains icu days
            prediction_list: the drug prediction
            mattching_list: the list from 1 stage
        Returns: icu days list
        """

        vectors = [(i['drug_hot'][0]).clone().detach() for i in mattching_list[seq]]

        similarities = torch.nn.functional.cosine_similarity(torch.stack(vectors), prediction_list[seq], dim=1)
        if len(vectors) >= 3:
            top_similarities, top_indices = similarities.topk(3)
            #'''
            if top_similarities[1]<0.35:
                top_similarities, top_indices = similarities.topk(1)
            else:
                if top_similarities[2]<0.35:
                    top_similarities, top_indices = similarities.topk(2)
            #'''
        else:
            top_similarities, top_indices = similarities.topk(1)
        tmp_list = []
        for i in top_indices:
            tmp_list.extend([mattching_list[seq][i]['icu_days']])
        icu_days_list.append([sum(tmp_list) / len(tmp_list)])
        return icu_days_list

    def first_stage(self,
                    #mattching_feature: str = 'procedures',
                    **kwargs
                    ):
        # 1st stage start: icd search (visit, drug, drug_onehot, icu_days)

        mattching_feature = 'procedures'
        if mattching_feature == 'procedures':
            keywords_list = []
            for i in kwargs['procedures']:
                if len(i)>1 and len(list(set(i[-1]).difference(set(i[-2])))) !=0:
                    keywords_list.append(list(set(i[-1]).difference(set(i[-2]))))

                else:
                    keywords_list.append(i[0])
            tmp_icd_result = []
            for i in keywords_list:
                tmp_icd_result = self.search_icd(self.dataset, i, tmp_icd_result)
        elif mattching_feature == 'conditions':
            #print(mattching_feature)
            keywords_list = []
            for i in kwargs['conditions']:
                if len(i) > 1 and len(list(set(i[-1]).difference(set(i[-2])))) != 0:
                    keywords_list.append(list(set(i[-1]).difference(set(i[-2]))))

                else:
                    keywords_list.append(i[0])
            tmp_icd_result = []
            for i in keywords_list:
                tmp_icd_result = self.search_icd(self.dataset, i,
                                             tmp_icd_result,
                                             search_feature='conditions')
            keywords_list = [i[0] for i in kwargs['procedures']]
            # print (len(icd_result))

        # 1st stage end: icd search (visit, drug, drug_onehot, icu_days)

        # the init rewards
        keywords_list_conditions = [i[-1] for i in kwargs['conditions']]
        keywords_list_procedures = [i[-1] for i in kwargs['procedures']]
        tmp_init_result = []
        for i, j in zip(keywords_list_procedures, keywords_list_conditions):
            tmp_init_result = self.search_1st_step(self.dataset, i, j, tmp_init_result)
        init_icudays = [i[0]['icu_days'] for i in tmp_init_result]
        tmp_init_mean_icudays = sum(init_icudays) / len(init_icudays)
        #print (tmp_icd_result)
        return tmp_icd_result, tmp_init_mean_icudays

    def second_stage(self,
                     tmp_y_prob,
                     tmp_icd_result):
        # 2nd stage start: mattching icu days
        day_list = []

        for i in range(len(tmp_y_prob)):
            day_list = self.search_drug(i,
                                        day_list,
                                        tmp_y_prob,
                                        tmp_icd_result)
        # 2nd stage end: mattching icu days

        day_reward = [i[0] for i in day_list]
        tmp_mattching_mean_icudays = sum(day_reward) / len(day_reward)
        return tmp_mattching_mean_icudays
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        The label `kwargs[self.label_key]` is a list of labels for each patient.

        Args:
            **kwargs: keyword arguments for the model. The keys must contain
                all the feature keys and the label key.

        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the loss.
                y_prob: a tensor representing the predicted probabilities.
                y_true: a tensor representing the true labels.
        """
        icd_result, init_mean_icudays = self.first_stage(**kwargs)

        patient_emb = []
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            level, Type = input_info["dim"], input_info["type"]

            # for case 1: [code1, code2, code3, ...]
            if (level == 2) and (Type == str):
                x = self.feat_tokenizers[feature_key].batch_encode_2d(
                    kwargs[feature_key]
                )
                # (patient, event)
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, event, embedding_dim)
                x = self.embeddings[feature_key](x)
                # (patient, event)
                mask = torch.sum(x, dim=2) != 0

            # for case 2: [[code1, code2], [code3, ...], ...]
            elif (level == 3) and (Type == str):
                x = self.feat_tokenizers[feature_key].batch_encode_3d(
                    kwargs[feature_key]
                )
                # (patient, visit, event)
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, visit, event, embedding_dim)
                x = self.embeddings[feature_key](x)
                # (patient, visit, embedding_dim)
                x = torch.sum(x, dim=2)
                # (patient, visit)
                mask = torch.sum(x, dim=2) != 0

            # for case 3: [[1.5, 2.0, 0.0], ...]
            elif (level == 2) and (Type in [float, int]):
                x, mask = self.padding2d(kwargs[feature_key])
                # (patient, event, values)
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                # (patient, event, embedding_dim)
                x = self.linear_layers[feature_key](x)
                # (patient, event)
                mask = torch.tensor(mask, dtype=torch.bool, device=self.device)

            # for case 4: [[[1.5, 2.0, 0.0], [1.8, 2.4, 6.0]], ...]
            elif (level == 3) and (Type in [float, int]):
                x, mask = self.padding3d(kwargs[feature_key])
                # (patient, visit, event, values)
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                # (patient, visit, embedding_dim)
                x = self.linear_layers[feature_key](x)
                # (patient, event)
                mask = torch.tensor(mask, dtype=torch.bool, device=self.device)

            else:
                raise NotImplementedError

            x = self.retain[feature_key](x, mask)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        mattching_mean_icudays = self.second_stage(y_prob,icd_result)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "init_mean_icudays": init_mean_icudays,  # TODO
            "mattching_mean_icudays": mattching_mean_icudays
        }


if __name__ == "__main__":
    from pyhealth.datasets import SampleDataset

    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "conditions": ["cond-33", "cond-86", "cond-80"],
            "procedures": [[1.0, 2.0, 3.5, 4]],
            "label": 0,
        },
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "conditions": ["cond-33", "cond-86", "cond-80"],
            "procedures": [[5.0, 2.0, 3.5, 4]],
            "label": 1,
        },
    ]

    input_info = {
        "conditions": {"dim": 1, "type": str},
        "procedures": {"dim": 2, "type": float, "input_dim": 4},
    }

    # dataset
    dataset = SampleDataset(samples=samples, dataset_name="test")
    dataset.input_info = input_info

    # data loader
    from pyhealth.datasets import get_dataloader

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    # model
    model = RETAIN(
        dataset=dataset,
        feature_keys=["conditions", "procedures"],
        label_key="label",
        mode="binary",
    )

    # data batch
    data_batch = next(iter(train_loader))

    # try the model
    ret = model(**data_batch)
    print(ret)

    # try loss backward
    ret["loss"].backward()

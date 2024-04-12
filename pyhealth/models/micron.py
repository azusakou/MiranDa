from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.utils import get_last_visit


class MICRONLayer(nn.Module):
    """MICRON layer.

    Paper: Chaoqi Yang et al. Change Matters: Medication Change Prediction
    with Recurrent Residual Networks. IJCAI 2021.

    This layer is used in the MICRON model. But it can also be used as a
    standalone layer.

    Args:
        input_size: input feature size.
        hidden_size: hidden feature size.
        num_drugs: total number of drugs to recommend.
        lam: regularization parameter for the reconstruction loss. Default is 0.1.

    Examples:
        >>> from pyhealth.models import MICRONLayer
        >>> patient_emb = torch.randn(3, 5, 32) # [patient, visit, input_size]
        >>> drugs = torch.randint(0, 2, (3, 50)).float()
        >>> layer = MICRONLayer(32, 64, 50)
        >>> loss, y_prob = layer(patient_emb, drugs)
        >>> loss.shape
        torch.Size([])
        >>> y_prob.shape
        torch.Size([3, 50])
    """

    def __init__(
        self, input_size: int, hidden_size: int, num_drugs: int, lam: float = 0.1
    ):
        super(MICRONLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_labels = num_drugs
        self.lam = lam

        self.health_net = nn.Linear(input_size, hidden_size)
        self.prescription_net = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_drugs)

        self.bce_loss_fn = nn.BCEWithLogitsLoss()

    @staticmethod
    def compute_reconstruction_loss(
        logits: torch.tensor, logits_residual: torch.tensor, mask: torch.tensor
    ) -> torch.tensor:
        rec_loss = torch.mean(
            torch.square(
                torch.sigmoid(logits[:, 1:, :])
                - torch.sigmoid(logits[:, :-1, :] + logits_residual)
            )
            * mask[:, 1:].unsqueeze(2)
        )
        return rec_loss

    def forward(
        self,
        patient_emb: torch.tensor,
        drugs: torch.tensor,
        mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Forward propagation.

        Args:
            patient_emb: a tensor of shape [patient, visit, input_size].
            drugs: a multihot tensor of shape [patient, num_labels].
            mask: an optional tensor of shape [patient, visit] where
                1 indicates valid visits and 0 indicates invalid visits.

        Returns:
            loss: a scalar tensor representing the loss.
            y_prob: a tensor of shape [patient, num_labels] representing
                the probability of each drug.
        """
        if mask is None:
            mask = torch.ones_like(patient_emb[:, :, 0])

        # (patient, visit, hidden_size)
        health_rep = self.health_net(patient_emb)#;print (health_rep.size())
        drug_rep = self.prescription_net(health_rep)
        logits = self.fc(drug_rep)
        logits_last_visit = get_last_visit(logits, mask)
        bce_loss = self.bce_loss_fn(logits_last_visit, drugs)

        # (batch, visit-1, input_size)
        health_rep_last = health_rep[:, :-1, :]
        # (batch, visit-1, input_size)
        health_rep_cur = health_rep[:, 1:, :]
        # (batch, visit-1, input_size)
        health_rep_residual = health_rep_cur - health_rep_last
        drug_rep_residual = self.prescription_net(health_rep_residual)
        logits_residual = self.fc(drug_rep_residual)
        rec_loss = self.compute_reconstruction_loss(logits, logits_residual, mask)

        loss = bce_loss + self.lam * rec_loss
        y_prob = torch.sigmoid(logits_last_visit)

        return loss, y_prob


class MICRON(BaseModel):
    """MICRON model.

    Paper: Chaoqi Yang et al. Change Matters: Medication Change Prediction
    with Recurrent Residual Networks. IJCAI 2021.

    Note:
        This model is only for medication prediction which takes conditions
        and procedures as feature_keys, and drugs as label_key. It only operates
        on the visit level.

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        embedding_dim: the embedding dimension. Default is 128.
        hidden_dim: the hidden dimension. Default is 128.
        **kwargs: other parameters for the MICRON layer.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        **kwargs
    ):
        super(MICRON, self).__init__(
            dataset=dataset,
            feature_keys=["conditions", "procedures"],
            label_key="drugs",
            mode="multilabel",
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.feat_tokenizers = self.get_feature_tokenizers()
        self.label_tokenizer = self.get_label_tokenizer()
        self.embeddings = self.get_embedding_layers(self.feat_tokenizers, embedding_dim)

        # validate kwargs for MICRON layer
        if "input_size" in kwargs:
            raise ValueError("input_size is determined by embedding_dim")
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")
        if "num_drugs" in kwargs:
            raise ValueError("num_drugs is determined by the dataset")
        self.micron = MICRONLayer(
            input_size=embedding_dim * 2,
            hidden_size=hidden_dim,
            num_drugs=self.label_tokenizer.get_vocabulary_size(),
            **kwargs
        )

    def search_icd(self,
                   data,
                   keywords,
                   result,
                   search_feature='procedures'):
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
            if len(data[i][search_feature]) > 1 and len(
                    list(set(data[i][search_feature][-1]).difference(set(data[i][search_feature][-2])))) != 0:
                if len(list(set(data[i][search_feature][-1]).difference(set(data[i][search_feature][-2])))) == len(
                        keywords):
                    if set(keywords).issubset(
                            set(list(set(data[i][search_feature][-1]).difference(set(data[i][search_feature][-2]))))):
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
            # '''
            if top_similarities[1] < 0.35:
                top_similarities, top_indices = similarities.topk(1)
            else:
                if top_similarities[2] < 0.35:
                    top_similarities, top_indices = similarities.topk(2)
            # '''
        else:
            top_similarities, top_indices = similarities.topk(1)
        tmp_list = []
        for i in top_indices:
            tmp_list.extend([mattching_list[seq][i]['icu_days']])
        icu_days_list.append([sum(tmp_list) / len(tmp_list)])
        return icu_days_list

    def first_stage(self,
                    #mattching_feature: str = 'procedures',
                    conditions, procedures

                    ):
        # 1st stage start: icd search (visit, drug, drug_onehot, icu_days)

        mattching_feature = 'procedures'
        if mattching_feature == 'procedures':
            keywords_list = []
            for i in procedures:
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
            for i in conditions:
                if len(i) > 1 and len(list(set(i[-1]).difference(set(i[-2])))) != 0:
                    keywords_list.append(list(set(i[-1]).difference(set(i[-2]))))

                else:
                    keywords_list.append(i[0])
            tmp_icd_result = []
            for i in keywords_list:
                tmp_icd_result = self.search_icd(self.dataset, i,
                                             tmp_icd_result,
                                             search_feature='conditions')
            #keywords_list = [i[0] for i in procedures]
            # print (len(icd_result))

        # 1st stage end: icd search (visit, drug, drug_onehot, icu_days)

        # the init rewards
        keywords_list_conditions = [i[-1] for i in conditions]
        keywords_list_procedures = [i[-1] for i in procedures]
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
    def forward(
        self,
        conditions: List[List[List[str]]],
        procedures: List[List[List[str]]],
        drugs: List[List[str]],
        **kwargs
    ) -> Dict[str, torch.Tensor]:

        icd_result, init_mean_icudays = self.first_stage(conditions, procedures)

        """Forward propagation.

        Args:
            conditions: a nested list in three levels [patient, visit, condition].
            procedures: a nested list in three levels [patient, visit, procedure].
            drugs: a nested list in two levels [patient, drug].

        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the loss.
                y_prob: a tensor of shape [patient, visit, num_labels] representing
                    the probability of each drug.
                y_true: a tensor of shape [patient, visit, num_labels] representing
                    the ground truth of each drug.
        """
        conditions = self.feat_tokenizers["conditions"].batch_encode_3d(conditions)
        # (patient, visit, code)
        conditions = torch.tensor(conditions, dtype=torch.long, device=self.device)
        # (patient, visit, code, embedding_dim)
        conditions = self.embeddings["conditions"](conditions)
        # (patient, visit, embedding_dim)
        conditions = torch.sum(conditions, dim=2)

        procedures = self.feat_tokenizers["procedures"].batch_encode_3d(procedures)
        # (patient, visit, code)
        procedures = torch.tensor(procedures, dtype=torch.long, device=self.device)
        # (patient, visit, code, embedding_dim)
        procedures = self.embeddings["procedures"](procedures)
        # (patient, visit, embedding_dim)
        procedures = torch.sum(procedures, dim=2)

        # (patient, visit, embedding_dim * 2)
        patient_emb = torch.cat([conditions, procedures], dim=2)
        # (patient, visit)
        mask = torch.sum(patient_emb, dim=2) != 0
        # (patient, num_labels)
        drugs = self.prepare_labels(drugs, self.label_tokenizer)

        loss, y_prob = self.micron(patient_emb, drugs, mask)

        mattching_mean_icudays = self.second_stage(y_prob,icd_result)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": drugs,
            "init_mean_icudays": init_mean_icudays,  # TODO
            "mattching_mean_icudays": mattching_mean_icudays
        }


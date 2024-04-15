import numpy as np
import torch.optim
import warnings
import pickle
from tqdm import tqdm

from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.tasks.drug_recommendationv2 import *
from pyhealth.trainer import Trainer
from pyhealth.tokenizer import Tokenizer
from pyhealth.models.utils import batch_to_multihot

import os
import wandb
from cfg import CFG

def read_date(sdataset = 'mimic3', read_data = True):
    if sdataset == 'mimic3':
        if read_data == True:
            with open('input/drug_recommendation_mimic3_text_alive_contain_onevisit.json', 'rb') as f:
                sample_dataset = pickle.load(f)#[0]

        else:
            from pyhealth.datasets import MIMIC3Dataset
            base_dataset = MIMIC3Dataset(
                root="./mimic3",
                tables=["DIAGNOSES_ICD",
                        "PROCEDURES_ICD",
                        "PRESCRIPTIONS",
                        "LABEVENTS",
                        #"TEXT"
                        ],
                code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
                #dev=True,
                #refresh_cache=True,
            )

            sample_dataset = base_dataset.set_task(drug_recommendation_mimic3_fn)
            if CFG.track_logging:
                print(sample_dataset.samples[1])
            del base_dataset

            with open('input/drug_recommendation_mimic3.json', 'wb') as f:
                pickle.dump(sample_dataset,f)
            del sample_dataset
        if CFG.track_logging:
            sample_dataset.stat()
        #print (sample_dataset.samples[1])

    elif sdataset == 'mimic4':
        if read_data == True:
            with open('input/drug_recommendation_mimic4_text_alive_contain_onevisit.json', 'rb') as f:
                sample_dataset = pickle.load(f)#[0]
                #sample_dataset = ageint(sample_dataset)
        else:
            from pyhealth.datasets import MIMIC4Dataset
            base_dataset = MIMIC4Dataset(
                root="./mimic4/hosp",
                tables=["diagnoses_icd",
                        "procedures_icd",
                        "prescriptions",
                        #"TEXT",
                        "labevents"
                        ],
                #dev=True,
                code_mapping={"NDC": "ATC"},
                refresh_cache=True,
            )
            sample_dataset = base_dataset.set_task(drug_recommendation_mimic4_fn)
            del base_dataset

            with open('input/drug_recommendation_mimic4.json', 'wb') as f:
                pickle.dump(sample_dataset, f)
            del sample_dataset
        if CFG.track_logging:
            sample_dataset.stat()
            print (sample_dataset.samples[0])

    elif sdataset == 'eicu':

        if read_data == True:
            with open('input/drug_recommendation_eicu_alive.json', 'rb') as f:
                sample_dataset = pickle.load(f)#[0]
            sample_dataset.stat()

        else:
            from pyhealth.datasets import eICUDataset
            eicu_base = eICUDataset(
                root = "./eicu/2.0",
                tables = ["diagnosis",
                          "medication",
                          "treatment",
                          #"physicalExam",
                          "lab"
                          ],
                code_mapping = {"NDC": "ATC"},
                #dev = True,
                refresh_cache=True,
            )

            eicu_base.stat()
            eicu_base.info()

            #from pyhealth.tasks import readmission_prediction_eicu_fn
            sample_dataset = eicu_base.set_task(drug_recommendation_eicu_fn)
            if CFG.track_logging:
                print (sample_dataset.samples[0])
            del eicu_base

            with open('input/drug_recommendation_eicu.json', 'wb') as f:
                pickle.dump(sample_dataset, f)
            del sample_dataset

    return sample_dataset

class retrieve_data:
    """
    first of all, read the original json data;
    generate SQL database as db
    read the retrieved data
    """
    def __init__(self, config=CFG):
        self.data_name = config.data_name
        self.save_memory_mode = config.save_memory
        self.train_the_last_mode = config.train_the_last
        self.save_memory_del_features = config.save_memory_del_features
        self.train_the_last_last_features = config.train_the_last_last_features

    def prepare_labels(self, labels,label_tokenizer):
        mode = "multilabel"
        if mode in ["binary"]:
            labels = label_tokenizer.convert_tokens_to_indices(labels)
            labels = torch.FloatTensor(labels).unsqueeze(-1)
        elif mode in ["multiclass"]:
            labels = label_tokenizer.convert_tokens_to_indices(labels)
            labels = torch.LongTensor(labels)
        elif mode in ["multilabel"]:
            # convert to indices
            labels_index = label_tokenizer.batch_encode_2d(
                labels, padding=False, truncation=False
            )
            # convert to multihot
            num_labels = label_tokenizer.get_vocabulary_size()
            labels = batch_to_multihot(labels_index, num_labels)
        else:
            raise NotImplementedError
        labels = labels.to("cuda")
        return labels

    def search_icd(self,
                   cur,
                   keywords,
                   k_age,
                   k_visit,
                   top_n = 30):
        matching = str(sorted(set(keywords)))
        cur.execute(f"SELECT * FROM visits WHERE NOT visit_id = ? AND procedures = ? AND age BETWEEN ? AND ? ORDER BY icu_days DESC LIMIT ?",
                    (k_visit, matching, float(k_age) - 5, float(k_age) + 5, top_n,))

        rows1 = cur.fetchall()

        tmp_results = []

        for row in rows1:
            row_dict = dict(zip((
                'visit_id',
                'age',
                'gender',
                'procedures',
                'icu_days',
                'drug',
                'basic_info'), row))

            row_dict['drug'] = pickle.loads(row_dict['drug'])
            # row_dict['age'] = str(row_dict['age'])
            # row_dict['icu_days'] = str(row_dict['icu_days'])
            # k_gender == row_dict['gender'] and
            tmp_results.append(row_dict)

        cur.execute(f"SELECT * FROM visits WHERE visit_id = ?",
                    (k_visit,))

        rows2 = cur.fetchall()
        for row in rows2:
            row_dict = dict(zip((
                'visit_id',
                'age',
                'gender',
                'procedures',
                'icu_days',
                'drug',
                'basic_info'), row))

            row_dict['drug'] = pickle.loads(row_dict['drug'])
            tmp_results.append(row_dict)

        return tmp_results

    def gene_db(self, dataset):

        filename = f'./input/sqlite/{self.data_name}.db'
        if os.path.exists(filename):
            print ("read db")
        else:
            print ("write new db")
            tokenizers = Tokenizer(tokens=dataset.get_all_tokens(key="drugs"),
                                   special_tokens=[])

            conn = sqlite3.connect(filename)
            cur = conn.cursor()

            cur.execute('''CREATE TABLE visits
            (visit_id TEXT,
            age INTEGER,
            gender TEXT,
            procedures TEXT,
            icu_days INTEGER,
            drug TEXT,
            basic_info TEXT)''')

            for row in dataset:
                cur.execute(
                    "INSERT INTO visits (visit_id, age, gender, procedures, icu_days, drug, basic_info) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (row['visit_id'],
                     int(row['age'].split(".")[0]),
                     row['gender'],
                     str(sorted(set(row['procedures'][-1]))),
                     row['readmission'],
                     pickle.dumps(self.prepare_labels([row['drugs']], tokenizers)),
                     row['multimodal'][0])
                    )

            conn.commit()
            conn.close()

    def match_icd(self, sample_dataset):
        filename = f'./input/retrieved_drug_recommendation_{self.data_name}.json'
        if os.path.exists(filename):
            print(f"read retrieved {self.data_name}.json")
            del sample_dataset
            with open(filename, 'rb') as f:
                sample_dataset = pickle.load(f)#[0]
            #print (sample_dataset.get_all_tokens(key="drugs"))
            if os.path.exists(f'output/drug_token_space_{self.data_name}.json'):
                print('token space exists')
            else:
                with open(f'output/drug_token_space_{self.data_name}.json', 'wb') as f:
                    pickle.dump(sample_dataset.get_all_tokens(key="drugs"), f)

        else:
            self.gene_db(sample_dataset)
            print(f"write retrieve {self.data_name}")
            filename_db = f'./input/sqlite/{self.data_name}.db'
            conn = sqlite3.connect(filename_db)
            cur = conn.cursor()
            for i in tqdm(range(len(sample_dataset))):
                sample_dataset.samples[i]["matching"] = self.search_icd(cur,
                                                                   sample_dataset.samples[i]['procedures'][-1],
                                                                   sample_dataset.samples[i]['age'],
                                                                   sample_dataset.samples[i]['visit_id'])

            with open(f'input/retrieved_drug_recommendation_{self.data_name}.json', 'wb') as f:
                pickle.dump(sample_dataset, f)

        return sample_dataset

    def save_memory(self, sample_dataset):
        if self.data_name == "eicu":
            return sample_dataset
        for i in tqdm(range(len(sample_dataset))):
            for j in self.save_memory_del_features:
                del sample_dataset.samples[i][j]
        return sample_dataset

    def train_the_last(self, sample_dataset):
        for i in tqdm(range(len(sample_dataset))):
            for j in self.train_the_last_last_features:
                sample_dataset.samples[i][j] = [sample_dataset.samples[i][j][-1]]
        #print (sample_dataset.samples[0])
        return sample_dataset

    def preprocess(self, sample_dataset):
        sample_dataset = self.match_icd(sample_dataset)
        if self.save_memory_mode:
            print('save memory')
            sample_dataset = self.save_memory(sample_dataset)
        if self.train_the_last_mode:
            print('train the last list')
            sample_dataset = self.train_the_last(sample_dataset)
        return sample_dataset

def ageint(sample_dataset):
    for i in tqdm(range(len(sample_dataset))):
        if int(float(sample_dataset.samples[i]["age"]))<90:
            sample_dataset.samples[i]["age"] = int(float(sample_dataset.samples[i]["age"]))
        else:
            sample_dataset.samples[i]["age"] = int(90)
    return sample_dataset
class ALL_train:
    def __init__(self,sample_dataset, config, epoch):
        train_ds, val_ds, test_ds = split_by_patient(dataset=sample_dataset,
                                                     ratios=[0.7, 0.15, 0.15],
                                                     seed=epoch)
        self.sample_dataset = sample_dataset
        self.dataset_name = config.data_name

        self.train_loader = get_dataloader(train_ds, batch_size=config.batch_size, shuffle=True,drop_last=True)
        self.val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)
        self.test_loader = get_dataloader(test_ds, batch_size=32, shuffle=False)
        self.all_metrics = [
            # "roc_auc_micro",
            # "roc_auc_macro",
            # "roc_auc_weighted",
            "roc_auc_samples",
            # "pr_auc_micro",
            # "pr_auc_macro",
            # "pr_auc_weighted",
            "pr_auc_samples",
            "accuracy",
            # "f1_micro",
            # "f1_macro",
            # "f1_weighted",
            "f1_samples",
            # "precision_micro",
            # "precision_macro",
            # "precision_weighted",
            "precision_samples",
            # "recall_micro",
            # "recall_macro",
            # "recall_weighted",
            "recall_samples",
            # "jaccard_micro",
            # "jaccard_macro",
            # "jaccard_weighted",
            "jaccard_samples",
            # "hamming_loss",
            "ddi_rate"
        ]
        self.epochs = config.epochs
        self.val_start_epoch = config.val_start_epoch
        self.trace_model = print
        self.track_logging = config.track_logging
        self.DEMO = config.DEMO_mode
        self.set_seed()

        del train_ds, val_ds, test_ds, sample_dataset

    def choose_model(self, name = 'transformer'):
        if name == 'transformer':
            from pyhealth.models import Transformer
            model = Transformer(
                dataset=self.sample_dataset,
                feature_keys=["conditions",
                              "procedures",
                              "lab",
                              "multimodal"
                              ],
                # "lab", "texts", "multimodal"
                label_key="drugs",
                mode="multilabel", ),

        elif name == 'retain':
            from pyhealth.models import RETAIN
            model = RETAIN(
                dataset=self.sample_dataset,
                feature_keys=["conditions", "procedures", "lab" ],
                # "lab", "texts", "multimodal"
                label_key="drugs",
                mode="multilabel", ),

        elif name == 'safedrug':
            from pyhealth.models import SafeDrug
            model = SafeDrug(self.sample_dataset),

        elif name == 'micron':
            from pyhealth.models import MICRON
            model = MICRON(self.sample_dataset),

        elif name == 'cnn':
            from pyhealth.models import CNN
            model = CNN(dataset=self.sample_dataset,
                   feature_keys=[
                       "conditions",
                       "procedures",
                       "lab"
                   ],
                   label_key="drugs",
                   mode="multilabel", ),

        elif name == 'rnn':
            from pyhealth.models import RNN
            model = RNN(dataset=self.sample_dataset,
                   feature_keys=[
                       "conditions",
                       "procedures",
                       "lab"
                   ],
                   label_key="drugs",
                   mode="multilabel", ),

        elif name == 'grasp':
            from pyhealth.models import GRASP
            model = GRASP(dataset=self.sample_dataset,
                       feature_keys=[
                           "conditions",
                           "procedures",
                           "lab"
                           # "lab", "texts","multimodal"
                       ],
                       label_key="drugs",
                       mode="multilabel",
                       use_embedding=[True, True])
            model = [model]

        elif name == 'agent':
            from pyhealth.models import Agent
            model = Agent(dataset=self.sample_dataset,
                       feature_keys=[
                           "conditions",
                           "procedures",
                           "lab"
                           # "lab", "texts","multimodal"
                       ],
                       #static_key="multimodal",
                       label_key="drugs",
                       mode="multilabel",)
            model = [model]

        elif name == 'adacare':
            from pyhealth.models import AdaCare
            model = AdaCare(dataset=self.sample_dataset,
                       feature_keys=[
                           "conditions",
                           "procedures",
                           "lab"
                           # "lab", "texts","multimodal"
                       ],
                       #static_key="multimodal",
                       label_key="drugs",
                       mode="multilabel",
                       use_embedding=[True, True, True, True],
                       hidden_dim=64,)
            model = [model]

        elif name == 'concare':
            from pyhealth.models import ConCare
            model = ConCare(dataset=self.sample_dataset,
                       feature_keys=[
                           "conditions",
                           "procedures",
                           "lab",
                           "multimodal"
                           # "lab", "texts","multimodal"
                       ],
                       #static_key="multimodal",
                       label_key="drugs",
                       mode="multilabel",
                       use_embedding=[True, True, True, True],
                       hidden_dim=64,)
            model = [model]

        return model

    def savedata2json(self, model_name,feature_list):
        self.model_name = model_name
        self.trace_model(self.model_name)
        model = self.choose_model(self.model_name)[0]
        trainer = Trainer(model=model,
                          metrics=self.all_metrics,
                          dataset_name=self.dataset_name,
                          track_logging=self.track_logging
                          )
        flaginput = False
        if flaginput:
            trainer.load_ckpt(f"output/normal_epoch7.ckpt")
            trainer.save_data(self.val_loader, feature_list,mode='val')
            trainer.save_data(self.test_loader,feature_list,mode='test')

        flagoutput = True
        if flagoutput:
            trainer.load_ckpt(f"output/normal_epoch7.ckpt")
            trainer.save_data(self.test_loader,
                              feature_list,
                              save_input_data=False,
                              save_output_data=True,
                              mode='test')
            trainer.load_ckpt(f"output/rl_epoch5.ckpt")
            trainer.save_data(self.test_loader,
                              feature_list,
                              save_input_data=False,
                              save_output_data=True,
                              mode='test')

    def before_earlystop(self, model_name):
        self.model_name = model_name
        self.trace_model(self.model_name)
        model = self.choose_model(self.model_name)[0]
        trainer = Trainer(model=model,
                          metrics=self.all_metrics,
                          dataset_name=self.dataset_name,
                          track_logging=self.track_logging
                          )
        trainer.load_ckpt(f"output/normal_epoch0.ckpt")
        score = trainer.evaluate(self.val_loader)
        print(score)
        if self.DEMO:
            for i in range(self.val_start_epoch):
                trainer.load_ckpt(f"output/normal_epoch{i}.ckpt")
                _ = trainer.inference(self.val_loader)
        if self.DEMO:
            for i in range(self.val_start_epoch):
                trainer.load_ckpt(f"output/rl_epoch{i}.ckpt")
                _ = trainer.inference(self.val_loader)

    def train_normal(self, model_name):
        self.model_name = model_name
        self.trace_model(self.model_name)
        model = self.choose_model(self.model_name)[0]
        trainer = Trainer(model=model,
                          metrics=self.all_metrics,
                          dataset_name=self.dataset_name,
                          track_logging=self.track_logging
                          )

        trainer.train(
            train_dataloader = self.train_loader,
            val_dataloader = self.val_loader,
            #test_dataloader = self.test_loader,
            epochs=self.epochs,
            val_start_epoch=self.val_start_epoch,
            monitor_criterion = "min",
            monitor='loss',
            # optimizer_params={"lr": 1e-4},
        )

        score_normal = trainer.evaluate(self.test_loader)

        return score_normal

    def train_rl(self, model_name,
                 reinforcement_confidence: float = 1,
                 reinforcement_alpha: float = 0.8,
                 reinforcement_beta: float = 0.35,
                 loss_decay: float = 0.5):
        self.model_name = model_name
        self.trace_model(self.model_name+'_reinforcement learning')
        model = self.choose_model(self.model_name)[0]
        trainer = Trainer(model=model,
                          metrics=self.all_metrics,
                          dataset_name=self.dataset_name,
                          track_logging=self.track_logging
                          )

        trainer.train_rl_a(
            train_dataloader = self.train_loader,
            val_dataloader = self.val_loader,
            epochs=self.epochs,
            val_start_epoch=self.val_start_epoch,
            monitor_criterion="min",
            monitor='loss',
            reinforcement_confidence = reinforcement_confidence,
            reinforcement_alpha = reinforcement_alpha,
            reinforcement_beta= reinforcement_beta,
            loss_decay = loss_decay
        )

        score_reinforcement_learning = trainer.evaluate(self.test_loader)
        return score_reinforcement_learning

    def set_seed(self, seed=2023):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def transage(sample_dataset):
    for i in tqdm(range(len(sample_dataset))):
        sample_dataset.samples[i]["age"] = int(float(sample_dataset.samples[i]["age"]))
    return sample_dataset


def final_experiment():
    towandb = False
    for CFG.data_name in CFG.dataset_list:
        sample_dataset = read_date(CFG.data_name, read_data=True)
        sample_dataset = retrieve_data(CFG).preprocess(sample_dataset)

        result_list = dict()
        for trial in range(CFG.trials_for_each_model):
            alltrain = ALL_train(sample_dataset, CFG, trial)
            tmp_list = []
            resume_record = False if trial == 0 else True
            for model_seq, CFG.model_name in enumerate(CFG.model_list):

                score = alltrain.train_normal(CFG.model_name)
                tmp_list.append(score)
                if towandb:
                    with wandb.init(project=f'FINAL_EXP_{CFG.data_name}',
                                    name=f'N_ID_{0}_{CFG.data_name}_{CFG.model_name}',
                                    entity="wangzh",
                                    id=f'N_ID_{0}_{CFG.data_name}_{CFG.model_name}',
                                    resume=resume_record
                                    ):
                        wandb.log(score)
                print (score)

                score_rl = alltrain.train_rl(CFG.model_name,
                                             reinforcement_confidence = 0.9,
                                             reinforcement_alpha = 0.2,
                                             reinforcement_beta = 0.5,
                                             loss_decay = 0.2
                                             )
                tmp_list.append(score_rl)

                if towandb:
                    with wandb.init(project=f'FINAL_EXP_{CFG.data_name}',
                                    name=f'RL_ID_{0}_{CFG.data_name}_{CFG.model_name}',
                                    entity="wangzh",
                                    id=f'RL_ID_{0}_{CFG.data_name}_{CFG.model_name}',
                                    resume=resume_record
                                    ):
                        wandb.log(score_rl)
                print(score_rl)
            result_list[str(trial)] = tmp_list


        print ('Finised!')
        np.savez(f'./output/all_{CFG.data_name}.npz', **result_list)




if __name__ == '__main__':final_experiment()


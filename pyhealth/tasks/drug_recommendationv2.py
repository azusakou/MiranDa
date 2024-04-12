from pyhealth.data import Patient, Visit
import sqlite3
import pickle
stopword_list=[]#[k.strip() for k in open('./stop_words_japanese.txt',encoding='utf8').readlines() if k.strip()!='']
add_sw = ['a','for','to','is',]
stopword_list.extend(add_sw)

# TODO
def _prop(original_data,stopword_list_f = stopword_list):
    token_type = "nltk"
    remove_repeat_words = True

    if token_type == "Bert":
        from transformers import AutoTokenizer
        Tokenizer = AutoTokenizer.from_pretrained("Gaborandi/Clinical-Longformer-MLM-pubmed",
                                                  max_length=4096)
        outdata = Tokenizer.tokenize(original_data[0])

    elif token_type == "nltk":
        import nltk
        outdata = nltk.word_tokenize(original_data[0])
        outdata = [word for word in outdata if not word in stopword_list_f]

    if remove_repeat_words:
        outdata = list(set(outdata))
    return outdata


def drug_recommendation_mimic3_fn(patient: Patient):
    """Processes a single patient for the drug recommendation task.

    Drug recommendation aims at recommending a set of drugs given the patient health
    history  (e.g., conditions and procedures).

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> mimic3_base = MIMIC3Dataset(
        ...    root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        ...    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        ...    code_mapping={"ICD9CM": "CCSCM"},
        ... )
        >>> from pyhealth.tasks import drug_recommendation_mimic3_fn
        >>> mimic3_sample = mimic3_base.set_task(drug_recommendation_mimic3_fn)
        >>> mimic3_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '19', '122', '98', '663', '58', '51']], 'procedures': [['1']], 'label': [['2', '3', '4']]}]
    """
    #time_window = 15
    samples = []

    for i in range(len(patient)):
        visit: Visit = patient[i]
        #next_visit: Visit = patient[i + 1]

        # get time difference between current visit and next visit
        time_diff = (visit.encounter_time - visit.discharge_time).days
        #readmission_label = 1 if time_diff < time_window else 0
        readmission_label = int(time_diff)
        
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        #texts = visit.get_code_list(table="TEXT") # TODO
        #lab = visit.get_code_list(table="LABEVENTS") # TODO

        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]

        # TODO age!!!
        age = ((visit.encounter_time-patient.birth_datetime).days)/365

        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0 or visit.discharge_status == 1 or age < 18: # * len(texts) * len(lab)
            continue
        #texts = "No text" if len(texts) == 0 else _prop(texts)

        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,

                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                #"drugs_all": drugs,
                #"lab": lab,
                #"texts": texts, # TODO
                "readmission": readmission_label,
                "gender": str(patient.gender),
                "ethnicity": str(patient.ethnicity),
                "age":str(age),
                #"discharge_status": str(visit.discharge_status),
                "multimodal": ["Age " + str(age) + ", gender " + patient.gender +", race " + patient.ethnicity + ", patient's " + str(i+1) +" th hospitalization"] # 性别，种族，是否出院 , str(visit.discharge_status)
                }
        )
    # exclude: patients with less than 2 visit
    exclude = True
    add_history_all = True
    add_history_wo_procedures = False

    if exclude:
        if len(samples) < 2:
            return []

    for i in range(len(samples)):
        samples[i]["conditions"] = [samples[i]["conditions"]]
        samples[i]["procedures"] = [samples[i]["procedures"]]
        #samples[i]["drugs_all"] = [samples[i]["drugs_all"]]

    if len(samples) > 1:

        if add_history_all:
            #samples[0]["lab"] = [samples[0]["lab"]] # TODO
            for i in range(1, len(samples)):
                samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]
                samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]

        if add_history_wo_procedures:
            #samples[0]["lab"] = [samples[0]["lab"]] # TODO
            for i in range(1, len(samples)):
                samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]
                #samples[i]["drugs_all"] = samples[i - 1]["drugs_all"] + samples[i]["drugs_all"]

    #print (samples)
    return samples


def drug_recommendation_mimic4_fn(patient: Patient):
    """Processes a single patient for the drug recommendation task.

    Drug recommendation aims at recommending a set of drugs given the patient health
    history  (e.g., conditions and procedures).

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key

    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> mimic4_base = MIMIC4Dataset(
        ...     root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        ...     tables=["diagnoses_icd", "procedures_icd"],
        ...     code_mapping={"ICD10PROC": "CCSPROC"},
        ... )
        >>> from pyhealth.tasks import drug_recommendation_mimic4_fn
        >>> mimic4_sample = mimic4_base.set_task(drug_recommendation_mimic4_fn)
        >>> mimic4_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '19', '122', '98', '663', '58', '51']], 'procedures': [['1']], 'label': [['2', '3', '4']]}]
    """
    samples = []
    time_window = 15
    for i in range(len(patient)):
        visit: Visit = patient[i]
        #next_visit: Visit = patient[i + 1]

        # get time difference between current visit and next visit
        time_diff = (visit.encounter_time - visit.discharge_time).days
        #readmission_label = 1 if time_diff < time_window else 0
        readmission_label = int(time_diff)

        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        #texts = visit.get_code_list(table="TEXT") # TODO
        lab = visit.get_code_list(table="labevents") # TODO

        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]
        age = ((visit.encounter_time - patient.birth_datetime).days) / 365

        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) * len(lab) == 0 or visit.discharge_status==1 or age <18:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,

                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                #"drugs_all": drugs,
                "lab": lab,
                "readmission": readmission_label,
                #"texts": texts, # TODO
                #"gender": str(patient.gender),
                #"ethnicity": str(patient.ethnicity),
                #"discharge_status": str(visit.discharge_status),
                "multimodal": ["Age " + str(age) + ", gender " + patient.gender +", race " + patient.ethnicity + ", patient's " + str(i+1) +" th hospitalization"] # 性别，种族，是否出院 , str(visit.discharge_status)
            }
        )
    # exclude: patients with less than 2 visit
    exclude = True
    add_history_all = True
    add_history_wo_procedures = False

    if exclude:
        if len(samples) < 2:
            return []

    for i in range(len(samples)):
        samples[i]["conditions"] = [samples[i]["conditions"]]
        samples[i]["procedures"] = [samples[i]["procedures"]]
        #samples[i]["drugs_all"] = [samples[i]["drugs_all"]]

    if len(samples) > 1:

        if add_history_all:
            #samples[0]["lab"] = [samples[0]["lab"]] # TODO
            for i in range(1, len(samples)):
                samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]
                samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]

        if add_history_wo_procedures:
            #samples[0]["lab"] = [samples[0]["lab"]] # TODO
            for i in range(1, len(samples)):
                samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]
                #samples[i]["drugs_all"] = samples[i - 1]["drugs_all"] + samples[i]["drugs_all"]


    return samples


def drug_recommendation_eicu_fn(patient: Patient):
    """Processes a single patient for the drug recommendation task.
    Drug recommendation aims at recommending a set of drugs given the patient health
    history  (e.g., conditions and procedures).
    Args:
        patient: a Patient object
    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key
    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> eicu_base = eICUDataset(
        ...     root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        ...     tables=["diagnosis", "medication"],
        ...     code_mapping={},
        ...     dev=True
        ... )
        >>> from pyhealth.tasks import drug_recommendation_eicu_fn
        >>> eicu_sample = eicu_base.set_task(drug_recommendation_eicu_fn)
        >>> eicu_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '98', '663', '58', '51']], 'procedures': [['1']], 'label': [['2', '3', '4']]}]
    """
    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]

        # get time difference between current visit and next visit
        time_diff = (visit.encounter_time - visit.discharge_time).days
        # readmission_label = 1 if time_diff < time_window else 0
        readmission_label = int(time_diff)

        conditions = visit.get_code_list(table="diagnosis")
        procedures = visit.get_code_list(table="treatment")
        drugs = visit.get_code_list(table="medication")
        lab = visit.get_code_list(table="lab")  # TODO
        age = ((visit.encounter_time - patient.birth_datetime).days) / 365 if patient.birth_datetime != None else 0
        drugs = [drug[:4] for drug in drugs]
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0 or age < 18 or visit.discharge_status!='Alive':
            continue
        lab = ['No record'] if len(lab) == 0 else lab
        # TODO: should also exclude visit with age < 18

        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "lab": lab,
                #"drugs_all": drugs,
                "readmission": readmission_label,
                "age":str(age),
                "gender": str(patient.gender),
                "ethnicity": str(patient.ethnicity),
                "discharge_status": str(visit.discharge_status),
                "multimodal": [str("Age " + str(age) + ", gender " + str(patient.gender) +", race " + str(patient.ethnicity) + ", patient's " + str(i+1) +" th hospitalization")] # 性别，种族，是否出院 , str(visit.discharge_status)
            }
        )
    # exclude: patients with less than 2 visit
    exclude = False
    add_history_all = True
    add_history_wo_procedures = False

    if exclude:
        if len(samples) < 2:
            return []

    for i in range(len(samples)):
        samples[i]["conditions"] = [samples[i]["conditions"]]
        samples[i]["procedures"] = [samples[i]["procedures"]]
        #samples[i]["drugs_all"] = [samples[i]["drugs_all"]]

    if len(samples) > 1:

        if add_history_all:
            #samples[0]["lab"] = [samples[0]["lab"]] # TODO
            for i in range(1, len(samples)):
                samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]
                samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]

        if add_history_wo_procedures:
            #samples[0]["lab"] = [samples[0]["lab"]] # TODO
            for i in range(1, len(samples)):
                samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]
                #samples[i]["drugs_all"] = samples[i - 1]["drugs_all"] + samples[i]["drugs_all"]


    return samples


def drug_recommendation_omop_fn(patient: Patient):
    """Processes a single patient for the drug recommendation task.

    Drug recommendation aims at recommending a set of drugs given the patient health
    history  (e.g., conditions and procedures).

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key

    Examples:
        >>> from pyhealth.datasets import OMOPDataset
        >>> omop_base = OMOPDataset(
        ...     root="https://storage.googleapis.com/pyhealth/synpuf1k_omop_cdm_5.2.2",
        ...     tables=["condition_occurrence", "procedure_occurrence"],
        ...     code_mapping={},
        ... )
        >>> from pyhealth.tasks import drug_recommendation_omop_fn
        >>> omop_sample = omop_base.set_task(drug_recommendation_eicu_fn)
        >>> omop_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '98', '663', '58', '51'], ['98', '663', '58', '51']], 'procedures': [['1'], ['2', '3']], 'label': [['2', '3', '4'], ['0', '1', '4', '5']]}]
    """

    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
        conditions = visit.get_code_list(table="condition_occurrence")
        procedures = visit.get_code_list(table="procedure_occurrence")
        drugs = visit.get_code_list(table="drug_exposure")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "drugs_all": drugs,
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_all"] = [samples[0]["drugs_all"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs"] = samples[i - 1]["drugs_all"] + [samples[i]["drugs_all"]]

    return samples


if __name__ == "__main__":
    from pyhealth.datasets import MIMIC3Dataset

    base_dataset = MIMIC3Dataset(
        root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        dev=True,
        code_mapping={"ICD9CM": "CCSCM", "NDC": "ATC"},
        refresh_cache=False,
    )
    sample_dataset = base_dataset.set_task(task_fn=drug_recommendation_mimic3_fn)
    sample_dataset.stat()
    print(sample_dataset.available_keys)

    from pyhealth.datasets import MIMIC4Dataset

    base_dataset = MIMIC4Dataset(
        root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        dev=True,
        code_mapping={"NDC": "ATC"},
        refresh_cache=False,
    )
    sample_dataset = base_dataset.set_task(task_fn=drug_recommendation_mimic4_fn)
    sample_dataset.stat()
    print(sample_dataset.available_keys)

    from pyhealth.datasets import eICUDataset

    base_dataset = eICUDataset(
        root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        tables=["diagnosis", "medication", "physicalExam"],
        dev=True,
        refresh_cache=False,
    )
    sample_dataset = base_dataset.set_task(task_fn=drug_recommendation_eicu_fn)
    sample_dataset.stat()
    print(sample_dataset.available_keys)

    from pyhealth.datasets import OMOPDataset

    base_dataset = OMOPDataset(
        root="/srv/local/data/zw12/pyhealth/raw_data/synpuf1k_omop_cdm_5.2.2",
        tables=["condition_occurrence", "procedure_occurrence", "drug_exposure"],
        dev=True,
        refresh_cache=False,
    )
    sample_dataset = base_dataset.set_task(task_fn=drug_recommendation_omop_fn)
    sample_dataset.stat()
    print(sample_dataset.available_keys)

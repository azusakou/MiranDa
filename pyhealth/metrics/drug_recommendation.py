from typing import List

import numpy as np
import dill

# TODO: this metric is very ad-hoc, need to be improved


def ddi_rate_score(medications: List[np.ndarray], ddi_matrix: np.ndarray) -> float:
    """DDI rate score.

    Args:
        medications: list of medications for each patient, where each medication
            is represented by the corresponding index in the ddi matrix.
        ddi_matrix: array-like of shape (n_classes, n_classes).

    Returns:
        result: DDI rate score.
    """
    all_cnt = 0
    ddi_cnt = 0
    for sample in medications:
        for i, med_i in enumerate(sample):
            for j, med_j in enumerate(sample):
                if j <= i:
                    continue
                all_cnt += 1
                if ddi_matrix[med_i, med_j] == 1 or ddi_matrix[med_j, med_i] == 1:
                    ddi_cnt += 1
    if all_cnt == 0:
        return 0
    return ddi_cnt / all_cnt

def ddi_rate_score_w(record, path="/home/user/Documents/CODE/MIMIC/PyHealth/pyhealth/metrics/ddi.pkl"):
    # ddi rate
    ddi_A = dill.load(open(path, "rb"))
    all_cnt = 0
    dd_cnt = 0
    for patient in record:
        for adm in patient:
            med_code_set = adm
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                        dd_cnt += 1
    if all_cnt == 0:
        return 0
    return dd_cnt / all_cnt
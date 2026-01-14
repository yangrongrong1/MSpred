import pandas as pd
import numpy as np
import math

data1 = pd.read_csv("./Data/data", sep = "\t")
std = list("ACDEFGHIKLMNPQRSTVWY")


def val(AA_1, AA_2, aa, mat):
    return sum([(mat[i][aa[AA_1]] - mat[i][aa[AA_2]]) ** 2 for i in range(len(mat))]) / len(mat)

def paac_1(sequences, lambdaval, w=0.05):
    dd = []
    aa = {amino_acid: i for i, amino_acid in enumerate(std)}
    # 计算标准化值
    for i in range(0, 3):
        mean = sum(data1.iloc[i][1:]) / 20
        rr = math.sqrt(sum([(p - mean) ** 2 for p in data1.iloc[i][1:]]) / 20)
        dd.append([(p - mean) / rr for p in data1.iloc[i][1:]])
    zz = pd.DataFrame(dd)
    head = []
    for n in range(1, lambdaval + 1):
        head.append('lam_' + str(n))

    ee = []
    for sequence in sequences:
        cc = []
        for n in range(1, lambdaval + 1):
            prod_sum = sum(val(sequence[p], sequence[p + n], aa, dd) for p in range(len(sequence) - n) if
                           p + n < len(sequence) and sequence[p] in aa and sequence[p + n] in aa)
            if len(sequence) - n > 0:
                cc.append(prod_sum / (len(sequence) - n))
            else:
                cc.append(0)
        if cc:  # 如果 cc 不为空，计算伪氨基酸
            ee.append([(w * p) / (1 + w * sum(cc)) for p in cc])
        else:
            ee.append([0] * len(head))  # 如果 cc 为空，填充0
    return pd.DataFrame(ee, columns=head) if ee else pd.DataFrame(columns=head)


def aac_comp(sequences, w):
    feature_columns = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W",
                       "Y"]
    feature_list = []
    for sequence in sequences:
        composition = {amino_acid: sequence.count(amino_acid) for amino_acid in std}
        composition = [composition[amino_acid] / (1 + w * len(sequence)) for amino_acid in std]
        feature_list.append(composition)
    return pd.DataFrame(feature_list, columns=feature_columns)


def paac_wp(sequences, lambdaval=3, w=0.05):
    features_paac = paac_1(sequences, lambdaval, w)
    features_aac = aac_comp(sequences, w)
    features = pd.concat([features_aac, features_paac], axis=1)
    return features

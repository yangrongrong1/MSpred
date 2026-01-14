import pandas as pd
import numpy as np
import math
data1 = pd.read_csv("./Data/data", sep = "\t")
std = list("ACDEFGHIKLMNPQRSTVWY")


def apaac_1(sequences, lambdaval=3, w=0.05):
    dd = []
    aa = {std[i]: i for i, std_amino_acid in enumerate(std)}
    # 计算标准化值
    for i in range(0, 3):
        mean = sum(data1.iloc[i][1:]) / 20
        rr = math.sqrt(sum([(p - mean) ** 2 for p in data1.iloc[i][1:]]) / 20)
        dd.append([(p - mean) / rr for p in data1.iloc[i][1:]])
    zz = pd.DataFrame(dd)
    head = []
    for n in range(1, lambdaval + 1):
        for e in ('hydrphobicity', 'hydrophilicity', 'sidechainmass'):
            head.append('lam_' + str(n) + "_" + str(e))
    ee = []
    for sequence in sequences:
        cc = []
        for n in range(1, lambdaval + 1):
            for b in range(0, len(zz)):
                prod_sum = sum(zz.loc[b][aa.get(sequence[p], 'N/A')] * zz.loc[b][aa.get(sequence[p + n], 'N/A')]
                               for p in range(len(sequence) - n) if p + n < len(sequence) and sequence[p] in aa and sequence[p + n] in aa)
                if len(sequence) - n > 0:
                    cc.append(prod_sum / (len(sequence) - n))
                else:
                    cc.append(0)
        ee.append([(w * p) / (1 + w * sum(cc)) for p in cc])
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


def apaac_wp(sequences, lambdaval=3, w=0.05):
    Apaac = apaac_1(sequences, lambdaval, w)
    Aac = aac_comp(sequences, w)
    features = pd.concat([Aac, Apaac], axis=1)
    return features

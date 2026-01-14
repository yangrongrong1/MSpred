import pandas as pd
import numpy as np

std = list("ACDEFGHIKLMNPQRSTVWY")


def qos_wp(sequences, gap=3, w=0.05):
    if not all(seq.isupper() and all(a in std for a in seq) for seq in sequences):
        print("Error: All sequences must be uppercase and contain only standard amino acids.")
        return None

    mat1 = pd.read_csv(
        "./Data/Schneider-Wrede.csv",
        index_col='Name')
    mat2 = pd.read_csv(
        "./Data/Grantham.csv",
        index_col='Name')

    s1 = []
    s2 = []
    for seq in sequences:
        for n in range(1, gap + 1):
            sum1 = 0
            sum2 = 0
            for j in range(len(seq) - n):
                if seq[j] in mat1.index and seq[j + n] in mat1.index:
                    sum1 += (mat1.loc[seq[j], seq[j + n]]) ** 2
                if seq[j] in mat2.index and seq[j + n] in mat2.index:
                    sum2 += (mat2.loc[seq[j], seq[j + n]]) ** 2
            s1.append(sum1)
            s2.append(sum2)

    zz = pd.DataFrame(np.array(s1).reshape(len(sequences), gap))
    zz["sum"] = zz.sum(axis=1)
    zz2 = pd.DataFrame(np.array(s2).reshape(len(sequences), gap))
    zz2["sum"] = zz2.sum(axis=1)

    h1 = [f'SC_{aa}' for aa in std]
    h2 = [f'G_{aa}' for aa in std]
    h3 = [f'SC_{n}' for n in range(1, gap + 1)]
    h4 = [f'G_{n}' for n in range(1, gap + 1)]
    feature_columns = h1 + h2 + h3 + h4

    feature_list = []
    for i, seq in enumerate(sequences):
        AA = {aa: seq.count(aa) for aa in std}
        seq_features = []
        for aa in std:
            seq_features.append(AA[aa] / (1 + w * zz['sum'][i]))  # Schneider-Wrede feature
            seq_features.append(AA[aa] / (1 + w * zz2['sum'][i]))  # Grantham feature
        for k in range(gap):
            seq_features.append((w * zz[k][i]) / (1 + w * zz['sum'][i]))  # Schneider-Wrede gap feature
            seq_features.append((w * zz2[k][i]) / (1 + w * zz2['sum'][i]))  # Grantham gap feature
        feature_list.append(seq_features)

    features = pd.DataFrame(feature_list, columns=feature_columns)
    return features

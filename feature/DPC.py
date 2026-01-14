import pandas as pd
import numpy as np


std = list("ACDEFGHIKLMNPQRSTVWY")


def dpc_wp(sequences, q=1):
    # 将序列转换为大写并创建DataFrame
    df = pd.DataFrame(sequences.str.upper())
    zz = df.iloc[:, 0]
    feature_list = []
    # 计算每个氨基酸对的百分比组成
    for i in range(len(zz)):
        feature_vector = []
        for j in std:
            for k in std:
                temp = j + k
                count = sum(temp == zz[i][m3:m3+q+1].upper() for m3 in range(len(zz[i])-q+1))
                composition = (count / (len(zz[i]) - q + 1)) * 100
                feature_vector.append(composition)
        feature_list.append(feature_vector)
    # 创建特征DataFrame
    feature_columns = [f'DPC_{j}{k}' for j in std for k in std]  # 为每个氨基酸对生成列名
    features = pd.DataFrame(feature_list, columns=feature_columns)
    return features

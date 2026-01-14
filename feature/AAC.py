import pandas as pd
import numpy as np


std = list("ACDEFGHIKLMNPQRSTVWY")


def aac_wp(sequences):
    # 将序列转换为大写并创建DataFrame
    df = pd.DataFrame(sequences.str.upper())
    zz = df.iloc[:, 0]
    feature_list = []

    # 计算每个氨基酸的百分比组成
    for j in zz:
        feature_vector = []
        for i in std:
            count = j.count(i)  # 计算每个氨基酸的数量
            composition = (count / len(j)) * 100  # 计算百分比组成
            feature_vector.append(composition)
        feature_list.append(feature_vector)

    # 创建特征DataFrame
    feature_columns = [f'AAC_{a}' for a in std]  # 为每个氨基酸生成列名
    features = pd.DataFrame(feature_list, columns=feature_columns)
    return features

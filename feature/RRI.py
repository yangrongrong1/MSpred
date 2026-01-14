import pandas as pd
import numpy as np


std = list("ACDEFGHIKLMNPQRSTVWY")


def rri_wp(sequences):
    # 属性名称头
    feature_columns = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                       'Y']

    # 初始化特征列表
    feature_list = []

    for seq in sequences:
        seq_features = []
        for aa in std:
            count = 0
            cc = []
            x = 0
            for j in seq:
                if j == aa:
                    count += 1
                    cc.append(count)
                else:
                    count = 0
            while x < len(cc):
                if x + 1 < len(cc):
                    if cc[x] != cc[x + 1]:
                        if cc[x] < cc[x + 1]:
                            cc[x] = 0
                x += 1
            cc1 = [e for e in cc if e != 0]
            cc2 = [e * e for e in cc1]
            zz = sum(cc2)
            zz1 = sum(cc1)
            if zz1 != 0:
                zz2 = zz / zz1
            else:
                zz2 = 0
            seq_features.append(zz2)

        feature_list.append(seq_features)



    # 创建带"RRI_"前缀的特征名列表
    feature_names = [f"RRI_{aa}" for aa in std]

    # 将特征列表转换为DataFrame
    features = pd.DataFrame(feature_list, columns=feature_names)
    return features
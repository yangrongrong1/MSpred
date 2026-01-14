import pandas as pd
import numpy as np
from collections import Counter
import math

std = list("ACDEFGHIKLMNPQRSTVWY")


def ser_wp(peptide_list):
    # 初始化结果列表
    GH = []

    # 定义氨基酸的编码
    my_list = {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0, 'P': 0,
               'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0}
    allowed = set(('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'))

    # 遍历每个肽序列
    for seq in peptide_list:
        # 检查序列是否包含非法字符
        is_data_invalid = set(seq).issubset(allowed)
        if not is_data_invalid:
            print("Error: Please check for invalid inputs in the sequence.")
            print("Sequence contains invalid characters.")
            return pd.DataFrame()  # 返回空DataFrame

        # 计算每个氨基酸的Shannon熵
        seq = seq.upper()
        num, length = Counter(seq), len(seq)
        num = dict(sorted(num.items()))
        C = list(num.keys())
        F = list(num.values())
        for key, value in my_list.items():
            for j in range(len(C)):
                if key == C[j]:
                    my_list[key] = -round((F[j] / length) * math.log(F[j] / length, 2), 3)

        # 将结果添加到GH列表中
        GH.append(list(my_list.values()))

    # 创建DataFrame
    df_headers = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    # 创建带"SER_"前缀的特征名列表
    feature_names = [f"SER_{aa}" for aa in std]

    df = pd.DataFrame(GH, columns=feature_names)
    return df
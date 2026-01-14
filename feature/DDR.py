import pandas as pd
import numpy as np


# 定义标准氨基酸列表
std = list("ACDEFGHIKLMNPQRSTVWY")


def ddr_wp(peptide_sequences):
    # 初始化存储特征的列表
    features = []
    # 遍历每个肽序列
    for sequence in peptide_sequences:
        seq_features = []
        s = sequence.upper()  # 将序列转换为大写
        p = s[::-1]  # 反转序列
        for aa in std:
            zz = [pos for pos, char in enumerate(s) if char == aa]  # 找到氨基酸在序列中的位置
            pp = [pos for pos, char in enumerate(p) if char == aa]  # 找到氨基酸在反转序列中的位置
            ss = []
            for i in range(len(zz) - 1):
                ss.append(zz[i + 1] - zz[i] - 1)  # 计算相邻氨基酸之间的距离
            if zz:  # 如果列表不为空
                ss.insert(0, zz[0])  # 插入第一个氨基酸的位置
                ss.append(pp[0])  # 插入反转序列中第一个氨基酸的位置
            cc1 = sum([e for e in ss]) + 1  # 所有距离的和加1
            cc = sum([e * e for e in ss])  # 所有距离平方的和
            zz2 = cc / cc1 if cc1 != 0 else 0  # 计算特征值
            seq_features.append(zz2)  # 将特征值添加到序列特征列表中
        features.append(seq_features)  # 将序列特征列表添加到总特征列表中

    # 创建带"DDR_"前缀的特征名列表
    feature_names = [f"DDR_{aa}" for aa in std]
    # 创建包含特征的DataFrame
    feature_df = pd.DataFrame(features, columns=feature_names)
    return feature_df
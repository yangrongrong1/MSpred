import pandas as pd
import numpy as np
from collections import Counter
import math

std = list("ACDEFGHIKLMNPQRSTVWY")


def entropy_single(seq):
    seq = seq.upper()
    num, length = Counter(seq), len(seq)
    return -sum(freq / length * math.log(freq / length, 2) for freq in num.values())


def sep_wp(peptide_sequences):
    # 初始化一个空列表来存储每个序列的香农熵值
    shannon_entropies = []

    # 遍历每个肽序列
    for sequence in peptide_sequences:
        # 计算香农熵并添加到列表中
        shannon_entropy = entropy_single(sequence)
        shannon_entropies.append(shannon_entropy)

    # 创建一个包含香农熵值的DataFrame
    se_df = pd.DataFrame(shannon_entropies, columns=["Shannon-Entropy"])

    return se_df

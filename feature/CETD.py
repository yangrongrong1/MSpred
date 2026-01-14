import pandas as pd
import numpy as np


std = list("ACDEFGHIKLMNPQRSTVWY")



def cetd_wp(peptide_list):
    # 加载氨基酸属性分组
    attr = pd.read_csv("./Data/aa_attr_group.csv")
    # 数据预处理
    df = pd.DataFrame({'sequence': peptide_list})
    df['sequence'] = df['sequence'].str.upper()
    # 初始化特征存储
    composition_data = []
    transition_data = []
    distribution_data = []
    # 7个属性组的名称
    prop_names = ['Hydrophobicity', 'NVWV', 'Polarity', 'Polarizability', 'Charge', 'SS', 'SA']
    # 遍历每个肽序列
    for seq in df['sequence']:
        # ====================== Composition ========================
        comp_row = []
        # 遍历每个属性组
        for prop_idx in range(len(attr)):
            counts = {1: 0, 2: 0, 3: 0}
            # 统计每个类别的出现次数
            for aa in seq:
                for group in [1, 2, 3]:
                    if aa in attr.iloc[prop_idx, group]:
                        counts[group] += 1
                        break
            # 计算百分比
            total = len(seq)
            for g in [1, 2, 3]:
                comp_row.append(counts[g] / total * 100 if total != 0 else 0)

        # ====================== Transition =========================
        trans_row = []
        for prop_idx in range(len(attr)):
            transitions = []
            # 获取当前序列的类别序列
            class_seq = []
            for aa in seq:
                for g in [1, 2, 3]:
                    if aa in attr.iloc[prop_idx, g]:
                        class_seq.append(g)
                        break
            # 统计转换次数
            trans_counts = {(i, j): 0 for i in [1, 2, 3] for j in [1, 2, 3]}
            for i in range(len(class_seq) - 1):
                pair = (class_seq[i], class_seq[i + 1])
                trans_counts[pair] += 1
            # 添加特征值
            for pair in [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]:
                trans_row.append(trans_counts[pair])

        # ====================== Distribution =======================
        dist_row = []
        for prop_idx in range(len(attr)):
            class_seq = []
            for aa in seq:
                for g in [1, 2, 3]:
                    if aa in attr.iloc[prop_idx, g]:
                        class_seq.append(g)
                        break
            # 计算每个类别的分布
            for target in [1, 2, 3]:
                positions = [i for i, g in enumerate(class_seq) if g == target]
                if not positions:
                    dist_row.extend([0] * 5)
                    continue
                # 计算百分位
                percentiles = [0, 25, 50, 75, 100]
                dist_values = []
                for p in percentiles:
                    idx = int((p / 100) * (len(positions) - 1))
                    pos = positions[idx] if len(positions) > 1 else positions[0]
                    scaled_pos = pos / len(class_seq) * 100
                    dist_values.append(scaled_pos)
                dist_row.extend(dist_values)

        # 添加当前序列的特征
        composition_data.append(comp_row)
        transition_data.append(trans_row)
        distribution_data.append(dist_row)

    # 生成特征DataFrame
    comp_columns = [f"{prop}_{g}" for prop in prop_names for g in [1, 2, 3]]
    trans_columns = [f"{prop}_{i}->{j}" for prop in prop_names for i in [1, 2, 3] for j in [1, 2, 3]]
    dist_columns = [f"{prop}_{g}_{p}%" for prop in prop_names for g in [1, 2, 3] for p in [0, 25, 50, 75, 100]]

    features = pd.DataFrame(
        data=[c + t + d for c, t, d in zip(composition_data, transition_data, distribution_data)],
        columns=comp_columns + trans_columns + dist_columns
    )

    return features

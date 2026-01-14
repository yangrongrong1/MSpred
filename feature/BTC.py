import pandas as pd
import numpy as np


std = list("ACDEFGHIKLMNPQRSTVWY")


def btc_wp(peptide_sequences):
    # 初始化存储键合特征的列表
    tota = []
    hy = []
    Si = []
    Du = []
    # 读取键合数据
    bonds = pd.read_csv("./Data/bonds.csv")
    # 遍历每个肽序列
    for sequence in peptide_sequences:
        tot = 0
        h = 0
        S = 0
        D = 0
        # 遍历序列中的每个氨基酸
        for aa in sequence:
            # 查找并累加键合特征
            for index, bond in bonds.iterrows():
                if bond['Name'] == aa:
                    tot += bond['nBonds_tot']
                    h += bond['Hydrogen_bonds']
                    S += bond['nBondsS']
                    D += bond['nBondsD']
        # 将结果添加到列表中
        tota.append(tot)
        hy.append(h)
        Si.append(S)
        Du.append(D)
    # 创建包含键合特征的DataFrame
    features_df = pd.DataFrame({
        "Total_Bonds": tota,
        "Hydrogen_Bonds": hy,
        "Single_Bonds": Si,
        "Double_Bonds": Du
    })
    return features_df
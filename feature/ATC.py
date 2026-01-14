import pandas as pd
import numpy as np

std = list("ACDEFGHIKLMNPQRSTVWY")

def atc_wp(sequences):
    # 读取原子组成数据
    atom = pd.read_csv("./Data/atom.csv", header=None)
    atom[0] = atom[0].apply(lambda x: x.strip() if isinstance(x, str) else x)
    
    # 创建原子组成DataFrame
    atom['C_atom'] = atom[1].apply(lambda x: x.count('C'))
    atom['H_atom'] = atom[1].apply(lambda x: x.count('H'))
    atom['N_atom'] = atom[1].apply(lambda x: x.count('N'))
    atom['O_atom'] = atom[1].apply(lambda x: x.count('O'))
    atom['S_atom'] = atom[1].apply(lambda x: x.count('S'))
    
    # 将序列转换为大写并创建DataFrame
    df = pd.DataFrame(sequences.str.upper())
    zz = df.iloc[:, 0]
    
    # 初始化特征列表
    feature_list = []
    
    # 计算每个序列的总原子组成百分比
    for j in zz:
        # 初始化原子计数器
        total_C = 0
        total_H = 0
        total_N = 0
        total_O = 0
        total_S = 0
        
        # 计算整个序列的总原子数
        for aa in j:
            if aa in std:  # 确保是标准氨基酸
                aa_row = atom[atom[0] == aa]
                if not aa_row.empty:
                    total_C += aa_row['C_atom'].iloc[0]
                    total_H += aa_row['H_atom'].iloc[0]
                    total_N += aa_row['N_atom'].iloc[0]
                    total_O += aa_row['O_atom'].iloc[0]
                    total_S += aa_row['S_atom'].iloc[0]
        
        # 计算总原子数
        total_atoms = total_C + total_H + total_N + total_O + total_S
        
        # 避免除以零错误
        if total_atoms > 0:
            # 计算每个原子的百分比
            composition_C = (total_C / total_atoms) * 100
            composition_H = (total_H / total_atoms) * 100
            composition_N = (total_N / total_atoms) * 100
            composition_O = (total_O / total_atoms) * 100
            composition_S = (total_S / total_atoms) * 100
        else:
            composition_C = composition_H = composition_N = composition_O = composition_S = 0
        
        # 添加特征向量
        feature_list.append([composition_C, composition_H, composition_N, composition_O, composition_S])
    
    # 创建特征DataFrame
    feature_columns = ['C_perc', 'H_perc', 'N_perc', 'O_perc', 'S_perc']
    features = pd.DataFrame(feature_list, columns=feature_columns)
    
    return features
import pandas as pd
import numpy as np
import math

PCP = pd.read_csv("./Data/PhysicoChemical.csv", header=None)
std = list("ACDEFGHIKLMNPQRSTVWY")

# 属性名称头
headers = [
    'Positively charged', 'Negatively charged', 'Neutral charged', 'Polarity', 'Non polarity',
    'Aliphaticity', 'Cyclic', 'Aromaticity', 'Acidicity', 'Basicity', 'Neutral (ph)',
    'Hydrophobicity', 'Hydrophilicity', 'Neutral', 'Hydroxylic', 'Sulphur content',
    'Secondary Structure(Helix)', 'Secondary Structure(Strands)', 'Secondary Structure(Coil)',
    'Solvent Accessibilty (Buried)', 'Solvent Accesibilty(Exposed)', 'Solvent Accesibilty(Intermediate)',
    'Tiny', 'Small', 'Large', 'z1', 'z2', 'z3', 'z4', 'z5'
]


def encode(peptide):
    # letter = {'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'};
    l = len(peptide);
    encoded = np.zeros(l);
    for i in range(l):
        if (peptide[i] == 'A'):
            encoded[i] = 0;
        elif (peptide[i] == 'C'):
            encoded[i] = 1;
        elif (peptide[i] == 'D'):
            encoded[i] = 2;
        elif (peptide[i] == 'E'):
            encoded[i] = 3;
        elif (peptide[i] == 'F'):
            encoded[i] = 4;
        elif (peptide[i] == 'G'):
            encoded[i] = 5;
        elif (peptide[i] == 'H'):
            encoded[i] = 6;
        elif (peptide[i] == 'I'):
            encoded[i] = 7;
        elif (peptide[i] == 'K'):
            encoded[i] = 8;
        elif (peptide[i] == 'L'):
            encoded[i] = 9;
        elif (peptide[i] == 'M'):
            encoded[i] = 10;
        elif (peptide[i] == 'N'):
            encoded[i] = 11;
        elif (peptide[i] == 'P'):
            encoded[i] = 12;
        elif (peptide[i] == 'Q'):
            encoded[i] = 13;
        elif (peptide[i] == 'R'):
            encoded[i] = 14;
        elif (peptide[i] == 'S'):
            encoded[i] = 15;
        elif (peptide[i] == 'T'):
            encoded[i] = 16;
        elif (peptide[i] == 'V'):
            encoded[i] = 17;
        elif (peptide[i] == 'W'):
            encoded[i] = 18;
        elif (peptide[i] == 'Y'):
            encoded[i] = 19;
        else:
            print('Wrong residue!');
    return encoded;


def lookup(peptide, featureNum):
    l = len(peptide);
    peptide = list(peptide);
    out = np.zeros(l);
    peptide_num = encode(peptide);
    for i in range(l):
        out[i] = PCP[peptide_num[i]][featureNum];
    return sum(out);



def pcp(sequences):
    # 处理序列列表
    sequenceFeature = [headers]  # 包含属性名称头
    for seq in sequences:
        nfeatures = PCP.shape[0]  # 参考表中的特征数量
        sequenceFeatureTemp = []
        for j in range(nfeatures):  # 遍历每个特征
            featureVal = lookup(seq, j)
            if len(seq) != 0:
                sequenceFeatureTemp.append(round(featureVal / len(seq), 3))
            else:
                sequenceFeatureTemp.append('NaN')
        sequenceFeature.append(sequenceFeatureTemp)
    return pd.DataFrame(sequenceFeature[1:], columns=headers)  # 排除属性名称头


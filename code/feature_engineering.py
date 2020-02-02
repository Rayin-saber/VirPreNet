# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 13:42:03 2019

@author: YINR0002
"""
import os
import pandas as pd
import random
import warnings

warnings.filterwarnings('ignore')

os.chdir('/content/drive/Colab Notebooks/NTU/code5')


def data_transform(input_file, vir_class):
    raw_data = []
    HA_seq = []
    HA_label = []
    for i in range(input_file.nrows):  # print each row
        raw_data.append(input_file.row_values(i))  # read the information
    for j in range(1, input_file.nrows):
        HA_seq.append(raw_data[j][5:input_file.ncols])
    for m in range(1, input_file.nrows):
        HA_label.append(raw_data[m][3])      # 2/3 classes

    return HA_seq, HA_label


def cnn_training_data(raw_seq, raw_label, vir_class):
    # replace unambiguous with substitutions
    Btworandom = 'DN'
    Jtworandom = 'IL'
    Ztworandom = 'EQ'
    Xallrandom = 'ACDEFGHIKLMNPQRSTVWY'
    for i in range(0, len(raw_seq)):
        for j in range(0, len(raw_seq[0])):
            seq = raw_seq[i][j]
            seq = seq.replace('B', random.choice(Btworandom))
            seq = seq.replace('J', random.choice(Jtworandom))
            seq = seq.replace('Z', random.choice(Ztworandom))
            seq = seq.replace('X', random.choice(Xallrandom))
            raw_seq[i][j] = seq

        if vir_class == 2:
            if raw_label[i] == 'Avirulent':
                raw_label[i] = 0
            elif raw_label[i] == 'Virulent':
                raw_label[i] = 1
            else:
                print('error')

        elif vir_class == 3:
            if raw_label[i] == 'LOW':
                raw_label[i] = 0
            elif raw_label[i] == 'INTERMEDIATE':
                raw_label[i] = 1
            elif raw_label[i] == 'HIGH':
                raw_label[i] = 2
            else:
                print('error')

    # embedding with ProVect    
    df = pd.read_csv('protVec_100d_3grams.csv', delimiter='\t')
    trigrams = list(df['words'])
    trigram_to_idx = {trigram: i for i, trigram in enumerate(trigrams)}
    trigram_vecs = df.loc[:, df.columns != 'words'].values

    feature = []
    label = raw_label
    # overlapped feature generation for 3-grams
    for i in range(0, len(raw_seq)):
        tri_embedding = []
        strain_embedding = []
        for j in range(0, len(raw_seq[0]) - 2):
            trigram = raw_seq[i][j:j + 3]
            if trigram[0] == '-' or trigram[1] == '-' or trigram[2] == '-':
                tri_embedding = trigram_vecs[trigram_to_idx['<unk>']]
            else:
                tri_embedding = trigram_vecs[trigram_to_idx["".join(trigram)]]

            strain_embedding.append(tri_embedding)

        feature.append(strain_embedding)

    #    #non-overlapped feature generation for 3-grams  
    #    for i in range(0, len(raw_seq)):
    #        tri_embedding = []
    #        strain_embedding = []
    #        for j in range(0, len(raw_seq[0])-2, 3):
    #            #insert a gap '-' for the last two amino acids
    #            if j == len(raw_seq[0])-2:
    #                tri_embedding = trigram_vecs[trigram_to_idx['<unk>']]
    #                strain_embedding.append(tri_embedding)
    #            else:
    #                trigram = raw_seq[i][j:j+3]
    #                if trigram[0] == '-' or trigram[1] == '-' or trigram[2] == '-':
    #                    tri_embedding = trigram_vecs[trigram_to_idx['<unk>']]
    #                else:
    #                    tri_embedding = trigram_vecs[trigram_to_idx["".join(trigram)]]
    #            strain_embedding.append(tri_embedding)
    #        feature.append(strain_embedding)

    return feature, label
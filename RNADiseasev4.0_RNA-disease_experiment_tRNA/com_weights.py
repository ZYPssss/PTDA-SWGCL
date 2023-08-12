import numpy as np
import os
import random
from sklearn.preprocessing import normalize
def normal_row(data):
    #normalization by row
    data1=data.transpose()
    data1=data1/(sum(data1))
    return data1.transpose()

def import_data(k):
    file_path = os.path.dirname(__file__)
    file_name = os.path.abspath(os.path.join(file_path, '../PUL_GCN/dataset/'))
    PS_seq = np.loadtxt('./mirna_gip_similarity.csv', delimiter=',', dtype=float)
    DS_doid = np.loadtxt('./disease_gip_similarity.csv', delimiter=',', dtype=float)

    adjPD = np.loadtxt('./adjTD.csv', dtype=int, delimiter=',')
    PD_ben_ind_label = np.load('./Datasets/flod{}/TD_ben_ind_label.npy'.format(k))
    return PS_seq,DS_doid,adjPD,PD_ben_ind_label


if __name__ == "__main__":
    for j in range(5):
        k = j + 1
        PS_seq, DS_doid,adjPD,PD_ben_ind_label = import_data(k)
        file_path = os.path.dirname(__file__)
        file_name = os.path.abspath(os.path.join(file_path, '../PUL_GCN/dataset/'))

        # Assigning weights to the 4th-19th disease-related associations with RF  GBDT  SVM
        score_dis_less=np.zeros(PD_ben_ind_label.shape)

        ML_num=5
        for i in range(ML_num):
            model_name='./Datasets/flod{}/rel_GB'.format(k)+str(i+1)+'.npy'
            score_cur=np.load(model_name)
            score_dis_less[:,:]=score_dis_less[:,:] + score_cur[:,:]

        for i in range(ML_num):
            model_name='./Datasets/flod{}/rel_RF'.format(k)+str(i+1)+'.npy'
            score_cur=np.load(model_name)
            score_dis_less[:,:]=score_dis_less[:,:] + score_cur[:,:]

        for i in range(ML_num):
            model_name='./Datasets/flod{}/rel_SVM'.format(k)+str(i+1)+'.npy'
            score_cur=np.load(model_name)
            score_dis_less[:,:]=score_dis_less[:,:] + score_cur[:,:]

        Complemental_matrix=np.zeros(PD_ben_ind_label.shape)
        ben_GCN_pos=np.where(PD_ben_ind_label == 1)
        Complemental_matrix[ben_GCN_pos] = 1

        Complemental_matrix=Complemental_matrix+(score_dis_less)/15 #+score_g_norm
        Complemental_matrix=Complemental_matrix#+score_g_norm

        np.save('./Datasets/flod{}/Complemental_label_matrix(rel_GB+rel_DT+rel_RF+rel_SVM).npy'.format(k),Complemental_matrix )









    



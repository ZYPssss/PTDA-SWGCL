import numpy as np
import os
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
#from thundersvm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import time
def import_data(k):
    PS_seq = np.loadtxt('./mirna_gip_similarity.csv',delimiter=',',dtype=float)
    DS_doid = np.loadtxt('./disease_gip_similarity.csv',delimiter=',',dtype=float)

    adjPD=np.loadtxt('./adjTD.csv',dtype=int,delimiter=',')
    PD_ben_ind_label=np.load('./Datasets/flod{}/TD_ben_ind_label.npy'.format(k))
    return PS_seq,DS_doid,adjPD,PD_ben_ind_label

def select_ben_neg(adjPD,PD_ben_ind_label):
    # select negative samples in benchmark set with the same number of positive samples in benchmark set
    PD_balance=np.copy(PD_ben_ind_label)
    neg_ben_orin_loc=np.where(PD_ben_ind_label==-20)
    PD_balance[neg_ben_orin_loc]=0
    PD_balance[:,0:3]=PD_ben_ind_label[:,0:3]
    allneg = []
    for i in range(len(PD_ben_ind_label)):
        for j in range(len(PD_ben_ind_label[i])):
            if((PD_ben_ind_label[i][j]==-20) or (PD_ben_ind_label[i][j]==0)):
                allneg.append([i,j])
    neg = random.sample(allneg,len(neg_ben_orin_loc[0]))
    for i in range(len(neg)):
        PD_balance[neg[i][0]][neg[i][1]] = -20
    # for iter_col in range(adjPD.shape[1]):
    #     Neg_loc = np.where((PD_balance[:, iter_col] == 0))
    #     pos_ben_loc=np.where((PD_balance[:, iter_col] == 1))
    #     random.seed(77)  # 设置随机种子，保证结果可以复现
    #     random.shuffle(Neg_loc[0])
    #     pos_num=len(pos_ben_loc[0])
    #     print(pos_num)
    #     neg_ind=Neg_loc[0][0:pos_num]
    #     PD_balance[neg_ind,iter_col]=-20


    a=np.where((PD_balance==-10))
    b=np.where(PD_balance== -1)
    print(len(a[0]),len(b[0]))
    return PD_balance


def rel_ML1(PD_ben_ind_label_cur,PD_ben_ind_label_GCN,pi_feature,dis_feature,i):
    # find sample index in benchmark and independent set
    train_index = np.where((PD_ben_ind_label_cur==1)|(PD_ben_ind_label_cur==-20))
    test_index = np.where((PD_ben_ind_label_GCN[:,:]!=1))

    # find labels for train and test samples
    PD_ben = np.zeros(PD_ben_ind_label.shape, dtype=int)
    PD_ind = np.zeros(PD_ben_ind_label.shape, dtype=int)
    PD_ind[(PD_ben_ind_label_GCN == -1)|(PD_ben_ind_label_GCN == 1)] = 1
    PD_ben[PD_ben_ind_label_cur == 1] = 1
    train_label = PD_ben[train_index]
    test_label = PD_ind[test_index]

    # construct bencharmk set
    train_data = []
    for (cur_train_row, cur_train_col) in zip(train_index[0], train_index[1]):
        print("{},{}".format(cur_train_row,cur_train_col))
        cur_feature = pi_feature[cur_train_row].tolist()
        cur_feature.extend(dis_feature[cur_train_col].tolist())
        train_data.append(cur_feature)
    print("end1")
    # 构造test集合
    test_data = []
    print(len(test_index[0]))
    print(test_index[1])
    for (cur_test_row, cur_test_col) in zip(test_index[0], test_index[1]):
        print("{},{}".format(cur_test_row,cur_test_col))
        cur_feature = pi_feature[cur_test_row].tolist()
        cur_feature.extend(dis_feature[cur_test_col].tolist())
        test_data.append(cur_feature)
    print("end2")

    # RF
    RF_model = RandomForestClassifier(n_estimators=100, max_leaf_nodes=10, n_jobs=-1, max_features=0.2)
    RF_model.fit(train_data, train_label)

    Score_ind_RF = RF_model.predict_proba(test_data)[:, 1]
    Score_RF=np.zeros((PD_ben_ind_label_GCN.shape))
    Score_RF[test_index]=Score_ind_RF

    # GBDT
    # GBDT_model = GradientBoostingClassifier()
    # GBDT_model.fit(train_data, train_label)
    #
    # Score_ind_GB = GBDT_model.predict_proba(test_data)[:, 1]
    # Score_GB = np.zeros((PD_ben_ind_label_GCN.shape))
    # Score_GB[test_index] = Score_ind_GB

    #SVM
    # SVM_model = svm.SVC(kernel='rbf', gamma=20)
    # SVM_model.fit(train_data, train_label)
    #
    # Score_ind_SVM = SVM_model.predict_proba(test_data)[:, 1]
    # Score_SVM = np.zeros((PD_ben_ind_label_GCN.shape))
    # Score_SVM[test_index] = Score_ind_SVM

    #将对unknown的打分存起来
    Score=np.zeros((PD_ben_ind_label_GCN.shape))
    Score[test_index]=Score_ind_RF

    return Score

def rel_ML2(PD_ben_ind_label_cur,PD_ben_ind_label_GCN,pi_feature,dis_feature,i):
    # find sample index in benchmark and independent set
    train_index = np.where((PD_ben_ind_label_cur==1)|(PD_ben_ind_label_cur==-20))
    test_index = np.where((PD_ben_ind_label_GCN[:,:]!=1))

    # find labels for train and test samples
    PD_ben = np.zeros(PD_ben_ind_label.shape, dtype=int)
    PD_ind = np.zeros(PD_ben_ind_label.shape, dtype=int)
    PD_ind[(PD_ben_ind_label_GCN == -1)|(PD_ben_ind_label_GCN == 1)] = 1
    PD_ben[PD_ben_ind_label_cur == 1] = 1
    train_label = PD_ben[train_index]
    test_label = PD_ind[test_index]

    # construct bencharmk set
    train_data = []
    for (cur_train_row, cur_train_col) in zip(train_index[0], train_index[1]):
        print("{},{}".format(cur_train_row,cur_train_col))
        cur_feature = pi_feature[cur_train_row].tolist()
        cur_feature.extend(dis_feature[cur_train_col].tolist())
        train_data.append(cur_feature)
    print("end1")
    # 构造test集合
    test_data = []
    print(len(test_index[0]))
    print(test_index[1])
    for (cur_test_row, cur_test_col) in zip(test_index[0], test_index[1]):
        print("{},{}".format(cur_test_row,cur_test_col))
        cur_feature = pi_feature[cur_test_row].tolist()
        cur_feature.extend(dis_feature[cur_test_col].tolist())
        test_data.append(cur_feature)
    print("end2")

    # RF
    # RF_model = RandomForestClassifier(n_estimators=100, max_leaf_nodes=10, n_jobs=-1, max_features=0.2)
    # RF_model.fit(train_data, train_label)
    #
    # Score_ind_RF = RF_model.predict_proba(test_data)[:, 1]
    # Score_RF=np.zeros((PD_ben_ind_label_GCN.shape))
    # Score_RF[test_index]=Score_ind_RF

    #GBDT
    GBDT_model = GradientBoostingClassifier()
    GBDT_model.fit(train_data, train_label)

    Score_ind_GB = GBDT_model.predict_proba(test_data)[:, 1]
    Score_GB = np.zeros((PD_ben_ind_label_GCN.shape))
    Score_GB[test_index] = Score_ind_GB

    #SVM
    # SVM_model = svm.SVC(kernel='rbf', gamma=20)
    # SVM_model.fit(train_data, train_label)
    #
    # Score_ind_SVM = SVM_model.predict_proba(test_data)[:, 1]
    # Score_SVM = np.zeros((PD_ben_ind_label_GCN.shape))
    # Score_SVM[test_index] = Score_ind_SVM

    #将对unknown的打分存起来
    Score=np.zeros((PD_ben_ind_label_GCN.shape))
    Score[test_index]=Score_ind_GB

    return Score

def rel_ML3(PD_ben_ind_label_cur,PD_ben_ind_label_GCN,pi_feature,dis_feature,i):
    # find sample index in benchmark and independent set
    train_index = np.where((PD_ben_ind_label_cur==1)|(PD_ben_ind_label_cur==-20))
    test_index = np.where((PD_ben_ind_label_GCN[:,:]!=1))

    # find labels for train and test samples
    PD_ben = np.zeros(PD_ben_ind_label.shape, dtype=int)
    PD_ind = np.zeros(PD_ben_ind_label.shape, dtype=int)
    PD_ind[(PD_ben_ind_label_GCN == -1)|(PD_ben_ind_label_GCN == 1)] = 1
    PD_ben[PD_ben_ind_label_cur == 1] = 1
    train_label = PD_ben[train_index]
    test_label = PD_ind[test_index]

    # construct bencharmk set
    train_data = []
    for (cur_train_row, cur_train_col) in zip(train_index[0], train_index[1]):
        print("{},{}".format(cur_train_row,cur_train_col))
        cur_feature = pi_feature[cur_train_row].tolist()
        cur_feature.extend(dis_feature[cur_train_col].tolist())
        train_data.append(cur_feature)
    print("end1")
    # 构造test集合
    test_data = []
    print(len(test_index[0]))
    print(test_index[1])
    for (cur_test_row, cur_test_col) in zip(test_index[0], test_index[1]):
        print("{},{}".format(cur_test_row,cur_test_col))
        cur_feature = pi_feature[cur_test_row].tolist()
        cur_feature.extend(dis_feature[cur_test_col].tolist())
        test_data.append(cur_feature)
    print("end2")

    # RF
    # RF_model = RandomForestClassifier(n_estimators=100, max_leaf_nodes=10, n_jobs=-1, max_features=0.2)
    # RF_model.fit(train_data, train_label)
    #
    # Score_ind_RF = RF_model.predict_proba(test_data)[:, 1]
    # Score_RF=np.zeros((PD_ben_ind_label_GCN.shape))
    # Score_RF[test_index]=Score_ind_RF

    # GBDT
    # GBDT_model = GradientBoostingClassifier()
    # GBDT_model.fit(train_data, train_label)
    #
    # Score_ind_GB = GBDT_model.predict_proba(test_data)[:, 1]
    # Score_GB = np.zeros((PD_ben_ind_label_GCN.shape))
    # Score_GB[test_index] = Score_ind_GB

    #SVM
    SVM_model = svm.SVC(kernel='rbf', gamma=20,probability=True,max_iter=1000)
    SVM_model.fit(train_data, train_label)

    Score_ind_SVM = SVM_model.predict_proba(test_data)[:, 1]
    Score_SVM = np.zeros((PD_ben_ind_label_GCN.shape))
    Score_SVM[test_index] = Score_ind_SVM

    #将对unknown的打分存起来
    Score=np.zeros((PD_ben_ind_label_GCN.shape))
    Score[test_index]=Score_ind_SVM
    return Score



if __name__ == "__main__":
    for j in range(5):
        k = j + 1
        PS_seq, DS_doid, adjPD,PD_ben_ind_label = import_data(k)

        #the number of each base predictor
        ML_num=5
        # for i in range(ML_num):
        #
        #     PD_ben_ind_label_cur=select_ben_neg(adjPD,PD_ben_ind_label)
        #
        #     start_time=time.clock()
        #     rel_RF=rel_ML1(PD_ben_ind_label_cur,PD_ben_ind_label,PS_seq,DS_doid,i)
        #     end_time=time.clock()
        #
        #     print('Running time: %s Seconds'%(end_time-start_time))
        #
        #     file_path = os.path.dirname(__file__)
        #     file_name = os.path.abspath(os.path.join(file_path, './PUL_GCN/'))
        #     np.save('./Datasets/flod{}/rel_RF'.format(k)+str(i+1)+'.npy',rel_RF)
        #
        # for i in range(ML_num):
        #     PD_ben_ind_label_cur = select_ben_neg(adjPD, PD_ben_ind_label)
        #
        #     start_time = time.clock()
        #     rel_RF = rel_ML2(PD_ben_ind_label_cur, PD_ben_ind_label, PS_seq, DS_doid, i)
        #     end_time = time.clock()
        #
        #     print('Running time: %s Seconds' % (end_time - start_time))
        #
        #     file_path = os.path.dirname(__file__)
        #     file_name = os.path.abspath(os.path.join(file_path, './PUL_GCN/'))
        #     np.save('./Datasets/flod{}/rel_GB'.format(k) + str(i + 1) + '.npy', rel_RF)
        #
        # for i in range(ML_num):
        #     PD_ben_ind_label_cur = select_ben_neg(adjPD, PD_ben_ind_label)
        #
        #     start_time = time.clock()
        #     rel_RF = rel_ML3(PD_ben_ind_label_cur, PD_ben_ind_label, PS_seq, DS_doid, i)
        #     end_time = time.clock()
        #
        #     print('Running time: %s Seconds' % (end_time - start_time))
        #
        #     file_path = os.path.dirname(__file__)
        #     file_name = os.path.abspath(os.path.join(file_path, '../PUL_GCN/dataset/'))
        #     np.save('./Datasets/flod{}/rel_SVM'.format(k) + str(i + 1) + '.npy', rel_RF)


	
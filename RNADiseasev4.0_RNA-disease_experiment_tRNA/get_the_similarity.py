import numpy as np

disease_gip_similarity = np.loadtxt('./disease_gip_similarity.csv',delimiter=',',dtype=float)
trna_gip_similarity = np.loadtxt('./trna_gip_similarity.csv',delimiter=',',dtype=float)
disease_semantic_similarity = np.load('./disease_semantic_similarity.npy')
trna_sequence_similarity = np.load('./trna_sequence_similarity.npy')

disease_similarity = disease_semantic_similarity
trna_similarity = trna_gip_similarity
for i in range(len(trna_similarity)):
    for j in range(len(trna_similarity[i])):
        if(trna_similarity[i][j]==0):
            trna_similarity[i][j] = trna_gip_similarity[i][j]
for i in range(len(disease_similarity)):
    for j in range(len(disease_similarity[i])):
        if(disease_similarity[i][j] == 0):
            disease_similarity[i][j] = disease_gip_similarity[i][j]
np.savetxt('./trna_similarity.csv',trna_similarity,delimiter=',')
np.savetxt('./disease_similarity.csv',disease_similarity,delimiter=',')

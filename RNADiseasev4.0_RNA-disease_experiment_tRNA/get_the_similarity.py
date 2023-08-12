import numpy as np

disease_gip_similarity = np.loadtxt('./disease_gip_similarity.csv',delimiter=',',dtype=float)
trna_gip_similarity = np.loadtxt('./trna_gip_similarity.csv',delimiter=',',dtype=float)
disease_semantic_similarity = np.load('./disease_semantic_similarity.npy')
trna_sequence_similarity = np.load('./trna_sequence_similarity.npy')
disease_similarity = (disease_gip_similarity + disease_semantic_similarity)/2
trna_similarity = (trna_sequence_similarity + trna_gip_similarity)/2
np.savetxt('./trna_similarity.csv',trna_similarity,delimiter=',')
np.savetxt('./disease_similarity.csv',disease_similarity,delimiter=',')

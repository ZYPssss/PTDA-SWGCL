import pickle
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics import roc_auc_score, average_precision_score

from Params import args
import scipy.sparse as sp
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

def tst_metric(label, score):
	print('auc:{}.aupr:{}'.format(roc_auc_score(label, score), average_precision_score(label, score)))
def sortlist(Complemental_label_matrix,TD_ben_ind_label):
	truplist=[]
	for i in range(len(Complemental_label_matrix)):
		for j in range(len(Complemental_label_matrix[i])):
			if((TD_ben_ind_label[i][j]!=1) and (TD_ben_ind_label[i][j]!=-1) and (TD_ben_ind_label[i][j]!=-20)):
				truplist.append([Complemental_label_matrix[i][j],i,j])
	for i in range(len(truplist)):
		for j in range(len(truplist)-1):
			if(truplist[j][0]<truplist[j+1][0]):
				w = truplist[j]
				truplist[j] = truplist[j+1]
				truplist[j+1] = w
	w1 = -1
	print(truplist)



class DataHandler:
	def __init__(self,k):
		if args.data == 'yelp':
			predir = './RNADiseasev4.0_RNA-disease_experiment_tRNA/Datasets/flod{}/'.format(k)
		self.predir = predir
		self.trnfile = predir + 'trnMat.pkl'
		self.tstfile = predir + 'tstMat.pkl'
		self.flod = k

	def loadOneFile(self, filename):
		with open(filename, 'rb') as fs:
			ret = (pickle.load(fs) != 0).astype(np.float32)
		if type(ret) != coo_matrix:
			ret = sp.coo_matrix(ret)
		return ret

	def normalizeAdj(self, mat):
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def makeTorchAdj(self, mat):
		# make ui adj
		a = sp.csr_matrix((args.tRNA, args.tRNA))
		b = sp.csr_matrix((args.disease, args.disease))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		#mat = (mat != 0) * 1.0
		mat = mat + sp.eye(mat.shape[0])
		mat = self.normalizeAdj(mat)

		# make cuda tensor
		idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = t.from_numpy(mat.data.astype(np.float32))
		shape = t.Size(mat.shape)
		return t.sparse.FloatTensor(idxs, vals, shape).to(device)

	def makeAllOne(self, torchAdj):
		idxs = torchAdj._indices()
		vals = torchAdj._values()
		shape = torchAdj.shape
		return t.sparse.FloatTensor(idxs, vals, shape).to(device)

	def LoadData(self):

		Complemental_label_matrix = np.load('./RNADiseasev4.0_RNA-disease_experiment_tRNA/Datasets/flod{}/Complemental_label_matrix.npy'.format(self.flod))
		TD_ben_ind_label = np.load('./RNADiseasev4.0_RNA-disease_experiment_tRNA/Datasets/flod{}/TD_ben_ind_label.npy'.format(self.flod))

		for i in range(len(Complemental_label_matrix)):
			for j in range(len(Complemental_label_matrix[i])):
				if(TD_ben_ind_label[i][j]==1):
					Complemental_label_matrix[i][j] = 1
				elif(Complemental_label_matrix[i][j]>=0.5):
					Complemental_label_matrix[i][j] = Complemental_label_matrix[i][j]
				else:
					Complemental_label_matrix[i][j] = 0

		a = np.eye(len(Complemental_label_matrix))
		b = np.eye(len(Complemental_label_matrix[0]))
		self.Complemental_matrix = np.vstack([np.hstack([a, Complemental_label_matrix]), np.hstack([Complemental_label_matrix.T, b])])
		trnMat = coo_matrix(Complemental_label_matrix)
		self.trnMat = trnMat
		tstMat = self.loadOneFile(self.tstfile)
		args.tRNA, args.disease = trnMat.shape
		self.torchBiAdj = self.makeTorchAdj(trnMat)
		self.allOneAdj = self.makeAllOne(self.torchBiAdj)
		trnData = TrnData(trnMat)
		self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
		tstData = TstData(tstMat, trnMat)
		self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)
	
class TrnData(data.Dataset):
	def __init__(self, coomat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok()
		self.negs = np.zeros(len(self.rows)).astype(np.int32)

	def negSampling(self):
		for i in range(len(self.rows)):
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(args.disease)
				if (u, iNeg) not in self.dokmat:
					break
			self.negs[i] = iNeg

	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
	def __init__(self, coomat, trnMat):
		self.csrmat = (trnMat.tocsr() != 0) * 1.0

		tstLocs = [None] * coomat.shape[0]
		tstTRNAs = set()
		for i in range(len(coomat.data)):
			row = coomat.row[i]
			col = coomat.col[i]
			if tstLocs[row] is None:
				tstLocs[row] = list()
			tstLocs[row].append(col)
			tstTRNAs.add(row)
		tstTRNAs = np.array(list(tstTRNAs))
		self.tstTRNAs = tstTRNAs
		self.tstLocs = tstLocs

	def __len__(self):
		return len(self.tstTRNAs)

	def __getitem__(self, idx):
		return self.tstTRNAs[idx], np.reshape(self.csrmat[self.tstTRNAs[idx]].toarray(), [-1])

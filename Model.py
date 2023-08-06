from statistics import mean
import torch as t
from torch import nn
import torch.nn.functional as F

from Params import args
import numpy as np

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		uEmbeds = np.loadtxt('./RNADiseasev4.0_RNA-disease_experiment_tRNA/tRAN_embeds.csv',delimiter=',')
		iEmbeds = np.loadtxt('./RNADiseasev4.0_RNA-disease_experiment_tRNA/disease_embeds.csv',delimiter=',')
		self.uEmbeds = nn.Parameter(t.FloatTensor(uEmbeds).to(device))
		self.iEmbeds = nn.Parameter(t.FloatTensor(iEmbeds).to(device))
		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gcn_layer)])
		self.gtLayers = nn.Sequential(*[GTLayer() for i in range(args.gt_layer)])
	
	def getEgoEmbeds(self):
		return t.cat([self.uEmbeds, self.iEmbeds], axis=0)

	def forward(self, encoderAdj, decoderAdj=None):
		# self.uEmbeds = nn.Parameter(init(self.linear1(self.tRNA_GIP)))
		# self.iEmbeds = nn.Parameter(init(self.linear2(self.disease_GIP)))
		embeds = t.cat([self.uEmbeds, self.iEmbeds], axis=0)
		embedsLst = [embeds]
		for i, gcn in enumerate(self.gcnLayers):
			embeds = gcn(encoderAdj, embedsLst[-1])
			embedsLst.append(embeds)
		if decoderAdj is not None:
			for gt in self.gtLayers:
				embeds = gt(decoderAdj, embedsLst[-1])
				embedsLst.append(embeds)
		embeds = sum(embedsLst)
		return embeds[:args.tRNA], embeds[args.tRNA:]

class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	def forward(self, adj, embeds):
		return t.spmm(adj, embeds)

class GTLayer(nn.Module):
	def __init__(self):
		super(GTLayer, self).__init__()
		self.qTrans = nn.Parameter(init(t.empty(args.latdim, args.latdim)))
		self.kTrans = nn.Parameter(init(t.empty(args.latdim, args.latdim)))
		self.vTrans = nn.Parameter(init(t.empty(args.latdim, args.latdim)))
	
	def forward(self, adj, embeds):
		indices = adj._indices()
		rows, cols = indices[0, :], indices[1, :]
		rowEmbeds = embeds[rows]
		colEmbeds = embeds[cols]

		qEmbeds = (rowEmbeds @ self.qTrans).view([-1, args.head, args.latdim // args.head])
		kEmbeds = (colEmbeds @ self.kTrans).view([-1, args.head, args.latdim // args.head])
		vEmbeds = (colEmbeds @ self.vTrans).view([-1, args.head, args.latdim // args.head])

		att = t.einsum('ehd, ehd -> eh', qEmbeds, kEmbeds)
		att = t.clamp(att, -10.0, 10.0)
		expAtt = t.exp(att)
		tem = t.zeros([adj.shape[0], args.head]).to(device)
		attNorm = (tem.index_add_(0, rows, expAtt))[rows]
		att = expAtt / (attNorm + 1e-8) # eh

		resEmbeds = t.einsum('eh, ehd -> ehd', att, vEmbeds).view([-1, args.latdim])
		tem = t.zeros([adj.shape[0], args.latdim]).to(device)
		resEmbeds = tem.index_add_(0, rows, resEmbeds) # nd
		return resEmbeds



class LocalGraph(nn.Module):
	def __init__(self):
		super(LocalGraph, self).__init__()
	
	def makeNoise(self, scores):
		noise = t.rand(scores.shape).to(device)
		noise = -t.log(-t.log(noise))
		return t.log(scores) + noise
	
	def forward(self, allOneAdj, embeds):
		# allOneAdj should be with self-loop
		# embeds should be zero-order embeds
		order = t.sparse.sum(allOneAdj, dim=-1).to_dense().view([-1, 1])
		fstEmbeds = t.spmm(allOneAdj, embeds) - embeds
		fstNum = order
		scdEmbeds = (t.spmm(allOneAdj, fstEmbeds) - fstEmbeds) - order * embeds
		scdNum = (t.spmm(allOneAdj, fstNum) - fstNum) - order
		subgraphEmbeds = (fstEmbeds + scdEmbeds) / (fstNum + scdNum + 1e-8)
		subgraphEmbeds = F.normalize(subgraphEmbeds, p=2)
		embeds = F.normalize(embeds, p=2)
		scores = t.sigmoid(t.sum(subgraphEmbeds * embeds, dim=-1))
		scores = self.makeNoise(scores)
		_, seeds = t.topk(scores, args.seedNum)
		return scores, seeds

class RandomMaskSubgraphs(nn.Module):
	def __init__(self):
		super(RandomMaskSubgraphs, self).__init__()
		self.flag = False
	
	def normalizeAdj(self, adj):
		degree = t.pow(t.sparse.sum(adj, dim=1).to_dense() + 1e-12, -0.5)
		newRows, newCols = adj._indices()[0, :], adj._indices()[1, :]
		rowNorm, colNorm = degree[newRows], degree[newCols]
		newVals = adj._values() * rowNorm * colNorm
		return t.sparse.FloatTensor(adj._indices(), newVals, adj.shape)

	def forward(self, adj, seeds,Complemental_matrix):
		rows = adj._indices()[0, :]
		cols = adj._indices()[1, :]

		maskNodes = [seeds]

		for i in range(args.maskDepth):
			curSeeds = seeds if i == 0 else nxtSeeds
			nxtSeeds = list()
			for seed in curSeeds:
				rowIdct = (rows == seed)
				colIdct = (cols == seed)
				idct = t.logical_or(rowIdct, colIdct)

				if i != args.maskDepth - 1:
					mskRows = rows[idct]
					mskCols = cols[idct]
					nxtSeeds.append(mskRows)
					nxtSeeds.append(mskCols)

				rows = rows[t.logical_not(idct)]
				cols = cols[t.logical_not(idct)]
			if len(nxtSeeds) > 0:
				nxtSeeds = t.unique(t.cat(nxtSeeds))
				maskNodes.append(nxtSeeds)
		sampNum = int((args.tRNA + args.disease) * args.keepRate)
		sampedNodes = t.randint(args.tRNA + args.disease, size=[sampNum]).to(device)
		if self.flag == False:
			l1 = adj._values().shape[0]
			l2 = rows.shape[0]
			print('-----')
			print('LENGTH CHANGE', '%.2f' % (l2 / l1), l2, l1)
			tem = t.unique(t.cat(maskNodes))
			print('Original SAMPLED NODES', '%.2f' % (tem.shape[0] / (args.tRNA + args.disease)), tem.shape[0], (args.tRNA + args.disease))
		maskNodes.append(sampedNodes)
		maskNodes = t.unique(t.cat(maskNodes))
		if self.flag == False:
			print('AUGMENTED SAMPLED NODES', '%.2f' % (maskNodes.shape[0] / (args.tRNA + args.disease)), maskNodes.shape[0], (args.tRNA + args.disease))
			self.flag = True
			print('-----')

		values = []
		for i in range(len(rows)):
			values.append(Complemental_matrix[rows[i]][cols[i]])
		values = np.array(values)
		encoderAdj = self.normalizeAdj(t.sparse.FloatTensor(t.stack([rows, cols], dim=0), t.FloatTensor(values).to(device), adj.shape))

		temNum = maskNodes.shape[0]
		temRows = maskNodes[t.randint(temNum, size=[adj._values().shape[0]]).to(device)]
		temCols = maskNodes[t.randint(temNum, size=[adj._values().shape[0]]).to(device)]

		newRows = t.cat([temRows, temCols, t.arange(args.tRNA+args.disease).to(device), rows])
		newCols = t.cat([temCols, temRows, t.arange(args.tRNA+args.disease).to(device), cols])

		# filter duplicated
		hashVal = newRows * (args.tRNA + args.disease) + newCols
		hashVal = t.unique(hashVal)
		newCols = hashVal % (args.tRNA + args.disease)
		newRows = ((hashVal - newCols) / (args.tRNA + args.disease)).long()

		values = []
		for i in range(len(newRows)):
			values.append(Complemental_matrix[newRows[i]][newCols[i]])
		values = np.array(values)
		decoderAdj = t.sparse.FloatTensor(t.stack([newRows, newCols], dim=0), t.FloatTensor(values).to(device).float(), adj.shape)
		return encoderAdj, decoderAdj

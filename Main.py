import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import Model, RandomMaskSubgraphs, LocalGraph
from DataHandler import DataHandler
import numpy as np
from Utils.Utils import *
from Utils.Utils import contrast
import os
from sklearn.metrics import auc, roc_auc_score, average_precision_score
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
auc = []
aupr = []

def tst_metric(label, score):
	return roc_auc_score(label, score), average_precision_score(label, score)

class Coach:
	def __init__(self, handler,k):
		self.handler = handler

		print('TRAN', args.tRNA, 'DISEASE', args.disease)
		print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
		self.metrics = dict()
		self.bestauc = -1
		self.bestaupr = -1
		self.flod = k
		mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')
		if args.load_model != None:
			self.loadModel()
			stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
		else:
			stloc = 0
			log('Model Initialized')
		bestRes = None
		for ep in range(stloc, args.epoch):
			tstFlag = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, tstFlag))
			reses = self.testEpoch()
			log(self.makePrint('Test', ep, reses, tstFlag))
			bestRes = reses if bestRes is None or reses['Recall'] > bestRes['Recall'] else bestRes
			print()
		reses = self.testEpoch()
		log(self.makePrint('Test', args.epoch, reses, True))
		log(self.makePrint('Best Result', args.epoch, bestRes, True))
		print('best_auc:{},best_aupr:{}'.format(self.bestauc,self.bestaupr))
		auc.append(self.bestauc)
		aupr.append(self.bestaupr)

	def prepareModel(self):
		self.model = Model().to(device)
		self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
		self.masker = RandomMaskSubgraphs()
		self.sampler = LocalGraph()
	
	def trainEpoch(self):
		trnLoader = self.handler.trnLoader
		epLoss, epPreLoss = 0, 0
		steps = trnLoader.dataset.__len__() // args.batch
		for i, tem in enumerate(trnLoader):
			if i % args.fixSteps == 0:
				centerScores, seeds = self.sampler(self.handler.allOneAdj, self.model.getEgoEmbeds())
				encoderAdj, decoderAdj = self.masker(self.handler.torchBiAdj, seeds,self.handler.Complemental_matrix)
			tRNA_index, disease_index, _ = tem
			tRNA_index = tRNA_index.long().to(device)
			disease_index = disease_index.long().to(device)
			tRNA_Embeds, disease_Embeds = self.model(encoderAdj, decoderAdj)
			ancEmbeds = tRNA_Embeds[tRNA_index]
			posEmbeds = disease_Embeds[disease_index]
			
			bprLoss = (-t.sum(ancEmbeds * posEmbeds, dim=-1)).mean()
			regLoss = calcRegLoss(self.model) * args.reg

			contrastLoss = (contrast(tRNA_index, tRNA_Embeds) + contrast(disease_index, disease_Embeds)) * args.ssl_reg + contrast(tRNA_index, tRNA_Embeds, disease_Embeds)
			
			loss = bprLoss + regLoss + contrastLoss
			
			if i % args.fixSteps == 0:
				localGlobalLoss = -centerScores.mean()
				loss += localGlobalLoss
			epLoss += loss.item()
			epPreLoss += bprLoss.item()
			self.opt.zero_grad()
			loss.backward()
			self.opt.step()
		ret = dict()
		ret['Loss'] = epLoss / (steps+1)
		ret['preLoss'] = epPreLoss / (steps+1)
		return ret

	def testEpoch(self):
		tstLoader = self.handler.tstLoader
		epLoss, epRecall, epNdcg = [0] * 3
		i = 0
		num = tstLoader.dataset.__len__()
		steps = num // args.tstBat
		for usr, trnMask in tstLoader:
			i += 1
			usr = usr.long().to(device)
			trnMask = trnMask.to(device)
			usrEmbeds, itmEmbeds = self.model(self.handler.torchBiAdj, self.handler.torchBiAdj)

			allPreds = t.mm(usrEmbeds[usr], t.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
			allPreds1 = allPreds.detach().cpu().numpy()
			_, topLocs = t.topk(allPreds, args.topk)
			recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
			epRecall += recall
			epNdcg += ndcg
		ret = dict()
		ret['Recall'] = epRecall / num
		ret['NDCG'] = epNdcg / num
		allscores = t.mm(usrEmbeds, t.transpose(itmEmbeds, 1, 0)).detach().cpu().numpy()
		score = []
		label = []
		test_positive = np.loadtxt('./RNADiseasev4.0_RNA-disease_experiment_tRNA/Datasets/flod{}/test_positive.csv'.format(self.flod), dtype=int, delimiter=',')
		#test_positive = np.loadtxt('./Datasets/sparse_yelp/test_positive.csv', dtype=int,delimiter=',')
		for j in range(len(test_positive)):
			score.append(allscores[test_positive[j][0]][test_positive[j][1]])
			label.append(1)

		test_negative = np.loadtxt('./RNADiseasev4.0_RNA-disease_experiment_tRNA/Datasets/flod{}/test_negative.csv'.format(self.flod), dtype=int, delimiter=',')
		#test_negative = np.loadtxt('./Datasets/sparse_yelp/test_negative.csv', dtype=int,delimiter=',')
		for j in range(len(test_negative)):
			score.append(allscores[test_negative[j][0]][test_negative[j][1]])
			label.append(0)
		auc, aupr = tst_metric(label,score)
		print('auc:{},aupr:{}'.format(auc,aupr))
		if(self.bestauc<auc):
			self.bestauc = auc
			self.bestaupr = aupr
			np.savetxt('./result/flod{}/testlabel{}.csv'.format(self.flod,self.flod),np.array(label),fmt='%d')
			np.savetxt('./result/flod{}/testscore{}.csv'.format(self.flod,self.flod),np.array(score))
			np.savetxt('./result/flod{}/allscores.csv'.format(self.flod),np.array(allscores))
			np.save('./result/flod{}/tRNA_Embeds.npy'.format(self.flod),usrEmbeds.detach().cpu().numpy())
			np.save('./result/flod{}/disease_Embeds.npy'.format(self.flod),itmEmbeds.detach().cpu().numpy())
		return ret

	def calcRes(self, topLocs, tstLocs, batIds):
		assert topLocs.shape[0] == len(batIds)
		allRecall = allNdcg = 0
		for i in range(len(batIds)):
			temTopLocs = list(topLocs[i])
			temTstLocs = tstLocs[batIds[i]]
			tstNum = len(temTstLocs)
			maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
			recall = dcg = 0
			for val in temTstLocs:
				if val in temTopLocs:
					recall += 1
					dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
			recall = recall / tstNum
			ndcg = dcg / maxDcg
			allRecall += recall
			allNdcg += ndcg
		return allRecall, allNdcg

if __name__ == '__main__':
	logger.saveDefault = True
	for i in range(5):
		print('--------------------------------------flod{}-------------------------------------------------'.format(i+1))
		log('Start')
		handler = DataHandler(i+1)
		handler.LoadData()
		log('Load Data')

		coach = Coach(handler,i+1)
		coach.run()
	print('mean auc:{},aupr:{}'.format(sum(auc) / 5, sum(aupr) / 5))
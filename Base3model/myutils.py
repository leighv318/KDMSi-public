from abc import abstractmethod
from typing import List, Tuple, Union
from scipy import stats
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, roc_auc_score
import sklearn
import random
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import pdb
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import torch.backends.cudnn as cudnn
import pandas as pd
from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses.approxNDCG import approxNDCGLoss


def logging(content, filename='logging.txt'):
	with open(filename, 'a') as f:
		f.write(content + '\n')

def log(filename='logging.txt'):
	pcc, spcc, ndcg, auc = [], [], [], []

	with open(filename) as f:
		lines = f.readlines()
		for line in lines:
			line = line.split(',')
			pcc.append(float(line[0].split(':')[-1].strip()))
			spcc.append(float(line[1].split(':')[-1].strip()))
			ndcg.append(float(line[2].split(':')[-1].strip()))
			auc.append(float(line[3].split(':')[-1].strip()))
				

	data = pd.DataFrame()
	data['pcc'], data['spcc'], data['ndcg'], data['auc'] = pcc, spcc, ndcg, auc
	data.to_csv(filename+'indicators.csv', index=0)

def init_seed(SEED):
	random.seed(SEED)
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	cudnn.enabled = True
	cudnn.deterministic = True
	cudnn.benchmark = False



#
# def linear_regression(x, y):
#	  N = len(x)
#	  sumx = sum(x)
#	  sumy = sum(y)
#	  sumx2 = sum(x ** 2)
#	  sumxy = sum(x * y)
#	  A = np.mat([[N, sumx], [sumx, sumx2]])
#	  b = np.array([sumy, sumxy])
#	  return np.linalg.solve(A, b)

def plot_scatter_curve(pred, label, save_path, title):
	# a0, a1 = linear_regression(pred, label)
	# # 生成拟合直线的绘制点
	# _X1 = [0, 1.]
	# _Y1 = [a0 + a1 * x for x in _X1]

	plt.scatter(pred, label, s=10, c='b') #点的大小默认20
	# plt.plot(_X1, _Y1, 'b', linewidth=0.5)
	plt.title('exp vs pred / {}'.format(title))
	plt.xlabel('prediction')
	plt.ylabel('experiment')
	plt.xlim(0., 1.)
	plt.ylim(0., 1.)
	plt.savefig(os.path.join(save_path, str(title) + '.png'))
	plt.close()

def plot_metric_curve(metrics_dict, train_or_valid, base_path, epoches: tuple):
	x_axis = np.arange(epoches[0], epoches[1])

	for key in metrics_dict.keys():
		value = metrics_dict[key]
		plt.plot(x_axis, value, c='r')
		plt.title(train_or_valid + '_' + key)
		plt.xlabel('epoches')
		plt.ylabel(key)
		plt.savefig(os.path.join(base_path, train_or_valid + '_' + key + '.png'))
		plt.close()


def ncdg_loss_multiclass(pred, truth, sources):
	# target mrna label: 0~42
	
	NDCGLoss = torch.zeros(43).to(pred.dtype).to(pred.device)
	for i in range(43):
		q_truth = torch.where(sources!=i, PADDED_Y_VALUE, truth)
		if q_truth.sum()!=q_truth.shape[0] * PADDED_Y_VALUE:
			NDCGLoss[i] = 1. + approxNDCGLoss(pred.unsqueeze(0), q_truth.unsqueeze(0))

	return torch.mean(NDCGLoss)


def ncdg_loss(pred, truth, alpha=1., k=65):
	# target mrna label: 0~42
	newpred = pred[:k]
	newtruth = truth[:k]
	rtn = 1. + approxNDCGLoss(newpred.unsqueeze(0), newtruth.unsqueeze(0), alpha=alpha)
	return rtn


class EarlyStopping:
	"""Early stops the training if validation loss doesn't improve after a given patience."""
	def __init__(self, model_save_path, model_save_name, patience=15, greater=False, delta=0):
		"""
			patience (int): How long to wait after last time validation loss improved.
							Default: 7
			greater (bool): the greater, the better.
							Default: False
			delta (float): Minimum change in the monitored quantity to qualify as an improvement.
							Default: 0
		"""
		self.patience = patience
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.delta = delta
		self.greater = greater
		self.path = model_save_path
		self.best_model_path = None
		self.model_name = model_save_name

	def __call__(self, score, iter, model):

		if self.greater:
			cur_score = score
		else:
			cur_score = -score

		if self.best_score is None:
			self.best_score = cur_score
			self.save_checkpoint(score, model, iter)
		elif cur_score < self.best_score + self.delta:
			self.counter += 1
			if self.counter >= self.patience:
				# logging(f'EarlyStopping counter: {self.counter} out of {self.patience}')
				self.early_stop = True
		else:
			self.best_score = cur_score
			self.save_checkpoint(score, model, iter)
			self.counter = 0

	def save_checkpoint(self, score, model, iter):
		# self.best_model_path = os.path.join(self.path, 'pcc_' + str(round(score, 4)) +'_epoch_' +str(iter).zfill(3)+'.pth.tar')
		self.best_model_path = os.path.join(self.path, self.model_name)
		torch.save(model.state_dict(), self.best_model_path)


def map_source_to_idx():
	data = pd.read_csv('./data/normalized_sirna_with_mrna.csv',header=0)
	from sklearn.preprocessing import LabelEncoder
	lf=LabelEncoder().fit(np.array(data['nm_names']))
	data['source_ndcg']=lf.transform(np.array(data['nm_names']))
	data.to_csv('./data/normalized_sirna_with_mrna.csv', index=0)

if __name__=='__main__':
	map_source_to_idx()
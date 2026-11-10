from scipy import stats
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, roc_auc_score
import sklearn
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import random
import pdb
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import torch.backends.cudnn as cudnn
import pandas as pd
from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses.approxNDCG import approxNDCGLoss

def init_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    cudnn.enabled = True
    cudnn.deterministic = True
    cudnn.benchmark = False


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


def get_rdkit_fingerprint():

    '''
    根据修饰的smiles表征，通过RDKIT的MACCKEYS获取167维度的分子指纹
    然后去除全为0/1的bit，剩下了101 bit是有差异性的，保存到./data/finerprints.npy
    '''
    path = './data/fingerprints.npy'

    # if not os.path.exists(path):
    #     from rdkit import Chem
    #     from rdkit.Chem import MACCSkeys
    #     result = {}
    #     for each_ in symbol_2_smiles.keys():
    #         if each_ is not None:
    #             molecule = Chem.MolFromSmiles(symbol_2_smiles[each_])
    #             fingerprints = MACCSkeys.GenMACCSKeys(molecule)
    #             result[each_] = fingerprints.ToBitString()
    #             # print(each_, '\n', result[each_])

    #     # # macckeys分子指纹是167长度，统计每个bit上，23种修饰是否相同，如果完全相同的话，就去掉这一bit
    #     # 统计出来总共101bit是不一样的
    #     differ_bit_ind = []
    #     for i in range(167):
    #         cnt = 0
    #         for v_ in result.keys():
    #             cnt += int(result[v_][i])
    #         if cnt != 0 and cnt != 167:
    #             differ_bit_ind.append(i)

    #     print('macckeys fingerprints\' shape is : {}'.format( len(differ_bit_ind)) )
    #     abbreviate_result = {}
    #     for k_ in result.keys():
    #         abbreviate_result[k_] = np.array([int(result[k_][i]) for i in differ_bit_ind])

    #     np.save(path, abbreviate_result, allow_pickle=True)

    result = np.load(path, allow_pickle=True)
    return result



import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

def map_source_to_idx():
	data = pd.read_excel('./data/ori_data.xlsx',header=0)
	# data.to_excel('./data/backup.xlsx',index=0)
	lf=LabelEncoder().fit(np.array(data['ori_as']))
	data['source_ndcg']=lf.transform(np.array(data['ori_as']))
	data.to_excel('./data/ori_data.xlsx', index=0)

if __name__=='__main__':
	map_source_to_idx()

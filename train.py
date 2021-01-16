import cv2
import numpy as np
from numpy.linalg import norm
import sys
import os
import json

SZ = 20          #训练图片长宽
MAX_WIDTH = 1000 #原始图片最大宽度
Min_Area = 2000  #车牌区域允许最大面积
PROVINCE_START = 1000
#不能保证包括所有省份
provinces = [
	"zh_cuan", "川",
	"zh_e", "鄂",
	"zh_gan", "赣",
	"zh_gan1", "甘",
	"zh_gui", "贵",
	"zh_gui1", "桂",
	"zh_hei", "黑",
	"zh_hu", "沪",
	"zh_ji", "冀",
	"zh_jin", "津",
	"zh_jing", "京",
	"zh_jl", "吉",
	"zh_liao", "辽",
	"zh_lu", "鲁",
	"zh_meng", "蒙",
	"zh_min", "闽",
	"zh_ning", "宁",
	"zh_qing", "靑",
	"zh_qiong", "琼",
	"zh_shan", "陕",
	"zh_su", "苏",
	"zh_sx", "晋",
	"zh_wan", "皖",
	"zh_xiang", "湘",
	"zh_xin", "新",
	"zh_yu", "豫",
	"zh_yu1", "渝",
	"zh_yue", "粤",
	"zh_yun", "云",
	"zh_zang", "藏",
	"zh_zhe", "浙"
]

class StatModel(object):
	def load(self, fn):
		self.model = self.model.load(fn)#从文件载入训练好的模型
	def save(self, fn):
		self.model.save(fn)#保存训练好的模型到文件中

class SVM(StatModel):
	def __init__(self, C = 1, gamma = 0.5):
		self.model = cv2.ml.SVM_create()#生成一个SVM模型
		self.model.setGamma(gamma) #设置Gamma参数，demo中是0.5
		self.model.setC(C)# 设置惩罚项， 为：1
		self.model.setKernel(cv2.ml.SVM_RBF)#设置核函数
		self.model.setType(cv2.ml.SVM_C_SVC)#设置SVM的模型类型：SVC是分类模型，SVR是回归模型
	#训练svm
	def train(self, samples, responses):
		self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)#训练
	#字符识别
	def predict(self, samples):
		r = self.model.predict(samples)#预测
		return r[1].ravel()

#来自opencv的sample，用于svm训练
def deskew(img):
	m = cv2.moments(img)
	if abs(m['mu02']) < 1e-2:
		return img.copy()
	skew = m['mu11']/m['mu02']
	M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
	img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
	return img

#来自opencv的sample，用于svm训练
def preprocess_hog(digits):
	samples = []
	for img in digits:
		gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
		gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
		mag, ang = cv2.cartToPolar(gx, gy)
		bin_n = 16
		bin = np.int32(bin_n*ang/(2*np.pi))
		bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
		mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
		hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
		hist = np.hstack(hists)

		# transform to Hellinger kernel
		eps = 1e-7
		hist /= hist.sum() + eps
		hist = np.sqrt(hist)
		hist /= norm(hist) + eps

		samples.append(hist)
	return np.float32(samples)


def save_traindata(model,modelchinese):
	if not os.path.exists("module\\svm.dat"):
		model.save("module\\svm.dat")
	if not os.path.exists("module\\svmchinese.dat"):
		modelchinese.save("module\\svmchinese.dat")

def train_svm():
	#识别英文字母和数字
	model = SVM(C=1, gamma=0.5)
	#识别中文
	modelchinese = SVM(C=1, gamma=0.5)
	if os.path.exists("svm.dat"):
		model.load("svm.dat")
	else:
		chars_train = []
		chars_label = []

		for root, dirs, files in os.walk("train\\chars2"):
			if len(os.path.basename(root)) > 1:
				continue
			root_int = ord(os.path.basename(root))
			for filename in files:
				filepath = os.path.join(root,filename)
				digit_img = cv2.imread(filepath)
				digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
				chars_train.append(digit_img)
				#chars_label.append(1)
				chars_label.append(root_int)

		chars_train = list(map(deskew, chars_train))
		#print(chars_train)
		chars_train = preprocess_hog(chars_train)
		#print(chars_train)
		#chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
		chars_label = np.array(chars_label)
		model.train(chars_train, chars_label)
	if os.path.exists("svmchinese.dat"):
		modelchinese.load("svmchinese.dat")
	else:
		chars_train = []
		chars_label = []
		for root, dirs, files in os.walk("train\\charsChinese"):
			if not os.path.basename(root).startswith("zh_"):
				continue
			pinyin = os.path.basename(root)
			index = provinces.index(pinyin) + PROVINCE_START + 1 #1是拼音对应的汉字
			for filename in files:
				filepath = os.path.join(root,filename)
				digit_img = cv2.imread(filepath)
				digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
				chars_train.append(digit_img)
				#chars_label.append(1)
				chars_label.append(index)
		chars_train = list(map(deskew, chars_train))
		chars_train = preprocess_hog(chars_train)
		#chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
		chars_label = np.array(chars_label)
		print(chars_train.shape)
		modelchinese.train(chars_train, chars_label)

	save_traindata(model,modelchinese)


train_svm()
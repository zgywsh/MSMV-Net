# coding=gbk

import re
import pylab
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib
matplotlib.use('Agg')



TXT = open("/home/wangz/verid/checkpoints and logs/mgn5/log.txt","r+")
txt=TXT.read()
l1=txt.split("\n")

TXT = open("/home/wangz/verid/checkpoints and logs/mgn5_0/log.txt","r+")
txt=TXT.read()
l2=txt.split("\n")

# print(l1)
mAP=[]
rank_1=[]
rank_5=[]
Loss=[]
Loss2=[]
mAP2=[]

def map_rank1_5() :

	for str in l1 :
	# 匹配“mAP”后面的数字
		pattern = re.compile(r'(?<=Rank-1  :)\d+\.?\d*')
		num=pattern.findall(str)
		if num != [] :
			num=float("".join(num))
			# print(type(num))
			rank_1.append(num)


	for str in l1 :
	# 匹配“mAP”后面的数字
		pattern = re.compile(r'(?<=mAP: )\d+\.?\d*')
		num=pattern.findall(str)
		if num != [] :
			num=float("".join(num))
			# print(type(num))
			mAP.append(num)


	for str in l2 :
	# 匹配“mAP”后面的数字
		pattern = re.compile(r'(?<=mAP: )\d+\.?\d*')
		num=pattern.findall(str)
		if num != [] :
			num=float("".join(num))
			# print(type(num))
			mAP2.append(num)


	for str in l2 :
	# 匹配“mAP”后面的数字
		pattern = re.compile(r'(?<=Rank-5  :)\d+\.?\d*')
		num=pattern.findall(str)
		if num != [] :
			num=float("".join(num))
			# print(type(num))
			rank_5.append(num)

	if len(mAP2)>len(mAP):
		for i in range(len(mAP2)-len(mAP)):
			mAP.append(mAP[-1])
			# rank_1.append(rank_1[-1])
			# rank_5.append(rank_5[-1])
	else:
		for i in range(len(mAP)-len(mAP2)):
			mAP2.append(mAP2[-1])

	y = mAP
	y1 = rank_1
	y2 = rank_5
	y3 = mAP2
	x = range(1, len(mAP)+1)
	plt.plot(x, y)
	# plt.plot(x, y1)
	# plt.plot(x, y2)
	plt.plot(x, y3)
	plt.title('MGN1_VeRi776')
	plt.xlabel('Epoch')
	plt.grid()  # 网格线
	# y以每10示
	y_major_locator = MultipleLocator(10)
	ax = plt.gca()
	ax.yaxis.set_major_locator(y_major_locator)

	plt.legend(['map', "map2"])



def loss():
	for str in l1 :
	# 匹配“mAP”后面的数字
		pattern = re.compile(r'(?<=Loss: )\d+\.?\d*')
		num=pattern.findall(str)
		if num != [] :
			num=float("".join(num))
			# print(type(num))
			Loss.append(num)
	for str in l2 :
	# 匹配“mAP”后面的数字
		pattern = re.compile(r'(?<=Loss: )\d+\.?\d*')
		num=pattern.findall(str)
		if num != [] :
			num=float("".join(num))
			# print(type(num))
			Loss2.append(num)

	if len(Loss2)>len(Loss):
		for i in range(len(Loss2)-len(Loss)):
			Loss.append(Loss[-1])
			# rank_1.append(rank_1[-1])
			# rank_5.append(rank_5[-1])
	else:
		for i in range(len(Loss)-len(Loss2)):
			Loss2.append(Loss2[-1])
	y3=Loss
	y4=Loss2
	x = range(1, len(y3) + 1)
	plt.plot(x, y3)
	plt.plot(x, y4)
	plt.title('MGN1_VeRi776')
	plt.xlabel('Epoch')
	plt.grid()  # 网格线
	y_major_locator = MultipleLocator(0.1)
	ax = plt.gca()
	ax.yaxis.set_major_locator(y_major_locator)

	plt.legend(['Loss'])



if __name__ == "__main__":

	map_rank1_5()
	plt.savefig("/home/wangz/verid/checkpoints and logs/MAP")
	plt.clf()
	loss()
	plt.savefig("/home/wangz/verid/checkpoints and logs/LOSS")
	# plt.show()
	# pylab.show()




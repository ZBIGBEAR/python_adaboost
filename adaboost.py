#adaboost算法实现
#参考：https://blog.csdn.net/guyuealian/article/details/70995333
import matplotlib.pyplot as plt
from numpy import *
import math
plt.switch_backend('agg')

#读取数据
def getData(filename):
	fr = open(filename)
	lines =[line.strip().split('\t') for line in fr.readlines()]
	#print(lines)
	data= []
	labels=[]
	for line in lines:
		data.append([float(line[0]),float(line[1])])
		labels.append(int(line[2]))
	return data,labels

#弱分类器1,x=2.5
def H1(point):
	if point[0]>2.5:
		return -1
	else:
		return 1

#弱分类器2,x=8.5
def H2(point):
	if point[0]>8.5:
		return -1
	else:
		return 1

#弱分类器3,y=6.5
def H3(point):
	if point[1]>6.5:
		return 1
	else:
		return -1

#初始化各个点的权值
def initD(points):
	m = len(points)
	ds = []
	for i in range(m):
		ds.append(1.0/m)
	return ds

#求分类器在特定权值下的误差
def getError(H,points,labels,ds):
	ret=[]
	for point in points:
		ret.append(H(point))
	error = 0.0
	m = len(points)
	for i in range(m):
		if labels[i]!=ret[i]:
			error +=ds[i]
	return error,ret

#求误差最小的分类器
def getMinError(points,labels,ds):
	error1,ds1 = getError(H1,points,labels,ds)
	error2,ds2 = getError(H2,points,labels,ds)
	error3,ds3 = getError(H3,points,labels,ds)
	#print([error1,error2,error3])
	#Min = min([error1,error2,error3])
	Min = error1
	func = H1
	dsRet = ds1
	if Min>error2:
		Min=error2
		func = H2
		dsRet=ds2
	if Min>error3:
		Min = error3
		func = H3
		dsRet=ds3
	param = (1.0/2)*math.log((1-Min)/Min)	
	#print("-----------------")
	#print(Min)
	#print(param)
	#print(labels)
	#print(dsRet)
	dss=[]
	m = len(ds)
	for i in range(m):
		if dsRet[i]==labels[i]:
			dss.append(ds[i]/(2*(1-Min)))
		else:
			dss.append(ds[i]/(2*Min))
	return param,func,dss

#求各个弱分类器的系数
def calcParas(points,labels):
	m =3
	params=[]
	funcs=[]
	ds = initD(points)
	for i in range(m):
		param,func,ds = getMinError(points,labels,ds)
		#print("当前选择")
		#print(param,func,ds)
		params.append(param)
		funcs.append(func)
		#print("calcError")
		#print(calcError(params,funcs,points,labels))
	return params,funcs

#符号函数
def sign(x):
	if x>0:
		return 1
	else:
		return -1

#根据当前还未完成的adaboost分类器计算误差率
def calcError(params,funcs,points,labels):
	ret = []
	m = len(funcs)
	n = len(points)
	error = 0
	for i in range(n):
		label= 0
		for j in range(m):
			label += params[j]*funcs[j](points[i])
		ret.append(sign(label))
	for i in range(n):
		if ret[i]!=labels[i]:
			error += 1
	return error*1.0/n

#获取测试点集合
def getTestPoints(filename):
	fr = open(filename)
	lines =[line.strip().split('\t') for line in fr.readlines()]
	data=[]
	for line in lines:
		data.append([line[0],line[1]])
	return data

#随机生成测试点集合
def getTestPointsByRand(n):
	dataSet = []
	for i in range(n):
		dataSet.append([random.uniform(0, 10),random.uniform(0, 10)])
	return dataSet

#将adaboost函数用于测试点集合
def test(params,funcs,points):
	ret=[]
	m=len(funcs)
	n=len(points)
	for i in range(n):
		label=0
		for j in range(m):
			print(points[i])
			label += params[j]+funcs[j](points[i])
		ret.append(sign(label))
	return ret

if __name__=='__main__':
	#print("H1",H1)
	#print("H2",H2)
	#print("H3",H3)
	filename='data/points.txt'
	points,labels = getData(filename)
	params,funcs = calcParas(points,labels)
	print(params)
	print(funcs)
	#print(points)
	#print(labels)
	#ds = initD(points)
	#print(ds)
	#error1=getError(H1,points,labels,ds)
	#error2=getError(H2,points,labels,ds)
	#error3=getError(H3,points,labels,ds)
	#print(error1)
	#print(error2)
	#print(error3)

	#画图
	xcord1=[]
	ycord1=[]
	xcord2=[]
	ycord2=[]
	m = len(points)
	for i in range(m):
		if labels[i]==1:
			xcord1.append(points[i][0])
			ycord1.append(points[i][1])
		else:
			xcord2.append(points[i][0])
			ycord2.append(points[i][1])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1,ycord1,s=30,c='green',marker='s')
	ax.scatter(xcord2,ycord2,s=30,c='red')
	x = arange(0.0,10.0,0.1)
	Y=[]
	for i in x:
		Y.append(6.5)
	ax.plot([2.5,2.5],[0,10])
	ax.plot([8.5,8.5],[0,10])
	ax.plot(x,Y)

	#读取测试点集合
	#filetest = 'data/test.txt'
	testSet = getTestPointsByRand(10000)
	funcsTest=[H1,H2,H3]
	testLabels = test(params,funcsTest,testSet)
	#print(testSet,testLabels)
	xtest1=[]
	ytest1=[]
	xtest2=[]
	ytest2=[]
	m = len(testSet)
	for i in range(m):
		if testLabels[i]==1:
			xtest1.append(testSet[i][0])
			ytest1.append(testSet[i][1])
		else:
			xtest2.append(testSet[i][0])
			ytest2.append(testSet[i][1])
	ax.scatter(xtest1,ytest1,s=30,c='green',marker='x')
	ax.scatter(xtest2,ytest2,s=30,c='red',marker='x')
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.savefig("adaboost.jpg")

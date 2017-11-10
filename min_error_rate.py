### Minimum Feilrate 

import numpy as np



def findMean(trainingSet, features):
	n = len(trainingSet)
	#m = len(trainingSet[0].features)
	m = len(features)
	sumVector = np.zeros([m,1])

	for elem in trainingSet:
		ac_features = np.zeros([0,1])
		for elem2 in features:
			ac_features = np.append(ac_features, elem.features[elem2])
		ac_features = ac_features.reshape([1,m])
		ac_features = ac_features.T
		sumVector += ac_features
	return sumVector/n	


def findCovariance(trainingSet, mean, features):
	n = len(trainingSet)
	m = len(features)
	sumMatrix = np.zeros([m, m])
	for elem in trainingSet:
		ac_features = np.zeros([0,1])
		for elem2 in features:
			ac_features = np.append(ac_features, elem.features[elem2])
		ac_features = ac_features.reshape([1,m])
		ac_features = ac_features.T
		sumMatrix += ((ac_features - mean) * (ac_features - mean).T)
	return sumMatrix / n



def makeDiag(matr):
	for i in range(matr.shape[0]):
		for j in range(matr.shape[1]):
			if i != j:
				matr[i,j] = 0


def discFunction(covMat, mean, x):
	bigW = -0.5 * np.linalg.inv(covMat)
	w = np.dot(np.linalg.inv(covMat), mean)
	w0 = -0.5 * np.dot(np.dot(mean.T,np.linalg.inv(covMat)), mean) - 0.5 * np.log(np.linalg.det(covMat)) + np.log(0.5)
	g = np.dot(np.dot(x.T, bigW), x) + np.dot(w.T, x) + w0
	return g


def MinErrClassifier(x, cov1, cov2, mean1, mean2):
	g1 = discFunction(cov1, mean1, x)
	g2 = discFunction(cov2, mean2, x)
	# print('cov1: ', cov1)
	# print('cov2: ', cov2)
	g = g1 - g2
	if g > 0:
		return 1
	else:
		return 2


def testClassifier(testSet, cov1, cov2, mean1, mean2, features):
		correctCount = 0
		W1asW1 = 0
		W2asW2 = 0
		W1asW2 = 0
		W2asW1 = 0
		for elem in testSet:
			ac_features = np.zeros([0,1])
			for elem2 in features:
				#print(elem.features[elem2])
				ac_features = np.append(ac_features, elem.features[elem2])
			clss = MinErrClassifier(ac_features, cov1, cov2, mean1, mean2)
			if clss == int(elem.clss):
				correctCount += 1
				if clss == 1:
					W1asW1 += 1
				else:
					W2asW2 += 1
			else:
				if int(elem.clss) == 1:
					W1asW2 += 1
				else:
					W2asW1 += 1
		c = np.matrix([[W1asW1, W1asW2], [W2asW1, W2asW2]])
		c_percent = correctCount*100 / len(testSet)
		return (c, c_percent)
		# print("Egenskaper brukt: ", features)
		# print('Prosent riktig klassifisert: ', correctCount*100 / len(testSet))
		# c = np.matrix([[W1asW1, W1asW2], [W2asW1, W2asW2]])
		# print('Forvirringsmatrise C:')
		# print(c)
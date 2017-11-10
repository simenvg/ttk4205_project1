##Nearest Neighbour
import numpy as np

def calcDistance(x1, x2, features):
	sumOfSquares = 0
	for i in range(len(x1)):
		sumOfSquares += ((x1[i] - x2[i]) * (x1[i] - x2[i]))
	return sumOfSquares


def findClass(x, trainingSet, features):
	minDist = 100
	clss = 0
	for elem in trainingSet:
		xnew = []	
		elemnew = []
		for elem2 in features:
			xnew.append(x.features[elem2])
			elemnew.append(elem.features[elem2] )
		dist = calcDistance(xnew, elemnew, features)
		if dist < minDist:
			minDist = dist		
			clss = elem.clss
	return clss


def checkClassifier(testSet, trainingSet, features):
	correctCount = 0
	W1asW1 = 0
	W2asW2 = 0
	W1asW2 = 0
	W2asW1 = 0
	for elem in testSet:
		est_clss = int(findClass(elem, trainingSet, features))
		if est_clss == int(elem.clss):
			correctCount += 1
			if est_clss == 1:
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
	# print('Prosent riktig klassifisert: ', correctCount*100 / len(testSet))
	# c = np.matrix([[W1asW1, W1asW2], [W2asW1, W2asW2]])
	# print('Forvirringsmatrise C:')
	# print(c)




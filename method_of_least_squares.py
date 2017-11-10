### Minste kvadraters metode
import numpy as np

def getY(trainingSet, features):
	Y = []
	for elem in trainingSet:
		y = [1]
		ac_features = []
		for elem2 in features:
			ac_features.append(elem.features[elem2])
		y.extend(ac_features)
		Y.append(y)
	return np.array(Y)

def getB(trainingSet):
	b = []
	for elem in trainingSet:
		if int(elem.clss) == 1:
			b.append(1)
		else:
			b.append(-1)
	return np.array(b)


def getA(Y, b):
	e1 = np.linalg.inv(np.dot(Y.T, Y)) 
	e2 = np.dot(Y.T, b)
	a = np.dot(e1, e2)
	#print('a = ', a)
	return a


def discFuncMLS(a, y):
	res = np.dot(a.T, y)
	#print('g = ', res)
	return res

# a = getA(getY(training), getB(training))


def checkClass(elem, a, features):
	y = [1]
	ac_features = []
	for elem2 in features:
		ac_features.append(elem.features[elem2])
	y.extend(ac_features)
	y = np.array(y)
	disc_val = discFuncMLS(a,y)
	if disc_val > 0:
		return 1
	else:
		return 2


def testClassifierMLS(testSet, a, features):
	correctCount = 0
	W1asW1 = 0
	W2asW2 = 0
	W1asW2 = 0
	W2asW1 = 0
	for elem in testSet:
		est_clss = checkClass(elem, a, features)
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


#testClassifierMLS(test, a)
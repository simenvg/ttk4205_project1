from min_error_rate import *
from method_of_least_squares import *
from nearest_neighbour import *
import itertools


class SetObject():

	def __init__(self, clss, features):
		self.clss = clss
		self.features = features


def readFile(fname):
	counter = 1
	testSet = []
	trainingSet = []
	with open(fname, 'r') as f:
		for line in f:
			lst = line.split()
			clss = lst.pop(0)
			features = np.array([lst], dtype=np.float64)
			features.reshape([1,len(lst)])
			features = features.T
			obj = SetObject(clss, features)
			if counter % 2 == 0:
				testSet.append(obj)
			else:
				trainingSet.append(obj)
			counter += 1
	return [testSet, trainingSet]


def divideSetByClass(trainingSet, numClasses):
	classes = [[] for i in range(numClasses)]

	for elem in trainingSet:
		classes[int(elem.clss) - 1].append(elem)
	return classes



def minErrRate(testData, trainingData, features):
		# print('\nMinste feilrate')

		mean1 = findMean(trainingData[0], features)
		mean2 = findMean(trainingData[1], features)
		cov1 = findCovariance(trainingData[0], mean1, features)
		cov2 = findCovariance(trainingData[1], mean2, features)
		makeDiag(cov1)
		makeDiag(cov2)

		(c_matr, correct_percent) = testClassifier(testData, cov1, cov2, mean1, mean2, elem)

		if correct_percent > 93:
			print('\nMinste feilrate')
			print("Egenskaper brukt: ", features)
			print('Prosent riktig klassifisert: ', correct_percent)
			print('Forvirringsmatrise C:')
			print(c_matr)

def MLS(testData, trainingData, features):
	a = getA(getY(trainingData, features), getB(trainingData))
	(c_matr, correct_percent) = testClassifierMLS(testData, a, features)

	if correct_percent > 90:
		print('\n Minste kvadraters metode')
		print("Egenskaper brukt: ", features)
		print('Prosent riktig klassifisert: ', correct_percent)
		print('Forvirringsmatrise C:')
		print(c_matr)

def nearestNeighbour(testSet, trainingSet, features):
	(c_matr, correct_percent) = checkClassifier(testSet, trainingSet, features)

	if correct_percent > 90:
		print('\nNæreste nabo')
		print("Egenskaper brukt: ", features)
		print('Prosent riktig klassifisert: ', correct_percent)
		print('Forvirringsmatrise C:')
		print(c_matr)

data = []
training_div = []

for i in range(3):
	data.append(readFile('ds-'+str(i+1)+'.txt'))
	training_div.append(divideSetByClass(data[i][1],2))

def findAllCombinations(lst):
	combinations = []
	nums = []
	for i in range(len(lst)):
		combinations.extend(itertools.combinations(lst, i+1))
	return combinations

features4 = findAllCombinations([0,1,2,3])
features3 = findAllCombinations([0,1,2])





for i in range(3):
	# i = 2
	if i == 0:
		print("\nDatasett 1")
		features = features4;
	elif i ==1:
		print("\nDatasett 2")
		features = features3
	else:
		print("\nDatasett 3")
		features = features4


	for elem in features:
		minErrRate(data[i][0], training_div[i], elem)
		MLS(data[i][0], data[i][1], elem)
		nearestNeighbour(data[i][0], data[i][1], elem)

		

	# break

import numpy as np
import os
import argparse
import collections
import random
import pickle
from ViolaJones import ViolaJones 
from Dataset import Dataset

def save_object(obj, filename):
	with open(filename, 'wb') as output:  # Overwrites any existing file.
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def main():
	# parse arguments
	parser = argparse.ArgumentParser(description='Pattern Recognition Final Project')
	parser.add_argument('-f', '--filepath', dest='filepath', type=str, required=True)
	parser.add_argument('-t', '--t', dest='t', type=int, required=True)
	parser.add_argument('-e', '--e', dest='error_type', type=str, required=False)
	args = parser.parse_args()

	if args.t < 1:
		print('-t Error:')
		print('# of classifier should larger or equal to 1')
		return 

	if args.error_type and args.error_type not in ('FN', 'FP', 'E', 'E+FP', 'E+FN'):
		print('Please specify the error type: FN or FP or E')
		print('FN: False Negative')
		print('FP: False Positive')
		print('E: Empirical Error')
		print('E+FP: Empirical Error w/ False Positive Error')
		print('E+FN: Empirical Error w/ False Negative Error')
		return 

	if not args.error_type:
		args.error_type = 'E' 

	"""
	Load Dataset
	"""
	# training set
	trainset = Dataset('train')
	if not os.path.exists(args.filepath + '/trainset.pkl'):
		trainset.process_data(args.filepath, 'trainset')
		print('training file loaded')
		save_object(trainset, args.filepath + '/trainset.pkl')
		print('training set saved')

	else:
		with open(args.filepath + '/trainset.pkl', 'rb') as inputfile:
			trainset = pickle.load(inputfile)
		print('training set loaded')			

	# testing set
	testset = Dataset('test')
	if not os.path.exists(args.filepath + '/testset.pkl'):
		testset.process_data(args.filepath, 'testset')
		print('testing file loaded')
		save_object(testset, args.filepath + '/testset.pkl')
		print('testing set saved')
	else:
		with open(args.filepath + '/testset.pkl', 'rb') as inputfile:
			testset = pickle.load(inputfile)
		print('testing set loaded')		

	"""
	Load feature into Classifier
	"""	
	# initialize violajones classifier
	classifier = ViolaJones(1)
	if not os.path.exists('./classifier.pkl'):
		classifier.load_feature(trainset, trainset.samples, trainset.pos, trainset.neg)
		print('feature loaded')
		save_object(classifier, './classifier.pkl')
		print('classifier saved')
	else:
		with open('./classifier.pkl', 'rb') as inputfile:
			classifier = pickle.load(inputfile)
		print('classifier loaded')		

	"""
	Train Classifier
	"""	
	classifier.T = int(args.t)
	classifier.train(args.error_type)
	print('training completed')
	save_object(classifier, './classifier_' + str(args.t) + '.pkl')
	print('classifier saved')

	"""
	Evaluate Classifier
	"""	
	for t in range(int(args.t) - 1, -1, -1):
		TP, FN, FP, TN = classifier.evaluate(trainset, -1)
		TP, FN, FP, TN = classifier.evaluate(testset, -1)
		classifier.clfs.pop()
		classifier.alphas.pop()

	return


if __name__ == "__main__":
	main()
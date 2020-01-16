import numpy as np
import os
import time
import argparse
import matplotlib.pyplot as plt
import collections
import random
import pickle
from ViolaJones import ViolaJones 
from Dataset import Dataset
from CascadeClassifier import CascadeClassifier

def save_object(obj, filename):
	with open(filename, 'wb') as output:  # Overwrites any existing file.
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def main():
	# parse arguments
	parser = argparse.ArgumentParser(description='Pattern Recognition Final Project')
	parser.add_argument('-f', '--filepath', dest='filepath', type=str, required=True)
	args = parser.parse_args()

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

	
	# with open('./classifier_cascade_40.pkl', 'rb') as inputfile:
	# 		classifier = pickle.load(inputfile)
	# print('classifier loaded')	
	# classifier.evaluate(trainset)
	# classifier.evaluate(testset)
	

	# Cascading System
	classifier = CascadeClassifier(max_round=40, max_cascade=40, testset=testset)
	
	if not os.path.exists('./classifier_cascade.pkl'):
		classifier.cal_feature(trainset, trainset.samples, trainset.pos, trainset.neg)
		print('feature loaded')
		save_object(classifier, './classifier_cascade.pkl')
		print('classifier saved')
	else:
		with open('./classifier_cascade.pkl', 'rb') as inputfile:
			classifier = pickle.load(inputfile)
		print('classifier loaded')	

	classifier.d = .75 # TPR for each classifier (.9)
	classifier.f = .9 # FPR for each classifier
	classifier.F_target = 0.9 ** 40 # target FPR for the cascaded classifier
	classifier.D_target = .75 # target TPR for the cascaded classifier (.90)
	start = time.time()	
	classifier.train()
	save_object(classifier, './classifier_cascade_' + str(40) + '.pkl')
	train_time = time.time() - start
	print("Average Training Time: %f" % (train_time))

	classifier.evaluate(trainset)
	classifier.evaluate(testset)

	return


if __name__ == "__main__":
	main()
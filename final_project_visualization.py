import numpy as np
import os
import argparse
import collections
import random
import pickle
from ViolaJones import ViolaJones 
from Dataset import Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def save_object(obj, filename):
	with open(filename, 'wb') as output:  # Overwrites any existing file.
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def gen_rects(window):
	rects = []
	# two vertical
	if window[0] == 1: 
		rects.append(patches.Rectangle((window[2], window[1]), window[4] // 2, window[3], fill=True, linewidth=1, edgecolor='r', facecolor='w'))
		rects.append(patches.Rectangle((window[2] + window[4] // 2, window[1]), window[4] // 2, window[3], fill=True, linewidth=1, edgecolor='r', facecolor='black'))
	# two horizontal
	elif window[0] == 2:
		rects.append(patches.Rectangle((window[2], window[1]), window[4], window[3] // 2, fill=True, linewidth=1, edgecolor='r', facecolor='w'))
		rects.append(patches.Rectangle((window[2], window[1] + window[3] // 2), window[4], window[3] // 2, fill=True, linewidth=1, edgecolor='r', facecolor='black'))
	# three horizontal
	elif window[0] == 3:
		rects.append(patches.Rectangle((window[2], window[1]), window[4], window[3] // 3, fill=True, linewidth=1, edgecolor='r', facecolor='w'))
		rects.append(patches.Rectangle((window[2], window[1] + 1 * window[3] // 3), window[4], window[3] // 3, fill=True, linewidth=1, edgecolor='r', facecolor='black'))
		rects.append(patches.Rectangle((window[2], window[1] + 2 * window[3] // 3), window[4], window[3] // 3, fill=True, linewidth=1, edgecolor='r', facecolor='w'))
	# three vertical
	elif window[0] == 4: 
		rects.append(patches.Rectangle((window[2], window[1]), window[4] // 3, window[3], fill=True, linewidth=1, edgecolor='r', facecolor='w'))
		rects.append(patches.Rectangle((window[2] + 1 * window[4] // 3, window[1]), window[4] // 3, window[3], fill=True, linewidth=1, edgecolor='r', facecolor='black'))
		rects.append(patches.Rectangle((window[2] + 2 * window[4] // 3, window[1]), window[4] // 3, window[3], fill=True, linewidth=1, edgecolor='r', facecolor='w'))
	# four
	elif window[0] == 5:
		rects.append(patches.Rectangle((window[2], window[1]), window[4] // 2, window[3] // 2, fill=True, linewidth=1, edgecolor='r', facecolor='w'))
		rects.append(patches.Rectangle((window[2] + window[4] // 2, window[1]), window[4] // 2, window[3] // 2, fill=True, linewidth=1, edgecolor='r', facecolor='black'))
		rects.append(patches.Rectangle((window[2], window[1] + window[3] // 2), window[4] // 2, window[3] // 2, fill=True, linewidth=1, edgecolor='r', facecolor='black'))
		rects.append(patches.Rectangle((window[2] + window[4] // 2, window[1] + window[3] // 2), window[4] // 2, window[3] // 2, fill=True, linewidth=1, edgecolor='r', facecolor='w'))

	return rects

def main():
	# parse arguments
	parser = argparse.ArgumentParser(description='Pattern Recognition Final Project')
	parser.add_argument('-f', '--filepath', dest='filepath', type=str, required=True)
	parser.add_argument('-i', '--i', dest='iteration', type=int, required=True)
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

	
	"""
	Visulize features
	"""
	classifier_path = './classifier_' + str(args.iteration) + '.pkl'
	if not os.path.exists(classifier_path):
		print('Error: ' + classifier_path + ' do not exist!')
		return 
	with open(classifier_path, 'rb') as inputfile:
			classifier = pickle.load(inputfile)
	print('classifier loaded')

	# evaluate (1, 3, 5, 10)
	# classifier.alphas = [classifier.alphas[2]]
	# classifier.clfs = [classifier.clfs[2]]
	# classifier.alphas = [classifier.alphas[4]]
	# classifier.clfs = [classifier.clfs[4]]
	# classifier.alphas = [classifier.alphas[9]]
	# classifier.clfs = [classifier.clfs[9]]
	# classifier.evaluate(trainset)
	# classifier.evaluate(testset)

	for i, dataset in enumerate([trainset, testset]):
		# random selected one face and one non face
		pos_index = random.randint(0, dataset.pos)
		neg_index = random.randint(dataset.pos + 1, dataset.pos + dataset.neg - 1)
		
		for iteration, clf in enumerate(classifier.clfs):
			if iteration + 1 not in (1, 3, 5, 10):
				continue

			best_feature_index = clf.feature_index
			window = classifier.window[best_feature_index]
			print(window)
			print('Threshold: ' + str(clf.threshold))
			rects_pos = gen_rects(window)
			rects_neg = gen_rects(window)

			fig, axs = plt.subplots(1, 2)
			axs[0].set_title('Iteration ' + str(iteration + 1) + ' on ' + dataset.name + 'set\n Face')
			axs[1].set_title('Iteration ' + str(iteration + 1) + ' on ' + dataset.name + 'set\n Non-Face')
			plt.subplots_adjust(wspace = 0.6)

			axs[0].imshow(dataset.samples[pos_index][0], cmap="viridis")
			axs[1].imshow(dataset.samples[neg_index][0], cmap="viridis")

			# Add the patch to the Axes
			for rect in rects_pos: 
				axs[0].add_patch(rect)
			for rect in rects_neg: 
				axs[1].add_patch(rect)	

			axs[0].set(xlabel='x', ylabel='y')
			axs[1].set(xlabel='x', ylabel='y')
			plt.show()

			extent_pos = axs[0].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
			extent_neg = axs[1].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
			
			# if i == 0:
			# 	fig.savefig('./figure/feature_train_pos_' + str(iteration + 1) + '.png', bbox_inches = extent_pos.expanded(1.1, 1.1))
			# 	fig.savefig('./figure/feature_train_neg_' + str(iteration + 1) + '.png', bbox_inches = extent_neg.expanded(1.1, 1.1))
			# else:
			# 	fig.savefig('./figure/feature_test_pos_' + str(iteration + 1) + '.png', bbox_inches = extent_pos.expanded(1.1, 1.1))
			# 	fig.savefig('./figure/feature_test_neg_' + str(iteration + 1) + '.png', bbox_inches = extent_neg.expanded(1.1, 1.1))
	return


if __name__ == "__main__":
	main()
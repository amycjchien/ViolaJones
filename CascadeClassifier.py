import numpy as np
import collections
import pickle
import math
from ViolaJones import ViolaJones 
from ViolaJones import RectangleRegion
from Dataset import Dataset

def save_object(obj, filename):
	with open(filename, 'wb') as output:  # Overwrites any existing file.
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def calII(img):
	# input: img (ndarrray)
	# output: integral_img(ndarray)
	row, col = img.shape
	integral_img = np.zeros((row, col))
	for i in range(row):
		for j in range(col):
			integral_img[i][j] = img[i][j]
			if i != 0:
				integral_img[i][j] += integral_img[i - 1][j]
			if j != 0:
				integral_img[i][j] += integral_img[i][j - 1]
			if i != 0 and j != 0:
				integral_img[i][j] -= integral_img[i - 1][j - 1]
	return integral_img

class CascadeClassifier():
	def __init__(self, max_round, max_cascade, testset):
		self.clfs = []
		# preset paramters
		self.d = .75 # TPR for each classifier
		self.f = .9 # FPR for each classifier
		self.F_target = 0.9 ** max_cascade # target FPR for the cascaded classifier
		self.D_target = .75 # target TPR for the cascaded classifier
		self.D = np.ones(max_cascade, dtype=np.float64) # initialised TPR
		self.F = np.ones(max_cascade, dtype=np.float64) # initialised FPR
		self.n = np.zeros(max_cascade) # classifiers in one cascade
		self.i = 0 # index of the number of cascade
		self.cascaded_classifiers = []
		self.max_cascade = max_cascade
		self.max_round = max_round
		self.testset = testset
		self.trainset = None
		self.training = None
		self.pos = 0
		self.neg = 0
		self.training_data = None
		self.window = None
		self.features = None
		self.X = None
		self.y = None

	def cal_feature(self, trainset, training, pos, neg):
		self.trainset = trainset
		self.training = training
		self.pos = pos
		self.neg = neg
		self.training_data = []
		for x in range(len(training)):
			self.training_data.append((calII(training[x][0]), training[x][1]))
			
		self.window, self.features = self.construct_features(self.training_data[0][0].shape)
		self.X, self.y = self.get_features(self.features, self.training_data)
		return 

	def test_cur_classfier(self, neg_training, neg_training_data, neg_X, list_classifiers):
		# return all the misclassified images to decrease the FPR
		new_neg_training = []
		new_neg_training_data = []
		new_neg_X = []
		hash_idx = {}

		for index, (ex, ex1) in enumerate(zip(neg_training, neg_training_data)):
			status = False
			for classifier in list_classifiers:
				# wrong
				if classifier.classify(ex[0]) == 1:
					status = True
				# correct
				else:
					status = False
					break
			if status:
				hash_idx[index] = 1
				new_neg_training.append((ex[0], 0))
				new_neg_training_data.append(ex1)
				new_neg_X.append(neg_X.T[index])

		new_neg_X = np.array(new_neg_X)
		new_neg_X = new_neg_X.T
		print(new_neg_X.shape)

		return new_neg_training, new_neg_training_data, new_neg_X

	def train(self):
		count = 0
		# Load Precalculated Feature
		new_pos_training, new_neg_training = [], []
		new_pos_training_data, new_neg_training_data = [], []
		for ex, ex1 in zip(self.training, self.training_data): 
			if ex[1] == 1:
				new_pos_training.append(ex)
				new_pos_training_data.append(ex1)

			else:
				new_neg_training.append(ex)
				new_neg_training_data.append(ex1)

		new_training = new_pos_training + new_neg_training
		new_training_data = new_pos_training_data + new_neg_training_data

		new_pos, new_neg = 0, 0
		new_X = np.zeros((len(self.features), len(self.training_data)))
		new_y = np.zeros((len(self.training_data), 1))
		for index, Y in enumerate(self.y):
			new_y[index] = Y
			if Y == 1:
				new_pos += 1
			else:
				new_neg += 1

		new_pos_X = np.zeros((len(self.features), new_pos))
		new_neg_X = np.zeros((len(self.features), new_neg))
		idx_pos = 0
		idx_neg = 0
		for index, x in enumerate(self.X.T):
			if new_y[index] == 1:
				new_pos_X.T[idx_pos] = x 
				idx_pos += 1
			else:
				new_neg_X.T[idx_neg] = x 
				idx_neg += 1
		new_X = np.concatenate((new_pos_X, new_neg_X), axis=1)

		while self.F[self.i] > self.F_target:
			# end when reaching maximum number of cascades
			if self.i == self.max_cascade or len(new_neg_training) == 0:
				break
			self.i += 1 # cascade 0 does not count 
			self.F[self.i] = self.F[self.i - 1]
			self.D[self.i] = self.D[self.i - 1]
			print("\ncascade no %d" % self.i)
			new_X = np.concatenate((new_pos_X, new_neg_X), axis=1)
			new_training = new_pos_training + new_neg_training
			new_training_data = new_pos_training_data + new_neg_training_data
			new_pos = len(new_pos_training)
			new_neg = len(new_neg_training)
			new_y = np.concatenate((np.ones(new_pos), np.zeros(new_neg)))
			clf = None
			while self.F[self.i] > self.f * self.F[self.i - 1] or self.D[self.i] < self.d * self.D[self.i - 1]:
				if count >= self.max_round:
					break
				self.n[self.i] += 1 # increase the number of features in the strong classifer
				print("\ntraining the strong classifier in cascade no %d" % self.i)
				if not clf:
					clf = ViolaJones(T = int(self.n[self.i]))
					clf.read_feature(new_X, new_y, new_training, new_training_data, self.features, new_pos, new_neg, self.trainset)
					clf.train('E')
				else:
					clf.T = int(self.n[self.i])
					clf.resume_train(clf.weights[-1], 'E')
				print("\nvalidating the strong classifier on the devset")
				TP, FN, FP, TN = clf.evaluate(self.trainset, -1)
				self.D[self.i] = TP / (TP + FN) # update TPR
				self.F[self.i] = FP / (FP + TN) # update FPR
				print('Detection Rate = ' + str(self.D[self.i]))
				print('FP Rate = ' + str(self.F[self.i]))
				count += 1
			
			self.clfs.append(clf)
			new_neg_training, new_neg_training_data, new_neg_X = self.test_cur_classfier(new_neg_training, new_neg_training_data, new_neg_X, [clf])
			self.save(self.i)
			if count >= self.max_round:
				break

		return

	def save(self, i):
		with open('./classifier_cascade_' + str(i) +".pkl", 'wb') as f:
			pickle.dump(self, f)

	
	def classify(self, image):
		for index, clf in enumerate(self.clfs):
			if clf.classify(image) == 0:
				return (index + 1, 0)
		return (index + 1, 1)

	def evaluate(self, dataset):
		TP = 0 # predict true and autual true
		TN = 0 # predict false and actual false
		FN = 0 # predict false and autual true
		FP = 0 # predict true and autual false
		total_sample = len(dataset.samples)
		correct = 0
		abandoned_images = collections.defaultdict(int)

		for sample, label in dataset.samples:
			layer, classified = self.classify(sample) 
			if classified == 0:
				abandoned_images[layer] += 1

			if classified == label:
				correct += 1
				if label == 1:
					TP += 1
				else:
					TN += 1
			else:
				if label == 1: # classified = 0
					FN += 1
				else:
					FP += 1

		print(dataset.name + "False Positive Rate: %d/%d (%f" % (FP, FP + TN, 100 * FP / (FP + TN)) + '%)')
		print(dataset.name + "False Negative Rate: %d/%d (%f" % (FN, FN + TP, 100 * FN / (FN + TP)) + '%)')
		print(dataset.name + "Accuracy: %d/%d (%f" % (correct, total_sample, 100 * correct / total_sample) + '%)')

	def construct_features(self, image_shape):
		height, width = image_shape
		features = []
		window = []
		# two vertical min_w: 2, min_h: 1, max_w: 18, max_h: 19, step_w: 2, step_h: 1
		min_w = 2
		min_h = 1
		max_w = width if width % 2 == 0 else width - 1
		max_h = height
		step_w = 2
		step_h = 1
		for i in range(height):
			for j in range(width):
				for r in range(min_h, max_h + step_h, step_h):
					for c in range(min_w, max_w + step_w, step_w):
						# check boundary
						if i + r - 1 < height and j + c - 1 < width:
							left = RectangleRegion((i, j), (i + r, j + c // 2))
							right = RectangleRegion((i, j + c // 2), (i + r, j + c))
							features.append(([left], [right]))
							window.append((1, i, j, r, c))

		# two horizontal min_h: 2, min_w: 1, max_w: 19, max_h: 18, step_w: 1, step_h: 2
		min_w = 1
		min_h = 2
		max_w = width
		max_h = height if height % 2 == 0 else height - 1
		step_w = 1
		step_h = 2
		for i in range(height):
			for j in range(width):
				for r in range(min_h, max_h + step_h, step_h):
					for c in range(min_w, max_w + step_w, step_w):
						# check boundary
						if i + r - 1< height and j + c - 1< width:
							top = RectangleRegion((i, j), (i + r // 2, j + c))
							bottom = RectangleRegion((i + r // 2, j), (i + r, j + c))
							features.append(([top], [bottom]))
							window.append((2, i, j, r, c))

		# three horizontal min_h: 3, min_w: 1, max_w: 19, max_h: 18, step_w: 1, step_h: 3
		min_w = 1
		min_h = 3
		max_w = width
		max_h = height
		while max_h % 3 != 0:
			max_h -= 1
		step_w = 1
		step_h = 3
		for i in range(height):
			for j in range(width):
				for r in range(min_h, max_h + step_h, step_h):
					for c in range(min_w, max_w + step_w, step_w):
						# check boundary
						if i + r - 1< height and j + c - 1< width:
							top = RectangleRegion((i, j), (i + 1 * (r // 3), j + c))
							middle = RectangleRegion((i + 1 * (r // 3), j), (i + 2 * (r // 3), j + c))
							bottom = RectangleRegion((i + 2 * (r // 3), j), (i + r, j + c))
							features.append(([top, bottom], [middle]))
							window.append((3, i, j, r, c))
		
		# three vertical min_h: 1, min_w: 3, max_w: 18, max_h: 19, step_w: 3, step_h: 1
		min_w = 3
		min_h = 1
		max_w = width
		max_h = height
		while max_w % 3 != 0:
			max_w -= 1
		step_w = 3
		step_h = 1
		for i in range(height):
			for j in range(width):
				for r in range(min_h, max_h + step_h, step_h):
					for c in range(min_w, max_w + step_w, step_w):
						# check boundary
						if i + r - 1 < height and j + c - 1 < width:
							left = RectangleRegion((i, j), (i + r, j + 1 * (c // 3)))
							middle = RectangleRegion((i, j + 1 * (c // 3)), (i + r, j + 2 * (c // 3)))
							right = RectangleRegion((i, j + 2 * (c // 3)), (i + r, j + c))
							features.append(([left, right], [middle]))
							window.append((4, i, j, r, c))

		# diagonal min_h: 2, min_w: 2, max_w: 18, max_h: 18, step_w: 2, step_h: 2
		min_w = 2
		min_h = 2
		max_w = width if width % 2 == 0 else width - 1
		max_h = height if height % 2 == 0 else height - 1
		step_w = 2
		step_h = 2
		for i in range(height):
			for j in range(width):
				for r in range(min_h, max_h + step_h, step_h):
					for c in range(min_w, max_w + step_w, step_w):
						# check boundary
						if i + r - 1 < height and j + c - 1< width:
							lefttop = RectangleRegion((i, j), (i + r // 2, j + c // 2))
							leftbottom = RectangleRegion((i + r // 2, j), (i + r, j + c // 2))
							righttop = RectangleRegion((i, j + c // 2), (i + r // 2, j + c))
							rightbottom = RectangleRegion((i + r // 2, j + c // 2), (i + r, j + c))
							features.append(([lefttop, rightbottom], [leftbottom, righttop]))
							window.append((5, i, j, r, c))

		return window, features

	def get_features(self, features, training_data):
		X = np.zeros((len(features), len(training_data)))
		y = np.zeros((len(training_data), 1))
		for i in range(len(training_data)):
			y[i] = training_data[i][1]
		i = 0
		for positive_regions, negative_regions in features:
			for j in range(len(training_data)):
				X[i][j] = sum([pos.calIIIndex(training_data[j][0]) for pos in positive_regions]) - sum([neg.calIIIndex(training_data[j][0]) for neg in negative_regions])
			i += 1
		return X, y
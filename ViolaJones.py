import numpy as np
import math

class RectangleRegion:
	def __init__(self, startPos, endPos):
		self.startPos = startPos
		self.endPos = endPos

	def calIIIndex(self, ii):
		# input: integral_img (ndarrray)
		# output: feature_value (float)
		lefttop = (self.startPos[0] - 1, self.startPos[1] - 1)
		righttop = (self.startPos[0] - 1, self.endPos[1] - 1)
		rightbottom = (self.endPos[0] - 1, self.endPos[1] - 1)
		leftbottom = (self.endPos[0] - 1, self.startPos[1] - 1)
		res = float(0)
		res += ii[lefttop] if lefttop[0] >= 0 and lefttop[1] >= 0 else 0
		res -= ii[righttop] if righttop[0] >= 0 and righttop[1] >= 0 else 0
		res -= ii[leftbottom] if leftbottom[0] >= 0 and leftbottom[1] >= 0 else 0
		res += ii[rightbottom] if rightbottom[0] >= 0 and rightbottom[1] >= 0 else 0
		return res

class WeakClassifier(object):
	def __init__(self, positive_regions, negative_regions, threshold, polarity, feature_index):
		self.positive_regions = positive_regions
		self.negative_regions = negative_regions
		self.threshold = threshold
		self.polarity = polarity
		self.feature_index = feature_index
	
	def classify_dp(self, X, index):
		return 1 if self.polarity * X[self.feature_index][index] < self.polarity * self.threshold else 0

	def classify(self, x):
		feature = lambda ii: sum([pos.calIIIndex(ii) for pos in self.positive_regions]) - sum([neg.calIIIndex(ii) for neg in self.negative_regions])
		return 1 if self.polarity * feature(x) < self.polarity * self.threshold else 0

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

class ViolaJones(object):
	def __init__(self, T):
		self.T = T
		self.alphas = []
		self.clfs = []
		self.weights = []
		self.best_feature_index = []
		self.X = None
		self.y = None
		self.features = None
		self.window = None
		self.training_data = None
		self.training = None
		self.trainset = None
		self.pos = 0
		self.neg = 0

	def load_feature(self, trainset, training, pos, neg):
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

	def read_feature(self, X, y, training, training_data, feature, pos, neg, trainset):
		self.trainset = trainset
		self.training = training
		self.pos = pos
		self.neg = neg
		self.training_data = training_data
		self.features = feature
		self.X = X 
		self.y = y
		return 

	def evaluate(self, dataset, threshold):
		TP = 0 # predict true and autual true
		TN = 0 # predict false and actual false
		FN = 0 # predict false and autual true
		FP = 0 # predict true and autual false
		total_sample = len(dataset.samples)
		correct = 0
		for sample, label in dataset.samples:
			if threshold == -1:
				classified = self.classify(sample) 
			else:
				classified = self.classify_threshold(sample, threshold) 

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

		print(dataset.name + ' Accuracy = ' + str(100 * correct / total_sample) + '%')
		print(dataset.name + ' False Positive = ' + str(100 * FP / (FP + TN)) + '%')
		print(dataset.name + ' False Negative = ' + str(100 * FN / (FN + TP)) + '%')
		return TP, FN, FP, TN

	def resume_train(self, weights, error_type):
		for t in range(len(self.alphas), self.T):
			print('T = ' + str(t + 1) + ' out of ' + str(self.T))
			weights = weights / np.sum(weights)
			weak_classifiers = self.construct_weak(self.X, self.y, self.features, weights)
			clf, error, accuracy = self.find_best(weak_classifiers, weights, self.training_data, error_type)
			if error == 0:
				epsilon = 0.01
				beta = (error + epsilon) / (1.0 - error + epsilon)
			else:
				beta = error / (1.0 - error)
			for i in range(len(accuracy)):
				weights[i] *= (beta ** (1 - accuracy[i]))
			alpha = math.log(1.0 / beta)
			self.alphas.append(alpha)
			self.clfs.append(clf)
			self.weights.append(weights)
			self.best_feature_index.append(clf.feature_index)
			TP, FN, FP, TN = self.evaluate(self.trainset, -1)

		print(self.alphas)
		print(self.best_feature_index)
		return
			
	def train(self, error_type):
		# initialization
		weights = np.zeros(len(self.training))
		for x in range(len(self.training)):
			if self.training[x][1] == 1:
				weights[x] = 1.0 / (2 * self.pos)
			else:
				weights[x] = 1.0 / (2 * self.neg)

		for t in range(self.T):
			print('T = ' + str(t + 1) + ' out of ' + str(self.T))
			weights = weights / np.sum(weights)
			weak_classifiers = self.construct_weak(self.X, self.y, self.features, weights)
			clf, error, accuracy = self.find_best(weak_classifiers, weights, self.training_data, error_type)
			if error == 0:
				epsilon = 0.01
				beta = (error + epsilon) / (1.0 - error + epsilon)
			else:
				beta = error / (1.0 - error)
			for i in range(len(accuracy)):
				weights[i] *= (beta ** (1 - accuracy[i]))
			alpha = math.log(1.0 / beta)
			self.alphas.append(alpha)
			self.clfs.append(clf)
			self.weights.append(weights)
			self.best_feature_index.append(clf.feature_index)
			TP, FN, FP, TN = self.evaluate(self.trainset, -1)

		print(self.alphas)
		print(self.best_feature_index)
		return

	def classify(self, image):
		total = float(0)
		ii = calII(image)
		for alpha, clf in zip(self.alphas, self.clfs):
			total += alpha * clf.classify(ii)
		return 1 if total >= 0.5 * sum(self.alphas) else 0	

	def classify_threshold(self, image, threshold):
		total = float(0)
		ii = calII(image)
		for alpha, clf in zip(self.alphas, self.clfs):
			total += alpha * clf.classify(ii)
		return 1 if total >= 0.5 * threshold else 0	

	def construct_weak(self, X, y, features, weights):
		total_pos, total_neg = float(0), float(0)
		for w, label in zip(weights, y):
			if label == 1:
				total_pos += w
			else:
				total_neg += w

		classifiers = []
		no_features = X.shape[0]
		for index, feature in enumerate(X):
			if index in self.best_feature_index:
				continue
			if len(classifiers) % 20000 == 0 and len(classifiers) != 0:
				print("Trained %d classifiers out of %d" % (len(classifiers), no_features))
			applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])
			pos_seen, neg_seen = 0, 0
			pos_weights, neg_weights = float(0), float(0)
			min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
			for w, f, label in applied_feature:
				error = min(neg_weights + total_pos - pos_weights, pos_weights + total_neg - neg_weights)
				if error < min_error:
					min_error = error
					best_feature_index = index
					best_feature = features[index]
					best_threshold = f
					best_polarity = 1 if pos_seen > neg_seen else -1
				if label == 1:
					pos_seen += 1
					pos_weights += w
				else:
					neg_seen += 1
					neg_weights += w
			clf = WeakClassifier(best_feature[0], best_feature[1], best_threshold, best_polarity, best_feature_index)
			classifiers.append(clf)
		return classifiers

	def find_best(self, classifiers, weights, training_data, error_type):
		best_clf, best_error, best_accuracy = None, float('inf'), None
		for clf in classifiers:
			error, accuracy = float(0), np.zeros((len(training_data), 1))
			index = 0
			for data, w in zip(training_data, weights):
				classified = clf.classify_dp(self.X, index)
				correctness = abs(classified - data[1])
				accuracy[index] = correctness
				if error_type == 'E':
					error += w * correctness
				elif error_type == 'FN':
					if int(classified) == 0 and int(data[1]) == 1:
						error += w * 1
				elif error_type == 'FP':
					if int(classified) == 1 and int(data[1]) == 0:
						error += w * 1
				elif error_type == 'E+FN':
					error += 0.2 * w * correctness
					if int(classified) == 0 and int(data[1]) == 1:
						error += 0.8 * w * 1
				elif error_type == 'E+FP':
					error += 0.2 * w * correctness
					if int(classified) == 1 and int(data[1]) == 0:
						error += 0.8 * w * 1
				index += 1
			if error < best_error:
				best_clf, best_error, best_accuracy = clf, error, accuracy
		return best_clf, best_error, best_accuracy

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
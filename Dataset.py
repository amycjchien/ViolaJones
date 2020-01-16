import os
import cv2
import pickle

class Dataset(object):
	def __init__(self, name):
		self.samples = None
		self.name = name
		self.pos = 0
		self.neg = 0

	def process_samples(self, pos_samples, neg_samples):
		self.samples = pos_samples + neg_samples
		self.pos = len(pos_samples)
		self.neg = len(neg_samples)

	def process_data(self, foldername, dirname):
		data = []
		pos = 0
		neg = 0
		for subdirname in ('faces', 'non-faces'):
			filenames = os.listdir('%s/%s/%s' % (foldername, dirname, subdirname))
			for filename in filenames:
				fullpath = '%s/%s/%s/%s' % (foldername, dirname, subdirname, filename)
				img = cv2.imread(fullpath, 0) # grayscale
				# img = cv2.resize(img, (6, 6), interpolation = cv2.INTER_AREA)
				if subdirname == 'faces':
					tmp = (img, 1)
					pos += 1
				elif subdirname == 'non-faces':
					tmp = (img, 0)
					neg += 1
				data.append(tmp)
		self.samples = data
		self.pos = pos
		self.neg = neg	
		return
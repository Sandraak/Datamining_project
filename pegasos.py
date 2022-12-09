import numpy as np
import matplotlib.pyplot as plt


class Pegasos:
	def __init__(self, n_iter:int = 10, lambda1:int = 1) -> None:
		# Number of iterations
		self.n_iter: int = n_iter
		self.lambda1: int = lambda1
		self.X: np.ndarray
		self.y: np.ndarray
		# Class index for index of each sample
		self.classes: dict = {}
		# Class names for every class number
		self.class_nums_names: dict = {}
		# Class number for every class name
		self.class_names_nums: dict = {}
		# Weights
		self.w: np.ndarray

	def fit(self, X:np.ndarray, y:np.ndarray) -> None:
		"""Fit data on pegasos svm.

		Args:
			X (np.ndarray): Features
			y (np.ndarray): Class labels
		"""
		self.X = X
		self.y = y

		self.find_classes()
		n_samples, n_features = self.X.shape[0], self.X.shape[1]

		# Initialize the weight vector for the perceptron with zeros
		self.w = np.zeros(n_features)

		for i in range(self.n_iter):
			learning_rate = 1. / (self.lambda1*(i+1))
			rand_sample_index = np.random.choice(n_samples, 1)[0]
			sample_X, sample_y = self.X[rand_sample_index], self.classes[rand_sample_index]
			score = self.w.dot(sample_X)
			
			if sample_y*score < 1:
				self.w = (1 - learning_rate*self.lambda1)*self.w + learning_rate*sample_y*sample_X
			else:
				self.w = (1 - learning_rate*self.lambda1)*self.w

		# TODO look at improving the magnitude?
		print(f"magnitude: {2/np.linalg.norm(self.w)}")


	# def fit2(self, X:np.ndarray, y:np.ndarray, c:float = 0.001) -> None:	
	# 	"""_summary_

	# 	Args:
	# 		X (np.ndarray): _description_
	# 		y (np.ndarray): _description_
	# 		c (float): learning rate?
	# 	"""
	# 	self.find_classes()
	# 	n_samples, n_features = self.X.shape[0], self.X.shape[1]

	# 	# Initialize the weight vector for the perceptron with zeros
	# 	self.w = np.zeros(n_features)

	# 	for i in range(1, self.n_iter):
	# 		# n_iter = epochs
	# 		rand_sample_index = np.random.choice(n_samples, 1)[0]
	# 		nt = 1/(c*i)
	# 		X_sample = self.X[rand_sample_index]
	# 		next_w = np.zeros(len(X_sample))
	# 		dot_product = 0

	# 		for j in range(len(X_sample)):
	# 			dot_product += self.w[j]*X_sample[j]

	# 		if dot_product*self.classes[rand_sample_index] < 1:
	# 			for j in range(len(X_sample)):
	# 				next_w[j] = self.w[j] - nt*c*self.w[j] + nt*self.classes[rand_sample_index]*X_sample[j]

	# 		else:
	# 			for j in range(len(X_sample)):
	# 				next_w[j] = self.w[j] - nt*c*self.w[j]
        

	def find_classes(self) -> None:
		"""Assign class numbers to classes and save a dictionary of both ways."""
		# TODO adapt for multiclass
		class_num = -1

		for i, label in enumerate(self.y):
			if label not in self.class_nums_names.values():
				self.class_nums_names[class_num] = label
				self.class_names_nums[label] = class_num
				class_num += 2
			self.classes[i] = self.class_names_nums[label]

	def predict(self, X:np.ndarray) -> list:
		"""Predict on features in X.

		Args:
			X (np.ndarray): Features for every sample to predict on

		Returns:
			list: class labels for every sample in X
		"""		
		# TODO adapt for multiclass
		predicted_labels = []
		for sample in X:
			dot_product = 0
			for i in range(len(sample)):
				dot_product += self.w[i] * sample[i]
			if dot_product >= 0:
				predicted_labels.append(1)
			else:
				predicted_labels.append(-1)

		return [self.class_nums_names[label_num] for label_num in predicted_labels]

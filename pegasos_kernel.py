
from random import randint

import numpy as np
import matplotlib.pyplot as plt


class PegasosKernel:
	def __init__(self, n_iter:int = 2000, lambda1:int = 1) -> None:
		# Number of iterations
		self.n_iter: int = n_iter
		# Lambda parameter
		self.lambda1: int = lambda1
		# Weights
		self.w: np.ndarray
		# Training data
		self.X: np.ndarray
		self.y: np.ndarray
		# Class index for index of each sample
		self.classes: dict = {}
		# Class names for every class number
		self.class_nums_names: dict = {}
		# Class number for every class name
		self.class_names_nums: dict = {}

	def kernel_function(self, X:np.ndarray, y:int) -> float:
		"""Transform X and y to a scalar.

		Args:
			X (np.ndarray): Set of attribute values for one record.
			y (int): Label for the record X.

		Returns:
			float: The scalar of X and y.
		"""
		mean = np.linalg.norm(X - y)**2
		variance = 1
		return np.exp(-mean/(2*variance))

	def fit(self, X:np.ndarray, y:np.ndarray) -> None:
		"""Fit training data on pegasos svm.

		Args:
			X (np.ndarray): Records.
			y (np.ndarray): Class labels.
		"""
		self.X = X
		self.y = y

		self.find_classes()
		n_samples, n_features = self.X.shape[0], self.X.shape[1]

		self.w = np.zeros(n_samples)

		for _ in range(self.n_iter):
			rand_sample_index = randint(0, n_samples - 1)
			decision = 0
			for j in range(n_samples):
				decision += self.w[j] * self.classes[rand_sample_index] * self.kernel_function(self.X[rand_sample_index], self.classes[j])
			decision *= self.classes[rand_sample_index]/self.lambda1
			if decision < 1:
				self.w[rand_sample_index] += 1

	def find_classes(self) -> None:
		"""Assign class numbers to classes and save a dictionary of both ways."""
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
			X (np.ndarray): Features for every sample to predict on.

		Returns:
			list: class labels for every sample in X.
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

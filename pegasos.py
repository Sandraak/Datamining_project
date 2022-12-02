import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt


class Pegasos:
	def __init__(self, epochs:int = 10, lambda1:int = 1) -> None:
		self.epochs: int = epochs
		self.lambda1: int = lambda1
		self.X: np.ndarray
		self.y: np.ndarray
		self.classes: dict = {}
		self.w: np.ndarray

	def fit(self, X:np.ndarray, y:np.ndarray):
		"""Fit data on pegasos svm.

		Args:
			X (np.ndarray): Features
			y (np.ndarray): Class labels (-1 for negative class and +1 for positive class)
		"""
		self.X = X
		self.y = y

		errors = []
		class_nums = self.find_classes()
		n_samples, n_features = self.X.shape[0], self.X.shape[1]

		# Initialize the weight vector for the perceptron with zeros
		self.w = np.zeros(n_features)

		for epoch in range(self.epochs):
			error = 0
			learning_rate = 1. / (self.lambda1*(epoch+1))
			rand_sample_index = np.random.choice(n_samples, 1)[0]
			sample_X, sample_y = self.X[rand_sample_index], class_nums[rand_sample_index]
			score = self.w.dot(sample_X)

			if class_nums*score < 1:
				self.w = (1 - eta*self.lambda1)*self.w + eta*class_nums*self.x
				error = 1
			else:
				self.w = (1 - eta*self.lambda1)*self.w

			errors.append(error)

		self.plot_errors(errors)

	def find_classes(self) -> np.ndarray:
		class_nums: list = []
		for i, label in enumerate(self.y):
			self.classes[i] = label 
			class_nums.append(i - 1)
		return class_nums


	def plot_errors(self, errors):
		plt.plot(errors, '|')
		plt.ylim(0.5,1.5)
		plt.axes().set_yticklabels([])
		plt.xlabel('Epoch')
		plt.ylabel('Misclassified')
		plt.show()

# Pegasos(epochs=20, lambda1=3)
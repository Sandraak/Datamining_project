import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


from pegasos import Pegasos
from pegasos_kernel import PegasosKernel
from cheat_svm import CheatSVM


def main():
	X_train, X_test, y_train, y_test = prepare_data("codon_usage.csv")
	
	# print("PEGASOS normal (simple)")
	# model = Pegasos(n_iter=100*len(X_train), lambda1=1)
	# model.fit(X_train, y_train)
	# y_pred = model.predict(X_test)
	# print(accuracy_score(y_test, y_pred))
	# print(classification_report(y_test, y_pred))

	# print("PEGASOS kernel")
	# model = PegasosKernel(n_iter=2000, lambda1=1)
	# model.fit(X_train, y_train)
	# y_pred = model.predict(X_test)
	# print(accuracy_score(y_test, y_pred))
	# print(classification_report(y_test, y_pred))

	print("CheatSVM")
	model = CheatSVM()
		
	train_classes: list = []
	# Class names for every class number
	class_nums_names: dict = {}
	# Class number for every class name
	class_names_nums: dict = {}
	class_num = -1
	for i, label in enumerate(y_train):
		if label not in class_nums_names.values():
			class_nums_names[class_num] = label
			class_names_nums[label] = class_num
			class_num += 2
		train_classes.append(class_names_nums[label])

	test_classes: list = []
	# Class names for every class number
	class_nums_names: dict = {}
	# Class number for every class name
	class_names_nums: dict = {}
	class_num = -1
	for i, label in enumerate(y_test):
		if label not in class_nums_names.values():
			class_nums_names[class_num] = label
			class_names_nums[label] = class_num
			class_num += 2
		test_classes.append(class_names_nums[label])

	model.fit(X_train, train_classes)
	y_pred = model.predict(X_test)
	print(accuracy_score(test_classes, y_pred))
	print(classification_report(test_classes, y_pred))

	print("SVM")
	clf = svm.SVC()
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	print(accuracy_score(y_test, y_pred))
	print(classification_report(y_test, y_pred))


def prepare_data(filename: str) -> tuple:
	"""Retrieve data for training and validation.

	Args:
		filename (_type_): Filename with the data for training.

	Returns:
		tuple: Train and test data.
	"""
	data = pd.read_csv(filename, low_memory=False)
	data.fillna(0,inplace=True)
	data = data[data['Kingdom'].isin(("bct", "vrl"))]
	X = data.iloc[:,-64:].to_numpy()
	y = data.iloc[:, 0].to_numpy()
	print(X.shape)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	return X_train, X_test, y_train, y_test


main()

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


from pegasos import Pegasos
from pegasos_kernel import PegasosKernel


def main():
	X_train, X_test, y_train, y_test = prepare_data("codon_usage.csv")
	# model = Pegasos(n_iter=10, lambda1=1)
	
	# print("PEGASOS normal (simple)")
	# model = Pegasos(n_iter=100*len(X_train), lambda1=1)
	# model.fit(X_train, y_train)
	# y_pred = model.predict(X_test)
	# print(accuracy_score(y_test, y_pred))
	# print(classification_report(y_test, y_pred))

	print("PEGASOS kernel")
	model = PegasosKernel(n_iter=2000, lambda1=1)
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	print(accuracy_score(y_test, y_pred))
	print(classification_report(y_test, y_pred))

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

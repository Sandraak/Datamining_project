import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from pegasos import Pegasos


def main():
	X_train, X_test, y_train, y_test = prepare_data("codon_usage.csv")
	
	print("PEGASOS")
	model = Pegasos(n_iter=10*len(X_train), lambda1=1)
	model.fit(X_train, y_train, X_test, y_test)
	y_pred = model.predict(X_test)
	print(accuracy_score(y_test, y_pred))
	print(classification_report(y_test, y_pred))

	plot_accuracy(
		model.training_accuracy_x, model.training_accuracy_y,
		model.validation_accuracy_x, model.validation_accuracy_y
		)
	plot_magnitude(model.magnitude_x, model.magnitude_y)


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

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	return X_train, X_test, y_train, y_test


def plot_accuracy(train_x, train_y, val_x, val_y) -> None:
	plt.plot(train_x, train_y, label='train')
	plt.plot(val_x, val_y, label='val')
	plt.title('Training accuracy')
	plt.xlabel('iteration')
	plt.ylabel('accuracy')
	plt.legend()
	plt.show()


def plot_magnitude(x: list, y: list) -> None:
	plt.plot(x, y)
	plt.title('Magnitude')
	plt.xlabel('iteration')
	plt.ylabel('magnitude')
	plt.show()


main()

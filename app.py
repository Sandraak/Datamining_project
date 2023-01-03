import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy import stats
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from pegasos import Pegasos
from pegasos_kernel import PegasosKernel


def main() -> None:
	X_train, X_test, y_train, y_test = prepare_data("codon_usage.csv")

	outfile = "output_only_pegasos.txt"

	train_pegasos(X_train, X_test, y_train, y_test, outfile, make_plots=False)

	# train_kernalized_pegasos(X_train, X_test, y_train, y_test, outfile)

	# train_svm(X_train, X_test, y_train, y_test, outfile)

def train_pegasos(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, outfile: str, make_plots: bool) -> None:
	"""Train multiple pegasos models with and without bias.

	Args:
		X_train (np.ndarray): Train records.
		X_test (np.ndarray): Test records.
		y_train (np.ndarray): Train labels.
		y_test (np.ndarray): Test labels.
		outfile (str): File to write performance scores to.
		make_plots (bool): If plots should be made of the training and validation accuracy.
	"""
	accuracy_no_bias = []
	accuracy_bias = []

	for i in range(50):
		bias = False
		model = Pegasos(n_iter=10*len(X_train), lambda1=1)
		model.fit(X_train, y_train, X_test, y_test, bias=bias, make_plots=make_plots)
		y_pred = model.predict(X_test)
		acc = accuracy_score(y_test, y_pred)
		with open(outfile, "a") as file:
			file.write(f"Pegasos model {i}, bias={bias}")
			file.write("\n")
			file.write(f"Bias: {model.b}")
			file.write("\n")
			file.write("Accuracy: " + str(acc))
			file.write("\n")
			file.write("Magnitude: " + str(2/np.linalg.norm(model.w)))
			file.write("\n")
			file.write(str(classification_report(y_test, y_pred)))
			file.write("\n")
			file.write("_"*20)
			file.write("\n")
		accuracy_no_bias.append(acc)
		if make_plots:
			plot_accuracy(
				model.training_accuracy_x, model.training_accuracy_y,
				model.validation_accuracy_x, model.validation_accuracy_y,
				f"Model_{i}_bias_{bias}_accuracy.png"
			)
			plot_magnitude(model.magnitude_x, model.magnitude_y, f"Model_{i}_bias_{bias}_magnitude.png")

	for i in range(50):
		bias = True
		model = Pegasos(n_iter=10*len(X_train), lambda1=1)
		model.fit(X_train, y_train, X_test, y_test, bias=bias, make_plots=make_plots)
		y_pred = model.predict(X_test)
		acc = accuracy_score(y_test, y_pred)
		with open(outfile, "a") as file:
			file.write(f"Pegasos model {i}, bias={bias}")
			file.write("\n")
			file.write(f"Bias: {model.b}")
			file.write("\n")
			file.write("Accuracy: " + str(acc))
			file.write("\n")
			file.write("Magnitude: " + str(2/np.linalg.norm(model.w)))
			file.write("\n")
			file.write(str(classification_report(y_test, y_pred)))
			file.write("\n")
			file.write("_"*20)
			file.write("\n")
		accuracy_bias.append(acc)
		if make_plots:
			plot_accuracy(
				model.training_accuracy_x, model.training_accuracy_y,
				model.validation_accuracy_x, model.validation_accuracy_y,
				f"Model_{i}_bias_{bias}_accuracy.png"
			)
			plot_magnitude(model.magnitude_x, model.magnitude_y, f"Model_{i}_bias_{bias}_magnitude.png")

	with open(outfile, "a") as file:
		file.write("T-test (equal_var = True)")
		file.write("\n")
		file.write(str(stats.ttest_ind(accuracy_no_bias, accuracy_bias)))
		file.write("\n")

		file.write("\n")
		file.write(f"Average accuracy no bias: {sum(accuracy_no_bias)/len(accuracy_no_bias)}")
		file.write("\n")
		file.write(f"Average accuracy with bias: {sum(accuracy_bias)/len(accuracy_bias)}")
		file.write("\n")
		
def train_kernalized_pegasos(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, outfile: str) -> None:
	"""Train a kernalized pegasos model.

	Args:
		X_train (np.ndarray): Train records.
		X_test (np.ndarray): Test records.
		y_train (np.ndarray): Train labels.
		y_test (np.ndarray): Test labels.
		outfile (str): File to write performance scores to.
	"""
	model = PegasosKernel(n_iter=10*len(X_train), lambda1=1)
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	acc = accuracy_score(y_test, y_pred)
	with open(outfile, "a") as file:
		file.write(f"Kernalized pegasos model")
		file.write("\n")
		file.write("Accuracy: " + str(acc))
		file.write("\n")
		file.write(str(classification_report(y_test, y_pred)))
		file.write("\n")
		file.write("_"*20)
		file.write("\n")


def train_svm(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, outfile: str) -> None:
	"""Train an SVM.

	Args:
		X_train (np.ndarray): Train records.
		X_test (np.ndarray): Test records.
		y_train (np.ndarray): Train labels.
		y_test (np.ndarray): Test labels.
		outfile (str): File to write performance scores to.
	"""
	clf = svm.SVC()
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)

	acc = accuracy_score(y_test, y_pred)
	with open(outfile, "a") as file:
		file.write(f"SVM model")
		file.write("\n")
		file.write("Accuracy: " + str(acc))
		file.write("\n")
		file.write(str(classification_report(y_test, y_pred)))
		file.write("\n")
		file.write("_"*20)
		file.write("\n")


def prepare_data(filename: str) -> tuple:
	"""Retrieve data for training and validation.

	Args:
		filename (str): Filename with the data for training.

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


def plot_accuracy(train_x: list, train_y: list, val_x: list, val_y: list, filename: str) -> None:
	"""Plot the development of the training and validation accuracy.

	Args:
		train_x (list): Numbers of iterations for the x-axis of the training accuracy.
		train_y (list): Accuracy values for the y-axis of the training accuracy.
		val_x (list): Numbers of iterations for the x-axis of the validation accuracy.
		val_y (list): Accuracy values for the y-axis of the validation accuracy.
		filename (str): Output filename of the plot.
	"""
	plt.plot(train_x, train_y, label='train')
	plt.plot(val_x, val_y, label='val')
	plt.title('Training and validation accuracy')
	plt.xlabel('iteration')
	plt.ylabel('accuracy')
	plt.legend()
	plt.savefig(filename)
	plt.show()


def plot_magnitude(x: list, y: list, filename: str) -> None:
	"""Plot the development of the magnitude.

	Args:
		x (list): Numbers of iterations for the x-axis of the plot.
		y (list): Magnitude values for the y-axis of the plot.
		filename (str): Output filename of the plot.
	"""
	plt.plot(x, y)
	plt.title('Magnitude')
	plt.xlabel('iteration')
	plt.ylabel('magnitude')
	plt.savefig(filename)
	plt.show()


main()

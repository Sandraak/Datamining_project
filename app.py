import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pegasos import Pegasos


def main():
	X_train, X_test, y_train, y_test = prepare_data("codon_usage.csv")
	model = Pegasos(n_iter=len(X_train)*100, lambda1=1)
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	print(accuracy_score(y_test, y_pred))

def prepare_data(filename):
	data = pd.read_csv(filename, low_memory=False)
	data.fillna(0,inplace=True)
	data = data[data['Kingdom'].isin(("bct", "vrl"))]

	X = data.iloc[:,-64:].to_numpy()
	y = data.iloc[:, 0].to_numpy()

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	return X_train, X_test, y_train, y_test


main()
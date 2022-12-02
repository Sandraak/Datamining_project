import pandas as pd
from sandra_app import Pegasos

def main():
    X, y = prepare_data("codon_usage.csv")
    model = Pegasos(epochs=10, lambda1=1)
    model.fit(X, y)
    predicted_label = model.predict(X)
    model.accuracy(y, predicted_label)
    # model.predict


def prepare_data(filename):
	data = pd.read_csv(filename, low_memory=False)
	data.fillna(0,inplace=True)
	data = data[data['Kingdom'].isin(("bct", "vrl"))]

	X = data.iloc[:,-64:].to_numpy()
	y = data.iloc[:, 0].to_numpy()

	return X, y

main()
import numpy as np

class Pegasos:
    def __init__(self, epochs: int = 10, lambda1: int = 1) -> None:
        self.w: np.array
        self.x: np.array
        self.y: np.array
        self.epochs: int = epochs
        self.lambda1: int = lambda1
        pass

    def fit(self, data:np.array, label:np.array) -> None:
        Y = data['Y']
        Yn = [np.sign(label, self.positive_class) for label in Y]
        X = data['X']
        m_samples, n_features = self.X.shape[0], self.X.shape[1]
        self.w = np.zeros( n_features)
        for sample in range(self.epochs):
            learning_rate = 1 / (self.lambda1*(sample+1))
            random_sample_index = np.random.choice(m_samples, 1)[0]
            self.x, self.y = X[random_sample_index], Yn[random_sample_index]
            score = self.w.dot(self.x)
            if self.y*score <1:
                self.w = (1 - learning_rate*self.lambda1)*self.w + learning_rate*self.y*self.x
            else:
                self.w = (1- learning_rate*self.lambda1)*self.w
        print("fit")

    def predict(data) -> np.array:
        print("predict")

    def accuracy(label, predicted_label) -> float:
        print("accuracy")
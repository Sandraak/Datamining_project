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

    def predict(self,data) -> np.array:
        predicted_labels:np.array
        xi:np.array
        for i in range(len(data)):
                xi = data[i]
                dot_product:float = 0.0
                for j in range(len(xi)):
                    dot_product += self.w[j]*xi[j]
                if(dot_product >= 0):
                    predicted_labels.append(1)
                else:
                    predicted_labels.append(-1)
        return predicted_labels


    def accuracy(labels:np.array, predicted_labels:np.array) -> float:
        correct_pred:int = 0
        for i in range(len(predicted_labels)):
            if labels[i] == predicted_labels[i]:
                correct_pred += 1

        return float(correct_pred/len(labels))


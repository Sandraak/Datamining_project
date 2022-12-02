import numpy as np
import matplotlib.pyplot as plt
class Pegasos:
    def __init__(self, epochs:int = 10, lambda1:int = 1) -> None:
        self.epochs: int = epochs
        self.lambda1: int = lambda1
        self.X: np.ndarray
        self.y: np.ndarray
        self.classes: dict = {}
        self.class_nums_names: dict = {}
        self.class_names_nums: dict = {}
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
        self.find_classes()
        n_samples, n_features = self.X.shape[0], self.X.shape[1]

        # Initialize the weight vector for the perceptron with zeros
        self.w = np.zeros(n_features)

        for epoch in range(self.epochs):
            error = 0
            learning_rate = 1. / (self.lambda1*(epoch+1))
            rand_sample_index = np.random.choice(n_samples, 1)[0]
            sample_X, sample_y = self.X[rand_sample_index], self.classes[rand_sample_index]
            score = self.w.dot(sample_X)

            print(sample_y)
            if sample_y*score < 1:
                self.w = (1 - learning_rate*self.lambda1)*self.w + learning_rate*sample_y*sample_X
                error = 1
            else:
                self.w = (1 - learning_rate*self.lambda1)*self.w

            errors.append(error)

        print(self.w)

    def find_classes(self):
        class_num = -1

        for i, label in enumerate(self.y):
            if label not in self.class_nums_names.values():
                self.class_nums_names[class_num] = label
                self.class_names_nums[label] = class_num
                class_num += 2
            self.classes[i] = self.class_names_nums[label]


    def predict(self,data) -> list:
        predicted_labels:list = []
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
        print("predicted labels: ", predicted_labels)
        return predicted_labels

    def accuracy(self, labels:np.array, predicted_labels:list) -> float:
        correct_pred:int = 0
        for i in range(len(predicted_labels)):
            if labels[i] == predicted_labels[i]:
                correct_pred += 1
        print("labels: ", labels)
        accuracy:float = float(correct_pred/len(predicted_labels))
        print("accuracy:", accuracy)
        return accuracy
import numpy as np

from sklearn.metrics import accuracy_score


class Pegasos:
    def __init__(self, n_iter:int = 10, lambda1:int = 1) -> None:
        # Number of iterations
        self.n_iter: int = n_iter
        # Lambda parameter
        self.lambda1: int = lambda1
        # Weights
        self.w: np.ndarray
        # Bias
        self.b: float = 0
        # Training data
        self.X: np.ndarray
        self.y: np.ndarray
        # Class index for index of each sample
        self.classes: dict = {}
        # Class names for every class number
        self.class_nums_names: dict = {}
        # Class number for every class name
        self.class_names_nums: dict = {}
        # Training accuracy
        self.training_accuracy_x: list = []
        self.training_accuracy_y: list = []
        # Validation accuracy
        self.validation_accuracy_x: list = []
        self.validation_accuracy_y: list = []
        # Magnitude
        self.magnitude_x: list = []
        self.magnitude_y: list = []

    def fit(self, X:np.ndarray, y:np.ndarray, X_test:np.ndarray, y_test:np.ndarray, bias=True, make_plots=True) -> None:
        """Fit data on pegasos svm.

        Args:
            X (np.ndarray): Records.
            y (np.ndarray): Class labels.
        """
        self.X = X
        self.y = y

        self.find_classes()
        n_samples, n_features = self.X.shape[0], self.X.shape[1]

        # Initialize the weight vector for the perceptron with zeros
        self.w = np.zeros(n_features)

        for i in range(self.n_iter):
            learning_rate = 1. / (self.lambda1*(i+1))
            rand_sample_index = np.random.choice(n_samples, 1)[0]
            sample_X, sample_y = self.X[rand_sample_index], self.classes[rand_sample_index]
            if bias:
                linear_model = np.dot(sample_X, self.w) + self.b
            else:
                linear_model = np.dot(sample_X, self.w)
            
            if sample_y*linear_model >= 1:
                self.w = self.lambda1 * self.w
            else:
                self.w = (1 - learning_rate*self.lambda1)*self.w + learning_rate*sample_y*sample_X
                if bias:
                    self.b -= learning_rate * (- sample_y)

            if not i%25 and make_plots:
                self.get_training_accuracy(i)
                self.get_validation_accuracy(i, X_test, y_test)
                self.get_magnitude(i)

    def find_classes(self) -> None:
        """Assign class numbers to classes and save a dictionary of both ways."""
        class_num = -1

        for i, label in enumerate(self.y):
            if label not in self.class_nums_names.values():
                self.class_nums_names[class_num] = label
                self.class_names_nums[label] = class_num
                class_num += 2
            self.classes[i] = self.class_names_nums[label]

    def get_training_accuracy(self, current_iter: int) -> None:
        """Calculate the training accuracy.

        Args:
            current_iter (int): Current iteration of fitting the model used for the x-axis.
        """
        y_pred = self.predict(self.X)
        acc = accuracy_score(self.y, y_pred)
        self.training_accuracy_x.append(current_iter)
        self.training_accuracy_y.append(acc)

    def get_validation_accuracy(self, current_iter: int, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Calculate the validation accuracy

        Args:
            current_iter (int): Current iteration of fitting the model used for the x-axis.
            X_test (np.ndarray): Test records.
            y_test (np.ndarray): Test labels.
        """
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        self.validation_accuracy_x.append(current_iter)
        self.validation_accuracy_y.append(acc)

    def get_magnitude(self, current_iter: int) -> None:
        magnitude = 2/np.linalg.norm(self.w)
        self.magnitude_x.append(current_iter)
        self.magnitude_y.append(magnitude)

    def predict(self, X:np.ndarray, use_bias: bool = False) -> list:
        """Predict on records in X.

        Args:
            X (np.ndarray): Records for every sample to predict on.

        Returns:
            list: Class labels for every sample in X
        """		
        predicted_labels = []
        for sample in X:
            dot_product = 0
            for i in range(len(sample)):
                dot_product += self.w[i] * sample[i]
            if use_bias:
                dot_product += self.b
            if dot_product >= 0:
                predicted_labels.append(1)
            else:
                predicted_labels.append(-1)

        return [self.class_nums_names[label_num] for label_num in predicted_labels]

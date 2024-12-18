import random
import math
import pandas as pd
from rich.console import Console
from sklearn.datasets import load_breast_cancer

class TumorClassificationCapstone:
    """
    A capstone project for tumor classification using the breast cancer dataset.
    https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
    """

    dataset: pd.DataFrame
    console: Console

    def __init__(self) -> None:
        self.logger = Console()
        raw_data = load_breast_cancer(as_frame=True)
        data = raw_data["data"]
        data["target"] = raw_data["target"]
        self.features = data.iloc[:, :-1]
        self.labels = data["target"]

    def divide_data(self, inputs, targets, ratio=0.2):
        idx = list(range(len(inputs)))
        random.shuffle(idx)
        cutoff = int(len(inputs) * ratio)
        test_idx, train_idx = idx[:cutoff], idx[cutoff:]
        train_inputs = [inputs[i] for i in train_idx]
        test_inputs = [inputs[i] for i in test_idx]
        train_targets = [targets[i] for i in train_idx]
        test_targets = [targets[i] for i in test_idx]
        return train_inputs, test_inputs, train_targets, test_targets

    def scale(self, inputs):
        means = [sum(col) / len(col) for col in zip(*inputs)]
        variances = [math.sqrt(sum((item - avg) ** 2 for item in col) / len(col)) for col, avg in zip(zip(*inputs), means)]
        return [[(val - avg) / std if std > 0 else val for val, avg, std in zip(row, means, variances)] for row in inputs]

    def activation(self, value):
        return 1 / (1 + math.exp(-value))

    def gradient_descent(self, inputs, targets, rate=0.01, iters=1000):
        coeffs = [0] * len(inputs[0])
        intercept = 0
        for _ in range(iters):
            for idx in range(len(inputs)):
                linear_out = sum(c * v for c, v in zip(coeffs, inputs[idx])) + intercept
                pred = self.activation(linear_out)
                diff = pred - targets[idx]
                for j in range(len(coeffs)):
                    coeffs[j] -= rate * diff * inputs[idx][j]
                intercept -= rate * diff
        return coeffs, intercept

    def classify(self, inputs, coeffs, intercept):
        return [1 if self.activation(sum(c * v for c, v in zip(coeffs, row)) + intercept) >= 0.5 else 0 for row in inputs]

    def score(self, actual, predicted):
        return sum(a == p for a, p in zip(actual, predicted)) / len(actual)

    def execute(self):
        input_data = self.features.values.tolist()
        output_data = self.labels.tolist()
        scaled_inputs = self.scale(input_data)
        train_in, test_in, train_out, test_out = self.divide_data(scaled_inputs, output_data)
        params, offset = self.gradient_descent(train_in, train_out)
        preds = self.classify(test_in, params, offset)
        performance = self.score(test_out, preds)
        self.logger.print(f"[bold green]Model Accuracy: {performance * 100:.2f}%[/bold green]")

if __name__ == "__main__":
    experiment = TumorClassificationCapstone()
    experiment.execute()
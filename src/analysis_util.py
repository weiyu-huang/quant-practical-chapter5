import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class StockDirectionAnalyzer:
    def __init__(self, y_true_, y_pred_, price_=None):
        if price_ is not None and len(y_true_) != len(price_):
            raise ValueError("price needs to be the same length as y_true")
        self.y_true = y_true_
        self.y_pred = y_pred_
        self.price_ = price_

    def compute_success_rates(self):
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
        up_predictive_value = tp / (fp + tp) * 100  # precision
        down_predicitive_value = tn / (tn + fn) * 100
        accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
        print(f"""
        Up correct: {up_predictive_value:.2f}%;
        Down correct: {down_predicitive_value:.2f}%;
        Accuracy: {accuracy:.2f}%;
        """)
        return up_predictive_value, down_predicitive_value, accuracy

    def plot_decisions(self):
        if self.price_ is None:
            return
        plt.figure(figsize=(8, 8))
        plt.plot(np.where(self.y_pred == self.y_true)[0], self.price_[self.y_pred == self.y_true], 'go')
        plt.plot(np.where(self.y_pred != self.y_true)[0], self.price_[self.y_pred != self.y_true], 'ro')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.show()

    def analyze(self, plot=True):
        up_predictive_value, down_predicitive_value, accuracy = self.compute_success_rates()
        if plot:
            self.plot_decisions()
        return up_predictive_value, down_predicitive_value, accuracy


if __name__ == '__main__':
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0])
    price = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
    analyzer = StockDirectionAnalyzer(y_true, y_pred, price)
    analyzer.analyze()
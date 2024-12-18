from rich.console import Console
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


class MpgRegressionCapstone:
    """
    A capstone project for MPG regression using the mpg dataset.
    """

    data: pd.DataFrame
    logger: Console

    def __init__(self) -> None:
        self.data = sns.load_dataset("mpg")
        self.logger = Console()

    def execute(self) -> None:
        self.logger.print("[bold blue]Initiating Analysis...[/bold blue]")
        self.inspect()
        self.data = self.prepare()
        self.analyze()

    def inspect(self) -> None:
        self.logger.print("[bold yellow]Inspecting Data[/bold yellow]")
        self.logger.print(self.data.sample(5))
        self.logger.print(self.data.info())
        self.logger.print("[bold green]Summary Metrics:[/bold green]")
        self.logger.print(self.data.describe())
        num_data = self.data.select_dtypes(include=["number"])
        sns.heatmap(num_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation")
        plt.show()

    def prepare(self) -> pd.DataFrame:
        cleaned = self.data.dropna()
        if "origin" in cleaned.columns:
            cleaned = pd.get_dummies(cleaned, columns=["origin"], drop_first=True)
        return cleaned

    def analyze(self) -> None:
        self.logger.print("[bold yellow]Executing Model Pipeline[/bold yellow]")
        predictors = self.data.drop(columns=["mpg", "name"], errors="ignore")
        target = self.data["mpg"]
        train_features, test_features, train_target, test_target = train_test_split(
            predictors, target, test_size=0.25, random_state=42
        )
        regressor = RandomForestRegressor(random_state=42)
        regressor.fit(train_features, train_target)
        predictions = regressor.predict(test_features)
        mae_value = mean_absolute_error(test_target, predictions)
        rmse_value = np.sqrt(mean_squared_error(test_target, predictions))
        r2_value = r2_score(test_target, predictions)
        self.logger.print(f"[bold green]MAE: {mae_value:.2f}[/bold green]")
        self.logger.print(f"[bold green]RMSE: {rmse_value:.2f}[/bold green]")
        self.logger.print(f"[bold green]R^2: {r2_value:.2f}[/bold green]")
        plt.scatter(test_target, predictions, alpha=0.7)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted Values")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    pipeline = MpgRegressionCapstone()
    pipeline.execute()
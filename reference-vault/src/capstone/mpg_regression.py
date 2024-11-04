from rich.console import Console
import pandas as pd
import seaborn as sns


class MpgRegressionCapstone:
    """
    A capstone project for MPG regression using the mpg dataset.
    https://www.kaggle.com/code/devanshbesain/exploration-and-analysis-auto-mpg
    """

    dataset: pd.DataFrame
    console: Console

    def __init__(self) -> None:
        self.dataset = sns.load_dataset("mpg")
        self.console = Console()

    def run(self) -> None:
        self.console.print(self.dataset.head())
        self.dataset.info()

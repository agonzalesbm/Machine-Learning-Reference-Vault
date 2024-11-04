import pandas as pd
from sklearn.datasets import load_breast_cancer  # type: ignore
from rich.console import Console


class TumorClassificationCapstone:
    """
    A capstone project for tumor classification using the breast cancer dataset.
    https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
    """

    dataset: pd.DataFrame
    console: Console

    def __init__(self) -> None:
        self.dataset = load_breast_cancer(as_frame=True)
        self.console = Console()

    def run(self) -> None:
        self.console.print(self.dataset.data.head())
        self.dataset.data.info()

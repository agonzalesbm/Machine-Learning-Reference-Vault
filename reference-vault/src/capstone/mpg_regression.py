from rich.console import Console
from ucimlrepo import fetch_ucirepo
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
        # fetch dataset
        auto_mpg = fetch_ucirepo(id=9)

        # data (as pandas dataframes)
        X = auto_mpg.data.features
        y = auto_mpg.data.targets

        # metadata
        print(auto_mpg.metadata)

        # variable information
        print(auto_mpg.variables)

mpg_capstone = MpgRegressionCapstone()
mpg_capstone.run()
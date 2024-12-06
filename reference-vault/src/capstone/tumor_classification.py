import pandas as pd
from sklearn.datasets import load_breast_cancer  # type: ignore
from rich.console import Console
from ucimlrepo import fetch_ucirepo


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
        # fetch dataset
        breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

        # data (as pandas dataframes)
        X = breast_cancer_wisconsin_diagnostic.data.features
        y = breast_cancer_wisconsin_diagnostic.data.targets

        # metadata
        print(breast_cancer_wisconsin_diagnostic.metadata)

        # variable information
        print(breast_cancer_wisconsin_diagnostic.variables)
    
    
tumor_capstone = TumorClassificationCapstone()
tumor_capstone.run()
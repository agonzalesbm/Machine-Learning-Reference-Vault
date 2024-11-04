from capstone.mpg_regression import MpgRegressionCapstone
from capstone.tumor_classification import TumorClassificationCapstone


def main() -> None:
    mpg_capstone = MpgRegressionCapstone()
    mpg_capstone.run()

    tumor_capstone = TumorClassificationCapstone()
    tumor_capstone.run()


if __name__ == "__main__":
    main()

import pandas as pd


class WineNormalization:

    def __init__(self, filepath):

        print("Loading wine dataset...")

        self.df = pd.read_csv(filepath)

        print(f"Dataset loaded with shape: {self.df.shape}")


    def min_max_normalization(self):

        print("\nApplying Min-Max normalization...")

        data = self.df.select_dtypes(include=['int64','float64'])

        normalized = (data - data.min()) / (data.max() - data.min())

        print("Min-Max normalized data preview:")
        print(normalized.head())

        return normalized


    def z_score_normalization(self):

        print("\nApplying Z-score normalization...")

        data = self.df.select_dtypes(include=['int64','float64'])

        normalized = (data - data.mean()) / data.std()

        print("Z-score normalized data preview:")
        print(normalized.head())

        return normalized


    def decimal_scaling(self):

        print("\nApplying Decimal Scaling normalization...")

        data = self.df.select_dtypes(include=['int64','float64'])

        max_vals = data.abs().max()

        j = max_vals.apply(lambda x: len(str(int(x))))

        normalized = data / (10 ** j)

        print("Decimal scaled data preview:")
        print(normalized.head())

        return normalized


def run_normalization(filepath):

    model = WineNormalization(filepath)

    minmax = model.min_max_normalization()
    zscore = model.z_score_normalization()
    decimal = model.decimal_scaling()

    print("\nAll normalization techniques applied successfully.")

    return minmax, zscore, decimal


if __name__ == "__main__":

    run_normalization("winequality-red.csv")

import pandas as pd


class DatasetProfiler:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run(self):

        print("\n" + "=" * 60)
        print("DATASET PROFILE")
        print("=" * 60)

        # Shape
        print("\n[SHAPE]")
        print(f"Rows:    {self.df.shape[0]:,}")
        print(f"Columns: {self.df.shape[1]:,}")

        # Head
        print("\n[HEAD]")
        print(self.df.head())

        # Data types
        print("\n[INFO]")
        self.df.info()

        # Duplicates
        print("\n[DUPLICATES]")
        print(
            f"Duplicate rows: "
            f"{self.df.duplicated().sum():,}"
        )

        # Missing values
        print("\n[MISSING VALUES]")
        print(
            self.df.isnull()
            .sum()
            .sort_values(ascending=False)
        )
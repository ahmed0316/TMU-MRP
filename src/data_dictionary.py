import pandas as pd


class DataDictionary:

    def __init__(self, datasets):
        self.datasets = datasets

    def build(self):

        dictionary = []

        for name, df in self.datasets.items():

            for column in df.columns:

                dictionary.append({
                    "Dataset": name,
                    "Column": column,
                    "Data Type": str(df[column].dtype),
                    "Missing": int(df[column].isna().sum()),
                    "Unique Values": int(df[column].nunique())
                })

        return pd.DataFrame(dictionary)
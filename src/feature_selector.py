import pandas as pd
import numpy as np


class FeatureSelector:

    def __init__(self, df):
        self.df = df.copy()

    def remove_metadata(self):

        metadata = [
            "geo",
            "dguid",
            "vector",
            "coordinate",
            "status",
            "symbol",
            "uom",
            "uom_id",
            "scalar_factor",
            "scalar_id",
            "terminated"
        ]

        self.df.drop(
            columns=[c for c in metadata if c in self.df.columns],
            inplace=True,
            errors="ignore"
        )

    def remove_constant_columns(self):

        constant = [
            c for c in self.df.columns
            if self.df[c].nunique(dropna=False) <= 1
        ]

        self.df.drop(columns=constant, inplace=True)

    def remove_high_missing(self, threshold=0.80):

        missing = self.df.isna().mean()

        remove = missing[missing > threshold].index

        self.df.drop(columns=remove, inplace=True)

    def keep_numeric(self):

        keep = ["date"]

        numeric = self.df.select_dtypes(include=np.number).columns.tolist()

        keep.extend(numeric)

        self.df = self.df[keep]

    def run(self):

        self.remove_metadata()
        self.remove_constant_columns()
        self.remove_high_missing()
        self.keep_numeric()

        return self.df
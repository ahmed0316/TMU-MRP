import pandas as pd
import numpy as np


class FeatureEngineer:

    def __init__(self, df):
        self.df = df.copy()

    def convert_to_monthly(self):
    
        if "date" not in self.df.columns:
            return
    
        self.df["date"] = pd.to_datetime(self.df["date"])
    
        self.df = self.df.sort_values("date")
    
        numeric = (
            self.df
            .select_dtypes(include=np.number)
            .columns
            .tolist()
        )
    
        self.df = (
            self.df[["date"] + numeric]
            .set_index("date")
            .resample("MS")
            .mean()
        )
    
        fvi_cols = [
            c for c in self.df.columns
            if c.startswith("fvi_")
        ]
    
        self.df[fvi_cols] = self.df[fvi_cols].ffill()
        self.df = self.df.reset_index()

    def create_date_features(self):

        if "date" not in self.df.columns:
            return

        self.df["year"] = self.df["date"].dt.year
        self.df["quarter"] = self.df["date"].dt.quarter
        self.df["month"] = self.df["date"].dt.month

        self.df["days_since_start"] = (
            self.df["date"] -
            self.df["date"].min()
        ).dt.days

    def create_lag_features(self):

        numeric = self.df.select_dtypes(include=np.number).columns

        for col in numeric:

            self.df[f"{col}_lag1"] = self.df[col].shift(1)
            self.df[f"{col}_lag3"] = self.df[col].shift(3)
            self.df[f"{col}_lag6"] = self.df[col].shift(6)

    def create_growth_features(self):

        numeric = self.df.select_dtypes(include=np.number).columns

        for col in numeric:

            self.df[f"{col}_pct_change"] = (
                self.df[col].pct_change()
            )

    def create_rolling_features(self):

        numeric = self.df.select_dtypes(include=np.number).columns

        for col in numeric:

            self.df[f"{col}_rolling3"] = (
                self.df[col]
                .rolling(3)
                .mean()
            )

            self.df[f"{col}_rolling12"] = (
                self.df[col]
                .rolling(12)
                .mean()
            )

    def engineer(self):

        #convert to monthly first!
        self.convert_to_monthly()

        self.create_date_features()
        self.create_lag_features()
        self.create_growth_features()
        self.create_rolling_features()

        return self.df
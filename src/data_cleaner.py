import re
import pandas as pd


class DataCleaner:

    def __init__(self):
        pass

    def clean_columns(self, df):

        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("-", "_")
            .str.replace(r"[()]", "", regex=True)
        )

        return df

    def clean_strings(self, df):

        for col in df.select_dtypes(include="object"):

            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
            )

        return df

    def convert_numeric(self, df):

        for col in df.columns:
            if df[col].dtype == object:

                x = (
                    df[col]
                    .str.replace(",", "", regex=False)
                    .str.replace("%", "", regex=False)
                )

                converted = pd.to_numeric(x, errors="ignore")
                df[col] = converted

        return df

    def convert_dates(self, df):
    
        for col in df.columns:
    
            if "date" in col:
    
                series = df[col]
                numeric = pd.to_numeric(series, errors="coerce")
                year_mask = numeric.between(1900, 2100)
    
                if year_mask.mean() > 0.80:
    
                    df[col] = pd.to_datetime(
                        numeric.astype("Int64").astype(str),
                        format="%Y",
                        errors="coerce"
                    )
    
                else:
    
                    df[col] = pd.to_datetime(
                        series,
                        errors="coerce"
                    )
    
        return df

    def remove_empty(self, df):

        df = df.dropna(axis=1, how="all")
        df = df.dropna(axis=0, how="all")

        return df

    def remove_duplicates(self, df):
        return df.drop_duplicates()

    def clean(self, df):
    
        df = self.clean_columns(df)
        df = self.remove_empty(df)
        df = self.clean_strings(df)
  
        df = self.convert_dates(df)
    
        df = self.convert_numeric(df)
        df = self.remove_duplicates(df)
    
        return df
from pathlib import Path
import pandas as pd


class DataProfiler:

    def __init__(self, datasets):
        self.datasets = datasets

    def profile(self):

        profiles = []

        for name, df in self.datasets.items():
            rows, cols = df.shape
            missing = df.isna().sum().sum()
            duplicate_rows = df.duplicated().sum()
            numeric = len(df.select_dtypes(include="number").columns)
            categorical = len(df.select_dtypes(exclude="number").columns)
            date_columns = []

            date_columns = []
            
            for c in df.columns:
            
                if "date" in c.lower():
                    date_columns.append(c)
            
                elif df[c].dtype == object:
                    parsed = pd.to_datetime(df[c], errors="coerce")

                    if parsed.notna().mean() > 0.8:
                        date_columns.append(c)

            profiles.append({
                "Dataset": name,
                "Rows": rows,
                "Columns": cols,
                "Numeric Columns": numeric,
                "Categorical Columns": categorical,
                "Missing Values": missing,
                "Duplicate Rows": duplicate_rows,
                "Date Columns": ", ".join(date_columns)

            })

        return pd.DataFrame(profiles)

    def save(self, output_path):
        profile = self.profile()
        profile.to_csv(output_path, index=False)

        return profile
import pandas as pd


class DataMerger:

    def __init__(self, datasets):
        self.datasets = datasets

    def merge(self):

        merged = None

        #datasets that should not be included in the time-series modeling merge
        exclude = ["11100021"]

        for name, df in self.datasets.items():

            #skip datasets shouldnt use for the time-series merge
            if name in exclude:
                print(f"Skipping {name} from time-series merge")
                continue

            df = df.copy()

            #standardize date column name
            if "ref_date" in df.columns:
                df = df.rename(columns={"ref_date": "date"})

            # only merge datasets containing a date
            if "date" not in df.columns:
                continue

            # safety check: each dataset used in the time-series merge
            # should contain no more than one observation per date
            duplicate_dates = df["date"].duplicated().sum()

            if duplicate_dates > 0:
                print(
                    f"WARNING: {name} has "
                    f"{duplicate_dates} duplicate date rows "
                    f"and will be skipped."
                )
                continue

            if merged is None:
                merged = df.copy()

            else:
                merged = pd.merge(
                    merged,
                    df,
                    on="date",
                    how="outer",
                    suffixes=("", "_" + name),
                    validate="one_to_one"
                )

        #sort
        if merged is not None:
            merged = merged.sort_values("date").reset_index(drop=True)

        return merged
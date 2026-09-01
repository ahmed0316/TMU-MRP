from pathlib import Path
import pandas as pd


class DataLoader:

    def __init__(self, data_folder):
        self.data_folder = Path(data_folder)

    def discover_files(self):
        return sorted(self.data_folder.glob("*.csv"))

    def read_statcan(self, file):
        return pd.read_csv(file)

    def read_worldbank(self, file):
        return pd.read_csv(file, skiprows=4)

    def read_export(self, file):
        return pd.read_csv(file, skiprows=1)

    def read_bankofcanada(self, file):

        with open(file, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        header = None

        for i, line in enumerate(lines):

            text = line.lower().strip().replace('"', "")

            if text.startswith("date,") or text.startswith("date,value"):
                header = i
                break

        if header is None:
            raise Exception(f"Could not locate data header in {file.name}")

        return pd.read_csv(file, skiprows=header)

    def read_csv(self, file):

        name = file.name

        if name.startswith("111"):
            return self.read_statcan(file)

        elif name.startswith("API_"):
            return self.read_worldbank(file)

        elif name.startswith("FVI_"):
            return self.read_bankofcanada(file)

        elif name.startswith("export"):
            return self.read_export(file)

        else:
            return pd.read_csv(file)

    def load_all(self):

        datasets = {}

        for file in self.discover_files():

            print(f"Loading {file.name}")

            try:
                datasets[file.stem] = self.read_csv(file)

            except Exception as e:
                print(f"Failed: {file.name}")
                print(e)

        return datasets
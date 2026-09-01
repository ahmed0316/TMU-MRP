from src.data_loader import DataLoader
from src.data_profiler import DataProfiler
from src.data_dictionary import DataDictionary
from src.data_cleaner import DataCleaner
from src.config import RAW_DATA, TABLES
from src.data_merger import DataMerger
from src.feature_engineering import FeatureEngineer
from src.feature_selector import FeatureSelector
from src.fvi import FinancialVulnerabilityIndex
from src.models import ModelTrainer
import warnings
import pandas as pd
from src.visualization import ModelVisualizer

warnings.simplefilter("ignore",pd.errors.PerformanceWarning) #hide performance warning so output is clearer
warnings.simplefilter("ignore", FutureWarning) #hide future warnings as well

loader = DataLoader(RAW_DATA)
datasets = loader.load_all()
cleaner = DataCleaner()
datasets = {
    name: cleaner.clean(df)
    for name, df in datasets.items()
}
profiler = DataProfiler(datasets)
summary = profiler.save(TABLES / "dataset_summary.csv")
dictionary = DataDictionary(datasets).build()
dictionary.to_csv(TABLES / "data_dictionary.csv", index=False)

print(summary)

merger = DataMerger(datasets)
master = merger.merge()

master.to_csv("data/processed/master_dataset.csv",index=False)

#build fvi
fvi = FinancialVulnerabilityIndex(master)

master = fvi.build()

master.to_csv("data/processed/fvi_dataset.csv",index=False)

#feature engineering
engineer = FeatureEngineer(master)

features = engineer.engineer()

features.to_csv("data/processed/master_features.csv",index=False)

#feature selection
selector = FeatureSelector(features)

model_data = selector.run()

model_data.to_csv("data/processed/model_dataset.csv",index=False)

#prints
print(master[["expert_fvi","pca_fvi"]].head())
print("FVI dataset:", master.shape)
print("Feature dataset:", features.shape)
print("Model dataset:", model_data.shape)
print()
print("Starting machine learning...")

trainer = ModelTrainer(model_data)
results = trainer.run()

#save results
results.to_csv("outputs/tables/model_results.csv")

#results
print(results)

predictions = pd.read_csv(
    "outputs/tables/model_predictions.csv"
)

visualizer = ModelVisualizer(predictions)

visualizer.actual_vs_predicted()
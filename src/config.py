from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA = BASE_DIR / "data" / "raw"
PROCESSED_DATA = BASE_DIR / "data" / "processed"

FIGURES = BASE_DIR / "outputs" / "figures"
TABLES = BASE_DIR / "outputs" / "tables"
MODELS = BASE_DIR / "outputs" / "models"

RANDOM_STATE = 42

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.15
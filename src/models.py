import pandas as pd

from pathlib import Path

from sklearn.linear_model import (
    LinearRegression,
    Ridge
)

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


class ModelTrainer:

    def __init__(self, df):

        self.df = df.copy()

    def prepare_data(self):

        df = self.df.copy()

        #date prep
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        
        df = (
            df.set_index("date")
            .select_dtypes(include="number")
            .resample("MS")
            .mean()
            .reset_index()
        )

        print(f"Monthly observations: {len(df)}")

        print(
            f"Monthly date range: "
            f"{df['date'].min()} -> "
            f"{df['date'].max()}"
        )

        #one month ahead target
        df["target_fvi"] = (
            df["expert_fvi"]
            .shift(-1)
        )

        #naive forecase
        df["naive_prediction"] = df["expert_fvi"]

        #predictor matrix
        drop_cols = [
            "date",
            "expert_fvi",
            "pca_fvi",
            "target_fvi",
            "naive_prediction"
        ]

        X = df.drop(
            columns=[
                c for c in drop_cols
                if c in df.columns
            ]
        )

        #remove target derived feature
        target_derived = [
            c for c in X.columns
            if c.startswith("expert_fvi")
            or c.startswith("pca_fvi")
        ]

        X = X.drop(
            columns=target_derived,
            errors="ignore"
        )

        print(
            f"Removed target-derived features: "
            f"{len(target_derived)}"
        )

        time_prefixes = [
            "year",
            "quarter",
            "month",
            "days_since_start"
        ]

        time_derived = [
            c for c in X.columns
            if any(
                c == prefix
                or c.startswith(prefix + "_")
                for prefix in time_prefixes
            )
        ]

        X = X.drop(
            columns=time_derived,
            errors="ignore"
        )

        print(
            f"Removed engineered time features: "
            f"{len(time_derived)}"
        )

        #raw economic indicators
        raw_features = [
            c for c in X.columns
            if c.startswith("fvi_")
            and "_lag" not in c
            and "_rolling" not in c
            and "_pct_change" not in c
        ]

        X = X[raw_features]

        print(
            f"Raw economic indicators available: "
            f"{len(raw_features)}"
        )

        
        model_start = pd.Timestamp(
            "2014-01-01"
        )

        df = df[
            df["date"] >= model_start
        ].copy()

        X = X.loc[df.index].copy()

        print(
            f"Forecasting sample: "
            f"{df['date'].min().date()} -> "
            f"{df['date'].max().date()}"
        )

        y = df["target_fvi"]

        #repplace infinity with missing
        X = X.replace(
            [
                float("inf"),
                float("-inf")
            ],
            pd.NA
        )

        valid = y.notna()

        X = X.loc[valid].copy()
        y = y.loc[valid].copy()

        dates = (
            df.loc[valid, "date"]
            .copy()
        )

        naive = (
            df.loc[
                valid,
                "naive_prediction"
            ]
            .copy()
        )

        #80/20 split
        split_index = int(
            len(X) * 0.80
        )

        self.X_train = (
            X.iloc[:split_index]
            .copy()
            .reset_index(drop=True)
        )

        self.X_test = (
            X.iloc[split_index:]
            .copy()
            .reset_index(drop=True)
        )

        self.y_train = (
            y.iloc[:split_index]
            .copy()
            .reset_index(drop=True)
        )

        self.y_test = (
            y.iloc[split_index:]
            .copy()
            .reset_index(drop=True)
        )

        self.train_dates = (
            dates.iloc[:split_index]
            .copy()
            .reset_index(drop=True)
        )

        self.test_dates = (
            dates.iloc[split_index:]
            .copy()
            .reset_index(drop=True)
        )

        self.naive_test = (
            naive.iloc[split_index:]
            .copy()
            .reset_index(drop=True)
        )

        debug = pd.DataFrame({
            "date": self.test_dates,
            "current_fvi": self.naive_test,
            "next_month_actual": self.y_test
        })

        print(
            debug
            .head(10)
            .to_string(index=False)
        )

        coverage = (
            self.X_train
            .notna()
            .mean()
        )

        eligible_features = (
            coverage[
                coverage >= 0.70
            ]
            .index
        )

        self.X_train = (
            self.X_train[
                eligible_features
            ]
        )

        self.X_test = (
            self.X_test[
                eligible_features
            ]
        )

        print(
            f"Features with >=70% "
            f"training coverage: "
            f"{len(eligible_features)}"
        )

        medians = (
            self.X_train
            .median()
        )

        self.X_train = (
            self.X_train
            .fillna(medians)
        )

        self.X_test = (
            self.X_test
            .fillna(medians)
        )


        usable = (
            self.X_train.columns[
                self.X_train
                .notna()
                .any()
            ]
        )

        self.X_train = (
            self.X_train[usable]
        )

        self.X_test = (
            self.X_test[usable]
        )

        non_constant = [
            c for c in self.X_train.columns
            if self.X_train[c].nunique() > 1
        ]

        self.X_train = (
            self.X_train[
                non_constant
            ]
        )

        self.X_test = (
            self.X_test[
                non_constant
            ]
        )

        correlations = (
            self.X_train
            .corrwith(self.y_train)
            .abs()
            .dropna()
            .sort_values(
                ascending=False
            )
        )

        top_features = (
            correlations
            .head(10)
            .index
        )

        print(
            "\n===== TOP 10 SELECTED FEATURES ====="
        )

        for i, feature in enumerate(
            top_features,
            start=1
        ):

            print(
                f"{i:2d}. {feature} "
                f"(correlation = "
                f"{correlations[feature]:.4f})"
            )

        print(
            "===================================\n"
        )

        feature_table = pd.DataFrame({
            "feature": top_features,
            "abs_training_correlation": [
                correlations[f]
                for f in top_features
            ]
        })

        Path(
            "outputs/tables"
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        feature_table.to_csv(
            "outputs/tables/"
            "selected_features.csv",
            index=False
        )

        self.X_train = (
            self.X_train[
                top_features
            ]
        )

        self.X_test = (
            self.X_test[
                top_features
            ]
        )

        print(
            f"Features used for modeling: "
            f"{len(top_features)}"
        )

        print(
            f"Training observations: "
            f"{len(self.X_train)} | "
            f"Testing observations: "
            f"{len(self.X_test)}"
        )

        print(
            f"Training period: "
            f"{self.train_dates.iloc[0].date()} "
            f"-> "
            f"{self.train_dates.iloc[-1].date()}"
        )

        print(
            f"Testing period: "
            f"{self.test_dates.iloc[0].date()} "
            f"-> "
            f"{self.test_dates.iloc[-1].date()}"
        )

    
    def evaluate_predictions(
        self,
        predictions
    ):

        return {
            "MAE": mean_absolute_error(
                self.y_test,
                predictions
            ),

            "RMSE": (
                mean_squared_error(
                    self.y_test,
                    predictions
                )
                ** 0.5
            ),

            "R2": r2_score(
                self.y_test,
                predictions
            )
        }

    #train

    def run(self):

        print("\nPreparing data...")

        self.prepare_data()

        results = {}


        print(
            "Evaluating Naive Baseline..."
        )

        naive_predictions = (
            self.naive_test
            .values
        )

        results[
            "Naive Baseline"
        ] = self.evaluate_predictions(
            naive_predictions
        )


        print(
            "Training Linear Regression..."
        )

        lr = LinearRegression()

        lr.fit(
            self.X_train,
            self.y_train
        )

        linear_predictions = (
            lr.predict(
                self.X_test
            )
        )

        print(
            "Evaluating Linear Regression..."
        )

        results[
            "Linear Regression"
        ] = self.evaluate_predictions(
            linear_predictions
        )

        scaler = StandardScaler()

        X_train_scaled = (
            scaler.fit_transform(
                self.X_train
            )
        )

        X_test_scaled = (
            scaler.transform(
                self.X_test
            )
        )

        print(
            "Training Ridge Regression..."
        )

        ridge = Ridge(
            alpha=1.0
        )

        ridge.fit(
            X_train_scaled,
            self.y_train
        )

        ridge_predictions = (
            ridge.predict(
                X_test_scaled
            )
        )

        print(
            "Evaluating Ridge Regression..."
        )

        results[
            "Ridge Regression"
        ] = self.evaluate_predictions(
            ridge_predictions
        )

        print(
            "Training Random Forest..."
        )

        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        rf.fit(
            self.X_train,
            self.y_train
        )

        rf_predictions = (
            rf.predict(
                self.X_test
            )
        )

        print(
            "Evaluating Random Forest..."
        )

        results[
            "Random Forest"
        ] = self.evaluate_predictions(
            rf_predictions
        )

        print(
            "Training Decision Tree..."
        )

        dt = DecisionTreeRegressor(
            max_depth=5,
            random_state=42
        )

        dt.fit(
            self.X_train,
            self.y_train
        )

        dt_predictions = (
            dt.predict(
                self.X_test
            )
        )

        print(
            "Evaluating Decision Tree..."
        )

        results[
            "Decision Tree"
        ] = self.evaluate_predictions(
            dt_predictions
        )

        print(
            "Training Gradient Boosting..."
        )

        gb = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )

        gb.fit(
            self.X_train,
            self.y_train
        )

        gb_predictions = (
            gb.predict(
                self.X_test
            )
        )

        print(
            "Evaluating Gradient Boosting..."
        )

        results[
            "Gradient Boosting"
        ] = self.evaluate_predictions(
            gb_predictions
        )

        print(
            "Training SVR..."
        )

        svr = SVR(
            kernel="rbf",
            C=1.0,
            epsilon=0.1
        )

        svr.fit(
            X_train_scaled,
            self.y_train
        )

        svr_predictions = (
            svr.predict(
                X_test_scaled
            )
        )

        print(
            "Evaluating SVR..."
        )

        results[
            "SVR"
        ] = self.evaluate_predictions(
            svr_predictions
        )

        print(
            "Training Neural Network..."
        )

        nn = MLPRegressor(
            hidden_layer_sizes=(16, 8),
            solver="lbfgs",
            alpha=0.001,
            max_iter=3000,
            random_state=42
        )

        nn.fit(
            X_train_scaled,
            self.y_train
        )

        nn_predictions = (
            nn.predict(
                X_test_scaled
            )
        )

        print(
            "Evaluating Neural Network..."
        )

        results[
            "Neural Network"
        ] = self.evaluate_predictions(
            nn_predictions
        )

        predictions = pd.DataFrame({

            "date":
                self.test_dates.values,

            "actual_fvi":
                self.y_test.values,

            "naive_baseline":
                naive_predictions,

            "linear_regression":
                linear_predictions,

            "ridge_regression":
                ridge_predictions,

            "random_forest":
                rf_predictions,

            "decision_tree":
                dt_predictions,

            "gradient_boosting":
                gb_predictions,

            "svr":
                svr_predictions,

            "neural_network":
                nn_predictions
        })

        Path(
            "outputs/tables"
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        predictions.to_csv(
            "outputs/tables/"
            "model_predictions.csv",
            index=False
        )

        print(
            "Finished training models!"
        )

        return (
            pd.DataFrame(results)
            .T
            .sort_values("RMSE")
        )
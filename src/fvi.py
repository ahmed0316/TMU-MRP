import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class FinancialVulnerabilityIndex:

    """
    Builds both:

        1. Expert Financial Vulnerability Index
        2. PCA Financial Vulnerability Index
    """

    #variables used in the index

    FVI_COLUMNS = [

        #banking Stability
        "fvi_nsfr_sib",
        "fvi_lcr_sib",
        "fvi_roe_sib",
        "fvi_roa_sib",

        #mortgage Market
        "fvi_mtg_rate_5y_fix",
        "fvi_mtg_rate_5y_var",

        #credit Risk
        "fvi_cc_avg_utilization_rate_allborrowers",
        "fvi_cc_dlq_rate_30dayplus_allborrowers",
        "fvi_cc_utilization_rate_above80_allborrowers",

        #debt burden
        "fvi_median_lti_mtg_all",
        "fvi_mtg_lti450_all",
        "fvi_median_mdsr_all",
        "fvi_mtg_mdsr_gt25_all",

        #financial stress
        "fvi_fin_stress",
        "fvi_nfb_cash_debt_ratio",
        "fvi_nfb_icr",
        "fvi_nfb_insolvency",
        "fvi_nfb_profit_margin",
        "fvi_fsi_can"

    ]

    def __init__(self, df):

        self.df = df.copy()

    def prepare(self):

        cols = []

        for c in self.FVI_COLUMNS:

            if c in self.df.columns:
                cols.append(c)

        print(f"Using {len(cols)} FVI variables")

        self.features = self.df[cols].copy()

        #numeric only
        self.features = self.features.apply(
            pd.to_numeric,
            errors="coerce"
        )

        #fill missing values
        self.features = self.features.fillna(
            self.features.median()
        )

    def standardize(self):

        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.features)

    def expert_fvi(self):

        self.df["expert_fvi"] = self.X.mean(axis=1)

    def pca_fvi(self):

        pca = PCA(n_components=1)
        scores = pca.fit_transform(self.X)
        self.df["pca_fvi"] = scores

        print(
            f"Explained Variance = "
            f"{pca.explained_variance_ratio_[0]:.2%}"
        )

    def build(self):

        self.prepare()
        self.standardize()
        self.expert_fvi()
        self.pca_fvi()

        return self.df
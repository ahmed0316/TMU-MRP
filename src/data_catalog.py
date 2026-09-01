"""
Central catalog for all the datasets in the mrp

each data set --> 1 catagory
merger and feature engineering will use this
"""

DATASET_CATEGORIES = {

    "interest_rates": [
        "FVI_BANKS_CASH",
        "FVI_BANKS_PROFIT",
        "FVI_BANKS_ROE",
        "FVI_BANKS_ROA"
    ],

    "mortgage": [
        "FVI_MORTGAGE_DEBT_SERVICE_RATIO",
        "FVI_MORTGAGE_ORIGINATIONS"
    ],

    "credit": [
        "FVI_LOAN_TO_INCOME",
        "FVI_LOAN_TO_VALUE",
        "FVI_HOUSEHOLD_CREDIT"
    ],

    "housing": [
        "FVI_HOUSE_PRICE_EXPECTATIONS",
        "FVI_HOUSE_FLIPPING"
    ],

    "financial_stability": [
        "FVI_BANKS_CET1",
        "FVI_BANKS_LEVERAGE",
        "FVI_BANKS_NPL"
    ],

    "macro": [
        "API_CAN_DS2_en_csv_v2_10977"
    ],

    "statistics_canada": [
        "11100021"
    ],

    "other": []

}
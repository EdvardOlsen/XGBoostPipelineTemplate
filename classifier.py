from argparse import ArgumentParser

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main(args):
    diabetes = pd.read_csv(args.file_path).dropna()
    X = diabetes.drop(columns=[args.target_column])
    y = diabetes[args.target_column]

    num_cols = X.select_dtypes(include="number").columns
    cat_cols = X.select_dtypes(exclude="number").columns
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("oh-encode", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="mean")), ("scale", StandardScaler())]
    )

    full_processor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, num_cols),
            ("categorical", categorical_pipeline, cat_cols),
        ]
    )

    X_processed = full_processor.fit_transform(X)
    y_processed = SimpleImputer(strategy="most_frequent").fit_transform(
        y.values.reshape(-1, 1)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=0.2
    )

    xgb_cl = xgb.XGBClassifier(n_estimators=100)
    xgb_cl.fit(X_train, y_train)

    preds = xgb_cl.predict(X_test)

    print(accuracy_score(y_test, preds))


def parse():
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--file_path",
        help="The path of the file containing the data for the xgboost to learn from",
    )
    parser.add_argument(
        "-t",
        "--target_column",
        help="The column in the csv file that is the target column (what you need to find)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    main(args)

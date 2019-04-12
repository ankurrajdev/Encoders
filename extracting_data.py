import pandas as pd
import numpy as np
import time
import category_encoders as ce

from random import randrange
from sleepmind.preprocessing import SumEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from xam.feature_extraction import BayesianTargetEncoder
from hccEncoding.EncoderForRegression import (
    BayesEncoding,
    BayesEncodingKfold,
    LOOEncoding,
    LOOEncodingKfold,
)


df_avocado = pd.read_csv("datasets/avocado.csv", index_col=0)
df_avocado.rename(columns={"AveragePrice": "price"}, inplace=True)
df_avocado = df_avocado.reset_index(drop=True)
df_avocado.loc[df_avocado.type == "conventional", "type"] = 1
df_avocado.loc[df_avocado.type == "organic", "type"] = 0
df_avocado = df_avocado.drop("Date", axis=1)

df_diamonds = pd.read_csv("datasets/diamonds.csv", index_col=0)

df_brooklyn = pd.read_csv(
    "datasets/brooklyn_sales_map.csv", index_col=0, low_memory=False
)
df_brooklyn.rename(columns={"sale_price": "price"}, inplace=True)
df_brooklyn = df_brooklyn.drop("easement", axis=1)


# this functions runs all the encoders and appends to a list
def pipeline(df, target, cat_columns, models):
    n_rows, n_cols = df.shape
    metrics = {
        "n_rows": [],
        "n_cols": [],
        "cardinality": [],
        "model": [],
        "column": [],
        "encoder": [],
        "rmse": [],
        "mae": [],
        "fit_time": [],
        "rmse_change": [],
        "mae_change": [],
        "fit_time_change": [],
    }
    columns = cat_columns

    for model_name in models:

        base_rmse, base_mae, base_fit_time = model(
            df=df,
            target=target,
            encoder=np.nan,
            col=np.nan,
            model_name=model_name,
            encoder_type="basic",
            encoder_name=[],
        )

        _append_metric(
            row_list=metrics,
            n_rows=n_rows,
            n_cols=n_cols,
            cardinality=np.nan,
            model_name=model_name,
            column=np.nan,
            name="basic",
            rmse=base_rmse,
            mae=base_mae,
            fit_time=base_fit_time,
            base_rmse=base_rmse,
            base_mae=base_mae,
            base_fit_time=base_fit_time,
        )

        for column in columns:
            print()
            print(column)
            cardinality = df[column].nunique()

            print("ohe")
            rmse, mae, fit_time = model(
                df=df,
                target=target,
                encoder=np.nan,
                col=column,
                model_name=model_name,
                encoder_type="basic",
                encoder_name="One Hot Encoder (pd.dummies)",
            )
            _append_metric(
                row_list=metrics,
                n_rows=n_rows,
                n_cols=n_cols,
                cardinality=cardinality,
                model_name=model_name,
                column=column,
                name="One Hot Encoder (pd.dummies)",
                rmse=rmse,
                mae=mae,
                fit_time=fit_time,
                base_rmse=base_rmse,
                base_mae=base_mae,
                base_fit_time=base_fit_time,
            )

            encoders = [
                ("Sum Encoder(sleepmind)", SumEncoder()),
                ("BinaryEncoder", ce.BinaryEncoder(cols=[column])),
                ("HashingEncoder", ce.HashingEncoder(cols=[column])),
                ("OneHotEncoder", ce.OneHotEncoder(cols=[column])),
                ("OrdinalEncoder", ce.OrdinalEncoder(cols=[column])),
                ("BaseNEncoder", ce.BaseNEncoder(cols=[column])),
                (
                    "BackwardDifferenceEncoder",
                    ce.BackwardDifferenceEncoder(cols=[column]),
                ),
                ("HelmertEncoder", ce.HelmertEncoder(cols=[column])),
                ("SumEncoder", ce.SumEncoder(cols=[column])),
                ("PolynomialEncoder", ce.PolynomialEncoder(cols=[column])),
                ("TargetEncoder", ce.TargetEncoder(cols=[column])),
                ("LeaveOneOutEncoder", ce.LeaveOneOutEncoder(cols=[column])),
                (
                    "XAM_bayesian_targetEncoder",
                    BayesianTargetEncoder(
                        columns=[column], prior_weight=3, suffix=""
                    ),
                ),
            ]

            for name, encoder in encoders:
                print(name)
                rmse, mae, fit_time = model(
                    df=df,
                    target=target,
                    encoder=encoder,
                    col=column,
                    model_name=model_name,
                    encoder_type="sklearn_encoding",
                    encoder_name=name,
                )
                _append_metric(
                    row_list=metrics,
                    n_rows=n_rows,
                    n_cols=n_cols,
                    cardinality=cardinality,
                    model_name=model_name,
                    column=column,
                    name=name,
                    rmse=rmse,
                    mae=mae,
                    fit_time=fit_time,
                    base_rmse=base_rmse,
                    base_mae=base_mae,
                    base_fit_time=base_fit_time,
                )

            bayes_encoders = [
                ("hcc_BayesEncoding", BayesEncoding),
                ("hcc_BayesEncodingKfold", BayesEncodingKfold),
                ("LOOEncoding", LOOEncoding),
                ("LOOEncodingKfold", LOOEncodingKfold),
            ]
            for name, bayes_encoder in bayes_encoders:
                print(name)
                rmse, mae, fit_time = model(
                    df=df,
                    target=target,
                    encoder=bayes_encoder,
                    col=column,
                    model_name=model_name,
                    encoder_name=name,
                    encoder_type="basic",
                    hcc_ind=1,
                )
                _append_metric(
                    row_list=metrics,
                    n_rows=n_rows,
                    n_cols=n_cols,
                    cardinality=cardinality,
                    model_name=model_name,
                    column=column,
                    name=name,
                    rmse=rmse,
                    mae=mae,
                    fit_time=fit_time,
                    base_rmse=base_rmse,
                    base_mae=base_mae,
                    base_fit_time=base_fit_time,
                )
    results = pd.DataFrame(metrics)
    return results


def model(
    df,
    target,
    encoder,
    col,
    model_name,
    encoder_name,
    encoder_type,
    cv=5,
    hcc_ind=0,
):
    if encoder_name == "One Hot Encoder (pd.dummies)":
        df = pd.get_dummies(df, columns=[col])
    NUM_VARS = (
        df.loc[:, df.columns != target]
        .select_dtypes(include=["int64", "float64", "uint8"])
        .copy()
        .columns
    )
    if encoder_type == "basic":
        tfms = make_column_transformer(
            (SimpleImputer(missing_values=np.nan, strategy="mean"), NUM_VARS),
            remainder="drop",
        )
    elif encoder_type == "sklearn_encoding":
        tfms = make_column_transformer(
            (SimpleImputer(missing_values=np.nan, strategy="mean"), NUM_VARS),
            (encoder, [col]),
            remainder="drop",
        )

    if model_name == "LR":
        learner = LinearRegression()
    elif model_name == "RF":
        learner = RandomForestRegressor(n_estimators=20)

    pipe = make_pipeline(tfms, learner)
    idx = cross_validation_split(dataset=df, folds=cv)
    if hcc_ind == 1:
        pipe = learner
    else:
        pipe = pipe
    rmse, mae, fit_time = cross_validation(
        df=df,
        target=target,
        pipe=pipe,
        idx=idx,
        hcc_ind=hcc_ind,
        col=col,
        encoder=encoder,
    )
    return rmse, mae, fit_time


# Split a dataset into k folds
def cross_validation_split(dataset, folds):
    dataset_split = list()
    dataset_copy = list(range(len((dataset))))
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def cross_validation(df, pipe, idx, hcc_ind, col, encoder, target):
    fit_time = []
    rmse = []
    mae = []
    hcc_time = 0
    for i in range(len(idx)):
        test_ids = df.index.isin(idx[i])
        X_train, y_train, X_test, y_test = splitting(df, test_ids, target)
        if hcc_ind == 1:
            X_train = pd.concat([X_train, y_train], axis=1, sort=False)
            hcc_start_time = time.time()
            X_train, X_test = encoder(
                train=X_train,
                test=X_test,
                target=y_train.name,
                feature=col,
                drop_origin_feature=True,
            )
            X_train = X_train.drop(y_train.name, axis=1)
            NUM_VARS = (
                X_train.select_dtypes(include=["int64", "float64", "uint8"])
                .copy()
                .columns
            )
            X_train = X_train[NUM_VARS]
            X_test = X_test[NUM_VARS]
            X_train = X_train[NUM_VARS]
            X_test = X_test[NUM_VARS]
            X_train = X_train.fillna(X_train.mean())
            X_test = X_test.fillna(X_test.mean())
            y_train = y_train.fillna(y_train.mean())
            y_test = y_test.fillna(y_test.mean())
            hcc_time = time.time() - hcc_start_time
        else:
            pass
        # Added X_train, X_test, y_rain, y_test because we need to reindex the
        # datasets to make some encoders work
        start_time = time.time()
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        fit_time.append((time.time() - start_time) + hcc_time)
        rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae.append(mean_absolute_error(y_test, y_pred))
    return np.mean(rmse), np.mean(mae), np.mean(fit_time)


def splitting(df, test_ids, target):
    test = df[test_ids]
    train = df[~test_ids]
    train = train.reset_index(drop=1)
    test = test.reset_index(drop=1)

    X_train = train.loc[:, train.columns != target]
    y_train = train[target]
    X_test = test.loc[:, test.columns != target]
    y_test = test[target]
    return X_train, y_train, X_test, y_test


def _append_metric(
    row_list,
    n_rows,
    n_cols,
    cardinality,
    column,
    name,
    model_name,
    rmse,
    mae,
    fit_time,
    base_rmse,
    base_mae,
    base_fit_time,
):
    row_list["column"].append(column)
    row_list["n_rows"].append(n_rows)
    row_list["n_cols"].append(n_cols)
    row_list["cardinality"].append(cardinality)
    row_list["model"].append(model_name)
    row_list["encoder"].append(name)
    row_list["rmse"].append(rmse)
    row_list["rmse_change"].append(((rmse / base_rmse) - 1) * 100)
    row_list["mae"].append(mae)
    row_list["mae_change"].append(((mae / base_mae) - 1) * 100)
    row_list["fit_time"].append(fit_time)
    row_list["fit_time_change"].append(((fit_time / base_fit_time) - 1) * 100)
    return row_list


avocado_result = pipeline(
    df_avocado, target="price", cat_columns=["region"], models=["LR", "RF"]
)
avocado_result.to_csv("datasets/avocado_result.csv")

diamonds_result = pipeline(
    df_diamonds,
    target="price",
    cat_columns=["cut", "color", "clarity"],
    models=["LR", "RF"],
)
diamonds_result.to_csv("datasets/diamonds_result.csv")

brooklyn_result = pipeline(
    df_brooklyn,
    target="price",
    cat_columns=[
        "neighborhood",
        "building_class_category",
        "tax_class",
        "FireComp",
        "SanitSub",
        "SPDist1",
        "OwnerType",
    ],
    models=["LR", "RF"],
)
brooklyn_result.to_csv("datasets/brooklyn_result.csv")

final_result = avocado_result.append(diamonds_result, sort=False).reset_index(
    drop=True
)
final_result = final_result.append(brooklyn_result, sort=False).reset_index(
    drop=True
)

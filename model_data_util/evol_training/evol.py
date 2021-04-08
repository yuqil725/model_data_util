"""
To use evolutionary algo, we still need all global variable.
As the evolutionary algo need to modify the original models, what layers can be modified, how to modified are required
to be specified by the global variables """

import argparse
import pickle
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tqdm import tqdm

from model_data_util.constant import OPTIONS
from model_data_util.create_tt_data.generate_tt_data import testTT
from model_data_util.create_tt_data.model_build import ModelBuild
from model_data_util.create_tt_data.model_data_convert import preprocessRawData, convertRawDataToModel


def percentageError(y_pred, y_test):
    """
    :return: the average percentage error between y_pred and y_test
    """
    return np.abs(np.mean((y_pred - y_test) / y_test))


def evaluateError(y_pred, y_test, metrix=percentageError):
    """
    :param metrix: error metrix, which can be customized
    :return: error between the actual and the prediction
    """
    y_pred = np.reshape(np.array(y_pred), (-1,))
    y_test = np.reshape(np.array(y_test), (-1,))
    return metrix(y_pred, y_test)


def evalChild(df, to):
    """
    :param df: the raw dataframe converted from model :param to: the strength of modification. "s": modify a model to
    a similar one; "d": modify a model to a different one; "dd": modify a model to a rather different one.
    :return: a new model dataframe after modification.
    """
    res_df: pd.DataFrame = df.copy()
    if to == "s":  # s stand for similar
        res_df = layerMutation(res_df)
    elif to == "d":  # d stand for slightly different
        res_df = layerMutation(res_df)
        res_df = layerAddition(res_df)
        res_df = layerRemoval(res_df)
    elif to == "dd":
        res_df = evalChild(evalChild(df, "d"), "d")
    return res_df


def layerAddition(df, max_add_layers=4, verbose=False):
    """
    :param df: raw model dataframe
    :param max_add_layers: max number of layer that will be added. The newly added row will duplicate the layer of the previous one but with different configuration.
    :param verbose: print the added rows if true
    :return: a new model dataframe after addition
    """

    # only remove the layer from part_conv
    def insert_row(idx, df, new_row):
        dfA = df.iloc[:idx, ]
        dfB = df.iloc[idx:, ]
        df = dfA.append(new_row, ignore_index=True).append(dfB).reset_index(drop=True)
        return df

    new_df = df.copy()
    max_add_layers = max(min(max_add_layers, new_df.shape[0] - 2), 1)
    # exclude the first and last layer
    add_layers = random.sample(range(1, max_add_layers + 1), 1)[0]
    selected_layers = random.sample([i for i in range(df.shape[0]) if
                                     df.iloc[i].layer in OPTIONS["Model"]["layer"]], add_layers)
    selected_layers = np.array(sorted(selected_layers))
    selected_layers += np.arange(0, selected_layers.shape[0], 1)
    if verbose:
        added_row = pd.DataFrame()
    mb = ModelBuild()
    for l in selected_layers:
        row = new_df.iloc[l]
        new_row = mb.chooseRandomComb(OPTIONS[row.layer])
        not_mu_layer = ["optimizer", "loss"]
        for i in new_row.keys():
            if i in not_mu_layer:
                new_row[i] = row[i]
        for k in row.index:
            if k not in new_row:
                new_row[k] = row[k]
        fixed_layer = {"strides": (1, 1), "padding": "same"}
        for k in fixed_layer.keys():
            if row[k] == row[k]:
                new_row[k] = fixed_layer[k]
        if verbose:
            added_row = added_row.append(new_row, ignore_index=True)
        new_df = insert_row(l, new_df, new_row)
    if verbose:
        print("Adding Row:")
        print(added_row[df.columns])
    return new_df


def layerRemoval(df, max_remv_layers=1, removable=["Conv2D", "MaxPooling2D", "Dropout"],
                 verbose=False):
    """
    :param df: the raw model df
    :param max_remv_layers: the max number of removal layers
    :param removable: the types of layer that are removable
    :param verbose: print the removed row if True
    :return: a new model dataframe after layer removal
    """
    # only remove the layer from part_conv
    new_df = df.copy()
    max_remv_layers = max(min(max_remv_layers, new_df.shape[0] - 2), 1)
    # exclude the first and last layer
    remv_layers = random.sample(range(1, max_remv_layers + 1), 1)[0]
    selected_layers = random.sample([i for i in range(df.shape[0]) if df.iloc[i].layer in removable], remv_layers)
    if verbose:
        print("Removing Row:")
        print(new_df.iloc[selected_layers])
    new_df = new_df.drop(index=selected_layers)
    return new_df


def layerMutation(df, max_mu_layers=50, verbose=False):
    """
    This method will modify the model dataframe by randomly change the configuration of some layers
    :param df: the raw model dataframe
    :param max_mu_layers: the maximum number of mutable layers
    :return:
    """
    new_df = df.copy()
    max_mu_layers = max(min(max_mu_layers, new_df.shape[0] - 2), 1)
    # exclude the first and last layer
    mu_layers = random.sample(range(1, max_mu_layers + 1), 1)[0]
    selected_layers = random.sample(list(range(new_df.shape[0]))[1:-1], mu_layers)
    if verbose:
        print("Before Mutation")
        print(new_df.iloc[selected_layers])
    for l in selected_layers:
        row = new_df.iloc[l]
        new_row = chooseRandomComb(OPTIONS[row.layer])
        not_mu_layer = []
        if row.layer in ["Conv2D", "MaxPooling2D"]:
            if row.padding == "same":
                not_mu_layer = ["strides", "padding", "optimizer", "loss"]
            else:
                not_mu_layer = ["strides", "padding", "optimizer", "loss", "kernel_size", "pool_size"]
        for i in new_row.keys():
            if i not in not_mu_layer:
                row[i] = new_row[i]
    if verbose:
        print("After Mutation")
        print(new_df.iloc[selected_layers])
    return new_df


def modelModifier(df, error, threshold={"low": 0.1, "high": 0.25}):
    """
    :param df: the raw model dataframe
    :param error: the error of tt_predictor
    :param threshold: the threshold used to  judge similar, different, and rather different models
    :return:
    """
    new_df = df.copy()
    if error >= threshold["high"]:
        new_df = evalChild(new_df, to="s")
    elif threshold["high"] > error >= threshold["low"]:
        new_df = evalChild(new_df, to="d")
    elif error < threshold["low"]:
        new_df = evalChild(new_df, to="dd")
    return new_df


def evalChildren(X_df_list, y_list, tt_predictor):
    """
    :param X_df_list: the test model dataframe
    :param y_list: the tt of the test models
    :param tt_predictor: the model used to predict the training time of models
    :return: a set of new data from evolution
    """
    new_X_df_list: list = []
    new_y_list: list = []
    for i, X_df in enumerate(tqdm(X_df_list)):
        x = preprocessRawData(X_df).values
        x = x.reshape((-1, *x.shape))
        y = y_list[i]
        y_pred = tt_predictor(x).numpy().reshape(-1)[0]
        error = evaluateError(y_pred, y)
        new_X_df = modelModifier(X_df, error)
        new_X_df_list.append(new_X_df)
        model, tmp1, tmp2 = convertRawDataToModel(new_X_df)
        test_res = testTT(model, OPTIONS["Data"]["num_data"])
        new_y_list.append(test_res['median'])
    return new_X_df_list, new_y_list


def concateModelData(X_df_list, y_list):
    """
    :param X_df_list: a list of raw model dataframe
    :param y_list: a list of tt
    :return: the preprocessed model data and tt
    """
    X = np.array([])
    y = np.array([])
    for i in range(len(X_df_list)):
        X_df = X_df_list[i]
        X_tmp = preprocessRawData(X_df).values
        X = X.reshape((-1, *np.array(X_tmp).shape))
        X_tmp = X_tmp.reshape((-1, *np.array(X_tmp).shape))
        X = np.concatenate((X, X_tmp))
    y = np.reshape(y_list, (-1, *np.array(y_list).shape[1:]))
    return X, y


def refitModel(tt_predictor, X_df_list, y_list, validation_split=0.1,
               batch_size=4, epochs=30, verbose=False):
    """
    :param tt_predictor: the predictor used to predict tt
    :param X_df_list: a list of raw model dataframe
    :param y_list: a list of tt
    :param verbose, validation_split, batch_size, epochs: the configurations while refit new data made by evolutionary process
    :return: the newly trained tt_predictor
    """
    X, y = concateModelData(X_df_list, y_list)
    tt_predictor.fit(X, y, validation_split=validation_split, epochs=epochs,
                     batch_size=batch_size, verbose=verbose)
    return tt_predictor


def evolStart(tt_predictor, num_evol, X_df_list, y_list, size_per_evol=None,
              verbose=True, disable_evol=False):
    """
    :param tt_predictor: the predictor used to predict tt
    :param num_evol: the number of evolution will experience
    :param X_df_list: a list of raw model dataframe
    :param y_list: a list of tt
    :param size_per_evol: the number of data points per evolution
    :param disable_evol: when disable evolution, the tt_predictor will be repetitively fed by the old test data
    :return: newly trained tt_predictor, the error history, the new model dataframes and corresponding tt
    """
    tt_predictor_new = keras.models.clone_model(tt_predictor)
    tt_predictor_new.compile(loss=tt_predictor.loss, optimizer=tt_predictor.optimizer.get_config()["name"])
    tt_predictor_new.set_weights(tt_predictor.get_weights())

    if size_per_evol is None:
        size_per_evol = len(X_df_list)
    assert size_per_evol <= len(X_df_list)
    error_his = []

    X_origin, y_origin = concateModelData(X_df_list, y_list)
    y_pred = tt_predictor_new.predict(X_origin)
    error = evaluateError(y_pred, y_origin)
    error_his.append(error)
    if verbose:
        print("Generation 0: error=%.3f" % error)

    index = random.sample(range(size_per_evol), size_per_evol)
    X_df_list_sample = [X_df_list[i] for i in index]
    y_list_sample = [y_list[i] for i in index]
    new_X_df_list_final = []
    new_y_list_final = []
    for i in tqdm(range(num_evol)):
        if not disable_evol:
            new_X_df_list, new_y_list = evalChildren(X_df_list_sample, y_list_sample, tt_predictor)
        else:
            new_X_df_list, new_y_list = X_df_list_sample.copy(), y_list_sample.copy()
        new_X_df_list_final.append(new_X_df_list)
        new_y_list_final.append(new_y_list)
        tt_predictor_new = refitModel(tt_predictor_new, new_X_df_list, new_y_list)
        y_pred = tt_predictor_new(X_origin)
        error = evaluateError(y_pred, y_origin)
        error_his.append(error)
        if verbose:
            print("Generation %d: error=%.3f" % (i + 1, error))
    return tt_predictor_new, error_his, new_X_df_list_final, new_y_list_final


def readData(data_path):
    tmp_res = pickle.load(open(data_path, "rb"))
    return tmp_res['X_df'], tmp_res['y']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_evol", type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    tt_predictor = tf.keras.models.load_model(args.model_path)
    X_df_list, y_list = readData(args.data_path)
    _, X_test, _, y_test = train_test_split(X_df_list, y_list)

    tt_predictor_new, error_his_evol, new_X_df_list_final, new_y_list_final = \
        evolStart(tt_predictor, args.num_evol, X_test, y_test, disable_evol=False)

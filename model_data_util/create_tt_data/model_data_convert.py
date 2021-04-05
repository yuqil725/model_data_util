import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import Sequential

from model_data_util.constant import OPTIONS, ONE_HOT_ENC, NUM_DIM, DEFAULT_INPUT_SHAPE


def createInputbyModel(model, data_points, data_shape=DEFAULT_INPUT_SHAPE):
    """
    Generate data according to model's conf
    The input shape is set in default
    """
    last_dense_conf = {}
    first_layer_conf = {}
    for l_conf in model.get_config()['layers']:
        if 'Dense' == l_conf['class_name']:
            last_dense_conf = l_conf['config']
        if first_layer_conf == {} and 'InputLayer' != l_conf['class_name']:
            first_layer_conf = l_conf
    assert last_dense_conf != {}
    out_shape = last_dense_conf['units']
    if first_layer_conf['class_name'] == 'Dense':
        data_shape = [out_shape] if type(out_shape) == int else out_shape
    x = np.ones((data_points, *data_shape))
    y = np.ones((data_points, out_shape))
    return (x, y)


def convertModelToRawData(model, num_data, batch_input_shape, columns=[], num_dim=NUM_DIM):
    """
    Given a model, convert model conf into dataframe
    The columns of dataframe depends on the global variable options
    """
    columns = [x for x in columns if "out_dim" not in x] + [f"out_dim_{x}" for x in range(num_dim)]
    df = pd.DataFrame(columns=columns)
    model_layers = model.get_config()['layers'].copy()
    for i, l in enumerate(model_layers):
        l_name = l['class_name']
        conf = l['config']
        new_row = dict(zip(list(OPTIONS[l_name].keys()), [conf[k] for k in OPTIONS[l_name].keys()]))

        new_row['layer'] = l_name
        new_row['optimizer'] = model.optimizer.get_config()['name']
        new_row['loss'] = model.loss
        new_row['active'] = 1
        new_row['num_data'] = num_data
        for j in range(num_dim):
            if j < batch_input_shape.shape[0]:
                new_row[f'out_dim_{j}'] = batch_input_shape[j]
            else:
                new_row[f'out_dim_{j}'] = np.nan
        df = df.append(new_row, ignore_index=True)
    return df


def convertRawDataToModel(df):
    """
    :param df: the raw dataframe used to convert into model
    :return: model made by df, and the num of data
    """
    assert len(df.optimizer.unique()) == 1
    assert "inputlayer" in df.layer.str.lower().unique()
    optimizer = df.optimizer.unique()[0]
    loss = df.loss.unique()[0]
    num_data = df.num_data.unique()[0]
    batch_input_shape = df[[x for x in df.columns if "out_dim" in x]].iloc[0].dropna().values
    model = Sequential()
    drop_cols = ["active", "optimizer", "layer", "loss", "num_data", *[x for x in df.columns if "out_dim" in x]]
    for i in range(df.shape[0]):
        l_name = df.iloc[i].layer
        if l_name not in OPTIONS["Model"]["layer"]:
            # only add those layer was in our layer type choices
            if l_name.lower() != "inputlayer":
                print(f"WARNING: {l_name} layer was skip as not listed in OPTIONS")
            continue
        kwargs = df.iloc[i].drop(drop_cols).dropna().to_dict()
        model.add(getattr(keras.layers, l_name)(**kwargs))
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model, num_data, batch_input_shape


def _preprocessRawData_Onehot_Helper(res, col, df, one_hot_enc):
    """
    :param res: return argument
    :param col: the column from df needs to be encoded into one hot
    :param df: the raw model dataframe
    :return: the one hot matrix
    """
    assert col in df.columns
    layer_one_hot = np.concatenate(
        [one_hot_enc[col].transform(np.reshape(l, (-1, 1))).toarray() for l in
         df[col].astype(str).values])
    layer_one_hot = layer_one_hot.astype(np.int)
    res = np.hstack((res, layer_one_hot))
    return res


def preprocessRawData(df, one_hot_enc=ONE_HOT_ENC, padding_to=64):
    """
    convert a raw model dataframe into one hot format
    """
    tmp_df = df.fillna(0)
    res = df.active.values.reshape(-1, 1).astype(np.int)
    columns = [["active"]]
    for col in ["layer", "optimizer", "loss", "activation"]:
        columns.append(["_".join([col, str(c)]) for c in one_hot_enc[col].categories_[0]])
        res = _preprocessRawData_Onehot_Helper(res, col, tmp_df, one_hot_enc)
    if 'use_bias' in df.columns:
        columns.append(['use_bias'])
        res = np.hstack((res, np.reshape(df['use_bias'].apply(lambda x: 1 if x else 0).values, (-1, 1))))
    for col in ["units", "filters", "num_data", *[x for x in df.columns if "out_dim" in x]]:
        if col in df.columns:
            columns.append([col])
            res = np.hstack((res, np.reshape(df[col].fillna(0).values, (-1, 1))))
    for col in ['strides', 'dilation_rate', 'kernel_size']:
        if col in df.columns:
            columns.append([col + "_r", col + "_c"])
            res = np.hstack(
                (res, np.reshape(np.array([list(x) if x == x else [0, 0] for x in df[col].values]).flatten(), (-1, 2))))
    columns = [x for y in columns for x in y]
    assert padding_to >= res.shape[0]
    res = np.concatenate([res, np.zeros((padding_to - res.shape[0], res.shape[1]))])

    return pd.DataFrame(res, columns=columns)

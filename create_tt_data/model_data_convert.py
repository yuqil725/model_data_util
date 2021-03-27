import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import Sequential

from constant import OPTIONS, ONE_HOT_ENC, NUM_DIM


def convertModelToRawData(model, columns, num_data, num_dim=NUM_DIM):
    """
    Given a model, convert model conf into dataframe
    The columns of dataframe depends on the global variable options
    """
    df = pd.DataFrame(columns=columns)
    for i, l in enumerate(model.get_config()['layers']):
        l_name = l['class_name']
        conf = l['config']
        new_row = dict(zip(list(OPTIONS[l_name].keys()), [conf[k] for k in OPTIONS[l_name].keys()]))

        new_row['layer'] = l_name
        new_row['optimizer'] = model.optimizer.get_config()['name']
        new_row['loss'] = model.loss
        new_row['active'] = 1
        new_row['num_data'] = num_data
        out_shape = None
        if i > 0:
            assert model.layers[i - 1].name == conf['name']
            out_shape = np.array(model.layers[i - 1].output.shape)
        elif i == 0 and l_name == "InputLayer":
            out_shape = np.array(conf['batch_input_shape'])
        if out_shape is not None:
            assert out_shape.shape[0] <= num_dim
            for j in range(num_dim):
                if j < out_shape.shape[0]:
                    new_row[f'out_dim_{j}'] = out_shape[j]
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
    optimizer = df.optimizer.unique()[0]
    loss = df.loss.unique()[0]
    num_data = df.num_data.unique()[0]
    model = Sequential()
    drop_cols = ["active", "optimizer", "layer", "loss", "num_data", *[x for x in df.columns if "out_dim" in x]]
    for i in range(df.shape[0]):
        l_name = df.iloc[i].layer
        if l_name not in OPTIONS["Model"]["layer"]:
            # only add those layer was in our layer type choices
            print(f"WARNING: {l_name} layer was skip as not listed in OPTIONS")
            continue
        kwargs = df.iloc[i].drop(drop_cols).dropna().to_dict()
        model.add(getattr(keras.layers, l_name)(**kwargs))
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model, num_data


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

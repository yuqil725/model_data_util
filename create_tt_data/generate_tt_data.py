import argparse
import collections
import random
import time

import numpy as np
from tensorflow import keras
from tqdm import tqdm

from constant import DEFAULT_INPUT_SHAPE, OPTIONS, COLUMNS
from create_tt_data.cnn_build_rule import CnnRules
from create_tt_data.ffnn_build_rule import FFnnRules
from create_tt_data.model_build import buildCnnModel, generateRandomModelConfigList, buildFFnnModel
from create_tt_data.model_data_convert import convertModelToRawData


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


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


def testTT(model, epochs=7, batch_size=4):
    """
    Given a model, test its TT. The first epoch is excluded to avoid setup time
    """
    num_data = random.choice(OPTIONS["Data"]["num_data"])
    x, y = createInputbyModel(model, num_data)
    time_callback = TimeHistory()
    model.fit(x, y, epochs=epochs, batch_size=batch_size, callbacks=[time_callback], verbose=False)
    times = time_callback.times
    test_res = {}
    test_res['median'] = np.median(times[2:])
    test_res['mean'] = np.mean(times[2:])
    test_res['std'] = np.std(times[2:])
    test_res['x_shape'] = np.array(x).shape
    test_res['y_shape'] = np.array(y).shape
    return test_res


def generateTTData(num_data, out_dim=10, max_layers=32, type="cnn", options=OPTIONS, columns=COLUMNS):
    result = collections.defaultdict(list)

    for _ in tqdm(range(num_data)):
        rules = None
        if type == "cnn":
            rules = CnnRules(max_layers=max_layers)
        elif type == "ffnn":
            rules = FFnnRules(max_layers=max_layers)
        assert rules is not None
        kwargs_list, layer_orders, image_shape_list = generateRandomModelConfigList(rules.layer_order, options=options)
        if type == "cnn":
            model = buildCnnModel(kwargs_list, layer_orders, out_dim)
        elif type == "ffnn":
            model = buildFFnnModel(kwargs_list, layer_orders)
        test_res = testTT(model)
        df = convertModelToRawData(model, columns, test_res['x_shape'][0])
        result["X_df"].append(df)
        result["y_median"].append(test_res['median'])
        result["y_mean"].append(test_res['mean'])
        result["y_std"].append(test_res['std'])
        if type == "cnn":
            result["image_shape_list"].append(image_shape_list)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_data', type=int, required=True)
    parser.add_argument('--out_dim', type=int, required=True)
    args = parser.parse_args()
    num_data = args.num_data
    out_dim = args.out_dim

    result = generateTTData(num_data, out_dim, type="ffnn")
    # print(convertRawDataToModel(result["X_df"][0]))
    # print(preprocessRawData(result["X_df"][0]).columns)

    # pickle.dump(result, open(f"../data/local_dp{DATA_POINTS}_CNN_Data_1.pkl", 'wb'))

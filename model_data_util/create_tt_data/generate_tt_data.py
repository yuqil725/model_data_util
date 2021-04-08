import argparse
import collections
import random
import time

import numpy as np
from tensorflow import keras
from tqdm import tqdm

from model_data_util.constant import OPTIONS, COLUMNS
from model_data_util.create_tt_data.cnn_build_rule import CnnRules
from model_data_util.create_tt_data.ffnn_build_rule import FFnnRules
from model_data_util.create_tt_data.model_build import ModelBuild
from model_data_util.create_tt_data.model_data_convert import convertModelToRawData, createInputbyModel, \
    convertRawDataToModel


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def testTT(model, epochs=7, batch_size=4, num_data_range=OPTIONS["Data"]["num_data"]):
    """
    Given a model, test its TT. The first epoch is excluded to avoid setup time
    """
    num_data = random.choice(num_data_range)
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
    test_res["batch_size"] = batch_size
    return test_res


def generateTTData(num_data, out_dim=10, max_layers=32, type="cnn", options=OPTIONS, columns=COLUMNS):
    result = collections.defaultdict(list)
    mb = ModelBuild(options)

    for _ in tqdm(range(num_data)):
        rules = None
        if type == "cnn":
            rules = CnnRules(max_layers=max_layers)
        elif type == "ffnn":
            rules = FFnnRules(max_layers=max_layers)
        assert rules is not None
        kwargs_list, layer_orders, image_shape_list = mb.generateRandomModelConfigList(rules.layer_order)
        if type == "cnn":
            model = mb.buildCnnModel(kwargs_list, layer_orders, out_dim)
        elif type == "ffnn":
            model = mb.buildFFnnModel(kwargs_list, layer_orders)
        test_res = testTT(model, num_data_range=options["Data"]["num_data"],
                          batch_size=kwargs_list[-1]["Fit"]["batch_size"])
        batch_input_shape = np.array([test_res["batch_size"], *test_res['x_shape'][1:]])
        df = convertModelToRawData(model, test_res['x_shape'][0], batch_input_shape, columns)
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

    result = generateTTData(num_data, out_dim, type="ffnn", max_layers=8)
    model, num_data, batch_input_shape = convertRawDataToModel(result["X_df"][0])
    convertModelToRawData(model, 1, batch_input_shape, result["X_df"][0].columns)
    print(result["X_df"][0].activation)

    # pickle.dump(result, open(f"../data/local_dp{DATA_POINTS}_CNN_Data_1.pkl", 'wb'))

import collections

import numpy as np
from sklearn.preprocessing import OneHotEncoder

# options: save changable configurations for the model
OPTIONS = collections.defaultdict(dict)

OPTIONS["Data"]["num_data"] = np.arange(1, 30, 1) * 32

OPTIONS["Model"]["layer"] = ["Conv2D", "Dense", "MaxPooling2D", "Dropout",
                             "Flatten"]  # the model's layer can be either Conv2D or Dense
OPTIONS["Model"]["pure_activation_rate"] = {1: 0.8, 2: 0.1, 3: 0.05, 4: 0.025, 5: 0.0125,
                                            6: 0.0125}  # {number of activation types in one NN: probability}
OPTIONS["Compile"]["optimizer"] = ["SGD", "RMSprop", "Adam", "Adadelta", "Adagrad", "Adamax", "Nadam", "Ftrl"]
OPTIONS["Compile"]["loss"] = ["categorical_crossentropy", "categorical_hinge", ]
OPTIONS["Fit"]["batch_size"] = [2, 4, 8, 16, 32, 64, 128, 256]

OPTIONS["Dense"]["units"] = range(1, 500)
OPTIONS["Dense"]["activation"] = ["linear", "relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu",
                                  "exponential"]
OPTIONS["Dense"]["use_bias"] = [True, False]

OPTIONS["Conv2D"]["filters"] = range(1, 100)
OPTIONS["Conv2D"]["kernel_size"] = list(zip(range(1, 50), range(1, 50)))  # tried product, but then RAM is in shortage
OPTIONS["Conv2D"]["strides"] = [(1, 1)] * 10 + list(zip(range(1, 3), range(1, 3)))
OPTIONS["Conv2D"]["padding"] = [*list(["same"] * 5), "valid"]
# options["Conv2D"]["dilation_rate"] = [1, 2]
OPTIONS["Conv2D"]["activation"] = ["linear", "relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu",
                                   "elu", "exponential"]
OPTIONS["Conv2D"]["use_bias"] = [True, False]

OPTIONS["MaxPooling2D"]["pool_size"] = list(zip(range(2, 10), range(2, 10)))
OPTIONS["MaxPooling2D"]["strides"] = [(1, 1)] * 10 + list(zip(range(1, 3), range(1, 3)))
OPTIONS["MaxPooling2D"]["padding"] = [*list(["same"] * 5), "valid"]

OPTIONS["Dropout"]["rate"] = list(np.arange(0, 0.5, 0.05))

NUM_DIM = 4  # the number of data dimension will be included in training data
dim_cols = [f"out_dim_{i}" for i in range(NUM_DIM)]

# columns: the columns name when convert model to a raw dataframe
tmp_column_set = set()
for k in OPTIONS.keys():
    if k == "Model":
        continue
    tmp_column_set = tmp_column_set.union(set(OPTIONS[k].keys()))
COLUMNS = ["active", "layer", *tmp_column_set, *dim_cols]

# one_hot_enc: the one hot encoder used to convert raw model dataframe into one hot
ONE_HOT_ENC = dict()
ONE_HOT_ENC["layer"] = OneHotEncoder(handle_unknown='ignore').fit(
    np.reshape(OPTIONS["Model"]["layer"], (-1, 1)))
ONE_HOT_ENC["optimizer"] = OneHotEncoder(handle_unknown='ignore').fit(
    np.reshape(OPTIONS["Compile"]["optimizer"], (-1, 1)))
ONE_HOT_ENC["loss"] = OneHotEncoder(handle_unknown='ignore').fit(np.reshape(OPTIONS["Compile"]["loss"], (-1, 1)))
ONE_HOT_ENC["activation"] = OneHotEncoder(handle_unknown='ignore').fit(
    np.reshape(list(set(OPTIONS["Dense"]["activation"]).union(set(OPTIONS["Conv2D"]["activation"]))), (-1, 1)))

# data_points: the number of data points used as the input to test model TT
DATA_POINTS = 32
# batch_size: the batch size used to test model TT
BATCH_SIZE = 4
# default_input_shape: the default input shape will be used when input needs to be created according to model's size
DEFAULT_INPUT_SHAPE = (32, 32, 3)

if __name__ == "__main__":
    print(f"batch_size={BATCH_SIZE}\n"
          f"default_input_shape={DEFAULT_INPUT_SHAPE}\n"
          f"set up constant options={OPTIONS}\n"
          f"set up one_hot_enc={ONE_HOT_ENC}\n"
          f"set up raw data columns={COLUMNS}")

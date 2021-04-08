import random

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

from model_data_util.constant import DEFAULT_INPUT_SHAPE, OPTIONS


class ModelBuild:
    def __init__(self, options=OPTIONS):
        self.options = options
        self.kwargs_list: list
        self.layer_orders: list

    def buildCnnModel(self, kwargs_list, layer_orders, out_dim):
        """
        convert a kwargs into a cnn model
        kwargs_list and layer_orders should have the same length
        """
        cnn = Sequential()
        for i, lo in enumerate(layer_orders):
            kwargs = kwargs_list[i]
            if lo == "Dense":
                cnn.add(Dense(**kwargs))
            elif lo == "Conv2D":
                cnn.add(Conv2D(**kwargs))
            elif lo == "MaxPooling2D":
                cnn.add(MaxPooling2D(**kwargs))
            elif lo == "Dropout":
                cnn.add(Dropout(**kwargs))
            elif lo == "Flatten":
                cnn.add(Flatten())
        cnn.add(Dense(out_dim, activation='softmax'))
        kwargs = kwargs_list[-1]
        cnn.compile(metrics=['accuracy'], **kwargs["Compile"])
        return cnn

    def buildFFnnModel(self, kwargs_list, layer_orders):
        """
        convert a kwargs into a ffnn model
        kwargs_list and layer_orders should have the same length
        """
        ffnn = Sequential()
        for i, lo in enumerate(layer_orders):
            kwargs = kwargs_list[i]
            if lo == "Dense":
                ffnn.add(Dense(**kwargs))
            elif lo == "Dropout":
                ffnn.add(Dropout(**kwargs))
        kwargs = kwargs_list[-1]
        ffnn.compile(metrics=['accuracy'], **kwargs["Compile"])
        return ffnn

    def chooseRandomComb(self, options_layer, activations=None):
        res = dict()
        for k, v in options_layer.items():
            if k == "activation" and activations is not None:
                res[k] = random.choice(activations)
            else:
                res[k] = (random.sample(v, 1)[0])
        return res

    def generateRandomModelConfigList(self, layer_orders,
                                      input_shape=DEFAULT_INPUT_SHAPE):
        """
        Use gloabl variable all_comb to generate random cnn model conf
        To build a model, pass the return to buildCnnModel method
        """

        def updateImageShape(_l, _kwargs, _image_shape):
            kernel_size: tuple
            if _l == "Conv2D":
                if type(_kwargs["kernel_size"]) == int:  # when kwargs["kernel_size"] was set by int
                    kernel_size = (_kwargs["kernel_size"], _kwargs["kernel_size"])
                else:
                    kernel_size = _kwargs["kernel_size"]
            elif _l == "MaxPooling2D":
                if type(_kwargs["pool_size"]) == int:  # when kwargs["kernel_size"] was set by int
                    # for program simplicity, I called pool_size as kernel_size
                    kernel_size = (_kwargs["pool_size"], _kwargs["pool_size"])
                else:
                    kernel_size = _kwargs["pool_size"]

            if type(_kwargs["strides"]) == int:  # when kwargs["strides"] was set by int
                strides = (_kwargs["strides"], _kwargs["strides"])
            else:
                strides = _kwargs["strides"]
            if _kwargs["padding"] == "valid":
                _image_shape[0] = (_image_shape[0] - kernel_size[0]) // strides[0] + 1
                _image_shape[1] = (_image_shape[1] - kernel_size[1]) // strides[1] + 1
            if _kwargs["padding"] == "same":
                if _image_shape[0] % strides[0] > 0:
                    _image_shape[0] = _image_shape[0] // strides[0] + 1
                else:
                    _image_shape[0] = _image_shape[0] // strides[0]
                if _image_shape[1] % strides[1] > 0:
                    _image_shape[1] = _image_shape[1] // strides[1] + 1
                else:
                    _image_shape[1] = _image_shape[1] // strides[1]
            assert _image_shape[0] > 0 and _image_shape[1] > 0
            return _image_shape

        def validKernelStridesSize(_l, _kwargs, _image_shape):
            if _l == "Conv2D":
                if type(_kwargs["kernel_size"]) == int:
                    kernel_size = (_kwargs["kernel_size"], _kwargs["kernel_size"])
                else:
                    kernel_size = _kwargs["kernel_size"]
            elif _l == "MaxPooling2D":
                if type(_kwargs["pool_size"]) == int:  # when kwargs["kernel_size"] was set by int
                    # for program simplicity, I called pool_size as kernel_size
                    kernel_size = (_kwargs["pool_size"], _kwargs["pool_size"])
                else:
                    kernel_size = _kwargs["pool_size"]

            if type(_kwargs["strides"]) == int:
                strides = (_kwargs["strides"], _kwargs["strides"])
            else:
                strides = _kwargs["strides"]
            judge = True
            if _l in ["Conv2D", "MaxPooling2D"]:
                judge = judge and (kernel_size[0] <= _image_shape[0] and kernel_size[1] <= _image_shape[1])
            judge = judge and (strides[0] <= _image_shape[0] and strides[1] <= _image_shape[1])
            if judge:
                return True
            else:
                return False

        options = self.options
        kwargs_list = []
        image_shape: list = list(input_shape[:2])
        image_shape_list: list = []
        # image_shape should end up in the same shape as model
        new_layer_orders = []
        max_strides = [3, 3]
        num_activations_types = {}
        pure_activation = {}
        num_activations_types["Dense"] = np.random.choice(list(self.options["Model"]["pure_activation_rate"].keys()),
                                                          p=list(
                                                              self.options["Model"]["pure_activation_rate"].values()))
        pure_activation["Dense"] = random.choices(self.options["Dense"]["activation"], k=num_activations_types["Dense"])
        for i, lo in enumerate(layer_orders):
            if lo == "Dense":
                kwargs = self.chooseRandomComb(options["Dense"], pure_activation["Dense"])
            elif lo == "Conv2D":
                if image_shape[0] == 1 or image_shape[1] == 1:
                    # if one of the image dim has only size one, we stop adding new conv2D
                    continue
                options_conv2d = options["Conv2D"].copy()
                # always ensure the kernel and strides size is smaller than the image
                options_conv2d["kernel_size"] = list(zip(range(1, image_shape[0]), range(1, image_shape[1])))
                options_conv2d["strides"] = [(1, 1)] * 10 + list(zip(range(1, max_strides[0]),
                                                                     range(1, max_strides[1])))
                kwargs = self.chooseRandomComb(options_conv2d)
                image_shape = updateImageShape(lo, kwargs, image_shape)
                max_strides = [min(max_strides[0], max(1, image_shape[0])), min(max_strides[1], max(1, image_shape[1]))]
            elif lo == "MaxPooling2D":
                if image_shape[0] == 1 or image_shape[1] == 1:
                    # if one of the image dim has only size one, we stop adding new conv2D
                    continue
                options_maxpooling2d = options["MaxPooling2D"].copy()
                options_maxpooling2d["pool_size"] = list(zip(range(1, image_shape[0]), range(1, image_shape[1])))
                options_maxpooling2d["strides"] = [(1, 1)] * 10 + list(zip(range(1, max_strides[0]),
                                                                           range(1, max_strides[1])))
                kwargs = self.chooseRandomComb(options_maxpooling2d)
                image_shape = updateImageShape(lo, kwargs, image_shape)
                max_strides = [min(max_strides[0], max(1, image_shape[0])), min(max_strides[1], max(1, image_shape[1]))]
            elif lo == "Dropout":
                kwargs = self.chooseRandomComb(options["Dropout"])
            elif lo == "Flatten":
                kwargs = {}
            # elif l == "AveragePooling2D":
            #   pass
            else:
                print("Error: layer order contained unsupported layer: %s" % lo)
            kwargs_list.append(kwargs)
            new_layer_orders.append(lo)
            image_shape_list.append(image_shape.copy())
        kwargs = {}
        for k in ["Compile", "Fit"]:
            kwargs[k] = {}
            for item in options[k].keys():
                kwargs[k][item] = random.sample(options[k][item], 1)[0]
        kwargs_list.append(kwargs)
        return kwargs_list, new_layer_orders, image_shape_list

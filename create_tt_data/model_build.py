import random

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

from constant import DEFAULT_INPUT_SHAPE


def buildCnnModel(kwargs_list, layer_orders, out_dim):
    '''
    convert a kwargs into a cnn model
    kwargs_list and layer_orders should have the same length
    '''
    cnn = Sequential()
    for i, l in enumerate(layer_orders):
        kwargs = kwargs_list[i]
        if l == "Dense":
            cnn.add(Dense(**kwargs))
        elif l == "Conv2D":
            cnn.add(Conv2D(**kwargs))
        elif l == "MaxPooling2D":
            cnn.add(MaxPooling2D(**kwargs))
        elif l == "Dropout":
            cnn.add(Dropout(**kwargs))
        elif l == "Flatten":
            cnn.add(Flatten())
    cnn.add(Dense(out_dim, activation='softmax'))
    cnn.compile(loss=kwargs_list[-1]["loss"], optimizer=kwargs_list[-1]["optimizer"], metrics=['accuracy'])
    return cnn


def buildFFnnModel(kwargs_list, layer_orders):
    '''
    convert a kwargs into a ffnn model
    kwargs_list and layer_orders should have the same length
    '''
    ffnn = Sequential()
    for i, l in enumerate(layer_orders):
        kwargs = kwargs_list[i]
        if l == "Dense":
            ffnn.add(Dense(**kwargs))
        elif l == "Dropout":
            ffnn.add(Dropout(**kwargs))
    ffnn.compile(loss=kwargs_list[-1]["loss"], optimizer=kwargs_list[-1]["optimizer"], metrics=['accuracy'])
    return ffnn


def chooseRandomComb(options_layer):
    res = dict()
    for k, v in options_layer.items():
        res[k] = (random.sample(v, 1)[0])
    return res


def generateRandomModelConfigList(layer_orders, options, max_num_hidden_layer=62, input_shape=DEFAULT_INPUT_SHAPE):
    '''
    Use gloabl variable all_comb to generate random cnn model conf
    To build a model, pass the return to buildCnnModel method
    '''

    def updateImageShape(l, kwargs, image_shape):
        if l == "Conv2D":
            if type(kwargs["kernel_size"]) == int:  # when kwargs["kernel_size"] was set by int
                kernel_size = (kwargs["kernel_size"], kwargs["kernel_size"])
            else:
                kernel_size = kwargs["kernel_size"]
        elif l == "MaxPooling2D":
            if type(kwargs["pool_size"]) == int:  # when kwargs["kernel_size"] was set by int
                # for program simplicity, I called pool_size as kernel_size
                kernel_size = (kwargs["pool_size"], kwargs["pool_size"])
            else:
                kernel_size = kwargs["pool_size"]

        if type(kwargs["strides"]) == int:  # when kwargs["strides"] was set by int
            strides = (kwargs["strides"], kwargs["strides"])
        else:
            strides = kwargs["strides"]
        if kwargs["padding"] == "valid":
            image_shape[0] = (image_shape[0] - kernel_size[0]) // strides[0] + 1
            image_shape[1] = (image_shape[1] - kernel_size[1]) // strides[1] + 1
        if kwargs["padding"] == "same":
            if image_shape[0] % strides[0] > 0:
                image_shape[0] = image_shape[0] // strides[0] + 1
            else:
                image_shape[0] = image_shape[0] // strides[0]
            if image_shape[1] % strides[1] > 0:
                image_shape[1] = image_shape[1] // strides[1] + 1
            else:
                image_shape[1] = image_shape[1] // strides[1]
        assert image_shape[0] > 0 and image_shape[1] > 0
        return image_shape

    def validKernelStridesSize(l, kwargs, image_shape):
        if l == "Conv2D":
            if type(kwargs["kernel_size"]) == int:
                kernel_size = (kwargs["kernel_size"], kwargs["kernel_size"])
            else:
                kernel_size = kwargs["kernel_size"]
        elif l == "MaxPooling2D":
            if type(kwargs["pool_size"]) == int:  # when kwargs["kernel_size"] was set by int
                # for program simplicity, I called pool_size as kernel_size
                kernel_size = (kwargs["pool_size"], kwargs["pool_size"])
            else:
                kernel_size = kwargs["pool_size"]

        if type(kwargs["strides"]) == int:
            strides = (kwargs["strides"], kwargs["strides"])
        else:
            strides = kwargs["strides"]
        judge = True
        if l in ["Conv2D", "MaxPooling2D"]:
            judge = judge and (kernel_size[0] <= image_shape[0] and kernel_size[1] <= image_shape[1])
        judge = judge and (strides[0] <= image_shape[0] and strides[1] <= image_shape[1])
        if judge:
            return True
        else:
            return False

    kwargs_list = []
    image_shape: list = list(input_shape[:2])
    image_shape_list: list = []
    # image_shape should end up in the same shape as model
    new_layer_orders = []
    max_strides = [3, 3]
    for i, l in enumerate(layer_orders):
        if l == "Dense":
            kwargs = chooseRandomComb(options["Dense"])
        elif l == "Conv2D":
            if (image_shape[0] == 1 or image_shape[1] == 1):
                # if one of the image dim has only size one, we stop adding new conv2D
                continue
            options_conv2d = options["Conv2D"].copy()
            # always ensure the kernel and strides size is smaller than the image
            options_conv2d["kernel_size"] = list(zip(range(1, image_shape[0]), range(1, image_shape[1])))
            options_conv2d["strides"] = [(1, 1)] * 10 + list(zip(range(1, max_strides[0]),
                                                                 range(1, max_strides[1])))
            kwargs = chooseRandomComb(options_conv2d)
            image_shape = updateImageShape(l, kwargs, image_shape)
            max_strides = [min(max_strides[0], max(1, image_shape[0])), min(max_strides[1], max(1, image_shape[1]))]
        elif l == "MaxPooling2D":
            if (image_shape[0] == 1 or image_shape[1] == 1):
                # if one of the image dim has only size one, we stop adding new conv2D
                continue
            options_maxpooling2d = options["MaxPooling2D"].copy()
            options_maxpooling2d["pool_size"] = list(zip(range(1, image_shape[0]), range(1, image_shape[1])))
            options_maxpooling2d["strides"] = [(1, 1)] * 10 + list(zip(range(1, max_strides[0]),
                                                                       range(1, max_strides[1])))
            kwargs = chooseRandomComb(options_maxpooling2d)
            image_shape = updateImageShape(l, kwargs, image_shape)
            max_strides = [min(max_strides[0], max(1, image_shape[0])), min(max_strides[1], max(1, image_shape[1]))]
        elif l == "Dropout":
            kwargs = chooseRandomComb(options["Dropout"])
        elif l == "Flatten":
            kwargs = {}
        # elif l == "AveragePooling2D":
        #   pass
        else:
            print("Error: layer order contained unsupported layer: %s" % l)
        kwargs_list.append(kwargs)
        new_layer_orders.append(l)
        image_shape_list.append(image_shape.copy())
    kwargs_list.append({"optimizer": random.sample(options['Model']['optimizer'], 1)[0],
                        "loss": random.sample(options['Model']['loss'], 1)[0]})
    return kwargs_list, new_layer_orders, image_shape_list

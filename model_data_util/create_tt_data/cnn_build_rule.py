"""
# Convolutional Layer:
## Rule 1: No Convolutional Layer After the First Dense Layer
## Rule 2: At least the half of layers are convolutional
## Rule 3: Convolutional layer is always the first one
#Maxpooling Layer
## Rule 3: No Maxpooling Layer After the First Dense Layer
## Rule 4: No two Maxpooling Layers Next to Each Other
## Rule 5: Maxpooling Layers are only after convolutional layers
#Dropout Layer
## Rule 6: No two Dropout Layers Next to Each Other
## Rule 7: No dropout rate higher than 0.5
## Rule 8: Maximum two Dropout Layers
#Flatten Layer and AveragePooling Layer
## Rule 9: All CNN end with one Flatten or AveragePooling Layer followed by maximum 2 dense layer

A CNN was divided into parts:
1. convolutional_part: composed by conv2d, maxpooling2d, dropout layers
2. flatten part: composed by a flatten layer
3. dense part: composed by dense and dropout layer
4. output part: compose by a dense layer with softmax as activation
"""
import random

import numpy as np


class CnnRules:
    def __init__(self, max_layers=64):
        self.first_dense_occurred = False  # Rule: No Convolutional Layer After the First Dense Layer
        self.max_dropout = 2
        self.remain_dropout = 2
        self.initial_layer = ["Conv2D"]
        self.max_layers = max_layers  # the total max layers
        self.layer_order = []
        self.generatePartAll()

    def generatePartLayers(self):
        """
        :return: the number of layers per parts
        """
        part_layers_num = {}
        part_layers_num["output"] = 1
        part_layers_num["flatten"] = 1
        part_layers_num["dense"] = random.choice(range(3))
        remain_max_layers = self.max_layers - np.sum(list(part_layers_num.values()))
        part_layers_num["conv"] = random.choice(range(3, remain_max_layers))
        return part_layers_num

    def generatePartConv(self):
        """
        Generate the layer_order of part_conv
        The available layers and their probabilities to occur are hard coded
        """

        def nextAvailableLayer(l_name):
            if l_name == "Conv2D":
                next_available_layer_weights = {"Conv2D": random.uniform(0.5, 1)}
                next_available_layer_weights["MaxPooling2D"] = (1 - next_available_layer_weights[
                    "Conv2D"]) * random.uniform(0.8, 1)
                next_available_layer_weights["Dropout"] = (1 - np.sum(list(next_available_layer_weights.values())))
                l = list(next_available_layer_weights.keys())
                l_w = list(next_available_layer_weights.values())
                next_layer = random.choices(l, weights=l_w)[0]
            elif l_name == "MaxPooling2D":
                # Rule: No two Maxpooling Layers Next to Each Other
                next_available_layer_weights = {"Conv2D": random.uniform(0.9, 1)}
                next_available_layer_weights["Dropout"] = (1 - np.sum(list(next_available_layer_weights.values())))
                l = list(next_available_layer_weights.keys())
                l_w = list(next_available_layer_weights.values())
                next_layer = random.choices(l, weights=l_w)[0]
            elif l_name == "Dropout":
                # Rule: the layer after dropout should only be Conv2D
                next_layer = "Conv2D"
            else:
                print("Error: received unsupported layer name: %s" % l_name)
            return next_layer

        self.layer_order = self.initial_layer.copy()
        for _ in range(len(self.layer_order), self.part_layers_num["conv"]):
            self.layer_order.append(nextAvailableLayer(self.layer_order[-1]))

    def generatePartFlatten(self):
        """
        Generate the layer_order of part_flatten
        The available layers and their probabilities to occur are hard coded
        """
        # self.layer_order.append(random.choice(["Flatten", "AveragePooling2D"]))
        self.layer_order.append("Flatten")

    def generatePartDense(self):
        """
        Generate the layer order of part_dense
        """
        if self.part_layers_num["dense"] == 1:
            self.layer_order.append(random.choice(["Dense", "Dropout"]))
        if self.part_layers_num["dense"] == 2:
            self.layer_order.append("Dense")
            self.layer_order.append("Dropout")

    def generatePartOutput(self):
        """
        Generate the layer order of part_output
        """
        self.layer_order.append("Dense")

    def generatePartAll(self):
        """
        Run all generatePartXXX functions
        """
        self.part_layers_num = self.generatePartLayers()
        self.generatePartConv()
        self.generatePartFlatten()
        self.generatePartDense()
        self.generatePartOutput()
        return self.layer_order


if __name__ == "__main__":
    cnn_rules = CnnRules()
    print(cnn_rules.generatePartAll())
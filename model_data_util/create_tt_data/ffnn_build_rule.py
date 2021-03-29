"""
The only rule of FFNN is that is was all made up by dense layers
"""

import random


class FFnnRules:
    def __init__(self, max_layers=64):
        self.max_layers = max_layers
        self.layer_order = []
        self.generateLayerOrder()

    def generateLayerOrder(self):
        self.layer_order = random.choice(range(1, self.max_layers + 1)) * ["Dense"]

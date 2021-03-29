import unittest

from model_data_util.create_tt_data.generate_tt_data import testTT
from model_data_util.create_tt_data.model_build import generateRandomModelConfigList, buildCnnModel
from model_data_util.create_tt_data.cnn_build_rule import CnnRules


class MyTestCase(unittest.TestCase):
    def validImageShape(self, model, image_shape_list):
        for i, s in enumerate(image_shape_list):
            if len(list(model.layers[i].output_shape[1:-1])) > 0:
                self.assertTrue(s == list(model.layers[i].output_shape[1:-1]), "Error: incorrect image shape")

    def testRandomModel(self):
        cnn_rules = CnnRules(max_layers=32)
        self.assertTrue(
            len(cnn_rules.layer_order) > 3,
            "Error: incorrect CNN layer structure")  # at least 1 convolutional layer, 1 flatten layer, and 1 dense layer
        out_dim = 10
        kwargs_list, layer_orders, image_shape_list = generateRandomModelConfigList(cnn_rules.layer_order)
        self.assertTrue(len(kwargs_list) == len(layer_orders) + 1 and len(kwargs_list) == len(image_shape_list) + 1,
                        f"Error: incorrect output {len(kwargs_list), len(layer_orders), len(image_shape_list)}")
        model = buildCnnModel(kwargs_list, layer_orders, out_dim)
        testTT(model)
        self.validImageShape(model, image_shape_list)


if __name__ == '__main__':
    unittest.main()

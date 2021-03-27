Utilities for Creating Random Model
---
The utils can be used to create random both CNN and FFNN. The model type depends on the constant variables and model
rules.

#####

To create 10 CNN data points to categorize 5 type of images, run the following command:

```buildoutcfg
python generate_tt_data.py --num_data 10 --out_dim 5
```

For further modification on randomizing model structure. We need to change the constant variables from _constant.py_ and
the rules in _model_build_rule.py_

For a simple demo, you could refer to this colab demo:
https://colab.research.google.com/drive/1bNakVoczxWkm9KdAY9T_GenYi3pKJGTk?authuser=1#scrollTo=r1Z9D3rPtpYG
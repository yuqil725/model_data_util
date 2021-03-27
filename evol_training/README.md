Utilities for Evolutionary Training
---
The algo is not being proved better than training by random data.

![alt text](https://bitbucket.org/AIpaca-Corp/model_data_util/raw/master/pic/Evolutionary%20Algo%20Flow%20Chart.png)

#####

To use the script, you need a tt_predictor model, number of evolution, training model dataframe and the corresponding training
time.

This command is an example to evolve the tt_predictor 10 times

```buildoutcfg
python evol.py --num_evol 10 --model_path XXXX --data_path XXXX
```

For further modification on randomizing model structure. We need to change the constant variables from _constant.py_ and
the rules in _model_build_rule.py_

For a simple demo, you could refer to this colab demo:
https://colab.research.google.com/drive/1bNakVoczxWkm9KdAY9T_GenYi3pKJGTk?authuser=1#scrollTo=r1Z9D3rPtpYG
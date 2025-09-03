主要的改动代码

1. `diffuser_actor/trajectory_optimization/diffuser_actor.py`中的`encode_inputs`函数，这里把`fusion_method`分为了`rgb_only`和`concat`并分别实现。
2. `diffuser_actor/utils/encoder.py`中主要增加`encode_pcds`函数，将`pcd_method`分为`cluster`和`patch`两种进行实现。


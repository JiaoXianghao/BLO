# __class&nbsp;&nbsp;MetaModel__
***
## Description

Special  model used for initialization optimization with MAML and MAML based methods.
Containing backbone model(CONV4, for example) and additional modules.

***
## Parameters

* __backbone__: Module
    Backbone model, could

* __learn_lr__: bool, default=False
    Whether to learning inner learning rate during outer optimization,
     i.e. use Meta-SGD method.
  
* __meta_lr__: float, default=0.1
    Learning rate of inner optimization.

* __use_t__: bool, default=False
    Whether to add T-layers, i.e. use MT-net method.

* __use_warp__: bool, default=False
    Whether to add Warp-blocks, i.e. use Warp-grad method.

* __num_warp_layers__: int, default=1
    Num of conv layers in one warp block.

* __use_forget__: bool, default=False
    Whether to add attenuator, i.e. use Learning-to-Forget method.

* __enable_inner_loop_optimizable_bn_params__: bool, default=False
    When use L2F method, whether to add the attenuation operation to the batch-norm modules.

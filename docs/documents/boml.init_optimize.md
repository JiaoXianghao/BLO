# __class&nbsp;&nbsp;Init__
***
## Description
Complete Meta-learning Process with MAML and MAML-based methods

Implements the meta learning procedure of MAML _`[1]`_ and four MAML based methods, 
Meta-SGD _`[2]`_, MT-net _`[3]`_, Warp-grad _`[4]`_ and L2F _`[5]`_.
***
## Parameters
* __model__: Module  
    Model containing backbone network and other auxiliary meta modules if using
            other MAML-based methods.
  
* __inner_objective__: callable  
    The inner loop optimization objective.

    Callable with signature callable(state). Defined based on modeling of
    the specific problem that need to be solved. Computing the loss of inner
    objective. The state object contains the following:

    - "data"(Tensor) - 
        Data used in inner optimization phase.
    - "target"(Tensor) - 
        Target used in inner optimization phase.
    - "model"(Module) - 
        Meta model to be updated.
    - "updated_weights"(List[Parameter]) - 
        Weights of model updated in inner-loop, will be used for forward propagation.

* __outer_objective__: callable
    The outer optimization objective.

    Callable with signature callable(state). Defined based on modeling of
    the specific problem that need to be solved. Computing the loss of outer
    objective. The state object contains the following:

    - "data"(Tensor) - 
        Data used in outer optimization phase.
    - "target"(Tensor) - 
        Target used in outer optimization phase.
    - "model"(Module) - 
        Meta model to be updated.
    - "updated_weights"(List[Parameter]) - 
        Weights of model updated in inner-loop, will be used for forward propagation.

* __inner_learning_rate__: float, default=0.01  
    Step size for inner optimization.

* __inner_loop__: int, default=5  
    Num of inner optimization steps.
  
* __use_second_order__ (optional): bool, default=True  
    Optional argument,whether to calculate precise second-order gradients during inner-loop.
  
* __learn_lr__ (optional): bool, default=False  
    Optional argument, whether to update inner learning rate during outer optimization,
    i.e. use MSGD method.

* __use_t__ (optional): bool, default=False  
    Optional argument, whether to using T-layers during optimization,i.e. use MT-net method.

* __use_warp__ (optional): bool, default=False  
    Optional argument, whether to using warp modules during optimization,i.e. use Warp-grad method.

* __use_forget__ (optional): bool, default=False  
    Optional argument, whether to add attenuation to each layers, i.e. use L2F method.

***
## Methods
??? important highlight blink "optimize()"

    The meta optimization process containing both inner loop phase and outer loop phase.
     Final grads will be calculated by outer objective and saved in the passed in model.

    Note that the implemented optimization procedure will compute the grads of meta model
    with only one single set of training and validation data samples in a batch. If
    batch size is larger than 1, then optimize() function should be called repeatedly to
    accumulate the grads of model variables for the whole batch. After that the update
    operation of model variable needs to be done outside this optimization module.

    Parameters:
    * train_data(Tensor) - The training data used in inner loop phase.

    * train_target(Tensor) - The labels of the samples in the train data.

    * validate_data(Tensor) - The validation data used in outer loop phase.

    * validate_target(Tensor) - The labels of the samples in the validation data.

    Returns

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;val_loss(Tensor) - The value of validation loss.
***
## References
_`[1]`_ C. Finn, P. Abbeel, S. Levine, "Model-Agnostic Meta-Learning for
     Fast Adaptation of Deep Networks", in ICML, 2017.

_`[2]`_ Z. Li, F. Zhou, F. Chen, H. Li, "Meta-SGD: Learning to Learn Quickly for
Few-Shot Learning", in arxiv, 2017.

_`[3]`_ Y. Lee and S. Choi, "Gradient-Based Meta-Learning with Learned Layer-wise
 Metric and Subspace", in ICML, 2018.

_`[4]`_ S. Flennerhag, A. Rusu, R. Pascanu, F. Visin, H. Yin, R. Hadsell, "Meta-learning
 with Warped Gradient Descent", in ICLR, 2020.

_`[5]`_ S. Baik, S. Hong, K. Lee, "Learning to Forget for Meta-Learning", in CVPR, 2020.

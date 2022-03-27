# __class&nbsp;&nbsp;BVFIM__
***
## Description
Calculation of the gradient of the upper model variables with BVFIM method

Implements the ul optimization procedure of  Value-Function Best-
Response (VFBR) type BLO methods, named i-level Value-Function-basedInterior-point
Method(BVFIM) _`[1]`_.

A wrapper of lower model that has been optimized in the lower optimization will
be used in this procedure.

Note that this ul optimization module should only use both.lower_optimize.Bvfim
module to finish ll optimization procedure.
***
## Parameters
* __upper_objective__: callable  
    The main optimization problem in a hierarchical optimization problem.
    
    Callable with signature callable(state). Defined based on modeling of
    the specific problem that need to be solved. Computing the loss of upper
    problem. The state object contains the following:
    
    - "data"(Tensor) - 
        Data used in the upper optimization phase.
    - "target"(Tensor) - 
        Target used in the upper optimization phase.
    - "upper_model"(Module) - 
        Upper model of the bi-level model structure.
    - "lower_model"(Module) - 
        Lower model of the bi-level model structure.

* __upper_model__: Module  
    Upper model in a hierarchical model structure whose parameters will be
    updated with upper objective and trained lower model.

* __lower_objective__: callable  
    An optimization problem which is considered as the constraint of ul problem.
    
    Callable with signature callable(state). Defined based on modeling of
    the specific problem that need to be solved. Computing the loss of ul
    problem. The state object contains the following:
    
    - "data"(Tensor) - 
        Data used in the ul optimization phase.
    - "target"(Tensor) - 
        Target used in the ul optimization phase.
    - "upper_model"(Module) - 
        Upper model of the bi-level model structure.
    - "lower_model"(Module) - 
        Lower model of the bi-level model structure.

* __z_l2_reg__ (optional): float, default=0.1
    Weight of L2 regularization term in the value function of the regularized
    LL problem, which is $\displaystyle f_\mu^*(x) = \min_{y\in\mathbb{R}^n} f(x,y) + \frac{\mu_1}{2}\|y\|^2 + \mu_2$.

* __y_l2_reg__ (optional): float, default=0.01
    Weight of L2 regularization term in the value function of the regularized
    UL problem, which is $\displaystyle \varphi(x) = \min_{y\in\mathbb{R}^n} F(x,y) + 
  \frac{\theta}{2}\|y\|^2 - \tau\ln(f_\mu^*(x)-f(x,y))$.

* __y_ln_reg__ (optional): float, default=10.
    Weight of the log-barrier penalty term in the value function of the regularized
    UL problem, as y_l2_reg.
  
***
## Methods
??? important highlight blink "compute_gradients()"

    Compute the grads of upper variable with one single set of validation data samples in the batch
    using upper objective. The grads will be saved in the passed in upper model.

    Note that the implemented ul optimization procedure will only compute
    the grads of upper variables with one single set of data samples in a batch. If
    batch size is larger than 1, then compute_gradients() function should be called
    repeatedly to accumulate the grads of upper variables for the whole batch. After
    that the update operation of upper variables needs to be done outside this module.

    Parameters:

    * validate_data(Tensor) - The validation data used for ul problem optimization.

    * validate_target(Tensor) - The labels of the samples in the validation data.

    * auxiliary_model(_MonkeyPatchBase) - Wrapper of lower model encapsulated by module higher, has been optimized in ll
        optimization phase.

    * auxiliary_model_extra(Module) - 
    
    * train_data(Tensor) - The training data used for ll problem optimization.

    * train_target(Tensor) - The labels of the samples in the train data.

    * reg_decay(float) - 
    Weight decay coefficient of L2 regularization term and log-barrier
    penalty term. The value increases with the number of iterations.

    Returns

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;upper_loss(Tensor) - The loss value of upper objective.
***
## References
_`[1]`_ R. Liu, X. Liu, X. Yuan, S. Zeng and J. Zhang, "A Value-Function-based
        Interior-point Method for Non-convex Bi-level Optimization", in ICML, 2021.
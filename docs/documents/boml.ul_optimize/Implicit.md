# __class&nbsp;&nbsp;Implicit__
***
## Description
Calculation of the gradient of the upper model variables with Implicit Gradient Based Methods.

Implements the ul optimization procedure of two implicit gradient
based methods (IGBMs), _linear system based_ method (LS) _`[1]`_ and _neumann series
based_ method (NS) _`[2]`_.

A wrapper of lower model which has been optimized in the ll optimization will
be used in this procedure.

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

* __lower_model__: Module  
    Lower model in a hierarchical model structure whose parameters will be
    updated with lower objective during ll optimization.

* __lower_learning_rate__: float  
    Step size for ll optimization.

* __k__: int  
    The maximum number of conjugate gradient iterations.

* __method__: {"LS", "NS"}, default="NS"  
    The method used in the ul problem optimization.

* __tolerance__: float, default=1e-10  
    End the method earlier when the norm of the residual is less than tolerance.
  
***
## Methods
??? important highlight blink "compute_gradients()"

    Compute the grads of upper variable with validation data samples in the batch
    using upper objective. The grads will be saved in the passed in upper model.

    Note that the implemented ul optimization procedure will only compute
    the grads of upper variables。 If the validation data passed in is only single data
    of the batch (such as few-shot learning experiment), then compute_gradients()
    function should be called repeatedly to accumulate the grads of upper variables
    for the whole batch. After that the update operation of upper variables needs
    to be done outside this module.

    Parameters:

    * validate_data(Tensor) - The validation data used for ul problem optimization.

    * validate_target(Tensor) - The labels of the samples in the validation data.

    * auxiliary_model(_MonkeyPatchBase) - Wrapper of lower model encapsulated by module higher, has been optimized in ll
        optimization phase.

    * train_data(Tensor) - The training data used for ll problem optimization.

    * train_target(Tensor) - The labels of the samples in the train data.

    Returns

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;upper_loss(Tensor) - The loss value of upper objective.
   
***
## References
_`[1]`_ A. Rajeswaran, C. Finn, S. M. Kakade, and S. Levine, "Meta-learning
 with implicit gradients", in NeurIPS, 2019.

_`[2]`_ J. Lorraine, P. Vicol, and D. Duvenaud, "Optimizing millions of
 hyperparameters by implicit differentiation", in AISTATS, 2020.
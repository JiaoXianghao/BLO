# __class&nbsp;&nbsp;Iapttgm__
***
## Description
UL Variable Gradients Calculation with IAPTT-GM Method

Implements the UL problem optimization procedure of Initialization Auxiliary and
Pessimistic Trajectory Truncated Gradient Method(IAPTT-GM)_`[1]`_, which designed
for nal BLO in the absence of Lower-Level Convexity(LLC) hypothesis.

A wrapper of lower model that has been optimized in the lower optimization will
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
  
* __lower_model__: Module  
    Lower model in a hierarchical model structure whose parameters will be
    updated with lower objective during ll optimization.
  
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

    * max_loss_iter(int) - The step of lower optimization loop which has the maximum
    loss value. The backpropagation trajectory shall be truncated here rather than the whole
    lower-loop.

    Returns

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;upper_loss(Tensor) - The loss value of upper objective.
***
## References
_`[1]`_ R. Liu, Y. Liu, S. Zeng, J. Zhang, "Towards Gradient-based Bilevel
     Optimization with Non-convex Followers and Beyond", in NeurIPS, 2021.
# __class&nbsp;&nbsp;RAD__
***
## Description
UL Variable Gradients Calculation with Reverse-mode AD

Implements the ul optimization procedure with Reverse-mode Auto Diff method _`[1]`_.

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

    Returns

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;upper_loss(Tensor) - The loss value of upper objective.
***
## References
_`[1]`_ L. Franceschi, P. Frasconi, S. Salzo, R. Grazzi, and M. Pontil, "Bilevel
     programming for hyperparameter optimization and meta-learning", in ICML, 2018.
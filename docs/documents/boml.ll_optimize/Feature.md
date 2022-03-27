# __class&nbsp;&nbsp;Feature__
***
## Description
Lower level model optimization procedure

Implements the ll problem optimization procedure of two explicit gradient based
methods (EGBMs) with lower-level singleton (LLS) assumption, _Reverse-mode AutoDiff_
method (RAD) _`[1]`_ and _Truncated RAD_ method (T-RAD) _`[2]`_, as well as two methods
without LLS, _Bi-level Descent Aggregation_ (BDA) _`[3]`_ and _Initialization Auxiliary
and Pessimistic Trajectory Truncated Gradient_ method (IAPTT-GM) _`[4]`_.

The implemented ll optimization procedure will optimize a wrapper of lower model 
for further using in the following ul optimization. 
***
## Parameters
* __lower_objective__: callable  
  An optimization problem which is considered as the constraint of ll problem.

    Callable with signature callable(state). Defined based on modeling of
    the specific problem that need to be solved. Computing the loss of ll
    problem. The state object contains the following:

    - "data"(Tensor) - 
        Data used in the ll optimization phase.
    - "target"(Tensor) - 
        Target used in the ll optimization phase.
    - "upper_model"(Module) - 
        Upper model of the bi-level model structure.
    - "lower_model"(Module) - 
        Lower model of the bi-level model structure.

* __lower_loop__: int  
  Updating iterations over ll optimization.

* __upper_model__: Module  
    Upper model in a hierarchical model structure whose parameters will be
    updated with upper objective.

* __upper_objective__: callable  
    The main optimization problem in a hierarchical optimization problem.

    Callable with signature callable(state). Defined based on modeling of
    the specific problem that need to be solved. Computing the loss of ul
    problem. The state object contains the following:

    - "data"(Tensor)
        Data used in the ul optimization phase.
    - "target"(Tensor)
        Target used in the ul optimization phase.
    - "upper_model"(Module)
        Upper model of the bi-level model structure.
    - "lower_model"(Module)
        Lower model of the bi-level model structure.

* __lower_model__: Module  
    Lower model in a hierarchical model structure whose parameters will be
    updated with lower objective during ll optimization.

* __acquire_max_loss__ (optional): bool, default=False  
    Optional argument,if set True then will use IAPTT-GM method as ll
    optimization method.

* __alpha__ (optional): float, default=0  
  The aggregation parameter for BDA method, where alpha ∈ (0, 1) denotes
  the ratio of lower objective to upper objective during lower optimizing.

* __truncate_iters__ (optional): int, default=0  
    Parameter for T-RAD method, defining number of iterations to truncate
    in the back propagation process during lower optimizing.

* __lower_opt__ (optional): Optimizer, default=None  
    The original optimizer of lower model.
  
***
## Methods
??? important highlight blink "optimize()"
    
    Execute the lower optimization procedure with training data samples using lower 
    objective. The passed in wrapper of lower model will be updated.
    
    Parameters:  
    
    * train_data(Tensor) - The training data used for ll problem optimization.  
    
    * train_target(Tensor) - The labels of the samples in the train data.
    
    * auxiliary_model(_MonkeyPatchBase) - 
        Wrapper of lower model encapsulated by module higher, will be optimized in ll
        optimization procedure.  
    
    * auxiliary_opt(DifferentiableOptimizer) - 
        Wrapper of ll optimizer encapsulated by module higher, will be used in ll
        optimization procedure.
    
    * validate_data(Tensor, optional, default=None) - 
        The validation data used for ul problem optimization. Needed 
        when using BDA method or IAPTT-GM method. 
    
    * validate_target(Tensor, optional, default=None) - 
        The labels of the samples in the validation data. Needed when using
        BDA method or IAPTT-GM method.
***
## References
_`[1]`_ L. Franceschi, P. Frasconi, S. Salzo, R. Grazzi, and M. Pontil, "Bilevel
     programming for hyperparameter optimization and meta-learning", in ICML, 2018

_`[2]`_ A. Shaban, C. Cheng, N. Hatch, and B. Boots, "Truncated backpropagation
 for bilevel optimization", in AISTATS, 2019.

_`[3]`_ R. Liu, P. Mu, X. Yuan, S. Zeng, and J. Zhang, "A generic first-order algorithmic
 framework for bi-level programming beyond lower-level singleton", in ICML, 2020.

_`[4]`_ R. Liu, Y. Liu, S. Zeng, and J. Zhang, "Towards Gradient-based Bilevel
 Optimization with Non-convex Followers and Beyond", in NeurIPS, 2021.


        

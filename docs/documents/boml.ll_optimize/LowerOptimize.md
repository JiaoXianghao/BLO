
# __boml.ll_optimize__
***
## __class&nbsp;&nbsp;LowerOptimize__
***
## __class&nbsp;&nbsp;BOMLLowerOptimizeFeature__
### Description
Lower level model optimization procedure with Implicit Gradient Based Methods  # todo

Implements the ll problem optimization procedure of two explicit gradient based
methods (EGBMs) with lower-level singleton (LLS) assumption, Reverse-mode AutoDiff
method (RAD) _`[1]`_ and Truncated RAD method (T-RAD) _`[2]`_, as well as two methods
without LLS, Bi-level descent aggregation (BDA) _`[3]`_ and Initialization Auxiliary
and Pessimistic Trajectory Truncated Gradient Method (IAPTT-GM) _`[4]`_.

The implemented ll optimization procedure will optimize a wrapper of lower model 
for further using in the following ul optimization. Only one set of training 
data samples in the batch will be used in this procedure.

###Parameters
* __lower_objective__(callable) - 
  An optimization problem which is considered as the constraint of ll problem.

    Callable with signature callable(state). Defined based on modeling of
    the specific problem that need to be solved. Computing the loss of ll
    problem. The state object contains the following:

    - "data"(Tensor) - 
        Data used in the lower optimization phase.
    - "target"(Tensor) - 
        Target used in the lower optimization phase.
    - "upper_model"(Module) - 
        Upper model of the bi-level model structure.
    - "lower_model"(Module) - 
        Lower model of the bi-level model structure.

* __lower_loop__(int) - Updating iterations over lower level optimization.

* __upper_model__(Module) - 
    Upper model in a hierarchical model structure whose parameters will be
    updated with upper objective.

* __upper_objective__(callable) - 
    The main optimization problem in a hierarchical optimization problem.

    Callable with signature callable(state). Defined based on modeling of
    the specific problem that need to be solved. Computing the loss of ul
    problem. The state object contains the following:

    - "data"(Tensor)
        Data used in the upper optimization phase.
    - "target"(Tensor)
        Target used in the upper optimization phase.
    - "upper_model"(Module)
        Upper model of the bi-level model structure.
    - "lower_model"(Module)
        Lower model of the bi-level model structure.

* __lower_model__(Module) - 
    Lower model in a hierarchical model structure whose parameters will be
    updated with lower objective during lower-level optimization.

* __acquire_max_loss__(bool, optional, default=False) - 
    Optional argument,if set True then will use IAPTT-GM method as lower
    optimization method.

* __alpha__(float, optional, default=0) - 
    The aggregation parameter for Bi-level Descent Aggregation method, where
    alpha ∈ (0, 1) denotes the ratio of lower objective to upper objective
    during lower optimizing.

* __truncate_iters__(int, optional, default=0) - 
    Parameter for Truncated Reverse method, defining number of
    iterations to truncate in the back propagation process during lower
    optimizing.

* __lower_opt__(Optimizer, optional, default=None) - 
    The original optimizer of lower model.
### Methods
optimize(train_data,  train_target, auxiliary_model, auxiliary_opt, validate_data=None, validate_target=None)
> Execute the lower optimization procedure with one single set of training data
samples in the batch using lower objective. The passed in wrapper of lower model.

> Parameters:  

> * train_data(Tensor) - The training data used for ll problem optimization.  

> * train_target(Tensor) - The labels of the samples in the train data.

> * auxiliary_model(_MonkeyPatchBase) - 
    Wrapper of lower model encapsulated by module higher, will be optimized in lower
    optimization procedure.  

> * auxiliary_opt(DifferentiableOptimizer) - 
    Wrapper of lower optimizer encapsulated by module higher, will be used in lower
    optimization procedure.

> * validate_data(Tensor, optional, default=None) - 
    The validation data used for ul problem optimization. Needed 
    when using BDA method or IAPTT-GM method. 

> * validate_target(Tensor, optional, default=None) - 
    The labels of the samples in the validation data. Needed when using
    BDA method or IAPTT-GM method.
***
###References
_`[1]`_ L. Franceschi, P. Frasconi, S. Salzo, R. Grazzi, and M. Pontil, "Bilevel
     programming for hyperparameter optimization and meta-learning", in ICML, 2018

_`[2]`_ A. Shaban, C. Cheng, N. Hatch, and B. Boots, "Truncated backpropagation
 for bilevel optimization", in AISTATS, 2019.

_`[3]`_ R. Liu, P. Mu, X. Yuan, S. Zeng, and J. Zhang, "A generic first-order algorithmic
 framework for bi-level programming beyond lower-level singleton", in ICML, 2020.

_`[4]`_ R. Liu, Y. Liu, S. Zeng, and J. Zhang, "Towards Gradient-based Bilevel
 Optimization with Non-convex Followers and Beyond", in NeurIPS, 2021.
## __class&nbsp;&nbsp;BOMLLowerOptimizeBvfim__
    selection:
      members:
        - adapt
        - clone


        -

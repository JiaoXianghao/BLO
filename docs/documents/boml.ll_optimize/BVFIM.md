# __class&nbsp;&nbsp;BVFIM__
***
## Description
Lower model optimization procedure of Value-Function-based Interior-point Method

Implements the ll problem optimization procedure of Value-Function Best-
Response (VFBR) type BLO methods, named i-level Value-Function-basedInterior-point
Method(BVFIM) _`[1]`_.

The implemented lower level optimization procedure will optimize a wrapper of lower
model for further using in the following upper level optimization.
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

* __z_loop__ (optional): int, default=5
    Num of steps to obtain a low ll problem value, i.e. optimize ll variable
    with ll problem. Regarded as $T_z$ in the paper.

* __y_loop__ (optional): int, default=5
    Num of steps to obtain a optimal ll variable under the ll problem value obtained
    after z_loop, i.e. optimize the updated ll variable with ul problem. Regarded as
    Regarded as $T_y$ in the paper.

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
    
    * auxiliary_model_extra: Module

    * auxiliary_opt_extra: Optimizer

    * validate_data(Tensor) - 
        The validation data used for ul problem optimization. Needed 
        when using BDA method or IAPTT-GM method. 
    
    * validate_target(Tensor) - 
        The labels of the samples in the validation data. Needed when using
        BDA method or IAPTT-GM method.

    * reg_decay(float) - 
        Weight decay coefficient of L2 regularization term and log-barrier
        penalty term. The value increases with the number of iterations
***
## References
_`[1]`_ R. Liu, X. Liu, X. Yuan, S. Zeng and J. Zhang, "A Value-Function-based
    Interior-point Method for Non-convex Bi-level Optimization", in ICML, 2021.


        

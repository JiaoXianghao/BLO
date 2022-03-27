# __class&nbsp;&nbsp;BOTHOptimizer__
***
## Description
Wrapper for performing bi-level optimization and gradient-based initialization optimization

BOTHOptimizer is the wrapper of Bi-Level Optimization(BLO) and Initialization Optimization(Initialization-based
EGBR) process which builds LL, UL and Initialization problem solver with corresponding method modules
and uses in training phase. The optimization process could also be done by using methods packages directly.
    
***
## Parameters

* __method__: str
    Define basic method for following training process, it should be included in
    ['MetaInit', 'MetaRepr']. 'MetaInit' type refers to meta-learning optimization
    strategy, including methods like 'MAML, FOMAML, TNet, WarpGrad, L2F'; 'MetaRepr'
    type refers to bi-level optimization strategy, includes methods like 'BDA, RHG,
    Truncated RHG, Implicit HG, Onestage, BVFIM, IAPTT-GM, BSG'.

* __lower_method__: str, default=None
    method chosen for solving LL problem, including ['Feature' ,'BVFIM'].

* __upper_method__: str, default=None
    Method chosen for solving UL problem, including ['Trad' ,'IAPTTGM', 'Implicit', 'BSG',
    'Onestage', 'BVFIM'].

* __lower_objective__: callable, default=None
    An optimization problem which is considered as the constraint of UL problem.

    Callable with signature callable(state). Defined based on modeling of
    the specific problem that need to be solved. Computing the loss of upper
    problem. The state object contains the following:

    - "data"
        Data used in the LL optimization phase.
    - "target"
        Target used in the LL optimization phase.
    - "upper_model"
        UL model of the bi-level model structure.
    - "lower_model"
        LL model of the bi-level model structure.

* __upper_objective__: callable, default=None
    The main optimization problem in a hierarchical optimization problem.

    Callable with signature callable(state). Defined based on modeling of
    the specific problem that need to be solved. Computing the loss of upper
    problem. The state object contains the following:

    - "data"
        Data used in the UL optimization phase.
    - "target"
        Target used in the UL optimization phase.
    - "upper_model"
        Ul model of the bi-level model structure.
    - "lower_model"
        LL model of the bi-level model structure.

* __inner_objective__: callable, default=None
    The inner loop optimization objective.

    Callable with signature callable(state). Defined based on modeling of
    the specific problem that need to be solved. Computing the loss of inner
    objective. The state object contains the following:

    - "data"
        Data used in inner optimization phase.
    - "target"
        Target used in inner optimization phase.
    - "model"
        Meta model to be updated.
    - "updated_weights"
        Weights of model updated in inner-loop, will be used for forward propagation.

* __outer_objective__: callable, default=None
    The outer optimization objective.

    Callable with signature callable(state). Defined based on modeling of
    the specific problem that need to be solved. Computing the loss of outer
    objective. The state object contains the following:

    - "data"
        Data used in outer optimization phase.
    - "target"
        Target used in outer optimization phase.
    - "model"
        Meta model to be updated.
    - "updated_weights"
        Weights of model updated in inner-loop, will be used for forward propagation.

* __lower_model__: Module, default=None
    The model whose parameters will be updated during upper-level optimization.

* __upper_model__: Module, default=None
    Upper model in a hierarchical model structure whose parameters will be
    updated with upper objective.

* __meta_model__: MetaModel or Module, default=None
    Model whose initial value will be optimized. If choose MAML method to optimize, any user-defined
    torch nn.Module could be used as long as the definition of forward() function meets the standard;
    but if choose other derived methods, internally defined both.utils.model.meta_model should be used
    for related additional modules.

* __total_iters__: int, default=60000
    Total iterations of the experiment, used to set weight decay.

***
## Methods
??? important highlight blink "build_ll_problem_solver()"

    Build LL-problem solver with both.lower_optimizer module,
    which will optimize lower model for further using in ul optimization
    procedure. Setting the value of parameters according to the selected method.

    Parameters:

    * lower_loop(int) - 
        The total number of iterations for lower gradient descent optimization.

    * lower_objective_optimizer(Optimizer) - 
        Optimizer of lower model, defined outside this module and will be used
        in LL optimization procedure.

    * update_lower_model_with_step_num(int, default=0) - 
        Whether to update lower model variables after ll optimization. Default
        value 0 means that lower model will maintain initial state after ll optimization
        process. If set this parameter to a positive integer k, then the lower
        model will save the updated results of step k of the ll optimization loop.
        Setting it when experiment doesn't have fine-tune stage.

    * truncate_iter(int, default=0) - 
        Specific parameter for Truncated Reverse AD method, defining number of
        iterations to truncate in the back propagation process during lower
        optimizing.

    * acquire_max_loss(bool, default=False) - 
        Specific parameter for IAPTT-GM method,if set True then will use IAPTT-GM method as lower
        optimization method.

    * alpha_init(float, default=0.0) - 
        Specify parameter for BDA method. The aggregation parameter for Bi-level descent
        aggregation method, where alpha ∈ (0, 1) denotes the ratio of lower objective
        to upper objective during lower optimizing.

    * learn_alpha(bool, default=False) - 
        Specify parameter for BDA method to decide whether to initialize alpha as a hyper
        parameter.

    * learn_alpha_itr(int, default=0) - 
        Specify parameter for BDA method to specify whether to initialize alpha as a vector, of which
        every dimension's value is step-wise scale factor fot the optimization process.

    * z_loop(int, default=5) - 
        Specify parameter for BVFIM method. Num of steps to obtain a low ll problem value, i.e.
         optimize ll variable with ll problem. Regarded as $T_z$ in the paper.

    * y_loop(int, default=5) - 
        Specify parameter for BVFIM method. Num of steps to obtain a optimal ll variable under the
        ll problem value obtained after z_loop, i.e. optimize the updated ll variable with ul
         problem. Regarded as Regarded as $T_y$ in the paper.

    * ll_l2_reg(float, default=0.1) - 
        Specify parameter for BVFIM method. Weight of L2 regularization term in the value
        function of the regularizedLL problem. Referring to module both.upper_optimize.bvfim
        for more details.

    * ul_l2_reg(float, default=0.01) - 
        Specify parameter for BVFIM method. Weight of L2 regularization term in the
        value function of the regularized UL problem. Referring to module
        both.upper_optimize.bvfim for more details.

    * ul_ln_reg(float, default=10.) - 
        Specify parameter for BVFIM method. Weight of the log-barrier penalty term in the
        value function of the regularized UL problem. Referring to module both.upper_optimize.bvfim
        for more details.

    * reg_decay(bool, default=True) - 
        Specify parameter for BVFIM method. Whether to use weight decay coefficient of
         L2 regularization term and log-barrier penalty term.

??? important highlight blink "build_ul_problem_solver()"

    Setting up UL optimization module. Select desired method through given parameters
    and set related experiment parameters.

    Details of parameter settings for each particular method are available in the specific
    method module of both.upper_optimize.

    Parameters:

    * ul_optimizer(Optimizer) - 
        Optimizer of upper model, defined outside this module and will be used
        in UL optimization procedure.

    * method_igbm(str, default=None) - 
        Specific parameter for Implicit method. The method used in the UL problem optimization.

    * k(int, default=10) - 
        Specific parameter for Implicit method. The maximum number of conjugate gradient iterations.

    * tolerance(float, default=1e-10) - 
        Specific parameter for Implicit method. End the method earlier when the norm of the
        residual is less than tolerance.

    * r(float, default=1e-2) - 
        Parameter for One-stage RAD method and used to adjust scalar epsilon. Value 0.01 of r is
        recommended for sufficiently accurate in the paper. Referring to module
        both.upper_optimize.onestage for more details.

    * ll_l2_reg(float, default=0.1) - 
        Specify parameter for BVFIM method. Weight of L2 regularization term in the value
        function of the regularizedLL problem. Referring to module both.upper_optimize.bvfim
        for more details.

    * ul_l2_reg(float, default=0.01) - 
        Specify parameter for BVFIM method. Weight of L2 regularization term in the
        value function of the regularized UL problem. Referring to module
        both.upper_optimize.bvfim for more details.

    * ul_ln_reg(float, default=10.) - 
        Specify parameter for BVFIM method. Weight of the log-barrier penalty term in the
        value function of the regularized UL problem. Referring to module both.upper_optimize.bvfim
        for more details.
    
??? important highlight blink "build_meta_problem_solver()"

    Setting up meta-learning optimization module. Select desired method through given parameters
    and set set related experiment parameters.

    Note that among three methods MT-net, Warpgrad and L2F, only one can be used; while First-order
    and MSGD can be combined with others.

    Parameters:

    * meta_optimizer(Optimizer) - 
        The optimizer used to update initial values of meta model after
        an iteration.

    * inner_loop(int, default=5) - 
        Num of inner optimization steps.

    * inner_learning_rate(float, default=0.01) - 
        Step size for inner optimization.

    * use_second_order(bool, default=True) - 
        Optional argument, whether to calculate precise second-order gradients during inner-loop.

    * learn_lr(bool, default=False) - 
        Optional argument, whether to update inner learning rate during outer optimization,
        i.e. use MSGD method.

    * use_t(bool, default=False) - 
        Optional argument, whether to using T-layers during optimization,i.e. use MT-net method.

    * use_warp(bool, default=False) - 
        Optional argument, whether to using warp modules during optimization,i.e. use Warp-grad method.

    * use_forget(bool, default=False) - 
        Optional argument, whether to add attenuation to each layers, i.e. use L2F method.

??? important highlight blink "run_iter()"

    Run an iteration with a data batch and updates the parameters of upper model or meta-model.

    Parameters:

    * train_data_batch(Tensor) - A batch of train data,which is used during lower optimizing.

    * train_target_batch(Tensor) - A batch of train target,which is used during lower optimizing.

    * validate_data_batch(Tensor) - A batch of test data,which is used during upper optimizing.

    * validate_target_batch(Tensor) - A batch of test target,which is used during upper optimizing.

    * batch_size(int) - The number of training samples in each batch.

    * current_iter(int) - The num of current iter.

    * forward_with_whole_batch(bool, default=True) - Whether to feed in the whole
      data batch when doing forward propagation.
                When setting to False, each single data in the batch will be fed into model
                during this iteration. This useful for some experiment having special setting,
                like few-shot learning.

    Returns

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;validation_loss(Tensor) - The value of validation loss value.

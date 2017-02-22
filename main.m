%%% main script
clear all;

%% INITIAL WORKSPACE SETUP 
if ~exist('models','dir')
    mkdir('models');
end

if ~exist('results','dir')
    mkdir('results');
end

%% FLAGS
MANUAL_TEST = 1;
MODEL_SELECTION = 0;
TEST = 0;

MOVEMENT_AAL = 0;
KITCHEN = 1;

assert(MOVEMENT_AAL + KITCHEN == 1);

%% READ DATA
global example
example = containers.Map();

if MOVEMENT_AAL
    addpath 'Movement AAL'\
    [ trainInputSequence, trainTargetSequence, testInputSequence, testTargetSequence ] = preprocessor('nonlocal4');
    
    % Configuring example
    example('dataset') = 'Movement AAL';
    example('objective_function') = @(preds, tgts) check_accuracy(preds, tgts);
    example('objective') = 'maximize';
else
    addpath Kitchen\
    [ trainInputSequence, trainTargetSequence, testInputSequence, testTargetSequence ] = preprocessor();
    
    % Configuring example
    example('dataset') = 'Kitchen';
    example('objective_function') = @(preds, tgts) compute_MAE(preds,tgts);
    example('objective') = 'minimize';
end

%% MANUAL ESN TEST PART

%%%% Fixed parameters
nInputUnits = size(trainInputSequence{1}, 2);
nOutputUnits = 1;

washout = 10;

if MANUAL_TEST
    
    nInternalUnits = 100;
    rho = 0.9;
    leaky_param = 0.5;
    
    % Ridge-regression
    lambda = 10;
    
    %RLS
    rls_delta = 100;
    rls_lambda = 0.9999995;
 
%         'methodWeightCompute', 'pseudoinverse' ...
%
%         'methodWeightCompute', 'ridge_regression', ...
%         'ridge_parameter', lambda ...
% 

    % ESN TEST
    my_esn = ESN( nInputUnits, nInternalUnits, nOutputUnits, ...
        'rho', rho, ...
        'type', 'leaky_esn', ...
        'leaky_parameter', leaky_param, ...
        'methodWeightCompute', 'rls', ...
        'rls_delta', rls_delta, ...
        'rls_lambda', rls_lambda ...
   );        
    
%     % DROPOUT ESN TEST
%     p = 0.8;
%     
%     my_esn = DropoutESN ( nInputUnits, nInternalUnits, nOutputUnits, ...
%         'rho', rho, ...
%         'type', 'leaky_esn', ...
%         'leaky_parameter', leaky_param , ...
%         'methodWeightCompute', 'rls', ...
%         'rls_lambda', rls_lambda, ...
%         'rls_delta', rls_delta, ...
%         'dropout_type', 'dropout', ...
%         'p', p ...
%     );
    
    
    % Training
    ls_tr = my_esn.train(trainInputSequence, trainTargetSequence, washout);
    tr_preds = my_esn.test(trainInputSequence, NaN, washout);
    
    tr_tgts = compute_mutiple_series_targets(trainTargetSequence, washout);
    tr_tgts = cat(1, tr_tgts{:});
    
    f = example('objective_function');
    tr_perf = f(tr_preds, tr_tgts);
   
    % Test
    ts_preds = my_esn.test(testInputSequence, NaN, washout);
    ts_tgts = compute_mutiple_series_targets(testTargetSequence, washout);
    ts_tgts = cat(1, ts_tgts{:});
    
    ts_perf = f(ts_preds, ts_tgts);
    
    if MOVEMENT_AAL
        fprintf('Training ACC: %g - Test ACC: %g \n', tr_perf, ts_perf);
    else
        fprintf('Training MAE: %g - Test MAE: %g \n', tr_perf, ts_perf);
    end
    fprintf('------------------------------- \n');
    
end


%% MODEL SELECTION PART
if MODEL_SELECTION
    
    %%%% Fixed parameters
    nInputUnits = size(trainInputSequence{1}, 2);
    nOutputUnits = 1;
    
   fixed_params = containers.Map( ...
        {'nInputUnits', 'nOutputUnits', 'rho', 'type', 'methodWeightCompute', 'rls_lambda'}, ...
        {nInputUnits, nOutputUnits, 0.99, 'leaky_esn', 'rls', 0.9999995} ...
        );
    
    % Hyperparameters
    nInternalUnits = [10, 20];
    leaky_parameter = [0.1];
    rls_delta = [0.001, 0.01];
    
    hyperparameters = containers.Map();
    hyperparameters('nInternalUnits') = nInternalUnits;
    hyperparameters('leaky_parameter') = leaky_parameter;
    hyperparameters('rls_delta') = rls_delta;
    
    %% MODEL SELECTION OF THE CLASSICAL ESN
    fprintf('Performing model selection... \n');
    
    model_selection_log_f = fopen('models/ESN.log','w');
    
    % Defining model type
    model_type = 'ESN';
    [ best_model , performance] = model_selection(model_type, fixed_params, hyperparameters, trainInputSequence, trainTargetSequence, model_selection_log_f);
    
    trained_models = best_model('trained_models');
    model = trained_models(best_model('best_model_idx'));
    
    fprintf(model_selection_log_f, '============ ESN Best model selected ============ \n');
    fprintf(model_selection_log_f, 'Hyperparameters: \n');
    fprintf(model_selection_log_f, ' - nInternalUnits: %d \n', model.nReservoirUnits);
    fprintf(model_selection_log_f, ' - leaky_parameter: %g \n', model.leaky_parameter);
    fprintf(model_selection_log_f, ' - rls_delta: %g \n', model.rls_delta);
    fprintf(model_selection_log_f, ' --> Expected performace (MAE): %g \n', performance);
    fprintf(model_selection_log_f, '================================================ \n');
    
    fclose(model_selection_log_f);
    
    %% MODEL SELECTION OF DROPOUT MODEL
    % Defining model type
    model_type = 'DropoutESN';
    
    p_param = [0.8, 0.5, 0.3];
    for i = 1:size(p_param,2)
        
        switch p_param(i)
            case 0.8
                filename = 'zero_otto.log';
            case 0.5
                filename = 'zero_cinque.log';
            case 0.3
                filename = 'zero_tre.log';
                
            otherwise
                error('ERROR: performing model selection on DropoutESN, unknown p set!')
        end
                
        model_selection_log_f = fopen(strcat('models/',filename),'w');
        
        % And dropout params
        fixed_params('dropout_type') = 'dropout';
        fixed_params('p') = p_param(i);
        
        [ best_model , performance] = model_selection(model_type, fixed_params, hyperparameters, trainInputSequence, trainTargetSequence, model_selection_log_f);
        
        trained_models = best_model('trained_models');
        model = trained_models(best_model('best_model_idx'));
        
        fprintf(model_selection_log_f, '============ DropoutESN Best model selected ============ \n');
        fprintf(model_selection_log_f, 'Percentage of input retaining: %g \n', p_param(i));
        fprintf(model_selection_log_f, 'Hyperparameters: \n');
        fprintf(model_selection_log_f, ' - nInternalUnits: %d \n', model.nReservoirUnits);
        fprintf(model_selection_log_f, ' - leaky_parameter: %g \n', model.leaky_parameter);
        fprintf(model_selection_log_f, ' - rls_delta: %g \n', model.rls_delta);
        fprintf(model_selection_log_f, ' --> Expected performace (MAE): %g \n', performance);
        fprintf(model_selection_log_f, '================================================ \n');
        
        fclose(model_selection_log_f);
        
    end

end

if TEST
    
    % Plain test
    plain_test_res = zeros(4,2);
    
    %% Loading models
    best_esn = load('models/esn.mat');
    trained_models = best_esn.best_model('trained_models');
    esn = trained_models(best_esn.best_model('best_model_idx'));    
    
    % Plain test
    [tr_avg_acc, ts_avg_acc] = plain_test(trained_models, trainInputSequence, trainTargetSequence, testInputSequence, testTargetSequence, washout);
    plain_test_res(1, :) = [tr_avg_acc, ts_avg_acc];
    
    best_desn_p_zero_otto = load('models/zero_otto.mat');
    trained_models = best_desn_p_zero_otto.best_model('trained_models');
    desn_p_zero_otto = trained_models(best_esn.best_model('best_model_idx'));
        
    % Plain test
    [tr_avg_acc, ts_avg_acc] = plain_test(trained_models, trainInputSequence, trainTargetSequence, testInputSequence, testTargetSequence, washout);
    plain_test_res(2, :) = [tr_avg_acc, ts_avg_acc];
    
    best_desn_p_zero_cinque = load('models/zero_cinque.mat');
    trained_models = best_desn_p_zero_cinque.best_model('trained_models');
    desn_p_zero_cinque = trained_models(best_esn.best_model('best_model_idx'));
    
    % Plain test
    [tr_avg_acc, ts_avg_acc] = plain_test(trained_models, trainInputSequence, trainTargetSequence, testInputSequence, testTargetSequence, washout);
    plain_test_res(3, :) = [tr_avg_acc, ts_avg_acc];
    
    best_desn_p_zero_tre = load('models/zero_tre.mat');
    trained_models = best_desn_p_zero_tre.best_model('trained_models');
    desn_p_zero_tre = trained_models(best_esn.best_model('best_model_idx'));
   
    % Plain test
    [tr_avg_acc, ts_avg_acc] = plain_test(trained_models, trainInputSequence, trainTargetSequence, testInputSequence, testTargetSequence, washout);
    plain_test_res(4, :) = [tr_avg_acc, ts_avg_acc];
    
    % Saving plain test results
    save('results/plain_test', 'plain_test_res');
    
    %% Selecting best topologies for Dropout test
    models = { esn, desn_p_zero_otto, desn_p_zero_cinque, desn_p_zero_tre };
    
    % Training every model on training data (TR+VL)
    for i = 1:size(models,2)
        model = models{i};
        model.train(trainInputSequence, trainTargetSequence, washout);
    end
    
    % First of all I want to test all models on test set (plain).
    plain_ts_res = zeros(size(models));
    
    ts_tgts = compute_mutiple_series_targets(testTargetSequence, washout);
    ts_tgts = cat(1,ts_tgts{:});
    
    for i=1:size(models,2)
        ts_preds = models{i}.test(testInputSequence, NaN, washout);
        plain_ts_res(i) = compute_MAE(ts_preds, ts_tgts);
    end
    
    save('results/best_models_plain_test', 'plain_ts_res');

    % Then it's the funny stuff: start removing units from ts_input!
    dropping_ts_res = {};
    for i=1:size(models,2)
        dropping_ts_res{i,1} = test_drop_units_incr(models{i}, testInputSequence, testTargetSequence, washout);
    end
    
    save('results/best_models_dropping_test', 'dropping_ts_res');
    
end

fprintf('Bye bye :-) \n');
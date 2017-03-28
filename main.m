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
MANUAL_TEST = 0;
MODEL_SELECTION = 0;
TEST = 1;

MOVEMENT_AAL = 0;
KITCHEN = 1;

assert(MOVEMENT_AAL + KITCHEN == 1);

if MOVEMENT_AAL
    type = 'seq2elem';
else
    type = 'seq2seq';
end

%% READ DATA
global example
example = containers.Map();

if MOVEMENT_AAL
    addpath 'Movement AAL'
    [ trainInputSequence, trainTargetSequence, testInputSequence, testTargetSequence ] = preprocessor('nonlocal4');
    
    % Configuring example
    example('dataset') = 'Movement AAL';
    example('objective_function') = @(preds, tgts) check_accuracy(tgts, preds);
    example('objective') = 'maximize';
else
    addpath 'Kitchen'
    [ trainInputSequence, trainTargetSequence, testInputSequence, testTargetSequence ] = preprocessor();
    
    % Configuring example
    example('dataset') = 'Kitchen';
    example('objective_function') = @(preds, tgts) compute_MAE(tgts, preds);
    example('objective') = 'minimize';
end

%% Fixed parameters
nInputUnits = size(trainInputSequence{1}, 2);
nOutputUnits = 1;

washout = 10;

%% MANUAL ESN TEST PART

if MANUAL_TEST
    
    nInternalUnits = 500;
    rho = 0.9;
    leaky_param = 0.5;
    
    % Ridge-regression
    lambda = 10;
    
    %RLS
    rls_delta = 100;
    rls_lambda = 0.9999995;
 
% Choose 'methodWeightCompute' to substitute into ESN initialization from
% commented parts below
%
%        'methodWeightCompute', 'pseudoinverse' ...
%
%         'methodWeightCompute', 'ridge_regression', ...
%         'ridge_parameter', lambda ...
% 
%     % ESN TEST
%     my_esn = ESN( nInputUnits, nInternalUnits, nOutputUnits, ...
%         'rho', rho, ...
%         'type', 'leaky_esn', ...
%         'leaky_parameter', leaky_param, ...
%         'methodWeightCompute', 'rls', ...
%         'rls_delta', rls_delta, ...
%         'rls_lambda', rls_lambda ...
%    );        
    
% Or test DropoutESN choosing input retaining probability
    p = 0.3;
    
    my_esn = DropoutESN ( nInputUnits, nInternalUnits, nOutputUnits, ...
        'rho', rho, ...
        'type', 'leaky_esn', ...
        'leaky_parameter', leaky_param , ...
        'methodWeightCompute', 'rls', ...
        'rls_lambda', rls_lambda, ...
        'rls_delta', rls_delta, ...
        'p', p ...
    );
    
    ls_tr = my_esn.train(trainInputSequence, trainTargetSequence, washout, type);
    tr_preds = my_esn.test(trainInputSequence, NaN, washout, type);
    
    if MOVEMENT_AAL
        % Squashing predictions to [-1,1] targets
        tr_preds = sign(tr_preds);
        tr_tgts = trainTargetSequence;
    else
        tr_tgts = compute_mutiple_series_targets(trainTargetSequence, washout);
        tr_tgts = cat(1, tr_tgts{:});
    end
    
    f = example('objective_function');
    tr_perf = f(tr_preds, tr_tgts);
   
    % Test
    ts_preds = my_esn.test(testInputSequence, NaN, washout, type);
    
    if MOVEMENT_AAL
        % Squashing predictions to [-1,1] targets
        ts_preds = sign(ts_preds);
        ts_tgts = testTargetSequence;
    else
        ts_tgts = compute_mutiple_series_targets(testTargetSequence, washout);
        ts_tgts = cat(1, ts_tgts{:});
    end
    
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
    nInternalUnits = [50, 100, 300, 500];
    leaky_parameter = [0.1, 0.2, 0.3, 0.5];
    rls_delta = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
    
    hyperparameters = containers.Map();
    hyperparameters('nInternalUnits') = nInternalUnits;
    hyperparameters('leaky_parameter') = leaky_parameter;
    hyperparameters('rls_delta') = rls_delta;
    
    %% MODEL SELECTION OF THE CLASSICAL ESN
    fprintf('Performing model selection... \n');
    
    model_selection_log_f = fopen('models/ESN.log','w');
    
    % Defining model type
    model_type = 'ESN';
    data = model_selection(model_type, fixed_params, hyperparameters, trainInputSequence, trainTargetSequence, washout, type, model_selection_log_f);
    
    trained_models = data('trained_models');
    best_model = trained_models(data('best_model_idx'));
    performance = data('best_avg_perf');
    
    fprintf(model_selection_log_f, '============ ESN Best model selected ============ \n');
    fprintf(model_selection_log_f, 'Hyperparameters: \n');
    fprintf(model_selection_log_f, ' - nInternalUnits: %d \n', best_model.nReservoirUnits);
    fprintf(model_selection_log_f, ' - leaky_parameter: %g \n', best_model.leaky_parameter);
    fprintf(model_selection_log_f, ' - rls_delta: %g \n', best_model.rls_delta);
    
    if MOVEMENT_AAL
        fprintf(model_selection_log_f, ' --> Expected performace (ACC): %g \n', performance);
    else
        fprintf(model_selection_log_f, ' --> Expected performace (MAE): %g \n', performance);
    end

    fprintf(model_selection_log_f, '================================================ \n');
    
    fclose(model_selection_log_f);
    
    %% MODEL SELECTION OF DROPOUT MODEL
    % Defining model type
    model_type = 'DropoutESN';
    
    p_param = [0.8, 0.5, 0.3];
    for i = 1:size(p_param,2)
        
        switch p_param(i)
            case 0.8
                filename = 'zero_eight.log';
            case 0.5
                filename = 'zero_five.log';
            case 0.3
                filename = 'zero_three.log';
                
            otherwise
                error('ERROR: performing model selection on DropoutESN, unknown p set!')
        end
                
        model_selection_log_f = fopen(strcat('models/',filename),'w');
        
        % And dropout params
        fixed_params('p') = p_param(i);
        
        data = model_selection(model_type, fixed_params, hyperparameters, trainInputSequence, trainTargetSequence, washout, type, model_selection_log_f);
        
        trained_models = data('trained_models');
        best_model = trained_models(data('best_model_idx'));
        performance = data('best_avg_perf');
        
        fprintf(model_selection_log_f, '============ DropoutESN Best model selected ============ \n');
        fprintf(model_selection_log_f, 'Percentage of input retaining: %g \n', p_param(i));
        fprintf(model_selection_log_f, 'Hyperparameters: \n');
        fprintf(model_selection_log_f, ' - nInternalUnits: %d \n', best_model.nReservoirUnits);
        fprintf(model_selection_log_f, ' - leaky_parameter: %g \n', best_model.leaky_parameter);
        fprintf(model_selection_log_f, ' - rls_delta: %g \n', best_model.rls_delta);
        

        if MOVEMENT_AAL
            fprintf(model_selection_log_f, ' --> Expected performace (ACC): %g \n', performance);
        else
            fprintf(model_selection_log_f, ' --> Expected performace (MAE): %g \n', performance);
        end

        fprintf(model_selection_log_f, '================================================ \n');
        
        fclose(model_selection_log_f);
        
    end

end

if TEST
    
    
    %% Loading models
    esn = load('models/esn.mat');
    desn_p_zero_eight = load('models/zero_eight.mat');
    desn_p_zero_five = load('models/zero_five.mat');
    desn_p_zero_three = load('models/zero_three.mat');
    
    models_to_test = {esn, desn_p_zero_eight, desn_p_zero_five, desn_p_zero_three};
    
    % Plain test
    models_stats = zeros(4,2);
    
    for i = 1:size(models_to_test,2)
        [results, best_model] = compute_model_statistics(models_to_test{i}, trainInputSequence, trainTargetSequence, testInputSequence, testTargetSequence, washout, type);
        
        models_stats(i, :) = results;
        best_models{i} = best_model;
    end
    
    % Saving plain test results
    save('results/models_stats', 'models_stats');
    
    %% Selecting best topologies for Dropout test    
    
    % Training every best_model on training data (TR+VL)
    for i = 1:size(best_models,2)
        best_model = best_models{i};
        best_model.train(trainInputSequence, trainTargetSequence, washout, type);
    end
    
    % First of all I want to test all models on test set (plain).
    best_models_plain_test = zeros(size(best_models));
    
    if MOVEMENT_AAL
       ts_tgts = testTargetSequence; 
    else
       ts_tgts = compute_mutiple_series_targets(testTargetSequence, washout);
       ts_tgts = cat(1,ts_tgts{:});
    end
    
    f = example('objective_function');
    for i=1:size(best_models,2)
        ts_preds = best_models{i}.test(testInputSequence, NaN, washout, type);
        if MOVEMENT_AAL
            best_models_plain_test(i) = f(sign(ts_preds), ts_tgts);
        else
            best_models_plain_test(i) = f(ts_preds, ts_tgts);
        end
    end
    
    save('results/best_models_plain_test', 'best_models_plain_test');
    
    % Then it's the funny stuff: start removing units from ts_input!
    best_models_dropping_test = {};
    for i=1:size(best_models,2)
        best_models_dropping_test{i,1} = test_drop_units_incr(best_models{i}, testInputSequence, testTargetSequence, washout, type);
    end
    
    save('results/best_models_dropping_test', 'best_models_dropping_test');
    
end

% Clearing PATH
if MOVEMENT_AAL
    rmpath 'Movement AAL'
else
    rmpath 'Kitchen'
end

fprintf('Bye bye :-) \n');

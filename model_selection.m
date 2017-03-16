function [data] = model_selection (model_type, fixedparams, hyperparams, inputSequence, targetSequence, washout, type, log_file )
    
    global example;
    ds = example('dataset');
    f = example('objective_function');
    objective = example('objective');
    
    MOVEMENT_AAL = 0;
    KITCHEN = 0;
    
    switch ds
        case 'Movement AAL'
            MOVEMENT_AAL = 1;
        case 'Kitchen'
            KITCHEN = 1;
        otherwise
                error('Unrecognized dataset!');
    end
    
    assert(MOVEMENT_AAL + KITCHEN == 1);

    DROPOUT = strcmp(model_type, 'DropoutESN');
    
    % Collecting classes
    classes = [];
    for i = 1:size(targetSequence,1)
        if MOVEMENT_AAL
            classes(i,1) = targetSequence(i);
        else
            classes(i,1) = targetSequence{i,1}(end);
        end
    end
   
    % perform K-fold CV ( k = 3 )
    K = 3;
    c = cvpartition(classes,'KFold',K);
    assert (c.NumTestSets == K);
    
    nInternalUnits = hyperparams('nInternalUnits');
    leaky = hyperparams('leaky_parameter');
    delta = hyperparams('rls_delta');
    
    nInit = 3;   
    
    switch objective
        case 'minimize'
            best_avg_perf = inf;
        case 'maximize'
            best_avg_perf = -inf;
        otherwise
                error('Unrecognized option!');
    end
    
    % Shouting model
    if DROPOUT
        fprintf(log_file, '\n========> MODEL: DROPOUT_ESN <======== \n\n');
    else
        fprintf(log_file, '\n========> MODEL: ESN <======== \n\n');
    end
    
    fprintf(log_file, '======== MODEL FIXED PARAMETERS ======== \n');
    fprintf(log_file, 'nInputUnits = %d \n', fixedparams('nInputUnits'));
    fprintf(log_file, 'nOutputUnits = %d \n', fixedparams('nOutputUnits'));
    fprintf(log_file, 'rho = %f \n', fixedparams('rho'));
    fprintf(log_file, 'methodWeightCompute = %s \n', fixedparams('methodWeightCompute'));
    fprintf(log_file, 'rls_lambda = %f \n', fixedparams('rls_lambda'));
    if DROPOUT
        fprintf(log_file, 'p = %f \n', fixedparams('p'));
    end
    
    fprintf(log_file, '======== MODEL HYPERPARAMETERS ======== \n');
    fprintf(log_file, '- nInternalUnits \n');
    fprintf(log_file, '- leaky_parameter \n');
    fprintf(log_file, '- rls_delta \n');
    fprintf(log_file, '======================================== \n');
    
    %% RESULT MATRICES
    TR_PERF = zeros(K, nInit * size(nInternalUnits,2) * size(leaky,2) * size(delta,2));
    VL_PERF = zeros(K, nInit * size(nInternalUnits,2) * size(leaky,2) * size(delta,2));
    % Creating models container (one container for each config)
    if DROPOUT
        MODELS = DropoutESN.empty();
    else
        MODELS = ESN.empty();
    end
    
    mc = 0;
    
    %% NESTED LOOPS        
    % Varying 'nInternalUnits'
    for n = 1:size(nInternalUnits,2)
        
        % Varying 'leaky_paramter'
        for a = 1:size(leaky,2)
            
            % Varying 'delta' (rls)
            for d = 1:size(delta,2)
            
                % for each fold
                fprintf(log_file, 'Testing hyperparameter config: \n');
                fprintf(log_file, 'nInternalUnits = %d \n', nInternalUnits(n));
                fprintf(log_file, 'leaky_parameter = %g \n', leaky(a));
                fprintf(log_file, 'rls_delta = %g \n', delta(d));

                % Stats for configuration
                current_tr_perf = zeros(K, nInit);
                current_vl_perf = zeros(K, nInit);
                if DROPOUT
                    current_models = DropoutESN.empty();
                else
                    current_models = ESN.empty();
                end

                % try 'nInit' different reservoir init.
                for k = 1: nInit

                    % generate model
                    if DROPOUT
                        my_esn = DropoutESN( ...
                            fixedparams('nInputUnits'), ...
                            nInternalUnits(n), ...
                            fixedparams('nOutputUnits'), ...
                            'rho', fixedparams('rho'), ...
                            'type', fixedparams('type'), ...
                            'leaky_parameter', leaky(a) , ...
                            'methodWeightCompute', fixedparams('methodWeightCompute'), ...
                            'rls_lambda', fixedparams('rls_lambda'), ...
                            'rls_delta', delta(d), ...
                            'p', fixedparams('p') ...
                        );
                    else
                        my_esn = ESN ( ...
                            fixedparams('nInputUnits'), ...
                            nInternalUnits(n), ...
                            fixedparams('nOutputUnits'), ...
                            'rho', fixedparams('rho'), ...
                            'type', fixedparams('type'), ...
                            'leaky_parameter', leaky(a) , ...
                            'methodWeightCompute', fixedparams('methodWeightCompute'), ...
                            'rls_lambda', fixedparams('rls_lambda'), ...
                            'rls_delta', delta(d) ...
                        );
                    end

                    orig_W_out = my_esn.W_out;

                    % for each fold of the Ks
                    for i = 1:K

                        % Re-initialize W_out
                        my_esn.W_out = orig_W_out;

                        % getting training and validation set
                        trIdxs = c.training(i);
                        tr_input = inputSequence(trIdxs, 1);
                        tr_target = targetSequence(trIdxs, 1);

                        vlIdxs = c.test(i);
                        vl_input = inputSequence(vlIdxs, 1);
                        vl_target = targetSequence(vlIdxs, 1);

                        % training on training set
                        my_esn.train( tr_input, tr_target, washout, type);
                        % evaluate on training set
                        tr_preds = my_esn.test( tr_input, NaN, washout, type);
                        
                        if MOVEMENT_AAL
                            tr_perf = f(tr_target, sign(tr_preds));
                        else
                            tr_tgts = compute_mutiple_series_targets(tr_target, washout);
                            tr_tgts = cat(1,tr_tgts{:});
                        
                            tr_perf = f(tr_preds, tr_tgts);
                        end

                        % evaluate on validation set
                        vl_preds = my_esn.test( vl_input, NaN, washout, type);
                        
                        if MOVEMENT_AAL
                            vl_perf = f(vl_target, sign(vl_preds));
                        else
                            vl_tgts = compute_mutiple_series_targets(vl_target, washout);
                            vl_tgts = cat(1,vl_tgts{:});
                        
                            vl_perf = f(vl_preds, vl_tgts);
                        end

                        % saving current results
                        current_tr_perf(i,k) = tr_perf;
                        current_vl_perf(i,k) = vl_perf;

                    end

                    % and trained model (resetting Wout, needs retraining)
                    my_esn.W_out = orig_W_out;
                    current_models(1,k) = my_esn;

                end

                % Computing hyperparams configuration AVG performance
                avg_perf = mean(current_vl_perf);
                avg_avg_perf = mean(avg_perf);
                
                % Selecting min or max depending on objective
                switch objective
                    case 'minimize'
                        [current_best_avg_perf, best_idx] = min(avg_perf);
                        is_better = @ (a,b) a < b;
                    case 'maximize'
                        [current_best_avg_perf, best_idx] = max(avg_perf);
                        is_better = @ (a,b) a > b;
                    otherwise
                            error('Unknown objective!');
                end                
                
                if MOVEMENT_AAL
                    fprintf(log_file, '\nBest AVG_VL_ACC for config: %f obtained with model n. %d \n', current_best_avg_perf, best_idx);
                end
                if KITCHEN
                    fprintf(log_file, '\nBest AVG_VL_MAE for config: %f obtained with model n. %d \n', current_best_avg_perf, best_idx);
                end

                if is_better(current_best_avg_perf,best_avg_perf)

                    best_avg_perf = current_best_avg_perf;
                    
                    fprintf(log_file, '\n======= UPDATING BEST MODEL! =======\n');
                    if MOVEMENT_AAL
                        fprintf(log_file, 'Expected ACC in: [%f,%f] \n',current_best_avg_perf, avg_avg_perf);
                    end
                    if KITCHEN
                        fprintf(log_file, 'Expected MAE in: [%f,%f] \n',current_best_avg_perf, avg_avg_perf);
                    end
                    fprintf(log_file, '====================================\n\n');

                    if DROPOUT
                        switch my_esn.p
                            case 0.8
                                filename = 'zero_eight';
                            case 0.5
                                filename = 'zero_five';
                            case 0.3
                                filename = 'zero_three';

                            otherwise
                                error('ERROR:I do not know why but p is not set!')
                        end
                    else
                        filename = 'esn';
                    end
                    
                    % Creating a dictionary containing the best model and
                    % its results
                    data = containers.Map();
                    data('best_model_idx') = best_idx;
                    data('best_avg_perf') = best_avg_perf;
                    data('trained_models') = current_models;
                    data('training_data') = current_tr_perf;
                    data('validation_data') = current_vl_perf;
                    
                    % Saving best model
                    save(strcat('models/',filename) , 'data');
                end

                fprintf(log_file, '-------------------------------------\n');


                % Saving config stats
                TR_PERF(:,mc*nInit+1:mc*nInit+k) = current_tr_perf;
                VL_PERF(:,mc*nInit+1:mc*nInit+k) = current_vl_perf;
                MODELS(1, mc*nInit+1:mc*nInit+k) = current_models;

                mc = mc + 1;
                
            end
        end
    end
    
    %% Saving general stats about model selection
    model_selection = containers.Map();
    % Describing the task
    if MOVEMENT_AAL
        model_selection('objective function') = 'ACC';
    end
    if KITCHEN
        model_selection('objective function') = 'MAE';
    end
    model_selection('objective') = objective;
    % And the stats obtained
    model_selection('TR_PERF') = TR_PERF;
    model_selection('VL_PERF') = VL_PERF;
    model_selection('MODELS') = MODELS;
    save('models/model_selection_recap', 'model_selection');
end


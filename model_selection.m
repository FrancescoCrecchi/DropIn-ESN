function [best_model, best_avg_mae] = model_selection (model_type, fixedparams, hyperparams, inputSequence, targetSequence, log_file )
    
    %% ACHTUNG! FIXED WASHOUT PARAMETER!
    washout = 10;

    DROPOUT = strcmp(model_type, 'DropoutESN');
    
    % Collecting classes
    classes = [];
    for i = 1:size(targetSequence,1)
        classes(i,1) = targetSequence{i,1}(end);
    end
   
    % perform K-fold CV ( k = 3 )
    K = 3;
    c = cvpartition(classes,'KFold',K);
    assert (c.NumTestSets == K);
    
    nInternalUnits = hyperparams('nInternalUnits');
    leaky = hyperparams('leaky_parameter');
    delta = hyperparams('rls_delta');
    
    nInit = 3;
    best_model = NaN;
    best_avg_mae = inf;
    
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
        fprintf(log_file, 'dropout_type = %s \n', fixedparams('dropout_type'));
        fprintf(log_file, 'p = %f \n', fixedparams('p'));
    end
    
    fprintf(log_file, '======== MODEL HYPERPARAMETERS ======== \n');
    fprintf(log_file, '- nInternalUnits \n');
    fprintf(log_file, '- leaky_parameter \n');
    fprintf(log_file, '- rls_delta \n');
    fprintf(log_file, '======================================== \n');
    
    %% RESULT MATRICES
    TR_MAE = zeros(K, nInit * size(nInternalUnits,2) * size(leaky,2) * size(delta,2));
    VL_MAE = zeros(K, nInit * size(nInternalUnits,2) * size(leaky,2) * size(delta,2));
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
                current_tr_mae = zeros(K, nInit);
                current_vl_mae = zeros(K, nInit);
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
                            'dropout_type', fixedparams('dropout_type'), ...
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
                        my_esn.train( tr_input, tr_target, washout);
                        % evaluate on training set
                        tr_preds = my_esn.test( tr_input, NaN, washout);
                        
                        tr_tgts = compute_mutiple_series_targets(tr_target, washout);
                        tr_tgts = cat(1,tr_tgts{:});
                        
                        tr_mae = compute_MAE(tr_preds, tr_tgts);

                        % evaluate on validation set
                        vl_preds = my_esn.test( vl_input, NaN, washout);
                        
                        vl_tgts = compute_mutiple_series_targets(vl_target, washout);
                        vl_tgts = cat(1,vl_tgts{:});
                        
                        vl_mae = compute_MAE(vl_preds, vl_tgts);

                        % saving current results
                        current_tr_mae(i,k) = tr_mae;
                        current_vl_mae(i,k) = vl_mae;

                    end

                    % and trained model (resetting Wout, needs retraining)
                    my_esn.W_out = orig_W_out;
                    current_models(1,k) = my_esn;

                end

                % Computing hyperparams configuration AVG performance
                avg_mae = mean(current_vl_mae);
                avg_avg_mae = mean(avg_mae);
                [min_avg_mae, min_idx] = min(avg_mae);
                fprintf(log_file, '\nBest AVG_VL_MAE for config: %f obtained with model n. %d \n', min_avg_mae, min_idx);

                if min_avg_mae < best_avg_mae

                    best_avg_mae = min_avg_mae;

                    fprintf(log_file, '\n======= UPDATING BEST MODEL! =======\n');
                    fprintf(log_file, 'Expected MAE in: [%f,%f] \n',min_avg_mae, avg_avg_mae);
                    fprintf(log_file, '====================================\n\n');

                    % Creating a dictionary containing the best model and
                    % its results
                    best_model = containers.Map();
                    best_model('best_model_idx') = min_idx;
                    best_model('trained_models') = current_models;
                    best_model('training_data') = current_tr_mae;
                    best_model('validation_data') = current_vl_mae;

                    if DROPOUT
                        switch my_esn.p
                            case 0.8
                                filename = 'zero_otto';
                            case 0.5
                                filename = 'zero_cinque';
                            case 0.3
                                filename = 'zero_tre';

                            otherwise
                                error('ERROR:I do not know why but p is not set!')
                        end
                    else
                        filename = 'esn';
                    end

                    % Saving best model
                    save(strcat('models/',filename) , 'best_model');
                end

                fprintf(log_file, '-------------------------------------\n');


                % Saving config stats
                TR_MAE(:,mc*nInit+1:mc*nInit+k) = current_tr_mae;
                VL_MAE(:,mc*nInit+1:mc*nInit+k) = current_vl_mae;
                MODELS(1, mc*nInit+1:mc*nInit+k) = current_models;

                mc = mc + 1;
                
            end
        end
    end
    
    %% Saving general stats about model selection
    model_selection = containers.Map();
    model_selection('TR_MAE') = TR_MAE;
    model_selection('VL_MAE') = VL_MAE;
    model_selection('MODELS') = MODELS;
    save('models/model_selection_recap', 'model_selection');
end


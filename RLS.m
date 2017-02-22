function [best_Wout, last_state] = RLS( esn, trainInputs, trainTargets, washout)
    % RLS ALGORITHM WITH EARLY STOPPING 
    global example;
    ds = example('dataset');
    f = example('objective_function');
    objective = example('objective');
    
    nTimeSeries = size(trainInputs, 1);        
    
    DEBUG = 0;
    
    % Collecting classes
    classes = [];
    for i = 1:size(trainTargets,1)
        classes(i,1) = trainTargets{i,1}(end);
    end
    
    %% Starting splitting data into train and validation data
    c = cvpartition(classes, 'HoldOut', 0.2);
    trainInputSequence = trainInputs(c.training);
    trainTargetSequence = trainTargets(c.training);

    validationInputSequence = trainInputs(c.test);
    validationTargetSequence = trainTargets(c.test);
    
    %% Computing validation state matrix and targets (NO CHANGES!)
    
    X_vl = compute_multiple_series_state_matrix(esn, validationInputSequence, washout, NaN, 'test');
    X_vl = cat(2,X_vl{:});
    
    y_vl = compute_mutiple_series_targets(validationTargetSequence, washout);
    y_vl = cat(1,y_vl{:});
    
    
    %% Early stopping method
    max_epochs = 100;
    max_patience = 10;           % look as this many epochs regardless
    min_vl_improvement = 5e-2;
    done = false;
    
    epoch = 0;
    
    switch objective
        case 'minimize'
            best_validation_perf = inf;
        case 'maximize'
            best_validation_perf = -inf;
        otherwise
                error('Unrecognized option!');
    end
    
    best_Wout = NaN;
    w = zeros(size(esn.W_out));
    patience = max_patience;
    
    tr_nTimeSeries = size(trainInputSequence,1);
    
    errors = zeros();
    weights = zeros();
    
    SInverse = (1 / esn.rls_delta) * eye(esn.nInputUnits + esn.nReservoirUnits);
    
    while epoch < max_epochs && ~done
        
        % SHUFFLING OF TRAINING DATA!
        perm = randperm(size(trainInputSequence,1));
        trainInputSequence = trainInputSequence(perm, :);
        trainTargetSequence = trainTargetSequence(perm, :);
        
        %% FOREACH SEQUENCE 
        for j=1:tr_nTimeSeries
            
            % Computing total number of timeseries steps
            nSteps = size(trainInputSequence{j,1}, 1) - washout;
            
            %% Sequence state matrix
            X = zeros( esn.nInputUnits + esn.nReservoirUnits, nSteps);
            
            curr_state = X(:,1);
            
            %% FOREACH TIMESTEP
            for jj = 1:size(trainInputSequence{j,1}, 1)
                
                sample = trainInputSequence{j}(jj, :);
                
                curr_state = esn.compute_statematrix( sample, curr_state, 'train');
                
                if jj < washout
                    continue;
                end
                
                % Collecting states into X
                X(:, jj-washout+1) = curr_state;
                
                netOut = w * curr_state;
                
                state = curr_state;
                phi = state' * SInverse;
                k = phi'/(esn.rls_lambda + phi * state);
                
                e = trainTargetSequence{j}(jj, 1) - netOut(1);
                % collect the error that will be plotted
                errors(epoch*nTimeSeries+j+jj-1, 1 ) = e*e ;
                
                % update the weights
                w = w + (k*e)' ;
                
                % collect the weights for plotting
                weights(epoch*nTimeSeries+j+jj-1, 1) = sum(abs(w(1,:))) ;
                
                SInverse = ( SInverse - k * phi ) / esn.rls_lambda ;
                
            end
            
            last_state = X(:, end);
        end
        
        %% END OF EPOCH HERE! => CHECK EARLY STOPPING CONDITION!
        
        % Computing MAE on the validation set
        vl_preds = (w * X_vl)';  
        val_perf = f(vl_preds, y_vl);
        
        if DEBUG
            X_tr = compute_multiple_series_state_matrix(esn, trainInputSequence, washout, NaN, 'test');
            X_tr = cat(2, X_tr{:});
            
            tr_preds = (w * X_tr)';  
            
            y_tr = compute_mutiple_series_targets(trainTargetSequence, washout);
            y_tr = cat(1, y_tr{:});
            
            TR_PERF(epoch+1,1) = f(tr_preds, y_tr);
            VL_PERF(epoch+1,1) = val_perf; 
        end

        if val_perf < best_validation_perf
            if abs(val_perf - best_validation_perf) >= min_vl_improvement
                best_Wout = w;
                % Reset patience
                patience = max_patience;
                best_validation_perf = val_perf;
            end
        else
            patience = patience - 1;
            % Check patience
            if patience == 0
                done = true;
            end
        end
        
        epoch = epoch+1;
        
    end
    
    if DEBUG
        figure(1);
        plot(TR_PERF);
        hold on;
        plot(VL_PERF);
        if strcmp(ds, 'Movement AAL')
            legend('TR ACC', 'VL ACC');
        else
            legend('TR MAE', 'VL MAE');
        end
        title('RLS - Performance plot');
        hold off;
    end
    
end


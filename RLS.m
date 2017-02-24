function [best_Wout, last_state] = RLS( esn, trainInputs, trainTargets, washout, type)
    % RLS ALGORITHM WITH EARLY STOPPING 
    global example;
    ds = example('dataset');
    f = example('objective_function');
    objective = example('objective');
    
    nTimeSeries = size(trainInputs, 1);        
    
    DEBUG = 0;
    
    % Collecting classes
    if strcmp(type, 'seq2elem')
        classes = trainTargets;
    else       
        for i = 1:size(trainTargets,1)
            classes(i,1) = trainTargets{i,1}(end);
        end
    end
    
    %% Starting splitting data into train and validation data
    c = cvpartition(classes, 'HoldOut', 0.2);
    trainInputSequence = trainInputs(c.training);
    trainTargetSequence = trainTargets(c.training);

    validationInputSequence = trainInputs(c.test);
    validationTargetSequence = trainTargets(c.test);
    
    %% Computing validation state matrix and targets
    
    X_vl = compute_multiple_series_state_matrix(esn, validationInputSequence, washout, NaN, 'test');
    switch type
        case 'seq2seq'
            % Concatenating states in a BIG matrix
            X_vl = cat(2,X_vl{:});
            % Same for targets (taking into account initial transient)
            y_vl = compute_mutiple_series_targets(validationTargetSequence, washout);
            y_vl = cat(1,y_vl{:});
            
        case 'seq2elem'
            % Root state mapping: selecting last state as
            % representative for the entire sequence
            foo = cellfun(@(x) x(:,end), X_vl, 'UniformOutput', 0);
            X_vl = cat(2,foo{:});
            y_vl = validationTargetSequence;
        otherwise
            error('Unrecognized type!');
    end
    
    
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
        
        switch type
            case 'seq2elem'
                
                % Sequence state matrix
                X = zeros( esn.nInputUnits + esn.nReservoirUnits, size(trainInputSequence,1));
                
                %% FOREACH SEQUENCE
                for j=1:tr_nTimeSeries
                    
                    curr_state = X(:,1);
                    sample = trainInputs{j,:};
                    
                    X_k = esn.compute_statematrix( sample, curr_state, 'train');
                    % keep only last state (sequence2element)
                    curr_state = X_k(:,end);
                    
                    X(:, j) = curr_state;
                    
                    state = curr_state;
                    phi = state' * SInverse;
                    k = phi'/(esn.rls_lambda + phi * state);
                    
                    netOut = w * curr_state;
                    
                    e = trainTargets(j, 1) - netOut(1);
                    % collect the error that will be plotted
                    errors(epoch*nTimeSeries+j, 1 ) = e*e ;
                    
                    % update the weights
                    w = w + (k*e)' ;
                    % collect the weights for plotting
                    weights(epoch*nTimeSeries+j, 1) = sum(abs(w(1,:))) ;
                    
                    SInverse = ( SInverse - k * phi ) / esn.rls_lambda ;
                    
                end
                
            case 'seq2seq'
                
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
                    
                end
                
            otherwise
                error('Unrecognized type option!');
        end
            
        last_state = X(:, end);
        
        %% END OF EPOCH HERE! => CHECK EARLY STOPPING CONDITION!
        
        switch ds
            case 'Movement AAL'
                
                % Computing performance on validation set
                vl_preds = (w * X_vl)';
                val_perf = f(y_vl,sign(vl_preds));
                
                if DEBUG
                    X_tr = compute_multiple_series_state_matrix(esn, trainInputSequence, washout, NaN, 'test');
                    foo = cellfun(@(x) x(:,end), X_tr, 'UniformOutput', 0);
                    X_tr = cat(2,foo{:});
                    
                    tr_preds = (w * X_tr)';
                    
                    y_tr = trainTargetSequence;
                    
                    TR_PERF(epoch+1,1) = f(y_tr, sign(tr_preds));
                    VL_PERF(epoch+1,1) = val_perf;
                end
                
                 if val_perf > best_validation_perf
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
                
            case 'Kitchen'
                
                % Computing performance on validation set
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
                
            otherwise
                    error('Unrecognized dataset!');
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


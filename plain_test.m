function [ tr_avg_perf, ts_avg_perf ] = plain_test( models, trInputs,trTargets, tsInputs, tsTargets, washout, type)
    
    global example;
    ds = example('dataset');
    f = example('objective_function');
    
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

    tr_perf = zeros(1,size(models,2));
    ts_perf = zeros(1,size(models,2));
    
    for i = 1:size(models,2)
        
        model = models(1,i);
        
        % Training 
        model.train( trInputs, trTargets, washout, type);
        % evaluate on training set
        tr_preds = model.test(trInputs, NaN, washout, type);
        
        if MOVEMENT_AAL
            tr_perf = f(sign(tr_preds), trTargets);
        else
            tr_tgts = compute_mutiple_series_targets(trTargets, washout);
            tr_tgts = cat(1,tr_tgts{:});
            
            tr_perf(1,i) = f(tr_preds, tr_tgts);
        end

        % Evaluate on test set
        ts_preds = model.test(tsInputs, NaN, washout, type);
        
        if MOVEMENT_AAL
            ts_perf = f(sign(ts_preds), tsTargets);
        else
            ts_tgts = compute_mutiple_series_targets(tsTargets, washout);
            ts_tgts = cat(1,ts_tgts{:});
            
            ts_perf(1,i) = f(ts_preds, ts_tgts);
        end
        
    end
    
    tr_avg_perf = mean(tr_perf);
    ts_avg_perf = mean(ts_perf);

end


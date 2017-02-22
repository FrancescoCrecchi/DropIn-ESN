function [ tr_avg_perf, ts_avg_perf ] = plain_test( models, trInputs,trTargets, tsInputs, tsTargets, washout )
    
    global example;
    f = example('objective_function');

    tr_perf = zeros(1,size(models,2));
    ts_perf = zeros(1,size(models,2));
    
    for i = 1:size(models,2)
        
        model = models(1,i);
        
        % Training 
        model.train( trInputs, trTargets, washout);
        % evaluate on training set
        tr_preds = model.test(trInputs, NaN, washout);
        tr_tgts = compute_mutiple_series_targets(trTargets, washout);
        tr_tgts = cat(1,tr_tgts{:});
        
        tr_perf(1,i) = f(tr_preds, tr_tgts);

        % Evaluate on test set
        ts_preds = model.test(tsInputs, NaN, washout);
        ts_tgts = compute_mutiple_series_targets(tsTargets, washout);
        ts_tgts = cat(1,ts_tgts{:});
        
        ts_perf(1,i) = f(ts_preds, ts_tgts);
    end
    
    tr_avg_perf = mean(tr_perf);
    ts_avg_perf = mean(ts_perf);

end


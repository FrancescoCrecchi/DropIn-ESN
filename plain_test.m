function [ tr_avg_mae, ts_avg_mae ] = plain_test( models, trInputs,trTargets, tsInputs, tsTargets, washout )
    
    tr_mae = zeros(1,size(models,2));
    ts_mae = zeros(1,size(models,2));
    
    for i = 1:size(models,2)
        
        model = models(1,i);
        
        % Training 
        model.train( trInputs, trTargets, washout);
        % evaluate on training set
        tr_preds = model.test(trInputs, NaN, washout);
        tr_tgts = compute_mutiple_series_targets(trTargets, washout);
        tr_tgts = cat(1,tr_tgts{:});
        
        tr_mae(1,i) = compute_MAE(tr_preds, tr_tgts);

        % Evaluate on test set
        ts_preds = model.test(tsInputs, NaN, washout);
        ts_tgts = compute_mutiple_series_targets(tsTargets, washout);
        ts_tgts = cat(1,ts_tgts{:});
        ts_mae(1,i) = compute_MAE(ts_preds, ts_tgts);
    end
    
    tr_avg_mae = mean(tr_mae);
    ts_avg_mae = mean(ts_mae);

end


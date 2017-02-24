function [ results, best_model ] = evaluate_model_on_plain_test( model, trainInputSequence, trainTargetSequence, testInputSequence, testTargetSequence, washout, type )
    
    trained_models = model.data('trained_models');
    best_model = trained_models(model.data('best_model_idx'));
    
    % Executing plain test
    [tr_avg_perf, ts_avg_perf] = plain_test(trained_models, trainInputSequence, trainTargetSequence, testInputSequence, testTargetSequence, washout, type);
    
    results = [tr_avg_perf, ts_avg_perf];

end


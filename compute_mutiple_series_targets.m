function [ Y ] = compute_mutiple_series_targets( targetSequences, washout )
    
    nTimeSeries = size(targetSequences,1);
    
    Y = cell(nTimeSeries, 1);
    for i = 1:nTimeSeries
        Y{i, 1} = targetSequences{i}(washout+1:end,1);
    end

end


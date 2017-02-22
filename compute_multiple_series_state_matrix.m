function [ X ] = compute_multiple_series_state_matrix(esn, inputSequences, washout, state, task )

    nTimeSeries = size(inputSequences,1);
     
    if strcmp(task, 'training')
        TRAINING = 1;
    else
        TRAINING = 0;
    end
    
    X = cell(nTimeSeries,1);
    
    for i = 1:nTimeSeries
        
        % reservoir states matrix
        X_i = zeros(esn.nInputUnits + esn.nReservoirUnits, size(inputSequences{i},1));
        
        if i == 1 && ~isnan(state)
            X_i(:,1) = state;
        end
        
        curr_state = X_i(:,1);
        sample = inputSequences{i};
        
        if TRAINING
            X_i = esn.compute_statematrix(sample, curr_state, 'training');
        else
            X_i = esn.compute_statematrix(sample, curr_state, 'test');
        end
        
        % saving result in X state matrix
        X{i} = X_i(:,washout+1:end);
        
        %cleaning
        clear X_k
        
    end

end


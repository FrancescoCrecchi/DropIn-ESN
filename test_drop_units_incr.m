function [ results ] = test_drop_units_incr( model, inputSequences, targetSequences, washout)
    
    global example;
    f = example('objective_function');
    ds = example('dataset');

    n_inputs = model.nInputUnits - 1;

    % Allocating results matrix
    results = zeros(n_inputs-2, n_inputs-1);
    
    % in Kitchen task avoid dropping X feature.
    k = 0;
    if strcmp(ds, 'Kitchen')
        k = 1;
    end
            
    for i = 1:(n_inputs-(k+1)) % Let at least one unit as input...
        % Number of input units to drop
        n_drop = i;

        % Dropping units in a RR fashion
        for j = 0:n_inputs-2
            
            missing_values_input = inputSequences;
            
            % Constructing dropping mask
            mask = ones(1, n_inputs);
            for d = 0:n_drop-1
                mask(mod(j+d, n_inputs-1)+1) = 0;
            end

            % Masking inputs
            for k = 1:size(missing_values_input, 1)
                for kk = 1:size(missing_values_input{k, 1})
                    missing_values_input{k}(kk,:) = missing_values_input{k}(kk,:) .* mask;
                end
            end

            % Testing model on this test input configuration
            predictedSequences = model.test(missing_values_input, NaN, washout);
            
            tgtSequences = compute_mutiple_series_targets(targetSequences, washout);
            tgtSequences = cat(1,tgtSequences{:});
            perf = f(predictedSequences, tgtSequences);
            
            % Updating results
            results(i,j+1) = perf;
        end
    end

end


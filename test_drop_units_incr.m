function [ results ] = test_drop_units_incr( model, inputSequences, targetSequences, washout, type)
    
    global example;
    f = example('objective_function');
    ds = example('dataset');

    n_inputs = model.nInputUnits - 1;
	

    % in Kitchen task avoid dropping X feature.
    z = 0;
    if strcmp(ds, 'Kitchen')
        z = 1;
    end

    % Allocating results matrix
    results = zeros(n_inputs-(z+1), n_inputs-z);

            
    for i = 1:(n_inputs-(z+1)) % Let at least one unit as input...
        % Number of input units to drop
        n_drop = i;

        % Dropping units in a RR fashion
        for j = 0:n_inputs-(z+1)
            
            missing_values_input = inputSequences;
            
            % Constructing dropping mask
            mask = ones(1, n_inputs);
            for d = 0:n_drop-1
                mask(mod(j+d, n_inputs-z)+1) = 0;
            end

            % Masking inputs
            for k = 1:size(missing_values_input, 1)
                for kk = 1:size(missing_values_input{k, 1})
                    missing_values_input{k}(kk,:) = missing_values_input{k}(kk,:) .* mask;
                end
            end

            % Testing model on this test input configuration
            predictedSequences = model.test(missing_values_input, NaN, washout, type);
            
            switch ds
                case 'Movement AAL'
                    perf = f(targetSequences, sign(predictedSequences));
                case 'Kitchen'
                    tgtSequences = compute_mutiple_series_targets(targetSequences, washout);
                    tgtSequences = cat(1,tgtSequences{:});
                    perf = f(tgtSequences, predictedSequences);
                otherwise
                    error('Unrecognized dataset!');
            end
            
            % Updating results
            results(i,j+1) = perf;
        end
    end

end


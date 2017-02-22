function [ trainInputSequence, trainOutputSequence, testInputSequence, testOutputSequence ] = preprocessor()
    
    input = {};
    target = {};
    classes = [];
    
    for i = 1:80
        % Importing data and targets
        [LIGHT1, PIR1, TEMP1, HUMID1, LIGHT2, PIR2, TEMP2, HUMID2, LIGHT3, PIR3, TEMP3, HUMID3, LIGHT4, PIR4, TEMP4, HUMID4, LIGHT5, PIR5, TEMP5, HUMID5, LIGHT6, PIR6, TEMP6, HUMID6, X, Y, THETA, TARGET] = signals_importer(['./Kitchen/data/kitchen', int2str(i), '.csv']);
        
        % ICF CONFIG
        input{i, 1} = [PIR1, PIR2, PIR3, PIR4, PIR5, X];        
        target{i, 1} = TARGET;
        classes(i, 1) = TARGET(end,1);
        
    end    
    
    %% Partitioning data (Hold out)
    c = cvpartition(classes, 'HoldOut', 0.2);
        
    trainInputSequence = input(c.training);
    trainOutputSequence = target(c.training);
    
    testInputSequence = input(c.test);
    testOutputSequence = target(c.test);
    
    %% Normalize training data mean0std1
    [trainInputSequence, tr_means, tr_vars] = normalizeData0Mean1Var(trainInputSequence, NaN, NaN);
    
    %% Normalize test data using training data means, vars
    [testInputSequence, ts_means, ts_vars] = normalizeData0Mean1Var(testInputSequence, tr_means, tr_vars);
    assert(~(sum(tr_means(:) ~= ts_means(:)) & sum(tr_vars(:) ~= ts_vars(:))));
    
end
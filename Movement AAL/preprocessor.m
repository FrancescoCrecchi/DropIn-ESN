function [ trainInputSequence, trainOutputSequence, testInputSequence, testOutputSequence ] = preprocessor( test_type )
    
    t = target_importer('./Movement AAL/data/MovementAAL_target.csv');
    
    input = {};
    target = [];
    
    for i = 1:210
        [RSS_anchor1,RSS_anchor2,RSS_anchor3,RSS_anchor4] = signals_importer(['./Movement AAL/data/MovementAAL_RSS_', int2str(i), '.csv']);

        x1 = RSS_anchor1;
        x2 = RSS_anchor2;
        x3 = RSS_anchor3;
        x4 = RSS_anchor4;
        
        assert( (sum(isnan(x1(:))) + sum(isinf(x1(:)))) + (sum(isnan(x2(:))) + sum(isinf(x2(:)))) + ...
            (sum(isnan(x3(:))) + sum(isinf(x3(:)))) + (sum(isnan(x4(:))) + sum(isinf(x4(:)))) == 0 );
        
        switch test_type
            case 'local'
                input{i,1} = x1;
            case 'nonlocal2'
                input{i,1} = [x1, x2];
            case 'nonlocal3'
                input{i,1} = [x1, x2, x3];
            case 'nonlocal4'
                input{i,1} = [x1, x2, x3, x4];
            otherwise
                error('Unrecognized preprocessor option!');
        end 
                
        target(i,1) = t(i);
        
    end
    
    c = cvpartition(target(:,1), 'HoldOut', 0.2);
        
    trainInputSequence = input(c.training);
    trainOutputSequence = target(c.training);
    
    testInputSequence = input(c.test);
    testOutputSequence = target(c.test);

end
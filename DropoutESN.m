classdef DropoutESN < ESN
    
    properties
        DropoutIn = NaN;
        DropoutWin = NaN;
        p = NaN;
        dropout_type = NaN;
    end
    
    methods
        function obj = DropoutESN(nInputUnits, nInternalUnits, nOutputUnits, varargin)
            
            % Calling superclass constructor
            obj@ESN(nInputUnits, nInternalUnits, nOutputUnits, varargin{1:end-4});
            
            % Adding a DropoutWrapper onto obj.W_in
            args = varargin;
            nargs= length(args);
            for i=1:2:nargs
                switch args{i}
                    case 'methodWeightCompute'
                        if strcmp(args{i+1}, 'ridge_regression') || strcmp(args{i+1}, 'pseudoinverse')
                            error('ERROR: DropoutESN needs an online method!');
                        end
                    case 'p'
                        obj.p = args{i+1};
                    case 'dropout_type'
                        obj.dropout_type = args{i+1};
                        if ~(strcmp(obj.dropout_type, 'dropout') || strcmp(obj.dropout_type, 'dropout_connect'))
                            error('Wrong type of DropoutESN set!');
                        end
                end
            end
            
            % Check correctness conditions
            if isnan(obj.p) || (~isnan(obj.p) && sum(isnan(obj.dropout_type)))
                error('Make sure you set probability (p) and type of DropoutESN!');
            end
            
            if strcmp(obj.dropout_type, 'dropout')
                obj.DropoutIn = DropoutWrapper(obj.p);
            else % dropout_connect cases
                % Adding DropoutWrapper onto W_in
                obj.DropoutWin = DropoutConnect(obj.W_in, obj.p);
            end
            
        end
        

        function X_i = compute_statematrix( obj, inputSequence, curr_state, task)
            
            nDataPoints = length(inputSequence(:,1));
            
            % current inputsequence state matrix
            X_i = zeros(obj.nInputUnits+obj.nReservoirUnits, nDataPoints);
            
            WIN = obj.W_in;
            mask = NaN;
            
            % DropoutIn?
            if strcmp(obj.dropout_type, 'dropout')
                if strcmp(task, 'train')
                    % Avoid dropping bias
                    foo = obj.DropoutIn.train(ones(1, obj.nInputUnits-1));
                    mask = cat(2, foo, 1);
%                 else
%                     WIN = obj.DropoutIn.test(WIN);
                end
            end
            
            % DropoutWin?
            if strcmp(obj.dropout_type, 'dropout_connect')
                if strcmp(task, 'train')
                    WIN = obj.DropoutWin.train;
                else % test
                    WIN = obj.DropoutWin.test;
                end
            end
                        
            for i = 1:nDataPoints
                
                in = [1 inputSequence(i,:)]';
                
                if ~isnan(mask) % DropoutIn training task
                    in = in .* mask';
                end
                
                x = curr_state(obj.nInputUnits+1:end, :);
                
                switch obj.type
                    case 'plain_esn', x = tanh (WIN * in + obj.W_hat * x);
                    case 'leaky_esn', ...
                            x = (1 - obj.leaky_parameter) * x + ...
                            obj.leaky_parameter * tanh (WIN * in  + obj.W_hat * x);
                        
                    otherwise
                        error('Unrecognised esn.type!');
                end
                
                X_i(:,i) = [in; x];
                curr_state = X_i(:,i);
                
            end
            
        end
        
        
        %% TRAIN
        function last_state = train (obj, trainInputs, trainTargets, washout, type)
           
            assert(size(trainInputs,1) == size(trainTargets,1));
           
            [obj.W_out, last_state] = RLS(obj, trainInputs, trainTargets, washout, type);
            
            % set trained flag 
            obj.trained = 1;
            
        end
        
       %% TEST
        function testPred = test (obj, testInputs, state, washout, type)
            testPred = test@ESN(obj, testInputs, state, washout, type);
        end
        
    end
        
end
    

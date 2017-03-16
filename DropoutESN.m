classdef DropoutESN < ESN
    
    properties
        DropoutIn = NaN;
        p = NaN;
    end
    
    methods
        function obj = DropoutESN(nInputUnits, nInternalUnits, nOutputUnits, varargin)
            
            % Calling superclass constructor
            obj@ESN(nInputUnits, nInternalUnits, nOutputUnits, varargin{1:end-2});
            
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
                end
            end
            
            % Check correctness conditions
            if isnan(obj.p)
                error('Make sure you set probability (p) and type of DropoutESN!');
            end
            
            obj.DropoutIn = DropoutWrapper(obj.p);
            
        end
        

        function X_i = compute_statematrix( obj, inputSequence, curr_state, task)
            
            nDataPoints = length(inputSequence(:,1));
            
            % current inputsequence state matrix
            X_i = zeros(obj.nInputUnits+obj.nReservoirUnits, nDataPoints);
            
            WIN = obj.W_in;
            mask = NaN;
            
            % Drop-In
            if strcmp(task, 'train')
                % Avoid dropping bias
                foo = obj.DropoutIn.train(ones(1, obj.nInputUnits-1));
                mask = cat(2, foo, 1);
                
% Although you can uncomment the part below to test the averaging factor multiplication of 
% input matrix (as suggested in any Dropout regularization approach) the
% autors do not reccomend this because it leads to worse performances since
% the reservoir itself takes into account for the model averaging factor.
% 
%                 else
%                     WIN = obj.DropoutIn.test(WIN);

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
    

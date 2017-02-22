classdef ESN < handle

    properties
        
        nInputUnits
        nReservoirUnits
        nOutputUnits
        
        W_in
        W_hat
        W_out
        
        rho
        leaky_parameter
        
        ridge_parameter
        type
        methodWeightCompute
        
        rls_delta
        rls_lambda
        
        trained
        
    end
    
    methods
        function obj = ESN(nInputUnits, nInternalUnits, nOutputUnits, varargin)
            
            addpath('../.');
            
            % set the number of units
            obj.nInputUnits = nInputUnits + 1; % +1 for the bias unit
            obj.nReservoirUnits = nInternalUnits; 
            obj.nOutputUnits = nOutputUnits;
            
            % esn parameters
            obj.rho = 0.9;
            obj.type = 'plain_esn';
            obj.methodWeightCompute = 'pseudoinverse';
            obj.leaky_parameter = NaN;
            obj.ridge_parameter = NaN;
            obj.rls_lambda = NaN;
            obj.rls_delta = NaN;

            args = varargin; 
            nargs= length(args);
            for i=1:2:nargs
                switch args{i}
                    case 'rho'
                        obj.rho = args{i+1} ;
                    case 'type'
                        obj.type = args{i+1} ;
                    case 'leaky_parameter'
                        obj.leaky_parameter = args{i+1};
                    case 'methodWeightCompute'
                        obj.methodWeightCompute = args{i+1};
                    case 'ridge_parameter'
                        obj.ridge_parameter = args{i+1};
                    case 'rls'
                        obj.methodWeightCompute = args{i+1};
                    case 'rls_lambda'
                        obj.rls_lambda = args{i+1};
                    case 'rls_delta'
                        obj.rls_delta = args{i+1};
                    otherwise
                        error('the option does not exist');
                end
            end
            
            % checking configurations
            if strcmp(obj.type, 'leaky_esn') && isnan(obj.leaky_parameter)
                error('Incompatible option choosed: leaky_esn but not leaky_parameter set!');
            end
            
            if strcmp(obj.methodWeightCompute, 'ridge_regression') && isnan(obj.ridge_parameter)
                error('Incompatible option choosed: ridge_regression but not ridge_parameter set!');
            end
            
            if strcmp(obj.methodWeightCompute, 'rls') && (isnan(obj.rls_lambda) || isnan(obj.rls_delta))
                error('Incompatible option choosed: RLS but parameters not set correctly!');
            end
            
            %%%% generate weight matrices
            obj.W_in = (0.4 - (-0.4)).*rand(obj.nReservoirUnits, obj.nInputUnits) - 0.4;

            connectivity = min([10/obj.nReservoirUnits, 1]); % 10% of connectivity in the reservoir
            obj.W_hat = obj.rho * generate_internal_weights(obj.nReservoirUnits, connectivity);

            obj.W_out = (0.5 - (-0.5)).*rand(obj.nOutputUnits, obj.nInputUnits+obj.nReservoirUnits) - 0.5;

            %%% trained flag
            obj.trained = 0;
          
        end
        
        function X_i = compute_statematrix ( obj, inputSequence, curr_state, ~ )
            
            nDataPoints = length(inputSequence(:,1));
            
            %%% current inputsequence state matrix
            X_i = zeros(obj.nInputUnits+obj.nReservoirUnits, nDataPoints);
            
            for i = 1:nDataPoints
                
                in = [1 inputSequence(i,:)]';
                x = curr_state(obj.nInputUnits+1:end, :);
                
                switch obj.type
                    case 'plain_esn', x = tanh (obj.W_in * in + obj.W_hat * x);
                    case 'leaky_esn', ...
                            x = (1 - obj.leaky_parameter) .* x + ...
                            obj.leaky_parameter .* tanh (obj.W_in * in  + obj.W_hat * x);
                        
                     otherwise
                        error('Unrecognised esn.type!');
                end
                
                X_i(:,i) = [in; x];
                curr_state = X_i(:,i);
                
            end
            
        end
        
        %% TRAIN
        function last_state = train (obj, trainInputs, trainTargets, washout)
            
            assert(size(trainInputs,1) == size(trainTargets,1));
            
            nTimeSeries = size(trainInputs, 1);
            
            % 'Is it online training?"
            if strcmp(obj.methodWeightCompute, 'rls')
                
                [obj.W_out, last_state] = RLS(obj, trainInputs, trainTargets, washout);
                
            else % direct methods case
                
                % Computing state matrix
                X = compute_multiple_series_state_matrix(obj, trainInputs, washout, NaN, 'training');
                
                % Computing targets taking into account intial transient
                Y = compute_mutiple_series_targets(trainTargets, washout);
                
                % Compressing all X_i into a BIG state matrix
                X_tr = cat(2, X{:});
                last_state = X_tr(:,end);
                
                % Compressing targets into a unique sequence
                y_tr = cat(1, Y{:});
                
                assert(size(X_tr,2) == size(y_tr,1));
                
                % Solve the linear system 'Y_tgt = W_out*X' in terms of W_out using
                switch obj.methodWeightCompute
                    case 'pseudoinverse'
                        obj.W_out = y_tr' * pinv(X_tr);
                    case 'ridge_regression'
                        I = eye(size(X_tr,1));
                        obj.W_out = y_tr' * X_tr' * inv(X_tr*X_tr' + obj.ridge_parameter*I);
                        
                    otherwise error('incorrect obj.methodWeightCompute!');
                end
                
            end

            % set trained flag 
            obj.trained = 1;
        end
        
        %% TEST
        function testPred = test (obj, testInputs, state, washout)
            
            if obj.trained == 0
                error('esn.test: TESTING A NON TRAINED NETWORK!');
            end
            
            X = compute_multiple_series_state_matrix(obj, testInputs, washout, state, 'test');
            % Compressing all X_i into a BIG state matrix
            X_ts = cat(2, X{:});

           % produce predictions real value'
            testPred = (obj.W_out * X_ts)';
            
        end
        
    end
    
end


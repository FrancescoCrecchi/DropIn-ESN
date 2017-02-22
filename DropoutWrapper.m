classdef DropoutWrapper
    
    properties
        p   % PERCENTAGE OF UNITS TO RETAIN
        
    end
    
    methods
        % Constructor
        function obj = DropoutWrapper(p)
            obj.p = p;
        end
        
        function output = train(obj, output)
            % Construct a Bernoulli(p) matrix
            % r = rand(size(output)) < obj.p;
            % output = output .* r;
            
            % Generating a random permutation
            n = size(output,2);
            perm = randperm(n);
            n_drop = round((1 - obj.p) * n);
            
            output(perm(1:n_drop)) = 0;
        end
        
        function weight_matrix = test(obj, weight_matrix)
            % Scaling matrix with the same probability
            weight_matrix = obj.p .* weight_matrix;
        end
        
    end
    
end


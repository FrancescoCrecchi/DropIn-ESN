function internalWeights = generate_internal_weights(nInternalUnits, ...
                                                  connectivity)
% GENERATE_INTERNAL_WEIGHTS creates a random reservoir for an ESN
%  
% inputs:
% nInternalUnits = the number of internal units in the ESN
% connectivity \in [0,1], says how many weights should be non-zero 
%
% output:
% internalWeights = matrix of size nInternalUnits x nInternalUnits
% internalWeights(i,j) = value of weight(synapse) from unit i to unit j
% internalWeights(i,j) might be different from internalWeights(j,i)

%
% Created April 30, 2006, D. Popovici
% Copyright: Fraunhofer IAIS 2006 / Patent pending
% Revision 1, Feb 23, 2007, H. Jaeger
% Revision 2, March 10, 2007, H. Jaeger (replaced eigs by myeigs)
% Revision 3, May 10, 2014, H. Jaeger (replaced myeigs again by eigs)
%
% MODIFIED IN 08/06/2016 BY F.CRECCHI TO FULFIT REQUIREMENTS

success = 0 ;                                               
while success == 0
    % following block might fail, thus we repeat until we obtain a valid
    % internalWeights matrix
    try,
        internalWeights = sprand(nInternalUnits, nInternalUnits, connectivity);

        % scale non-zero weights into [-1, 1] interval
%         internalWeights(internalWeights ~= 0) = ...
%               2 * internalWeights(internalWeights ~= 0) - 1;
        
        internalWeights(internalWeights ~= 0) = ...
            (0.4 - (-0.4)).*internalWeights(internalWeights ~= 0) - 0.4;
              
        opts.disp = 0;
        maxVal = max(abs(eigs(internalWeights,1, 'lm', opts)));
        internalWeights = internalWeights/maxVal;
        success = 1 ;
    catch,
        success = 0 ; 
    end
end


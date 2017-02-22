function [normData, means, vars] = normalizeData0Mean1Var(data, avg, stdev)
% Normalizes a multivariate dataset data (where each data vector is a row in data) 
% by scaling and shifting such that in each column mean/variance becomes 0/1.
% If the variance of a column is zero, the var of this column is not
% changed. 
%
% Shifts and scalings are applied in the order such that
%   normData(:,d) = (data(:,d) + shifts(1,d)) * scalings(1,d);
% 
% Input arg: 
% data: a dataset, either real-valued array of size N by dim or a cell array of size
%    [nrSamples, 1], where the i-th cell is a real-valued array of size N_i by dim 
%
% Outputs:
% normData: the dataset normalized to columns with min/max = 0/1. Each
%    column in normData is computed from the corresponding column in data by 
%    normalizedColumn = scalefactor * (originalColum + shiftconstant). If
%    the input is a cell structure, the same scalefactors and shiftconstants are
%    applied across all cells, such that the *global* mean/var of normData
%    becomes 0/1.
% scalings: a row vector of lenght dim giving the scalefactors
% shifts: a row vector of lenght dim giving the shiftconstants
%
% Created by H. Jaeger, June 07, 2009
  


if isnumeric(data)
    dim = size(data,2);
    
    if ~sum(isnan(avg(:))) && ~sum(isnan(stdev(:)))
        means = avg;
        vars = stdev;
    else
        means = mean(data);
        vars = var(data,1);
    end
       
    normData = data;
    scalings = ones(1,dim);
    shifts = zeros(1,dim);
    for d = 1:dim
        if vars(1,d) > 0
            scalings(1,d) = 1/sqrt(vars(1,d));
            shifts(1,d) = -means(1,d);
            normData(:,d) = (data(:,d) + shifts(1,d)) * scalings(1,d);
        else
            shifts(1,d) = 0;
            normData(:,d) = data(:,d);
            scalings(1,d) = 1;
        end
    end
elseif iscell(data)
    dim = size(data{1},2);
    nrSamples = size(data,1);
    % check if all cells have same dim, and compute total length tl of
    % all data
    tl = 0;
    for n = 1:nrSamples
        if size(data{n},2) ~= dim
            error('all cells must have same row dim');
        end
        tl = tl + size(data{n},1);
    end
    % concatenate all cells and operate on that
    cellsConcat = zeros(tl, dim);
    currentInd = 1;
    for n = 1:nrSamples
       thisCellLength = size(data{n},1); 
       cellsConcat(currentInd:currentInd + thisCellLength - 1, :) = ...
           data{n};
       currentInd = currentInd + thisCellLength;
    end
    normDataConcat = cellsConcat;
    
    if ~sum(isnan(avg(:))) && ~sum(isnan(stdev(:)))
        means = avg;
        vars = stdev;
    else
        means = mean(cellsConcat);
        vars = var(cellsConcat);
    end
    
    scalings = ones(1,dim);
    shifts = zeros(1,dim);
       for d = 1:dim
        if vars(1,d) > 0
            scalings(1,d) = 1/sqrt(vars(1,d));
            shifts(1,d) = -means(1,d);
            normDataConcat(:,d) = (cellsConcat(:,d) + shifts(1,d)) * scalings(1,d); ...
                
        else
            shifts(1,d) = 0;
            normDataConcat(:,d) = cellsConcat(:,d);
            scalings(1,d) = 1;
        end       
       end
    % redistribute over cells
    normData = data;
    currentInd = 1;
    for n = 1:nrSamples
      thisCellLength = size(data{n},1);
      normData{n} = normDataConcat(currentInd:currentInd + ...
                                   thisCellLength-1,:);
      currentInd = currentInd + thisCellLength;
    end
 else error('input data must be array or cell structure');
end




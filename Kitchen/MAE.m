function [error] = MAE( y, y_hat )
    
    assert(size(y,1) == size(y_hat,1));
    
    % Computing Mean Absolute Error
    n = size(y, 1);
    error = (1/n) * sum(abs(y_hat(:,1) - y(:,1)));

end


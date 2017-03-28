function [ mae ] = compute_MAE(tgts, preds)

    % Check training results
    mae = MAE(tgts, (preds > 0.5) + 0);

end


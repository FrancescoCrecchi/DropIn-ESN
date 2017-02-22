function [ mae ] = compute_MAE(preds, tgts)

    % Check training results
    mae = MAE(tgts, (preds > 0.5) + 0);

end


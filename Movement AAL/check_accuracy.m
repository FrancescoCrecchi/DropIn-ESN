function [ acc, cm ] = check_accuracy( y_true, y_hat )

    cm = confusionmat(y_true, y_hat);
    acc = (trace(cm)/sum(cm(:)));

end


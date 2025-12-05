function [d,varargout] = trainDecoder(X,y,regularization)

%% Compute decoder
switch regularization.name
    case 'none'
        d = (X'*X)\(X'*y);
        varargout{1} = 0;
    case 'ridge'
        Rxx = X'*X;
        d = (Rxx+regularization.lambda*trace(Rxx)./size(X,2)*eye(size(X,2)))\(X'*y);
        varargout{1} = regularization.lambda;
    case 'lasso'
        d = admm(X, y, regularization.lambda*norm(X'*y,'inf'), 1, 1);
        varargout{1} = regularization.lambda;        
    case 'lwcov'
        Rxx = lwcov(X);
        d = Rxx\X'*y;
        varargout{1} = 0;
end

end
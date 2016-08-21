function [sequence, L, dE,  dw1, dw2, dw3] = viterbi_deep_crf_2nd_order(X, model,layers, T, rho, EX, bflag)
%
% Performs the Viterbi algorithm on time series X in the deep CRFs
% specified in model, to find the most likely underlying state sequence.
% After predict yhat, it will compute gradient and backpropagate to get 
% all weights in each layer and return them.
% (C) Gang Chen, gangchen@buffalo.edu


    % Initialize some variables
    N = size(X, 2);
    K = numel(model.pi);
    no_hidden = size(model.E, 2);
    ind = zeros(K, K, N);
    sequence = zeros(1, N);
    if N == 0
        L = [];
        return;
    end
    
   
    w1 = layers.w1;
    w2 = layers.w2;
    if isfield(layers, 'w3')
        w3 = layers.w3;
        numlayers = 3;
    else
        numlayers = 2;
        
    end
    dw3 = 0;
    
    
    if ~exist('bflag', 'var') || isempty(bflag)
        bflag = true;
    end
    
    if isfield(layers, 'fvar')
        bflag = false;
    end
    N = size(X,2);
    if bflag
    % feature learning here
        data  = X';
    else
        data = X./repmat(sqrt(layers.fvar), 1, N);
        data = data';
    end
    
    %--------------- forward the deep neural network ----------------
    % [N, numdims] = size(data);
    data = [data ones(N,1)];
    w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)];
    w2probs = 1./(1 + exp(-w1probs*w2)); 
    if numlayers ==3
        w2probs = [w2probs ones(N,1)];
        w3probs = 1./(1 + exp(-w2probs*w3)); %w3probs = [w3probs  ones(N,1)];
        feats = w3probs';
    else
        feats = w2probs';
    end
    if ~exist('EX', 'var') || isempty(EX)
        EX = bsxfun(@plus, model.E' * feats, model.E_bias');
    end
    emission = EX; %exp(bsxfun(@minus, EX, max(EX, [], 1)));

    
    % vectorize y
    yvec = zeros(K, size(T, 2));
    for ti = 1: size(T, 2)
        yvec(T(ti), ti) =1;
    end
    
    % Add margin constraint to emissions
    if exist('T', 'var') && exist('rho', 'var') && ~isempty(T) && ~isempty(rho)
        ii = sub2ind(size(emission), T, 1:length(T));
        emission = emission + rho;
        emission(ii) = emission(ii) - rho;
    end
    
    % Compute message for first two state variables
    omega = model.pi + emission(:,1);
    if N > 1
        omega = bsxfun(@plus, bsxfun(@plus, model.pi2, omega), emission(:,2)');
    end
    
    % Perform forward pass
    for n=3:N
        [omega, ind(:,:,n)] = max(bsxfun(@plus, model.A, omega), [], 1);    % max over variable n - 2
        omega = bsxfun(@plus, squeeze(omega), emission(:,n)');
    end
    
    % Add message for last hidden variable
    if N > 1
        omega = bsxfun(@plus, omega, model.tau');
        omega = omega + model.tau2;
    else
        omega = omega + model.tau;
    end
    
    % Perform backtracking to determine final sequence
    if N > 1
        [L, ii] = max(omega(:));                                            % max over variable N and N - 1
        [sequence(N - 1), sequence(N)] = ind2sub(size(omega), ii);
    else
        [L, sequence(N)] = max(omega, [], 1);
    end
    for n=N - 2:-1:1
        sequence(n) = ind(sequence(n + 1), sequence(n + 2), n + 2);
    end
    
%     % Construct matrix with hidden unit states
%     if nargout > 2
%         Z = repmat(false, [no_hidden N]);
%         for n=1:N
%             Z(:,n) = hidden(:,n,sequence(n));
%         end
%     end

if nargin >3 && exist('T', 'var') && ~isempty(T)
    % vectorize y
    yhat = zeros(K, size(T, 2));
    for ti = 1: size(sequence, 2)
        yhat(sequence(ti), ti) =1;
    end
    
    % add softmax 
    emission = exp(bsxfun(@minus, EX, max(EX, [], 1)));
    emission = emission./repmat(sum(emission,1),K, 1);
    %----------------------- backpropagation -------------------------
    IO = (emission-yvec);
    Ix_class=IO; 
    dE = feats * (yvec- yhat)'; % - lambda2*dw_class1;
    
    % dE = dE +  (dw_class1) + lambda2*dw_class1;
    % dE = dE +  (-dw_class1 + dw_class2);
    % Dfeats = model.E * gamma;
    if numlayers ==3
        Ix3 = Ix_class'*model.E'.*w3probs.*(1-w3probs);
        % Ix3 = Ix3(:,1:end-1);
        dw3 =   w2probs'*Ix3;
        % backpropagation the features 
        Ix2 = (Ix3*w3').*w2probs.*(1-w2probs); 
        Ix2 = Ix2(:,1:end-1);
        dw2 =   w1probs'*Ix2;
    else
        Ix2 = Ix_class'*model.E'.*w2probs.*(1-w2probs); 
        % Ix2 = Ix2(:,1:end-1);
        dw2 =   w1probs'*Ix2;
    end
    Ix1 = (Ix2*w2').*w1probs.*(1-w1probs); 
    Ix1 = Ix1(:,1:end-1);
    dw1 =  data'*Ix1;
end
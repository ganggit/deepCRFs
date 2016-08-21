function [pred_T, model] = deep_crf_2nd_online(train_X, train_T, test_X, test_T, type, layers, average_models, base_eta, rho, max_iter, burnin_iter)
%
% Performs the online deep CRFs on the data in the train_X, and the corresponding targets train_T. 
% The function performs target prediction on the time series in test_X (return pred_T). 
% The variable average_models can be set to (default = false) for online learning, which updates model on each instance.
% The variable base_eta is the base step size (default = 1). The variable
% max_iter indicates the number of iterations (default = 100). The variable
% burnin_iter specifies the burn-in time (default = 10).
% This function will return model (learned via our online deepCRFs) and
% predictions on the test data (test_X) pred_T (part of the code is from
% Laurens van der Maaten)
% 
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose
%
% Gang Chen, SUNY at Buffalo, gangchen@buffalo.edu

    if ~exist('average_models', 'var') || isempty(average_models)
        average_models = false;
    end
    if ~exist('base_eta', 'var') || isempty(base_eta)
        base_eta = 1;
    end
    if ~exist('rho', 'var') || isempty(rho)
        rho = 0;
    end
    if ~exist('max_iter', 'var') || isempty(max_iter)
        max_iter = 100;
    end
    if ~exist('burnin_iter', 'var') || isempty(burnin_iter)
        burnin_iter = 10;
    end

    % Initialize some variables
    if isstruct(type)
        model = type;
        type = model.type;
        no_hidden = size(model.labE, 1);
    end
    
    % added by Gang Chen
    
    if isfield(layers, 'w3')
        numlayers = 3;
        
    else
        numlayers = 2;
    end
    
    [num_dims, num_class] = size(layers.w_class);
    % layers.w_class = 0.1*randn(size(w3,2)+1,numclasses);
    no_hidden = num_dims-1;
    
    
    % Compute total length of data
    n = length(train_X);
    m = length(test_X);
    total_length = 0;
    for i=1:n
        total_length = total_length + length(train_T{i});
    end
    pred_interval = min(2000, n / 10);
    
    % Compute number of features / dimensionality and number of labels
    if strcmpi(type, 'drbm_discrete')
        D = 0;
        for i=1:n
            for j=1:length(train_X{i})
                D = max(D, max(train_X{i}{j}));
            end
        end
        for i=1:m
            for j=1:length(test_X{i})
                D = max(D, max(test_X{i}{j}));
            end
        end
    elseif strcmpi(type, 'drbm_continuous')
        D = size(train_X{1}, 1);
    else
        error('Data type should be discrete or continuous.');
    end
    K = 0;
    for i=1:n
        K = max(K, max(train_T{i}));
    end
    
    % Initialize model
    if ~exist('model', 'var')
        model.type = type;
        model.A  = zeros(K, K, K);
        model.E = randn(no_hidden, K) * .0001;
        model.labE = randn(K, no_hidden) * .0001;
        model.E_bias = zeros(1, K);
        model.labE_bias = zeros(1, K);
        model.pi   = zeros(K, 1);
        model.pi2  = zeros(K, K);
        model.tau  = zeros(K, 1);
        model.tau2 = zeros(K, K);
    end

    % Initialize mean model, or training and test predictions
    if average_models
        mean_model = model;
        ii = 0;
    else
        pred_trn_T = cell(length(train_X), 1);
        pred_tst_T = cell(length(test_X),  1);
        for i=1:length(train_X)
            pred_trn_T{i} = zeros(K, size(train_X{i}, 2));
        end    
        for i=1:length(test_X)
            pred_tst_T{i} = zeros(K, size(test_X{i}, 2));
        end    
    end   
    
    % Compute step sizes
    eta_P  = base_eta / (total_length * numel(model.pi));
    eta_T  = base_eta / (total_length * numel(model.tau)); 
    eta_P2 = base_eta / (total_length * numel(model.pi2));
    eta_T2 = base_eta / (total_length * numel(model.tau2)); 
    eta_A  = base_eta / (total_length * K * K);
    eta_E2 = base_eta / (total_length * numel(model.labE));
    eta_E2_bias = base_eta / (total_length * numel(model.labE));
    
    % weights from the deep learning part
    eta_w1 = base_eta / (total_length * numel(layers.w1));
    eta_w2 = base_eta / (total_length * numel(layers.w2));
    if numlayers ==3
        eta_w3 = base_eta / (total_length * numel(layers.w3));
    end
    if strcmpi(type, 'drbm_discrete')
        eta_E1 = base_eta / (total_length * numel(model.labE));
        eta_E1_bias = base_eta / (total_length * numel(model.labE));
    else
        eta_E1 = base_eta / (total_length * numel(model.E));    
        eta_E1_bias = base_eta / (total_length * numel(model.E));
    end
    
    
    rescale = 0;
    if rescale
        coeff = 0.01;
        eta_A = eta_A * coeff;
        eta_T =eta_T*coeff;
        eta_P = eta_P *coeff;
        
        eta_T2 =eta_T2*coeff;
        eta_P2 = eta_P2 *coeff;
    end
    
    
    % Perform sweeps through training data
    for iter=1:max_iter
        
        % Print out progress
        disp(['Iteration ' num2str(iter) ' of ' num2str(max_iter) '...']);
        old_P = model.pi; old_P2 = model.pi2; old_T = model.tau; old_T2 = model.tau2; old_A = model.A; old_E1 = model.E; old_E2 = model.labE; old_bE1 = model.E_bias; old_bE2 = model.labE_bias;
        ind = randperm(n);
        train_X = train_X(ind);
        train_T = train_T(ind);
        train_err = 0;
        
        %-------------------reinitialization over top layer weight---------
        % this step is very important to improve performance over testing
        % data set, or generalization performance 
        if (iter ==20)
            model.E = randn(no_hidden, K) * .0001;
        end
        % Sweep through all training time series
        for i=1:n
            
            % deep forward here
            feats = deep_project(train_X{i}, layers, numlayers);
            
            % Compute hidden unit states (positive phase)
            if strcmpi(model.type, 'drbm_continuous')
                % EX = bsxfun(@plus, model.E' * train_X{i}, model.E_bias');
                EX = bsxfun(@plus, model.E' * feats, model.E_bias');
            elseif strcmpi(model.type, 'drbm_discrete')
                EX = zeros(no_hidden, length(train_X{i}));
                for j=1:length(train_X{i})
                    EX(:,j) = sum(model.E(train_X{i}{j},:), 1)';
                end
                EX = bsxfun(@plus, EX, model.E_bias');
            end
            lab = zeros(K, length(train_T{i}));
            lab(sub2ind(size(lab), train_T{i}, 1:length(train_T{i}))) = 1;
            % Z_pos = (EX + model.labE' * lab > 0);
            if (numlayers ==3)            
            % Run Viterbi decoder (negative phase)
                [cur_T, ~, dE, dw1, dw2, dw3] = viterbi_deep_crf_2nd_order(train_X{i}, model,layers, train_T{i}, rho, EX);
            else
                [cur_T, ~, dE, dw1, dw2] = viterbi_deep_crf_2nd_order(train_X{i}, model,layers, train_T{i}, rho, EX);
            end
            train_err = train_err + sum(cur_T ~= train_T{i});
            
            % Update co-occurring state parameters (positive phase)
            model.pi(train_T{i}(1)) = model.pi(train_T{i}(1)) + eta_P;
            if length(train_T{i}) > 1
                model.pi2(train_T{i}(1), train_T{i}(2)) = model.pi2(train_T{i}(1), train_T{i}(2)) + eta_P2;
            end
            model.tau(train_T{i}(end)) = model.tau(train_T{i}(end)) + eta_T;
            if length(train_T{i}) > 1
                model.tau2(train_T{i}(end - 1), train_T{i}(end)) = model.tau2(train_T{i}(end - 1), train_T{i}(end)) + eta_T2;
            end            
            for j=3:length(train_T{i})
                model.A(train_T{i}(j - 2), train_T{i}(j - 1), train_T{i}(j)) = ...
                model.A(train_T{i}(j - 2), train_T{i}(j - 1), train_T{i}(j)) + eta_A;
            end
            
            
            % Update co-occurring state parameters (negative phase)
            model.pi(cur_T(1)) = model.pi(cur_T(1)) - eta_P;
            if length(cur_T) > 1
                model.pi2(cur_T(1), cur_T(2)) = model.pi2(cur_T(1), cur_T(2)) - eta_P2;
            end
            model.tau(cur_T(end)) = model.tau(cur_T(end)) - eta_T;
            if length(cur_T) > 1
                model.tau2(cur_T(end - 1), cur_T(end)) = model.tau2(cur_T(end - 1), cur_T(end)) - eta_T2;
            end
            for j=3:length(cur_T)
                model.A(cur_T(j - 2), cur_T(j - 1), cur_T(j)) = model.A(cur_T(j - 2), cur_T(j - 1), cur_T(j)) - eta_A;
            end
            
            
            % deep model here by update the hidden weights for each layer
            model.E = model.E + eta_E1*dE;
            yhat = zeros(K, length(cur_T));
            yhat(sub2ind(size(yhat), cur_T, 1:length(cur_T))) = 1;
            model.E_bias =model.E_bias + eta_E1_bias*sum((lab - yhat),2)';
            layers.w1 =layers.w1 -  eta_w1*dw1;
            layers.w2 =layers.w2 -  eta_w2*dw2;
            if numlayers ==3
                layers.w3 =layers.w3 -  eta_w3*dw3;             
            end
            % Make test predictions
            if iter >= burnin_iter && ~average_models && ~rem(i, pred_interval)
                for j=1:m
                    sequence = viterbi_deep_crf_2nd_order(test_X{j}, model, layers, test_T{j});
                    pred = zeros(K, length(sequence));
                    pred(sub2ind(size(pred), sequence, 1:length(sequence))) = 1;
                    pred_tst_T{j} = pred_tst_T{j} + pred;
                end
            end
        end
        
        % Print out parameter change
        change = sum(abs(old_P - model.pi)) + sum(abs(old_P2(:) - model.pi2(:))) + sum(abs(old_T - model.tau)) + sum(abs(old_T2(:) - model.tau2(:))) + sum(abs(old_A(:) - model.A(:))) + sum(abs(old_E1(:) - model.E(:))) + sum(abs(old_E2(:) - model.labE(:))) + sum(abs(old_bE1(:) - model.E_bias(:))) + sum(abs(old_bE2(:) - model.labE_bias(:)));
        disp(['Cumulative parameter change: ' num2str(change)]);
        disp(['Training error this iteration: ' num2str(train_err / total_length)]);            
        
        % Only if we already have predictions or a mean model
        if iter >= burnin_iter
            pred_T = cell(m, 1);
            
            % Average models and perform prediction
            if average_models
                ii = ii + 1;
                mean_model.pi   = ((ii - 1) / ii) .* mean_model.pi   + (1 / ii) .* model.pi;                
                mean_model.pi2  = ((ii - 1) / ii) .* mean_model.pi2  + (1 / ii) .* model.pi2;                
                mean_model.tau  = ((ii - 1) / ii) .* mean_model.tau  + (1 / ii) .* model.tau;
                mean_model.tau2 = ((ii - 1) / ii) .* mean_model.tau2 + (1 / ii) .* model.tau2;
                mean_model.A    = ((ii - 1) / ii) .* mean_model.A    + (1 / ii) .* model.A;
                mean_model.E    = ((ii - 1) / ii) .* mean_model.E    + (1 / ii) .* model.E;
                mean_model.labE = ((ii - 1) / ii) .* mean_model.labE + (1 / ii) .* model.labE;
                mean_model.E_bias    = ((ii - 1) / ii) .* mean_model.E_bias    + (1 / ii) .* model.E_bias;
                mean_model.labE_bias = ((ii - 1) / ii) .* mean_model.labE_bias + (1 / ii) .* model.labE_bias;
                for i=1:m
                    pred_T{i} = viterbi_deep_crf_2nd_order(test_X{i}, mean_model, layers);
                end
                
            % Get most likely sequences after voting
            else
                pred_T = cell(m, 1);
                for i=1:m
                    [~, pred_T{i}] = max(pred_tst_T{i}, [], 1);
                end
            end
            
            % Compute current test error
            err = 0; len = 0;
            for i=1:m
                len = len + length(pred_T{i});
                err = err + sum(pred_T{i} ~= test_T{i});
            end            
            disp([' - test error: ' num2str(err / len)]);
        end
    end
    model = mean_model;
    
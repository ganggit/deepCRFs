function err = deep_ocr_experiment(numhids, lambda, rho, window)
%OCR_EXPERIMENT Runs experiment on OCR data set
% Version 1.000
%
% Code provided by Gang Chen, SUNY at Buffalo
% gangchen@buffalo.edu
% 
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% 
% Reference: 
% Sequential Labeling with online Deep Learning, in ECML 2016
% Authors: Gang Chen , Ran Xu and Sargur Srihari

    % Process inputs
    if ~exist('numhids', 'var') || isempty(numhids)
        %numhids = [1000 400 100];
        numhids = [400 200 100];
    end
    if ~exist('lambda', 'var') || isempty(lambda)
        lambda = 0.2;
    end
    if ~exist('rho', 'var') || isempty(rho)
        rho = 0;
    end
    
    % base_eta = 100 * max(1, no_hidden / 100); 
    base_eta = 10000;
    batchsize = 100;
    bflag = true; %false ; % it indicates whether the data is binary or not
    
    % Load data (and convert to [-1, 1] representation)
    X = cell(0);
    
    fdir = './data/';
    fname = 'ocr';
    load([fdir filesep fname]);
    %for i=1:length(X)
    %    X{i} = X{i} >0; %sign(X{i} - .5);%
    %end
    numclasses = length(words);

    % Perform 10-fold cross-validation
    
    % fname =[fname '3'];
    no_folds = 10;
    err = ones(no_folds, 1);
    word_err = ones(no_folds,1);
    perm = randperm(length(X)); ind = 1;
    fold_size = floor(length(X) ./ no_folds);
    for fold=1:no_folds
    
        % Split into training and test set
        disp(['Fold ' num2str(fold) ' of ' num2str(no_folds) '...']);
        train_ind = perm([1:ind - 1 ind + fold_size:end]);
        test_ind = perm(ind:ind + fold_size - 1);        
        train_X = X(train_ind);
        train_T = labels(train_ind);
        test_X  = X(test_ind);
        test_T  = labels(test_ind);
        ind = ind + fold_size;
        

        init_model = [fname '_weights.mat'];
        if ~exist(init_model, 'file')
        % pretraining with RBM/parameters for each layer
            if ~exist('layers_ocr.mat', 'file')
            layers = pretraining(train_X, train_T, numhids, batchsize, 100, ~bflag); % 3 layers
            save('layers_ocr.mat', 'layers');
            else
                load('layers_ocr.mat');
                w1 = layers(1).w1;
                w2 = layers(2).w2;
                w3 = layers(3).w3;
                %w4 = layers(4).w4;
                w_class = 0.1*randn(size(w3,2)+1,numclasses);
            end
            [w1, w2, w3, w_class] =learn_deepneuralnetwork(train_X, train_T,  layers, fname, 100, bflag);
        else
            % load layers.mat;
            load(init_model);
        end
        % initialize regularization parameters
        layers = [];
        if ~bflag
            layers.fvar = fvar;
        end
        layers.w1 = w1;
        layers.w2 = w2;
        layers.w3 = w3;
        layers.w_class = w_class;
        layers.lambda = lambda;
        layers.lambda1 = 1; % lambda1;
        layers.lambda2 = 0.2;%0.5 % lambda2;

        % Perform learning and predictions using deepCRFs
        pred_T = deep_crf_2nd_online(train_X, train_T, test_X, test_T, 'drbm_continuous', layers, false, base_eta, rho);
                
    
        % Measure per-character tagging error
        err(fold) = 0; tot = 0;
        for i=1:length(pred_T)
            tot = tot + length(test_T{i});
            err(fold) = err(fold) + sum(pred_T{i} ~= test_T{i});
        end
        err(fold) = err(fold) / tot;
        disp(['Per-character tagging error (test set): ' num2str(err(fold))]);
        
        % Measure per-word tagging error
        word_err(fold) = 0; tot = 0;
        for i=1:length(pred_T)
            tot = tot + 1;%length(test_T{i});
            word_err(fold) = word_err(fold) + (sum(pred_T{i} == test_T{i})==length(test_T{i}));
        end
        word_err(fold) = (tot - word_err(fold)) / tot;
        disp(['Per-word tagging error (test set): ' num2str(word_err(fold))]);
        
    %end
    disp(['Mean error over ' num2str(no_folds) ' folds (lambda = ' num2str(lambda) '): ' num2str(mean(err)) ' (std. dev. ' num2str(std(err)) ')']);
    save('ocr_deepcrfs_result.mat', 'err', 'word_err');
    end

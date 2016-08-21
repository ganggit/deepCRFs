function [w1, w2, w3, w_class] = learn_deepneuralnetwork(X, labels,  layers, fname, batchsize, bflag)

% Version 1.000
%
% Code provided by Ruslan Salakhutdinov and Geoff Hinton
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% This program fine-tunes an autoencoder with backpropagation.
% Weights of the autoencoder are going to be saved in mnist_weights.mat
% and trainig and test reconstruction errors in mnist_error.mat
% You can also set maxepoch, default value is 200 as in our paper.  

if nargin <5
    batchsize =100;
end
if nargin <6
   bflag = true; 
end


maxepoch=100;% 200
fprintf(1,'\nTraining discriminative model on MNIST by minimizing cross entropy error. \n');
fprintf(1,'60 batches of 1000 cases each. \n');

disp('split the training data for cross-validation...');
perm = randperm(length(X));
train_ind = perm(1:round(.9 * length(X)));
test_ind  = perm(1+round(.9 * length(X)):end); 
train_X = X(train_ind); train_T = labels(train_ind);
test_X  = X(test_ind);  test_T  = labels(test_ind); 


% get the sigma for gaussian
numdims = size(train_X{1}, 1);
if ~bflag && isfield(layers(1), 'fvar')
    fvar = layers(1).fvar;
    mdata = layers(1).mdata;
    if size(mdata,2) ~= size(fvar,2)
        mdata = mdata';
    end
else
    fvar = ones(1, numdims);
    mdata = zeros(1, numdims);
end 

data = [];
labels = [];
% only consider the data that has conditional probability


for i=1: length(train_X)  
    
    numcases = size(train_X{i}, 2);
    if ~bflag  % bsxfun(@minus, Data, avedata);
        norm_X = bsxfun(@minus, train_X{i}', mdata)./repmat(sqrt(fvar), numcases, 1);
    else
       norm_X = bsxfun(@minus, train_X{i}', mdata);%train_X{i}';
    end
    % data = [data; train_X{i}'];
    data = [data; norm_X];
    labels = [labels; train_T{i}'];   
end


numclasses = max(labels(:));

testdata = [];
testlabels = [];
% only consider the data that has conditional probability

numdims = size(test_X{1}, 1);
for i=1: length(test_X)  
    numcases = size(test_X{i}, 2);
    if ~bflag 
        norm_X = bsxfun(@minus, test_X{i}', mdata)./repmat(sqrt(fvar), numcases, 1);
    else 
	norm_X = bsxfun(@minus, test_X{i}', mdata);%test_X{i}';    
    end
    % testdata = [testdata; test_X{i}'];
    testdata = [testdata; norm_X];
    testlabels = [testlabels; test_T{i}'];   
end

fprintf(1,'Converting Raw files into Matlab format \n');
% [batchdata, batchtargets, testbatchdata, testbatchtargets] = dividedbatches(data, labels, testdata, testlabels, batchsize); 
[batchdata, batchtargets] = getbatches(data, labels, numclasses, batchsize);
[testbatchdata, testbatchtargets] = getbatches(testdata, testlabels, numclasses, batchsize);


[numcases numdims numbatches]=size(batchdata);
N=numcases; 

%%%% PREINITIALIZE WEIGHTS OF THE DISCRIMINATIVE MODEL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load mnistvhclassify
% load mnisthpclassify
% load mnisthp2classify
% w1=[vishid; hidrecbiases];
% w2=[hidpen; penrecbiases];
% w3=[hidpen2; penrecbiases2];

w1 = layers(1).w1;
w2 = layers(2).w2;
w3 = layers(3).w3;
w_class = 0.1*randn(size(w3,2)+1,numclasses);
 

%%%%%%%%%% END OF PREINITIALIZATIO OF WEIGHTS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

l1=size(w1,1)-1;
l2=size(w2,1)-1;
l3=size(w3,1)-1;
l4=size(w_class,1)-1;
l5=numclasses; 
test_err=[];
train_err=[];


disp('learning the deep (full connected) neural network ...');
for epoch = 1:maxepoch

%%%%%%%%%%%%%%%%%%%% COMPUTE TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err=0; 
err_cr=0;
counter=0;
[numcases numdims numbatches]=size(batchdata);
N=numcases;
 for batch = 1:numbatches
  data = [batchdata(:,:,batch)];
  target = [batchtargets(:,:,batch)];
  data = [data ones(N,1)];
  w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)];
  w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
  w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
  targetout = exp(w3probs*w_class);
  targetout = targetout./repmat(sum(targetout,2),1,numclasses);

  [I J]=max(targetout,[],2);
  [I1 J1]=max(target,[],2);
  counter=counter+length(find(J==J1));
  err_cr = err_cr- sum(sum( target(:,1:end).*log(targetout))) ;
 end
 train_err(epoch)=(numcases*numbatches-counter);
 train_crerr(epoch)=err_cr/numbatches;

%%%%%%%%%%%%%% END OF COMPUTING TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%% COMPUTE TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err=0;
err_cr=0;
counter=0;
[testnumcases testnumdims testnumbatches]=size(testbatchdata);
N=testnumcases;
for batch = 1:testnumbatches
  data = [testbatchdata(:,:,batch)];
  target = [testbatchtargets(:,:,batch)];
  data = [data ones(N,1)];
  w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)];
  w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
  w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
  targetout = exp(w3probs*w_class);
  targetout = targetout./repmat(sum(targetout,2),1,numclasses);

  [I J]=max(targetout,[],2);
  [I1 J1]=max(target,[],2);
  counter=counter+length(find(J==J1));
  err_cr = err_cr- sum(sum( target(:,1:end).*log(targetout))) ;
end
%  test_err(epoch)=(testnumcases*testnumbatches-counter);
 test_err(epoch)= 1-counter/(testnumcases*testnumbatches);
 test_crerr(epoch)=err_cr/testnumbatches;
 fprintf(1,'Before epoch %d Train # misclassified: %d (from %d). Test # misclassified: %d (from %d) \t \t \n',...
            epoch,train_err(epoch),numcases*numbatches,test_err(epoch),testnumcases*testnumbatches);

%%%%%%%%%%%%%% END OF COMPUTING TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 tt=0;
 for batch = 1:numbatches/10
 fprintf(1,'epoch %d batch %d\r',epoch,batch);

%%%%%%%%%%% COMBINE 10 MINIBATCHES INTO 1 LARGER MINIBATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 tt=tt+1; 
 data=[];
 targets=[]; 
 for kk=1:10
  data=[data 
        batchdata(:,:,(tt-1)*10+kk)]; 
  targets=[targets
        batchtargets(:,:,(tt-1)*10+kk)];
 end 

%%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  max_iter=3;

  if epoch<6  % First update top-level weights holding other weights fixed. 
    N = size(data,1);
    XX = [data ones(N,1)];
    w1probs = 1./(1 + exp(-XX*w1)); w1probs = [w1probs  ones(N,1)];
    w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
    w3probs = 1./(1 + exp(-w2probs*w3)); %w3probs = [w3probs  ones(N,1)];

    VV = [w_class(:)']';
    Dim = [l4; l5];
    [X, fX] = minimize(VV,'CG_CLASSIFY_INIT',max_iter,Dim,w3probs,targets);
    w_class = reshape(X,l4+1,l5);

  else
    VV = [w1(:)' w2(:)' w3(:)' w_class(:)']';
    Dim = [l1; l2; l3; l4; l5];
    [X, fX] = minimize(VV,'CG_CLASSIFY',max_iter,Dim,data,targets);

    w1 = reshape(X(1:(l1+1)*l2),l1+1,l2);
    xxx = (l1+1)*l2;
    w2 = reshape(X(xxx+1:xxx+(l2+1)*l3),l2+1,l3);
    xxx = xxx+(l2+1)*l3;
    w3 = reshape(X(xxx+1:xxx+(l3+1)*l4),l3+1,l4);
    xxx = xxx+(l3+1)*l4;
    w_class = reshape(X(xxx+1:xxx+(l4+1)*l5),l4+1,l5);

  end
%%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 end
 if bflag
    save([fname '_weights.mat'], 'w1', 'w2', 'w3', 'w_class');
 else
     save([fname '_weights.mat'], 'w1','fvar','mdata', 'w2', 'w3', 'w_class');
 end
 save ocr_classify_error test_err test_crerr train_err train_crerr;

end



function [batchdata, batchtargets] = getbatches(data, labels, nclasses, batchsize)

if nargin <4
    batchsize = 100;
end
% for training
[N, numdims] = size(data);
N = floor(N/batchsize)*batchsize;

if size(labels,2)==1
u= unique(labels);
if nargin <3
    nclasses = length(u);
end
targets= zeros(N, nclasses);
%Create targets: 1-of-k encodings for each discrete label
for i=1:length(u)
    targets(labels==u(i),i)=1;
end
else
    targets = labels;
    nclasses = size(labels,2);
end

%Create batches
numbatches= floor(N/batchsize);
batchdata = zeros(batchsize, numdims, numbatches);
batchtargets = zeros(batchsize, nclasses, numbatches);
groups= repmat(1:numbatches, 1, batchsize);
groups= groups(1:N);
groups = groups(randperm(N));
for i=1:numbatches
    batchdata(:,:,i)= data(groups==i,:);
    batchtargets(:,:,i)= targets(groups==i,:);
end




function layers = pretraining(X, Y, numhids, batchsize, maxepoch, continuous)

if nargin <3
    numhid=500; numpen=500; numpen2=2000; 
    
else
    
    numhid = numhids(1);
    numpen = numhids(2);
    numpen2 = numhids(3);
    
end


if nargin <4
    batchsize =100;
end
if nargin <5
    maxepoch=100; 
end
if nargin <6
    continuous=false; 
end
if strcmp(class(X), 'cell')
    % transform it into double
    [data, labels] = cell2double(X,Y);
    
else
    data = X;
    labels = Y;
end
fprintf(1,'Pretraining a deep autoencoder. \n');
fprintf(1,'The Science paper used 50 epochs. This uses %3i \n', maxepoch);


[batchdata, batchtargets] = getbatches(data', labels', batchsize);


[numcases numdims numbatches]=size(batchdata);

fprintf(1,'Pretraining Layer 1 with RBM: %d-%d \n',numdims,numhid);
restart=1;

if ~continuous

    rbm;
else
    %[vishid, visbiases, hidbiases, fvar, batchposhidprobs]= grbm_1layer(data, labels, numhid);
    [vishid, visbiases, hidbiases, fvar, batchposhidprobs]= grbm_1layer(batchdata, batchtargets, numhid);
end
hidrecbiases=hidbiases; 
save mnistvhclassify vishid hidrecbiases visbiases;

fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d \n',numhid,numpen);
batchdata=batchposhidprobs;
numhid=numpen;
restart=1;
rbm;
hidpen=vishid; penrecbiases=hidbiases; hidgenbiases=visbiases;
save mnisthpclassify hidpen penrecbiases hidgenbiases;

fprintf(1,'\nPretraining Layer 3 with RBM: %d-%d \n',numpen,numpen2);
batchdata=batchposhidprobs;
numhid=numpen2;
restart=1;
rbm;
hidpen2=vishid; penrecbiases2=hidbiases; hidgenbiases2=visbiases;
save mnisthp2classify hidpen2 penrecbiases2 hidgenbiases2;


% -------------------------------------------------
% return parameters here
% -------------------------------------------------
layers = [];
load mnistvhclassify
load mnisthpclassify
load mnisthp2classify

% --------------------------------------------------
% PREINITIALIZE WEIGHTS OF THE DISCRIMINATIVE MODEL
%---------------------------------------------------
w1=[vishid; hidrecbiases];
w2=[hidpen; penrecbiases];
w3=[hidpen2; penrecbiases2];

layers(1).w1 = w1;
if continuous
    layers(1).fvar = fvar;
end
layers(2).w2 = w2;
layers(3).w3 = w3;
end


function [data, labels] = cell2double(X,Y)
nums = 0;
for i =1: length(X)
    nums = nums + size(X{i}, 2);
end

data = zeros(size(X{1}, 1), nums);
labels = zeros(1, nums);

idx = 1;
for i =1: length(X)
    data(:, idx:idx+size(X{i}, 2)-1) = X{i};
    labels(idx:idx+size(X{i}, 2)-1) = Y{i};
    
    idx = idx+size(X{i}, 2);
end
end

function [batchdata, batchtargets] = getbatches(data, labels, batchsize)

if nargin <3
    batchsize = 100;
end
% for training
[N, numdims] = size(data);
N = floor(N/batchsize)*batchsize;

if size(labels,2)==1
u= unique(labels);
nclasses = length(u);
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
end
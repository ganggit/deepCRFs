function output = deep_project(X, layers, numlayers, bflag)

%--------- forward in deep neural network ---------------------
% Gang Chen, SUNY at Buffalo, gangchen@buffalo.edu
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our webpage.


if nargin <3 && length(layers)==1
    

    if isfield(layers, 'w3')
        numlayers =3;
    else
        numlayers = 2;
    end
elseif nargin <3 && length(layers)>1
    
    numlayers = length(layers);
    % for i = 1:layers(ilayer)
    %     eval(['w' num2str(i)]) = layers(i).['w' num2str(i)];
    % end
    if numlayers <3
        w1= layers(1).w1;
        w2 = layers(2).w2;
    else
        w1= layers(1).w1;
        w2 = layers(2).w2;
        w3 = layers(3).w3;
    end
end

if length(layers)==1
    
    w1= layers.w1;
    if isfield(layers, 'w2')
    w2 = layers.w2;
    numlayers = 2;
    end

    if isfield(layers, 'w3')
    w3 = layers.w3;
    numlayers =3;
    end       
end   

if nargin <4 
    bflag = true;
end

if isfield(layers, 'fvar')
    bflag = false;
end

% if nargin <3
% 
%     
% 
%         w1= layers(1).w1;
%         w2 = layers(2).w2;
%         w3 = layers(3).w3;
%     else
% 
%         w1= layers.w1;
%         w2 = layers.w2;
%         w3 = layers.w3;
%     end
% 
% 
% 
%     data  = X';
%     [N, numdims] = size(data);
%     data = [data ones(N,1)];
%     w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)];
%     w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
%     w3probs = 1./(1 + exp(-w2probs*w3)); %w3probs = [w3probs  ones(N,1)];
%     output = w3probs';

N = size(X,2);    
if bflag    
    data  = X';
else
    data = X./repmat(sqrt(layers.fvar), 1, N);
    data = data';
end
%[N, numdims] = size(data);
data = [data ones(N,1)];
w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)];
w2probs = 1./(1 + exp(-w1probs*w2));

if numlayers ==2


    output = w2probs';
else
    w2probs_plus = [w2probs ones(N,1)];
    w3probs = 1./(1 + exp(-w2probs_plus*w3)); %w3probs = [w3probs  ones(N,1)];
    output = w3probs';
end

end
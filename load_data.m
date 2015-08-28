function [train, valid, test, vocab, I] = load_data(datafile, I, N, K)
% This method loads the training, validation and test set.
% It also divides the training set into mini-batches.
% Inputs:
%   I: Index of the target word
%   N: Mini-batch size.
%   K: Number of mini-batch to keep
% Outputs:
%   train_input: An array of size D X N X M, where
%                 D: number of input dimensions (in this case, 3).
%                 N: size of each mini-batch (in this case, 100).
%                 M: number of minibatches.
%   train_target: An array of size 1 X N X M.
%   valid_input: An array of size D X number of points in the validation set.
%   test: An array of size D X number of points in the test set.
%   vocab: Vocabulary containing index to word mapping.

% default settings of the input parameters
if nargin < 4, K = Inf; end
if nargin < 3, N = 100; end
if nargin < 2, I = 'last'; end

% load data (train/test/validate) and vocabulary (250 words)
load(datafile);

% get input dimension 
numdims = size(data.train, 1);

% validate the input index for target word
errMessage1 = ['Target word index can only be string "first", ', ...
    '"last", or an integer number!'];
errMessage2 = ['Target word index should in the range of [1, ', ...
            num2str(numdims), ']'];
if ischar(I)
    if strcmp(I, 'first')
        I = 1;
    elseif strcmp(I, 'last')
        I = numdims;
    else
        error(errMessage1);
    end
elseif ((isnumeric(I) && isequal(I, fix(I))) || isinteger(I)) && I > 0
    if I > numdims
        error(errMessage2);
    end
else
    error(errMessage1)
end

% -1 for removing the dimension of predicted word
D = numdims - 1;

% get # of minibatches (take floor to remove the residuals)
M = floor(size(data.train, 2) / N); 

% update M to the # of minibatches needed to keep
M = min(M,K);

% seperate training data into inputs and tagets with minibaches
train.input = reshape(data.train([1:I-1,I+1:numdims], 1:N*M), D, N, M);
train.target = reshape(data.train(I, 1:N*M), 1, N, M);

% seperate test and validation data into inputs and tagets
valid.input = data.valid([1:I-1,I+1:numdims], :);
valid.target = data.valid(I, :);
test.input = data.test([1:I-1,I+1:numdims], :);
test.target = data.test(I, :);

% get vocabulary
vocab = data.vocab;

end

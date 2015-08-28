function states = fprop(input_batch, weights, bias, cand_idx, probtype)
% This method forward propagates through a neural network.
% Inputs:
%   input_batch: The input data as a matrix of size numwords X batchsize
%   where, numwords is the number of words, batchsize is the number of data
%   points. So, if input_batch(i, j) = k then the ith word in data point j
%   is word index k of the vocabulary.
%
%   word_embedding_weights (weights{1}): Word embedding as a matrix of size
%   vocab_size X numhid1, where vocab_size is the size of the vocabulary
%   numhid1 is the dimensionality of the embedding space.
%
%   embed_to_hid_weights (weights{2:numlayers-2}): Weights between the word
%   embedding layer and hidden layer as a matrix of size numhid1*numwords X
%   numhid2, numhid2 is the number of hidden units.
%
%   hid_to_output_weights (weights{numlayers-1}): Weights between the
%   hidden layer and output softmax unit as a matrix of size numhid2 X V
%
%   hid_bias (bias{1:end-1}): Bias of the hidden layer as a matrix of size
%   numhid2 X 1.
%
%   output_bias (bias{end}): Bias of the output layer as a matrix of size
%   V X 1.
%
% Outputs:
%   embedding_layer_state (states{1}): State of units in the embedding
%   layer as a matrix of size numhid1*numwords X batchsize
%
%   hidden_layer_state (states{2:end-1}): State of units in the hidden
%   layer as a matrix of size numhid2 X batchsize
%
%   output_layer_state (states{end}): State of units in the output layer as
%   a matrix of size vocab_size X batchsize
%

% default setting for probability type
if nargin < 5, probtype = 'regular'; end
if nargin < 4, cand_idx = 1:length(bias{end}); end

% obtain parameters
[numwords, mbsz] = size(input_batch);
numemb = size(weights{1}, 2);
numlayers = length(weights) + 1;
% V = size(weights{1}, 1);
v = length(cand_idx); % use 'v' instead of 'v' if short list is used
numhid = zeros(1, numlayers-2);
for i = 1:numlayers-2
   numhid(i) = size(weights{i+1}, 1); 
end

% get dimensions
nb = length(bias);
nw = length(weights);
nz = numlayers - 2;
ny = numlayers - 1;

% initialize inputs and states
inputs = cell(nz, 1);
states = cell(ny, 1);

%% COMPUTE STATE OF WORD EMBEDDING LAYER.

% Look up the inputs word indices in the word_embedding_weights matrix.
% Reshape input_batch (numwords X batchsize, e.g. 3 X 100) to one column 
% (1 X numwords*batchsize, e.g. 1 X 300)
tmp1 = reshape(input_batch, [], 1); 
% Find corresponding word embedding weights by indeces in tmp1 and
% transpose (numhid1 X numwords*batchsize, e.g. 50 X 300)

tmp2 = zeros(numemb, numwords*mbsz);
% assign weights corresponding to the non-missing input words
tmp2(:, tmp1~=0) = weights{1}(tmp1(tmp1~=0),:)';
% assign mean weights to missing words
if any(tmp1==0)
    idx = tmp1==0;
    tmp2(:, idx) = mean(weights{1})' * ones(1, sum(idx)); 
end
% tmp2 = weights{1}(tmp1, :)';
% Reshape again to get states of word embedding layer, i.e. weights of 3
% words in single column X batchsize (e.g. 150 X 100)
states{1} = reshape(tmp2, numhid(1), []);
clear tmp1 tmp2

%% COMPUTE STATE OF HIDDEN LAYER.

for i = 2:numlayers-2
    
    % Compute inputs to hidden units, i.e. z_hid = w_{emb-hid}' * y_emb + B_hid.
    inputs{i-1} = weights{i}' * states{i-1} + repmat(bias{i-1}, 1, mbsz);
    % Apply logistic activation function, i.e. y_hid = 1 / (1 + exp(-z_hid)).
    states{i} = 1 ./ (1 + exp(-inputs{i-1}));
end

%% COMPUTE STATE OF OUTPUT LAYER.

% Compute inputs to softmax, i.e. z_out = w_{hid-out}' * y_hid + B_out.
% use short list if v < V
weights_short = weights{nw}(:, cand_idx);
bias_short = bias{nb}(cand_idx);
inputs{end} = weights_short' * states{ny-1} + repmat(bias_short, 1, mbsz); 
% inputs{end} = weights{nw}' * states{ny-1} + repmat(bias{nb}, 1, mbsz); 

% Subtract maximum. 
% Remember that adding or subtracting the same constant from each input to a
% softmax unit does not affect the outputs. Here we are subtracting maximum to
% make all inputs <= 0. This prevents overflows when computing their
% exponents.
inputs_to_softmax_max = max(inputs{end});
inputs{nz} = inputs{nz} - repmat(inputs_to_softmax_max, v, 1);
% inputs{nz} = inputs{nz} - repmat(inputs_to_softmax_max, V, 1);

% Compute exp.
states{ny} = exp(inputs{nz});

% Normalize to get probability distribution.
output_layer_state_sum = sum(states{ny});
states{ny} = states{ny} ./ repmat(output_layer_state_sum, v, 1);
% states{ny} = states{ny} ./ repmat(output_layer_state_sum, V, 1);

% option for log likelihood output
if strcmp(probtype, 'loglik')
   states{ny} = sqrt(-2.0*log(states{ny})); 
end

end

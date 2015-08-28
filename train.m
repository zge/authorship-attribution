function model = train(datafile, para, varargin)
% This function trains a neural network language model.
%
% Inputs with hyperparameters:
%   epochs: Number of epochs (passes through training set) to run.
%   mbsz: Mini-batch size
%   lr: Learning rate; default = 0.1.
%   momentum: Momentum; default = 0.9.
%   wc: weight cost (for regulization using penalty)
%   V: vocabulary size.
%   numemb: Dimensionality of embedding space; default = 50
%   numhid1: dimension of the concatenation of embedding layers
%   numhid2: dimensions of the hidden layer(s), can be row vector
%   numhid: [numhid1, numhid2].
%   w0: Standard deviation of the normal distribution, which is
%   sampled to get the initial weights; default = 0.01
% Output:
%   model: A struct containing the learned weights and biases and vocabulary.

% start clock
if size(ver('Octave'),1)
    OctaveMode = 1;
    start_time = time;
else
    OctaveMode = 0;
    start_time = clock;
end

% set default parameters
if nargin > 1 && ischar(para)
    varargin = [para, varargin];
    para = struct;
end
if nargin < 2, para = struct; end

% set default parameter values using process options
[epochs, mbsz, lr, momentum, wc, V, numemb, numhid2, w0, lrDecayAfter, halving, ...
    showTrainCEAfter, showValidCEAfter, targetIdx] = process_options(varargin, ...
    'epochs', 10, 'mbsz', 100, 'lr', 0.1, 'momentum', 0.9, 'wc', 0.0, 'V', Inf, ...
    'numemb', 50, 'numhid2', 200, 'w0', 0.01, 'lrDecayAfter', 5, 'halving', 0.9, ...
    'showTrainCEAfter', 500, 'showValidCEAfter', Inf, 'targetIdx', 'last');

% changes parameters specified in para (override values in process options)
vars = fields(para);
for i = 1:length(vars)
   %assignin('caller', vars{i}, para.(vars{i})); 
   fprintf('update %s: %s -> %s\n', vars{i}, num2str(eval(vars{i})), ...
       num2str(para.(vars{i})))
   eval([vars{i}, ' = ', num2str(para.(vars{i})), ';']); 
end
fprintf('\n');

% LOAD DATA.
% datafile = 'data.mat';
[train, valid, test, vocab, targetIdx] = load_data(datafile, targetIdx, mbsz);
[numwords, ~, numbatches] = size(train.input);
V = min(length(vocab), V);
numhid = [numwords * numemb, numhid2];
% total # of layers = # of hidden layers (including concatenated emedding
% layer) + input layer + output layer
numlayers = length(numhid) + 2; 

% display parameter settings
fprintf('target word index: %d\n', targetIdx);
fprintf('# of epochs: %d\n', epochs);
fprintf('mini-batch size: %d\n', mbsz);
fprintf('initial learning rate: %.3f\n', lr);
fprintf('weight cost: %.3f\n', wc);
fprintf('weight scale factor: %.3f\n', w0);
hid_str = cellfun(@num2str, num2cell(numhid2), 'UniformOutput', 0);
fprintf('model structure: (%d x %d) (%d x %d) (%s) (%d)\n', numwords, V, ...
    numwords, numemb, strjoin(hid_str,' : '), V);
fprintf('learning rate decay after: %d\n', lrDecayAfter);
fprintf('learning rate halving factor: %.3f\n', halving);
fprintf('show train CE after: %d\n', showTrainCEAfter);
fprintf('show validation CE after: %d\n', showValidCEAfter);

% % randomize training data and labels
% sel = randperm(numbatches);
% train_input = train_input(:, :, sel);
% train_target = train_target(:, :, sel);

% INITIALIZE MODEL (WEIGHTS, BIASES and their DELTA).
lrs = ones(epochs, 1) * lr;
weights = cell(numlayers-1, 1);
bias = cell(numlayers-2, 1); % only hidden and output layers have bias
delta_weights = cell(numlayers-1, 1);
delta_bias = cell(numlayers-2, 1);
weights{1} = w0 * randn(V, numemb);
delta_weights{1} = zeros(V, numemb);

% emedding-to-hidden and hidden-to-hidden
for i = 2:numlayers-2
    weights{i} = w0 * randn(numhid(i-1), numhid(i));
    bias{i-1} = zeros(numhid(i), 1);
    delta_weights{i} = zeros(numhid(i-1), numhid(i));
    delta_bias{i-1} = zeros(numhid(i), 1);
end
% hidden-to-output
weights{end} = w0 * randn(numhid(end), V);
bias{end} = zeros(V, 1);
delta_weights{end} = zeros(numhid(end), V);
delta_bias{end} = zeros(V, 1);

% INITIALIZE FINAL MODEL
weightsFinal = weights;
biasFinal = bias;

% initialize the best validation CE used for early termination
best_validset_CE = Inf;

% TRAIN.
hist_trainset_CE = zeros(epochs, 1);
hist_validset_CE = zeros(epochs, 1);
tElapsed = zeros(epochs, 1);
for ep = 1:epochs
    
    tStart = tic;
    
    % reset parameters
    fprintf(1, '\rEpoch %d\n', ep);
    count = 0;
    this_chunk_CE = 0;
    trainset_CE = 0;
    
    % update learning rate
    if ep > lrDecayAfter
        lrs(ep) = lrs(ep-1) * halving;
        fprintf(1, 'Reduce learning rate: %.3f -> %.3f\n', ...
            lrs(ep-1), lrs(ep));
    end
    
    % LOOP OVER MINI-BATCHES.
    for m = 1:numbatches
        input_batch = train.input(:, :, m);
        target_batch = train.target(:, :, m);
        
        % FORWARD PROPAGATE.
        % Compute the state of each layer in the network given the input batch
        % and all weights and biases
        states = fprop(input_batch, weights, bias);
        
        % COMPUTE DERIVATIVE AND BACK PROPAGATE.
        [CE, gradient_weights, gradient_bias] = bprop(input_batch, ...
            target_batch, states, weights(2:end));
        
        count =  count + 1;
        this_chunk_CE = this_chunk_CE + (CE - this_chunk_CE) / count;
        trainset_CE = trainset_CE + (CE - trainset_CE) / m;
        
        if mod(m, showTrainCEAfter) == 0
            fprintf(1, 'Batch %d Train CE %.3f\n', m, this_chunk_CE);
            count = 0;
            this_chunk_CE = 0;
        end
        if OctaveMode, fflush(1); end
        
        % UPDATE WEIGHTS AND BIASES. momentum determines how much should be
        % remebered from the trained NNs, weight gradient should be divided
        % by mini-batch size (mbsz) since (normalization) previously we are
        % deal with the whole batch without averaging - zge
        
        % weight delta is an averaged version of weight gradient which carries
        % information from the past
        
        % update weights
        for i = 1:numlayers-1
            delta_weights{i} = momentum .* delta_weights{i} + ...
                gradient_weights{i} ./ mbsz - wc * weights{i};
            weights{i} = weights{i} - lrs(ep) * delta_weights{i};
        end

        % update bias
        for i = 1:numlayers-2
           delta_bias{i} = momentum .* delta_bias{i} + gradient_bias{i} ./ mbsz;
           bias{i} = bias{i} - lrs(ep) * delta_bias{i};
        end
        
        % VALIDATE.
        if mod(m, showValidCEAfter) == 0
            fprintf(1, 'epoch %d, batch %d: ...', ep, m);
            if OctaveMode, fflush(1); end
            CE = valid_data(valid, weights, bias);
            fprintf(1, ' Validation CE %.3f\n', CE);
            if OctaveMode, fflush(1); end
        end
        
    end
    fprintf(1, 'Moving Avg. Training CE %.3f\n', trainset_CE);
    
    % EVALUATION ON TRAINING AND VALIDATION SET
    fprintf(1, 'Running evaluation (epoch %d): ...\n', ep);
    
    % EVALUATE ON TRAINING SET.
    if OctaveMode, fflush(1); end
    %trainset_CE = eval_data(train, weights, bias);
    perc = 0.1; perm = 1;
    trainset_CE = eval_data(train, weights, bias, perc, perm);
    fprintf(1, ' Training CE %.3f (%.2f%%)\n', trainset_CE, perc*100);
    hist_trainset_CE(ep) = trainset_CE;
    if OctaveMode, fflush(1); end
    
    % EVALUATE ON VALIDATION SET.
    
    if OctaveMode, fflush(1); end
    validset_CE = eval_data(valid, weights, bias);
    fprintf(1, ' Validation CE %.3f (100%%)\n', validset_CE);
    hist_validset_CE(ep) = validset_CE;
    if OctaveMode, fflush(1); end
    
    % check early termination
    if validset_CE > best_validset_CE
        fprintf(1, ['Validation error increasing! Training stopped. ', ...
            'Returning weights after epoch %d.\n'], ep);
        break
    end
    
    % save current model as final
    last_trainset_CE = trainset_CE;
    best_validset_CE = validset_CE;
    weightsFinal = weights;
    biasFinal = bias;
    
    % obtain the running time of this epoch
    tElapsed(ep) = toc(tStart);
    
end

% print final trainset CE
fprintf(1, 'Finished Training.\n');
if OctaveMode, fflush(1); end
fprintf(1, 'Final Training CE %.3f (%.2f%%)\n', last_trainset_CE, perc*100);

% EVALUATE ON TEST SET.
fprintf(1, 'Running test ...\n');
if OctaveMode, fflush(1); end
testset_CE = eval_data(test, weightsFinal, biasFinal);
fprintf(1, 'Final Test CE %.3f (100%%)\n', testset_CE);
if OctaveMode, fflush(1); end

% save model
model.weights = weightsFinal;
model.bias = biasFinal;
model.vocab = vocab;
model.targetIdx = targetIdx;
model.lr = lrs;
model.CE = [last_trainset_CE, best_validset_CE, testset_CE];
model.hist_CE = [hist_trainset_CE, hist_validset_CE];
model.compcost = tElapsed;

% save meta data to model
model.meta.epochs = epochs;
model.meta.mbsz = mbsz;
model.meta.lr = lr;
model.meta.momentum = momentum;
model.meta.wc = wc;
model.meta.numemb = numemb;
model.meta.numhid2 = numhid2;
model.meta.w0 = w0;
model.meta.lrDecayAfter = lrDecayAfter;
model.meta.halving = halving;
model.meta.datafile = datafile;
model.meta.numwords = numwords;

% sort fields in model.meta
model.meta = orderfields(model.meta);

% stop clock
if OctaveMode
    end_time = time;
    diff = end_time - start_time;
else
    end_time = clock;
    diff = etime(end_time, start_time);
end
fprintf(1, 'Training took %.2f seconds\n', diff);

end

function [prob, rank] = seq_probability(seqs, model, candidates, show_result)
% find the sequence probability p(w_1, w_2, ..., w_N)
% build on top of predict_target_word.m
%
% Zhenhao (Roger) Ge, 2015-08-08

if nargin < 4, show_result = 1; end
if nargin < 3, candidates = {}; end

% find target word index
I = model.targetIdx;

% find parameters
numemb = size(model.weights{1},2);
numhid1 = size(model.weights{2}, 1);
numwords = numhid1 / numemb;
numdims = numwords + 1;
len_seqs = length(seqs);

% sanity check on the matching of model word dimension and seq. length 
if len_seqs ~= numdims
    fprintf('model word dimension: %d\n', numdims);
    fprintf('seq. length: %d\n', len_seqs);
    error('seq. length must match the model word dimension!');
end

% seperate target word from the input word sequence
idx = logical(1:numdims);
idx(I) = 0;
input_words = seqs(idx);
target_word = seqs(I);

% map target word to <unk> if it is not in vocabulary
if sum(strcmp(model.vocab, target_word)) == 0
   target_word = '<unk>'; 
end

% if candidate list is non-empty, then target word must be in it
if ~isempty(candidates)
   id = strcmp(candidates, target_word);
   if sum(id) == 0
      fprintf('target word %s is not in candidate list!\n', target_word);
      return
   end
end

% set default value of the number of output predicted target words
k = Inf;

% predict the target word
close_vocab = 1;
show_func_result = 0;
[predicted, p] = predict_target_word(input_words, model, k, ...
    candidates, close_vocab, show_func_result);

% find the probability corresponding to the target word
rank = find(strcmp(predicted, target_word)==1);
prob = p(rank);

% show result if needed
if show_result == 1
    seqs2 = seqs;
    seqs2{I} = ['(', seqs{I}, ')'];
    fprintf('%s | Prob: %.5f\n', strjoin(seqs2, ' '), prob);
end

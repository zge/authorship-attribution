function [predicted, p] = predict_target_word(words, model, k, ...
    candidates, close_vocab, show_result)
% Predicts the target word.
% Inputs:
%   words: Input words as strings.
%   model: Model returned by the training script.
%   I: target word index
%   k: The k most probable predictions are shown.
% Outputs:
%   predicted: predicted target words
%   p: probabilities of the predicted target words
% Example usage:
%   predict_next_word('john', 'might', 'be', model, 3);
%   predict_next_word('life', 'in', 'new', model, 3);

if nargin < 6, show_result = 1; end
if nargin < 5, close_vocab = 1; end 
if nargin < 4, candidates = {}; end
if nargin < 3, k = Inf; end

vocab = model.vocab;
if isempty(candidates), candidates = vocab; end
numwords = length(words);
numdims = numwords + 1;
ids = zeros(numwords, 1);
for i = 1:numwords
    if ~isempty(words{i}) % skip missing (empty) word(s)
        id = find(strcmp(words{i}, vocab));
        if ~any(id) % find OOVs
            fprintf('Word ''%s\'' not in vocabulary.\n', words{i});
            if close_vocab == 1 % close vocabulary problem
                fprintf('Convert ''%s\'' to <unk> ...\n', words{i});
                idx_unk = find(strcmp(model.vocab,'<unk>')==1, 1);
                if ~isempty(idx_unk) % <unk> in vocabulary
                ids(i) = idx_unk;
                else
                    fprintf('There is no <unk> in vocabluary!\n');
                    return
                end
            else
                return
            end
        else
            ids(i) = id;
        end
    end
end

% keep only the candidates that in the vocabulary
set_diff = setdiff(candidates, vocab);
if ~isempty(set_diff)
    fprintf(['Warning: following candidates are not in vocabulary, ', ...
        'so removed:\n']);
    fprintf([strjoin(set_diff, ' '), '\n']);
    candidates = setdiff(candidates, set_diff);
end

v = length(candidates);
if v == 1
   warning(['Only 1 word in candidate list, ', ...
     'so the output probability after softmax must be 1, which is useless!']); 
end
if k > v
    k = v;
%     fprintf('There are %d predictions: \nInput: %s, target index: %d\n', ...
%         k, strjoin(words), model.targetIdx),
end

[~, cand_idx] = intersect(vocab, candidates);
vocab_short = vocab(cand_idx);

% compute sorted probabilities with their corresponding word indices
states = fprop(int32(ids), model.weights, model.bias, cand_idx);
% states = fprop_backup(ids, model.weights, model.bias);
[prob, indices] = sort(states{end}, 'descend');
can_sorted = vocab_short(indices);

% create sentence except the target word
sentence = cell(1, numdims);
I = model.targetIdx;
for i = 1:numdims
   % fill in input words 
   if i < I
       sentence{i} = words{i};
   elseif i > I
       sentence{i} = words{i-1};
   end
   % deal with missing input word
   if i ~= I && isempty(sentence{i})
      sentence{i} = '*'; 
   end
end

predicted = can_sorted(1:k)';
% predicted = vocab(indices(1:k))';
p = prob(1:k);

% show result if needed
if show_result == 1
    fprintf('show first %d of %d candidates\n', k, v);
    for i = 1:k
        sentence{I} = ['(', predicted{i} , ')'];
        fprintf('%s | Prob: %.5f\n', strjoin(sentence, ' '), p(i));
    end
end

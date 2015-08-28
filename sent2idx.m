function d = sent2idx(sentences, vocab, rm_sil)
% prepare the dev/test data
% convert sentences from sequences of actual words to sequences of word
% indices
% map words uncovered in vocab as <unk> (with the largest word index)
%
% Zhenhao (Roger) Ge, 2015-08-21

% set default parameter
if nargin < 3, rm_sil = 1; end

% sentences should be 2-layer cells
if ischar(sentences{1})
   sentences = {sentences}; 
end

% get the # of words, i.e. the index for <unk>
% idx_unk = find(strcmp(vocab, '<unk>')==1, 1);
nwords = length(vocab);

% remove end-of-silence mark
if rm_sil == 1
    sentences = cellfun(@(s) s(1:end-1), sentences, 'UniformOutput', 0);
end

% convert word to its index in double-loop
num_sent = length(sentences);
d = cell(num_sent, 1);
for i = 1:num_sent
    len_sent = length(sentences{i});
    d{i} = int32(zeros(1, len_sent));
    for j = 1:len_sent
        d{i}(j) = mod(word2idx(vocab, sentences{i}{j})-1, nwords) + 1;
    end
end
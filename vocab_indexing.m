function [vocab, freq, word_idx] = vocab_indexing(sentences, varargin)
% Index vocabulary with data in sentences
% Inputs:
%   sentences: text sentences
%   ordertype (natural): if we need to randomize the order of sentences, or
%   sort the sentences in ascend/descend order
%   count_lim (0): require word to be counted more than 'count_lim' of
%   times, in order to add into vocabulary
%   vsize_lim (Inf): upper bound of vocabulary size
%   freq_lim (0): lower bound of word frequency limit
% Outputs:
%   vocab: sorted vocabulary (not contain end_sil mark)
%   freq: word freq of words in vocab (not contain end_sil mark)
%   word_idx [size N X 1, value range: 1~V]: vocabulary word index for
%   each word in the raw word list
%
% Zhenhao (Roger) Ge, 2015-08-19 

% set default parameters
[ordertype, count_lim, vsize_lim, freq_lim] = process_options(varargin, ...
    'ordertype', 'natural', 'count_lim', 0, 'vsize_lim', Inf, 'freq_lim', 0);

% sort sentences if needed
if strcmp(ordertype, 'ascend') || strcmp(ordertype, 'descend')
   sentences = sort(sentences, ordertype); 
end

% concatentate sentences
sent_cat = [sentences{:}]';
N = length(sent_cat);  % total number of word tokens

% obtain vocabulary and word indices
[vocab, ~, word_idx] = unique(sent_cat);

% count word appearance (1gram) and get word frequency
count = zeros(size(vocab));
for i = 1:N
    idx = strcmp(vocab, sent_cat{i});
    count(idx) = count(idx) + 1;
end
freq = count/N;

% combine count limit with freq. limit
freq_lim = max(count_lim/N, freq_lim);
% combine freq. limit with vocab. size limit
vsize_lim = min(sum(freq>freq_lim), vsize_lim);

% prune vocabulary based on vocab. size limit and update freq and word
% indices
freq_sorted = sort(freq, 'descend');
freq_lim_new = freq_sorted(vsize_lim);
flag_freq = freq>=freq_lim_new;
vocab_new = vocab(flag_freq);
freq_new = freq(flag_freq);

% suppress 'dead' words in word_idx (assign 0 idx) to create word_idx_new

% method 1
% word_idx_new = zeros(N,1);
% for i = 1:N
%    if flag_freq(word_idx(i)) == 1
%        word_idx_new(i) = word_idx(i);
%    end
% end

% method 2
word_idx_new = zeros(N,1);
survive = flag_freq(word_idx) == 1;
word_idx_new(survive) = word_idx(survive);

% map old idx to new idx for 'survived' words

% % method 1
% tmp1 = find(flag_freq==1);
% word_idx_new2 = zeros(N, 1);
% for i = 1:N
%     if word_idx_new(i) > 0 
%         word_idx_new2(i) = find(tmp1==word_idx_new(i));
%     end
% end

% method 2
[~, ~, ic] = unique(word_idx_new);
min_idx = min(word_idx_new);
word_idx_new2 = ic + min_idx - 1;

% append new word '<unk>'
if min_idx == 0
    vocab_new = [vocab_new; '<unk>'];
    freq_new = [freq_new; sum(word_idx_new2==0)/N];
    word_idx_new2(word_idx_new2==0) = vsize_lim + 1;
end

% update the orginal parameters
vocab = vocab_new;
freq = freq_new;
word_idx = word_idx_new2;
V = length(vocab);

% option to randomize words in vocabulary
if strcmp(ordertype, 'random')
    
    % randonmize vocabulary with its frequencies and indices
    ind_rand = randperm(V);
    vocab_rand = vocab(ind_rand);
    freq_rand = freq(ind_rand);
    [~, ind_map] = sort(ind_rand);
    % new_ind = ind_map(old_ind);
    word_indices_rand = ind_map(word_idx);
    
    % sanity check by print out the first 10 words with both regular and
    % rand vocabularies
    
    % print out the first 10 words in content using sorted vocabulary
    fprintf('first 10 words in sorted vocabulary ...\n');
    fprintf([strjoin(vocab(word_idx(1:10))', ' '), '\n']);
    
    % print out the first 10 words in contenct using rand vocabulary
    fprintf(['first 10 words in random vocabulary ', ...
        '(without silence mark removal) ...\n']);
    fprintf([strjoin(vocab_rand(word_indices_rand(1:10))', ' '), '\n']);
    
    % update the vocabulary from regular (alphabet sorted) to random
    vocab = vocab_rand;
    freq = freq_rand;
    word_idx = word_indices_rand;
    
    clear vocab_rand freq_rand word_indices_rand
    
end

% deal with end-of-sentence silence mark (</s>)
% assign word index -1 to </s> and update vocab and freq
end_sil = '</s>';
idx_sil = find(strcmp(vocab, end_sil)==1, 1);
if ~isempty(idx_sil)
    idx_range = 1:V;
    word_idx(word_idx==idx_sil) = -1;
    if idx_sil ~= V
        idx = word_idx > idx_sil;
        word_idx(idx) = word_idx(idx) - 1;
    end

    % update vocabulary and frequency list   
    vocab = vocab(idx_range~=idx_sil);
    freq = freq(idx_range~=idx_sil);   
    % V = V - 1;

    % re-normalize word frequency after removing end_sil mark
    freq = freq ./ sum(freq);
    
    % sanity check after dealing with silence mark
    % print out the first 10 words in contenct using rand vocabulary
    fprintf('first 10 words in random vocabulary ...\n');
    strs = cell(1, 10);
    for i = 1:10
        if word_idx(i) ~= -1
            strs{i} = vocab{word_idx(i)};
        else
            strs{i} = '</s>';
        end
    end
    fprintf([strjoin(strs, ' '), '\n']);

end

word_idx = int32(word_idx);
    
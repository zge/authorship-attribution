function data = gen_data(directory, datafile, varargin)
% generate data
% Options:
%  - add_sil (1): flag to add end-of-silience mark, 1 (add), or 0 (noadd)
%  - count_lim (1): word occurance count, words with > count_lim will be
%  added into vocabulary
%  - freq_lim (1/100000, i.e. log10(freq_lim) = -5): word freq. count,
%  freq.(word) with > freq_lim will be added into vocabulary
%
% Zhenhao (Roger) Ge, 2015-08-21
% Zhenhao (Roger) Ge, 2015-08-26, modified the default options

[add_sil, count_lim, freq_lim] = process_options(varargin, 'add_sil', ...
    1, 'count_lim', 0, 'freq_lim', 0);

[~, dataname] = fileparts(datafile);
underscores = strfind(dataname, '_');
course_instructor = dataname(1:underscores(1)-1);
seed_int = str2double(dataname(underscores(2)-2:underscores(2)-1));
ngram = str2double(dataname(underscores(2)+1));

% % find all data sets for the selected course
% datasets = getfile(directory.split, 'txt');
% idx = cellfun(@(i) ~isempty(i), strfind(datasets, course_instructor));
% datasets = datasets(idx);

% get train, valid, test datasets
setnames = {'train', 'valid', 'test'};
for i = 1:length(setnames)
    
    % extract sentences from raw file
    dataset.(setnames{i}) = [course_instructor, '_', setnames{i}, ...
        '_rand', num2str(seed_int, '%02d'), '.txt'];
    sentences.(setnames{i}) = extract_sentences([directory.split, '\', ...
        dataset.(setnames{i})], add_sil);
end

% vocabulary indexing with pruning for train set
disp('vocab. indexing with pruning for train set ...')
[vocab, freq, word_idx] = vocab_indexing(sentences.train, ...
    'count_lim', count_lim, 'freq_lim', freq_lim);

% prepare the ngram-wise train dataset
data.train = prep_ngram(word_idx, ngram);

% prepare the sentence-wise valid/test data
sent_dev = sent2idx(sentences.valid, vocab);
sent_test = sent2idx(sentences.test, vocab);

% prepare the ngram-wise valid/test data
sent_dev_sil = cellfun(@(s) [s, int32(-1)], sent_dev, 'UniformOutput', 0);
sent_test_sil = cellfun(@(s) [s, int32(-1)], sent_test, 'UniformOutput', 0);
word_idx_dev = horzcat(sent_dev_sil{:})';
word_idx_test = horzcat(sent_test_sil{:})';
data.valid = prep_ngram(word_idx_dev, ngram);
data.test = prep_ngram(word_idx_test, ngram);

% save train vocab/freq to data
data.vocab = vocab;
data.freq = freq;
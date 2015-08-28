% Main script to test porterStemmer.m
% It stem word by word
%
% Zhenhao (Roger) Ge, 2015-08-10

% read in text file
fid = fopen('stem.txt', 'r');
text = textscan(fid, '%s', 'delimiter', '\n');
fclose(fid);

% lower down case since it does not handeled inside the function
text = lower(text{1});

% split sentences into words
words = regexp(text{1}, ' ', 'split');

% stemmization
% nwords = length(words);
words2 = cellfun(@porterStemmer, words, 'UniformOutput', 0);


function word = idx2word(vocab, idx)
% find word in vocabulary given word index
% word = '' if the word index is out of vocabulary boundary

V = length(vocab);
if idx > V
    fprintf(1, 'Index %d is out of vocabulary boundaru.\n', idx);
    word = '';
else
    word = vocab{idx};
end
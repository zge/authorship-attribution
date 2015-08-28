function idx = word2idx(vocab, word, msg)
% find word index in vocabulary given word
% idx = 0 if the word is not in vocabulary

% set the default value of message showing flag
if nargin < 3, msg = 0; end

id = strcmp(word, vocab);
if ~any(id)
    if msg == 1
        fprintf(1, 'Word ''%s\'' not in vocabulary.\n', word);
    end
    idx = 0;
else
    idx = find(id, 1);
end


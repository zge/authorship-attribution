function distance = word_distance(word1, word2, model)
% Shows the normalized L2 distance between word1 and word2 in the
% word_embedding_weights.
% Inputs:
%   word1: The first word as a string.
%   word2: The second word as a string.
%   model: Model returned by the training script.
% Example usage:
%   word_distance('school', 'university', model);

vocab = model.vocab;

id1 = find(strcmp(word1, vocab));
id2 = find(strcmp(word2, vocab));
if ~any(id1)
  fprintf(1, 'Word ''%s\'' not in vocabulary.\n', word1);
  return;
end
if ~any(id2)
  fprintf(1, 'Word ''%s\'' not in vocabulary.\n', word2);
  return;
end

word_rep1 = model.weights{1}(id1, :);
word_rep2 = model.weights{1}(id2, :);
diff = word_rep1 - word_rep2;
numemb = length(diff);
distance = sqrt(sum(diff .* diff) / numemb);

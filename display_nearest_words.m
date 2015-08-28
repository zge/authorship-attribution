function [nearest, dist] = display_nearest_words(word, model, k)
% Shows the k-nearest words to the query word by their normalized
% distances.
% Inputs:
%   word: The query word as a string.
%   model: Model returned by the training script.
%   k: The number of nearest words to display.
% Outputs:
%   nearest: nearest words
%   dist: corresponding distance of the nearest words
% Example usage:
%   display_nearest_words('school', model, 10);

vocab = model.vocab;
id = strcmp(word, vocab);
if sum(id) == 0
  fprintf(1, 'Word ''%s\'' not in vocabulary.\n', word);
  return;
end

% Compute distance to every other word.
V = length(vocab);
word_rep = model.weights{1}(id, :);
diff = model.weights{1} - repmat(word_rep, V, 1);

% get embedding space dimension for normalization so distance from
% different embedding space can be compared
numemb = length(word_rep); 
% compute the normalized distance (dimension normalization should be inside
% sqrt)
distance = sqrt(sum(diff .* diff, 2) / numemb);

% Sort by distance.
[~, order] = sort(distance);

% % approach 1 (slighly slower)
% for i = 2:k+1 % The nearest word is the query word itself, skip that.
%   fprintf('%s %.2f\n', vocab{order(i)}, d(i));
% end

% approach 2
order = order(2:k+1);  % The nearest word is the query word itself, skip that.
nearest = vocab(order(1:k))';
dist = distance(order(1:k));
for i = 1:k
    fprintf('%s %.3f\n', nearest{i}, dist(i));
end



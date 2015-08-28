function d = prep_ngram(word_indices, numdims, randomize)
% Prepare the training data
% generate phrases with length specified in 'numdims' without silence
% data format: d [numdims X numsamples], i.e. one gram/phrase per column
%
% Parent: gen_data.m
% 
% Zhenhao (Roger) Ge, 2015-08-20

if nargin < 3, randomize = 0; end

% obtain phrase data
N = length(word_indices);
d = int32(zeros(N+numdims-1, numdims));
for i = 1:numdims
    d(numdims-i+1:numdims-i+N,i) = int32(word_indices);
end
d = d(numdims:N,:);

% remove phrases with silence tags
nosil = logical(1-sum(d==-1, 2)>0);
d = d(nosil, :)';

% randomize data
if randomize == 1
    sel = randperm(size(d, 2));
    d = d(:, sel);
end




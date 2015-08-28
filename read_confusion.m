function confusion = read_confusion(filename)
% read the confusion matrix from csv file
%
% Zhenhao (Roger) Ge, 2015-08-26

M = csvread(filename);

group_length = max(M(:,1));
ngram = max(M(:,2));
num_set = size(M,2)-2;
confusion = zeros(group_length, ngram, num_set);

for i = 1:group_length
    for j = 1:ngram
        idx = (i-1)*ngram + j;
        confusion(i,j,:) = M(idx, 3:end);
    end
end

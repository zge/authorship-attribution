function nbest = read_nbest(filename)
% read nbest from csv file, nbest(i,j,:) is the nbest of i-gram and jth course
% with various group length
%
% Zhenhao (Roger) Ge, 2015-08-26

M = csvread(filename);
num_set = max(M(:,1));
ngram = max(M(:,2));
group_length = size(M,2)-2;
nbest = zeros(ngram, num_set, group_length);

for i = 1:ngram
    for j = 1:num_set
        idx = (j-1)*ngram + i;
%         disp(['set: ', num2str(j), ', gram: ', num2str(i), ...
%             ', idx: ', num2str(idx)]);
        nbest(i,j,:) = M(idx,3:end); 
    end    
end
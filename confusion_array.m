function conf_array = confusion_array(ppl)
% get the min of each row and indicate the confusion matrix 1 for the index
% of min, o.w. 0
% then, sum up w.r.t rows and normalize to get the prob. 
% return confusion array as a row vector
%
% Zhenhao (Roger) Ge, 2015-08-25

[num_sent, num_set] = size(ppl);

conf_mtx = zeros(num_sent, num_set);
for i = 1:num_sent
    [~, idx] = min(ppl(i,:));
    conf_mtx(i,idx) = 1;
end

conf_array = sum(conf_mtx) / num_sent;

function nbest = nbest_accuracy(ppl, target)
% find the avg. nbest accuracy among all sentences

[num_sent, num_set] = size(ppl);
pos = zeros(num_sent, 1);
for i = 1:num_sent
   s = ppl(i,:);
   [~, idx] = sort(s);
   pos(i) = find(idx==target, 1);
end

nbest = zeros(1, num_set);
for i = 1:num_set
   nbest(i) = sum(pos<=i) / num_sent; 
end

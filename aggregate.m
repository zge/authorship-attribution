function ppl_agg = aggregate(ppl, gl)
% group sentence-level ppl with specified group length (gl)
%
% Zhenhao (Roger) Ge, 2015-08-25

[num_sent, num_set] = size(ppl);
ppl_agg = zeros(num_sent-gl+1, num_set);
for i = 1:(num_sent-gl+1)
    ppl_agg(i,:) = sum(ppl(i:i+gl-1,:), 1);
end
function CE = eval_data(data, weights, bias, perc, perm)
% Evaluate data by computing cross entropy through forward propation
% Inputs:
%   perc: percent to use for evaluation
%   perm: flag for random permutation (1 for randomization, 0 otherwise)

% set default parameter values
if nargin < 5, perm = 0; end
if nargin < 4, perc = 1; end

% initialize parameters
V = length(bias{end});
% expansion_matrix = eye(V);
tiny = exp(-30);

% get the # of mini-batch
dims = size(data.input);
if length(dims) == 2
    M = 1;
else
    M = dims(3); 
end

% get the # of samples to keep in mini-batch
mbsz = dims(2);
K = round(mbsz*perc);    

% randomize the order if needed

if perm == 1
   sel = randperm(mbsz); 
else
   sel = 1:mbsz; 
end

CEs = zeros(M, 1);
for i = 1:M
    
    % split to mini-batch
    data_batch.input = data.input(:,sel(1:K),i);
    data_batch.target = data.target(:,sel(1:K),i);
    
    % forward propagation
    states = fprop(data_batch.input, weights, bias);
    
    % compute cross entropy
    % expanded_data_target = expansion_matrix(:, data_target);
    expandedDataTarget = sparse(double(data_batch.target), 1:K, 1, V, K);
    CEs(i) = full(-sum(sum(expandedDataTarget .* log(states{end} + tiny)))) / K;
    
end
CE = mean(CEs);



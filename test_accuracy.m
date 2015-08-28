function [accuracies, prob, indices] = test_accuracy(data, model, k)
% Test the k-best accuracy.
% Inputs:
%   data: Data batch with 4 word indices per column.
%   model: Model returned by the training script.
%   k: The k-best accuracies are shown.
% Example usage: 
%   accuracies = test_accuracy(data.testData, model_default, 5);

% get the mini-batch size and the # of mini-batches
[~, mbsz, M] = size(data.input);
dbsz = mbsz * M;

show_progress_after = 5000;

prob = zeros(k, mbsz, M);
indices = zeros(k, mbsz, M);
for i = 1:M
    
    for j = 1:mbsz
        
        states = fprop(data.input(:,j,i), model.weights, model.bias);
        [prob_all, indices_all] = sort(states{end}, 'descend');
        prob(:, j, i) = prob_all(1:k);
        indices(:, j, i) = indices_all(1:k);
        
        numsamples = (i-1)*mbsz+j;
        if mod(numsamples, show_progress_after) == 0
            fprintf(1, 'completed %d out of %d samples\r', numsamples, dbsz);
        end
        
    end
    
end

flag = indices == double(repmat(data.target, k, 1));
flag = reshape(flag, k, []);
accuracies = sum(cumsum(flag),2) / dbsz;



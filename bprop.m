function [CE, gradient_weights, gradient_bias] = bprop(input_batch, ...
            target_batch, states, weights)
% function [CE, word_embedding_weights_gradient, embed_to_hid_weights_gradient, ...
%     hid_to_output_weights_gradient, hid_bias_gradient, output_bias_gradient] ...
%     = bprop(input_batch, target_batch, embedding_layer_state, embed_to_hid_weights, ...
%     hidden_layer_state, hid_to_output_weights, output_layer_state)

% Outputs:
%   hid_to_output_weights_gradient (gradient_weights{end})
%
%   output_bias_gradient (gradient_weights{end})

% get dimensions
ny = length(states);
nw = length(weights);

% initialize parameters
numlayers = length(states) + 1;
[V, mbsz] = size(states{ny});
numwords = size(input_batch, 1);
numhid1 = size(states{1}, 1);
numemb = numhid1 / numwords;
tiny = exp(-30);

% COMPUTE DERIVATIVE.
% Expand the target to a sparse 1-of-K vector.
expanded_target_batch = sparse(double(target_batch), 1:mbsz, 1, V, mbsz);
% Compute derivative of cross-entropy loss function.
% dC/dZ_output = y_output - t_output
error_deriv = states{ny} - expanded_target_batch;

% MEASURE LOSS FUNCTION.
% C = -sum_output [t_output*log(y_output)]
CE = full(-sum(sum(expanded_target_batch .* log(states{ny} + tiny)))) / mbsz;

gradient_weights = cell(numlayers-1, 1);
gradient_bias = cell(numlayers-2, 1);
back_prop_deriv = cell(numlayers-2, 1);
% BACK PROPAGATE.

%% OUTPUT TO HIDDEN.
% dC/dw_{hid-out} = y_hid * dC/dz_out [e.g. size 200 X 250]
gradient_weights{ny} =  states{ny-1} * error_deriv';
% dC/db_out = dC/dz_out * dz_out/db_out = dC/dz_out [e.g. size 250 X 1]
gradient_bias{nw} = sum(error_deriv, 2);

% dC/dy_hid = sum_out[w_{hid-out} * dC/dz_out]
tmp1 = weights{nw} * error_deriv;
% dy_hid/dz_hid = y_hid * (1-y_hid), since hidden layer is logistic
tmp2 = states{ny-1} .* (1 - states{ny-1});
% dC/dz_hid = (dC/dy_hid).*(dy_hid/dz_hid)
back_prop_deriv{1} = tmp1 .* tmp2;

%% HIDDEN TO HIDDEN.

% perform when there are more than 1 hidden layer (the other three layers
% are word, embedding and output layers)
if numlayers > 4 
    for i = 2:numlayers-3
        % dC/dw_{hid(lower)-hid(upper)} = y_hid(lower) * dC/dz_hid(upper)
        gradient_weights{ny-i+1} = states{ny-i} * back_prop_deriv{i-1}';
        % dC/db_hid(upper) = dC/dz_hid(upper) * dz_hid(upper)/db_hid(upper)
        % = dC/dz_hid(upper)
        gradient_bias{nw-i+1} = sum(back_prop_deriv{i-1}, 2);
        
        % dC/dy_hid(lower) = sum_out[w_{hid(lower)-hid(upper)} * dC/dz_hid(upper)]
        tmp1 = weights{nw-i+1} * back_prop_deriv{i-1};
        % dy_hid(lower)/dz_hid(lower) = y_hid(lower) * (1-y_hid(lower)),
        tmp2 = states{ny-i} .* (1 - states{ny-i});
        % dC/dz_hid(lower) = (dC/dy_hid(lower)).*(dy_hid(lower)/dz_hid(lower))
        back_prop_deriv{i} = tmp1 .* tmp2;
    end
end

%% HIDDEN TO EMBEDDING.
% dC/dw_{emb-hid} = y_emb * dC/dz_hid
gradient_weights{2} = states{1} * back_prop_deriv{nw-1}';
% dC/db_hid = dC/dz_hid * dz_hid/db_hid = dC/dz_hid
gradient_bias{1} = sum(back_prop_deriv{nw-1}, 2);

% dC/dy_emb = sum_hid[w_{emb-hid} * dC/dz_hid]
tmp1 = weights{1} * back_prop_deriv{nw-1};
% dy_emb/dz_emb = 1, since embedding layer is linear
tmp2 = 1;
back_prop_deriv{nw} = tmp1 .* tmp2;

%% EMBEDDING TO WORD.
% similar to previous two weights gradient, but need to sum over the
% number of words - zge
gradient_weights{1} = zeros(V, numemb);
for w = 1:numwords
    % obtain y_word
%     expanded_input_batch = expansion_matrix(:, input_batch(w, :));
    expanded_input_batch = sparse(double(input_batch(w, :)), 1:mbsz, 1, ...
    V, mbsz);
    % add dC/d_{word-emb} = y_word * dC/dy_emb word by word
    gradient_weights{1} = gradient_weights{1} + expanded_input_batch * ...
        (back_prop_deriv{nw}(1 + (w-1)*numemb : w*numemb, :)');
end
clear expanded_input_batch
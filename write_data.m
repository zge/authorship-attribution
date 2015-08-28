function write_data
% write data (train/test/validate) and vocabulary (250 words) into
% seperated csv files
%
% Zhenhao (Roger) Ge, 2015-07-27

% load data
load data.mat;

% write data iteratively
fnames = fieldnames(data);
for i = 1:length(fnames)
    name = [fnames{i}, '.csv'];
    if iscell(data.(fnames{i}))
        cell2csv(name, data.(fnames{i})');
    else
        cell2csv(name, num2cell(data.(fnames{i})'));
    end
end

function sentences = extract_sentences(rawfile, add_sil)
% Extract sentences from raw file
%
% Zhenhao (Roger) Ge, 2015-08-07

% set default parameters
if nargin < 2, add_sil = 0; end

% open raw file
fid = fopen(rawfile);
if fid == -1
    error([rawfile, 'does not exist!']);
end

% get raw text
fprintf('read in raw text from %s\n', rawfile)
text = textscan(fid, '%s', 'delimiter', '\n');
text = text{1};

% lower case
text = lower(text);

% add end of sentence silence tag
if add_sil 
    end_sil = '</s>';
    text = cellfun(@(str) [str, ' ', end_sil], text, 'UniformOutput', 0);
end

% get sententences in cells
% fprintf('get sentences from raw text ...\n');

% simple one-line command
sentences = cellfun(@(str) strsplit(str), text, 'UniformOutput', 0);

% alternatively
% nlines = length(text);
% sentences = cell(nlines, 1);
% for i = 1:nlines
%     
%     % show progress
%     if mod(i, 10000) == 0
%        fprintf('%d lines processed ...\r', i); 
%     end
%     
%     sentences{i} = strsplit(text{i});
% end

% close raw file
fclose(fid);

end
function data = dataprep(rawfile, numdims)

% open raw file
fid = fopen(rawfile);
if fid == -1
    error([rawfile, 'does not exist!']);
end

% set flag to show progress after processing 1000 lines of raw file
show_progress_after = 1000;

% initialize phrases and vocabulary
phrases = [];
vocab = [];

tline = fgets(fid);
numlines = 1;
while ischar(tline)
    
    % disp line content
    % disp(tline)
    
    if mod(numlines, show_progress_after) == 0
        fprintf(1, '%s: %d lines processed ...\r', rawfile, numlines);
    end
    
    % split line into words
    sentence = strsplit(tline, ' ');
    % remove empty cells
    sentence = sentence(~cellfun('isempty', sentence));
    % dump words into vocabulary
    vocab = [vocab; sentence'];
    
    numwords = length(sentence);
    if numwords >= numdims
        for i = 1: numwords-numdims+1
            phrases = [phrases, sentence(i:i+numdims-1)'];
        end
    end
    
    tline = fgets(fid);
    numlines = numlines + 1;

end

% close raw file
fclose(fid)

save('raw1.mat', 'vocab', 'phrases');

vocab2 = vocab(~cellfun('isempty', vocab));
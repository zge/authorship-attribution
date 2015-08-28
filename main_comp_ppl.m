% Main script to compute the perplexity of a test dataset, givin the
% corresponding NPLMs, looped over courses
%
% Zhenhao (Roger) Ge, 2015-08-25

%% Initialization

% set up directory
directory.work = pwd;
directory.root = fileparts(directory.work);
directory.data = [directory.root, '\data'];
directory.stem = [directory.data, '\stem'];
directory.split = [directory.data, '\split'];
directory.mat = [directory.data, '\mat'];
directory.lm = [directory.data, '\lm'];

% find data file names
textnames = getfile(directory.stem, 'txt');
num_sets = length(textnames);

% set dataset parameters
ngram = 4; % specify grams (# of context + target wrods)
num_seeds = 10;

% set other parameters
setnames = {'train', 'valid', 'test'};
add_sil = 0; % flag of adding silence in sentence extraction
rm_sil = 0; % flag of removing silence after sentence extraction

%% Perplexity cmputation

ppl = zeros(num_sets, num_seeds);
timecost = zeros(num_sets, num_seeds);
show_result = 0;
for i = 1:num_sets
    
    % specify set index
    set_idx = i;
    
    % get course and instructor info
    textname = textnames{set_idx};
    [~, course_instructor] = fileparts(textname);
    
    for j = 1:num_seeds
        
        % display progress
        disp([' *** processing set: ', num2str(i),', seed: ', ...
            num2str(j-1, '%02d'), ' ... ***']),
        
        % specify seed
        seed_int = j - 1;
        
        % start timer
        tStart = tic;

        % load the trained NPLM
        file.lm = [directory.lm, '\', course_instructor, '_rand', ...
            num2str(seed_int, '%02d'), '_', num2str(ngram), 'gram_lm.mat'];
        load(file.lm),
        
        % extract sentences from corresponding test file
        file.test = [directory.split, '\', course_instructor, '_test_rand', ...
            num2str(seed_int, '%02d'), '.txt'];
        sentences = extract_sentences(file.test, add_sil);

        % compute the local ppl for sentences in current test set
        num_sent = length(sentences);
        ppls = zeros(num_sent, 1);
        logprobs = zeros(num_sent, 1);
        N = zeros(num_sent, 1);
        for k = 1:num_sent
            [ppls(k), logprobs(k), N(k)] = seq_ppl(sentences{k}, ...
                model, show_result);
        end
        
        % get avg. ppl for current test set
        ppl(i,j) = 10 ^ (-sum(logprobs)/sum(N));
        
        % log time cost
        timecost(i,j) = toc(tStart);
        
    end
    
end

% save results
save('main_comp_ppl', 'ppl', 'timecost');
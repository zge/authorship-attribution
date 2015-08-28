% Main script to get classification of Authorship Attribution (AA) project
% using NPLM
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
directory.result = [directory.root, '\results'];

% find data file names
textnames = getfile(directory.stem, 'txt');
num.set = length(textnames);

% set dataset parameters
num.gram = 4; % specify grams (# of context + target wrods)
num.seed = 5;

% set other parameters
add_sil = 0; % flag of adding silence in sentence extraction
rm_sil = 0; % flag of removing silence after sentence extraction
group_len_max = 50;
refresh = 0;

%% classification over course and seed

show_result = 0;
for i = 1:num.set
    
    % specify set index
    set_idx = i;
    
    % get course and instructor info
    textname = textnames{set_idx};
    [~, course_instructor] = fileparts(textname);
    
    for j = 1:num.seed
        
        % specify seed
        seed_int = j - 1;
        
        % specify the ppl file
        file.score = [directory.result, '\scores\', course_instructor, ...
            '_test_rand', num2str(seed_int, '%02d'), '.mat'];
        
        if exist(file.score, 'file')
            
            disp(['loading existed ', file.score, ' ...']),
            load(file.score);
            
        else
            
            disp(['creating new ', file.score, ' ...']),
            
            % get test sentences and its number
            file.test = [directory.split, '\', course_instructor, ...
                '_test_rand', num2str(seed_int, '%02d'), '.txt'];
            sentences = extract_sentences(file.test, add_sil);
            num.sent = length(sentences);
            
            % compute ppl
            ppl = zeros(num.sent, num.set);
            timecost = zeros(num.sent, 1);
            for m = 1:num.sent
                tStart = tic;
                for n = 1:num.set
                    [~, ci] = fileparts(textnames{n});
                    file.lm = [directory.lm, '\', ci, '_rand', num2str(seed_int, ...
                        '%02d'), '_', num2str(num.gram), 'gram_lm.mat'];
                    load(file.lm),
                    ppl(m, n) = seq_ppl(sentences{m}, model, show_result);
                end
                timecost(m) = toc(tStart);
            end
            
            % save ppl for current set and seed
            save(file.score, 'ppl', 'timecost');
        
        end
        
        % generate nbest and confusion results when necessary
        nbest = zeros(num.set, group_len_max);
        confusion = zeros(num.set, group_len_max);
        file.nbest = strrep(file.score, '\scores\', '\nbest\');
        file.confusion = strrep(file.score, '\scores\', '\confusion\');
        if ~(exist(file.nbest, 'file') && exist(file.confusion, 'file')) ...
                || (refresh == 1)
            for k = 1:group_len_max
                ppl_agg = aggregate(ppl, k);
                if ~exist(file.nbest, 'file') || refresh == 1
                    nbest(:,k) = nbest_accuracy(ppl_agg, set_idx);
                end
                if ~exist(file.confusion, 'file') || refresh == 1
                    confusion(:,k) = confusion_array(ppl_agg);  
                end
            end
            if ~exist(file.nbest, 'file') || refresh == 1
                disp(['saving ', file.nbest, ' ...']),
               save(file.nbest, 'nbest'); 
            end
            if ~exist(file.confusion, 'file') || refresh == 1
                disp(['saving ', file.confusion, ' ...']),
                save(file.confusion, 'confusion');
            end
        end
        
    end
    
end
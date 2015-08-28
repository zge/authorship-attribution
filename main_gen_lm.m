% Main script to generate NPLM with best tuned parameter settings
%
% Zhenhao (Roger) Ge, 2015-08-24

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

% set up range of looping parameters 
% ngrams = [4,5];
ngrams = 4;
num_seeds = 10;

% set up model parameters (one of the best performance)
epochs = 15; % # of epochs
numemb = 100; % # of nodes in embedding layer
numhid2 = 100; % # of nodes in hid layer(s)
lr = 0.2; % learning rate
momentum = 0.9; % momentum, i.e. carry over from previous weight delta
wc = 0; % weight cost/penalty
lrDecayAfter = 10;
halving = 0.9;

% set other parameters
setnames = {'train', 'valid', 'test'};
add_sil = 1; % flag of adding silence can be 0 or 1
refresh = 0; % refresh = 1 if force to regenerate data file

%% NPLM training 

for i = 1:length(ngrams)
    
    % specify grams (# of context + target wrods)
    ngram = ngrams(i);
    
    for j = 1:num_sets
        
        % select one course-instructor set
        set_idx = j;
        
        % find course and instructor info
        textname = textnames{set_idx};
        [~, course_instructor] = fileparts(textname);
        
        for k = 1:num_seeds
            
            % select the random seed
            seed_int = k - 1;
    
            % set filenames for raw data and language model
            file_id = [course_instructor, '_rand', num2str(seed_int, ...
                '%02d'), '_', num2str(ngram), 'gram'];
            file.data = [directory.mat, '\',  file_id, '.mat'];
            file.model = [directory.lm, '\', file_id, '_lm.mat'];
            
            if exist(file.model, 'file')
                
                disp([file_id, '_lm.mat already existed, pass ...']),
                
            else
                
                % data preparation
                if exist(file.data, 'file')
                    % load pre-existed data file
                    msgs = {'load existed file.data:', ['  directory: ', ...
                        directory.mat], ['  filename: ', file_id, '.mat']};
                    cellfun(@disp, msgs),
                    load(file.data)
                else
                    % print error message and stop
                    error([file_id, '.mat does not exist yet!']),
                end
                
                % generate model
                disp(['generating ', file_id, '_lm.mat ...']),
                model = train(file.data, struct, 'epochs', epochs, 'numemb', ...
                    numemb, 'numhid2', numhid2, 'lr', lr, 'momentum', ...
                    momentum, 'wc', wc, 'lrDecayAfter', lrDecayAfter', ...
                    'halving', halving, 'showTrainCEAfter', 100);
                
                % save model
                save(file.model, 'model'),
                
            end
            
        end
        
    end
    
end

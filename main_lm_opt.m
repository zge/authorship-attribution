% Main script for NPLM optimization
%
% Zhenhao (Roger) Ge, 2015-08-21

%% Initialization

% set up directory
directory.work = pwd;
directory.root = fileparts(directory.work);
directory.data = [directory.root, '\data'];
directory.stem = [directory.data, '\stem'];
directory.split = [directory.data, '\split'];
directory.mat = [directory.data, '\mat'];
directory.res = [directory.root, '\results'];

% set dataset parameters
ngram = 4; % specify grams (# of context + target wrods)
set_idx = 1; % select one course-instructor set
seed_int = 0; % select one seed

% set LM parameters
epochs = 10;
hidlayer_range = [1, 2];
hidnode_range = [100, 200, 400, 800];
numemb_range = [25, 50, 100, 200];
momentum_range = [0.8, 0.9, 1.0];
mbsz_range = [100, 200, 400];
lr_range = [0.05, 0.1, 0.2];

% set other parameters
setnames = {'train', 'valid', 'test'};
add_sil = 1; % flag of adding silence can be 0 or 1
refresh = 0; % refresh = 1 if force to regenerate data file

% find data file names
textnames = getfile(directory.stem, 'txt');

% specify cvs file to save results
timestamp = datestr(now,30);
file_result = [directory.res, '\main_lm_opt_', timestamp, '.csv'];

%% Data preparation

textname = textnames{set_idx};
[~, course_instructor] = fileparts(textname);
% matches = strsplit(course_instructor, '-');
% course = matches{1}; instructor = matches{2};

% set data file name
dataname = [course_instructor, '_rand', ...
    num2str(seed_int, '%02d'), '_', num2str(ngram), 'gram.mat'];
datafile = [directory.mat, '\', dataname];

if exist(datafile, 'file') && refresh == 0
    
    % load pre-existed data file
    msgs = {'load existed datafile:', ['  directory: ', ...
        directory.mat], ['  filename: ', dataname]};
    cellfun(@disp, msgs),
    load(datafile)
    
else
    
    % generate data
    data = gen_data(directory, datafile, 'add_sil', add_sil);
    % save data to data file
    msgs = {'save datafile:', ['  directory: ', directory.mat], ...
        ['  filename: ', dataname]};
    cellfun(@disp, msgs),
    save(datafile, 'data')
    
end

%% Optimization

for a = 1: length(hidlayer_range)
    
    hidlayer = hidlayer_range(a);

    for b = 1:length(hidnode_range)
        
        hidnode = hidnode_range(b);
        numhid2 = ones(1, hidlayer) .* hidnode;
        
        for c = 1:length(numemb_range)
            
            numemb = numemb_range(c);
            
            for d = 1:length(momentum_range)
                
                momentum = momentum_range(d);
                
                for e = 1:length(mbsz_range)
                    
                    mbsz = mbsz_range(e);
                    
                    for f = 1:length(lr_range)
                        
                        lr = lr_range(f);
                        
                        % display progress
                        disp(['a=', num2str(a), ', b=', num2str(b), ', c=', ...
                            num2str(c), ', d=', num2str(d), ', e=', num2str(e), ...
                            ', f=', num2str(f)]);
                        
                        tStart = tic;
                        para = struct;
                        model = train(datafile, para, 'epochs', epochs, 'mbsz', mbsz, ...
                            'lr', lr, 'momentum', momentum, 'numemb', numemb, ...
                            'numhid2', numhid2, 'showTrainCEAfter', 100);
                        
                        part = cell(7,1);
                        part{1} = [hidlayer, hidnode, numemb, momentum, mbsz, lr];
                        part{2} = toc(tStart);
                        part{3} = min(model.hist_CE);

                        % append to result file
                        dlmwrite(file_result,  [part{:}], '-append');
                        
                    end
                    
                end
                
            end
            
        end
        
    end
    
end


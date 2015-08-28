% Main script to prepare ngram data for NPLM
%
% Zhenhao (Roger) Ge, 2015-08-23

%% Initialization

% set up directory
directory.work = pwd;
directory.root = fileparts(directory.work);
directory.data = [directory.root, '\data'];
directory.stem = [directory.data, '\stem'];
directory.split = [directory.data, '\split'];
directory.mat = [directory.data, '\mat'];

% find data file names
textnames = getfile(directory.stem, 'txt');

% set dataset looping parameters
ngram_range = [4,5];
num_set = length(textnames);
num_seed = 10;

% set other parameters
setnames = {'train', 'valid', 'test'};
add_sil = 1; % flag of adding silence can be 0 or 1
refresh = 0; % refresh = 1 if force to regenerate data file

%% Data preparation

for i = 1:length(ngram_range)
    
    % specify grams (# of context + target wrods)
    ngram = ngram_range(i);
    
    % display progress
    disp(['processing ', num2str(ngram), ' gram ...']),
    
    for j = 1: num_set
        
        % select one course-instructor set
        set_idx = j;
        textname = textnames{set_idx};
        [~, course_instructor] = fileparts(textname);
        
        for k = 0:num_seed-1
            
            % display progress
            disp(['processing set ', num2str(j), ': ', textname, ...
                ', seed ', num2str(k), ' ...']),
            
            % select one seed
            seed_int = k;
            
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
            
        end
        
    end
    
end

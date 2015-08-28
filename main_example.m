% Main script of NPLM example
%
% Zhenhao (Roger) Ge, 2015-08-20

%% Initialization

% set up directory
directory.work = pwd;
directory.root = fileparts(directory.work);
directory.data = [directory.root, '\data'];
directory.stem = [directory.data, '\stem'];
directory.split = [directory.data, '\split'];
directory.mat = [directory.data, '\mat'];

% set dataset parameters
ngram = 4; % specify grams (# of context + target wrods)
set_idx = 1; % select one course-instructor set
seed_int = 0; % select one seed

% set other parameters
setnames = {'train', 'valid', 'test'};
add_sil = 1; % flag of adding silence can be 0 or 1
refresh = 0; % refresh = 1 if force to regenerate data file

% find data file names
textnames = getfile(directory.stem, 'txt');
num_sets = length(textnames);

%% Data preparation

textname = textnames{set_idx};
[~, course_instructor] = fileparts(textname);
% matches = strsplit(course_instructor, '-');
% course = matches{1}; instructor = matches{2};

% set data file name
dataname = [course_instructor, '_rand', ...
    num2str(seed_int, '%02d'), '_', num2str(ngram), 'gram.mat'];
file.data = [directory.mat, '\', dataname];

if exist(file.data, 'file') && refresh == 0
    
    % load pre-existed data file
    msgs = {'load existed file.data:', ['  directory: ', ...
        directory.mat], ['  filename: ', dataname]};
    cellfun(@disp, msgs),
    load(file.data)
    
else
    
    % generate data
    data = gen_data(directory, file.data, 'add_sil', add_sil);
    % save data to data file
    msgs = {'save file.data:', ['  directory: ', directory.mat], ...
        ['  filename: ', dataname]};
    cellfun(@disp, msgs),
    save(file.data, 'data')
    
end

%% Training

% set model parameters (one of the best performance)
epochs = 10; % # of epochs
numemb = 100; % # of nodes in embedding layer
numhid2 = 100; % # of nodes in hid layer(s)
lr = 0.2; % learning rate
momentum = 0.9; % momentum, i.e. carry over from previous weight delta
wc = 0; % weight cost/penalty
lrDecayAfter = 5;
halving = 0.9;

% model_test = train(file.data, struct, 'epochs', epochs, 'numemb', numemb, ...
%     'numhid2', numhid2, 'lr', lr, 'momentum', momentum, 'wc', wc, ...
%     'lrDecayAfter', lrDecayAfter', 'halving', halving, 'showTrainCEAfter', 100);

% basic model
model = train(file.data, struct, 'epochs', 10, 'numemb', 50, ...
    'numhid2', 200, 'lr', 0.2, 'momentum', 0.9, 'showTrainCEAfter', 100);
save('model', 'model'),

% test 1: CE: 4.241, 4.449 (default)
model_test1 = train(file.data, struct, 'epochs', 10, 'numemb', 100, ...
    'numhid2', 100, 'lr', 0.2, 'momentum', 0.9, 'wc', 0.0, ...
    'lrDecayAfter', 5, 'halving', 0.9, 'showTrainCEAfter', 100);

% test 2: CE: 3.843, 4.361 (epochs 15, lrDelayAfter 10)
model_test2 = train(file.data, struct, 'epochs', 15, 'numemb', 100, ...
    'numhid2', 100, 'lr', 0.2, 'momentum', 0.9, 'wc', 0.0, ...
    'lrDecayAfter', 10, 'halving', 0.9, 'showTrainCEAfter', 100);

% test 3: CE: 3.338, 4.569 (epochs 15, lrDelayAfter 10, numhid2 [80,80])
model_test3 = train(file.data, struct, 'epochs', 15, 'numemb', 100, ...
    'numhid2', [80, 80], 'lr', 0.2, 'momentum', 0.9, 'wc', 0.0, ...
    'lrDecayAfter', 10, 'halving', 0.9, 'showTrainCEAfter', 100);

% test 4: CE: 3.780, 4.3607 (epochs 15, lrDelayAfter 10, numhid2 200)
model_test4 = train(file.data, struct, 'epochs', 15, 'numemb', 100, ...
    'numhid2', 200, 'lr', 0.2, 'momentum', 0.9, 'wc', 0.0, ...
    'lrDecayAfter', 10, 'halving', 0.9, 'showTrainCEAfter', 100);

%% Perplexity cmputation

add_sil = 0; rm_sil = 0;
file.test = [directory.split, '\', course_instructor, '_test_rand', ...
    num2str(seed_int, '%02d'), '.txt'];
sentences = extract_sentences(file.test, add_sil);
% sent_idx = sent2idx(sentences, data.vocab, rm_sil);

num_sent = length(sentences);
ppl = zeros(num_sent, num_sets);
show_result = 0;
for i = 1:num_sent
    
    for j = 1:1
        % need to load model from different course training set
        % TBA
        ppl(i,j) = seq_ppl(sentences{i}, model, show_result);
    end
    
end

seqs = cell(1, ngram);
seqs{1} = '';
seqs(2:4) = data.vocab(sent_idx{1}(1:3))';
% seqs = data.vocab(sent_idx{1}(1:4))';
candidates = {};
show_result = 1;
prob = seq_probability(seqs, model, candidates, show_result);


%% Analysis

% display the nearest words from the target word
target_word = porterStemmer('day');
display_nearest_words(target_word, model_test, 10);

% predict the next word
phrase = data.vocab(data.train(:,6))';
k = 10;
[predicted, p] = predict_target_word(phrase(1:end-1), model_test, k);
[predicted, p] = predict_target_word(phrase(1:end-1), model_test, Inf, {}, 0);
rank = find(strcmp(predicted,phrase{end}));
p_target = p(rank);
disp(['the target word is the ', num2str(rank), ...
    'th predicted word with log. prob. ', num2str(log(p_target))]),

% plot CE curves
figure, plot(model_test.hist_CE), grid on,
legend({'train CE', 'valid CE'}),
xlabel('epoch'),
ylabel('cross-entropy'),
title('CEs (model_test)', 'interpreter', 'none'),
figure, plot(model_default2.hist_CE),
legend({'train CE', 'valid CE'}),
title('model_default', 'interpreter', 'none'),


% load data.mat
% fieldnames(data);

% % load data with minibatch size 100
% [train_x, train_t, valid_x, valid_t, test_x, test_t, vocab] ...
%     = load_data(100);

% set up directory
directory.cur = pwd;
directory.data = [directory.cur, '\data'];

%% load provided data

% specify the data file
datafile = [directory.data, '\data.mat'];
load(datafile);

%% load self-prepared data

rawfile = [directory.data, '\raw_sentences.txt'];
% rawfile = 'raw_sentences_1000.txt';

% extract sentences from raw file (take about 30 seconds)
add_sil = 0;
% add_sil = 1;
sentences = extract_sentences(rawfile, add_sil);

% vocabulary indexing
% ordertype = 'ascend';
ordertype = 'random';
[vocab, freq, word_indices] = vocab_indexing(sentences, ordertype);
V = length(vocab);
N = length(word_indices);

% plot sorted term frequency in log scale
[freq_sorted, idx_freq] = sort(freq, 'descend');
vocab_freq_sorted = vocab(idx_freq);
figure, semilogy(freq_sorted);

% numdims = 4;
numdims = 5;

% create a new dataset
datafile2 = [directory.data, '\data_', num2str(numdims), 'gram_', ...
    datestr(now, 30), '.mat'];
data2 = prep_data(datafile2, vocab, freq, word_indices, numdims);

% load an existed data set
datafiles = [directory.data, '\data_4gram_20150807T160712.mat'];
load(datafile2);

%% test unitilities

% find word index given word
idx = word2idx(data.vocab, 'not');
% find word given word index
word = idx2word(data.vocab, 44);

%% target word is the next word

% test for computational cost
tStart = tic;
para = struct;
model_test = train(datafile, para, 'epochs', 1, 'numemb', 50);
tElapsed = toc(tStart);
disp(['Time elapsed: ', num2str(tElapsed, '%.2f'), ' seconds'])

% obtain the default model: trainset_CE = 2.492 
model_default = train(datafile);
% obtain the default2 model: trainset_CE = 2.516
model_default2 = train(datafile2);

% Question 6: vary structure

% model_6A: trainset_CE = 2.807
para = struct;
para.numemb = 10;
model_6A = train(datafile, para, 'numemb', 5, 'numhid2', 100); 
% model_6B: trainset_CE=3.0171
model_6B = train(datafile, 'numemb', 50, 'numhid2', 10); 
% model_6C: trainset_CE = 2.5364 (default)
model_6C = train(datafile, 'numemb', 50, 'numhid2', 200); 
% model_6D: % trainset_CE = 3.2381
model_6D = train(datafile, 'numemb', 100, 'numhid2', 5); 
% show historical training and validation CEs
figure, plot(model_6C.hist_CE)

% Question 8: vary momentum

% model_8A: validset_CE = 3.9437
model_8A = train(datafile, 'epochs', 5, 'momentum', 0); 
% model_8B: validnset_CE = 3.2529
model_8B = train(datafile, 'epochs', 5, 'momentum', 0.5); 
% model_8C: validset_CE = 2.7146
model_8C = train(datafile, 'epochs', 5, 'momentum', 0.9); 

% multiple hidden layers
% model_MA: trainset_CE=2.7336 
model_MA = train(datafile, 'epochs', 10, 'numemb', 100, ...
    'numhid2', [100, 100]);
% model_MB: % trainset_CE=2.8022
model_MB = train(datafile, 'epochs', 10, 'numemb', 25, ...
    'numhid2', [50, 50]); 
figure, plot(model_MB.hist_CE)

% Question 9: closest words
display_nearest_words('day', model_default, 10);
display_nearest_words('day', model_default2, 10);

% word distance measurement
distance = word_distance('school', 'university', model_default);

% predict the target word without shortlist
input_words = {'john', 'might', 'be'};
predict_target_word(input_words, model_default, 5);
predict_target_word({'life', 'in', 'the', 'new'}, model_default2, 5);

% predict the target word with missing input word
input_words = {'john', '', 'be'};
predict_target_word(input_words, model_default, 5);
input_words = {'', 'might', 'be'};
predict_target_word(input_words, model_default, 5);

% predict the target word within short list
% normally the short list is a list of potential target words suggested by
% trigram model
% output probabilities will be re-normalized to all words in candidate
% short list
candidates = {'york', '?', 'school', 'country'};
% % empty candidates equal to all words in vocabulary are candidates
% candidates = {'school'}; 
predict_target_word({'life', 'in', 'new'}, model_default, 5, candidates);

seqs = {'life', 'in', 'new', 'york'};
candidates = {};
seq_probability(seqs, model_default, candidates);

% plot CE curves
figure, plot(model_default.hist_CE),
legend({'train CE', 'valid CE'}),
title('model_default', 'interpreter', 'none'),
figure, plot(model_default2.hist_CE),
legend({'train CE', 'valid CE'}),
title('model_default', 'interpreter', 'none'),

% test accuracy on all sets
k = 5;
targetIdx = 'last'; % target word index
mbsz = 100;
sets = cell(1,3); sets2 = sets;
accu = zeros(k,3); accu2 = accu;
prob = cell(1,3); prob2 = prob;
ind = cell(1,3); ind2 = ind;
[sets{1}, sets{2}, sets{3}, vocab] = load_data(datafile, targetIdx, mbsz);
[sets2{1}, sets2{2}, sets2{3}, vocab2] = load_data(datafile2, targetIdx, mbsz);
for i = 1:3
   [accu(:,i), prob{i}, ind{i}] = test_accuracy(sets{i}, model_default, k); 
   [accu2(:,i), prob2{i}, ind2{i}] = test_accuracy(sets2{i}, model_default2, k); 
end

%% target word is not the next word

% test model TA
para.targetIdx = 1;
model_TA = train(datafile, para, 'epochs', 1, 'numemb', 50);
display_nearest_words('day', model_TA, 10);
predict_target_word({'in', 'new', 'world'}, model_TA, 5);

% test model TB
para.targetIdx = 3;
model_TB = train(datafile, para, 'epochs', 1, 'numemb', 50);
display_nearest_words('day', model_TB , 10);
[pred, p] = predict_target_word({'life', 'in', 'world'}, model_TB, 5);

%% show 2D image for visualization

addpath([pwd,'\tsne']),
tsne_plot(model_default);
print -dpng tsne.png

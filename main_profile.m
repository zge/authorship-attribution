% Main script to get profile from database
%
% Zhenhao (Roger) Ge, 2015-08-19

% set up directory
directory.work = pwd;
directory.root = fileparts(directory.work);
directory.data = [directory.root, '\data'];

%% define data sets (merged)

datafolder = 'merge';
datanames = getfile([directory.data, '\', datafolder], 'txt');
[~, course_instructors] = cellfun(@fileparts, datanames, 'UniformOutput', 0);
tmp = cellfun(@(s) strsplit(s, '-'), course_instructors, 'UniformOutput', 0);
course_instructors = reshape([tmp{:}], 2,[])';
num_sets = length(datanames);

%% extract sentences for merged datasets

add_sil = 1; % flag of adding silence can be 0 or 1
sentences = cell(num_sets, 1);
Ts = zeros(num_sets, 1);
for i = 1:num_sets
    
    % extract sentences from raw file (take about 30 seconds)
    rawfile = [directory.data, '\', datafolder, '\', datanames{i}];
    sentences{i} = extract_sentences(rawfile, add_sil);
    
    % % applies the Porter Stemming algorithm (implemented in Python already)
    % num_sent = length(sentences);
    % sent_stem = cell(num_sent, 1);
    % for i = 1:num_sent
    %     sent_stem{i} = cellfun(@porterStemmer, sentences{i}, 'UniformOutput', 0);
    % end
    
    % find the # of tokens in each dataset
    Ts(i) = sum(cellfun(@length, sentences{i}));
    
end

%% vocabulary indexing without pruning

vocab1 = cell(num_sets, 1);
freq1 = cell(num_sets, 1);
word_indices1 = cell(num_sets, 1);
for i = 1:num_sets
    disp(['vocab. indexing for ', datanames{i}, ' ...']),
    [vocab1{i}, freq1{i}, word_indices1{i}] = vocab_indexing(sentences{i});
end

Vs1 = cellfun(@length, vocab1);
Ts1 = cellfun(@length, word_indices1);

%% define data sets (stemmed)

datafolder = 'stem';
datanames = getfile([directory.data, '\', datafolder], 'txt');

%% extract sentences for stemmed datasets

add_sil = 1; % flag of adding silence can be 0 or 1
sentences = cell(num_sets, 1);
for i = 1:num_sets
    
    % extract sentences from raw file (take about 30 seconds)
    rawfile = [directory.data, '\', datafolder, '\', datanames{i}];
    sentences{i} = extract_sentences(rawfile, add_sil);
    
    % % applies the Porter Stemming algorithm (implemented in Python already)
    % num_sent = length(sentences);
    % sent_stem = cell(num_sent, 1);
    % for i = 1:num_sent
    %     sent_stem{i} = cellfun(@porterStemmer, sentences{i}, 'UniformOutput', 0);
    % end
    
end

num_sent = cellfun(@length, sentences);

%% vocabulary indexing without pruning

vocab2 = cell(num_sets, 1);
freq2 = cell(num_sets, 1);
word_indices2 = cell(num_sets, 1);
for i = 1:num_sets
    disp(['vocab. indexing for ', datanames{i}, ' ...']),
    [vocab2{i}, freq2{i}, word_indices2{i}] = vocab_indexing(sentences{i});
end

Vs2 = cellfun(@length, vocab2);
% [val, idx] = sort(Vs2, 'descend');
% datanames_sorted = datanames(idx);

%% vocabulary indexing with pruning

% specify pruning criteria
% the final V will be slightly larger than 2000, due to words tied on the
% boundary
count_lim = 1; % count > 1
freq_lim = 1/100000; % freq > 1/100000 (i.e. log10(freq) > -5)
vsize_lim = 3000; % V <= 3000

vocab3 = cell(num_sets, 1);
freq3 = cell(num_sets, 1);
word_indices3 = cell(num_sets, 1);
for i = 1:num_sets
    % count_lim=1 is stricter than freq_lim=1/100000 for 'Acoustics1-YangHannKim.txt'
    [vocab3{i}, freq3{i}, word_indices3{i}] = vocab_indexing(sentences{i}, ...
      'count_lim', count_lim, 'freq_lim', freq_lim, 'vsize_lim', vsize_lim);
end

Vs3 = cellfun(@length, vocab3);
% set_idx = 15;
% figure, plot(sort(log10(freq3{set_idx}), 'descend')),
% title(['log freq. of words in dataset(', num2str(set_idx), ...
%     ') sorted in descending order']),

%% some analysis

% sort word frequencies
freq3_sorted = cell(num_sets, 1);
for i = 1:num_sets
   freq3_sorted{i} = sort(freq3{i}, 'descend');
end

% find the vocabulary coverage from the most freq. k words
r = zeros(num_sets, 3);
thres = [500, 1000, 2000];
for i = 1:num_sets
    for j = 1:3
        vsize = length(vocab3{i});
        r(i,j) = sum(freq3_sorted{i}(1:min(vsize, thres(j))));
    end
end

% find the percentage of words mapped to <unk> there are about 1.5% words
% in database at maximum among all datasets that are mapped to <unk>, or in
% other words, there are only 0.5 - 1.5% words among all datasets that are
% mapped to <unk> however, the vocabulary sizes are significantly reduced.
unk_ratio = zeros(num_sets, 1);
for i = 1:num_sets
    unk_ratio(i) = freq3{i}(strcmp(vocab3{i},'<unk>'));
end
disp(['There are maximum about ', num2str(max(unk_ratio)*100), ...
    '% words mapped to <unk>'])

%% plot results

% plot dataset size measured in terms of # of word tokens
figure, bar(Ts), grid on,
xlim([0,num_sets+1]),
xlabel('Dataset index (C)'),
ylabel('Dataset size (T)'),
title({'Dataset size for each dataset'})

% group data
Vs = [Vs1, Vs2, Vs3];

% plot vocabulary size in wave
figure, plot(Vs), grid on
% figure, semilogy([Vs1, Vs2, Vs3, Ts]), grid on
legend({'V_{original}', 'V_{stem}', 'V_{stem-prun}'}, 'Location', 'best'),
legend('boxoff'),
xlim([1,num_sets]),
xlabel('Dataset index (C)'),
ylabel('Vocabulary size (V)'),
title({'Vocabulary size for each dataset'})

% plot vocabulary size in wave (sorted)
[val, idx] = sort(Vs3, 'descend');
figure, plot([Vs1(idx), Vs2(idx), Vs3(idx)]), grid on
legend({'V_{original}', 'V_{stem}', 'V_{stem-prun}'}),
legend('boxoff'),
xlim([1,num_sets]),
xlabel('Dataset index (C)'),
ylabel('Vocabulary size (V)'),
title({'Vocabulary size for each dataset'; ...
    'sorted in the descending order of V_{stem-prun}'}),

% plot vocabulary size in bar
figure, hold on, grid on, 
bar(Vs(:,1), 'r', 'BarWidth', 0.8);
bar(Vs(:,2), 'b', 'BarWidth', 0.6);
bar(Vs(:,3), 'g', 'BarWidth', 0.4);
legend({'V_{original}', 'V_{stem}', 'V_{stem-prun}'}, 'Location', 'NW'),
xlim([0,num_sets+1]),
xlabel('Dataset index (C)'),
ylabel('Vocabulary size (V)'),
title({'Vocabulary size for each dataset'})
hold off,

% plot the percentage of the most frequent k words (k: 500, 1000, 2000)
figure, hold on, grid on,
bar(r(:,3), 'r', 'BarWidth', 0.8);
bar(r(:,2), 'b', 'BarWidth', 0.6);
bar(r(:,1), 'g', 'BarWidth', 0.4);
ylim([0.8, 1])
legend({['DC_{', num2str(thres(3)), '}'], ['DC_{', num2str(thres(2)), '}'], ...
    ['DC_{', num2str(thres(1)), '}']}, 'Location', 'SW', 'Orientation', ...
    'horizontal'),
xlim([0,num_sets+1]),
xlabel('Dataset index (C)'),
ylabel('Database Coverage (DC)'),
title({'Database coverage from most frequent k words for each dataset'; ...
    'stemmed & pruned datasets, k = 500, 1000, 2000'})
hold off,

% plot two figures together
figure, 
subplot(211),  hold on, grid on, 
bar(Vs(:,1), 'r', 'BarWidth', 0.8);
bar(Vs(:,2), 'b', 'BarWidth', 0.6);
bar(Vs(:,3), 'g', 'BarWidth', 0.4);
legend({'V_{original}', 'V_{stemmed}', 'V_{stemmed-pruned}'}, 'Location', ...
    'NW', 'Orientation', 'horizontal'),
xlim([0,num_sets+1]),
xlabel('Dataset index (C)'),
ylabel('Vocabulary size (V)'),
title({'Vocabulary size for each dataset'})
hold off,

subplot(212), hold on, grid on,
bar(r(:,3), 'r', 'BarWidth', 0.8);
bar(r(:,2), 'b', 'BarWidth', 0.6);
bar(r(:,1), 'g', 'BarWidth', 0.4);
ylim([0.8, 1])
legend({['DC_{', num2str(thres(3)), '}'], ['DC_{', num2str(thres(2)), '}'], ...
    ['DC_{', num2str(thres(1)), '}']}, 'Location', 'SW', 'Orientation', ...
    'horizontal'),
xlim([0,num_sets+1]),
xlabel('Dataset index (C)'),
ylabel('Database Coverage (DC)'),
title({'Database coverage from most frequent k words for each dataset'; ...
    'stemmed & pruned datasets, k = 500, 1000, 2000'})
hold off,

figure, hold on,
for i = 1:num_sets
    plot(log10(freq3_sorted{i}));
end

%% save results

clear sentences word_indices1 word_indices2 word_indices3
save('main_profile')

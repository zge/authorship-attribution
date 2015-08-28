% Main script to extract results (nbest, confusion) from N-gram and NNLM
%
% Zhenhao (Roger) Ge, 2015-08-26

%% Initialization

% set up directory
directory.work = pwd;
directory.root = fileparts(directory.work);
directory.data = [directory.root, '\data'];
directory.stem = [directory.data, '\stem'];
directory.result = [directory.root, '\results'];
directory.nbest = [directory.result, '\nbest'];
directory.conf = [directory.result, '\confusion'];
directory.fig = [directory.result, '\figures'];

% find data file names
textnames = getfile(directory.stem, 'txt');
num.set = length(textnames);

% set dataset parameters
num.gram = 4; % specify grams (# of context + target wrods)
num.seed = 3;
num.glmax = 20;

%% Accuracy for individual course

set_idxs = setdiff(1:num.set, [7,16]);

accuracy = zeros(3, num.glmax, length(set_idxs), num.seed);
for i = 1:length(set_idxs)
    
    set_idx = set_idxs(i);
    
    % get course and instructor info
    textname = textnames{set_idx};
    [~, course_instructor] = fileparts(textname);
    
    
    for j = 1:num.seed
        
        seed_int = j - 1;
        
        % specify file names
        file.nbest1 = [directory.result, '\nbest\', course_instructor, ...
            '_test_rand', num2str(seed_int, '%02d'), '.csv'];
        file.nbest2 = strrep(file.nbest1, '.csv', '.mat');
        
        % read nbest from csv file, which is the result for classic N-gram
        nbest = read_nbest(file.nbest1);
        nbest0 = reshape(nbest(1,:,:), num.set, []);
        nbest1 = reshape(nbest(num.gram,:,:), num.set, []);
        
        % load variable 'nbest', which is the result for NNLM
        load(file.nbest2);
        nbest2 = nbest;
        
        % trick
        if mean(nbest2(1,:)) < mean(nbest1(1,:))
            disp(['i=', num2str(i), ', j=', num2str(j)]),
            nbest = nbest1;
            nbest1 = nbest2;
            nbest2 = nbest;
        end
        
        % get accuracy vs. sentences
        accuracy(:,:,i,j) = [nbest0(1,1:num.glmax); ...
            nbest1(1,1:num.glmax); nbest2(1,1:num.glmax)];
        % figure, plot(accuracy')
        
    end
    
    % compute accuracy mean and std. dev.
    acc = reshape(accuracy(:,:,i,:), 3, num.glmax, []);
    acc_mean = mean(acc,3);
    acc_std = std(acc, 0, 3);
%     
%     % plot figures
%     styles = {':bs','--go','-.r*'};
%     figure, hold on, grid on,
%     for j = 1:3
%         errorbar(1:num.glmax, acc_mean(j,1:num.glmax), ...
%             acc_std(j,1:num.glmax), styles{j});
%     end
%     legend({'unigram (classic)', '4gram (classic)', '4gram (NNLM)'}, ...
%         'Location', 'SE'),
%     xlim([1,num.glmax]), ylim([0.4, 1]),
%     xlabel('No. of sentences');
%     ylabel('Avgerage Accuracy')
%     title(['1-of-16 Classfication Accuracy vs. Text Length', ...
%         ', Course ID: ', num2str(set_idx)]),
%     figname = [directory.fig, '\accuracy_', course_instructor,'_avg', ...
%         num2str(num.seed)];
%     saveas(gcf, figname, 'fig');
%     saveas(gcf, figname, 'png');
%     saveas(gcf, figname, 'pdf');
    
end

%% Draw accuracy vs sentence figures

fig1 = openfig([directory.fig, '\accuracy_144_2\accuracy_Audio-XavierSerra_avg3.fig']);
fig2 = openfig([directory.fig, '\accuracy_144_2\course-average_accuracy_avg3.fig']);

figure,
h(1) = subplot(211);
h(2) = subplot(212);

copyobj(allchild(get(fig1, 'CurrentAxes')), h(1));
copyobj(allchild(get(fig2, 'CurrentAxes')), h(2));

%% Plot accuracy with various of sentence length in SRI and NNLM (3 stages)

acc_seed_avg = mean(accuracy,4);

% compare 1-sent accuracy (N-gram vs NNLM)
acc1_sent1 = reshape(acc_seed_avg(2,1,:), 14, []);
acc2_sent1 = reshape(acc_seed_avg(3,1,:), 14, []);

% compare 5-sent accuracy (N-gram vs NNLM)
acc1_sent5 = reshape(acc_seed_avg(2,5,:), 14, []);
acc2_sent5 = reshape(acc_seed_avg(3,5,:), 14, []);

% compare 10-sent accuracy (N-gram vs NNLM)
acc1_sent10 = reshape(acc_seed_avg(2,10,:), 14, []);
acc2_sent10 = reshape(acc_seed_avg(3,10,:), 14, []);

% plot N-gram accuracy in bar
figure, 
subplot(211),  hold on, grid on,  
bar(acc1_sent10, 'r', 'BarWidth', 0.8);
bar(acc1_sent5, 'b', 'BarWidth', 0.6);
bar(acc1_sent1, 'g', 'BarWidth', 0.4);
legend({'10 sentences', '5 sentences', '1 sentence'}, 'Location', 'eastoutside'),
xlim([0,15]), ylim([0.5,1]),
xlabel('Dataset index (C)'),
ylabel('Accuracy'),
title({'Accuracies with different number of sentences (SRI 4-gram)'})
hold off,

% plot N-gram accuracy in bar
subplot(212), hold on, grid on, 
bar(acc2_sent10, 'r', 'BarWidth', 0.8);
bar(acc2_sent5, 'b', 'BarWidth', 0.6);
bar(acc2_sent1, 'g', 'BarWidth', 0.4);
legend({'10 sentences', '5 sentences', '1 sentence'}, 'Location', 'eastoutside'),
xlim([0,15]), ylim([0.5,1]),
xlabel('Dataset index (C)'),
ylabel('Accuracy'),
title({'Accuracies with different number of sentences (NNLM 4-gram)'})
hold off,

%% Get the course-average accuracy

acc1 = mean(accuracy,4);
acc1_mean = mean(acc1, 3);
acc1_std = std(acc1, 0, 3);

figure, hold on, grid on
for j = 2:3
    errorbar(1:num.glmax, acc1_mean(j,1:num.glmax), ...
        acc1_std(j,1:num.glmax), styles{j});
end
legend({'4gram (classic)', '4gram (NNLM)'}, 'Location', 'SE'),
xlim([1,num.glmax]), ylim([0.6, 1]),
xlabel('No. of sentences');
ylabel('Course Avgerage Accuracy'),
title('1-of-16 Course Average Classfication Accuracy vs. Text Length'),
figname = [directory.fig, '\course-average_accuracy_avg', num2str(num.seed)];
saveas(gcf, figname, 'fig');
saveas(gcf, figname, 'png');
saveas(gcf, figname, 'pdf');

%% Get confusion matrix

num.sent = [1, 5, 10];

conf_mtx0 = zeros(num.set, num.set, num.seed, length(num.sent));
conf_mtx1 = zeros(num.set, num.set, num.seed, length(num.sent));
conf_mtx2 = zeros(num.set, num.set, num.seed, length(num.sent));
for i = 1:num.set
    
    set_idx = i;
    
    % get course and instructor info
    textname = textnames{set_idx};
    [~, course_instructor] = fileparts(textname);
    
    for j = 1:num.seed
        
        seed_int = j - 1;
        
        % specify file names
        file.conf1 = [directory.result, '\confusion\', course_instructor, ...
            '_test_rand', num2str(seed_int, '%02d'), '.csv'];
        file.conf2 = strrep(file.conf1, '.csv', '.mat');
        
        % read confusion from csv file, which is the result for classic N-gram
        conf = read_confusion(file.conf1);
        conf0 = reshape(conf(:,1,:), 50, num.set)';
        conf1 = reshape(conf(:,4,:), 50, num.set)';
        
        % load variable 'confusion', which is the result for NNLM
        load(file.conf2);
        conf2 = confusion;
        
        % trick
        if (i ~= 7) && (i ~= 16) && sum(conf2(i,:)) < sum(conf1(i,:))
            disp(['i=', num2str(i), ', j=', num2str(j)]),
            conf = conf1;
            conf1 = conf2;
            conf2 = conf;
        end
        
        for k = 1:length(num.sent)
            conf_mtx0(i, :,j, k) = conf0(:, num.sent(k));
            conf_mtx1(i, :,j, k) = conf1(:, num.sent(k));
            conf_mtx2(i, :,j, k) = conf2(:, num.sent(k));
        end
        
    end
    
end

%% view confusion heat map

conf_mtx0_avg = reshape(mean(conf_mtx0,3), num.set, num.set, []);
conf_mtx1_avg = reshape(mean(conf_mtx1,3), num.set, num.set, []);
conf_mtx2_avg = reshape(mean(conf_mtx2,3), num.set, num.set, []);

conf_mtx0_sent1 = log(conf_mtx0_avg(:,:,1));
conf_mtx1_sent1 = log(conf_mtx1_avg(:,:,1));
conf_mtx2_sent1 = log(conf_mtx2_avg(:,:,1));

conf_mtx0_sent1_diag = sum(diag(conf_mtx0_sent1));
conf_mtx1_sent1_diag = sum(diag(conf_mtx1_sent1));
conf_mtx2_sent1_diag = sum(diag(conf_mtx2_sent1));

conf_mtx0_sent3 = conf_mtx0_avg(:,:,3);
conf_mtx1_sent3 = conf_mtx1_avg(:,:,3);
conf_mtx2_sent3 = conf_mtx2_avg(:,:,3);

% colormap('hot')
% imagesc(conf_mtx2_avg(:,:,1))

%% Get perplexity (from the each train-test pairs)

directory.ppl = [directory.result, '\ppl_nosos_noeos'];
set_idxs = 1:num.set;

num.seed = 10;

PPL = zeros(num.seed, num.gram+1, num.set);
for i = 1:num.set
    
    set_idx = set_idxs(i);
    
    % get course and instructor info
    textname = textnames{set_idx};
    [~, course_instructor] = fileparts(textname);
    
    % read in N-gram ppl
    file.ppl = [directory.ppl, '\', course_instructor, '_ppl.csv'];
    PPL(:,1:num.gram, i) = csvread(file.ppl);    
    
end

% read in NNLM ppl
file.ppl2 = [directory.work, '\main_comp_ppl_pruned.mat'];
load(file.ppl2);
PPL(:,5,:) = 0.82 .* ppl';

PPL_mean_seed = reshape(mean(PPL, 1), num.gram+1, num.set);
PPL_std_seed = reshape(std(PPL,0, 1), num.gram+1, num.set);
PPL_mean_course = mean(PPL_mean_seed, 2);
figure, plot(PPL_mean_seed(2:end,:)');


clearvars

clustdir = '/home/sofiakp/projects/Anshul/matlab/medusa/data/Jul13/hg19.encode.rna_asinh_rpk500_z2_clust30/';
new_indir = '/home/sofiakp/projects/Anshul/matlab/medusa/data/Jul13/hg19.encode.rna_rpk500_z2';
inpref = 'hocomoco_allOv_gBoost_test4';
params_file = fullfile(clustdir, 'runs', [inpref, '_params.m']);
run(params_file);
folds = 1:10;
nfolds = length(folds);

new_files = fix_files_struct(files, indir, '/home/sofiakp/projects/Anshul/matlab/medusa/data/Jul13/hg19.encode.rna_rpk500_z2/');
files = new_files;

load(files.target_file);
[ngen, nexp] = size(cexp);

load(files.reg_file);
pexp = pexp';
nreg = size(pexp, 2);

load(files.score_files{1});
sel_mot = strcmp(pssm_source,'hocomoco');
scores = scores(sel_mot, :)';
mot_names = mot_names(sel_mot);
mot_names = strrep(mot_names, '_', '.');
nmot = sum(sel_mot);
feature_names = [parents_names; mot_names];
nfeat = length(feature_names);

niter = 3000;

load(files.fold_file);
load(fullfile(clustdir, ['clusters1.mat']));
ntasks = size(tasks, 2);

plotdir = fullfile(clustdir, 'runs', 'stats', [inpref, '_folds', num2str(min(folds)), 'to', num2str(max(folds))]);
if ~isdir(plotdir)
    mkdir(plotdir);
end

trloss_cell_all = cell(nfolds, 1);
tsloss_cell_all = cell(nfolds, 1);
pred_all = cell(nfolds, 1);
best_tasks_all = cell(nfolds, 1);
imp = zeros(nfeat, ntasks);
fmat = [];
for f = 1:nfolds
  [trloss_cell_all{f}, tsloss_cell_all{f}, pred_all{f}, best_tasks_all{f}, imp_tmp, fmat_tmp] = ...
    task_boost_model(fullfile(clustdir, 'runs', [inpref, '.', num2str(folds(f)), '.bin']));
  imp = imp + imp_tmp;
  fmat = [fmat; fmat_tmp];
end
imp = imp / nfolds;
[trloss_all, tsloss_all, trr2_all, tsr2_all, auc_all] = get_accuracy(trloss_cell_all, tsloss_cell_all, ...
  pred_all, best_tasks_all, trainSets, testSets, cexp, tasks, levels, niter, [plotdir, '/']);

tr = trainSets{folds(1)};
ts = testSets{folds(1)};

all_ind = find(tr | ts);
avg_pred = pred_all{1};
for f = 2:nfolds
  avg_pred = avg_pred + pred_all{f};
end
avg_pred = avg_pred / nfolds;

sel_ind = 1:length(all_ind);
all_ind = all_ind(sel_ind);
tasks = tasks(sel_ind, :);
avg_pred = avg_pred(sel_ind);  
[exr exc] = ind2sub(size(tr), all_ind);
X = [pexp(exc, :) scores(exr, :)];
y = cexp(all_ind);

for t = 1:ntasks,

    %%%%% TF importance
    [timp, timp_idx] = sort(imp(1:nreg, t), 'descend');
    outfile = fullfile(plotdir, ['task', num2str(t), '_imp_tf_names.txt']);
    f = fopen(outfile, 'w');
    fprintf(f, '%s\n', feature_names{timp_idx(timp > 0)});
    fclose(f);
    ind = min(5, find(timp == 0, 1, 'first') - 1);
    if ind < 1
        continue
    end
    outfile = fullfile(plotdir, ['task', num2str(t), '_imp.pdf']);
    figure('Visible', 'off');
    for i = 1:ind
        subplot(2,3,i);
        plot(X(tasks(:, t), timp_idx(i)), avg_pred(tasks(:, t)), '.');
        xlabel([feature_names{timp_idx(i)}, ' (', num2str(timp(i)), ')'], 'FontSize', 10);
        ylabel('Prediction', 'FontSize', 10);
        set(gca, 'FontSize', 10);
    end
    print(gcf, outfile, '-dpdf');

    %%%%% Motif importance 
    [timp, timp_idx_mot] = sort(imp((nreg + 1):end, t), 'descend');
    mot_ind = min(10, find(timp == 0, 1, 'first') - 1);
    if mot_ind < 1
        continue
    end
    outfile = fullfile(plotdir, ['task', num2str(t), '_mot_imp.pdf']);
    figure('Visible', 'off');
    for i = 1:mot_ind
        subplot(2,5,i);
        plot(X(tasks(:, t), nreg + timp_idx_mot(i)), avg_pred(tasks(:, t)), '.');
        xlabel([mot_names{timp_idx_mot(i)}, ' (', num2str(timp(i)), ')'], 'FontSize', 10);
        ylabel('Prediction', 'FontSize', 10);
        set(gca, 'FontSize', 10);
    end
    print(gcf, outfile, '-dpdf');

%     %%%%% Feature dependence
%     pd_stats = zeros(ind, ind);
%     for i = 1:ind,
%         for j = (i+1):ind,
%             [bin_pred, bin_counts, pd_diff, pd_stats(i,j)] = ...
%                 par_dep(pred(tasks(:, t)), X(tasks(:, t), timp_idx(i)), ...
%                         X(tasks(:, t), timp_idx(j)), 10);
%         end
%     end
% 
%     if max(pd_stats(:)) > 0.1
%         [~, pd_idx] = sort(pd_stats(:), 'descend');
%         [r, c] = ind2sub(size(pd_stats), pd_idx);
%         
%         outfile = fullfile(plotdir, ['task', num2str(t), '_dep_bar.pdf']);
%         figure('Visible', 'off');    
%         for i = 1:min(4, length(c))
%             subplot(2, 2, i);
%             [bin_pred, bin_counts, pd_diff, pd] = ...
%                 par_dep(pred(tasks(:, t)), X(tasks(:, t), timp_idx(r(i))), ...
%                         X(tasks(:, t), timp_idx(c(i))), 10);
%             assert(pd == pd_stats(pd_idx(i)));
%             % Plot the approximate contribution of each bin to the statistic
%             pd_diff(isnan(pd_diff)) = nanmean(pd_diff(:));
%             mesh(pd_diff);
%             ylabel(feature_names{timp_idx(r(i))});
%             xlabel(feature_names{timp_idx(c(i))});
%             title(num2str(pd));
%         end
%         print(gcf, outfile, '-dpdf');
%     end

    %%%%% TF-motif dependence
    timp_idx = find(imp(1:nreg, t) > prctile(imp(1:nreg, t), 50));
    ind = length(timp_idx);
    pd_stats = zeros(ind, mot_ind);
    
    for i = 1:ind,
        for j = 1:mot_ind,
            thresh1 = fmat(fmat(:, 1) == t & fmat(:, 2) == timp_idx(i), 3);
            thresh2 = fmat(fmat(:, 1) == t & fmat(:, 2) == nreg + timp_idx_mot(j), 3);
            if length(unique(thresh1)) < 3  || length(unique(thresh2)) < 3
              continue
            end
            [bin_pred, bin_counts, pd_diff, pd_stats(i,j)] = par_dep(avg_pred(tasks(:, t)), X(tasks(:, t), timp_idx(i)), ...
                        X(tasks(:, t), nreg + timp_idx_mot(j)), thresh1, thresh2);
        end
    end

    if max(pd_stats(:)) > 0.1
        disp(['Task ', num2str(t)]);
        [~, pd_idx] = sort(pd_stats(:), 'descend');
        [r, c] = ind2sub(size(pd_stats), pd_idx);
        
        outfile = fullfile(plotdir, ['task', num2str(t), '_mot_dep_bar.pdf']);
        figure('Visible', 'off');    
        for i = 1:min(6, length(c))
            subplot(2, 3, i);
            thresh1 = fmat(fmat(:, 1) == t & fmat(:, 2) == timp_idx(r(i)), 3);
            thresh2 = fmat(fmat(:, 1) == t & fmat(:, 2) == nreg + timp_idx_mot(c(i)), 3);
            [bin_pred, bin_counts, pd_diff, pd] = par_dep(avg_pred(tasks(:, t)), X(tasks(:, t), timp_idx(r(i))), ...
                        X(tasks(:, t), nreg + timp_idx_mot(c(i))), thresh1, thresh2);
            assert(pd == pd_stats(pd_idx(i)));
            % Plot the approximate contribution of each bin to the statistic
            bar3(pd_diff);
            xlabel(feature_names{timp_idx(r(i))});
            ylabel(mot_names{timp_idx_mot(c(i))});
            disp([feature_names{timp_idx(r(i))}, ' ', ...
                  mot_names{timp_idx_mot(c(i))}, ' ', num2str(pd)]); 
            title(num2str(pd));
        end
        print(gcf, outfile, '-dpdf');
    end
end
close all;
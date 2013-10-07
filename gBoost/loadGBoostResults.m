clearvars

clustdir = '/home/sofiakp/projects/Anshul/matlab/medusa/data/Jul13/hg19.encode.rna_asinh_rpk500_z2_clust30/';
new_indir = '/home/sofiakp/projects/Anshul/matlab/medusa/data/Jul13/hg19.encode.rna_rpk500_z2';
inpref = 'hocomoco_allOv_gBoost_test2';
params_file = fullfile(clustdir, 'runs', [inpref, '_params.m']);
run(params_file);
fold = 1;

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

load(files.fold_file);
load(fullfile(clustdir, ['clusters', num2str(fold), '.mat']));

plotdir = fullfile(clustdir, 'runs', 'stats', [inpref, '_folds', ...
                   num2str(fold), 'to', num2str(fold)]);
if ~isdir(plotdir)
    mkdir(plotdir);
end

[trloss, tsloss, pred, bestTasks, imp] = task_boost_model(fullfile(clustdir, ...
                                                  'runs', [inpref, ...
                    '.', num2str(fold), '.bin']));
tr = trainSets{fold};
ts = testSets{fold};
ntasks = size(tasks, 2);
niter = find(trloss == 0, 1, 'first');

all_ind = find(tr | ts);
get_accuracy(trloss, tsloss, pred, bestTasks, tr, ts, cexp, tasks, ...
                     levels, [plotdir, '/']);

sel_ind = 1:5:length(all_ind);
all_ind = all_ind(sel_ind);
all_tr_ind = ismember(all_ind, find(tr));
all_ts_ind = ismember(all_ind, find(ts));
tasks = tasks(sel_ind, :);
pred = pred(sel_ind);  
[exr exc] = ind2sub(size(tr), all_ind);
X = [pexp(exc, :) scores(exr, :)];
y = cexp(all_ind);

for t = 1:ntasks,

    %%%%% Feature importance
    [timp, timp_idx] = sort(imp(:, t), 'descend');
    ind = min(5, find(timp == 0, 1, 'first') - 1);
    if ind < 1
        continue
    end
    outfile = fullfile(plotdir, ['task', num2str(t), '_imp.pdf']);
    figure('Visible', 'off');
    for i = 1:ind
        subplot(2,3,i);
        plot(X(tasks(:, t), timp_idx(i)), pred(tasks(:, t)), '.');
        xlabel([feature_names{timp_idx(i)}, ' (', num2str(timp(i)), ')'], 'FontSize', 10);
        ylabel('Prediction', 'FontSize', 10);
        set(gca, 'FontSize', 10);
    end
    print(gcf, outfile, '-dpdf');

    %%%%% Feature dependence
    pd_stats = zeros(ind, ind);
    for i = 1:ind,
        for j = (i+1):ind,
            [bin_pred, bin_counts, pd_diff, pd_stats(i,j)] = ...
                par_dep(pred(tasks(:, t)), X(tasks(:, t), timp_idx(i)), ...
                        X(tasks(:, t), timp_idx(j)), 10);
        end
    end

    if max(pd_stats(:)) > 0.1
        [~, pd_idx] = sort(pd_stats(:), 'descend');
        [r, c] = ind2sub(size(pd_stats), pd_idx);
        
        outfile = fullfile(plotdir, ['task', num2str(t), '_dep_bar.pdf']);
        figure('Visible', 'off');    
        for i = 1:min(4, length(c))
            subplot(2, 2, i);
            [bin_pred, bin_counts, pd_diff, pd] = ...
                par_dep(pred(tasks(:, t)), X(tasks(:, t), timp_idx(r(i))), ...
                        X(tasks(:, t), timp_idx(c(i))), 10);
            assert(pd == pd_stats(pd_idx(i)));
            % Plot the approximate contribution of each bin to the statistic
            bar3(pd_diff);
            ylabel(feature_names{timp_idx(r(i))});
            xlabel(feature_names{timp_idx(c(i))});
            title(num2str(pd));
        end
        print(gcf, outfile, '-dpdf');
    end

    %%%%% Motif importance 
    [timp, timp_idx_mot] = sort(imp((nreg + 1):end, t), 'descend');
    mot_ind = min(4, find(timp == 0, 1, 'first') - 1);
    if mot_ind < 1
        continue
    end
    outfile = fullfile(plotdir, ['task', num2str(t), '_mot_imp.pdf']);
    figure('Visible', 'off');
    for i = 1:mot_ind
        subplot(2,2,i);
        plot(X(tasks(:, t), nreg + timp_idx_mot(i)), pred(tasks(:, t)), '.');
        xlabel([mot_names{timp_idx_mot(i)}, ' (', num2str(timp(i)), ')'], 'FontSize', 10);
        ylabel('Prediction', 'FontSize', 10);
        set(gca, 'FontSize', 10);
    end
    print(gcf, outfile, '-dpdf');

    %%%%% TF-motif dependence
    pd_stats = zeros(ind, mot_ind);
    for i = 1:ind,
        for j = 1:mot_ind,
            if timp_idx(i) == nreg + timp_idx_mot(j)
                continue;
            end
            [bin_pred, bin_counts, pd_diff, pd_stats(i,j)] = ...
                par_dep(pred(tasks(:, t)), X(tasks(:, t), timp_idx(i)), ...
                        X(tasks(:, t), nreg + timp_idx_mot(j)), 10);
        end
    end

    if max(pd_stats(:)) > 0.1
        disp(['Task ', num2str(t)]);
        [~, pd_idx] = sort(pd_stats(:), 'descend');
        [r, c] = ind2sub(size(pd_stats), pd_idx);
        
        outfile = fullfile(plotdir, ['task', num2str(t), '_mot_dep_bar.pdf']);
        figure('Visible', 'off');    
        for i = 1:min(4, length(c))
            subplot(2, 2, i);
            [bin_pred, bin_counts, pd_diff, pd] = ...
                par_dep(pred(tasks(:, t)), X(tasks(:, t), timp_idx(r(i))), ...
                        X(tasks(:, t), nreg + timp_idx_mot(c(i))), ...
                        10);
            assert(pd == pd_stats(pd_idx(i)));
            % Plot the approximate contribution of each bin to the statistic
            bar3(pd_diff);
            ylabel(feature_names{timp_idx(r(i))});
            xlabel(mot_names{timp_idx_mot(c(i))});
            title(num2str(pd));
        end
        print(gcf, outfile, '-dpdf');
    end
end
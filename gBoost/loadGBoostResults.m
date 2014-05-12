clearvars

clustdir = '/home/sofiakp/projects/Anshul/matlab/medusa/data/Jul13/hg19.encode.rna_asinh_rpk500_z2_clust30_noMerge/';
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

if ~isempty(files.score_files{1})
  load(files.score_files{1});
  sel_mot = strcmp(pssm_source,'hocomoco');
  scores = scores(sel_mot, :)';
  mot_names = mot_names(sel_mot);
  mot_names = strrep(mot_names, '_', '.');
  nmot = sum(sel_mot);
  feature_names = [parents_names; mot_names];
else
  nmot = 0;
  feature_names = [parents_names];
end
nfeat = length(feature_names);

niter = 3000;

load(files.fold_file);
trainSets = trainSets(folds);
testSets = testSets(folds);

load(fullfile(clustdir, ['clusters1.mat']));
ntasks = size(tasks, 2);

plotdir = fullfile(clustdir, 'runs', 'stats', [inpref, '_folds', num2str(min(folds)), 'to', num2str(max(folds)), '_iter', num2str(niter)]);
if ~isdir(plotdir)
    mkdir(plotdir);
end

tr = trainSets{folds(1)};
ts = testSets{folds(1)};

all_ind = find(tr | ts);
[exr exc] = ind2sub(size(tr), all_ind);
if ~isempty(files.score_files{1})
  X = [pexp(exc, :) scores(exr, :)];
else
  X = pexp(exc, :);
end
y = cexp(all_ind);

trloss_cell_all = cell(nfolds, 1);
tsloss_cell_all = cell(nfolds, 1);
pred_all = cell(nfolds, 1);
best_tasks_all = cell(nfolds, 1);
imp = zeros(nfeat, ntasks);
clust_imp = zeros(size(imp, 1), 30); % (i,j) is > 0 if feature i is used in any fold in cluster j
fmat = [];
for f = 1:nfolds
  [trloss_cell_all{f}, tsloss_cell_all{f}, pred_all{f}, best_tasks_all{f}, imp_tmp, fmat_tmp] = ...
    task_boost_model(fullfile(clustdir, 'runs', [inpref, '.', num2str(folds(f)), '.bin']), niter);
  istr = ismember(all_ind, find(trainSets{f}));
  
  for t = 1:ntasks
    % Get "baseline error" and use it to normalize the importance.
    base_loss = sum((y(tasks(:, t) & istr) - mean(y(tasks(:, t) & istr))).^2);
    imp_tmp(:, t) = imp_tmp(:, t) / base_loss;
  end

  % Normalize the importance by the number of times each task was selected
  imp = imp + imp_tmp ./ max(repmat(accumarray(best_tasks_all{f}, ones(niter, 1), [ntasks, 1])', nfeat, 1), 1);
  fmat = [fmat; fmat_tmp];
end
%%
for t = 1:30
  for c = 1:ntasks
    if any(tasks(:, t) & tasks(:, c))
      clust_imp(:, t) = clust_imp(:, t) + imp(:, c);
    end
  end
end

disp(['Averare % features per cluster: ' num2str(mean(sum(clust_imp > 0, 1)) / nfeat)])
nts = cellfun(@(x) nnz(x), testSets)';
ntr = cellfun(@(x) nnz(x), trainSets)'; 

[trloss_all, tsloss_all, trr2_all, tsr2_all, auc_all] = get_accuracy(trloss_cell_all, tsloss_cell_all, ...
  pred_all, best_tasks_all, trainSets, testSets, cexp, tasks, levels, niter, [plotdir, '/']);
disp(['Mean loss: ' num2str(mean(tsloss_all(end,:) ./ nts))])
disp(['Median loss: ' num2str(median(tsloss_all(end,:) ./ nts))])
save(fullfile(plotdir, 'stats.mat'), 'trloss_all', 'tsloss_all', 'trr2_all', 'tsr2_all', 'auc_all', 'fmat', 'imp');

% nout_feat = 10;
% for k = 1:30
%   outfile = fullfile(plotdir, [inpref, '_clust', num2str(k), '_imp_feat_', num2str(nout_feat), '.txt']);
%   [~, idx] = sort(abs(clust_imp(:, k)), 1, 'descend');
%   f = fopen(outfile, 'w');
%   fprintf(f, '%s\n', feature_names{idx(1:nout_feat)});
%   fclose(f);
% end

avg_pred = pred_all{1};
for f = 2:nfolds
  avg_pred = avg_pred + pred_all{f};
end
avg_pred = avg_pred / nfolds;
%%
for t = 56:ntasks,
  
  base_imp = median(imp(:, t));
  
  %%%%% TF importance
  [timp, timp_idx] = sort(imp(1:nreg, t), 1, 'descend');
  outfile = fullfile(plotdir, ['task', num2str(t), '_imp_tf_names.txt']);
  f = fopen(outfile, 'w');
  fprintf(f, '%s\n', feature_names{timp_idx(timp > base_imp)});
  fclose(f);
  outfile = fullfile(plotdir, ['task', num2str(t), '_imp.pdf']);
  figure('Visible', 'off');
  for i = 1:min(6, sum(timp > base_imp))
    subplot(2,3,i);
    plot(X(tasks(:, t), timp_idx(i)), avg_pred(tasks(:, t)), '.');
    xlabel([feature_names{timp_idx(i)}, ' (', num2str(timp(i)), ')'], 'FontSize', 10);
    ylabel('Prediction', 'FontSize', 10);
    set(gca, 'FontSize', 10);
  end
  print(gcf, outfile, '-dpdf');
  
  %%%% Feature dependence
  timp = timp(timp > base_imp);
  pd_stats = zeros(length(timp), length(timp));
  for i = 1:length(timp),
    thresh1 = fmat(fmat(:, 1) == t & fmat(:, 2) == timp_idx(i), 3);
    if length(unique(thresh1)) < 3
      continue
    end
    for j = (i+1):length(timp),
      thresh2 = fmat(fmat(:, 1) == t & fmat(:, 2) == timp_idx(j), 3);
      if length(unique(thresh2)) < 3
        continue
      end
      [bin_pred, bin_counts, pd_diff, pd_stats(i,j)] = par_dep(avg_pred(tasks(:, t)), ...
        X(tasks(:, t), timp_idx(i)), X(tasks(:, t), timp_idx(j)), thresh1, thresh2);
    end
  end
  pd_stats(abs(pd_stats - 1) < 1e-15) = 0;
  
  if max(pd_stats(:)) > 0.1
    [~, pd_idx] = sort(pd_stats(:), 1, 'descend');
    [r, c] = ind2sub(size(pd_stats), pd_idx);
    
    outfile = fullfile(plotdir, ['task', num2str(t), '_dep_bar.pdf']);
    figure('Visible', 'off');
    for i = 1:min(6, sum(pd_stats(:) > 0))
      subplot(2, 3, i);
      thresh1 = fmat(fmat(:, 1) == t & fmat(:, 2) == timp_idx(r(i)), 3);
      thresh2 = fmat(fmat(:, 1) == t & fmat(:, 2) == timp_idx(c(i)), 3);
      [bin_pred, bin_counts, pd_diff, pd, cuts1, cuts2] = par_dep(avg_pred(tasks(:, t)), X(tasks(:, t), timp_idx(r(i))), ...
        X(tasks(:, t), timp_idx(c(i))), thresh1, thresh2);
      assert(pd == pd_stats(pd_idx(i)));
      % Plot the approximate contribution of each bin to the statistic
      pd_diff(isnan(pd_diff)) = nanmean(pd_diff(:));
      bar3c(pd_diff);
      set(gca, 'FontSize', 5);
      set(gca, 'XTickLabel', cuts1);
      set(gca, 'YTickLabel', cuts2);
      ylabel(feature_names{timp_idx(r(i))});
      xlabel(feature_names{timp_idx(c(i))});
      disp([feature_names{timp_idx(r(i))}, ' ', feature_names{timp_idx(c(i))}, ' ', num2str(pd)]);
      title(num2str(pd));
    end
    print(gcf, outfile, '-dpdf');
  end
  
  %%%%% Motif importance
  [timp, timp_idx_mot] = sort(imp((nreg + 1):end, t), 'descend');
  mot_ind = min(6, find(timp == 0, 1, 'first') - 1);
  outfile = fullfile(plotdir, ['task', num2str(t), '_mot_imp.pdf']);
  figure('Visible', 'off');
  for i = 1:min(6, sum(timp > 0))
    subplot(2,3,i);
    plot(X(tasks(:, t), nreg + timp_idx_mot(i)), avg_pred(tasks(:, t)), '.');
    xlabel([mot_names{timp_idx_mot(i)}, ' (', num2str(timp(i)), ')'], 'FontSize', 10);
    ylabel('Prediction', 'FontSize', 10);
    set(gca, 'FontSize', 10);
  end
  print(gcf, outfile, '-dpdf');
  
  %%%%% TF-motif dependence
  timp_idx = find(imp(1:nreg, t) > 0);
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
  pd_stats(abs(pd_stats - 1) < 1e-15) = 0;
  
  if max(pd_stats(:)) > 0.1
    disp(['Task ', num2str(t)]);
    [~, pd_idx] = sort(pd_stats(:), 'descend');
    [r, c] = ind2sub(size(pd_stats), pd_idx);
    
    outfile = fullfile(plotdir, ['task', num2str(t), '_mot_dep_bar.pdf']);
    figure('Visible', 'off');
    for i = 1:min([6, length(c), sum(pd_stats(:) > 0)])
      subplot(2, 3, i);
      thresh1 = fmat(fmat(:, 1) == t & fmat(:, 2) == timp_idx(r(i)), 3);
      thresh2 = fmat(fmat(:, 1) == t & fmat(:, 2) == nreg + timp_idx_mot(c(i)), 3);
      [bin_pred, bin_counts, pd_diff, pd, cuts1, cuts2] = par_dep(avg_pred(tasks(:, t)), X(tasks(:, t), timp_idx(r(i))), ...
        X(tasks(:, t), nreg + timp_idx_mot(c(i))), thresh1, thresh2);
      assert(pd == pd_stats(pd_idx(i)));
      % Plot the approximate contribution of each bin to the statistic
      bar3c(pd_diff);
      set(gca, 'FontSize', 5);
      set(gca, 'XTickLabel', cuts1);
      set(gca, 'YTickLabel', cuts2); 
      xlabel(feature_names{timp_idx(r(i))});
      ylabel(mot_names{timp_idx_mot(c(i))});
      disp([feature_names{timp_idx(r(i))}, ' ', mot_names{timp_idx_mot(c(i))}, ' ', num2str(pd)]);
      title(num2str(pd));
    end
    print(gcf, outfile, '-dpdf');
  end
end
close all;
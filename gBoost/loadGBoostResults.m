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
gene_imp = zeros(size(imp, 1), ngen);
for t = 1:ntasks
  gene_imp(imp(:, t) > 0, tasks(1:ngen, t)) = 1;
end

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
  
  % Get "baseline error":
  base_loss = sum((y(tasks(:, t)) - mean(y(tasks(:, t)))).^2);
  base_imp = .1 * base_loss;
  
  %%%%% TF importance
  [timp, timp_idx] = sort(imp(1:nreg, t), 'descend');
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
  
  %%%%% Feature dependence
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
    [~, pd_idx] = sort(pd_stats(:), 'descend');
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
  mot_ind = min(10, find(timp == 0, 1, 'first') - 1);
  outfile = fullfile(plotdir, ['task', num2str(t), '_mot_imp.pdf']);
  figure('Visible', 'off');
  for i = 1:min(10, sum(timp > 0))
    subplot(2,5,i);
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
    for i = 1:min(6, length(c))
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
function [trloss_all, tsloss_all, trr2_all, tsr2_all, auc_all] = get_accuracy(...
  trloss_cell_all, tsloss_cell_all, pred_all, best_tasks_all, tr_all, ts_all, ...
  cexp, tasks, levels, niter, outpref)
% Loads Booster results from a set of folds and computes performance
% statistics.
% trloss_cell_all,...,best_tasks_all : cell arrays, each element is the
% results of a call to task_boost_model for a specific fold.
% tr_all, ts_all: cell arrays with matrices of training and test set
% indicators

nfolds = length(trloss_cell_all);
assert(length(tsloss_cell_all) == nfolds);
assert(length(pred_all) == nfolds);
assert(length(best_tasks_all) == nfolds);
assert(length(tr_all) >= nfolds);
assert(length(ts_all) >= nfolds);

nlevels = max(levels);
cum_levels = zeros(niter, nlevels);
ntasks = size(tasks, 2);

trloss_all = zeros(niter, nfolds);
tsloss_all = zeros(niter, nfolds);
trr2_all = zeros(niter, nfolds);
tsr2_all = zeros(niter, nfolds);
task_loss = zeros(ntasks, 2); % Average loss per task
nbest_tasks = zeros(ntasks, 1); % Number of times each task was selected as the best task
auc_cuts = [0.5 1 2]; % Will compute AUC for values > cut or < -cut.
ncuts = length(auc_cuts);
auc_all = zeros(nfolds, ncuts + 1);
aucx = cell(ncuts + 1, 1);
aucy = cell(ncuts + 1, 1);

for f = 1:nfolds
  trloss = trloss_cell_all{f}(1:niter);
  tsloss = tsloss_cell_all{f}(1:niter);
  pred = pred_all{f};
  best_tasks = best_tasks_all{f}(1:niter);
  tr = tr_all{f};
  ts = ts_all{f}; 
  
  if length(trloss) < niter || trloss(niter) == 0
    error('MyErr:get_accuracy', ['Fold ', num2str(f), ' does not have enough iterations']);
  end
  
  all_ind = find(tr | ts);
  y = cexp(all_ind);
  all_tr_ind = ismember(all_ind, find(tr));
  all_ts_ind = ismember(all_ind, find(ts));
  trr2_all(:, f) = 1 - trloss / sum((y(all_tr_ind) - mean(y(all_tr_ind))).^2);
  tsr2_all(:, f) = 1 - tsloss(1:niter) / sum((y(all_ts_ind) - mean(y(all_ts_ind))).^2);
  trloss_all(:, f) = trloss;
  tsloss_all(:, f) = tsloss;
  
  % I should be averaging the x and y values across folds, but the plot
  % doesn't change noticeably across folds...
  [aucx{1}, aucy{1}, ~, auc_all(f, 1)] = perfcurve(double(cexp(all_ts_ind) > 0), pred(all_ts_ind), 1);
  for c = 1:ncuts
    bin = zeros(size(cexp));
    bin(cexp > auc_cuts(c)) = 1;
    bin(cexp < -auc_cuts(c)) = -1;
    [aucx{c + 1}, aucy{c + 1}, ~, auc_all(f, c + 1)] = perfcurve(bin(all_ts_ind & bin(:) ~= 0), pred(all_ts_ind & bin(:) ~= 0), 1);
  end

  % Get a unique set of points in the x-axis (FPR) of the ROC curves for all
  % folds.
%   [auc_x, uniq_idx] = unique(auc_x);
%   if nnz(good_folds) == 1
%     auc_x_ref = auc_x; % Store this set of points to use as reference for all folds
%     auc_y = auc_y(uniq_idx);
%   else
%     auc_y = interp1(auc_x, auc_y(uniq_idx), auc_x_ref); % Use interpolation to get the y coordinate (TPR) for the reference points.
%   end
%   all_auc_y = [all_auc_y auc_y];
  
  for t = 1:ntasks,
    task_loss(t, 1) = task_loss(t, 1) + mean((pred(all_tr_ind & tasks(:, t)) - y(all_tr_ind & tasks(:, t))).^2);
    task_loss(t, 2) = task_loss(t, 2) + mean((pred(all_ts_ind & tasks(:, t)) - y(all_ts_ind & tasks(:, t))).^2);
  end
  nbest_tasks = nbest_tasks + accumarray(best_tasks(1:niter), ones(niter, 1), [ntasks, 1], @sum, 0);

  for i = 1:nlevels
    cum_levels(:, i) = cum_levels(:, i) + cumsum(levels(best_tasks(1:niter)) == i);
  end
end

nts = repmat(cellfun(@(x) nnz(x), ts_all)', niter, 1);
ntr = repmat(cellfun(@(x) nnz(x), tr_all)', niter, 1); 

figure('Visible', 'off');
subplot(2,2,1);
plot(1:niter, mean(trloss_all ./ ntr, 2), '-b', 1:niter, mean(tsloss_all ./ nts, 2), '--b');
xlabel('Iteration', 'FontSize', 10);
ylabel('Loss', 'FontSize', 10);
legend({['Train (', num2str(mean(trloss_all(end, :) ./ ntr(end, :))), ')'], ...
  ['Test (', num2str(mean(tsloss_all(end, :) ./ nts(end, :))), ')']}, 'Location', 'NorthEast');
set(gca, 'FontSize', 10);
subplot(2,2,2);
plot(1:niter, mean(trr2_all, 2), '-b', 1:niter, mean(tsr2_all, 2), '--b');
xlabel('Iteration', 'FontSize', 10);
ylabel('R-squared', 'FontSize', 10);
legend({'Train', 'Test'}, 'Location', 'SouthEast');
set(gca, 'FontSize', 10);
subplot(2,2,3);
plot(aucx{1}, aucy{1}, 'b', 'LineWidth', 0.6); hold on;
styles = {'g', 'b--', 'g--'};
for c = 1:ncuts,
  plot(aucx{c + 1}, aucy{c + 1}, styles{c}, 'LineWidth', 0.6);
end
legends = {['All (', num2str(mean(auc_all(:, 1))), ')']};
for c = 1:ncuts
  legends = [legends, ['abs(z) > ', num2str(auc_cuts(c)), ' (', num2str(mean(auc_all(:, c + 1))), ')']];
end
legend(legends, 'Location', 'SouthEast');
xlabel('FPR', 'FontSize', 10);
ylabel('TPR', 'FontSize', 10);
set(gca, 'FontSize', 10);
print(gcf, [outpref, 'acc.pdf'], '-dpdf');

task_loss = task_loss / nfolds;
nbest_tasks = nbest_tasks / nfolds;
cum_levels = cum_levels / nfolds;

figure('Visible', 'off');
subplot(2,2,1);
bar(1:ntasks, nbest_tasks);
xlabel('Task', 'FontSize', 10);
ylabel('# times selected', 'FontSize', 10);
set(gca, 'FontSize', 10);
subplot(2,2,2);
bar(1:ntasks, task_loss);
xlabel('Task', 'FontSize', 10);
ylabel('Avg loss', 'FontSize', 10);
legend({'Train', 'Test'}, 'Location', 'NorthEast');
set(gca, 'FontSize', 10);
subplot(2,2,3);
plot(1:niter, cum_levels);
xlabel('Iteration', 'FontSize', 10);
ylabel('Cumulative level sum', 'FontSize', 10);
legend(arrayfun(@(x) {num2str(x)}, 1:nlevels), 'Location', 'NorthWest');
set(gca, 'FontSize', 10);
print(gcf, [outpref, 'task_stats.pdf'], '-dpdf');
end

%function compute_tp(pred, cexp, ind)
%acc = zeros(3, 1);
%end
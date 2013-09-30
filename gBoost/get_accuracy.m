function get_accuracy(trloss, tsloss, pred, best_tasks, tr, ts, cexp, ...
                              tasks, levels, outpref)

if trloss(end) > 0
    niter = size(trloss, 1);
else 
    niter = find(trloss == 0, 1, 'first') - 1;
    assert(~isempty(niter));
end
ntasks = size(tasks, 2);

all_ind = find(tr | ts);
y = cexp(all_ind);
all_tr_ind = ismember(all_ind, find(tr));
all_ts_ind = ismember(all_ind, find(ts));
trr2 = 1 - trloss(1:niter) / sum((y(all_tr_ind) - mean(y(all_tr_ind))).^2);
tsr2 = 1 - tsloss(1:niter) / sum((y(all_ts_ind) - mean(y(all_ts_ind))).^2);
[aucx, aucy, thresh, auc] = perfcurve(double(cexp(all_ts_ind) > 0), pred(all_ts_ind), 1);
bin = zeros(size(cexp));
bin(cexp > 1) = 1;
bin(cexp < -1) = -1;
[aucx_z1, aucy_z1, ~, auc_z1] = perfcurve(bin(all_ts_ind & bin(:) ~= 0), pred(all_ts_ind & bin(:) ~= 0), 1);

% Average loss per task
task_loss = zeros(ntasks, 2);
for t = 1:ntasks,
    task_loss(t, 1) = mean((pred(all_tr_ind & tasks(:, t)) - ...
                            y(all_tr_ind & tasks(:, t))).^2);
    task_loss(t, 2) = mean((pred(all_ts_ind & tasks(:, t)) - ...
                            y(all_ts_ind & tasks(:, t))).^2);
end
nbest_tasks = accumarray(best_tasks(1:niter), ones(size(best_tasks)), [ntasks, 1], @sum, 0);

nlevels = max(levels);
cum_levels = zeros(niter, nlevels);
for i = 1:nlevels
    cum_levels(:, i) = cumsum(levels(best_tasks(1:niter)) == i);
end

figure('Visible', 'off');
subplot(2,2,1);
plot(1:niter, trloss(1:niter) / nnz(tr), '-b', 1:niter, tsloss(1:niter) ...
     / nnz(ts), '--b');
xlabel('Iteration', 'FontSize', 10);
ylabel('Loss', 'FontSize', 10);
legend({'Train', 'Test'}, 'Location', 'NorthEast');
set(gca, 'FontSize', 10);
subplot(2,2,2);
plot(1:niter, trr2, '-b', 1:niter, tsr2, '--b');
xlabel('Iteration', 'FontSize', 10);
ylabel('R-squared', 'FontSize', 10);
legend({'Train', 'Test'}, 'Location', 'SouthEast');
set(gca, 'FontSize', 10);
subplot(2,2,3);
plot(aucx, aucy, 'b', aucx_z1, aucy_z1, 'g', 'LineWidth', 1);
%text(0.5, 0.5, {['AUC = ', num2str(auc)], [, 'FontSize', 10);
legend({['All (', num2str(auc), ')'], ['abs(z) > 1 (', num2str(auc_z1), ...
                    ')']}, 'Location', 'SouthEast');
xlabel('FPR', 'FontSize', 10);
ylabel('TPR', 'FontSize', 10);
set(gca, 'FontSize', 10);
%subplot(2,2,4);
%plot(aucx_z1, aucy_z1, 'LineWidth', 3);
%text(0.5, 0.5, ['AUC = ', num2str(auc_z1)], 'FontSize', 10);
%xlabel('FPR', 'FontSize', 10);
%ylabel('TPR', 'FontSize', 10);
%set(gca, 'FontSize', 10);
print(gcf, [outpref, 'acc.pdf'], '-dpdf');

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
legend(arrayfun(@(x) {num2str(x)}, 1:niter), 'Location', 'NorthWest');
set(gca, 'FontSize', 10);
print(gcf, [outpref, 'task_stats.pdf'], '-dpdf');
end

%function compute_tp(pred, cexp, ind)
%acc = zeros(3, 1);
%end
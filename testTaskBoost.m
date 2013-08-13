function testTaskBoost(params_file, outfile_root, folds)

run(params_file);
 
outdir = fullfile(clust_dir, 'runs');
if ~isdir(outdir)
 mkdir(outdir);
end
outfile_root = fullfile(outdir, outfile_root);

load(files.target_file);
[ngen, nexp] = size(cexp);

load(files.reg_file);
pexp = pexp';
nreg = size(pexp, 2);

load(files.score_files{1});
sel_mot = strcmp(pssm_source,'hocomoco');
scores = scores(sel_mot, :)';
mot_names = mot_names(sel_mot);
nmot = sum(sel_mot);

load(files.fold_file);
load(fullfile(clust_dir, 'clusters.mat'));

for f = 1:length(folds)
  outfile = [outfile_root, '.', num2str(folds(f)), '.mat'];
  tr = trainSets{folds(f)};
  ts = testSets{folds(f)};
  
  if exist(outfile, 'file')
    warning('Warning:testTaskBoost', 'Output file %s exists. Skipping.', outfile);
    continue;
  end
  naive_task_boost(tasks, cexp, pexp, scores, tr, ts, params, train_params, outfile);
end
end

function naive_task_boost(tasks, cexp, pexp, scores, tr, ts, params, train_params, outfile)
ntasks = size(tasks, 2);

trstats = zeros(params.max_iter, 3);
tsstats = zeros(params.max_iter, 3);
models = cell(params.max_iter, 1);

pred = zeros(size(cexp));
best_task = zeros(params.max_iter, 1);
task_models = cell(ntasks, 1);
task_err = inf(ntasks, 1);


for i = 1:params.max_iter
  
  for k = 1:(ntasks - 2),
    sel_genes = tasks(:, k);
        
    [tr_r tr_c] = find(bsxfun(@times, tr, sel_genes));
    X = [pexp(tr_c, :) scores(tr_r, :)];
    tr_ind = sub2ind(size(cexp), tr_r, tr_c);
    other_tr_ind = find(bsxfun(@times, tr, ~sel_genes));
    
    if i == 1 || any(sel_genes & tasks(:, best_task(i - 1)))
      task_models{k} = SQBMatrixTrain(single(X), cexp(tr_ind) - pred(tr_ind), uint32(1), train_params);
    end
    pred_tmp = SQBMatrixPredict(task_models{k}, single(X));
    task_err(k) = sum((pred(other_tr_ind) - cexp(other_tr_ind)).^2) + sum((pred_tmp + pred(tr_ind) - cexp(tr_ind)).^2);
  end
  
  [min_err, best_task(i)] = min(task_err);
  models{i} = task_models{best_task(i)};
  
  sel_genes = tasks(:, best_task(i));
  [sel_r sel_c] = find(bsxfun(@times, tr | ts, sel_genes));
  X = [pexp(sel_c, :) scores(sel_r, :)];
  sel_ind = sub2ind(size(cexp), sel_r, sel_c);
  pred_tmp = SQBMatrixPredict(models{i}, single(X));
  pred(sel_ind) = pred(sel_ind) + pred_tmp;
  
  trstats(i, 1) = corr(pred(tr), cexp(tr));
  tsstats(i, 1) = corr(pred(ts), cexp(ts));
  trstats(i, 2) = 1 - sum((pred(tr) - cexp(tr)).^2) / sum((cexp(tr) - mean(cexp(tr))).^2);
  tsstats(i, 2) = 1 - sum((pred(ts) - cexp(ts)).^2) / sum((cexp(ts) - mean(cexp(ts))).^2);
  trstats(i, 3) = min_err / nnz(tr);
  tsstats(i, 3) = sum((pred(ts) - cexp(ts)).^2) / nnz(ts);
  
  save(outfile, 'trstats', 'tsstats', 'models', 'best_task', 'pred');
end
end
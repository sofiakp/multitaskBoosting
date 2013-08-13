function testTaskBoost(params_file, outfile_root)

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

outfile = [outfile_root, '.1.mat'];
tr = trainSets{1};
ts = testSets{1};

naive_task_boost(tasks, cexp, pexp, scores, tr, ts, params, train_params, outfile);
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
  [tr_r tr_c] = find(bsxfun(@times, tr, sel_genes));
  [ts_r ts_c] = find(bsxfun(@times, ts, sel_genes));
  X = [pexp([tr_c; ts_c], :) scores([tr_r; ts_r], :)];
  tr_ind = sub2ind(size(cexp), tr_r, tr_c);
  ts_ind = sub2ind(size(cexp), ts_r, ts_c);
  pred_tmp = SQBMatrixPredict(models{i}, single(X));
  pred([tr_ind; ts_ind]) = pred([tr_ind; ts_ind]) + pred_tmp;
  
  trstats(i, 1) = corr(pred(tr_ind), cexp(tr_ind));
  tsstats(i, 1) = corr(pred(ts_ind), cexp(ts_ind));
  trstats(i, 2) = 1 - sum((pred(tr_ind) - cexp(tr_ind)).^2) / sum((cexp(tr_ind) - mean(cexp(tr_ind))).^2);
  tsstats(i, 2) = 1 - sum((pred(ts_ind) - cexp(ts_ind)).^2) / sum((cexp(ts_ind) - mean(cexp(ts_ind))).^2);
  trstats(i, 3) = min_err / nnz(tr);
  tsstats(i, 3) = sum((pred(ts) - cexp(ts)).^2) / nnz(ts);
  
  save(outfile, 'trstats', 'tsstats', 'models', 'best_task', 'pred');
end
end
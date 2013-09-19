function runBooster(params_file, outfile_root, resume, folds)
rs = RandStream('mt19937ar', 'Seed', 1);
RandStream.setGlobalStream(rs);

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

for f = 1:length(folds)
  load(fullfile(clust_dir, ['clusters', num2str(folds(f)), '.mat']));
  outfile = [outfile_root, '.', num2str(folds(f)), '.bin'];
  tr = trainSets{folds(f)};
  ts = testSets{folds(f)};
  ntasks = size(tasks, 2);
  
  all_ind = find(tr | ts); % The rows of tasks correspond to these indices of cexp
  all_ind = all_ind(1:10:end); % FOR TEST PURPOSES ONLY
  tasks = tasks(1:10:end, :);
  all_tr_ind = ismember(all_ind, find(tr)); % Indicators of rows of tasks that correspond to training examples
  all_ts_ind = ismember(all_ind, find(ts));
  
  [exr exc] = ind2sub(size(tr), all_ind);
  X = [pexp(exc, :) scores(exr, :)];
  
  taskOv = zeros(ntasks, ntasks);
  for i = 1:ntasks,
    for j = 1:ntasks,
      taskOv(i, j) = any(tasks(:, i) & tasks(:, j) & all_tr_ind);
    end
  end
  
  if exist(outfile, 'file') && ~resume
    warning('Warning:runBooster', 'Output file %s exists. Skipping.', outfile);
    continue;
  elseif ~exist(outfile, 'file') && resume
    warning('Warning:runBooster', 'Output file does not exist, setting resume to false');
    resume = false;
  end
  [trloss, tsloss, pred, imp] = task_boost_learn(sparse(bsxfun(@times, tasks, all_tr_ind)), ...
    sparse(bsxfun(@times, tasks, all_ts_ind)), taskOv, X, cexp(all_ind), ...
    params.niter, train_params.maxDepth, train_params.minNodes, 1e-6, train_params.fracFeat, train_params.shrink, outfile);
end
end
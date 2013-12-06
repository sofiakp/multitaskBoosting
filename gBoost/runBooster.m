function runBooster(params_file, outfile_root, resume, folds, debug)
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

has_scores = ~isempty(files.score_files{1});
if has_scores
    load(files.score_files{1});
    sel_mot = strcmp(pssm_source,'hocomoco');
    nmot = sum(sel_mot);
    scores = scores(sel_mot, :)'; % now scores is genes x mot
    scores = max(scores, 0); % Negative scores don't make much sense
    scores = scores ./ repmat(max(scores, [], 1), ngen, 1); % normalize by the maximum across each motif
    mot_names = mot_names(sel_mot);
end
load(files.fold_file);

load(fullfile(clust_dir, 'clusters1.mat'));

for f = 1:length(folds)
  outfile = [outfile_root, '.', num2str(folds(f)), '.bin'];
  tr = trainSets{folds(f)};
  ts = testSets{folds(f)};
  ntasks = size(tasks, 2);
  
  all_ind = find(tr | ts); % The rows of tasks correspond to these indices of cexp
  if debug
    all_ind = all_ind(1:10:end);
    tasks = tasks(1:10:end, :);
  end
  all_tr_ind = ismember(all_ind, find(tr)); % Indicators of rows of tasks that correspond to training examples
  all_ts_ind = ismember(all_ind, find(ts));
  
  [exr exc] = ind2sub(size(tr), all_ind);
  if has_scores
    if isfield(files, 'mot_activity_list') && ~isempty(files.mot_activity_list) && ...
        isfield(train_params, 'signal_weight') && train_params.signal_weight > 0
      % First col is the path to the file with the motif activities, second
      % col is experiment (cell line).
      fi = fopen(files.mot_activity_list, 'r');
      C = textscan(fi, '%s%s');
      fclose(fi);
      filenames = C{1};
      cell_lines = C{2};
      assert(all(ismember(exptnames, cell_lines)));
      
      X = scores(exr, :);
      
      for i = 1:nexp
        sel_file = strcmp(cell_lines, exptnames{i});
        if ~any(sel_file)
          error('MyErr:runBooster', ['No signal file provided for ', exptnames{i}]);
        elseif nnz(sel_file) > 1
          error('MyErr:runBooster', ['Multiple signal files provided for ', exptnames{i}]);
        else
          signal_struct = load(fullfile(indir, filenames{sel_file}));
          [~, midx] = ismember(mot_names, signal_struct.mot_names);
          assert(all(midx));
          signal_struct.signal = signal_struct.signal(midx, :)'; % transpose so that the result is genes x mot
          sel_ex = exc == i; % get examples corresponding to this experiment
          sel_gene = exr(sel_ex); % get the gene index of these examples
          X(sel_ex, :) = X(sel_ex, :) + train_params.signal_weight ...
              * signal_struct.signal(sel_gene, :) + ...
              train_params.signal_weight_inter * X(sel_ex, :) .* signal_struct.signal(sel_gene, :);
        end
      end
      X = [pexp(exc, :) X];
    else
      X = [pexp(exc, :) scores(exr, :)];
    end
  else
      X = pexp(exc, :);
  end
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
    sparse(bsxfun(@times, tasks, all_ts_ind)), levels, X, cexp(all_ind), ...
    params.niter, train_params.maxDepth, train_params.minNodes, 1e-6, train_params.fracFeat, train_params.shrink, ...
    resume, outfile);
end
end
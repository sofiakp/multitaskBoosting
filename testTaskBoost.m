function testTaskBoost()

outfile_root = 'hocomoco_sqb_test3';
indir = '/home/sofiakp/projects/Anshul/matlab/medusa/data/Jul13/hg19.encode.rna_rpk100_z2';
clust_dir = '/home/sofiakp/projects/Anshul/matlab/medusa/data/Jul13/hg19.encode.rna_asinh_rpk100_z2_clust30';

files = struct;
files.fold_file = fullfile(indir, 'folds10_examples.mat');
files.target_file = fullfile(indir,'targets_asinh.mat');
files.reg_file = fullfile(indir,'parents_regulatorList3_asinh.mat');
files.score_files = {fullfile(indir, 'hocomoco_pouya_stam_jaspar_proxRegions_norm_scores.mat')};

params.max_iter = 2;

train_params.loss = 'squaredloss';
train_params.shrinkageFactor = 0.01;
train_params.subsamplingFactor = 1;
train_params.maxTreeDepth = uint32(2);
train_params.randSeed = uint32(1);
 
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

trstats = zeros(params.max_iter, 2);
tsstats = zeros(params.max_iter, 2);
models = cell(params.max_iter, 1);

pred = ones(size(cexp));
best_task = zeros(params.max_iter, 1);

for i = 1:params.max_iter
  min_err = Inf;

  for k = 1:(ntasks - 1),
    sel_genes = tasks(:, k);
    [tr_r tr_c] = find(bsxfun(@times, tr, sel_genes));
    X = [pexp(tr_c, :) scores(tr_r, :)];
    tr_ind = sub2ind(size(cexp), tr_r, tr_c);
    
    model = SQBMatrixTrain(single(X), cexp(tr_ind) - pred(tr_ind), uint32(1), train_params);
    pred_tmp = SQBMatrixPredict(model, single(X));
    err = sum((pred_tmp - cexp(tr_ind) + pred(tr_ind)).^2);
    if err < min_err
      min_err = err;
      best_task(i) = k;
      models{i} = model;
    end
  end
  
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
  
  save(outfile, 'trstats', 'tsstats', 'models', 'best_task', 'pred');
end
end
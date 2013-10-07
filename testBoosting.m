clear all

outfile_root = 'hocomoco_sqb_test3';
indir = '/home/sofiakp/projects/Anshul/matlab/medusa/data/Jul13/hg19.encode.rna_rpk100_z2';
clust_dir = '/home/sofiakp/projects/Anshul/matlab/medusa/data/Jul13/hg19.encode.rna_asinh_rpk100_z2_clust30';

files = struct;
files.fold_file = fullfile(indir, 'folds10_examples.mat');
files.target_file = fullfile(indir,'targets_asinh.mat');
files.reg_file = fullfile(indir,'parents_regulatorList3_asinh.mat');
files.score_files = {fullfile(indir, 'hocomoco_pouya_stam_jaspar_proxRegions_norm_scores.mat')};

params.max_iter = uint32(10);
params.reg_mode = 'sqb';

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
nclust = size(centroids, 1);

trstats = zeros(nclust, 2);
tsstats = zeros(nclust, 2);
models = cell(nclust, 1);

outfile = [outfile_root, '.1.mat'];

sel_genes = km == 1;
[tr_r tr_c] = find(bsxfun(@times, trainSets{1}, sel_genes'));
[ts_r ts_c] = find(bsxfun(@times, testSets{1}, sel_genes'));
X = [pexp([tr_c; ts_c], :) scores([tr_r; ts_r], :)];
ntr = length(tr_r);
nts = length(ts_r);
tr_ind = sub2ind(size(cexp), tr_r, tr_c);
ts_ind = sub2ind(size(cexp), ts_r, ts_c);

trstats = zeros(2, 2);
tsstats = zeros(2, 2);

%%% These two produce the same result
% Version 1
model1 = SQBMatrixTrain(single(X(1:ntr, :)), cexp(tr_ind), params.max_iter, train_params);
pred1 = SQBMatrixPredict(model1, single(X));

trstats(1, 1) = corr(pred1(1:ntr), cexp(tr_ind));
tsstats(1, 1) = corr(pred1((ntr + 1):end), cexp(ts_ind));
trstats(1, 2) = 1 - sum((pred1(1:ntr) - cexp(tr_ind)).^2) / sum((cexp(tr_ind) - mean(cexp(tr_ind))).^2);
tsstats(1, 2) = 1 - sum((pred1((ntr + 1):end) - cexp(ts_ind)).^2) / sum((cexp(ts_ind) - mean(cexp(ts_ind))).^2);

% Version 2
niter = length(model1);
params.max_iter = uint32(1);
pred2 = zeros(ntr + nts, 1);
for i = 1:niter
  train_params.randSeed = uint32(i);
  model2 = SQBMatrixTrain(single(X(1:ntr, :)), cexp(tr_ind) - pred2(1:ntr), params.max_iter, train_params);
  pred2 = pred2 + SQBMatrixPredict(model2, single(X));
end
trstats(2, 1) = corr(pred2(1:ntr), cexp(tr_ind));
tsstats(2, 1) = corr(pred2((ntr + 1):end), cexp(ts_ind));
trstats(2, 2) = 1 - sum((pred2(1:ntr) - cexp(tr_ind)).^2) / sum((cexp(tr_ind) - mean(cexp(tr_ind))).^2);
tsstats(2, 2) = 1 - sum((pred2((ntr + 1):end) - cexp(ts_ind)).^2) / sum((cexp(ts_ind) - mean(cexp(ts_ind))).^2);
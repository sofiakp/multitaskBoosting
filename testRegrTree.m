clear all

outfile_root = 'hocomoco_sqb_test1';
params_file = '../../data/Jul13/hg19.encode.rna_asinh_rpk100_z2_clust30/runs/hocomoco_sqb_test1_params.m';
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

% Xr = zeros(numel(cexp), nreg);
% for i = 1:nexp,
%   Xr(((i - 1) * ngen + 1):(i * ngen), :) = repmat(pexp(:, i), ngen, 1);
% end

load(files.fold_file);
load(fullfile(clust_dir, 'clusters.mat'));
nclust = size(centroids, 1);

trstats = zeros(nclust, 2);
tsstats = zeros(nclust, 2);
models = cell(nclust, 1);

outfile = [outfile_root, '.1.mat'];

tr = trainSets{1};
ts = testSets{1};
pred = zeros(size(cexp));

for k = 1:nclust
  disp(['Starting cluster ', num2str(k)]);
  sel_genes = km == k;
  [tr_r tr_c] = find(bsxfun(@times, tr, sel_genes'));
  [ts_r ts_c] = find(bsxfun(@times, ts, sel_genes'));
  X = [pexp([tr_c; ts_c], :) scores([tr_r; ts_r], :)];
  ntr = length(tr_r);
  nts = length(ts_r);
  tr_ind = sub2ind(size(cexp), tr_r, tr_c);
  ts_ind = sub2ind(size(cexp), ts_r, ts_c);
  
  % Xr(sub2ind(size(Xr), [tr_r; ts_r], [tr_c; ts_c]), :)
  if strcmp(params.reg_mode, 'sqb')
    model = SQBMatrixTrain(single(X(1:ntr, :)), cexp(tr_ind), params.max_iter, train_params);
    pred([tr_ind; ts_ind]) = SQBMatrixPredict(model, single(X));
  else
    model = RegressionTree.fit(X(1:ntr, :), cexp(tr_ind), 'MinParent', train_params.min_par, 'PredictorNames', [parents_names; mot_names]);
    pred([tr_ind; ts_ind]) = predict(model, X);
  end
  trstats(k, 1) = corr(pred(tr_ind), cexp(tr_ind));
  tsstats(k, 1) = corr(pred(ts_ind), cexp(ts_ind));
  trstats(k, 2) = 1 - sum((pred(tr_ind) - cexp(tr_ind)).^2) / sum((cexp(tr_ind) - mean(cexp(tr_ind))).^2);
  tsstats(k, 2) = 1 - sum((pred(ts_ind) - cexp(ts_ind)).^2) / sum((cexp(ts_ind) - mean(cexp(ts_ind))).^2);
  
  models{k} = model;
end

trstats(k + 1, 1) = corr(pred(tr), cexp(tr));
tsstats(k + 1, 1) = corr(pred(ts), cexp(ts));
trstats(k + 1, 2) = 1 - sum((pred(tr) - cexp(tr)).^2) / sum((cexp(tr) - mean(cexp(tr))).^2);
tsstats(k + 1, 2) = 1 - sum((pred(ts) - cexp(ts)).^2) / sum((cexp(ts) - mean(cexp(ts))).^2);

save(outfile, 'models', 'trstats', 'tsstats');

clearvars

clustdir = '/home/sofiakp/projects/Anshul/matlab/medusa/data/Jul13/hg19.encode.rna_asinh_rpk500_z2_clust30/';
new_indir = '/home/sofiakp/projects/Anshul/matlab/medusa/data/Jul13/hg19.encode.rna_rpk500_z2';
inpref = 'hocomoco_allOv_gBoost_test4';
params_file = fullfile(clustdir, 'runs', [inpref, '_params.m']);
run(params_file);
folds = 1:1; 
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
trainSets = trainSets(folds);
testSets = testSets(folds);

load(fullfile(clustdir, ['clusters1.mat']));
ntasks = size(tasks, 2);

plotdir = fullfile(clustdir, 'runs', 'stats', [inpref, '_folds', num2str(min(folds)), 'to', num2str(max(folds))]);
if ~isdir(plotdir)
    mkdir(plotdir);
end

tr = trainSets{1};
ts = testSets{1};

all_ind = find(tr | ts);
[exr exc] = ind2sub(size(tr), all_ind);
if ~isempty(files.score_files{1})
  X = [pexp(exc, :) scores(exr, :)];
else
  X = pexp(exc, :);
end
y = cexp(all_ind);

trloss_cell_all = cell(nfolds, 1);
tsloss_cell_all = cell(nfolds, 1);
pred_all = cell(nfolds, 1);
best_tasks_all = cell(nfolds, 1);
imp = zeros(nfeat, ntasks);
clust_imp = zeros(size(imp, 1), 30); % (i,j) is > 0 if feature i is used in any fold in cluster j
fmat = [];
for f = 1:nfolds
  [trloss_cell_all{f}, tsloss_cell_all{f}, pred_all{f}, best_tasks_all{f}, imp_tmp, fmat_tmp] = ...
    task_boost_model(fullfile(clustdir, 'runs', [inpref, '.', num2str(folds(f)), '.bin']));
  istr = ismember(all_ind, find(trainSets{f}));
  
  for t = 1:ntasks
    % Get "baseline error" and use it to normalize the importance.
    base_loss = sum((y(tasks(:, t) & istr) - mean(y(tasks(:, t) & istr))).^2);
    imp_tmp(:, t) = imp_tmp(:, t) / base_loss;
  end

  % Normalize the importance by the number of times each task was selected
  imp = imp + imp_tmp ./ max(repmat(accumarray(best_tasks_all{f}, ones(niter, 1), [ntasks, 1])', nfeat, 1), 1);
  fmat = [fmat; fmat_tmp];
end
%%
[~, sidx] = sort(imp(:, 38), 1, 'descend');
sel_feat = sidx(1:1);
sel_expt = find(strcmp(exptnames, 'GM12878') | strcmp(exptnames, 'CD20+') | strcmp(exptnames, 'Monocytes-CD14+'));
sel_ex = tasks(:, 38) & ismember(exc, sel_expt);
ex_imp = task_boost_ex_imp(sparse(tasks), X, sel_feat, fullfile(clustdir, 'runs', [inpref, '.', num2str(folds(f)), '.bin']));
%%
oth_ex = tasks(:, 38) & ~ismember(exc, sel_expt);
%%
oth_imp = task_boost_ex_imp(sparse(bsxfun(@times, tasks, oth_ex)), X, sel_feat, fullfile(clustdir, 'runs', [inpref, '.', num2str(folds(f)), '.bin']));

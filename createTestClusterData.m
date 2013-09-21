clear all
rs = RandStream('mt19937ar', 'Seed', 1);
RandStream.setGlobalStream(rs);

indir = '/home/sofiakp/projects/Anshul/matlab/medusa/data/Jul13/hg19.encode.rna_rpk100_z2';

% This clusters the expression data (i.e. the output) which is not really
% what we want for multi-task regression, because in this way we're
% effectively using the output we're trying to predict.
%
% nclust = 30;
% 
% outdir = ['/home/sofiakp/projects/Anshul/matlab/medusa/data/Jul13/hg19.encode.rna_asinh_rpk100_z2_clust', num2str(nclust)];
% if ~isdir(outdir)
%  mkdir(outdir);
% end
% 
% load(fullfile(indir, 'targets_asinh.mat'));
% sel_genes = true(size(cexp, 1), 1);
% sel_expt = true(size(cexp, 2), 1);
% cexp = cexp(sel_genes, sel_expt);
% [ngen, nexp] = size(cexp);
% km = litekmeans(cexp', nclust);
% centroids = zeros(nclust, nexp);
% [~, kidx] = sort(km, 'ascend');
% 
% for k = 1:nclust
%   sel = km == k;
%   centroids(k, :) = mean(cexp(sel, :), 1);
% end  
% 
% c = clustergram(centroids, 'RowLabels', strcat('Cluster ', arrayfun(@num2str, 1:nclust, 'UniformOutput', false)'), ...
%   'ColumnLabels', exptnames, 'Colormap', redbluecmap, 'Linkage', 'ward');
% 
% Z = linkage(centroids, 'Ward');
% 
% tasks = false(length(km), 2*nclust - 1);
% for i = 1:nclust
%   tasks(:, i) = km == i;
% end
% for i = 1:size(Z, 1)
%   tasks(:, nclust + i) = tasks(:, Z(i, 1)) | tasks(:, Z(i, 2));
% end
% assert(all(tasks(:, end)));
% save(fullfile(outdir, 'clusters.mat'), 'km', 'centroids', 'tasks');
% dlmwrite(fullfile(outdir, 'clusters.txt'), centroids, 'delimiter', '\t');
% dlmwrite(fullfile(outdir, 'cluster_ind.txt'), km');
%%
% load(fullfile(indir, 'parents_regulatorList3_asinh.mat'));
% pexp = pexp'; % make pexp expt-by-regulators
% 
% load(fullfile(indir, 'hocomoco_pouya_stam_jaspar_proxRegions_norm_scores.mat'));
% sel_mot = strcmp(pssm_source,'hocomoco');
% scores = scores(sel_mot, :)';
% mot_names = mot_names(sel_mot);
% nmot = sum(sel_mot);
% 
% load(fullfile(indir, 'folds10_examples'));
% tr = trainSets{1};
% ts = testSets{1};
% [r, c] = find(tr | ts); % all examples
% X = [pexp(c, :) scores(r, :)];
%%
clear all
rs = RandStream('mt19937ar', 'Seed', 1);
RandStream.setGlobalStream(rs);

indir = '/home/sofiakp/projects/Anshul/matlab/medusa/data/Jul13/hg19.encode.rna_rpk100_z2';

nclust_pexp = 5;
nclust_scores = 30;
min_clust_size = 50;

outdir = ['/home/sofiakp/projects/Anshul/matlab/medusa/data/Jul13/hg19.encode.rna_asinh_rpk100_z2_pClust', ...
  num2str(nclust_pexp), '_sClust', num2str(nclust_scores)];
if ~isdir(outdir)
  mkdir(outdir);
end

load(fullfile(indir, 'folds10_examples'));
tr = trainSets{1};
ts = testSets{1};
[rows, cols] = find(tr | ts); % all examples

load(fullfile(indir, 'parents_regulatorList3_asinh.mat'));
% Cluster the cell lines based on the expression of regulators.
km = litekmeans(pexp, nclust_pexp);
centroids = zeros(nclust_pexp, size(pexp, 1)); % Expression of regulators in each of the clusters
for k = 1:nclust_pexp
  centroids(k, :) = nanmean(pexp(:, km == k), 2);
end
km_lines = km;

Z = linkage(centroids, 'Ward'); % Do a hiearchical clustering of the clusters
tot_npexp = 2 * nclust_pexp - 1;
levels_pexp = zeros(tot_npexp, 1);
% Create one task per cluster or combination of clusters. Put each training
% and testing example into the appropriate tasks.
tasks_pexp = false(length(rows), tot_npexp);
for i = 1:nclust_pexp
  tasks_pexp(:, i) = ismember(cols, find(km == i)); % find(km==i) gives the cell lines that belong to the i-th cluster
end
for i = 1:size(Z, 1)
  levels_pexp(nclust_pexp + i) = max(levels_pexp(Z(i, 1)), levels_pexp(Z(i,2))) + 1;
  tasks_pexp(:, nclust_pexp + i) = tasks_pexp(:, Z(i, 1)) | tasks_pexp(:, Z(i, 2));
end
%tasks_pexp = tasks_pexp(:, size(tasks_pexp, 2):-1:1);  % reverse so the clusters are roughly from largest to smallest
levels_pexp = max(levels_pexp) - levels_pexp;

load(fullfile(indir, 'hocomoco_pouya_stam_jaspar_proxRegions_norm_scores.mat'));
sel_mot = strcmp(pssm_source,'hocomoco');
scores = scores(sel_mot, :);
mot_names = mot_names(sel_mot);
nmot = sum(sel_mot);

% Cluster the genes based on the motif scores
km = litekmeans(scores, nclust_scores);
centroids = zeros(nclust_scores, nmot); % Average motif scores in each cluster
for k = 1:nclust_scores
  centroids(k, :) = nanmean(scores(:, km == k), 2);
end
km_scores = km;

Z = linkage(centroids, 'Ward');
tot_nscores = 2 * nclust_scores - 1;
levels_scores = zeros(tot_nscores, 1);
tasks_scores = false(length(rows), tot_nscores);
for i = 1:nclust_scores
  tasks_scores(:, i) = ismember(rows, find(km == i));
end
for i = 1:size(Z, 1)
  levels_scores(nclust_scores + i) = max(levels_scores(Z(i, 1)), levels_scores(Z(i,2))) + 1;
  tasks_scores(:, nclust_scores + i) = tasks_scores(:, Z(i, 1)) | tasks_scores(:, Z(i, 2));
end
%tasks_scores = tasks_scores(:, size(tasks_scores, 2):-1:1);
levels_scores = max(levels_scores) - levels_scores;
tasks_scores = tasks_scores(:, levels_scores > 0); % Remove the "root" task
levels_scores = levels_scores(levels_scores > 0);

levels = [levels_pexp; levels_scores]; 
tasks = [tasks_pexp tasks_scores];
%pord = repmat(1:tot_npexp, tot_nscores, 1); pord = pord(:); % 1 1 1... 1 2 2 2... 2 3 3 3...
%sord = repmat([1:tot_nscores]', tot_npexp, 1); % 1 2 3... 1 2 3... 1 2 3...
%tasks = tasks_pexp(:, pord) & tasks_scores(:, sord);
assert(size(levels, 1) == size(tasks, 2));
save(fullfile(outdir, 'clusters1.mat'), 'tasks', 'km_lines', 'km_scores', 'levels');
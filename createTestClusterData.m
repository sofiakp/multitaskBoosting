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

% First cluster the cell lines
load(fullfile(indir, 'parents_regulatorList3_asinh.mat'));

km = litekmeans(pexp, nclust_pexp);
centroids = zeros(nclust_pexp, size(pexp, 1));
for k = 1:nclust_pexp
  centroids(k, :) = mean(pexp(:, km == k), 2);
end

% Do a hiearchical clustering of the cell lines
Z = linkage(centroids, 'Ward');
tot_npexp = 2 * nclust_pexp - 1;
tasks_pexp = false(length(rows), tot_npexp);
for i = 1:nclust_pexp
  tasks_pexp(:, i) = ismember(cols, find(km == i));
end
for i = 1:size(Z, 1)
  tasks_pexp(:, nclust_pexp + i) = tasks_pexp(:, Z(i, 1)) | tasks_pexp(:, Z(i, 2));
end
tasks_pexp = tasks_pexp(:, size(tasks_pexp, 2):-1:1);

load(fullfile(indir, 'hocomoco_pouya_stam_jaspar_proxRegions_norm_scores.mat'));
sel_mot = strcmp(pssm_source,'hocomoco');
scores = scores(sel_mot, :);
mot_names = mot_names(sel_mot);
nmot = sum(sel_mot);

km = litekmeans(scores, nclust_scores);
centroids = zeros(nclust_scores, nmot);

for k = 1:nclust_scores
  centroids(k, :) = mean(scores(:, km == k), 2);
end

Z = linkage(centroids, 'Ward');
tot_nscores = 2 * nclust_scores - 1;
tasks_scores = false(length(rows), tot_nscores);
for i = 1:nclust_scores
  tasks_scores(:, i) = ismember(rows, find(km == i));
end
for i = 1:size(Z, 1)
  tasks_scores(:, nclust_scores + i) = tasks_scores(:, Z(i, 1)) | tasks_scores(:, Z(i, 2));
end
tasks_scores = tasks_scores(:, size(tasks_scores, 2):-1:1);

pord = repmat(1:tot_npexp, tot_nscores, 1); pord = pord(:);
sord = repmat([1:tot_nscores]', tot_npexp, 1);

tasks = tasks_pexp(:, pord) & tasks_scores(:, sord);
save(fullfile(outdir, 'clusters.mat'), 'tasks');
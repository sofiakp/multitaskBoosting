clear all
rs = RandStream('mt19937ar', 'Seed', 1);
RandStream.setGlobalStream(rs);

% Cluster the cell lines based on the expression of regulators and
% the genes based on the motifs.

indir = '/home/sofiakp/projects/Anshul/matlab/medusa/data/Jul13/hg19.encode.rna_rpk500_z2';

nclust_pexp = 21;
nclust_scores = 0;
min_clust_size = 50;
max_percent = 0.8; % Tasks containing more than this fraction of
                   % examples will be removed
asinh = false;
if asinh
   outdir = ['/home/sofiakp/projects/Anshul/matlab/medusa/data/Jul13/hg19.encode.rna_asinh_rpk500_z2', '_pClust', num2str(nclust_pexp), '_sClust', num2str(nclust_scores)];
else
 outdir = ['/home/sofiakp/projects/Anshul/matlab/medusa/data/Jul13/hg19.encode.rna_rpk500_z2', '_pClust', num2str(nclust_pexp), '_sClust', num2str(nclust_scores)];
end
if ~isdir(outdir)
  mkdir(outdir);
end

fold = 1;
outfile = fullfile(outdir, ['clusters', num2str(fold), '_info.txt']);
f = fopen(outfile, 'w');

load(fullfile(indir, 'folds10_examples'));
tr = trainSets{fold};
ts = testSets{fold};
[rows, cols] = find(tr | ts); % all examples
nex = length(cols);

load(fullfile(indir, 'parents_regulatorList3.mat'));
% Cluster the cell lines based on the expression of regulators.
if nclust_pexp < size(pexp, 2)
    km = litekmeans(pexp, nclust_pexp);
    centroids = zeros(nclust_pexp, size(pexp, 1)); % Expression of regulators in each of the clusters
    for k = 1:nclust_pexp
        centroids(k, :) = nanmean(pexp(:, km == k), 2);
    end
    Z = linkage(centroids, 'Ward'); % Do a hiearchical clustering of the clusters
else
    km = 1:size(pexp, 2);
    Z = linkage(pexp', 'Ward'); % Do a hiearchical clustering of
                                % the cell lines
    figure(); dendrogram(Z, 'Orientation', 'left', 'Labels', exptnames);
    print(gcf, fullfile(outdir, ['clusters', num2str(fold), '_pexp.pdf']), ...
          '-dpdf');
end

tot_npexp = 2 * nclust_pexp - 1;
levels_pexp = zeros(tot_npexp, 1);
% Create one task per cluster or combination of clusters. Put each training
% and testing example into the appropriate tasks.
tasks_pexp = false(length(rows), tot_npexp);
for i = 1:nclust_pexp
  tasks_pexp(:, i) = ismember(cols, find(km == i)); % find(km==i) gives the cell lines that belong to the i-th cluster
  fprintf(f, 'Pexp Cluster %d: %s\n', i, exptnames{km == i});
end
for i = 1:size(Z, 1)
  levels_pexp(nclust_pexp + i) = max(levels_pexp(Z(i, 1)), levels_pexp(Z(i,2))) + 1;
  tasks_pexp(:, nclust_pexp + i) = tasks_pexp(:, Z(i, 1)) | tasks_pexp(:, Z(i, 2));
  fprintf(f, 'Pexp Cluster %d: %d + %d (%.2f of examples)\n', nclust_pexp + ...
          i, Z(i, 1), Z(i, 2), sum(tasks_pexp(:, nclust_pexp + i)) * 100 / nex);
end
levels_pexp = max(levels_pexp) - levels_pexp;
Z = [[1:nclust_pexp; 1:nclust_pexp; zeros(1, nclust_pexp)]'; Z];
%tasks_pexp = tasks_pexp(:, size(tasks_pexp, 2):-1:1);  % reverse so the clusters are roughly from largest to smallest
good_clust = sum(tasks_pexp, 1)/nex < max_percent;
tasks_pexp = tasks_pexp(:, good_clust);
clust_map = zeros(tot_npexp, 1);
clust_map(good_clust) = 1:nnz(good_clust);
Z_pexp = Z(good_clust, :);
Z_pexp(:, 1:2) = clust_map(Z_pexp(:, 1:2));
levels_pexp = levels_pexp(good_clust);
km_pexp = clust_map(km);

load(fullfile(indir, 'hocomoco_pouya_stam_jaspar_proxRegions_norm_scores.mat'));
sel_mot = strcmp(pssm_source,'hocomoco');
scores = scores(sel_mot, :);
mot_names = mot_names(sel_mot);
nmot = sum(sel_mot);
feature_names = [parents_names; mot_names];

if nclust_scores > 0
        
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
        tasks_scores(:, nclust_scores + i) = tasks_scores(:, Z(i, 1)) | ...
            tasks_scores(:, Z(i, 2));
        fprintf(f, 'Scores Cluster %d: %d + %d (%.2f of examples)\n', nclust_scores + ...
                i, Z(i, 1), Z(i, 2), sum(tasks_scores(:, nclust_scores + i)) * 100 / nex);
    end
    levels_scores = max(levels_scores) - levels_scores;
    Z = [[1:nclust_scores; 1:nclust_scores; zeros(1, nclust_scores)]'; Z];
    good_clust = sum(tasks_scores, 1)/nex < max_percent;
    clust_map = zeros(tot_nscores, 1);
    clust_map(good_clust) = 1:nnz(good_clust);
    tasks_scores = tasks_scores(:, good_clust);
    Z_scores = Z(good_clust, :);
    Z_scores(:, 1:2) = clust_map(Z_scores(:, 1:2));
    levels_scores = levels_scores(good_clust);
    km_scores = clust_map(km); % This will fail if we removed any
                               % of the initial clusters
    
    levels = [levels_pexp; levels_scores]; 
    tasks = [tasks_pexp tasks_scores];
    %pord = repmat(1:tot_npexp, tot_nscores, 1); pord = pord(:); % 1 1 1... 1 2 2 2... 2 3 3 3...
    %sord = repmat([1:tot_nscores]', tot_npexp, 1); % 1 2 3... 1 2 3... 1 2 3...
    %tasks = tasks_pexp(:, pord) & tasks_scores(:, sord);
    assert(size(levels, 1) == size(tasks, 2));

    save(fullfile(outdir, ['clusters', num2str(fold), '.mat']), ...
         'tasks', 'feature_names', 'Z_pexp', 'km_pexp', 'Z_scores', ...
         'km_scores', 'levels');
else
    tasks = tasks_pexp;
    levels = levels_pexp;
    save(fullfile(outdir, ['clusters', num2str(fold), '.mat']), ...
         'tasks', 'feature_names', 'Z_pexp', 'km_pexp', 'levels');
end
fclose(f);

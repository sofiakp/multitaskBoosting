clear all
rs = RandStream('mt19937ar', 'Seed', 1);
RandStream.setGlobalStream(rs);

indir = '/home/sofiakp/projects/Anshul/matlab/medusa/data/Jul13/hg19.encode.rna_rpk500_z2';

% This clusters the expression data (i.e. the output) which is not really
% what we want for multi-task regression, because in this way we're
% effectively using the output we're trying to predict.

nclust = 30;
merge = false; % If false, it won't do hierarchical merging
fold = 1;
max_percent = 0.8;
outdir = ['/home/sofiakp/projects/Anshul/matlab/medusa/data/Jul13/hg19.encode.rna_asinh_rpk500_z2_clust', num2str(nclust)];
if ~merge
    outdir = [outdir, '_noMerge'];
end
if ~isdir(outdir)
    mkdir(outdir);
 end

load(fullfile(indir, 'targets_asinh.mat'));
[ngen, nexp] = size(cexp);
km = litekmeans(cexp', nclust); % Cluster the genes
centroids = zeros(nclust, nexp);

for k = 1:nclust
    sel = km == k;
    centroids(k, :) = mean(cexp(sel, :), 1);
end  

% c = clustergram(centroids, 'RowLabels', strcat('Cluster ', arrayfun(@num2str, 1:nclust, 'UniformOutput', false)'), 'ColumnLabels', exptnames, 'Colormap', redbluecmap, 'Linkage', 'ward');

dlmwrite(fullfile(outdir, 'clusters.txt'), centroids, 'delimiter', '\t');
dlmwrite(fullfile(outdir, 'cluster_ind.txt'), km');

load(fullfile(indir, 'folds10_examples'));
tr = trainSets{fold};
ts = testSets{fold};
[rows, cols] = find(tr | ts); % all examples
nex = length(cols);

tasks = false(nex, 2*nclust - 1);
levels = zeros(2*nclust - 1, 1);
for i = 1:nclust
    tasks(:, i) = ismember(rows, find(km == i));
end
if merge
    Z = linkage(centroids, 'Ward');
    f = fopen(fullfile(outdir, 'clusters_info.txt'), 'w');
    for i = 1:size(Z, 1)
        levels(nclust + i) = max(levels(Z(i, 1)), levels(Z(i, 2))) + 1;
        tasks(:, nclust + i) = tasks(:, Z(i, 1)) | tasks(:, Z(i, 2));
        fprintf(f, 'Cluster %d: %d + %d (%.2f of examples)\n', nclust + ...
                i, Z(i, 1), Z(i, 2), sum(tasks(:, nclust + i)) * 100 / nex);
    end
    fclose(f);
    assert(all(tasks(:, end)));
    levels = max(levels) - levels;
    good_clust = sum(tasks, 1) / nex < max_percent;
    tasks = tasks(:, good_clust);
    % Here I'm making the assumption that all the initial clusters
    % (before hierarchical agglomeration) pass the cutoff, otherwise
    % I'd have to change km and centroids.
    clust_map = zeros(2*nclust - 1, 1);
    clust_map(good_clust) = 1:nnz(good_clust);
    Z = [[1:nclust; 1:nclust; zeros(1, nclust)]'; Z];
    Z = Z(good_clust, :);
    Z(:, 1:2) = clust_map(Z(:, 1:2));
    levels = levels(good_clust);
    
    save(fullfile(outdir, ['clusters', num2str(fold), '.mat']), 'km', ...
         'Z', 'centroids', 'tasks', 'levels');
else
    levels = ones(size(levels(1:nclust)));
    tasks = tasks(:, 1:nclust);
    save(fullfile(outdir, ['clusters', num2str(fold), '.mat']), 'km', ...
         'centroids', 'tasks', 'levels');
end


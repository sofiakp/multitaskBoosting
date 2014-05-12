clear all

outfile_root = 'hocomoco_sqb_test3';
params_file = '/home/sofiakp/projects/Anshul/matlab/medusa/data/Jul13/hg19.encode.rna_asinh_rpk100_z2_clust30/runs/hocomoco_sqb_test1_params.m';
run(params_file);
outdir = fullfile(clust_dir, 'runs');
outfile_root = fullfile(outdir, outfile_root);

outfile = [outfile_root, '.1.mat'];
load(outfile);

load(files.reg_file);
load(files.score_files{1});
sel_mot = strcmp(pssm_source,'hocomoco');
scores = scores(sel_mot, :)';
mot_names = mot_names(sel_mot);

var_names = [parents_names; mot_names];
%%
m = models{4};

% This is a crazy expression to get all indices arrayfun(@(x) cell2mat(arrayfun(@(y) y.varIdx, x.tree, 'UniformOutput', false)), m, 'UniformOutput', false)
var_idx = [];
is_leaf = [];
value = [];
thresh = [];
left = [];
right = [];

for i = 1:length(m)
  var_idx = [var_idx; [m(i).tree.varIdx]'];
  is_leaf = [is_leaf; [m(i).tree.isLeaf]'];
  value = [value; [m(i).tree.value]'];
  thresh = [thresh; [m(i).tree.threshold]'];
  ltmp = [m(i).tree.leftNodeIdx]';
  ltmp(ltmp > 0) = ltmp(ltmp > 0) + length(left) + 1;
  left = [left; ltmp];
  rtmp = [m(i).tree.rightNodeIdx]';
  rtmp(rtmp > 0) = rtmp(rtmp > 0) + length(right) + 1;
  right = [right; rtmp];
end
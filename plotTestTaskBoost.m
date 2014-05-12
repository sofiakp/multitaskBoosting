load('/home/sofiakp/projects/Anshul/matlab/medusa/data/Jul13/hg19.encode.rna_asinh_rpk100_z2_clust30/runs/hocomoco_sqb_test3.1.mat')
%%
iters = 920;
subplot(2, 2, 1);
plot(1:iters, trstats(1:iters, 1), 'b', 1:iters, tsstats(1:iters, 1), 'b--', 'LineWidth', 1);
legend({'train', 'test'}, 'Location', 'SouthEast');
xlabel('Iteration','FontSize', 10);
ylabel('Correlation', 'FontSize', 10);
set(gca, 'FontSize', 10);
subplot(2, 2, 2);
plot(1:iters, trstats(1:iters, 2), 'b', 1:iters, tsstats(1:iters, 2), 'b--', 'LineWidth', 1);
legend({'train', 'test'}, 'Location', 'SouthEast');
xlabel('Iteration', 'FontSize', 10);
ylabel('R-squared', 'FontSize', 10);
set(gca, 'FontSize', 10);
subplot(2, 2, 3);
plot(1:iters, trstats(1:iters, 3), 'b', 1:iters, tsstats(1:iters, 3), 'b--', 'LineWidth', 1);
legend({'train', 'test'}, 'Location', 'NorthEast');
xlabel('Iteration', 'FontSize', 10);
ylabel('Mean squared error', 'FontSize', 10);
set(gca, 'FontSize', 10);

print(gcf, '/home/sofiakp/projects/Anshul/matlab/medusa/data/Jul13/hg19.encode.rna_asinh_rpk100_z2_clust30/runs/hocomoco_sqb_test3.1_stats.pdf', '-dpdf');
load carbig
MPG(isnan(MPG)) = 0;
X = [Acceleration, Displacement, Horsepower, Weight];
X(isnan(X)) = 0;
Xtest = [1000 4; 2000 6; 3000 8];

% start = tic;
% tree = RegressionTree.fit(X(1:2:end,:),MPG(1:2:end,:),'MinParent',50); %,'PredictorNames',{'A','D', 'H', 'W'})
% pred1 = predict(tree,X(2:2:end,:));
% time1 = toc(start);
% 
% start = tic;
% model = regression_tree_learn(X, MPG, ones(size(MPG)), [0:2:(size(X,1)-1)]', 6, 50, 1e-6, 1.0);
% pred2 = regression_tree_predict(model, X, [1:2:size(X,1)-1]');
% disp(['Time 1 ', num2str(time1)]);
% disp(['Time 2 ', num2str(toc(start))]);
% %%
% a = double(rand(5,4) > 0.5);
% unit_test_mex(sparse(a))

task_ind = ones(size(X, 1), 3);
task_ind(1:200, 2) = 0;
task_ind(1:300, 3) = 0;
err = task_boost_learn(sparse(task_ind), X, MPG, 10, 2, 50, 1e-6, 1.0, 1.0);
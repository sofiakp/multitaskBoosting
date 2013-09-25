rs = RandStream('mt19937ar', 'Seed', 1);
RandStream.setGlobalStream(rs);

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

%rs = RandStream('mt19937ar', 'Seed', s); % set s
%RandStream.setGlobalStream(rs);

% task_ind = ones(size(X, 1), 3);
% task_ind(1:200, 2) = 0;
% task_ind(201:end, 3) = 0;
% [trloss, tsloss, pred, imp] = task_boost_learn(sparse(task_ind), X, MPG, 100, 3, 50, 1e-6, 1.0, 0.1);

nex = 1000;
t0Ind = 1:1000;
t1Ind = 1:400;
t2Ind = 401:1000;
trind = [1:200 401:801];
X = randn(nex, 3);
X = [X X + randn(nex, 3) * 0.1];
y = zeros(nex, 1);
y(t0Ind) = X(t0Ind, 1);
y(t1Ind) = X(t1Ind, 1) + X(t1Ind, 2);
y(t2Ind) = X(t2Ind, 1) + X(t2Ind, 3);
task = zeros(nex, 3);
task(t0Ind, 1) = 1;
task(t1Ind, 2) = 1;
task(t2Ind, 3) = 1;
test_task = task;
test_task(1:2:end, :) = 0;
taskOv = [1 1 1; 0 1 0; 0 0 1];
task_boost_learn(sparse(task), sparse(test_task), [0 1 1]', X, y, 100, 2, 20, 1e-6, 1.0, 0.1, false, '/home/sofiakp/test.bin');
task_boost_learn(sparse(task), sparse(test_task), [0 1 1]', X, y, 110, 2, 20, 1e-6, 1.0, 0.1, true, '/home/sofiakp/test.bin');
[trloss, tsloss, pred, imp] = task_boost_model('/home/sofiakp/test.bin');

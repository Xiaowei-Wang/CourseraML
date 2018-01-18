%% Machine Learning Online Class
%  Exercise 6 | Spam Classification with SVMs

%% Initialization
clear ; close all; clc

% load spam email data from file 
% created by function selfDataset.m
load('selfData.mat');
m = size(X,1);

rand_rows = randperm(m); % shuffle the numbers from 1 to m
X_train = X(rand_rows(1:m*0.6),:);
y_train = y(rand_rows(1:m*0.6),:);
X_cv    = X(rand_rows(m*0.6+1:m*0.8),:);
y_cv    = y(rand_rows(m*0.6+1:m*0.8),:);
X_test  = X(rand_rows(m*0.8+1:m),:);
y_test  = y(rand_rows(m*0.8+1:m),:);

% C = 1;
% sigma = 0.3;
[C,sigma] = dataset3Params(X, y, X_cv, y_cv)
model = svmTrain( X_train, y_train, C, ... 
            @(x1, x2) gaussianKernel(x1, x2, sigma), ... 
            1e-3, 50);
predictions = svmPredict(model, X_cv)




clear ; close all; clc

load ('ex5data1.mat');

lambda = 3;
p = 8;

X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(size(X_poly, 1), 1), X_poly];     
theta = trainLinearReg(X_poly, y, lambda);

X_poly_test = polyFeatures(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones


J = linearRegCostFunction(X_poly_test, ytest, theta, 0)

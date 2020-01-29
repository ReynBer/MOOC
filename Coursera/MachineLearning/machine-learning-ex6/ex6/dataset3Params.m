function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction simor using 
%        mean(double(predictions ~= yval))
%
t=[0.01,0.03,0.1,0.3,1,3,10,30];
max=size(t)(2);
isfirst = 1;
%simor=0;
for iC = max:-1:1
	tC = t(iC);
	for isigma = 1:max
		tsigma = t(isigma);

		model = svmTrain(X, y, tC, @(x1, x2) gaussianKernel(x1, x2, tsigma));
		predictions = svmPredict(model, Xval);
		cur_simor = mean(double(predictions ~= yval));
		if (isfirst == 1)
			simor = cur_simor;
			C=tC;
			sigma=tsigma;
			isfirst=0;
		else
			if (simor > cur_simor)
				simor = cur_simor;
				C=tC;
				sigma=tsigma;
%			else
%				break;
			endif
		endif
	endfor
endfor


% =========================================================================

end

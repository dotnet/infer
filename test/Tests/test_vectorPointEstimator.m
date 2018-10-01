% Licensed to the .NET Foundation under one or more agreements.
% The .NET Foundation licenses this file to you under the MIT license.
% See the LICENSE file in the project root for more information.
%rng(0);
d = 3;
X = randn(d,d)*1;
%X(:,1) = X(:,1)*100;
X(1,:) = X(1,:)*1e-8;
H = -X*X';
b = randn(d,1);
A = [H b];
n = 10;
C = randn(d,d)/1e3;
X1 = [C*randn(d,n); ones(1,n)];
%X1(1,:) = X1(1,:)*1e10;
%X1 = X1*10;
Y = A*X1;
[Uy,Sy,Vy] = svd(Y,'econ');
diag(Sy);
%Y = Y + randn(d,n)/1e11;
[H,b] = estimateHessian(Y,X1);
e = [H b] - A;
%e = (Y/X1) - A;
rmse = sqrt(mean(e(:).^2));
% estimate the amount of noise in H
machEps = 1e-16;
maxAbsY = max(abs(Y),[],2);
Vy = (maxAbsY*machEps).^2;
%maxAbsY = min(max(abs(Y),[],1));
maxAbsY = max(max(abs(Y)));
[U,S,V] = svd(X1,'econ');
diag(S);
svX = min(diag(S));
%sqrt(eig(kron(diag(Vy),inv(X1*X1'))))
noise = maxAbsY*machEps/svX;
fprintf('svX = %g maxAbsY = %g\n', svX, maxAbsY);
fprintf('RMSE = %g expected %g\n', rmse, noise);
A2 = Y/X1;
H2 = A2(1:d,1:d);
b2 = A2(1:d,d+1);
H2 = (H2 + H2')/2;
e2 = [H2 b2] - A;
rmse2 = sqrt(mean(e2(:).^2));
fprintf('/ RMSE = %g\n', rmse2);

if 0
for i = 1:cols(X1)
    fprintf('buffer.prevPoints.Add(DenseVector.FromArray(%g,%g,%g));\n', X1(1,i),X1(2,i),X1(3,i));
end
for i = 1:cols(Y)
    fprintf('buffer.prevDerivs.Add(DenseVector.FromArray(%g,%g,%g));\n', Y(1,i),Y(2,i),Y(3,i));
end
end

if 0
% addpath(fullfile(pkg_dir,'Infer.NET'))
fprintf('np.set_printoptions(precision=16)\n');
fprintf('Y = %s\n', toPythonArray(Y));
fprintf('X1 = %s\n', toPythonArray(X1));
fprintf('np.linalg.lstsq(X1.transpose(),Y.transpose())\n');
end

%U'*[A; b' 0]*U

% Licensed to the .NET Foundation under one or more agreements.
% The .NET Foundation licenses this file to you under the MIT license.
% See the LICENSE file in the project root for more information.
load expTest.mat
n = 1:100;
color.ExpOp_LaplaceProp = 'b';
color.ExpOp_Laplace = 'g';
color.ExpOp_Laplace3 = 'r';
color.ExpOp = 'k';
color.ExpOp_BFGS = 'm';
color.VMP = 'm';
labeled_curves(n,maxdiff,'color',color)
ax = axis;
axis([ax(1) ax(2) 0 max(maxdiff.ExpOp_Laplace)])

% Licensed to the .NET Foundation under one or more agreements.
% The .NET Foundation licenses this file to you under the MIT license.
% See the LICENSE file in the project root for more information.
load student.mat

inc = precs(2)-precs(1);
w = ones(size(precs))';
w(1) = 0.5;
w(end) = 0.5;
w = w*inc;
[w*exp(z) w*exp(z2) w*exp(z3)]

plot(precs, z, precs, z2, precs, z3)
plot(precs, z, precs, z2)
%plot(precs, exp(z), precs, exp(z2), precs, exp(z3))
semilogx(precs, z, precs, z2, precs, z3)
y = log(precs);
semilogx(precs, z+y, precs, z2+y, precs, z3+y)

% suppose we find the approx with largest integral or smallest abs(2nd deriv)
% need to trade-off quality of fit with height

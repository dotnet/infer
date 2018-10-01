% Licensed to the .NET Foundation under one or more agreements.
% The .NET Foundation licenses this file to you under the MIT license.
% See the LICENSE file in the project root for more information.
function is_unstable = test_stability(N,pm,em,prec)

universe = Universe(repmat(2,1,N),{'a' 'b' 'c' 'd'});
y = ones(N,1)/10;
W = pm*eye(N);
h = N/2;
if floor(h) ~= h
    error('N must be even')
end
i1 = 1:h;
i2 = (h+1):N;
W(i1,i2) = em;
W(i2,i1) = em;
% factor is y(i) ~ N(W(i,:)*x, 1/prec)
pots = {};
priors = {};
for i = 1:N
    priors{i} = NormalPot(i,universe,0,1);
    %pots{end+1} = priors{i};
end
for i = 1:length(y)
    pots{end+1} = LowRankNormalPot(1:N, universe, W(i,:)', y(i), [], 1/prec);
end
net = MarkovNet(pots);
q = DisconnectedNet(priors);
niter = 1000;
[q,approx,iter] = do_ep(pots,q,[],niter,0.5);
is_unstable = (iter == niter);

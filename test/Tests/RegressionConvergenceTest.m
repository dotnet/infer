% Licensed to the .NET Foundation under one or more agreements.
% The .NET Foundation licenses this file to you under the MIT license.
% See the LICENSE file in the project root for more information.
pm = 1;
em = -0.1;
prec = 0.01;
%prec = 0.1;
%prec = 0.05; % fails
%prec = 0.04; % fails
%prec = 0.03; % fails
%prec = 0.02; % works with damping
%prec = 10;
y = [10 7 10 19]';
N = length(y);
N = 10;
y = ones(N,1)/10;
RegressionConvergenceTest_Helper(N,1,em,prec)
return

pm = 1;
ems = linspace(0.5,0.6,10);
precs = exp(linspace(4,5,2));
xs = ndgridmat(ems,precs);
f = [];
for row = 1:rows(xs)
em = xs(row,1);
prec = xs(row,2);
f(row) = RegressionConvergenceTest_Helper(N,1,em,prec);
end
f = reshape(f,length(ems),length(precs));
contour(log(precs),ems,f)
xlabel('log(precs)')
ylabel('em')
return

%do_bp(net,'constructor','NormalPot')

% write using pairwise factors
% W*x = y
% N(y; W*x, vI) = exp(-0.5(y - Wx)'(y-Wx)/v) = exp(-0.5/v (y'y - y'Wx - x'W'y + x'W'Wx))
Q = W'*W*prec;% + eye(4);
h = W'*y*prec;
pots2 = {};
approx = {};
for i = 1:N
    %priors{end+1} = NormalPot(i,universe,h(i)/Q(i,i),'inv',Q(i,i));
    for j = (i+1):N
        if 0
            pots2{end+1} = NormalPot([i j],universe,zeros(2,1),'inv',[0 Q(i,j); Q(i,j) 0]);
        else
            Qij = [Q(i,i)/(N-1) Q(i,j); Q(i,j) Q(j,j)/(N-1)];
            mij = Qij\(h([i; j])/(N-1));
            pots2{end+1} = NormalPot([i j],universe,mij,'inv',Qij);
        end
        approx{end+1} = DisconnectedNet(priors([i j])).^1000;
    end
end
multiplyAll([priors pots2])
if 0
%q = NormalPot(1:4,universe,zeros(4,1),eye(4));
priornet = simplify(MarkovNet(priors));
q = DisconnectedNet(priornet.factors);
q = multiplyAll([{q} approx]);
do_ep(pots2,q,approx,10000,1e-3)
%do_bp(MarkovNet(pots2),'constructor','NormalPot')
end

Q = W'*W*prec + eye(4);
d = diag(1./sqrt(diag(Q)));
R = d*Q*d - eye(4);
R = abs(R);
eig(R)

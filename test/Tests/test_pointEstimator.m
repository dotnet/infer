% Licensed to the .NET Foundation under one or more agreements.
% The .NET Foundation licenses this file to you under the MIT license.
% See the LICENSE file in the project root for more information.
folder = '..\TestApp\bin\Debug\';
load([folder 'pointEstimatorHistory.mat']);
nParams = 3;
m = history(2:nParams:end,:);
%m = m(1:587,:);
if isempty(history)
    load([folder 'pointEstimatorHistory2.mat']);
    m = history(:,[1 4]);
    %m = history(:,[2 5]);
    %m = history(:,[2 4]);
    %m = history(:,[1 3]);
    if 0
        i = (rows(m)-6):rows(m);
        x = history(i,1:2);
        y = history(i,3:4);
        x1 = [x ones(rows(x),1)];
        x1\y
    end
end
    
figure(1)
plot(m(:,1),m(:,2),'.')
if rows(m) > 100
    i = (rows(m)-100):rows(m);
else
    i = 1:rows(m);
end
%m = m(40:60,:);
%m = m(3:30,:);
figure(2)
plot(m(i,1),m(i,2),'.')

slope = [];
slope2 = [];
for i = 1:rows(m)
    slope2(i) = (m(end,2) - m(i,2))/(m(end,1) - m(i,1));
    x = m(i:end,1);
    y = m(i:end,2);
    x1 = [ones(size(x)) x];
    c = x1\y;
    slope(i) = c(2);
    %slope2(i) = sum((x-mean(x)).*y)/sum((x-mean(x)).^2);
    %[u,s,v] = svd(x1,'econ')
    %c = pinv(x1)*y;
    %slope2(i) = c(2);
end
[slope2' slope' m]

if 0
x = history(:,1);
x = reshape(x,nParams,length(x)/nParams);
g = history(:,2);
g = reshape(g,nParams,length(g)/nParams);
i = 1:cols(x);
%i = (cols(x)-20):cols(x);
%i = (g(2,:) > -20) & (g(2,:) < 20);
%i = 10:40;
figure(3)
plot3(x(1,i),x(2,i),g(2,i),'.')
zlabel('g1')
xlabel('mult')
ylabel('offset')

%plot(x(2,i),g(2,i),'.')
end

if exist([folder 'logProbs.mat'])
    load([folder 'logProbs.mat']);
    figure(4)
    plot(shifts,logProbs)
    inc = shifts(2)-shifts(1);
    g = gradient(logProbs)/inc;
    plot(shifts,g)
    h = gradient(g)/inc;
    plot(shifts,h)
    1+median(h)
end

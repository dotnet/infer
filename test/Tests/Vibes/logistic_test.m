% Licensed to the .NET Foundation under one or more agreements.
% The .NET Foundation licenses this file to you under the MIT license.
% See the LICENSE file in the project root for more information.
javaaddpath(fullfile(pwd,'Vibes2_0.jar'))
import cam.jmw39.app.vibes.matlab.*;
%Vibes.run
net = Vibes.read('logistic_test.xml')
if 0
    y = 0;
    net.getNode('y').setData(y);
end
algorithm = Vibes.init(net);
algorithm.update(500);
for iter = 1:10
    algorithm.update(1);
end
w = Vibes.get(net,'w')
if 1
    y = Vibes.get(net,'y')
end

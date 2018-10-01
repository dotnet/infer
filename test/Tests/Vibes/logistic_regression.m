% Licensed to the .NET Foundation under one or more agreements.
% The .NET Foundation licenses this file to you under the MIT license.
% See the LICENSE file in the project root for more information.
javaaddpath(fullfile(pwd,'Vibes2_0.jar'))
import cam.jmw39.app.vibes.matlab.*;
%Vibes.run
net = Vibes.read('logistic_regression.xml')
x = [1 2; -3 4; 5 -6; -7 -8]';
y = [1 0 1 0];
net.getNode('x').setData(x);
net.getNode('y').setData(y);
algorithm = Vibes.init(net);
algorithm.update(500);
w = Vibes.get(net,'w')

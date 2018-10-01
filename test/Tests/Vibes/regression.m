% Licensed to the .NET Foundation under one or more agreements.
% The .NET Foundation licenses this file to you under the MIT license.
% See the LICENSE file in the project root for more information.
javaaddpath(fullfile(pwd,'Vibes2_0.jar'))
import cam.jmw39.app.vibes.matlab.*;
%Vibes.run
net = Vibes.read('regression.xml')
wts=[2.5 2.7];
x=1:2;
wx=7.9
net.getNode('x').setData(x);
net.getNode('yNoisy').setData(wx);
algorithm = Vibes.init(net);
algorithm.update(500);
m = Vibes.get(net,'m')
prec = Vibes.get(net,'prec')
w = Vibes.get(net,'w')
y = Vibes.get(net,'y')

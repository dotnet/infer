% Licensed to the .NET Foundation under one or more agreements.
% The .NET Foundation licenses this file to you under the MIT license.
% See the LICENSE file in the project root for more information.
javaaddpath(fullfile(pwd,'Vibes2_0.jar'))
import cam.jmw39.app.vibes.matlab.*;
%Vibes.run
net = Vibes.read('sumTest.xml')
net.getNode('xNoisy').setData(11);
algorithm = Vibes.init(net);
algorithm.update(500);
z = Vibes.get(net,'z')
y = Vibes.get(net,'y')
x = Vibes.get(net,'x')

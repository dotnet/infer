% Licensed to the .NET Foundation under one or more agreements.
% The .NET Foundation licenses this file to you under the MIT license.
% See the LICENSE file in the project root for more information.
javaaddpath(fullfile(pwd,'Vibes2_0.jar'))
import cam.jmw39.app.vibes.matlab.*;
%Vibes.run
net = Vibes.read('sum3ElementArrayTest.xml')
net.getNode('xNoisy').setData(5);
algorithm = Vibes.init(net);
algorithm.update(500);
a1 = Vibes.get(net,'a1')
a2 = Vibes.get(net,'a2')
a1Plusa2 = Vibes.get(net,'a1Plusa2')
a3 = Vibes.get(net,'a3')
x = Vibes.get(net,'x')

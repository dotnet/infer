% Licensed to the .NET Foundation under one or more agreements.
% The .NET Foundation licenses this file to you under the MIT license.
% See the LICENSE file in the project root for more information.
javaaddpath(fullfile(pwd,'Vibes2_0.jar'))
import cam.jmw39.app.vibes.matlab.*;
%Vibes.run
net = Vibes.read('Product2.xml')
net.getNode('abcNoisy').setData(4.0);
net.getNode('abdNoisy').setData(8.0);
algorithm = Vibes.init(net);
algorithm.update(800);
a = Vibes.get(net,'a')
b = Vibes.get(net,'b')
ab = Vibes.get(net,'ab')
c = Vibes.get(net,'c')
d = Vibes.get(net,'d')

% Licensed to the .NET Foundation under one or more agreements.
% The .NET Foundation licenses this file to you under the MIT license.
% See the LICENSE file in the project root for more information.
javaaddpath(fullfile(pwd,'Vibes2_0.jar'))
import cam.jmw39.app.vibes.matlab.*;
%Vibes.run
net = Vibes.read('ProductObservedTwice.xml')
net.getNode('abNoisy').setData(8.0);
net.getNode('ab2Noisy').setData(11.0);
algorithm = Vibes.init(net);
algorithm.update(800);
a = Vibes.get(net,'a')
b = Vibes.get(net,'b')
ab = Vibes.get(net,'ab')

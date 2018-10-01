% Licensed to the .NET Foundation under one or more agreements.
% The .NET Foundation licenses this file to you under the MIT license.
% See the LICENSE file in the project root for more information.
javaaddpath(fullfile(pwd,'Vibes2_0.jar'))
import cam.jmw39.app.vibes.matlab.*;
%Vibes.run
net = Vibes.read('sumOfProduct.xml')
net.getNode('xNoisy').setData(18.0);
algorithm = Vibes.init(net);
algorithm.update(500);
a = Vibes.get(net,'a')
b = Vibes.get(net,'b')
c = Vibes.get(net,'c')
ab = Vibes.get(net,'ab')
bc = Vibes.get(net,'bc')
x = Vibes.get(net,'x')

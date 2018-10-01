% Licensed to the .NET Foundation under one or more agreements.
% The .NET Foundation licenses this file to you under the MIT license.
% See the LICENSE file in the project root for more information.
javaaddpath(fullfile(pwd,'Vibes2_0.jar'))
import cam.jmw39.app.vibes.matlab.*;
%Vibes.run
net = Vibes.read('Mixture1a.xml')
r=[7];

net.getNode('m').setData([5.5 8]);
net.getNode('x').setData(r);
algorithm = Vibes.init(net);


algorithm.update(1000);
m = Vibes.get(net,'m')

c = Vibes.get(net,'c')
D = Vibes.get(net,'D')

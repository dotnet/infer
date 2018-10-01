% Licensed to the .NET Foundation under one or more agreements.
% The .NET Foundation licenses this file to you under the MIT license.
% See the LICENSE file in the project root for more information.
javaaddpath(fullfile(pwd,'Vibes2_0.jar'))
import cam.jmw39.app.vibes.matlab.*;
%Vibes.run
net = Vibes.read('discrete.xml');
net.getNode('x').setData(2);
%Vibes.set(net,'pi',[.3 .1 .5 .7]);

algorithm = Vibes.init(net);
algorithm.update(500);
pi = Vibes.get(net,'pi')

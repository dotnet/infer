# Licensed to the .NET Foundation under one or more agreements.
# The .NET Foundation licenses this file to you under the MIT license.
# See the LICENSE file in the project root for more information.
# Use PyMC to check the results on RunCyclingTime1
samples = [13,17,16];
nsamples = len(samples);

import pymc as pm
averageTime = pm.Normal('averageTime', mu=15, tau=0.01)
# mean is alpha/beta
trafficNoise = pm.Gamma('trafficNoise', alpha=2, beta=1/0.5)
observations = pm.Container([pm.Normal('samples_model%i' % i, 
                   mu=averageTime, tau=trafficNoise, 
                   value=samples[i], observed=True) for i in range(nsamples)])
model = pm.Model([observations, averageTime, trafficNoise])
mcmc = pm.MCMC(model)
mcmc.sample(100000)
mcmc.summary()

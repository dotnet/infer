# Licensed to the .NET Foundation under one or more agreements.
# The .NET Foundation licenses this file to you under the MIT license.
# See the LICENSE file in the project root for more information.
#-----------------------------------------------------------------------------------
# Infer.NET IronPython example: Two Coins
#-----------------------------------------------------------------------------------

import InferNetWrapper
from InferNetWrapper import *

# two coins example
def two_coins() :
    print("\n\n------------------ Infer.NET Two Coins example ------------------\n");

    # The model
    b = Microsoft.ML.Probabilistic.Distributions.Bernoulli(0.5)
    firstCoin = Variable.Bernoulli(0.5)
    secondCoin = Variable.Bernoulli(0.5)
    bothHeads = firstCoin & secondCoin

    # The inference
    ie = InferenceEngine()
    print "Probability both coins are heads:", ie.Infer(bothHeads)
    bothHeads.ObservedValue = False
    print "Probability distribution over firstCoin:", ie.Infer(firstCoin)
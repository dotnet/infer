# Licensed to the .NET Foundation under one or more agreements.
# The .NET Foundation licenses this file to you under the MIT license.
# See the LICENSE file in the project root for more information.
import InferNetWrapper
from InferNetWrapper import *

import TwoCoins
from TwoCoins import *

import TruncatedGaussian
from TruncatedGaussian import *

import ClinicalTrial
from ClinicalTrial import *

import GaussianMixture
from GaussianMixture import *

import GaussianRanges
from GaussianRanges import *

import BayesPointMachine
from BayesPointMachine import *

two_coins()

truncated_gaussian()
gaussian_ranges()
bayes_point_machine()
clinical_trial()
gaussian_mixture()
print("Pulse una tecla para Salir")
sys.stdin.readline()
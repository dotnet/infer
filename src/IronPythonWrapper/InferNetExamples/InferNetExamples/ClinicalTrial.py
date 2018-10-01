# Licensed to the .NET Foundation under one or more agreements.
# The .NET Foundation licenses this file to you under the MIT license.
# See the LICENSE file in the project root for more information.
#-----------------------------------------------------------------------------------
# Infer.NET IronPython example: clinical trial
#-----------------------------------------------------------------------------------

import InferNetWrapper
from InferNetWrapper import *

def clinical_trial():

    print("\n\n------------------ Infer.NET Clinical Trial example ------------------\n");

    controlGroup = Variable.Observed[bool](System.Array[bool]((False, False, True, False, False)))
    treatedGroup = Variable.Observed[bool](System.Array[bool]((True, False, True, True, True )))
    i = controlGroup.Range
    j = treatedGroup.Range

    # Prior on being an effective treatment
    isEffective = Variable.Bernoulli(0.5).Named("isEffective");
    
    # If block
    with (Variable.If(isEffective)) :
       probIfControl = Variable.Beta(1, 1).Named("probIfControl")
       controlGroup[i] = Variable.Bernoulli(probIfControl).ForEach(i) 
       probIfTreated = Variable.Beta(1, 1).Named("probIfTreated")
       treatedGroup[j] = Variable.Bernoulli(probIfTreated).ForEach(j)
    
    # If Not block
    with (Variable.IfNot(isEffective)) :
       probAll = Variable.Beta(1, 1).Named("probAll")
       controlGroup[i] = Variable.Bernoulli(probAll).ForEach(i)
       treatedGroup[j] = Variable.Bernoulli(probAll).ForEach(j)
    
    # The inference
    ie = InferenceEngine()
    print "Probability treatment has an effect = ", ie.Infer(isEffective)
    print "Probability of good outcome if given treatment = ", ie.Infer[Beta](probIfTreated).GetMean()
    print "Probability of good outcome if control = ", ie.Infer[Beta](probIfControl).GetMean()





// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tutorials
{
    [Example("Tutorials", "Model comparison example to determine if a new medical treatment is effective.", Prefix = "5.")]
    public class ClinicalTrial
    {
        public void Run()
        {
            // Data from clinical trial
            VariableArray<bool> controlGroup = 
                Variable.Observed(new bool[] { false, false, true, false, false }).Named("controlGroup");
            VariableArray<bool> treatedGroup = 
                Variable.Observed(new bool[] { true, false, true, true, true }).Named("treatedGroup");
            Range i = controlGroup.Range.Named("i");
            Range j = treatedGroup.Range.Named("j");

            // Prior on being effective treatment
            Variable<bool> isEffective = Variable.Bernoulli(0.5).Named("isEffective");
            Variable<double> probIfTreated, probIfControl;
            using (Variable.If(isEffective))
            {
                // Model if treatment is effective
                probIfControl = Variable.Beta(1, 1).Named("probIfControl");
                controlGroup[i] = Variable.Bernoulli(probIfControl).ForEach(i);
                probIfTreated = Variable.Beta(1, 1).Named("probIfTreated");
                treatedGroup[j] = Variable.Bernoulli(probIfTreated).ForEach(j);
            }

            using (Variable.IfNot(isEffective))
            {
                // Model if treatment is not effective
                Variable<double> probAll = Variable.Beta(1, 1).Named("probAll");
                controlGroup[i] = Variable.Bernoulli(probAll).ForEach(i);
                treatedGroup[j] = Variable.Bernoulli(probAll).ForEach(j);
            }

            InferenceEngine engine = new InferenceEngine();
            if (!(engine.Algorithm is Algorithms.GibbsSampling))
            {
                Console.WriteLine("Probability treatment has an effect = " + engine.Infer(isEffective));
                Console.WriteLine("Probability of good outcome if given treatment = "
                                        + (float)engine.Infer<Beta>(probIfTreated).GetMean());
                Console.WriteLine("Probability of good outcome if control = "
                                        + (float)engine.Infer<Beta>(probIfControl).GetMean());
            }
            else
            {
                Console.WriteLine("This model is not supported by Gibbs sampling.");
            }
        }
    }
}

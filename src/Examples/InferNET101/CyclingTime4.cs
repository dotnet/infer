// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;

namespace CyclingModels
{
    public class CyclistWithEvidence : CyclistTraining
    {
        protected Variable<bool> Evidence;

        public override void CreateModel()
        {
            Evidence = Variable.Bernoulli(0.5);
            using (Variable.If(Evidence))
            {
                base.CreateModel();
            }
        }

        public double InferEvidence(double[] trainingData)
        {
            double logEvidence;
            ModelData posteriors = base.InferModelData(trainingData);
            logEvidence = InferenceEngine.Infer<Bernoulli>(Evidence).LogOdds;

            return logEvidence;
        }
    }

    public class CyclistMixedWithEvidence : CyclistMixedTraining
    {
        protected Variable<bool> Evidence;

        public override void CreateModel()
        {
            Evidence = Variable.Bernoulli(0.5);
            using (Variable.If(Evidence))
            {
                base.CreateModel();
            }
        }

        public double InferEvidence(double[] trainingData)
        {
            double logEvidence;
            ModelDataMixed posteriors = base.InferModelData(trainingData);
            logEvidence = InferenceEngine.Infer<Bernoulli>(Evidence).LogOdds;

            return logEvidence;
        }
    }
}

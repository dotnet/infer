// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace CyclingModels
{
    public class CyclistBase
    {
        public InferenceEngine InferenceEngine;

        protected Variable<double> AverageTime;
        protected Variable<double> TrafficNoise;
        protected Variable<Gaussian> AverageTimePrior;
        protected Variable<Gamma> TrafficNoisePrior;

        public virtual void CreateModel()
        {
            AverageTimePrior = Variable.New<Gaussian>();
            TrafficNoisePrior = Variable.New<Gamma>();
            AverageTime = Variable<double>.Random(AverageTimePrior);
            TrafficNoise = Variable<double>.Random(TrafficNoisePrior);
            if (InferenceEngine == null)
            {
                InferenceEngine = new InferenceEngine();
            }
        }

        public virtual void SetModelData(ModelData priors)
        {
            AverageTimePrior.ObservedValue = priors.AverageTimeDist;
            TrafficNoisePrior.ObservedValue = priors.TrafficNoiseDist;
        }
    }

    public class CyclistTraining : CyclistBase
    {
        protected VariableArray<double> TravelTimes;
        protected Variable<int> NumTrips;

        public override void CreateModel()
        {
            base.CreateModel();
            NumTrips = Variable.New<int>();
            Range tripRange = new Range(NumTrips);
            TravelTimes = Variable.Array<double>(tripRange);
            using (Variable.ForEach(tripRange))
            {
                TravelTimes[tripRange] = Variable.GaussianFromMeanAndPrecision(AverageTime, TrafficNoise);
            }
        }

        public ModelData InferModelData(double[] trainingData)
        {
            ModelData posteriors;

            NumTrips.ObservedValue = trainingData.Length;
            TravelTimes.ObservedValue = trainingData;
            posteriors.AverageTimeDist = InferenceEngine.Infer<Gaussian>(AverageTime);
            posteriors.TrafficNoiseDist = InferenceEngine.Infer<Gamma>(TrafficNoise);
            return posteriors;
        }
    }

    public class CyclistPrediction : CyclistBase
    {
        private Gaussian tomorrowsTimeDist;
        public Variable<double> TomorrowsTime;

        public override void CreateModel()
        {
            base.CreateModel();
            TomorrowsTime = Variable.GaussianFromMeanAndPrecision(AverageTime, TrafficNoise);
        }

        public Gaussian InferTomorrowsTime()
        {
            tomorrowsTimeDist = InferenceEngine.Infer<Gaussian>(TomorrowsTime);
            return tomorrowsTimeDist;
        }

        public Bernoulli InferProbabilityTimeLessThan(double time)
        {
            return InferenceEngine.Infer<Bernoulli>(TomorrowsTime < time);
        }
    }

    public struct ModelData
    {
        public Gaussian AverageTimeDist;
        public Gamma TrafficNoiseDist;

        public ModelData(Gaussian mean, Gamma precision)
        {
            AverageTimeDist = mean;
            TrafficNoiseDist = precision;
        }
    }
}

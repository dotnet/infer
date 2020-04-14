// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Algorithms;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace CyclingModels
{
    public class CyclistMixedBase
    {
        protected InferenceEngine InferenceEngine;

        protected int NumComponents;

        protected VariableArray<Gaussian> AverageTimePriors;
        protected VariableArray<Gamma> TrafficNoisePriors;
        protected Variable<Dirichlet> MixingPrior;

        protected VariableArray<double> AverageTime;
        protected VariableArray<double> TrafficNoise;
        protected Variable<Vector> MixingCoefficients;

        public virtual void CreateModel()
        {
            NumComponents = 2;
            Range ComponentRange = new Range(NumComponents);
            InferenceEngine = new InferenceEngine(new VariationalMessagePassing());
            InferenceEngine.ShowProgress = false;

            AverageTimePriors = Variable.Array<Gaussian>(ComponentRange);
            TrafficNoisePriors = Variable.Array<Gamma>(ComponentRange);
            AverageTime = Variable.Array<double>(ComponentRange);
            TrafficNoise = Variable.Array<double>(ComponentRange);

            using (Variable.ForEach(ComponentRange))
            {
                AverageTime[ComponentRange] = Variable<double>.Random(AverageTimePriors[ComponentRange]);
                TrafficNoise[ComponentRange] = Variable<double>.Random(TrafficNoisePriors[ComponentRange]);
            }

            // Mixing coefficients
            MixingPrior = Variable.New<Dirichlet>();
            MixingCoefficients = Variable<Vector>.Random(MixingPrior);
            MixingCoefficients.SetValueRange(ComponentRange);
        }

        public virtual void SetModelData(ModelDataMixed modelData)
        {
            AverageTimePriors.ObservedValue = modelData.AverageTimeDist;
            TrafficNoisePriors.ObservedValue = modelData.TrafficNoiseDist;
            MixingPrior.ObservedValue = modelData.MixingDist;
        }
    }

    public class CyclistMixedTraining : CyclistMixedBase
    {
        protected Variable<int> NumTrips;
        protected VariableArray<double> TravelTimes;
        protected VariableArray<int> ComponentIndices;

        public override void CreateModel()
        {
            base.CreateModel();

            NumTrips = Variable.New<int>();
            Range tripRange = new Range(NumTrips);
            TravelTimes = Variable.Array<double>(tripRange);
            ComponentIndices = Variable.Array<int>(tripRange);

            using (Variable.ForEach(tripRange))
            {
                ComponentIndices[tripRange] = Variable.Discrete(MixingCoefficients);
                using (Variable.Switch(ComponentIndices[tripRange]))
                {
                    TravelTimes[tripRange].SetTo(
                        Variable.GaussianFromMeanAndPrecision(
                            AverageTime[ComponentIndices[tripRange]],
                            TrafficNoise[ComponentIndices[tripRange]]));
                }
            }
        }

        public ModelDataMixed InferModelData(double[] trainingData) // Training model
        {
            ModelDataMixed posteriors;

            // Set Priors and training data
            NumTrips.ObservedValue = trainingData.Length;
            TravelTimes.ObservedValue = trainingData;

            posteriors.AverageTimeDist = InferenceEngine.Infer<Gaussian[]>(AverageTime);
            posteriors.TrafficNoiseDist = InferenceEngine.Infer<Gamma[]>(TrafficNoise);
            posteriors.MixingDist = InferenceEngine.Infer<Dirichlet>(MixingCoefficients);

            return posteriors;
        }
    }

    public class CyclistMixedPrediction : CyclistMixedBase
    {
        private Gaussian TomorrowsTimeDist;
        private Variable<double> TomorrowsTime;

        public override void CreateModel()
        {
            base.CreateModel();

            Variable<int> componentIndex = Variable.Discrete(MixingCoefficients);
            TomorrowsTime = Variable.New<double>();

            using (Variable.Switch(componentIndex))
            {
                TomorrowsTime.SetTo(
                      Variable.GaussianFromMeanAndPrecision(
                      AverageTime[componentIndex],
                      TrafficNoise[componentIndex]));
            }
        }

        public Gaussian InferTomorrowsTime() // Prediction mode
        {
            TomorrowsTimeDist = InferenceEngine.Infer<Gaussian>(TomorrowsTime);
            return TomorrowsTimeDist;
        }
    }

    public struct ModelDataMixed
    {
        public Gaussian[] AverageTimeDist;
        public Gamma[] TrafficNoiseDist;
        public Dirichlet MixingDist;
    }
}

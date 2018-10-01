// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;

namespace CyclingModels
{
    public class TwoCyclistsTraining
    {
        private CyclistTraining cyclist1, cyclist2;

        public void CreateModel()
        {
            cyclist1 = new CyclistTraining();
            cyclist1.CreateModel();
            cyclist2 = new CyclistTraining();
            cyclist2.CreateModel();
        }

        public void SetModelData(ModelData modelData)
        {
            cyclist1.SetModelData(modelData);
            cyclist2.SetModelData(modelData);
        }

        public ModelData[] InferModelData(double[] trainingData1,
                                          double[] trainingData2)
        {
            ModelData[] posteriors = new ModelData[2];

            posteriors[0] = cyclist1.InferModelData(trainingData1);
            posteriors[1] = cyclist2.InferModelData(trainingData2);

            return posteriors;
        }
    }

    public class TwoCyclistsPrediction
    {
        private CyclistPrediction cyclist1, cyclist2;
        private Variable<double> TimeDifference;
        private Variable<bool> Cyclist1IsFaster;
        private InferenceEngine CommonEngine;

        public void CreateModel()
        {
            CommonEngine = new InferenceEngine();

            cyclist1 = new CyclistPrediction() { InferenceEngine = CommonEngine };
            cyclist1.CreateModel();
            cyclist2 = new CyclistPrediction() { InferenceEngine = CommonEngine };
            cyclist2.CreateModel();

            TimeDifference = cyclist1.TomorrowsTime - cyclist2.TomorrowsTime;
            Cyclist1IsFaster = cyclist1.TomorrowsTime < cyclist2.TomorrowsTime;
        }

        public void SetModelData(ModelData[] modelData)
        {
            cyclist1.SetModelData(modelData[0]);
            cyclist2.SetModelData(modelData[1]);
        }

        public Gaussian[] InferTomorrowsTime()
        {
            Gaussian[] tomorrowsTime = new Gaussian[2];

            tomorrowsTime[0] = cyclist1.InferTomorrowsTime();
            tomorrowsTime[1] = cyclist2.InferTomorrowsTime();
            return tomorrowsTime;
        }

        public Gaussian InferTimeDifference()
        {
            return CommonEngine.Infer<Gaussian>(TimeDifference);
        }

        public Bernoulli InferCyclist1IsFaster()
        {
            return CommonEngine.Infer<Bernoulli>(Cyclist1IsFaster);
        }
    }
}

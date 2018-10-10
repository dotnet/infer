// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Text;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Xunit;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Assert = Xunit.Assert;

namespace Microsoft.ML.Probabilistic.Tests
{
    public class CyclistTests
    {
        public class CyclistBase
        {
            public InferenceEngine InferenceEngine;
            protected static int count;

            protected Variable<double> AverageTime;
            protected Variable<double> TrafficNoise;
            protected Variable<Gaussian> AverageTimePrior;
            protected Variable<Gamma> TrafficNoisePrior;

            public virtual void CreateModel()
            {
                count++;
                AverageTimePrior = Variable.New<Gaussian>().Named("AverageTimePrior" + count);
                TrafficNoisePrior = Variable.New<Gamma>().Named("TrafficNoisePrior" + count);
                AverageTime = Variable.Random<double, Gaussian>(AverageTimePrior).Named("AverageTime" + count);
                TrafficNoise = Variable.Random<double, Gamma>(TrafficNoisePrior).Named("TrafficNoise" + count);
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
                NumTrips = Variable.New<int>().Named("NumTrips" + count);
                Range tripRange = new Range(NumTrips).Named("tripRange" + count);
                TravelTimes = Variable.Array<double>(tripRange).Named("TravelTimes" + count);
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
                TomorrowsTime = Variable.GaussianFromMeanAndPrecision(AverageTime, TrafficNoise).Named("TomorrowsTime" + count);
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
        public class MultipleCyclistsTraining
        {
            private CyclistTraining cyclist1, cyclist2, cyclist3, cyclict4, cyclist5, cyclist6;

            public void CreateModel()
            {
                cyclist1 = new CyclistTraining();
                cyclist1.CreateModel();
                cyclist2 = new CyclistTraining();
                cyclist2.CreateModel();
                cyclist3 = new CyclistTraining();
                cyclist3.CreateModel();
                cyclict4 = new CyclistTraining();
                cyclict4.CreateModel();
                cyclist5 = new CyclistTraining();
                cyclist5.CreateModel();
                cyclist6 = new CyclistTraining();
                cyclist6.CreateModel();
            }

            public void SetModelData(ModelData modelData)
            {
                cyclist1.SetModelData(modelData);
                cyclist2.SetModelData(modelData);
                cyclist3.SetModelData(modelData);
                cyclict4.SetModelData(modelData);
                cyclist5.SetModelData(modelData);
                cyclist6.SetModelData(modelData);
            }

            public ModelData[] InferModelData(double[] trainingData1,
                                              double[] trainingData2,
                                              double[] trainingData3,
                                              double[] trainingData4,
                                              double[] trainingData5,
                                              double[] trainingData6)
            {
                ModelData[] posteriors = new ModelData[6];

                posteriors[0] = cyclist1.InferModelData(trainingData1);
                posteriors[1] = cyclist2.InferModelData(trainingData2);
                posteriors[2] = cyclist3.InferModelData(trainingData3);
                posteriors[3] = cyclict4.InferModelData(trainingData4);
                posteriors[4] = cyclist5.InferModelData(trainingData5);
                posteriors[5] = cyclist6.InferModelData(trainingData6);
                return posteriors;
            }

        }

        public class MultipleCyclistsPrediction
        {
            private CyclistPrediction cyclist1, cyclist2, cyclist3, cyclist4, cyclist5, cyclist6;
            private InferenceEngine CommonEngine;
            private Variable<int> winner = Variable.DiscreteUniform(6).Named("winner");

            public void CreateModel()
            {
                CommonEngine = new InferenceEngine();

                cyclist1 = new CyclistPrediction()
                {
                    InferenceEngine = CommonEngine
                };
                cyclist1.CreateModel();
                cyclist2 = new CyclistPrediction()
                {
                    InferenceEngine = CommonEngine
                };
                cyclist2.CreateModel();
                cyclist3 = new CyclistPrediction()
                {
                    InferenceEngine = CommonEngine
                };
                cyclist3.CreateModel();
                cyclist4 = new CyclistPrediction()
                {
                    InferenceEngine = CommonEngine
                };
                cyclist4.CreateModel();
                cyclist5 = new CyclistPrediction()
                {
                    InferenceEngine = CommonEngine
                };
                cyclist5.CreateModel();
                cyclist6 = new CyclistPrediction()
                {
                    InferenceEngine = CommonEngine
                };
                cyclist6.CreateModel();

                using (Variable.Case(winner, 0))
                {
                    Variable.ConstrainTrue(cyclist1.TomorrowsTime < cyclist2.TomorrowsTime & cyclist1.TomorrowsTime < cyclist3.TomorrowsTime &
                                           cyclist1.TomorrowsTime < cyclist4.TomorrowsTime & cyclist1.TomorrowsTime < cyclist5.TomorrowsTime &
                                           cyclist1.TomorrowsTime < cyclist6.TomorrowsTime);
                }
                using (Variable.Case(winner, 1))
                {
                    Variable.ConstrainTrue(cyclist2.TomorrowsTime < cyclist1.TomorrowsTime & cyclist2.TomorrowsTime < cyclist3.TomorrowsTime &
                                           cyclist2.TomorrowsTime < cyclist4.TomorrowsTime & cyclist2.TomorrowsTime < cyclist5.TomorrowsTime &
                                           cyclist2.TomorrowsTime < cyclist6.TomorrowsTime);
                }
                using (Variable.Case(winner, 2))
                {
                    Variable.ConstrainTrue(cyclist3.TomorrowsTime < cyclist1.TomorrowsTime & cyclist3.TomorrowsTime < cyclist2.TomorrowsTime &
                                           cyclist3.TomorrowsTime < cyclist4.TomorrowsTime & cyclist3.TomorrowsTime < cyclist5.TomorrowsTime &
                                           cyclist3.TomorrowsTime < cyclist6.TomorrowsTime);
                }
                using (Variable.Case(winner, 3))
                {
                    Variable.ConstrainTrue(cyclist4.TomorrowsTime < cyclist1.TomorrowsTime & cyclist4.TomorrowsTime < cyclist2.TomorrowsTime &
                                           cyclist4.TomorrowsTime < cyclist3.TomorrowsTime & cyclist4.TomorrowsTime < cyclist5.TomorrowsTime &
                                           cyclist4.TomorrowsTime < cyclist6.TomorrowsTime);
                }
                using (Variable.Case(winner, 4))
                {
                    Variable.ConstrainTrue(cyclist5.TomorrowsTime < cyclist1.TomorrowsTime & cyclist5.TomorrowsTime < cyclist2.TomorrowsTime &
                                           cyclist5.TomorrowsTime < cyclist3.TomorrowsTime & cyclist5.TomorrowsTime < cyclist4.TomorrowsTime &
                                           cyclist5.TomorrowsTime < cyclist6.TomorrowsTime);
                }
                using (Variable.Case(winner, 5))
                {
                    Variable.ConstrainTrue(cyclist6.TomorrowsTime < cyclist1.TomorrowsTime & cyclist6.TomorrowsTime < cyclist2.TomorrowsTime &
                                           cyclist6.TomorrowsTime < cyclist3.TomorrowsTime & cyclist6.TomorrowsTime < cyclist4.TomorrowsTime &
                                           cyclist6.TomorrowsTime < cyclist5.TomorrowsTime);
                }
            }

            public void SetModelData(ModelData[] modelData)
            {
                cyclist1.SetModelData(modelData[0]);
                cyclist2.SetModelData(modelData[1]);
                cyclist3.SetModelData(modelData[2]);
                cyclist4.SetModelData(modelData[3]);
                cyclist5.SetModelData(modelData[4]);
                cyclist6.SetModelData(modelData[5]);
            }

            public Gaussian[] InferTomorrowsTime()
            {
                Gaussian[] tomorrowsTime = new Gaussian[6];

                tomorrowsTime[0] = cyclist1.InferTomorrowsTime();
                tomorrowsTime[1] = cyclist2.InferTomorrowsTime();
                tomorrowsTime[2] = cyclist3.InferTomorrowsTime();
                tomorrowsTime[3] = cyclist4.InferTomorrowsTime();
                tomorrowsTime[4] = cyclist5.InferTomorrowsTime();
                tomorrowsTime[5] = cyclist6.InferTomorrowsTime();
                return tomorrowsTime;
            }

            public Discrete InferWinner()
            {
                return (Discrete)CommonEngine.Infer(winner);
            }
        }
        public static Discrete RunMultipleCyclistInference(Dictionary<int, double[]> trainingData)
        {
            ModelData initPriors = new ModelData(
                Gaussian.FromMeanAndPrecision(29.5, 0.01),
                Gamma.FromShapeAndScale(1.0, 0.5));

            //Train the model
            MultipleCyclistsTraining cyclistsTraining = new MultipleCyclistsTraining();
            cyclistsTraining.CreateModel();
            cyclistsTraining.SetModelData(initPriors);

            ModelData[] posteriors1 = cyclistsTraining.InferModelData(trainingData[0], trainingData[1], trainingData[2], trainingData[3], trainingData[4], trainingData[5]);

            Console.WriteLine("Cyclist 1 average travel time: {0}", posteriors1[0].AverageTimeDist);
            Console.WriteLine("Cyclist 1 traffic noise: {0}", posteriors1[0].TrafficNoiseDist);

            //Make predictions based on the trained model
            MultipleCyclistsPrediction cyclistsPrediction = new MultipleCyclistsPrediction();
            cyclistsPrediction.CreateModel();
            cyclistsPrediction.SetModelData(posteriors1);

            Gaussian[] posteriors2 = cyclistsPrediction.InferTomorrowsTime();

            return cyclistsPrediction.InferWinner();
        }
        [Fact]
        public void MultipleCyclistTest()
        {
            Dictionary<int, double[]> trainingData = new Dictionary<int, double[]>();
            trainingData[0] = new double[] { 29.91, 28.79, 30.58, 30.17, 30.01 };
            trainingData[1] = new double[] { 30.0, 29.99, 28.9 };
            trainingData[2] = new double[] { 29.72, 29.69, 30.26, 30.12, 29.89 };
            trainingData[3] = new double[] { 30.44, 29.67, 29.8 };
            trainingData[4] = new double[] { 29.95, 30.1, 29.3, 30.13, 29.51 };
            trainingData[5] = new double[] { 29.81, 29.67, 30.08 };

            Console.WriteLine(RunMultipleCyclistInference(trainingData));
        }

        /// <summary>
        /// Test that initialized variables are reset correctly when number of iterations is changed.
        /// </summary>
        [Fact]
        public void RunCyclingTime1()
        {
            // [1] The model
            Variable<double> averageTime = Variable.GaussianFromMeanAndPrecision(15, 0.01);
            Variable<double> trafficNoise = Variable.GammaFromShapeAndScale(2.0, 0.5);

            Variable<double> travelTimeMonday = Variable.GaussianFromMeanAndPrecision(averageTime, trafficNoise);
            Variable<double> travelTimeTuesday = Variable.GaussianFromMeanAndPrecision(averageTime, trafficNoise);
            Variable<double> travelTimeWednesday = Variable.GaussianFromMeanAndPrecision(averageTime, trafficNoise);

            // [2] Train the model
            travelTimeMonday.ObservedValue = 13;
            travelTimeTuesday.ObservedValue = 17;
            travelTimeWednesday.ObservedValue = 16;

            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;
            averageTime.InitialiseTo(new Gaussian(15.29, 1.559));
            trafficNoise.InitialiseTo(new Gamma(1.458, 0.3944));
            engine.NumberOfIterations = 2;
            Gaussian averageTimeExpected = engine.Infer<Gaussian>(averageTime);
            Gamma trafficNoiseExpected = engine.Infer<Gamma>(trafficNoise);

            engine.NumberOfIterations = 1;
            Gaussian averageTimePosterior = engine.Infer<Gaussian>(averageTime);
            Gamma trafficNoisePosterior = engine.Infer<Gamma>(trafficNoise);

            engine.NumberOfIterations = 2;
            Gaussian averageTimeActual = engine.Infer<Gaussian>(averageTime);
            Gamma trafficNoiseActual = engine.Infer<Gamma>(trafficNoise);

            Assert.Equal(averageTimeActual, averageTimeExpected);
            Assert.Equal(trafficNoiseActual, trafficNoiseExpected);

            // These are the results expected from EP.
            // The exact results can be obtained from Gibbs sampling or cyclingTest.py
            averageTimeExpected = new Gaussian(15.33, 1.32);
            trafficNoiseExpected = new Gamma(2.242, 0.2445);

            engine.NumberOfIterations = 50;
            averageTimeActual = engine.Infer<Gaussian>(averageTime);
            trafficNoiseActual = engine.Infer<Gamma>(trafficNoise);

            Assert.Equal(averageTimeActual.ToString(), averageTimeExpected.ToString());
            Assert.Equal(trafficNoiseActual.ToString(), trafficNoiseExpected.ToString());
        }
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Globalization;
using System.Linq;
using System.Collections.Generic;
using Xunit;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using System.IO;
using Microsoft.ML.Probabilistic.Algorithms;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
    public class BPMNoiseTests2
    {
        public static void NoiseTest(double noiseVariance)
        {
            int d = 2;
            int n = 1000;

            // generate data
            var N0I = new VectorGaussian(Vector.Zero(d), PositiveDefiniteMatrix.Identity(d));
            var wTrue = N0I.Sample();
            Normalize(wTrue);
            Vector[] x = new Vector[n];
            bool[] y = new bool[n];
            for (int i = 0; i < n; i++)
            {
                x[i] = N0I.Sample();
                y[i] = (x[i].Inner(wTrue) + Gaussian.Sample(0, 1.0/noiseVariance)) > 0.0;
            }

            // evaluate models
            var fixedNoise = new BPM_FixedNoise(d, n);
            var noiseRange = new double[] {1, 2, 10, 20, 30, 100, 1000, 1e4};
            foreach (double noiseTrain in noiseRange)
            {
                Vector wTrain = fixedNoise.Train(x, y, noiseTrain);
                Normalize(wTrain);
                double err = System.Math.Acos(wTrain.Inner(wTrue)) / System.Math.PI;
                //double err = Math.Sqrt(wTrue.Inner(wTrue) -2*wTrue.Inner(wTrain) + wTrain.Inner(wTrain));
                Console.WriteLine("noiseTrain = {0}, error = {1}", noiseTrain, err);
            }
        }

        // normalize a Vector to unit length
        public static void Normalize(Vector w)
        {
            w.Scale(1.0/ System.Math.Sqrt(w.Inner(w)));
        }

        public class BPM_FixedNoise
        {
            private Variable<Vector> w;
            private VariableArray<bool> y;
            private VariableArray<Vector> x;
            private Variable<double> noise;
            private InferenceEngine engine;

            public BPM_FixedNoise(int d, int n)
            {
                Range j = new Range(n).Named("j");
                y = Variable.Array<bool>(j).Named("y");
                x = Variable.Array<Vector>(j).Named("x");
                noise = Variable.New<double>().Named("noise");
                w = Variable.Random(new VectorGaussian(Vector.Zero(d),
                                                       PositiveDefiniteMatrix.Identity(d))).Named("w");
                engine = new InferenceEngine();
                y[j] = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(w, x[j]).Named("innerProduct"), noise) > 0;
            }

            public Vector Train(Vector[] xTrain, bool[] yTrain, double noiseTrain)
            {
                x.ObservedValue = xTrain;
                y.ObservedValue = yTrain;
                noise.ObservedValue = noiseTrain;
                var wPost = engine.Infer<VectorGaussian>(w);
                return wPost.GetMean();
            }
        }
    }

    
    public class BPMNoiseTests
    {
        public string baseDir; // = @"../../../Tests/Data/synthetic/";

        private Vector[] xtrain;
        private Vector[] xtest;
        private bool[] ytrain;
        private bool[] ytest;

        public void BigBPMTest()
        {
            var noiseRange = new double[] {1.0, 2, 10, 20, 30, 100, 1000, 1e4};
            var noisePrecRateRangeEP = new double[] {1e-1, 1, 1e1, 1e2, 1e3, 1e4};
            var noisePrecRateRangeVMP = new double[] {1e-2, 1e-1, 1, 1e1};
            var noiseVarRateRange = new double[] {1e-2, 1e-1, 1, 1e1};
            var countsRange = new double[] {8, 12, 16};
            var epsRange = new double[] {1e-3, 1e-2, 1e-1};
            var aRange = new double[] {0.5, 1, 2.0};

            TestHelper(SimpleAverageModel, new double[] {0}, "dumb", LoadData);
            TestHelper(TestBPM_FixedNoise, noiseRange, "fix", LoadData);
            TestHelper(TestBPM_NoisePrecisionEP, noisePrecRateRangeEP, "prec", LoadData);
            TestHelper(TestBPM_NoisePrecisionVMP, noisePrecRateRangeVMP, "vmp", LoadData);
            TestHelper(TestBPM_NoiseVariance, noiseVarRateRange, "var", LoadData);
            TestHelper(TestBPM_NoiseVariancePlusLabel, noiseVarRateRange, "varlab", LoadData);
            TestHelper(TestBPM_LabelNoise, epsRange, "label", LoadData);
            TestHelper(TestBPM_LabelNoiseConstrained, new double[] {.5, 1, 5, 10}, "labelc", LoadData);
            TestHelper(TestBPM_LabelNoiseFixed, noiseRange, "labfix", LoadData);
            TestHelper(TestNaiveBayes, aRange, "nb", LoadData);
            TestHelper(TestNaiveBayesDiagonal, aRange, "nbd", LoadData);
            TestHelper(TestNaiveBayesDiagonalHierarchical, aRange, "nbdh", LoadData);
        }

        public int SyntheticData(int D, int N, int Ntest, double noiseVariance)
        {
            xtrain = new Vector[N];
            xtest = new Vector[Ntest];
            ytrain = new bool[N];
            ytest = new bool[Ntest];
            var N0I = new VectorGaussian(Vector.Zero(D), PositiveDefiniteMatrix.Identity(D));
            var trueW = N0I.Sample();
            for (int i = 0; i < N; i++)
            {
                xtrain[i] = N0I.Sample();
                ytrain[i] = (xtrain[i].Inner(trueW) + Gaussian.Sample(0, 1.0/noiseVariance)) > 0.0;
            }
            for (int i = 0; i < Ntest; i++)
            {
                xtest[i] = N0I.Sample();
                ytest[i] = (xtest[i].Inner(trueW) + Gaussian.Sample(0, 1.0/noiseVariance)) > 0.0;
            }
            return 0;
        }

        public void OutputSyntheticDatasets(int D, int N, int Ntest)
        {
            var noiseVariance = (Enumerable.Range(-5, 10).ToArray()).Select(i => System.Math.Pow(10, i)).ToArray();
            Converter<int, int> load = i => SyntheticData(D, N, Ntest, noiseVariance[i]);
            Directory.CreateDirectory(baseDir + "data");

            for (int i = 0; i < noiseVariance.Length; i++)
            {
                load(i);
                using (var sw = new StreamWriter(baseDir + "data/train_x_" + i + ".csv"))
                    foreach (var line in xtrain.Select(j => j.ToArray().Select(a => a.ToString(CultureInfo.InvariantCulture)).Aggregate((k, l) => k + "," + l)))
                        sw.WriteLine(line);
                using (var sw = new StreamWriter(baseDir + "data/test_x_" + i + ".csv"))
                    foreach (var line in xtest.Select(j => j.ToArray().Select(a => a.ToString(CultureInfo.InvariantCulture)).Aggregate((k, l) => k + "," + l)))
                        sw.WriteLine(line);
                using (var sw = new StreamWriter(baseDir + "data/train_y_" + i + ".csv"))
                    foreach (var line in ytrain)
                        sw.WriteLine(line);
                using (var sw = new StreamWriter(baseDir + "data/test_y_" + i + ".csv"))
                    foreach (var line in ytest)
                        sw.WriteLine(line);
            }
        }

        public void SyntheticBPMTest(int D, int N, int Ntest)
        {
            var noiseRange = new double[] {1.0, 2, 10, 20, 30, 100, 1000, 1e4};
            var noisePrecRateRangeEP = new double[] {1e-1, 1, 1e1, 1e2, 1e3, 1e4};
            var noisePrecRateRangeVMP = new double[] {1e-2, 1e-1, 1, 1e1, 1e2};
            var noiseVarRateRange = new double[] {1e-2, 1e-1, 1, 1e1};
            var countsRange = new double[] {8, 12, 16};
            var epsRange = new double[] {1e-3, 1e-2, 1e-1};
            var aRange = new double[] {0.5, 1, 2.0};

            var noiseVariance = (Enumerable.Range(-5, 10).ToArray()).Select(i => System.Math.Pow(10, i)).ToArray();
            Converter<int, int> load = i => SyntheticData(D, N, Ntest, noiseVariance[i]);

            TestHelper(SimpleAverageModel, new double[] {0}, "dumb", load);
            TestHelper(TestBPM_FixedNoise, noiseRange, "fix", load);
            TestHelper(TestBPM_NoisePrecisionEP, noisePrecRateRangeEP, "prec", load);
            TestHelper(TestBPM_NoisePrecisionVMP, noisePrecRateRangeVMP, "vmp", load);
            TestHelper(TestBPM_NoiseVariance, noiseVarRateRange, "var", load);
            TestHelper(TestBPM_NoiseVariancePlusLabel, noiseVarRateRange, "varlab", load);
            TestHelper(TestBPM_LabelNoise, epsRange, "label", load);
            TestHelper(TestBPM_LabelNoiseConstrained, new double[] {.5, 1, 5, 10}, "labelc", load);
            TestHelper(TestBPM_LabelNoiseFixed, noiseRange, "labfix", load);
            TestHelper(TestNaiveBayes, aRange, "nb", load);
            TestHelper(TestNaiveBayesDiagonal, aRange, "nbd", load);
            TestHelper(TestNaiveBayesDiagonalHierarchical, aRange, "nbdh", load);
        }


        public void TestHelper(TestMethod testMethod, double[] range, string prefix, Converter<int, int> load, int reps = 10)
        {
            using (var sw = new StreamWriter(baseDir + "noisy_bpm_" + prefix + ".csv"))
            using (var sw2 = new StreamWriter(baseDir + "noisy_bpm_" + prefix + "_err.csv"))
            using (var sw3 = new StreamWriter(baseDir + "noisy_bpm_" + prefix + "_noise.csv"))
            {
                sw.WriteLine("it," + range.Select(x => prefix + x.ToString(CultureInfo.InvariantCulture)).Aggregate((x, y) => x + "," + y));
                sw2.WriteLine("it," + range.Select(x => prefix + x.ToString(CultureInfo.InvariantCulture)).Aggregate((x, y) => x + "," + y));
                sw3.WriteLine("it," + range.Select(x => prefix + x.ToString(CultureInfo.InvariantCulture)).Aggregate((x, y) => x + "," + y));
                for (int i = 0; i < reps; i++)
                {
                    sw.Write(i);
                    sw2.Write(i);
                    sw3.Write(i);
                    for (int j = 0; j < range.Length; j++)
                    {
                        load(i);
                        double res = 0, err = 0;
                        double noiseEstimate = double.NaN;
                        try
                        {
                            var ytPost = testMethod(range[j], out noiseEstimate);
                            for (int k = 0; k < ytest.Length; k++)
                            {
                                res += ytPost[k].GetLogProb(ytest[k]);
                                if ((ytPost[k].GetMean() > .5) != ytest[k])
                                    err++;
                            }
                            res /= ytest.Length;
                            Console.WriteLine("logPred = " + res);
                            Console.WriteLine("Errors = " + err);
                        }
                        catch
                        {
                            err = double.NaN;
                            res = double.NaN;
                        }
                        sw.Write("," + res);
                        sw2.Write("," + err);
                        sw3.Write("," + noiseEstimate);
                    }
                    sw.WriteLine();
                    sw2.WriteLine();
                    sw3.WriteLine();
                }
            }
        }

        private double propTest = .2;

        public int LoadData(int r)
        {
            Rand.Restart(r);
            var xTrainList = new List<Vector>();
            var xTestList = new List<Vector>();
            var yTrainList = new List<bool>();
            var yTestList = new List<bool>();
            var ydata = BioTests.ReadCSV(baseDir + "y.csv");
            var xdata = BioTests.ReadCSV(baseDir + "x.csv");
            int N = ydata.GetLength(0);
            int K = xdata.GetLength(1);
            for (int i = 0; i < N; i++)
            {
                var x = Vector.Zero(K);
                for (int j = 0; j < K; j++)
                {
                    x[j] = xdata[i, j];
                }
                if (Rand.Double() < propTest)
                {
                    yTestList.Add(ydata[i, 0] == 1.0);
                    xTestList.Add(x);
                }
                else
                {
                    yTrainList.Add(ydata[i, 0] == 1.0);
                    xTrainList.Add(x);
                }
            }
            xtrain = xTrainList.ToArray();
            xtest = xTestList.ToArray();
            ytrain = yTrainList.ToArray();
            ytest = yTestList.ToArray();
            return 0;
        }

        public delegate DistributionArray<Bernoulli> TestMethod(double x, out double noiseEstimate);


        public DistributionArray<Bernoulli> SimpleAverageModel(double dummy, out double noiseEstimate)
        {
            noiseEstimate = double.NaN;
            int K = xtrain[0].Count;
            var probTrue = Variable.Beta(1, 1);
            Range n = new Range(ytrain.Length);
            var data = Variable.Array<bool>(n);
            data[n] = Variable.Bernoulli(probTrue).ForEach(n);
            data.ObservedValue = ytrain;
            InferenceEngine engine = new InferenceEngine();
            var probPost = engine.Infer<Beta>(probTrue);
            var array = Enumerable.Range(0, xtest.Length).Select(i => BernoulliFromBetaOp.SampleAverageConditional(probPost)).ToArray();
            return (DistributionArray<Bernoulli>) Distribution<bool>.Array(array);
        }


        public DistributionArray<Bernoulli> TestBPM_FixedNoise(double noise, out double noiseEstimate)
        {
            noiseEstimate = double.NaN;
            int K = xtrain[0].Count;
            // Create target y
            VariableArray<bool> y = Variable.Observed(ytrain).Named("y");
            Variable<Vector> w = Variable.Random(new VectorGaussian(Vector.Zero(K),
                                                                    PositiveDefiniteMatrix.Identity(K))).Named("w");
            var mean = Variable.GaussianFromMeanAndPrecision(0, .1);
            BayesPointMachine_FixedNoise(xtrain, w, y, mean, noise);
            InferenceEngine engine = new InferenceEngine();
            VectorGaussian wPosterior = engine.Infer<VectorGaussian>(w);
            //Console.WriteLine("Dist over w=\n" + wPosterior);
            var meanPost = engine.Infer<Gaussian>(mean);
            VariableArray<bool> yt = Variable.Array<bool>(new Range(ytest.Length)).Named("ytest");
            BayesPointMachine_FixedNoise(xtest, Variable.Random(wPosterior).Named("w"), yt, Variable.Random(meanPost), noise);
            return engine.Infer<DistributionArray<Bernoulli>>(yt);
        }

        public void BayesPointMachine_FixedNoise(Vector[] xdata, Variable<Vector> w, VariableArray<bool> y, Variable<double> mean, double noise)
        {
            // Create x vector, augmented by 1
            Range j = y.Range.Named("person");
            VariableArray<Vector> x = Variable.Observed(xdata, j).Named("x");

            // Bayes Point Machine
            y[j] = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(w, x[j]).Named("innerProduct") + mean, noise) > 0;
        }


        public DistributionArray<Bernoulli> TestBPM_NoiseVariance(double noiseRate, out double noiseEstimate)
        {
            int K = xtrain[0].Count;
            // Create target y
            VariableArray<bool> y = Variable.Observed(ytrain).Named("y");
            Variable<Vector> w = Variable.Random(new VectorGaussian(Vector.Zero(K),
                                                                    PositiveDefiniteMatrix.Identity(K))).Named("w");
            var mean = Variable.GaussianFromMeanAndPrecision(0, .1);
            var noise = Variable.GammaFromShapeAndRate(2, noiseRate);
            //var noise = Variable.Random(Gamma.PointMass(.1));
            BayesPointMachine_NoiseVariance(xtrain, w, y, mean, noise);
            InferenceEngine engine = new InferenceEngine();
            VectorGaussian wPosterior = engine.Infer<VectorGaussian>(w);

            var noisePost = engine.Infer<Gamma>(noise);
            noiseEstimate = noisePost.GetMean();
            //Console.WriteLine("Dist over w=\n" + wPosterior);
            //Console.WriteLine("Dist over noise=\n" + noisePost);
            var meanPost = engine.Infer<Gaussian>(mean);
            VariableArray<bool> yt = Variable.Array<bool>(new Range(ytest.Length)).Named("ytest");
            BayesPointMachine_NoiseVariance(xtest, Variable.Random(wPosterior).Named("w"), yt, Variable.Random(meanPost), Variable.Random(noisePost));
            return engine.Infer<DistributionArray<Bernoulli>>(yt);
        }

        public void BayesPointMachine_NoiseVariance(Vector[] xdata, Variable<Vector> w, VariableArray<bool> y, Variable<double> mean, Variable<double> noiseVariance)
        {
            // Create x vector, augmented by 1
            Range j = y.Range.Named("person");
            VariableArray<Vector> x = Variable.Observed(xdata, j).Named("x");

            // Bayes Point Machine
            y[j] = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(w, x[j]).Named("innerProduct") + mean, noiseVariance) > 0;
        }

        public DistributionArray<Bernoulli> TestBPM_NoiseVariancePlusLabel(double noiseRate, out double noiseEstimate)
        {
            int K = xtrain[0].Count;
            // Create target y
            VariableArray<bool> y = Variable.Observed(ytrain).Named("y");
            Variable<Vector> w = Variable.Random(new VectorGaussian(Vector.Zero(K),
                                                                    PositiveDefiniteMatrix.Identity(K))).Named("w");
            var ProbTrue = Variable.Beta(1, 10).Named("TPR");
            var ProbFalse = Variable.Beta(10, 1).Named("FNR");

            var mean = Variable.GaussianFromMeanAndPrecision(0, .1);
            var noise = Variable.GammaFromShapeAndRate(2, noiseRate);
            //var noise = Variable.Random(Gamma.PointMass(.1));
            BayesPointMachine_NoiseVariancePlusLabel(xtrain, w, y, mean, noise, ProbTrue, ProbFalse);
            InferenceEngine engine = new InferenceEngine();
            VectorGaussian wPosterior = engine.Infer<VectorGaussian>(w);
            var noisePost = engine.Infer<Gamma>(noise);
            noiseEstimate = noisePost.GetMean();
            Beta probTruePost = engine.Infer<Beta>(ProbTrue);
            Beta probFalsePost = engine.Infer<Beta>(ProbFalse);
            //Console.WriteLine("Dist over w=\n" + wPosterior);
            //Console.WriteLine("Dist over noise=\n" + noisePost);
            var meanPost = engine.Infer<Gaussian>(mean);
            VariableArray<bool> yt = Variable.Array<bool>(new Range(ytest.Length)).Named("ytest");
            BayesPointMachine_NoiseVariancePlusLabel(xtest, Variable.Random(wPosterior).Named("w"), yt, Variable.Random(meanPost), Variable.Random(noisePost),
                                                     Variable.Random(probTruePost), Variable.Random(probFalsePost));
            return engine.Infer<DistributionArray<Bernoulli>>(yt);
        }

        public void BayesPointMachine_NoiseVariancePlusLabel(Vector[] xdata, Variable<Vector> w, VariableArray<bool> y, Variable<double> mean, Variable<double> noiseVariance,
                                                             Variable<double> probTrue, Variable<double> probFalse)
        {
            // Create x vector, augmented by 1
            Range j = y.Range.Named("person");
            VariableArray<Vector> x = Variable.Observed(xdata, j).Named("x");

            var yTrue = Variable.Array<bool>(j).Named("yTrue");

            // Bayes Point Machine
            yTrue[j] = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(w, x[j]).Named("innerProduct") + mean, noiseVariance) > 0;
            using (Variable.ForEach(j))
            {
                using (Variable.If(yTrue[j]))
                    y[j] = Variable.Bernoulli(probTrue);
                using (Variable.IfNot(yTrue[j]))
                    y[j] = Variable.Bernoulli(probFalse);
            }
        }


        public DistributionArray<Bernoulli> TestBPM_NoisePrecisionEP(double noiseRate, out double noiseEstimate)
        {
            int K = xtrain[0].Count;
            // Create target y
            VariableArray<bool> y = Variable.Observed(ytrain).Named("y");
            Variable<Vector> w = Variable.Random(new VectorGaussian(Vector.Zero(K),
                                                                    PositiveDefiniteMatrix.Identity(K))).Named("w");
            var mean = Variable.GaussianFromMeanAndPrecision(0, .1);
            var noise = Variable.GammaFromShapeAndRate(2, noiseRate);
            //var noise = Variable.Random(Gamma.PointMass(.1));
            BayesPointMachine_NoisePrecisionEP(xtrain, w, y, mean, noise);
            InferenceEngine engine = new InferenceEngine();
            VectorGaussian wPosterior = engine.Infer<VectorGaussian>(w);
            var noisePost = engine.Infer<Gamma>(noise);
            noiseEstimate = 1.0/noisePost.GetMean();
            var meanPost = engine.Infer<Gaussian>(mean);
            VariableArray<bool> yt = Variable.Array<bool>(new Range(ytest.Length)).Named("ytest");
            BayesPointMachine_NoisePrecisionEP(xtest, Variable.Random(wPosterior).Named("w"), yt, Variable.Random(meanPost), Variable.Random(noisePost));
            return engine.Infer<DistributionArray<Bernoulli>>(yt);
        }

        public void BayesPointMachine_NoisePrecisionEP(Vector[] xdata, Variable<Vector> w, VariableArray<bool> y, Variable<double> mean, Variable<double> noisePrecision)
        {
            // Create x vector, augmented by 1
            Range j = y.Range.Named("person");
            VariableArray<Vector> x = Variable.Observed(xdata, j).Named("x");

            // Bayes Point Machine
            y[j] = Variable.GaussianFromMeanAndPrecision(Variable.InnerProduct(w, x[j]).Named("innerProduct") + mean, noisePrecision) > 0;
        }

        public DistributionArray<Bernoulli> TestBPM_NoisePrecisionVMP(double noiseRate, out double noiseEstimate)
        {
            int K = xtrain[0].Count;
            // Create target y
            VariableArray<bool> y = Variable.Observed(ytrain).Named("y");
            Variable<Vector> w = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(K),
                                                                             Variable.WishartFromShapeAndScale(.1, PositiveDefiniteMatrix.Identity(K))).Named("w");
            var mean = Variable.GaussianFromMeanAndPrecision(0, .1);
            var noise = Variable.GammaFromShapeAndRate(2, noiseRate);
            //var noise = Variable.Random(Gamma.PointMass(.1));
            BayesPointMachine_NoisePrecisionVMP(xtrain, w, y, mean, noise);
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            VectorGaussian wPosterior = engine.Infer<VectorGaussian>(w);
            var noisePost = engine.Infer<Gamma>(noise);
            noiseEstimate = 1.0/noisePost.GetMean();
            //Console.WriteLine("Dist over w=\n" + wPosterior);
            //Console.WriteLine("Dist over noise=\n" + noisePost);
            var meanPost = engine.Infer<Gaussian>(mean);
            VariableArray<bool> yt = Variable.Array<bool>(new Range(ytest.Length)).Named("ytest");
            BayesPointMachine_NoisePrecisionVMP(xtest, Variable.Random(wPosterior).Named("w"), yt, Variable.Random(meanPost), Variable.Random(noisePost));
            return engine.Infer<DistributionArray<Bernoulli>>(yt);
        }

        public void BayesPointMachine_NoisePrecisionVMP(Vector[] xdata, Variable<Vector> w, VariableArray<bool> y, Variable<double> mean, Variable<double> noisePrecision)
        {
            // Create x vector, augmented by 1
            Range j = y.Range.Named("person");
            VariableArray<Vector> x = Variable.Observed(xdata, j).Named("x");

            // Bayes Point Machine
            y[j] = Variable.Bernoulli(Variable.Logistic(Variable.GaussianFromMeanAndPrecision(Variable.InnerProduct(w, x[j]).Named("innerProduct") + mean, noisePrecision)));
        }


        public DistributionArray<Bernoulli> TestBPM_LabelNoiseConstrained(double initCount, out double noiseEstimate)
        {
            noiseEstimate = double.NaN;
            // data
            int K = xtrain[0].Count;
            // Create target y
            VariableArray<bool> y = Variable.Observed(ytrain).Named("y");
            Variable<Vector> w = Variable.Random(new VectorGaussian(Vector.Zero(K),
                                                                    PositiveDefiniteMatrix.Identity(K))).Named("w");
            var probTrueLogOdds = Variable.GaussianFromMeanAndVariance(initCount, 10);
            var ProbFalseLogOdds = Variable.GaussianFromMeanAndVariance(-initCount, 10);
            Variable.ConstrainPositive(probTrueLogOdds - ProbFalseLogOdds);
            var ProbTrue = Variable.Logistic(probTrueLogOdds).Named("TPR");
            var ProbFalse = Variable.Logistic(ProbFalseLogOdds).Named("FNR");
            var mean = Variable.GaussianFromMeanAndPrecision(0, .1).Named("mean");
            BayesPointMachine_LabelNoise(xtrain, w, y, ProbTrue, ProbFalse, mean);
            InferenceEngine engine = new InferenceEngine();
            VectorGaussian wPosterior = engine.Infer<VectorGaussian>(w);
            Beta probTruePost = engine.Infer<Beta>(ProbTrue);
            Beta probFalsePost = engine.Infer<Beta>(ProbFalse);
            var meanPost = engine.Infer<Gaussian>(mean);
            VariableArray<bool> yt = Variable.Array<bool>(new Range(ytest.Length)).Named("ytest");
            BayesPointMachine_LabelNoise(xtest, Variable.Random(wPosterior).Named("w"), yt, Variable.Random(probTruePost), Variable.Random(probFalsePost),
                                         Variable.Random(meanPost));
            return engine.Infer<DistributionArray<Bernoulli>>(yt);
        }


        public DistributionArray<Bernoulli> TestBPM_LabelNoise(double initCount, out double noiseEstimate)
        {
            noiseEstimate = double.NaN;
            // data
            int K = xtrain[0].Count;
            // Create target y
            VariableArray<bool> y = Variable.Observed(ytrain).Named("y");
            Variable<Vector> w = Variable.Random(new VectorGaussian(Vector.Zero(K),
                                                                    PositiveDefiniteMatrix.Identity(K))).Named("w");
            var ProbTrue = Variable.Beta(1, initCount).Named("TPR");
            var ProbFalse = Variable.Beta(initCount, 1).Named("FNR");
            var mean = Variable.GaussianFromMeanAndPrecision(0, .1).Named("mean");
            BayesPointMachine_LabelNoise(xtrain, w, y, ProbTrue, ProbFalse, mean);
            InferenceEngine engine = new InferenceEngine();
            VectorGaussian wPosterior = engine.Infer<VectorGaussian>(w);
            Beta probTruePost = engine.Infer<Beta>(ProbTrue);
            Beta probFalsePost = engine.Infer<Beta>(ProbFalse);
            //Console.WriteLine("Dist over w=\n" + wPosterior);
            //Console.WriteLine("Dist over p_t=\n" + probTruePost);
            //Console.WriteLine("Dist over p_f=\n" + probFalsePost);
            var meanPost = engine.Infer<Gaussian>(mean);
            VariableArray<bool> yt = Variable.Array<bool>(new Range(ytest.Length)).Named("ytest");
            BayesPointMachine_LabelNoise(xtest, Variable.Random(wPosterior).Named("w"), yt, Variable.Random(probTruePost), Variable.Random(probFalsePost),
                                         Variable.Random(meanPost));
            return engine.Infer<DistributionArray<Bernoulli>>(yt);
        }


        public DistributionArray<Bernoulli> TestBPM_LabelNoiseFixed(double eps, out double noiseEstimate)
        {
            noiseEstimate = double.NaN;
            int K = xtrain[0].Count;
            // Create target y
            VariableArray<bool> y = Variable.Observed(ytrain).Named("y");
            Variable<Vector> w = Variable.Random(new VectorGaussian(Vector.Zero(K),
                                                                    PositiveDefiniteMatrix.Identity(K))).Named("w");
            var ProbTrue = Variable.Observed<double>(1.0 - eps).Named("TPR");
            var ProbFalse = Variable.Observed<double>(eps).Named("FNR");
            var mean = Variable.GaussianFromMeanAndPrecision(0, .1).Named("mean");
            BayesPointMachine_LabelNoise(xtrain, w, y, ProbTrue, ProbFalse, mean);
            //InferenceEngine.DefaultEngine.Compiler.UseSerialSchedules = true;
            InferenceEngine engine = new InferenceEngine();
            VectorGaussian wPosterior = engine.Infer<VectorGaussian>(w);
            var meanPost = engine.Infer<Gaussian>(mean);
            VariableArray<bool> yt = Variable.Array<bool>(new Range(ytest.Length)).Named("ytest");
            BayesPointMachine_LabelNoise(xtest, Variable.Random(wPosterior).Named("w"), yt, ProbTrue, ProbFalse, Variable.Random(meanPost));
            return engine.Infer<DistributionArray<Bernoulli>>(yt);
        }

        public void BayesPointMachine_LabelNoise(Vector[] xdata, Variable<Vector> w, VariableArray<bool> y, Variable<double> probTrue, Variable<double> probFalse,
                                                 Variable<double> mean)
        {
            // Create x vector, augmented by 1
            Range j = y.Range.Named("person");
            VariableArray<Vector> x = Variable.Observed(xdata, j).Named("x");
            var yTrue = Variable.Array<bool>(j).Named("yTrue");

            // Bayes Point Machine
            yTrue[j] = (Variable.InnerProduct(w, x[j]).Named("innerProduct") + mean).Named("g") > 0;
            using (Variable.ForEach(j))
            {
                using (Variable.If(yTrue[j]))
                    y[j] = Variable.Bernoulli(probTrue);
                using (Variable.IfNot(yTrue[j]))
                    y[j] = Variable.Bernoulli(probFalse);
            }
        }


        public DistributionArray<Bernoulli> TestNaiveBayes(double a, out double noiseEstimate)
        {
            noiseEstimate = double.NaN;
            int K = xtrain[0].Count;
            // Create target y
            VariableArray<bool> y = Variable.Observed(ytrain).Named("y");
            Variable<Vector> meanTrue = Variable.Random(new VectorGaussian(Vector.Zero(K),
                                                                           PositiveDefiniteMatrix.Identity(K))).Named("m1");
            Variable<Vector> meanFalse = Variable.Random(new VectorGaussian(Vector.Zero(K),
                                                                            PositiveDefiniteMatrix.Identity(K))).Named("m2");
            var precTrue = Variable.Random(new Wishart(a, PositiveDefiniteMatrix.Identity(K)));
            var precFalse = Variable.Random(new Wishart(a, PositiveDefiniteMatrix.Identity(K)));
            NaiveBayes(xtrain, meanTrue, meanFalse, precTrue, precFalse, y);
            //InferenceEngine.DefaultEngine.Compiler.UseSerialSchedules = true;
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            var meanTruePost = engine.Infer<VectorGaussian>(meanTrue);
            var meanFalsePost = engine.Infer<VectorGaussian>(meanFalse);
            var precTruePost = engine.Infer<Wishart>(precTrue);
            var precFalsePost = engine.Infer<Wishart>(precFalse);
            var testRange = new Range(ytest.Length);
            VariableArray<bool> yt = Variable.Array<bool>(testRange).Named("ytest");
            yt[testRange] = Variable.Bernoulli(0.5).ForEach(testRange);
            NaiveBayes(xtest, Variable.Random(meanTruePost), Variable.Random(meanFalsePost), Variable.Random<PositiveDefiniteMatrix>(precTruePost),
                       Variable.Random<PositiveDefiniteMatrix>(precFalsePost), yt);
            return engine.Infer<DistributionArray<Bernoulli>>(yt);
        }

        public void NaiveBayes(Vector[] xdata, Variable<Vector> meanTrue, Variable<Vector> meanFalse, Variable<PositiveDefiniteMatrix> precTrue,
                               Variable<PositiveDefiniteMatrix> precFalse, VariableArray<bool> y)
        {
            // Create x vector, augmented by 1
            Range j = y.Range.Named("person");
            VariableArray<Vector> x = Variable.Observed(xdata, j).Named("x");

            // Bayes Point Machine
            using (Variable.ForEach(j))
            {
                using (Variable.If(y[j]))
                    x[j] = Variable.VectorGaussianFromMeanAndPrecision(meanTrue, precTrue);
                using (Variable.IfNot(y[j]))
                    x[j] = Variable.VectorGaussianFromMeanAndPrecision(meanFalse, precFalse);
            }
        }


        public DistributionArray<Bernoulli> TestNaiveBayesDiagonal(double a, out double noiseEstimate)
        {
            noiseEstimate = double.NaN;
            var xtrainArray = xtrain.Select(i => i.ToArray()).ToArray();
            var xtestArray = xtest.Select(i => i.ToArray()).ToArray();
            int K = xtrain[0].Count;
            var k = new Range(K);
            // Create target y
            VariableArray<bool> y = Variable.Observed(ytrain).Named("y");
            var meanTrue = Variable.Array<double>(k);
            var meanTruePriorObs = Enumerable.Range(0, K).Select(i => new Gaussian(0, 1)).ToArray();
            var meanTruePrior = Variable.Observed<Gaussian>(meanTruePriorObs, k);
            meanTrue[k] = Variable<double>.Random(meanTruePrior[k]);
            var precTrue = Variable.Array<double>(k);
            var precTruePriorObs = Enumerable.Range(0, K).Select(i => new Gamma(1, 1)).ToArray();
            var precTruePrior = Variable.Observed<Gamma>(precTruePriorObs, k);
            precTrue[k] = Variable<double>.Random(precTruePrior[k]);
            var meanFalse = Variable.Array<double>(k);
            var meanFalsePriorObs = Enumerable.Range(0, K).Select(i => new Gaussian(0, 1)).ToArray();
            var meanFalsePrior = Variable.Observed<Gaussian>(meanFalsePriorObs, k);
            meanFalse[k] = Variable<double>.Random(meanFalsePrior[k]);
            var precFalse = Variable.Array<double>(k);
            var precFalsePriorObs = Enumerable.Range(0, K).Select(i => new Gamma(1, 1)).ToArray();
            var precFalsePrior = Variable.Observed<Gamma>(precFalsePriorObs, k);
            precFalse[k] = Variable<double>.Random(precFalsePrior[k]);
            NaiveBayesDiagonal(xtrainArray, meanTrue, meanFalse, precTrue, precFalse, y);
            //InferenceEngine.DefaultEngine.Compiler.UseSerialSchedules = true;
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            var meanTruePost = engine.Infer<Gaussian[]>(meanTrue);
            var meanFalsePost = engine.Infer<Gaussian[]>(meanFalse);
            var precTruePost = engine.Infer<Gamma[]>(precTrue);
            var precFalsePost = engine.Infer<Gamma[]>(precFalse);
            var testRange = new Range(ytest.Length);
            VariableArray<bool> yt = Variable.Array<bool>(testRange).Named("ytest");
            yt[testRange] = Variable.Bernoulli(0.5).ForEach(testRange);
            meanTruePrior.ObservedValue = meanTruePost;
            meanFalsePrior.ObservedValue = meanFalsePost;
            precTruePrior.ObservedValue = precTruePost;
            precFalsePrior.ObservedValue = precFalsePost;
            meanTrue = Variable.Array<double>(k);
            meanTrue[k] = Variable<double>.Random(meanTruePrior[k]);
            precTrue = Variable.Array<double>(k);
            precTrue[k] = Variable<double>.Random(precTruePrior[k]);
            meanFalse = Variable.Array<double>(k);
            meanFalse[k] = Variable<double>.Random(meanFalsePrior[k]);
            precFalse = Variable.Array<double>(k);
            precFalse[k] = Variable<double>.Random(precFalsePrior[k]);
            NaiveBayesDiagonal(xtestArray, meanTrue, meanFalse, precTrue, precFalse, yt);
            return engine.Infer<DistributionArray<Bernoulli>>(yt);
        }

        public DistributionArray<Bernoulli> TestNaiveBayesDiagonalHierarchical(double a, out double noiseEstimate)
        {
            noiseEstimate = double.NaN;
            var xtrainArray = xtrain.Select(i => i.ToArray()).ToArray();
            var xtestArray = xtest.Select(i => i.ToArray()).ToArray();
            int K = xtrain[0].Count;
            var k = new Range(K);
            // Create target y
            VariableArray<bool> y = Variable.Observed(ytrain).Named("y");
            var meanTrue = Variable.Array<double>(k);
            meanTrue[k] = Variable.GaussianFromMeanAndPrecision(
                Variable.GaussianFromMeanAndPrecision(0, 1),
                Variable.GammaFromShapeAndRate(.1, .1)).ForEach(k);
            var precTrue = Variable.Array<double>(k);
            precTrue[k] = Variable.GammaFromShapeAndRate(
                1,
                Variable.GammaFromShapeAndRate(.1, .1)).ForEach(k);
            var meanFalse = Variable.Array<double>(k);
            meanFalse[k] = Variable.GaussianFromMeanAndPrecision(
                Variable.GaussianFromMeanAndPrecision(0, 1),
                Variable.GammaFromShapeAndRate(.1, .1)).ForEach(k);
            var precFalse = Variable.Array<double>(k);
            precFalse[k] = Variable.GammaFromShapeAndRate(
                1,
                Variable.GammaFromShapeAndRate(.1, .1)).ForEach(k);
            NaiveBayesDiagonal(xtrainArray, meanTrue, meanFalse, precTrue, precFalse, y);
            //InferenceEngine.DefaultEngine.Compiler.UseSerialSchedules = true;
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            var meanTruePost = engine.Infer<Gaussian[]>(meanTrue);
            var meanFalsePost = engine.Infer<Gaussian[]>(meanFalse);
            var precTruePost = engine.Infer<Gamma[]>(precTrue);
            var precFalsePost = engine.Infer<Gamma[]>(precFalse);
            var testRange = new Range(ytest.Length);
            VariableArray<bool> yt = Variable.Array<bool>(testRange).Named("ytest");
            yt[testRange] = Variable.Bernoulli(0.5).ForEach(testRange);
            var meanTruePrior = Variable.Observed<Gaussian>(meanTruePost, k);
            var precTruePrior = Variable.Observed<Gamma>(precTruePost, k);
            var meanFalsePrior = Variable.Observed<Gaussian>(meanFalsePost, k);
            var precFalsePrior = Variable.Observed<Gamma>(precFalsePost, k);
            meanTrue = Variable.Array<double>(k);
            meanTrue[k] = Variable<double>.Random(meanTruePrior[k]);
            precTrue = Variable.Array<double>(k);
            precTrue[k] = Variable<double>.Random(precTruePrior[k]);
            meanFalse = Variable.Array<double>(k);
            meanFalse[k] = Variable<double>.Random(meanFalsePrior[k]);
            precFalse = Variable.Array<double>(k);
            precFalse[k] = Variable<double>.Random(precFalsePrior[k]);
            NaiveBayesDiagonal(xtestArray, meanTrue, meanFalse, precTrue, precFalse, yt);
            return engine.Infer<DistributionArray<Bernoulli>>(yt);
        }

        public void NaiveBayesDiagonal(double[][] xdata, VariableArray<double> meanTrue, VariableArray<double> meanFalse, VariableArray<double> precTrue,
                                       VariableArray<double> precFalse, VariableArray<bool> y)
        {
            // Create x vector, augmented by 1
            Range j = y.Range.Named("person");
            var k = meanFalse.Range;
            var x = Variable.Array(Variable.Array<double>(k), j);
            x.ObservedValue = xdata;

            // Bayes Point Machine
            using (Variable.ForEach(j))
                //using (Variable.ForEach(k))
            {
                using (Variable.If(y[j]))
                    x[j][k] = Variable.GaussianFromMeanAndPrecision(meanTrue[k], precTrue[k]);
                using (Variable.IfNot(y[j]))
                    x[j][k] = Variable.GaussianFromMeanAndPrecision(meanFalse[k], precFalse[k]);
            }
        }


        public void GateModel()
        {
            var trueY = Variable.Bernoulli(.5).Named("trueY");
            var obsY = Variable.New<bool>().Named("obsY");
            var probTrue = Variable.Beta(1, 3).Named("TPR");
            var probFalse = Variable.Beta(3, 1).Named("FNR");
            using (Variable.If(trueY))
                obsY = Variable.Bernoulli(probTrue);
            using (Variable.IfNot(trueY))
                obsY = Variable.Bernoulli(probFalse);
            obsY.ObservedValue = true;
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(trueY));
        }
    }
}
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Xunit;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Assert = Xunit.Assert;
using System.IO;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Serialization;

namespace Microsoft.ML.Probabilistic.Tests
{
    /// <summary>
    /// VMP Msl tests (scalar and array)
    /// </summary>
    public class VmpMslTests
    {
        [Fact]
        [Trait("Category", "CsoftModel")]
        public void SimpleGaussianTest()
        {
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(SimpleGaussianModel);
            ca.Execute(30);
            Gaussian meanExpected = Gaussian.FromMeanAndVariance(5.9603207170807826, 0.66132138200164436);
            Gamma precisionExpected = Gamma.FromShapeAndRate(2, 2.6628958274937107);
            Gaussian meanActual = ca.Marginal<Gaussian>("mean");
            Gamma precisionActual = ca.Marginal<Gamma>("precision");
            Console.WriteLine("mean = {0} should be {1}", meanActual, meanExpected);
            Console.WriteLine("precision = {0} should be {1}", precisionActual, precisionExpected);
            Assert.True(meanExpected.MaxDiff(meanActual) < 1e-4);
            Assert.True(precisionExpected.MaxDiff(precisionActual) < 1e-4);
        }

        private void SimpleGaussianModel()
        {
            double mean = Factor.Random(new Gaussian(0, 100));
            double precision = Factor.Random(Gamma.FromShapeAndScale(1, 1));
            double x1 = Gaussian.Sample(mean, precision);
            double x2 = Gaussian.Sample(mean, precision);
            Constrain.Equal(x1, 5.0);
            Constrain.Equal(x2, 7.0);
            InferNet.Infer(mean, nameof(mean));
            InferNet.Infer(precision, nameof(precision));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void SimpleGaussianWithLoopsMsl()
        {
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            double[] data = {5, 7}; //, 9, 11, 17, 41, 32 };
            var ca = engine.Compiler.Compile(SimpleGaussianWithLoopsModel, data);
            ca.Execute(20);
            Console.WriteLine("Marginal for mean = " + ca.Marginal("mean"));
            Console.WriteLine("Marginal for precision = " + ca.Marginal("precision"));
            Gaussian meanExpected = Gaussian.FromMeanAndVariance(5.9603207170807826, 0.66132138200164436);
            Gaussian meanActual = ca.Marginal<Gaussian>("mean");
            Assert.True(meanExpected.MaxDiff(meanActual) < 1e-4);
            Gamma precisionExpected = Gamma.FromShapeAndRate(2, 2.6628958274937107);
            Gamma precisionActual = ca.Marginal<Gamma>("precision");
            Assert.True(precisionExpected.MaxDiff(precisionActual) < 1e-4);
        }

        private void SimpleGaussianWithLoopsModel(double[] data)
        {
            double mean = Factor.Random(new Gaussian(0, 100));
            double precision = Factor.Random(Gamma.FromShapeAndRate(1, 1));
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = Gaussian.Sample(mean, precision);
            }
            InferNet.Infer(mean, nameof(mean));
            InferNet.Infer(precision, nameof(precision));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void SimpleGaussianWithTwoLoopsMsl()
        {
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            double[] data = {5, 7}; //, 9, 11, 17, 41, 32 };
            Gaussian meanExpected = Gaussian.FromMeanAndVariance(5.9603207170807826, 0.66132138200164436);
            Gamma precisionExpected = Gamma.FromShapeAndRate(2, 2.6628958274937107);
            var ca = engine.Compiler.Compile(SimpleGaussianWithTwoLoopsModel, data);
            ca.Execute(20);
            Console.WriteLine("Marginal for mean = " + ca.Marginal("mean"));
            Console.WriteLine("Marginal for precision = " + ca.Marginal("precision"));
            DistributionArray<Gaussian> meanActual = ca.Marginal<DistributionArray<Gaussian>>("mean");
            DistributionArray<Gamma> precisionActual = ca.Marginal<DistributionArray<Gamma>>("precision");
            for (int j = 0; j < meanActual.Count; j++)
            {
                Assert.True(meanExpected.MaxDiff(meanActual[j]) < 1e-4);
                Assert.True(precisionExpected.MaxDiff(precisionActual[j]) < 1e-4);
            }
        }

        private void SimpleGaussianWithTwoLoopsModel(double[] data)
        {
            double[] mean = new double[2];
            double[] precision = new double[2];
            for (int j = 0; j < 2; j++)
            {
                mean[j] = Factor.Random(new Gaussian(0, 100));
                precision[j] = Factor.Random(Gamma.FromShapeAndRate(1, 1));
                for (int i = 0; i < data.Length; i++)
                {
                    data[i] = Gaussian.Sample(mean[j], precision[j]);
                }
            }
            InferNet.Infer(mean, nameof(mean));
            InferNet.Infer(precision, nameof(precision));
        }


        [Fact]
        [Trait("Category", "CsoftModel")]
        public void FactorAnalysisMslTest()
        {
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            double[,] dataIn = MatlabReader.ReadMatrix(new double[400,64],
                Path.Combine(TestUtils.DataFolderPath, "pca.txt"),
                ' ');
            int C = 8;
            engine.ShowProgress = true;
            engine.ShowTimings = true;
            engine.NumberOfIterations = 20;
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(FactorAnalysisModel, C, dataIn,
                                             BioTests.RandomGaussianArray(C, dataIn.GetLength(1)),
                                             BioTests.RandomGaussianArray(dataIn.GetLength(0), C));
            ca.Execute(engine.NumberOfIterations);
            DistributionArray2D<Gaussian> wMarginal = ca.Marginal<DistributionArray2D<Gaussian>>("W");
            DistributionArray2D<Gaussian> xMarginal = ca.Marginal<DistributionArray2D<Gaussian>>("x");
            //WriteMatrix(wMarginal.ToArray<Gaussian[,]>(), @"..\..\faresultsW.txt");
            //WriteMatrix(xMarginal.ToArray<Gaussian[,]>(), @"..\..\faresultsX.txt");

            // Reconstruct
            DistributionArray2D<Gaussian> productMarginal = MatrixMultiplyOp.MatrixMultiplyAverageLogarithm(xMarginal, wMarginal, null);
            double error = 0;
            for (int i = 0; i < productMarginal.GetLength(0); i++)
            {
                for (int j = 0; j < productMarginal.GetLength(1); j++)
                {
                    //Assert.True(productMarginal[i,j].GetLogProb(dataIn[i,j]) > -130);
                    error += System.Math.Abs(productMarginal[i, j].GetMean() - dataIn[i, j]);
                }
            }
            error /= productMarginal.Count;
            // error = 0.121231278712027
            Console.WriteLine("error = {0}", error);
            Assert.True(error < 0.15); // C=2: 0.15
        }

        private void FactorAnalysisModel(int C, double[,] data, IDistribution<double[,]> initW, IDistribution<double[,]> initX)
        {
            int N = data.GetLength(0);
            int d = data.GetLength(1);
            // Mixing matrix
            double[,] W = new double[C,d];
            Attrib.InitialiseTo(W, initW);
            for (int j = 0; j < C; j++)
            {
                for (int j2 = 0; j2 < d; j2++)
                {
                    W[j, j2] = Factor.Random(Gaussian.FromMeanAndVariance(0, 1));
                }
            }

            // Noise
            double[] tau = new double[d];
            for (int i = 0; i < d; i++)
            {
                tau[i] = Factor.Random(Gamma.FromShapeAndRate(0.001, 0.001));
                //tau[i] = Factor.Random(Gamma.PointMass(10));
            }
            // Sources
            double[,] x = new double[N,C];
            Attrib.InitialiseTo(x, initX);
            for (int k = 0; k < N; k++)
            {
                for (int k3 = 0; k3 < C; k3++)
                {
                    x[k, k3] = Factor.Random(Gaussian.FromMeanAndVariance(0, 1));
                }
            }
            double[,] XtimesW = new double[N,d];
            XtimesW = Factor.MatrixMultiply(x, W);
            for (int k4 = 0; k4 < N; k4++)
            {
                for (int k2 = 0; k2 < d; k2++)
                {
                    data[k4, k2] = Factor.Gaussian(XtimesW[k4, k2], tau[k2]);
                }
            }
            InferNet.Infer(W, nameof(W));
            InferNet.Infer(x, nameof(x));
            InferNet.Infer(tau, nameof(tau));
        }
    }
}
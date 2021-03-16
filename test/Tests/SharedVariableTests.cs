// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Xunit;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Tests
{
    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
    using Microsoft.ML.Probabilistic.Utilities;
    using GaussianArrayArray = DistributionRefArray<DistributionStructArray<Gaussian, double>, double[]>;
    using GaussianArray = DistributionStructArray<Gaussian, double>;
    using DirichletArray = DistributionRefArray<Dirichlet, Vector>;
    using GaussianArrayArrayArray = DistributionRefArray<DistributionRefArray<DistributionStructArray<Gaussian, double>, double[]>, double[][]>;
    using Microsoft.ML.Probabilistic.Algorithms;
    using Range = Microsoft.ML.Probabilistic.Models.Range;

    /// <summary>
    /// Provides test routines for shared variables on Gaussian and Discrete.
    /// Each test case uses a jagged 2D array of data points, where the outer dimension refers to chunks.
    /// 
    /// The routines refer to:
    /// - Estimating a single Gaussian from a chunked array (TestGaussian)
    /// - Estimating a single Discrete from a chunked array (TestDiscrete)
    /// - Estimating an 2D-array of Gaussians (one per data entry) (TestGaussianArrayArray)
    /// - Estimating an 1D-array of Gaussians (one for each inner-index) (TestGaussianArray)  !!!  DOES NOT WORK !!!
    /// - Estimating an 1D-array of Discrete (one for each inner-index) (TestDiscreteArray)  !!!  DOES NOT WORK !!!
    /// </summary>
    public class SharedVariableTests
    {
        [Fact]
        public void SharedVariableArray_DeepJagged2()
        {
            Range outer = new Range(1);
            Range inner = new Range(1);
            Range innerinner = new Range(1);

            var prior = new GaussianArrayArrayArray(new GaussianArrayArray(
                new GaussianArray(Gaussian.Uniform(), innerinner.SizeAsInt), 
                inner.SizeAsInt), outer.SizeAsInt);
            var x = SharedVariable<double[][][]>.Random(
                Variable.Array(Variable.Array<double>(innerinner), inner), outer, prior);
        }

        [Fact]
        public void SharedVariableArray_DeepJagged()
        {
            int depth = 5;
            Range[] ranges = new Range[depth];
            for (int i = 0; i < depth; i++)
            {
                ranges[i] = new Range(2);
            }

            GaussianArrayArray gaa = new GaussianArrayArray(2);
            gaa[0] = new GaussianArray(2);
            gaa[1] = new GaussianArray(2);

            var a = SharedVariable<double[][]>.Random(
                Variable<double>.Array(ranges[1]), ranges[0], gaa);
        }

        [Fact]
        public void SharedVariableArrayJaggedTest()
        {
            SharedVariableArrayJagged(true);
            SharedVariableArrayJagged(false);
        }

        private void SharedVariableArrayJagged(bool divideMessages)
        {
            double[][][] data = new double[][][]
                {
                    new double[][] {new double[] {1, 2}, new double[] {3, 4}},
                    new double[][] {new double[] {5, 6}, new double[] {7, 8}}
                };
            GaussianArrayArray expectedW;
            double expectedEvidence = GaussianModel(data, out expectedW);
            sharedVariableArrayJagged_GaussianModel(data, expectedW, expectedEvidence, divideMessages);
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        private double GaussianModel(double[][][] data, out GaussianArrayArray posteriorW)
        {
            Range user = new Range(data.Length);
            Range outer = new Range(data[0].Length);
            Range inner = new Range(data[0][0].Length);
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            var evidenceIfBlock = Variable.If(evidence);
            var w = Variable.Array<double>(Variable.Array<double>(inner), outer).Named("w");
            if (false)
            {
                // this should be equivalent to below
                GaussianArrayArray priorW = new GaussianArrayArray(new GaussianArray(new Gaussian(0, 1), inner.SizeAsInt), outer.SizeAsInt);
                w.SetTo(Variable<double[][]>.Random(Variable.Constant(priorW).Named("priorW")));
            }
            else
            {
                using (Variable.ForEach(outer))
                {
                    w[outer][inner] = Variable.GaussianFromMeanAndPrecision(0.0, 1.0).ForEach(inner);
                }
            }
            //var x = Variable.Array<double>(Variable.Array<double>(Variable.Array<double>(inner),outer),user).Named("x");
            var x = Variable.Constant(data, user, outer, inner).Named("x");
            x[user][outer][inner] = Variable.GaussianFromMeanAndPrecision(w[outer][inner], 1.0).ForEach(user);
            evidenceIfBlock.CloseBlock();
            InferenceEngine engine = new InferenceEngine();
            posteriorW = engine.Infer<GaussianArrayArray>(w);
            return engine.Infer<Bernoulli>(evidence).LogOdds;
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        private void sharedVariableArrayJagged_GaussianModel(double[][][] data, GaussianArrayArray expectedW, double expectedEvidence, bool divideMessages)
        {
            int numChunks = data.Length;
            Range outer = new Range(data[0].Length);
            Range inner = new Range(data[0][0].Length);
            GaussianArrayArray priorW = new GaussianArrayArray(new GaussianArray(new Gaussian(0, 1), inner.SizeAsInt), outer.SizeAsInt);

            // w is a jagged SharedVariableArray
            var w = SharedVariable<double>.Random(Variable.Array<double>(inner), outer, priorW, divideMessages).Named("w");
            var evidence = SharedVariable<bool>.Random(new Bernoulli(0.5), divideMessages).Named("evidence");
            evidence.IsEvidenceVariable = true;

            Model model = new Model(numChunks);
            var evidenceIfBlock = Variable.If(evidence.GetCopyFor(model));
            var x = Variable.Array<double>(Variable.Array<double>(inner), outer).Named("x");
            x[outer][inner] = Variable.GaussianFromMeanAndPrecision(w.GetCopyFor(model)[outer][inner], 1.0);
            evidenceIfBlock.CloseBlock();
            InferenceEngine engine = new InferenceEngine(); //new VariationalMessagePassing());

            for (int pass = 0; pass < 5; pass++)
            {
                for (int c = 0; c < numChunks; c++)
                {
                    x.ObservedValue = data[c];
                    model.InferShared(engine, c);
                }
            }

            GaussianArrayArray actualW = w.Marginal<GaussianArrayArray>();
            //double actualEvidence = evidence.Marginal<Bernoulli>().LogOdds;
            double actualEvidence = Model.GetEvidenceForAll(model);
            Console.WriteLine(StringUtil.JoinColumns("w = ", actualW, " should be ", expectedW));
            Console.WriteLine("evidence = {0} (should be {1})", actualEvidence, expectedEvidence);
            Assert.True(expectedW.MaxDiff(actualW) < 1e-4);
            Assert.Equal(expectedEvidence, actualEvidence, 1e-4);
            Gaussian[][] wActualArray = w.Marginal<Gaussian[][]>();
            Assert.True(expectedW.MaxDiff(Distribution<double>.Array(wActualArray)) < 1e-4);
        }

        [Fact]
        public void SharedVariableArray_Regression()
        {
            sharedVariableArray_Regression(true);
            sharedVariableArray_Regression(false);
        }

        private void sharedVariableArray_Regression(bool divideMessages)
        {
            double[][] inputs = new double[][] {new double[] {1, 2}};
            double[] outputs = new double[] {7.9};

            GaussianArray expectedW = RegressionModel(inputs, outputs);
            //GaussianArray expectedW = RegressionModel2(inputs[0], outputs[0]);
            //Console.WriteLine(expectedW);
            sharedVariableArray_RegressionModel(inputs, outputs, expectedW, divideMessages);

            inputs = new double[][] {new double[] {1, 2}, new double[] {3, 4}};
            outputs = new double[] {7.9, 1.3};

            expectedW = RegressionModel(inputs, outputs);
            sharedVariableArray_RegressionModel(inputs, outputs, expectedW, divideMessages);
        }

        private GaussianArray RegressionModel(double[][] inputs, double[] outputs)
        {
            int ndims = inputs[0].Length;
            Range dim = new Range(ndims).Named("dim");

            VariableArray<double> w = Variable.Array<double>(dim).Named("w");
            w[dim] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(dim);
            int dataCount = outputs.Length;
            Range item = new Range(dataCount).Named("item");
            var x = Variable.Array<double>(Variable.Array<double>(dim), item).Named("x");
            var ytmp = Variable.Array<double>(Variable.Array<double>(dim), item).Named("ytmp");
            ytmp[item][dim] = w[dim]*x[item][dim];
            VariableArray<double> y = Variable.Array<double>(item).Named("y");
            y[item] = Variable.Sum(ytmp[item]);
            VariableArray<double> yNoisy = Variable.Array<double>(item).Named("yNoisy");
            yNoisy[item] = Variable.GaussianFromMeanAndVariance(y[item], 0.1).Named("yNoisy");

            x.ObservedValue = inputs;
            yNoisy.ObservedValue = outputs;

            // does not work with VMP because Sum(array) cannot take a derived argument
            InferenceEngine engine = new InferenceEngine(); //new VariationalMessagePassing());
            engine.NumberOfIterations = 100;
            GaussianArray actualW = engine.Infer<GaussianArray>(w);
            return actualW;
        }

        private GaussianArray RegressionModel2(double[] inputs, double outputs)
        {
            int ndims = inputs.Length;
            Range dim = new Range(ndims).Named("dim");

            VariableArray<double> w = Variable.Array<double>(dim).Named("w");
            w[dim] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(dim);
            VariableArray<double> x = Variable.Array<double>(dim).Named("x");
            VariableArray<double> ytmp = Variable.Array<double>(dim).Named("ytmp");
            ytmp[dim] = w[dim]*x[dim];
            Variable<double> y = Variable.Sum(ytmp).Named("y");
            Variable<double> yNoisy = Variable.GaussianFromMeanAndVariance(y, 0.1).Named("yNoisy");

            x.ObservedValue = inputs;
            yNoisy.ObservedValue = outputs;

            // does not work with VMP because Sum(array) cannot take a derived argument
            InferenceEngine engine = new InferenceEngine(); //new VariationalMessagePassing());
            engine.NumberOfIterations = 100;
            GaussianArray actualW = engine.Infer<GaussianArray>(w);
            return actualW;
        }

        private void sharedVariableArray_RegressionModel(double[][] inputs, double[] outputs, GaussianArray expectedW, bool divideMessages)
        {
            int numChunks = outputs.Length;
            int ndims = inputs[0].Length;
            GaussianArray priorW = new GaussianArray(new Gaussian(0, 1), ndims);
            Range dim = new Range(ndims).Named("dim");

            SharedVariableArray<double> w =
                SharedVariable<double>.Random(dim, priorW, divideMessages).Named("w");

            Model model = new Model(numChunks);
            VariableArray<double> x = Variable.Array<double>(dim).Named("x");
            VariableArray<double> ytmp = Variable.Array<double>(dim).Named("ytmp");
            VariableArray<double> wModel = w.GetCopyFor(model);
            ytmp[dim] = wModel[dim]*x[dim];
            Variable<double> y = Variable.Sum(ytmp).Named("y");
            Variable<double> yNoisy = Variable.GaussianFromMeanAndVariance(y, 0.1).Named("yNoisy");

            // does not work with VMP because Sum(array) cannot take a derived argument
            InferenceEngine engine = new InferenceEngine(); //new VariationalMessagePassing());

            for (int pass = 0; pass < 50; pass++)
            {
                for (int c = 0; c < numChunks; c++)
                {
                    x.ObservedValue = inputs[c];
                    yNoisy.ObservedValue = outputs[c];
                    model.InferShared(engine, c);
                }
            }

            GaussianArray actualW = w.Marginal<GaussianArray>();
            Console.WriteLine(" w marginal = {0} (should be {1})", actualW, expectedW);
            Assert.True(expectedW.MaxDiff(actualW) < 1e-4);
        }

        [Fact]
        public void SharedVariable_LearningGaussian()
        {
            sharedVariable_LearningGaussian(true);
            sharedVariable_LearningGaussian(false);
        }

        private void sharedVariable_LearningGaussian(bool divideMessages)
        {
            double[][] dataSets = new double[][] {new double[] {5, 5.1, 5.2, 4.9, -5.1, -5.2, -5.3, -4.9}};
            Gaussian expectedMean = new Gaussian(-0.01017, 0.7288);
            Gamma expectedPrec = new Gamma(5, 0.009301);
            double expectedEvidence = nonSharedVariable_LearningGaussianModel(dataSets[0], out expectedMean, out expectedPrec);
            sharedVariable_LearningGaussianModel(dataSets, expectedMean, expectedPrec, expectedEvidence, divideMessages);
            sharedVariable_LearningGaussianModel2(dataSets, expectedMean, expectedPrec, expectedEvidence, divideMessages);

            dataSets = new double[][] {new double[] {5, 5.1, 5.2, 4.9}, new double[] {-5.1, -5.2, -5.3, -4.9}};
            sharedVariable_LearningGaussianModel(dataSets, expectedMean, expectedPrec, expectedEvidence, divideMessages);
            sharedVariable_LearningGaussianModel2(dataSets, expectedMean, expectedPrec, expectedEvidence, divideMessages);
            dataSets = new double[][] {new double[] {5, 5.1}, new double[] {5.2, 4.9}, new double[] {-5.1, -5.2}, new double[] {-5.3, -4.9}};
            sharedVariable_LearningGaussianModel(dataSets, expectedMean, expectedPrec, expectedEvidence, divideMessages);
            sharedVariable_LearningGaussianModel2(dataSets, expectedMean, expectedPrec, expectedEvidence, divideMessages);
            dataSets = new double[][]
                {
                    new double[] {5}, new double[] {5.1}, new double[] {5.2}, new double[] {4.9},
                    new double[] {-5.1}, new double[] {-5.2}, new double[] {-5.3}, new double[] {-4.9}
                };
            sharedVariable_LearningGaussianModel(dataSets, expectedMean, expectedPrec, expectedEvidence, divideMessages);
            sharedVariable_LearningGaussianModel2(dataSets, expectedMean, expectedPrec, expectedEvidence, divideMessages);

            dataSets = new double[][] {new double[] {5, 6}};
            expectedMean = new Gaussian(0.7732, 0.8594);
            expectedPrec = new Gamma(2, 0.0409);
            expectedEvidence = nonSharedVariable_LearningGaussianModel(dataSets[0], out expectedMean, out expectedPrec);
            sharedVariable_LearningGaussianModel(dataSets, expectedMean, expectedPrec, expectedEvidence, divideMessages);
            sharedVariable_LearningGaussianModel2(dataSets, expectedMean, expectedPrec, expectedEvidence, divideMessages);

            dataSets = new double[][] {new double[] {5}, new double[] {6}};
            sharedVariable_LearningGaussianModel(dataSets, expectedMean, expectedPrec, expectedEvidence, divideMessages);
            sharedVariable_LearningGaussianModel2(dataSets, expectedMean, expectedPrec, expectedEvidence, divideMessages);
        }

        private double nonSharedVariable_LearningGaussianModel(
            double[] dataSet, out Gaussian expectedMean, out Gamma expectedPrec)
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5);
            IfBlock evidenceBlock = Variable.If(evidence);
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 1).Named("mean");
            Variable<double> prec = Variable.GammaFromShapeAndScale(1, 1).Named("prec");
            Variable<int> dataCount = Variable.New<int>().Named("dataCount");
            Range item = new Range(dataCount).Named("item");
            VariableArray<double> data = Variable.Array<double>(item).Named("data");
            data[item] = Variable.GaussianFromMeanAndPrecision(mean, prec).ForEach(item);
            evidenceBlock.CloseBlock();
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.ShowProgress = false;
            data.ObservedValue = dataSet;
            dataCount.ObservedValue = dataSet.Length;

            expectedMean = engine.Infer<Gaussian>(mean);
            expectedPrec = engine.Infer<Gamma>(prec);
            return engine.Infer<Bernoulli>(evidence).LogOdds;
        }

        private void sharedVariable_LearningGaussianModel(
            double[][] dataSets, Gaussian expectedMean, Gamma expectedPrec, double expectedEvidence, bool divideMessages)
        {
            int numChunks = dataSets.Length;
            Gaussian priorMean = Gaussian.FromMeanAndVariance(0, 1);
            Gamma priorPrec = Gamma.FromShapeAndScale(1, 1);
            //priorMean = Gaussian.PointMass(0);
            //priorPrec = Gamma.PointMass(1);

            SharedVariable<bool> evidence = SharedVariable<bool>.Random(new Bernoulli(0.5), divideMessages);
            evidence.IsEvidenceVariable = true;
            SharedVariable<double> mean = SharedVariable<double>.Random(priorMean, divideMessages).Named("mean");
            SharedVariable<double> prec = SharedVariable<double>.Random(priorPrec, divideMessages).Named("prec");

            Model model = new Model(numChunks);
            IfBlock evidenceBlock = Variable.If(evidence.GetCopyFor(model));
            Variable<int> dataCount = Variable.New<int>().Named("dataCount");
            Range item = new Range(dataCount).Named("item");
            VariableArray<double> data = Variable.Array<double>(item).Named("data");
            data[item] = Variable.GaussianFromMeanAndPrecision(mean.GetCopyFor(model), prec.GetCopyFor(model)).ForEach(item);
            evidenceBlock.CloseBlock();

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.ShowProgress = false;

            for (int pass = 0; pass < 15; pass++)
            {
                for (int c = 0; c < numChunks; c++)
                {
                    data.ObservedValue = dataSets[c];
                    dataCount.ObservedValue = dataSets[c].Length;
                    model.InferShared(engine, c);
                    // another interface would be:  ie.InferShared(model,c);
                }
            }

            Gaussian actualMean = mean.Marginal<Gaussian>();
            Gamma actualPrec = prec.Marginal<Gamma>();
            //double actualEvidence = evidence.Marginal<Bernoulli>().LogOdds + model.GetEvidenceCorrectionForVmp();
            double actualEvidence = Model.GetEvidenceForAll(model);
            bool verbose = false;
            if (verbose)
            {
                Console.WriteLine(" mean marginal = {0} (should be {1})", actualMean, expectedMean);
                Console.WriteLine(" prec marginal = {0} (should be {1})", actualPrec, expectedPrec);
                Console.WriteLine(" evidence = {0} (should be {1})", actualEvidence, expectedEvidence);
            }
            Assert.True(expectedMean.MaxDiff(actualMean) < 1e-4);
            Assert.True(expectedPrec.MaxDiff(actualPrec) < 1e-2);
            Assert.Equal(expectedEvidence, actualEvidence, 1e-4);
        }

        private void sharedVariable_LearningGaussianModel2(
            double[][] dataSets, Gaussian expectedMean, Gamma expectedPrec, double expectedEvidence, bool divideMessages)
        {
            Gaussian priorMean = Gaussian.FromMeanAndVariance(0, 1);
            Gamma priorPrec = Gamma.FromShapeAndScale(1, 1);
            //priorMean = Gaussian.PointMass(0);
            //priorPrec = Gamma.PointMass(1);
            SharedVariable<bool> evidence = SharedVariable<bool>.Random(new Bernoulli(0.5), divideMessages);
            evidence.IsEvidenceVariable = true;
            SharedVariable<double, Gaussian> mean = new SharedVariable<double, Gaussian>(priorMean, divideMessages).Named("mean");
            SharedVariable<double, Gamma> prec = new SharedVariable<double, Gamma>(priorPrec, divideMessages).Named("prec");

            Model[] models = new Model[dataSets.Length];
            Variable<int>[] dataCountForModel = new Variable<int>[dataSets.Length];
            VariableArray<double>[] dataForModel = new VariableArray<double>[dataSets.Length];
            InferenceEngine[] engines = new InferenceEngine[dataSets.Length];
            for (int i = 0; i < models.Length; i++)
            {
                Model model = new Model(1);
                IfBlock evidenceBlock = Variable.If(evidence.GetCopyFor(model));
                Variable<int> dataCount = Variable.New<int>().Named("dataCount");
                Range item = new Range(dataCount).Named("item");
                VariableArray<double> data = Variable.Array<double>(item).Named("data");
                data[item] = Variable.GaussianFromMeanAndPrecision(mean.GetCopyFor(model), prec.GetCopyFor(model)).ForEach(item);
                evidenceBlock.CloseBlock();

                dataForModel[i] = data;
                dataCountForModel[i] = dataCount;
                models[i] = model;
                engines[i] = new InferenceEngine(new VariationalMessagePassing());
                engines[i].ShowProgress = false;
            }

            for (int pass = 0; pass < 200; pass++)
            {
                for (int i = 0; i < models.Length; i++)
                {
                    dataForModel[i].ObservedValue = dataSets[i];
                    dataCountForModel[i].ObservedValue = dataSets[i].Length;
                    models[i].InferShared(engines[i], 0);
                }
            }

            Gaussian actualMean = mean.Marginal();
            Gamma actualPrec = prec.Marginal();
            //double actualEvidence = evidence.Marginal<Bernoulli>().LogOdds;
            double actualEvidence = Model.GetEvidenceForAll(models);
            bool verbose = false;
            if (verbose)
            {
                Console.WriteLine(" mean marginal = {0} (should be {1})", actualMean, expectedMean);
                Console.WriteLine(" prec marginal = {0} (should be {1})", actualPrec, expectedPrec);
                Console.WriteLine(" evidence = {0} (should be {1})", actualEvidence, expectedEvidence);
            }
            Assert.True(expectedMean.MaxDiff(actualMean) < 1e-4);
            Assert.True(expectedPrec.MaxDiff(actualPrec) < 1e-2);
            Assert.Equal(expectedEvidence, actualEvidence, 1e-4);
        }

        [Fact]
        public void SharedVariable_GaussianGamma()
        {
            sharedVariable_GaussianGamma(true);
            sharedVariable_GaussianGamma(false);
        }

        private void sharedVariable_GaussianGamma(bool divideMessages)
        {
            // The data
            double[][] dataSets = new double[][]
                {
                    new double[] {11, 5, 8, 9},
                    new double[] {-1, -3, 2, 3, -5}
                };
            int numChunks = dataSets.Length;
            // The model
            Gaussian priorMean = Gaussian.FromMeanAndVariance(0, 100);
            Gamma priorPrec = Gamma.FromShapeAndScale(1, 1);
            SharedVariable<double> mean = SharedVariable<double>.Random(priorMean, divideMessages);
            SharedVariable<double> precision = SharedVariable<double>.Random(priorPrec, divideMessages);
            Model model = new Model(numChunks);
            Variable<int> dataCount = Variable.New<int>();
            Range item = new Range(dataCount);
            VariableArray<double> data = Variable.Array<double>(item);
            data[item] = Variable.GaussianFromMeanAndPrecision(mean.GetCopyFor(model), precision.GetCopyFor(model)).ForEach(item);


            // Set the inference algorithm
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            for (int pass = 0; pass < 5; pass++)
            {
                // Run the inference on each data set
                for (int c = 0; c < numChunks; c++)
                {
                    dataCount.ObservedValue = dataSets[c].Length;
                    data.ObservedValue = dataSets[c];
                    model.InferShared(engine, c);
                }
            }

            // Retrieve the posterior distributions
            Gaussian marginalMean = mean.Marginal<Gaussian>();
            Gamma marginalPrec = precision.Marginal<Gamma>();
            Console.WriteLine("mean=" + marginalMean);
            Console.WriteLine("prec=" + marginalPrec);
        }

        [Fact]
        public void SharedVariable_Dirichlet()
        {
            sharedVariable_Dirichlet(true);
            sharedVariable_Dirichlet(false);
        }

        private void sharedVariable_Dirichlet(bool divideMessages)
        {
            // The data
            int[][] dataSets = new int[][]
                {
                    new int[] {1, 5, 8, 3},
                    new int[] {7},
                    new int[] {1, 2, 1, 5, 8}
                };
            int maxV = 9;
            Dirichlet phiExpected;
            double evidenceExpected = nonSharedVariable_Dirichlet(dataSets, maxV, out phiExpected);

            int numChunks = dataSets.Length;
            // The model
            SharedVariable<bool> evidence = SharedVariable<bool>.Random(new Bernoulli(0.5), divideMessages).Named("evidence");
            evidence.IsEvidenceVariable = true;
            Dirichlet priorMean = Dirichlet.Uniform(maxV);
            SharedVariable<Vector> phi = SharedVariable<Vector>.Random(priorMean, divideMessages).Named("phi");
            Model model = new Model(numChunks);
            IfBlock block = Variable.If(evidence.GetCopyFor(model));
            Variable<int> dataCount = Variable.New<int>();
            Range itemRange = new Range(dataCount);
            VariableArray<int> data = Variable.Array<int>(itemRange);
            data[itemRange] = Variable.Discrete(phi.GetCopyFor(model)).ForEach(itemRange);
            block.CloseBlock();

            // Set the inference algorithm
            InferenceEngine engine = new InferenceEngine(); //new VariationalMessagePassing());
            for (int pass = 0; pass < 5; pass++)
            {
                // Run the inference on each data set
                for (int c = 0; c < numChunks; c++)
                {
                    dataCount.ObservedValue = dataSets[c].Length;
                    data.ObservedValue = dataSets[c];
                    model.InferShared(engine, c);
                }
            }

            // Retrieve the posterior distributions
            Dirichlet phiActual = phi.Marginal<Dirichlet>();
            Console.WriteLine("phi = {0} (should be {1})", phiActual, phiExpected);
            Assert.True(phiExpected.MaxDiff(phiActual) < 1e-4);
            double evidenceActual = Model.GetEvidenceForAll(model);
            Console.WriteLine("evidence = {0} (should be {1})", evidenceActual, evidenceExpected);
            Assert.Equal(evidenceExpected, evidenceActual, 1e-4);
        }

        private double nonSharedVariable_Dirichlet(int[][] dataSets, int maxV, out Dirichlet phiExpected)
        {
            // The model
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Dirichlet priorMean = Dirichlet.Uniform(maxV);
            Variable<Vector> phi = Variable<Vector>.Random(priorMean);
            Range user = new Range(dataSets.Length);
            VariableArray<int> dataCount = Variable.Array<int>(user);
            dataCount.ObservedValue = Util.ArrayInit(dataSets.Length, i => dataSets[i].Length);
            Range item = new Range(dataCount[user]);
            var data = Variable.Array(Variable.Array<int>(item), user);
            data[user][item] = Variable.Discrete(phi).ForEach(user, item);
            block.CloseBlock();
            data.ObservedValue = dataSets;

            // Set the inference algorithm
            InferenceEngine engine = new InferenceEngine(); //new VariationalMessagePassing());
            phiExpected = engine.Infer<Dirichlet>(phi);
            return engine.Infer<Bernoulli>(evidence).LogOdds;
        }

        [Fact]
        public void SharedVariableArray_GaussianJagged()
        {
            sharedVariableArray_GaussianJagged(true);
            sharedVariableArray_GaussianJagged(false);
        }

        private void sharedVariableArray_GaussianJagged(bool divideMessages)
        {
            double[][][] data = new double[][][]
                {
                    new double[][]
                        {
                            new double[] {11, 5, 8, 9},
                            new double[] {-1, -3, 2, 3}
                        },
                    new double[][]
                        {
                            new double[] {2, 2, 4, 5},
                            new double[] {-10, -30, 20, 3}
                        }
                };
            int numChunks = data.Length;
            Range outer = new Range(data[0].Length).Named("outer");
            Range inner = new Range(data[0][0].Length).Named("inner");
            GaussianArrayArray priorW = new GaussianArrayArray(new GaussianArray(new Gaussian(0, 1), inner.SizeAsInt), outer.SizeAsInt);

            // w is a jagged SharedVariableArray
            var w = SharedVariable<double>.Random(Variable.Array<double>(inner), outer, priorW, divideMessages)
                                          .Named("w");
            Model model = new Model(numChunks);
            var x = Variable.Array<double>(Variable.Array<double>(inner), outer).Named("x");
            x[outer][inner] = Variable.GaussianFromMeanAndPrecision(w.GetCopyFor(model)[outer][inner], 1.0);

            InferenceEngine engine = new InferenceEngine(); //new VariationalMessagePassing());
            for (int pass = 0; pass < 50; pass++)
            {
                for (int c = 0; c < numChunks; c++)
                {
                    x.ObservedValue = data[c];
                    model.InferShared(engine, c);
                }
            }

            GaussianArrayArray actualW = w.Marginal<GaussianArrayArray>();
            Console.WriteLine("actualW = " + actualW);
        }

        [Fact]
        public void SharedVariableArray_Gaussian()
        {
            sharedVariableArray_Gaussian(true);
            sharedVariableArray_Gaussian(false);
        }

        private void sharedVariableArray_Gaussian(bool divideMessages)
        {
            double[][][] data = new double[][][]
                {
                    new double[][]
                        {
                            new double[] {11, 5, 8, 9},
                            new double[] {-1, -3, 2, 3}
                        },
                    new double[][]
                        {
                            new double[] {2, 2, 4, 5},
                            new double[] {-10, -30, 20, 3}
                        }
                };
            int numChunks = data.Length;
            Range outer = new Range(data[0].Length).Named("outer");
            Range inner = new Range(data[0][0].Length).Named("inner");
            GaussianArray priorW = new GaussianArray(new Gaussian(0.0, 1.0), inner.SizeAsInt);
            SharedVariableArray<double> w = SharedVariable<double>.Random(inner, priorW, divideMessages).Named("w");

            Model model = new Model(numChunks);
            var x = Variable.Array<double>(Variable.Array<double>(inner), outer).Named("x");
            var wCopy = w.GetCopyFor(model);
            using (Variable.ForEach(outer))
            {
                using (Variable.ForEach(inner))
                {
                    x[outer][inner] = Variable.GaussianFromMeanAndPrecision(wCopy[inner], 1.0);
                }
            }
            InferenceEngine engine = new InferenceEngine(); //new VariationalMessagePassing());
            for (int pass = 0; pass < 50; pass++)
            {
                for (int c = 0; c < numChunks; c++)
                {
                    x.ObservedValue = data[c];
                    model.InferShared(engine, c);
                }
            }

            GaussianArray actualW = w.Marginal<GaussianArray>();
            Console.WriteLine("actualW = " + actualW);
        }

        [Fact]
        public void SharedVariableArray_GaussianMix()
        {
            sharedVariableArray_GaussianMix(true);
            sharedVariableArray_GaussianMix(false);
        }

        private void sharedVariableArray_GaussianMix(bool divideMessages)
        {
            int cnum = 4;
            Range cRange = new Range(cnum).Named("cRange");
            double[][][] data = new double[][][]
                {
                    new double[][]
                        {
                            new double[] {11, 5, 8, 9},
                            new double[] {-1, -3, 2, 3}
                        },
                    new double[][]
                        {
                            new double[] {2, 2, 4, 5},
                            new double[] {-10, -30, 20, 3}
                        }
                };
            int numChunks = data.Length;
            Range outer = new Range(data[0].Length).Named("outer");
            Range inner = new Range(data[0][0].Length).Named("inner");
            GaussianArray priorW = new GaussianArray(new Gaussian(0.0, 1.0), cnum);
            // gaussians
            SharedVariableArray<double> w = SharedVariable<double>.Random(cRange, priorW, divideMessages).Named("w");
            // observed vars
            var x = Variable.Array<double>(Variable.Array<double>(inner), outer).Named("x");
            // latent vars selecting one of the gaussians
            var z = Variable.Array<int>(Variable.Array<int>(inner), outer).Named("z");
            // mixing distribution
            var theta = Variable.DirichletUniform(cRange).Named("theta");

            Model model = new Model(numChunks);
            var wCopy = w.GetCopyFor(model);
            using (Variable.ForEach(outer))
            {
                using (Variable.ForEach(inner))
                {
                    z[outer][inner] = Variable.Discrete(theta);
                    using (Variable.Switch(z[outer][inner]))
                    {
                        x[outer][inner] = Variable.GaussianFromMeanAndPrecision(wCopy[z[outer][inner]], 1.0);
                    }
                }
            }
            Rand.Restart(0);
            var zInit = Util.ArrayInit(outer.SizeAsInt, i =>
                Util.ArrayInit(inner.SizeAsInt, j =>
                Discrete.PointMass(Rand.Int(cRange.SizeAsInt), cRange.SizeAsInt)));
            z.InitialiseTo(Distribution<int>.Array(zInit));
            bool geForceProper = GateEnterOp<double>.ForceProper;
            try
            {
                GateEnterOp<double>.ForceProper = true;
                InferenceEngine engine = new InferenceEngine();
                engine.ShowProgress = false;
                for (int pass = 0; pass < 50; pass++)
                {
                    for (int c = 0; c < numChunks; c++)
                    {
                        x.ObservedValue = data[c];
                        model.InferShared(engine, c);
                    }
                }

                GaussianArray actualW = w.Marginal<GaussianArray>();
                Console.WriteLine("actualW = " + actualW);
            }
            finally
            {
                GateEnterOp<double>.ForceProper = geForceProper;
            }
        }

        [Fact]
        public void SharedVariableArray_Dirichlet()
        {
            sharedVariableArray_Dirichlet(true);
            sharedVariableArray_Dirichlet(false);
        }

        private void sharedVariableArray_Dirichlet(bool divideMessages)
        {
            int[][][] data = new int[][][]
                {
                    new int[][]
                        {
                            new int[] {1, 5, 5, 8, 3},
                            new int[] {1, 2, 1, 5, 8}
                        },
                    new int[][]
                        {
                            new int[] {5, 3, 3, 3, 3},
                            new int[] {6, 7, 6, 5, 8}
                        }
                };
            int maxV = 9;
            int numChunks = data.Length;
            Range outer = new Range(data[0].Length).Named("outer");
            Range inner = new Range(data[0][0].Length).Named("inner");

            DirichletArray prior = new DirichletArray(new Dirichlet[]
                {
                    Dirichlet.Uniform(maxV),
                    Dirichlet.Uniform(maxV),
                    Dirichlet.Uniform(maxV),
                    Dirichlet.Uniform(maxV),
                    Dirichlet.Uniform(maxV)
                });

            SharedVariableArray<Vector> phi = SharedVariable<Vector>.Random(inner, prior, divideMessages).Named("phi");
            Model model = new Model(numChunks);
            var x = Variable.Array<int>(Variable.Array<int>(inner), outer).Named("x");
            var phiCopy = phi.GetCopyFor(model);
            using (Variable.ForEach(outer))
            {
                using (Variable.ForEach(inner))
                {
                    x[outer][inner] = Variable.Discrete(phiCopy[inner]);
                }
            }

            InferenceEngine engine = new InferenceEngine(); //new VariationalMessagePassing());

            for (int pass = 0; pass < 50; pass++)
            {
                for (int c = 0; c < numChunks; c++)
                {
                    x.ObservedValue = data[c];
                    model.InferShared(engine, c);
                }
            }

            var actualPhi = phi.Marginal<DirichletArray>();
            Console.WriteLine("actualPhi = " + actualPhi);
        }

        private double EstimateGaussianMeanAndPrec(
            InferenceEngine engine, double[] data,
            Gaussian meanPrior, Gamma precPrior,
            out Gaussian meanPost, out Gamma precPost)
        {
            Range n = new Range(data.Length).Named("n");
            bool calculateEvidence = !(engine.Algorithm is GibbsSampling);

            Variable<bool> evidence = null;
            IfBlock ifEvidence = null;

            if (calculateEvidence)
            {
                evidence = Variable.Bernoulli(0.5);
                ifEvidence = Variable.If(evidence);
            }
            var vMeanPrior = Variable.New<Gaussian>().Named("meanPrior");
            var mean = Variable.Random<double, Gaussian>(vMeanPrior).Named("mean");
            var vPrecPrior = Variable.New<Gamma>().Named("precPrior");
            var prec = Variable.Random<double, Gamma>(vPrecPrior).Named("prec");
            var x = Variable.Array<double>(n).Named("x");
            x[n] = Variable.GaussianFromMeanAndPrecision(mean, prec).ForEach(n);
            if (calculateEvidence)
                ifEvidence.CloseBlock();

            x.ObservedValue = data;
            vMeanPrior.ObservedValue = meanPrior;
            vPrecPrior.ObservedValue = precPrior;
            meanPost = engine.Infer<Gaussian>(mean);
            precPost = engine.Infer<Gamma>(prec);
            double result = calculateEvidence ? engine.Infer<Bernoulli>(evidence).LogOdds : 0.0;
            Console.WriteLine("Actual mean marginal posterior = " + meanPost);
            Console.WriteLine("Actual prec marginal posterior = " + precPost);
            Console.WriteLine("Actual evidence = " + result);
            return result;
        }

        private void EstimateGaussianSharedDefinition(
            InferenceEngine engine1, InferenceEngine engine2,
            double[][] data, Gaussian meanPrior, Gamma precPrior,
            double expectedEvidence, Gaussian expectedMean, Gamma expectedPrec, bool divideMessages)
        {
            bool calculateEvidence = !(engine1.Algorithm is GibbsSampling || engine2.Algorithm is GibbsSampling);

            int numChunks = data.Length;
            var chunkSize = Variable.New<int>();
            Range n = new Range(chunkSize).Named("n");

            Model paramModel = new Model(1);
            Model dataModel = new Model(numChunks);
            var sharedMean = SharedVariable<double>.Random(Gaussian.Uniform(), divideMessages).Named("sharedMean");
            var sharedPrec = SharedVariable<double>.Random(Gamma.Uniform(), divideMessages).Named("sharedPrec");
            SharedVariable<bool> sharedEvidence = null;
            IfBlock ifParamEvidence = null;
            IfBlock ifDataEvidence = null;

            if (calculateEvidence)
            {
                sharedEvidence = SharedVariable<bool>.Random(new Bernoulli(0.5), divideMessages).Named("Evidence");
                sharedEvidence.IsEvidenceVariable = true;
                ifParamEvidence = Variable.If(sharedEvidence.GetCopyFor(paramModel));
            }
            var vMeanPrior = Variable.New<Gaussian>();
            var mean = Variable.Random<double, Gaussian>(vMeanPrior);
            var vPrecPrior = Variable.New<Gamma>();
            var prec = Variable.Random<double, Gamma>(vPrecPrior);
            sharedMean.SetDefinitionTo(paramModel, mean);
            sharedPrec.SetDefinitionTo(paramModel, prec);
            if (calculateEvidence)
                ifParamEvidence.CloseBlock();


            if (calculateEvidence)
                ifDataEvidence = Variable.If(sharedEvidence.GetCopyFor(dataModel));
            var x = Variable.Array<double>(n).Named("x");
            x[n] = Variable.GaussianFromMeanAndPrecision(
                sharedMean.GetCopyFor(dataModel), sharedPrec.GetCopyFor(dataModel)).ForEach(n);
            if (calculateEvidence)
                ifDataEvidence.CloseBlock();

            vMeanPrior.ObservedValue = meanPrior;
            vPrecPrior.ObservedValue = precPrior;
            for (int pass = 0; pass < 10; pass++)
            {
                paramModel.InferShared(engine1, 0);
                for (int c = 0; c < numChunks; c++)
                {
                    chunkSize.ObservedValue = data[c].Length;
                    x.ObservedValue = data[c];
                    dataModel.InferShared(engine2, c);
                }
            }

            var actualMean = sharedMean.Marginal<Gaussian>();
            var actualPrec = sharedPrec.Marginal<Gamma>();

            double actualEvidence = calculateEvidence ? Model.GetEvidenceForAll(paramModel, dataModel) : 0.0;
            Console.WriteLine("Actual mean marginal posterior = " + actualMean);
            Console.WriteLine("Actual prec marginal posterior = " + actualPrec);
            if (calculateEvidence)
                Console.WriteLine("Actual evidence = " + actualEvidence);
            Assert.True(expectedMean.MaxDiff(actualMean) < 1e-4);
            Assert.True(expectedPrec.MaxDiff(actualPrec) < 1e-2);
            if (calculateEvidence)
                Assert.Equal(expectedEvidence, actualEvidence, 1e-4);
        }

        private void SharedVariableDefinition(InferenceEngine engine1, InferenceEngine engine2, bool divideMessages)
        {
            engine1.ShowProgress = false;
            engine2.ShowProgress = false;
            double[][] dataSets = new double[][] {new double[] {5, 5.1, 5.2, 4.9, -5.1, -5.2, -5.3, -4.9}};
            Gaussian meanPrior = Gaussian.FromMeanAndPrecision(0, 100);
            Gamma precPrior = Gamma.FromShapeAndRate(.1, .1);
            Gaussian expectedMeanPost;
            Gamma expectedPrecPost;
            Console.WriteLine(String.Format("\nNo shared variables ({0})", engine2.Algorithm.Name));
            var expectedEvidence = EstimateGaussianMeanAndPrec(engine2,
                                                               dataSets[0], meanPrior, precPrior, out expectedMeanPost, out expectedPrecPost);
            Console.WriteLine(String.Format("\n1 chunk ({0}/{1})", engine1.Algorithm.Name, engine2.Algorithm.Name));
            EstimateGaussianSharedDefinition(engine1, engine2, dataSets,
                                             meanPrior, precPrior, expectedEvidence, expectedMeanPost, expectedPrecPost, divideMessages);
            dataSets = new double[][] {new double[] {5, 5.1, 5.2, 4.9}, new double[] {-5.1, -5.2, -5.3, -4.9}};
            Console.WriteLine(String.Format("\n2 chunks ({0}/{1})", engine1.Algorithm.Name, engine2.Algorithm.Name));
            EstimateGaussianSharedDefinition(engine1, engine2, dataSets,
                                             meanPrior, precPrior, expectedEvidence, expectedMeanPost, expectedPrecPost, divideMessages);
        }

        [Fact]
        public void SharedVariableDefinitionVMP()
        {
            SharedVariableDefinition(
                new InferenceEngine(new VariationalMessagePassing()),
                new InferenceEngine(new VariationalMessagePassing()), true);
            SharedVariableDefinition(
                new InferenceEngine(new VariationalMessagePassing()),
                new InferenceEngine(new VariationalMessagePassing()), false);
        }

        [Fact]
        public void SharedVariableDefinitionEP()
        {
            SharedVariableDefinition(new InferenceEngine(), new InferenceEngine(), true);
            SharedVariableDefinition(new InferenceEngine(), new InferenceEngine(), false);
        }

        [Fact]
        public void SharedVariableDefinitionHybridVMPEP()
        {
            SharedVariableDefinition(new InferenceEngine(new VariationalMessagePassing()), new InferenceEngine(), true);
            SharedVariableDefinition(new InferenceEngine(new VariationalMessagePassing()), new InferenceEngine(), false);
        }

        [Fact]
        public void SharedVariableDefinitionHybridEPVMP()
        {
            SharedVariableDefinition(new InferenceEngine(), new InferenceEngine(new VariationalMessagePassing()), true);
            SharedVariableDefinition(new InferenceEngine(), new InferenceEngine(new VariationalMessagePassing()), false);
        }

        [Fact]
        public void SharedVariableDefinitionHybridGibbsEP()
        {
            SharedVariableDefinition(new InferenceEngine(new GibbsSampling()), new InferenceEngine(), true);
            SharedVariableDefinition(new InferenceEngine(new GibbsSampling()), new InferenceEngine(), false);
        }

        [Fact]
        public void SharedVariableDefinitionHybridGibbsVMP()
        {
            SharedVariableDefinition(new InferenceEngine(new GibbsSampling()), new InferenceEngine(new VariationalMessagePassing()), true);
            SharedVariableDefinition(new InferenceEngine(new GibbsSampling()), new InferenceEngine(new VariationalMessagePassing()), false);
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void SharedVariableDefinitionHybridEPGibbs()
        {
            //            sharedVariableDefinition(new InferenceEngine(), new InferenceEngine(new GibbsSampling()), true);
            SharedVariableDefinition(new InferenceEngine(), new InferenceEngine(new GibbsSampling()), false);
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void SharedVariableDefinitionHybridVMPGibbs()
        {
            //            sharedVariableDefinition(new InferenceEngine(new VariationalMessagePassing()), new InferenceEngine(new GibbsSampling()), true);
            SharedVariableDefinition(new InferenceEngine(new VariationalMessagePassing()), new InferenceEngine(new GibbsSampling()), false);
        }

        [Fact]
        public void OutputMessageTest()
        {
            int sizeVocab = 1;
            int numTopics = 1;

            Range W = new Range(sizeVocab).Named("W");
            Range T = new Range(numTopics).Named("T");
            var Theta = Variable<Vector>.DirichletSymmetric(numTopics, 0.125).Named("Theta");
            Theta.SetValueRange(T);
            var Phi = Variable.Array<Vector>(T).Named("Phi");
            Phi.SetValueRange(W);
            var PhiPrior = Variable.Array<Dirichlet>(T).Named("PhiPrior");
            Phi[T] = Variable<Vector>.Random(PhiPrior[T]);
            var Word = Variable.New<int>().Named("Word");
            Variable<int> topic = Variable.Discrete(Theta).Named("topic");
            using (Variable.Switch(topic))
            {
                Word = Variable.Discrete(Phi[topic]);
            }

            var engine = new InferenceEngine(new VariationalMessagePassing());
            engine.Compiler.ShowWarnings = false;
            engine.NumberOfIterations = 1;

            Word.ObservedValue = 0;
            PhiPrior.ObservedValue = Util.ArrayInit(numTopics, d => Dirichlet.Symmetric(sizeVocab, 0.125));
            Phi.AddAttribute(QueryTypes.Marginal);
            Phi.AddAttribute(QueryTypes.MarginalDividedByPrior);
            var PhiOutput = engine.Infer<Dirichlet[]>(Phi, QueryTypes.MarginalDividedByPrior)[0];
            var phiMarg = engine.Infer<Dirichlet[]>(Phi)[0];
            var phiManualOut = new Dirichlet(phiMarg);
            phiManualOut.SetToRatio(phiMarg, PhiPrior.ObservedValue[0], false);

            // Since we check for exact equality here, the numbers in the problem must be exactly representable.
            Assert.Equal(phiManualOut, PhiOutput);
        }
    }
}
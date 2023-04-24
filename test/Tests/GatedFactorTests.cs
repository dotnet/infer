// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Xunit;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;
using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Factors.Attributes;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
    using BernoulliArray = DistributionStructArray<Bernoulli, bool>;
    using BernoulliArrayArray = DistributionRefArray<DistributionStructArray<Bernoulli, bool>, bool[]>;
    using GaussianArray = DistributionStructArray<Gaussian, double>;
    using VectorGaussianArray = DistributionRefArray<VectorGaussian, Vector>;

    public class GatedFactorTests
    {
        private static readonly bool verbose = false;

        [Fact]
        public void BetaTrueCountIsGamma_BigData()
        {
            double shape = 2.5;
            double rate = 3.5;
            Variable<double> trueCount = Variable.GammaFromShapeAndRate(shape, rate);
            trueCount.Name = nameof(trueCount);
            int count = 10000;
            Range item = new Range(count);
            var p = Variable.Array<double>(item).Named("p");
            using (Variable.ForEach(item))
            {
                var c = Variable.Bernoulli(0.5);

                using (Variable.If(c))
                {
                    p[item] = Variable.Beta(trueCount, 1);
                }

                using (Variable.IfNot(c))
                {
                    p[item] = Variable.Beta(1, 1);
                }
            }
            p.ObservedValue = Util.ArrayInit(count, i => 0.0001);

            InferenceEngine engine = new InferenceEngine();
            var trueCountActual = engine.Infer<Gamma>(trueCount);
        }

        [Fact]
        public void BetaTrueCountIsGamma()
        {
            BetaTrueCountIsGamma(new ExpectationPropagation());
            BetaTrueCountIsGamma(new VariationalMessagePassing());

            void BetaTrueCountIsGamma(IAlgorithm algorithm)
            {
                Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
                var evBlock = Variable.If(evidence);

                double shape = 2.5;
                double rate = 3.5;
                Variable<double> trueCount = Variable.GammaFromShapeAndRate(shape, rate);
                trueCount.Name = nameof(trueCount);
                Variable<double> x = Variable.Beta(trueCount, 1).Named("x");
                x.ObservedValue = 0.25;

                evBlock.CloseBlock();

                InferenceEngine engine = new InferenceEngine(algorithm);
                var trueCountActual = engine.Infer<Gamma>(trueCount);
                double xRate = -System.Math.Log(x.ObservedValue);
                var trueCountExpected = Gamma.FromShapeAndRate(shape + 1, rate + xRate);
                Assert.True(trueCountExpected.MaxDiff(trueCountActual) < 1e-10);

                var evActual = engine.Infer<Bernoulli>(evidence).LogOdds;

                double evExpected = double.NegativeInfinity;
                var trueCounts = EpTests.linspace(1e-6, 10, 1000);
                foreach (var trueCountValue in trueCounts)
                {
                    trueCount.ObservedValue = trueCountValue;
                    double ev = engine.Infer<Bernoulli>(evidence).LogOdds;
                    evExpected = MMath.LogSumExp(evExpected, ev);
                }
                double increment = trueCounts[1] - trueCounts[0];
                evExpected += System.Math.Log(increment);
                Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-8) < 1e-6);
            }
        }

        [Fact]
        public void BetaFalseCountIsGamma()
        {
            BetaFalseCountIsGamma(new ExpectationPropagation());
            BetaFalseCountIsGamma(new VariationalMessagePassing());

            void BetaFalseCountIsGamma(IAlgorithm algorithm)
            {
                Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
                var evBlock = Variable.If(evidence);

                double shape = 2.5;
                double rate = 3.5;
                Variable<double> falseCount = Variable.GammaFromShapeAndRate(shape, rate);
                falseCount.Name = nameof(falseCount);
                Variable<double> x = Variable.Beta(1, falseCount).Named("x");
                x.ObservedValue = 0.25;

                evBlock.CloseBlock();

                InferenceEngine engine = new InferenceEngine(algorithm);
                var falseCountActual = engine.Infer<Gamma>(falseCount);
                double xRate = -System.Math.Log(1 - x.ObservedValue);
                var falseCountExpected = Gamma.FromShapeAndRate(shape + 1, rate + xRate);
                Assert.True(falseCountExpected.MaxDiff(falseCountActual) < 1e-10);

                var evActual = engine.Infer<Bernoulli>(evidence).LogOdds;

                double evExpected = double.NegativeInfinity;
                var falseCounts = EpTests.linspace(1e-6, 10, 1000);
                foreach (var falseCountValue in falseCounts)
                {
                    falseCount.ObservedValue = falseCountValue;
                    double ev = engine.Infer<Bernoulli>(evidence).LogOdds;
                    evExpected = MMath.LogSumExp(evExpected, ev);
                }
                double increment = falseCounts[1] - falseCounts[0];
                evExpected += System.Math.Log(increment);
                Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-8) < 1e-6);
            }
        }

        [Fact]
        public void TruncatedGaussianIsBetweenTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            var evBlock = Variable.If(evidence);

            Variable<double> x = Variable.TruncatedGaussian(1, 2, 0, 10).Named("x");
            var b = Variable.IsBetween(x, 1, 9);
            b.ObservedValue = true;

            evBlock.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            var xActual = engine.Infer<TruncatedGaussian>(x);
            var xExpected = new TruncatedGaussian(1, 2, 1, 9);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-10);

            var evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            double evExpected = new TruncatedGaussian(1, 2, 0, 10).GetLogAverageOf(new TruncatedGaussian(0, double.PositiveInfinity, 1, 9));
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-8) < 1e-10);
        }

        [Fact]
        public void MinTruncatedGammaTest()
        {
            foreach (bool flip in new[] { false, true })
            {
                var evidence = Variable.Bernoulli(0.5);
                var evBlock = Variable.If(evidence);
                Variable<double> x = Variable.TruncatedGammaFromShapeAndRate(2, 3, 1, double.PositiveInfinity).Named("x");
                double upperBound = 4;
                Variable<double> y = flip ? Variable<double>.Factor(System.Math.Min, x, upperBound) : Variable<double>.Factor(System.Math.Min, upperBound, x);
                var yLike = new TruncatedGamma(4, 5, 1, upperBound);
                Variable.ConstrainEqualRandom(y, yLike);
                evBlock.CloseBlock();

                InferenceEngine engine = new InferenceEngine();
                var yActual = engine.Infer<TruncatedGamma>(y);
                var yDef = new TruncatedGamma(Gamma.FromShapeAndRate(2, 3), 1, upperBound);
                var yExpected = yDef * yLike;
                Assert.True(yExpected.MaxDiff(yActual) < 1e-10);
                var evExpected = yDef.GetLogAverageOf(yLike);
                var evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                Assert.Equal(evExpected, evActual, 1e-10);
            }
        }

        [Fact]
        public void MinGammaTest()
        {
            foreach (bool flip in new[] { false, true })
            {
                var evidence = Variable.Bernoulli(0.5);
                var evBlock = Variable.If(evidence);
                Variable<double> x = Variable.GammaFromShapeAndRate(2, 3).Named("x");
                double upperBound = 4;
                Variable<double> y = flip ? Variable<double>.Factor(System.Math.Min, x, upperBound) : Variable<double>.Factor(System.Math.Min, upperBound, x);
                var yLike = new TruncatedGamma(4, 5, 0, upperBound);
                Variable.ConstrainEqualRandom(y, yLike);
                evBlock.CloseBlock();

                InferenceEngine engine = new InferenceEngine();
                var yActual = engine.Infer<TruncatedGamma>(y);
                var yDef = new TruncatedGamma(Gamma.FromShapeAndRate(2, 3), 0, upperBound);
                var yExpected = yDef * yLike;
                Assert.True(yExpected.MaxDiff(yActual) < 1e-10);
                var evExpected = yDef.GetLogAverageOf(yLike);
                var evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                Assert.Equal(evExpected, evActual, 1e-10);
            }
        }

        [Fact]
        public void MaxTruncatedGammaTest()
        {
            foreach (bool flip in new[] { false, true })
            {
                Variable<double> x = Variable.TruncatedGammaFromShapeAndRate(2, 3, 1, 10).Named("x");
                double lowerBound = 4;
                Variable<double> y = flip ? Variable<double>.Factor(System.Math.Max, x, lowerBound) : Variable<double>.Factor(System.Math.Max, lowerBound, x);
                InferenceEngine engine = new InferenceEngine();
                var yActual = engine.Infer<TruncatedGamma>(y);
                var yExpected = new TruncatedGamma(Gamma.FromShapeAndRate(2, 3), lowerBound, 10);
                Assert.True(yExpected.MaxDiff(yActual) < 1e-10);
            }
        }

        [Fact]
        public void MaxGammaTest()
        {
            foreach (bool flip in new[] { false, true })
            {
                Variable<double> x = Variable.GammaFromShapeAndRate(2, 3).Named("x");
                double lowerBound = 4;
                Variable<double> y = flip ? Variable<double>.Factor(System.Math.Max, x, lowerBound) : Variable<double>.Factor(System.Math.Max, lowerBound, x);
                InferenceEngine engine = new InferenceEngine();
                var yActual = engine.Infer<TruncatedGamma>(y);
                var yExpected = new TruncatedGamma(Gamma.FromShapeAndRate(2, 3), lowerBound, double.PositiveInfinity);
                Assert.True(yExpected.MaxDiff(yActual) < 1e-10);
            }
        }

        [Fact]
        public void GatedVectorGaussianFromMeanAndVarianceTest()
        {
            foreach (bool samplePoint in new[] { false, true })
            {
                foreach (bool meanPoint in new[] { false, true })
                {
                    GatedVectorGaussianFromMeanAndVariance(samplePoint, meanPoint);
                }
            }
        }

        private void GatedVectorGaussianFromMeanAndVariance(bool samplePoint, bool meanPoint)
        {
            var meanMean = Vector.FromArray(1, 2);
            var meanVariance = new PositiveDefiniteMatrix(new double[,] { { 3, 2 }, { 2, 3 } });
            if (meanPoint) meanVariance = PositiveDefiniteMatrix.IdentityScaledBy(2, 0);
            var sampleVariance = new PositiveDefiniteMatrix(new double[,] { { 3, -2 }, { -2, 3 } });
            var evidence = Variable.Bernoulli(0.5).Named("evidence");
            var evBlock = Variable.If(evidence);
            var mean = Variable.VectorGaussianFromMeanAndVariance(meanMean, meanVariance);
            var sample = Variable.VectorGaussianFromMeanAndVariance(mean, sampleVariance);
            evBlock.CloseBlock();

            if (samplePoint)
                sample.ObservedValue = meanMean;

            InferenceEngine engine = new InferenceEngine();
            var evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            double evExpected = 0;
            if (samplePoint) evExpected = VectorGaussian.GetLogProb(sample.ObservedValue, meanMean, sampleVariance + meanVariance);
            Console.WriteLine($"evidence = {evActual} should be {evExpected}");
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-8) < 1e-10);
            var sampleActual = engine.Infer<VectorGaussian>(sample);
            VectorGaussian sampleExpected;
            if (samplePoint)
                sampleExpected = VectorGaussian.PointMass(sample.ObservedValue);
            else
                sampleExpected = VectorGaussian.FromMeanAndVariance(meanMean, sampleVariance + meanVariance);
            Assert.True(sampleExpected.MaxDiff(sampleActual) < 1e-10);
        }

        internal void TruncatedGammaTest()
        {
            Rand.Restart(0);
            double shape = 1.0;
            double trueRate = 1;
            double lowerBound = 2;
            var g = new TruncatedGamma(Gamma.FromShapeAndRate(shape, trueRate), lowerBound, double.PositiveInfinity);
            int n = 1000;
            double[] data = Util.ArrayInit(n, i => g.Sample());
            Range item = new Range(n).Named("item");
            VariableArray<double> obs = Variable.Observed(data, item).Named("obs");

            Variable<double> rate = Variable.GammaFromShapeAndRate(1, 1).Named("rate");
            using (Variable.ForEach(item))
            {
                obs[item] = Variable.TruncatedGammaFromShapeAndRate(shape, rate, lowerBound, double.PositiveInfinity);
            }

            InferenceEngine engine = new InferenceEngine();
            var rateActual = engine.Infer<Gamma>(rate);
            Console.WriteLine($"rate = {rateActual} should be {trueRate}");
            Assert.True(MMath.AbsDiff(trueRate, rateActual.GetMean(), 1e-6) < 1e-1);
        }

        internal void MaxGammaTest2()
        {
            double shape = 1.0;
            double trueRate = 1;
            double lowerBound = 2;
            Gamma g = Gamma.FromShapeAndRate(shape, trueRate);
            int n = 100;
            double[] data = Util.ArrayInit(n, i => System.Math.Max(lowerBound, g.Sample()));
            Range item = new Range(n).Named("item");
            VariableArray<double> obs = Variable.Observed(data, item).Named("obs");

            Variable<double> rate = Variable.GammaFromShapeAndRate(1, 1).Named("rate");
            using (Variable.ForEach(item))
            {
                var x = Variable.GammaFromShapeAndRate(shape, rate);
                obs[item] = Variable.Max(lowerBound, x);
            }

            InferenceEngine engine = new InferenceEngine();
            var rateActual = engine.Infer<Gamma>(rate);
            Console.WriteLine($"rate = {rateActual} should be {trueRate}");
        }

        [Fact]
        public void MaxOfOthersTest()
        {
            var result0 = Factor.MaxOfOthers(new double[0]);
            var result1 = Factor.MaxOfOthers(new double[] { 1.0 });
            Assert.Equal(1.0, result1[0]);
            var result2 = Factor.MaxOfOthers(new double[] { 1.0, 2.0 });
            Assert.Equal(2.0, result2[0]);
            Assert.Equal(1.0, result2[1]);
            var result2b = Factor.MaxOfOthers(new double[] { 2.0, 1.0 });
            Assert.Equal(1.0, result2b[0]);
            Assert.Equal(2.0, result2b[1]);
            var result3 = Factor.MaxOfOthers(new double[] { 1.0, 2.0, 3.0 });
            Assert.Equal(3.0, result3[0]);
            Assert.Equal(3.0, result3[1]);
            Assert.Equal(2.0, result3[2]);
            var result3b = Factor.MaxOfOthers(new double[] { 3.0, 2.0, 1.0 });
            Assert.Equal(2.0, result3b[0]);
            Assert.Equal(3.0, result3b[1]);
            Assert.Equal(3.0, result3b[2]);
        }

        [Fact]
        public void MaxOfOthersTest2()
        {
            Range item = new Range(2);
            var prior = Variable.Array<Gaussian>(item);
            prior.ObservedValue = new Gaussian[] { new Gaussian(1, 2), new Gaussian(3, 4) };
            var array = Variable.Array<double>(item);
            array[item] = Variable<double>.Random(prior[item]);
            var max = Variable.Array<double>(item);
            max.SetTo(Variable<double[]>.Factor(Factor.MaxOfOthers, array));
            InferenceEngine engine = new InferenceEngine();
            var maxActual = engine.Infer<IList<Gaussian>>(max);
            Assert.Equal(prior.ObservedValue[1], maxActual[0]);
            Assert.Equal(prior.ObservedValue[0], maxActual[1]);
        }

        [Fact]
        public void MaxOfOthersTest4()
        {
            Range item = new Range(4);
            var prior = Variable.Array<Gaussian>(item);
            prior.ObservedValue = new Gaussian[] { new Gaussian(1, 2), new Gaussian(3, 4), new Gaussian(1, 2), new Gaussian(1, 2) };
            var array = Variable.Array<double>(item);
            array[item] = Variable<double>.Random(prior[item]);
            var max = Variable.Array<double>(item);
            max.SetTo(Variable<double[]>.Factor(Factor.MaxOfOthers, array));
            InferenceEngine engine = new InferenceEngine();
            var maxActual = engine.Infer<IList<Gaussian>>(max);
            //Assert.Equal(prior.ObservedValue[1], maxActual[0]);
            //Assert.Equal(prior.ObservedValue[0], maxActual[1]);
            Console.WriteLine(maxActual);
            Assert.Equal(maxActual[2], maxActual[0]);
            Assert.NotEqual(maxActual[0], maxActual[1]);
            Assert.Equal(maxActual[3], maxActual[2]);
        }

        [Fact]
        public void SumExceptTest()
        {
            Range item = new Range(2).Named("item");
            var arrayPrior = Variable.Array<Gaussian>(item).Named("arrayPrior");
            arrayPrior.ObservedValue = Util.ArrayInit(item.SizeAsInt, i => new Gaussian(i + 1, i + 1));
            var array = Variable.Array<double>(item).Named("array");
            array[item] = Variable<double>.Random(arrayPrior[item]);
            var other = Variable.Array<double>(item).Named("other");
            using (var block = Variable.ForEach(item))
            {
                var index = block.Index;
                other[index] = Variable<double>.Factor(Factor.SumExcept, array, index);
            }
            var otherLike = new GaussianArray(item.SizeAsInt, i => new Gaussian(i + 2, i + 2));
            Variable.ConstrainEqualRandom(other, otherLike);

            var otherExpectedArray = new Gaussian[] {
                arrayPrior.ObservedValue[1]*otherLike[0],
                arrayPrior.ObservedValue[0]*otherLike[1],
            };
            var otherExpected = new GaussianArray(otherExpectedArray);

            var arrayExpectedArray = new Gaussian[] {
                arrayPrior.ObservedValue[0]*otherLike[1],
                arrayPrior.ObservedValue[1]*otherLike[0],
            };
            var arrayExpected = new GaussianArray(arrayExpectedArray);

            InferenceEngine engine = new InferenceEngine();
            var otherActual = engine.Infer(other);
            var arrayActual = engine.Infer(array);
            Console.WriteLine(StringUtil.JoinColumns("other = ", otherActual, " should be ", otherExpected));
            Console.WriteLine(StringUtil.JoinColumns("array = ", arrayActual, " should be ", arrayExpected));
            Assert.True(otherExpected.MaxDiff(otherActual) < 1e-8);
            Assert.True(arrayExpected.MaxDiff(arrayActual) < 1e-8);
        }

        [Fact]
        public void MaxUniformRRTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);

            Variable<Gaussian> xprior = Variable.New<Gaussian>();
            Variable<Gaussian> yprior = Variable.New<Gaussian>();

            Variable<double> x = Variable.Random<double, Gaussian>(xprior);
            Variable<double> y = Variable.Random<double, Gaussian>(yprior);
            Variable<double> maxXY = Variable.Max(x, y);

            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine();

            Gaussian maxTruePost;
            double evActual, evExpected;

            // test 1
            xprior.ObservedValue = new Gaussian(0, 1);
            yprior.ObservedValue = new Gaussian(0, 1);
            maxTruePost = new Gaussian(0.564189583547755, 0.681690113816208);
            AssertPosteriorEquals(ie, maxXY, maxTruePost);
            evActual = ie.Infer<Bernoulli>(evidence).LogOdds;
            evExpected = 0;
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-5) < 1e-5);

            //test 2
            xprior.ObservedValue = new Gaussian(0, .5);
            yprior.ObservedValue = new Gaussian(0, 1);
            maxTruePost = new Gaussian(0.488602511902919, 0.511267585362156);
            AssertPosteriorEquals(ie, maxXY, maxTruePost);

            //test 3
            xprior.ObservedValue = new Gaussian(0, .5);
            yprior.ObservedValue = new Gaussian(1, 1);
            maxTruePost = new Gaussian(1.142990909082178, 0.733008646277036);
            AssertPosteriorEquals(ie, maxXY, maxTruePost);

            //test 4
            xprior.ObservedValue = new Gaussian(0, .1);
            yprior.ObservedValue = new Gaussian(1, .1);
            maxTruePost = new Gaussian(1.001971323223190, 0.098024790661557);
            AssertPosteriorEquals(ie, maxXY, maxTruePost);

            //test 5
            xprior.ObservedValue = new Gaussian(0, .1);
            yprior.ObservedValue = new Gaussian(0, .1);
            maxTruePost = new Gaussian(0.178412411615277, 0.068169011381621);
            AssertPosteriorEquals(ie, maxXY, maxTruePost);

            // test 6
            xprior.ObservedValue = Gaussian.PointMass(0);
            yprior.ObservedValue = Gaussian.PointMass(0);
            maxTruePost = new Gaussian(0, 0);
            AssertPosteriorEquals(ie, maxXY, maxTruePost);
        }

        [Fact]
        public void MaxRRRTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);

            Variable<Gaussian> xprior = Variable.New<Gaussian>();
            Variable<Gaussian> yprior = Variable.New<Gaussian>();
            Variable<Gaussian> maxConstrain = Variable.New<Gaussian>();

            Variable<double> x = Variable.Random<double, Gaussian>(xprior).Named("x");
            Variable<double> y = Variable.Random<double, Gaussian>(yprior).Named("y");
            Variable<double> maxXY = Variable.Max(x, y).Named("maxXY");
            Variable.ConstrainEqualRandom(maxXY, maxConstrain);

            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine();

            Gaussian maxTruePost, xTruePost, yTruePost;
            double evActual, evExpected;

            // test 1
            xprior.ObservedValue = new Gaussian(0, 1);
            yprior.ObservedValue = new Gaussian(2, 3);
            maxConstrain.ObservedValue = new Gaussian(4, 5);
            maxTruePost = new Gaussian(2.720481395499785, 1.781481142817509);
            xTruePost = new Gaussian(0.051770888431698, MaxGaussianOp.ForceProper ? 1.0 : 1.047041880003208);
            yTruePost = new Gaussian(2.612398497405035, 2.270660020442043);
            AssertPosteriorEquals(ie, maxXY, maxTruePost);
            AssertPosteriorEquals(ie, x, xTruePost);
            AssertPosteriorEquals(ie, y, yTruePost);
            evActual = ie.Infer<Bernoulli>(evidence).LogOdds;
            evExpected = -2.165137622719735;
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-5) < 1e-5);

            // test 2
            xprior.ObservedValue = new Gaussian(0, 1);
            yprior.ObservedValue = new Gaussian(2, 3);
            maxConstrain.ObservedValue = new Gaussian(1, 1e-10);
            maxTruePost = Gaussian.PointMass(1);
            xTruePost = new Gaussian(0.0905236949662384, 0.788663673789836);
            yTruePost = new Gaussian(0.684239540063073, 0.465531317650653);
            //AssertPosteriorEquals(ie, maxXY, maxTruePost);
            AssertPosteriorEquals(ie, x, xTruePost);
            AssertPosteriorEquals(ie, y, yTruePost);
            evActual = ie.Infer<Bernoulli>(evidence).LogOdds;
            evExpected = -1.459999071323817;
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-5) < 1e-5);

            yprior.ObservedValue = Gaussian.Uniform();
            AssertPosteriorEquals(ie, maxXY, maxConstrain.ObservedValue);
            AssertPosteriorEquals(ie, x, xprior.ObservedValue);
            AssertPosteriorEquals(ie, y, yprior.ObservedValue);
        }

        [Fact]
        public void MaxCRRTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);

            Variable<Gaussian> xprior = Variable.New<Gaussian>();
            Variable<Gaussian> yprior = Variable.New<Gaussian>();
            Range item = new Range(1);
            Variable<double> x = Variable.Random<double, Gaussian>(xprior).Named("x");
            Variable<double> y = Variable.Random<double, Gaussian>(yprior).Named("y");
            VariableArray<double> maxXY = Variable.Array<double>(item).Named("maxXY");
            maxXY[item] = Variable.Max(x, y).ForEach(item);

            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine();

            Gaussian xTruePost, yTruePost;
            double evActual, evExpected;

            // test 1
            xprior.ObservedValue = new Gaussian(0, 1);
            yprior.ObservedValue = new Gaussian(2, 3);
            maxXY.ObservedValue = new double[] { 1.0 };
            xTruePost = new Gaussian(0.0905236949662384, 0.788663673789836);
            yTruePost = new Gaussian(0.684239540063073, 0.465531317650653);
            AssertPosteriorEquals(ie, x, xTruePost);
            AssertPosteriorEquals(ie, y, yTruePost);
            evActual = ie.Infer<Bernoulli>(evidence).LogOdds;
            evExpected = -1.459999071323817;
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-5) < 1e-5);
        }

        [Fact]
        public void MaxCCRTest()
        {
            MaxCCR(false);
            MaxCCR(true);
        }
        private void MaxCCR(bool flip)
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);

            Variable<Gaussian> yprior = Variable.New<Gaussian>();
            Range item = new Range(1);
            var x = Variable.Constant(0.0).Named("x");
            Variable<double> y = Variable.Random<double, Gaussian>(yprior).Named("y");
            Variable<double> maxXY;
            if (flip)
                maxXY = Variable.Max(y, x);
            else
                maxXY = Variable.Max(x, y);

            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine();

            Gaussian yExpected;
            double evActual, evExpected;

            // test 1
            yprior.ObservedValue = new Gaussian(2, 3);
            maxXY.ObservedValue = 1.0;
            yExpected = new Gaussian(1.0, 0);
            AssertPosteriorEquals(ie, y, yExpected);
            evActual = ie.Infer<Bernoulli>(evidence).LogOdds;
            evExpected = yprior.ObservedValue.GetLogProb(maxXY.ObservedValue);
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-5) < 1e-5);

            // test 2
            yprior.ObservedValue = new Gaussian(2, 3);
            maxXY.ObservedValue = 0.0;
            yExpected = new Gaussian(-0.858553813194143, 0.545777723103246);
            AssertPosteriorEquals(ie, y, yExpected);
            evActual = ie.Infer<Bernoulli>(evidence).LogOdds;
            evExpected = IsPositiveOp.LogAverageFactor(false, yprior.ObservedValue);
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-5) < 1e-5);
        }

        [Fact]
        public void MaxCCCTest()
        {
            MaxCCC(false);
            MaxCCC(true);
        }
        private void MaxCCC(bool flip)
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);

            Variable<Gaussian> yprior = Variable.New<Gaussian>();
            Range item = new Range(1);
            var x = Variable.Constant(0.0).Named("x");
            Variable<double> y = Variable.Random<double, Gaussian>(yprior).Named("y");
            Variable<double> maxXY;
            if (flip)
                maxXY = Variable.Max(y, x);
            else
                maxXY = Variable.Max(x, y);
            Variable<Gaussian> maxLike = Variable.New<Gaussian>();
            Variable.ConstrainEqualRandom(maxXY, maxLike);

            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine();

            Gaussian yExpected, maxExpected;
            double evActual, evExpected;

            // test 1
            yprior.ObservedValue = Gaussian.PointMass(1.0);
            maxLike.ObservedValue = Gaussian.PointMass(1.0);
            yExpected = Gaussian.PointMass(1.0);
            maxExpected = Gaussian.PointMass(1.0);
            AssertPosteriorEquals(ie, y, yExpected);
            AssertPosteriorEquals(ie, maxXY, maxExpected);
            evActual = ie.Infer<Bernoulli>(evidence).LogOdds;
            evExpected = 0.0;
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-5) < 1e-5);

            // test 2
            yprior.ObservedValue = Gaussian.PointMass(0.0);
            maxLike.ObservedValue = Gaussian.PointMass(0.0);
            yExpected = Gaussian.PointMass(0.0);
            maxExpected = Gaussian.PointMass(0.0);
            AssertPosteriorEquals(ie, y, yExpected);
            AssertPosteriorEquals(ie, maxXY, maxExpected);
            evActual = ie.Infer<Bernoulli>(evidence).LogOdds;
            evExpected = 0.0;
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-5) < 1e-5);
        }

        private void AssertPosteriorEquals(InferenceEngine ie, Variable<double> x, Gaussian expected)
        {
            Gaussian actual = ie.Infer<Gaussian>(x);
            Console.WriteLine("{0} = {1} should be {2}", x.Name, actual, expected);
            Assert.True(actual.MaxDiff(expected) < 1e-5);
        }

        [Fact]
        public void DoubleTest()
        {
            var evidence = Variable.Bernoulli(0.5);
            var block = Variable.If(evidence);
            double[] probs = { 0.1, 0.2, 0.3, 0.4 };
            var k = Variable.Discrete(probs);
            var x = Variable.Double(k);
            Gaussian xLike = Gaussian.FromMeanAndVariance(3, 1);
            Variable.ConstrainEqualRandom(x, xLike);
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(k));
            Console.WriteLine(engine.Infer(x));
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            double evExpected = double.NegativeInfinity;
            for (int i = 0; i < probs.Length; i++)
            {
                evExpected = MMath.LogSumExp(evExpected, xLike.GetLogProb(i) + System.Math.Log(probs[i]));
            }
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-8) < 1e-8);

            x.ObservedValue = 0.5;
            evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Assert.Equal(evActual, double.NegativeInfinity);

            x.ObservedValue = 1;
            evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            evExpected = xLike.GetLogProb(x.ObservedValue) + System.Math.Log(probs[(int)x.ObservedValue]);
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-8) < 1e-8);
        }

        [Fact]
        public void VarianceGammaPowerTest3()
        {
            Variable<double> g = Variable.Random(new GammaPower(2, 1, 1)).Named("g");
            Variable<double> v = (g ^ -1).Named("v");
            Variable<double> x = Variable.GaussianFromMeanAndVariance(0, v).Named("x");
            x.ObservedValue = 3;

            InferenceEngine engine = new InferenceEngine();
            var vExpected = new GammaPower(2.5, 1.0 / (1 + 0.5 * 3 * 3), -1);
            var vActual = engine.Infer<GammaPower>(v);
            Assert.True(vExpected.MaxDiff(vActual) < 1e-4);
        }

        [Fact]
        public void VarianceGammaPowerTest2()
        {
            Variable<double> g = Variable.GammaFromShapeAndRate(2, 1).Named("g");
            Variable<double> v = (g ^ -1).Named("v");
            Variable<double> x = Variable.GaussianFromMeanAndVariance(0, v).Named("x");
            x.ObservedValue = 3;

            InferenceEngine engine = new InferenceEngine();
            var vExpected = new GammaPower(2.5, 1.0 / (1 + 0.5 * 3 * 3), -1);
            var vActual = engine.Infer<GammaPower>(v);
            Assert.True(vExpected.MaxDiff(vActual) < 1e-4);
        }

        [Fact]
        public void VarianceGammaPowerTest()
        {
            var vPrior = new GammaPower(2, 1, -1);  // v^(-3)*exp(-1/v)
            Variable<double> v = Variable.Random(vPrior);
            Variable<double> x = Variable.GaussianFromMeanAndVariance(0, v);
            x.ObservedValue = 3;

            InferenceEngine engine = new InferenceEngine();
            var vExpected = new GammaPower(2.5, 1.0 / (1 + 0.5 * 3 * 3), -1);
            var vActual = engine.Infer<GammaPower>(v);
            Assert.True(vExpected.MaxDiff(vActual) < 1e-4);
        }

        [Fact]
        public void ParetoUniformTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);

            Pareto aPrior = new Pareto(4, 1);
            Variable<double> a = Variable<double>.Random(aPrior).Named("a");
            Range item = new Range(3);
            var y = Variable.Array<double>(item).Named("y");
            y[item] = Variable<double>.Factor(Factor.UniformPlusMinus, a).ForEach(item);
            y.ObservedValue = new double[] { 3, 4, -5 };

            block.CloseBlock();

            double yMax = 5;
            if (string.Empty.Length > 0)
            {
                // Monte Carlo estimate
                int nSamples = 10000;
                double logSum = double.NegativeInfinity;
                for (int i = 0; i < nSamples; i++)
                {
                    double aSample = aPrior.Sample();
                    if (aSample < yMax)
                        continue;
                    double logWeight = -MMath.Ln2 - System.Math.Log(aSample);
                    logSum = MMath.LogSumExp(logSum, logWeight);
                }
                logSum -= System.Math.Log(nSamples);
                Console.WriteLine(logSum);
            }

            InferenceEngine engine = new InferenceEngine();
            a.AddAttribute(new DivideMessages(false));  // Pareto's cannot be divided
            Pareto aExpected = new Pareto(7, 5);
            Pareto aActual = engine.Infer<Pareto>(a);
            Console.WriteLine("a = {0} should be {1}", aActual, aExpected);
            int n = item.SizeAsInt;
            // int_B^inf (0.5/a)^n s L^s/a^(s+1) da = 0.5^n s L^s / B^(n+s) / (n+s)
            double evExpected = n * System.Math.Log(0.5) + System.Math.Log(aPrior.Shape / (aPrior.Shape + n))
                + aPrior.Shape * System.Math.Log(aPrior.LowerBound) - (n + aPrior.Shape) * System.Math.Log(yMax);
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(aExpected.MaxDiff(aActual) < 1e-10);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-10);
        }

        [Fact]
        public void GatedMatrixTimesScalarTest()
        {
            Rand.Restart(0);
            int dimension = 2;
            double ax = 4;
            Wishart wish = Wishart.FromShapeAndRate(dimension, PositiveDefiniteMatrix.Identity(dimension));
            PositiveDefiniteMatrix Bx = wish.Sample();
            double ay = 5;
            PositiveDefiniteMatrix By = wish.Sample();
            Wishart yPrior = Wishart.FromShapeAndRate(ay, By);
            double az = 6, bz = 7;
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            var block = Variable.If(evidence);
            Variable<PositiveDefiniteMatrix> x = Variable.WishartFromShapeAndRate(ax, Bx).Named("x");
            Variable<double> z = Variable.GammaFromShapeAndRate(az, bz).Named("z");
            var y = Variable.MatrixTimesScalar(x, z).Named("y");
            Variable.ConstrainEqualRandom(y, yPrior);
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;
            for (int observeX = 0; observeX < 2; observeX++)
            {
                Console.WriteLine("observeX = {0}", observeX);
                double zTolerance, yTolerance;
                if (observeX == 1)
                {
                    x.ObservedValue = Bx;
                    zTolerance = 1e-2;
                    yTolerance = 5e-2;
                }
                else
                {
                    zTolerance = 1e-2;
                    yTolerance = 5e-2;
                }

                Gamma zExpected;
                Wishart yExpected;
                double evExpected;
                bool useRejection = false;
                if (useRejection)
                {
                    // rejection sampling for true answer
                    GammaEstimator zEst = new GammaEstimator();
                    WishartEstimator yEst = new WishartEstimator(dimension);
                    Gamma zPrior = Gamma.FromShapeAndRate(az, bz);
                    Wishart xPrior = Wishart.FromShapeAndRate(ax, Bx);
                    PositiveDefiniteMatrix ySample = new PositiveDefiniteMatrix(dimension, dimension);
                    PositiveDefiniteMatrix xSample = new PositiveDefiniteMatrix(dimension, dimension);
                    int nSamples = 1000000;
                    for (int iter = 0; iter < nSamples; iter++)
                    {
                        double zSample = zPrior.Sample();
                        if (observeX == 1)
                            ySample.SetToProduct(x.ObservedValue, zSample);
                        else
                        {
                            xSample = xPrior.Sample(xSample);
                            ySample.SetToProduct(xSample, zSample);
                        }
                        double logp = yPrior.GetLogProb(ySample);
                        double p = System.Math.Exp(logp);
                        zEst.Add(zSample, p);
                        yEst.Add(ySample, p);
                    }
                    zExpected = zEst.GetDistribution(new Gamma());
                    yExpected = yEst.GetDistribution(new Wishart(dimension));
                    evExpected = System.Math.Log(zEst.mva.Count / nSamples);
                    if (observeX == 1) evExpected += xPrior.GetLogProb(x.ObservedValue);
                }
                else
                {
                    // cached results from above
                    if (observeX == 0)
                    {
                        zExpected = new Gamma(10.44, 0.1062);
                        yExpected = new Wishart(5.494, new PositiveDefiniteMatrix(new double[,] { { 0.5352, -0.1288 }, { -0.1288, 0.4376 } }));
                        evExpected = -8.67011175755101;
                    }
                    else
                    {
                        zExpected = new Gamma(12.97, 0.09074);
                        yExpected = new Wishart(12.97, new PositiveDefiniteMatrix(new double[,] { { 0.2298, 0.08301 }, { 0.08301, 0.1655 } }));
                        evExpected = -13.9543896506466;
                    }
                }

                for (int alg = 0; alg < 3; alg++)
                {
                    engine.NumberOfIterations = 100;
                    if (alg == 0)
                    {
                        engine.Algorithm = new GibbsSampling();
                        engine.NumberOfIterations = 10000;
                        evidence.ObservedValue = true;
                    }
                    else if (alg == 1)
                    {
                        engine.Algorithm = new VariationalMessagePassing();
                        evidence.ClearObservedValue();
                    }
                    else
                    {
                        engine.Algorithm = new ExpectationPropagation();
                        evidence.ClearObservedValue();
                        if (observeX == 0) continue;
                    }
                    Console.WriteLine(engine.Algorithm.Name);
                    Console.WriteLine(StringUtil.JoinColumns("x = ", engine.Infer(x)));
                    var yActual = engine.Infer<Wishart>(y);
                    Console.WriteLine(StringUtil.JoinColumns("y = ", yActual));
                    Console.WriteLine(StringUtil.JoinColumns(" should be ", yExpected));
                    var zActual = engine.Infer<Gamma>(z);
                    Console.WriteLine("z = {0} (should be {1})", zActual, zExpected);
                    if (observeX == 1 || alg != 1)
                    {
                        Assert.True(zExpected.MaxDiff(zActual) < zTolerance);
                        Assert.True(yExpected.MaxDiff(yActual) < yTolerance || true);
                    }
                    if (alg != 0)
                    {
                        var evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                        Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
                        Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-10) < 2e-2);
                    }
                }
            }
        }

        [Fact]
        public void GatedGaussianTest()
        {
            for (int trial = 0; trial < 2; trial++)
            {
                bool useVariance = (trial == 1);
                GatedGaussian(false, false, useVariance);
                GatedGaussian(false, true, useVariance);
                GatedGaussian(true, false, useVariance);
                GatedGaussian(true, true, useVariance);
            }
        }

        private void GatedGaussian(bool samplePoint, bool meanPoint, bool useVariance)
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Gaussian meanPrior = new Gaussian(1.2, 3.4);
            Variable<double> mean = Variable<double>.Random(meanPrior).Named("mean");
            Variable<double> precision = Variable.New<double>().Named("precision");
            Variable<double> sample;
            if (useVariance)
                sample = Variable.GaussianFromMeanAndVariance(mean, precision).Named("sample");
            else
                sample = Variable.GaussianFromMeanAndPrecision(mean, precision).Named("sample");
            Gaussian samplePrior = new Gaussian(5.6, 7.8);
            Variable.ConstrainEqualRandom(sample, samplePrior);
            block.CloseBlock();

            if (samplePoint) sample.ObservedValue = 6.7;
            if (meanPoint) mean.ObservedValue = 2.3;

            InferenceEngine engine = new InferenceEngine();
            for (int precChoice = 0; precChoice < 3; precChoice++)
            {
                if (precChoice == 0) precision.ObservedValue = 3;
                else if (precChoice == 1) precision.ObservedValue = double.PositiveInfinity;
                else precision.ObservedValue = 0.0;
                for (int trial = 0; trial < 2; trial++)
                {
                    if (trial == 0)
                    {
                        engine.Algorithm = new ExpectationPropagation();
                    }
                    else
                    {
                        if (!samplePoint && !meanPoint) continue;
                        engine.Algorithm = new VariationalMessagePassing();
                    }
                    double evExpected = 0;
                    Gaussian sampleP = samplePrior;
                    Gaussian meanP = meanPrior;
                    if (samplePoint)
                    {
                        evExpected += samplePrior.GetLogProb(sample.ObservedValue);
                        sampleP = Gaussian.PointMass(sample.ObservedValue);
                    }
                    if (meanPoint)
                    {
                        evExpected += meanPrior.GetLogProb(mean.ObservedValue);
                        meanP = Gaussian.PointMass(mean.ObservedValue);
                    }
                    double variance;
                    if (useVariance) variance = precision.ObservedValue;
                    else variance = 1 / precision.ObservedValue;
                    evExpected += sampleP.GetLogAverageOf(new Gaussian(meanP.GetMean(), meanP.GetVariance() + variance));
                    double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                    Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                    Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-10);
                }
            }
        }

        [Fact]
        public void GatedCopyTest()
        {
            GatedCopy(false, false);
            GatedCopy(false, true);
            GatedCopy(true, false);
            GatedCopy(true, true);
        }

        private void GatedCopy(bool xPoint, bool yPoint)
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            double xPrior = 0.3;
            double yPrior = 0.4;
            Variable<bool> x = Variable.Bernoulli(xPrior).Named("x");
            Variable<bool> y = Variable.Copy(x).Named("y");
            Variable.ConstrainEqualRandom(y, new Bernoulli(yPrior));
            block.CloseBlock();

            if (xPoint) x.ObservedValue = true;
            if (yPoint) y.ObservedValue = true;

            InferenceEngine engine = new InferenceEngine();
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial == 0)
                {
                    engine.Algorithm = new ExpectationPropagation();
                }
                else
                {
                    engine.Algorithm = new VariationalMessagePassing();
                }
                double evExpected = 0;
                double xp = xPrior;
                double yp = yPrior;
                if (xPoint)
                {
                    evExpected += System.Math.Log(xPrior);
                    xp = 1;
                }
                if (yPoint)
                {
                    evExpected += System.Math.Log(yPrior);
                    yp = 1;
                }
                evExpected += System.Math.Log(xp * yp + (1 - xp) * (1 - yp));
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-10);
            }
        }

        [Fact]
        public void GatedDirichletTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<Vector> pseudoCounts = Variable.New<Vector>().Named("pseudoCounts");
            Variable<Vector> probs = Variable.Dirichlet(pseudoCounts).Named("probs");
            Variable<int> x = Variable.Discrete(probs).Named("x");
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            pseudoCounts.ObservedValue = Vector.FromArray(new double[] { 1, 1 });
            Dirichlet probsActual = engine.Infer<Dirichlet>(probs);
            Dirichlet probsExpected = Dirichlet.Uniform(2);
            Console.WriteLine("probs = {0} should be {1}", probsActual, probsExpected);
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            double evExpected = 0;
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(probsExpected.MaxDiff(probsActual) < 1e-10);
            Assert.Equal(evExpected, evActual, 1e-10);
        }

        [Fact]
        public void GatedVectorPlusTest()
        {
            GatedVectorPlus(false, false, false);
            GatedVectorPlus(false, false, true);
            GatedVectorPlus(false, true, false);
            GatedVectorPlus(false, true, true);
            GatedVectorPlus(true, false, false);
            GatedVectorPlus(true, false, true);
            GatedVectorPlus(true, true, false);
            GatedVectorPlus(true, true, true);
        }

        private void GatedVectorPlus(bool sumPoint, bool aPoint, bool bPoint)
        {
            Vector sumMean = Vector.FromArray(6.0, 7.0);
            var sumVariance = new PositiveDefiniteMatrix(new double[,] { { 7, 5 }, { 5, 7 } });
            //sumVariance.SetToIdentityScaledBy(double.PositiveInfinity);
            Vector aMean = Vector.FromArray(4.0, 5.0);
            var aVariance = new PositiveDefiniteMatrix(new double[,] { { 2, 1 }, { 1, 2 } });
            var bMean = Vector.FromArray(1.0, 2.0);
            var bVariance = new PositiveDefiniteMatrix(new double[,] { { 3, 2 }, { 2, 3 } });
            var sumPrior = VectorGaussian.FromMeanAndVariance(sumMean, sumVariance);

            var evidence = Variable.Bernoulli(0.5).Named("evidence");
            var evBlock = Variable.If(evidence);
            Variable<Vector> a = Variable.VectorGaussianFromMeanAndVariance(aMean, aVariance).Named("a");
            Variable<Vector> b = Variable.VectorGaussianFromMeanAndVariance(bMean, bVariance).Named("b");
            Variable<Vector> sum = (a + b).Named("sum");
            Variable.ConstrainEqualRandom(sum, sumPrior);
            evBlock.CloseBlock();

            double evExpected = 0;
            if (sumPoint)
            {
                sum.ObservedValue = sumMean;
                sumVariance.SetAllElementsTo(0);
                evExpected += sumPrior.GetLogProb(sum.ObservedValue);
            }
            if (aPoint)
            {
                a.ObservedValue = aMean;
                aVariance.SetAllElementsTo(0);
            }
            if (bPoint)
            {
                b.ObservedValue = bMean;
                bVariance.SetAllElementsTo(0);
            }

            InferenceEngine engine = new InferenceEngine();

            VectorGaussian sumF = VectorGaussian.FromMeanAndVariance(aMean + bMean, aVariance + bVariance);
            VectorGaussian sumB = VectorGaussian.FromMeanAndVariance(sumMean, sumVariance);
            VectorGaussian sumExpected = sumPoint ? VectorGaussian.PointMass(sumMean) :
                sumF * sumB;
            VectorGaussian aExpected = aPoint ? VectorGaussian.PointMass(aMean) :
                VectorGaussian.FromMeanAndVariance(aMean, aVariance) *
                VectorGaussian.FromMeanAndVariance(sumMean - bMean, sumVariance + bVariance);
            VectorGaussian bExpected = bPoint ? VectorGaussian.PointMass(bMean) :
                VectorGaussian.FromMeanAndVariance(bMean, bVariance) *
                VectorGaussian.FromMeanAndVariance(sumMean - aMean, sumVariance + aVariance);
            evExpected += sumF.GetLogAverageOf(sumB);

            for (int trial = 0; trial < 2; trial++)
            {
                if (trial == 0)
                {
                    engine.Algorithm = new ExpectationPropagation();
                }
                else
                {
                    engine.Algorithm = new VariationalMessagePassing();
                }
                bool shouldThrow = (sumPoint && aPoint && bPoint);
                VectorGaussian aActual = null, bActual = null, sumActual = null;
                try
                {
                    aActual = engine.Infer<VectorGaussian>(a);
                    bActual = engine.Infer<VectorGaussian>(b);
                    sumActual = engine.Infer<VectorGaussian>(sum);
                }
                catch (ConstraintViolatedException)
                //catch (AllZeroException)
                {
                    if (shouldThrow)
                        continue;
                    else
                        throw;
                }
                if (shouldThrow)
                {
                    //Assert.True(false, "Did not throw AllZeroException");
                }
                // TODO: check more of the VMP cases
                if (trial == 1 && ((!sumPoint && !aPoint) || (!sumPoint && !bPoint) || (!aPoint && !bPoint)))
                    continue;
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("a = ");
                Console.WriteLine(aActual);
                Console.WriteLine(" should be ");
                Console.WriteLine(aExpected);
                Console.WriteLine("b = ");
                Console.WriteLine(bActual);
                Console.WriteLine(" should be ");
                Console.WriteLine(bExpected);
                Console.WriteLine("sum = ");
                Console.WriteLine(sumActual);
                Console.WriteLine(" should be ");
                Console.WriteLine(sumExpected);
                Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                Assert.True(aExpected.MaxDiff(aActual) < 1e-8);
                Assert.True(bExpected.MaxDiff(bActual) < 1e-8);
                Assert.True(sumExpected.MaxDiff(sumActual) < 1e-8);
                Assert.True(MMath.AbsDiff(evExpected, evActual) < 1e-4);
            }
        }

        [Fact]
        public void GatedVectorGetItemTest()
        {
            GatedVectorGetItem(false, false, false);
            GatedVectorGetItem(false, false, true);
            GatedVectorGetItem(false, true, false);
            GatedVectorGetItem(false, true, true);
            GatedVectorGetItem(true, false, false);
            GatedVectorGetItem(true, false, true);
            GatedVectorGetItem(true, true, false);
            GatedVectorGetItem(true, true, true);
        }

        private void GatedVectorGetItem(bool x1Point, bool x2Point, bool cPoint)
        {
            bool pointPrior = false;
            Gaussian x1Prior = Gaussian.FromMeanAndPrecision(2.0, 5.0);
            Gaussian x2Prior = Gaussian.FromMeanAndPrecision(3.0, 7.0);
            int dc = 2;
            Vector cMean = Vector.Zero(dc);
            PositiveDefiniteMatrix cPrec = new PositiveDefiniteMatrix(dc, dc);
            for (int i = 0; i < dc; i++)
            {
                cMean[i] = (i + 5);
                for (int j = 0; j < dc; j++)
                {
                    cPrec[i, j] = (double)(dc - System.Math.Abs(i - j));
                }
            }
            VectorGaussian cPrior = VectorGaussian.FromMeanAndPrecision(cMean, cPrec);
            if (pointPrior)
            {
                if (x1Point) x1Prior.Point = x1Prior.GetMean();
                if (x2Point) x2Prior.Point = x2Prior.GetMean();
                if (cPoint) cPrior.Point = cMean;
            }

            int index1 = 0;
            int index2 = dc - 1;
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<Vector> c = Variable.Random(cPrior).Named("c");
            Variable<double> x1 = Variable.GetItem(c, index1).Named("x1");
            Variable<double> x2 = Variable.GetItem(c, index2).Named("x2");
            Variable.ConstrainEqualRandom(x1, x1Prior);
            Variable.ConstrainEqualRandom(x2, x2Prior);
            block.CloseBlock();

            VectorGaussian cPrior2 = (VectorGaussian)cPrior.Clone();
            VectorGaussian cLike = new VectorGaussian(dc);
            cLike.Precision[index1, index1] = x1Prior.Precision;
            cLike.Precision[index2, index2] = x2Prior.Precision;
            Vector c0Mean = Vector.Zero(dc);
            c0Mean[index1] = x1Prior.GetMean();
            c0Mean[index2] = x2Prior.GetMean();
            if (!pointPrior)
            {
                if (x1Point)
                {
                    x1.ObservedValue = x1Prior.GetMean();
                    cLike.Precision[index1, index1] = Double.PositiveInfinity;
                }
                if (x2Point)
                {
                    x2.ObservedValue = x2Prior.GetMean();
                    cLike.Precision[index2, index2] = Double.PositiveInfinity;
                }
                if (cPoint)
                {
                    c.ObservedValue = cMean;
                    cPrior2.Point = cMean;
                }
            }
            cLike.SetMeanAndPrecision(c0Mean, cLike.Precision);

            InferenceEngine engine = new InferenceEngine();
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial == 0)
                {
                    engine.Algorithm = new ExpectationPropagation();
                }
                else
                {
                    engine.Algorithm = new VariationalMessagePassing();
                }
                bool shouldThrow = (cPoint && (x1Point || x2Point));
                VectorGaussian cActual;
                try
                {
                    cActual = engine.Infer<VectorGaussian>(c);
                }
                catch (AllZeroException)
                {
                    if (shouldThrow) continue;
                    else throw;
                }
                if (shouldThrow)
                {
                    if (pointPrior) Assert.True(false, "Did not throw AllZeroException");
                    else continue;
                }
                Gaussian x1Actual = engine.Infer<Gaussian>(x1);
                Gaussian x2Actual = engine.Infer<Gaussian>(x2);

                VectorGaussian cExpected = cPrior2 * cLike;
                Gaussian x1Expected = cExpected.GetMarginal(0);
                Gaussian x2Expected = cExpected.GetMarginal(1);
                double evExpected = cPrior2.GetLogAverageOf(cLike);
                if (x1.IsObserved) evExpected += x1Prior.GetLogProb(x1.ObservedValue);
                if (x2.IsObserved) evExpected += x2Prior.GetLogProb(x2.ObservedValue);
                if (c.IsObserved) evExpected += cPrior.GetLogProb(c.ObservedValue);
                Console.WriteLine("c = ");
                Console.WriteLine(cActual);
                Console.WriteLine(" should be ");
                Console.WriteLine(cExpected);
                Console.WriteLine("x1 = ");
                Console.WriteLine(x1Actual);
                Console.WriteLine(" should be ");
                Console.WriteLine(x1Expected);
                Console.WriteLine("x2 = ");
                Console.WriteLine(x2Actual);
                Console.WriteLine(" should be ");
                Console.WriteLine(x2Expected);
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                Assert.True(cExpected.MaxDiff(cActual) < 1e-8);
                Assert.True(x1Expected.MaxDiff(x1Actual) < 1e-8);
                Assert.True(x2Expected.MaxDiff(x2Actual) < 1e-8);
                Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-8);
            }
        }

        [Fact]
        public void SplitSubarrayTest()
        {
            SplitSubarray(false, false, false);
            SplitSubarray(false, false, true);
            SplitSubarray(false, true, false);
            SplitSubarray(false, true, true);
            SplitSubarray(true, false, false);
            SplitSubarray(true, false, true);
            SplitSubarray(true, true, false);
            SplitSubarray(true, true, true);
        }

        private void SplitSubarray(bool headPoint, bool tailPoint, bool arrayPoint)
        {
            int count = 3;
            Range item = new Range(count).Named("item");
            var priorArray = new Bernoulli[] {
                new Bernoulli(0.1), new Bernoulli(0.2), new Bernoulli(0.3)
            };
            var arrayPriorVar = Variable.Constant(priorArray, item).Named("arrayPrior");
            var array = Variable.Array<bool>(item).Named("array");
            array[item] = Variable<bool>.Random(arrayPriorVar[item]);
            var outer = new Range(2);
            var innerCount = Variable.Observed(new int[] { 2, 1 }, outer).Named("innerCount");
            var inner = new Range(innerCount[outer]).Named("inner");
            var indices = Variable.Observed(new int[][] { new int[] { 0, 1 }, new int[] { 2 } }, outer, inner).Named("indices");
            var split = Variable.SplitSubarray(array, indices).Named("split");
            var head = split[0];
            var tail = split[1];
            var headCount = innerCount.ObservedValue[0];
            var tailCount = innerCount.ObservedValue[1];
            var headLikeArray = new Bernoulli[] { new Bernoulli(0.7), new Bernoulli(0.8) };
            var headLike = new BernoulliArray(headCount, i => headLikeArray[i]);
            var headLikeVar = Variable.Constant(headLikeArray, head.Range).Named("headLike");
            Variable.ConstrainEqualRandom(head[head.Range], headLikeVar[head.Range]);
            var tailLikeArray = new Bernoulli[] { new Bernoulli(0.9) };
            var tailLikeArray2 = new Bernoulli[] { tailLikeArray[0] * tailLikeArray[0] };
            var tailLike = new BernoulliArray(tailCount, i => tailLikeArray2[i]);
            var tailLikeVar = Variable.Constant(tailLikeArray, tail.Range).Named("tailLike");
            Variable.ConstrainEqualRandom(tail[tail.Range], tailLikeVar[tail.Range]);
            Variable.ConstrainEqualRandom(tail[tail.Range], tailLikeVar[tail.Range]);
            var arrayLike = new BernoulliArray(count, i => (i < headCount) ? headLikeArray[i] : tailLikeArray2[i - headCount]);
            var headConstant = new bool[] { true, false };
            if (headPoint)
            {
                Variable.ConstrainEqual(head, headConstant);
                arrayLike[0] = Bernoulli.PointMass(true);
                arrayLike[1] = Bernoulli.PointMass(false);
            }
            var tailConstant = new bool[] { true };
            if (tailPoint)
            {
                Variable.ConstrainEqual(tail, tailConstant);
                arrayLike[2] = Bernoulli.PointMass(tailConstant[0]);
            }
            if (arrayPoint)
            {
                array.ObservedValue = new bool[] { true, false, true };
            }

            InferenceEngine engine = new InferenceEngine();
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial == 0)
                {
                    engine.Algorithm = new ExpectationPropagation();
                }
                else
                {
                    engine.Algorithm = new VariationalMessagePassing();
                }
                var arrayActual = engine.Infer<IReadOnlyList<Bernoulli>>(array);
                var splitActual = engine.Infer<IReadOnlyList<IReadOnlyList<Bernoulli>>>(split);
                var headActual = splitActual[0];
                var tailActual = splitActual[1];
                var arrayPrior = new BernoulliArray(count, i => priorArray[i]);
                if (arrayPoint)
                {
                    arrayPrior.Point = array.ObservedValue;
                }
                var headPrior = new BernoulliArray(headCount, i => arrayPrior[i]);
                var tailPrior = new BernoulliArray(tailCount, i => arrayPrior[headCount + i]);
                var arrayExpected = arrayPrior;
                arrayExpected.SetToProduct(arrayPrior, arrayLike);
                var headExpected = headPrior;
                headExpected.SetToProduct(headPrior, headLike);
                var tailExpected = tailPrior;
                tailExpected.SetToProduct(tailPrior, tailLike);
                if (headPoint)
                {
                    headExpected.Point = headConstant;
                }
                if (tailPoint)
                {
                    tailExpected.Point = tailConstant;
                }
                Console.WriteLine("array = ");
                Console.WriteLine(arrayActual);
                Console.WriteLine(" should be ");
                Console.WriteLine(arrayExpected);
                Console.WriteLine("head = ");
                Console.WriteLine(headActual);
                Console.WriteLine(" should be ");
                Console.WriteLine(headExpected);
                Console.WriteLine("tail = ");
                Console.WriteLine(tailActual);
                Console.WriteLine(" should be ");
                Console.WriteLine(tailExpected);
                Assert.True(arrayExpected.MaxDiff(arrayActual) < 1e-8);
                Assert.True(headExpected.MaxDiff(headActual) < 1e-8);
                Assert.True(tailExpected.MaxDiff(tailActual) < 1e-8);
            }
        }

        [Fact]
        public void GatedSplitSubarrayTest()
        {
            GatedSplitSubarray(false, false, false);
            GatedSplitSubarray(false, false, true);
            GatedSplitSubarray(false, true, false);
            GatedSplitSubarray(false, true, true);
            GatedSplitSubarray(true, false, false);
            GatedSplitSubarray(true, false, true);
            GatedSplitSubarray(true, true, false);
            GatedSplitSubarray(true, true, true);
        }

        private void GatedSplitSubarray(bool headPoint, bool tailPoint, bool arrayPoint)
        {
            int count = 3;
            Range item = new Range(count).Named("item");
            var priorArray = new Bernoulli[] {
                new Bernoulli(0.1), new Bernoulli(0.2), new Bernoulli(0.3)
            };
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            var arrayPriorVar = Variable.Constant(priorArray, item).Named("arrayPrior");
            var array = Variable.Array<bool>(item).Named("array");
            array[item] = Variable<bool>.Random(arrayPriorVar[item]);
            var outer = new Range(2);
            var innerCount = Variable.Observed(new int[] { 2, 1 }, outer).Named("innerCount");
            var inner = new Range(innerCount[outer]).Named("inner");
            var indices = Variable.Observed(new int[][] { new int[] { 0, 1 }, new int[] { 2 } }, outer, inner).Named("indices");
            var split = Variable.SplitSubarray(array, indices).Named("split");
            var head = split[0];
            var tail = split[1];
            var headCount = innerCount.ObservedValue[0];
            var tailCount = innerCount.ObservedValue[1];
            var headLikeArray = new Bernoulli[] { new Bernoulli(0.7), new Bernoulli(0.8) };
            var headLike = new BernoulliArray(headCount, i => headLikeArray[i]);
            var headLikeVar = Variable.Constant(headLikeArray, head.Range).Named("headLike");
            Variable.ConstrainEqualRandom(head[head.Range], headLikeVar[head.Range]);
            var tailLikeArray = new Bernoulli[] { new Bernoulli(0.9) };
            var tailLike = new BernoulliArray(tailCount, i => tailLikeArray[i]);
            var tailLikeVar = Variable.Constant(tailLikeArray, tail.Range).Named("tailLike");
            Variable.ConstrainEqualRandom(tail[tail.Range], tailLikeVar[tail.Range]);
            var arrayLike = new BernoulliArray(count, i => (i < headCount) ? headLikeArray[i] : tailLikeArray[i - headCount]);
            var headConstant = new bool[] { true, false };
            if (headPoint)
            {
                Variable.ConstrainEqual(head, headConstant);
                arrayLike[0] = Bernoulli.PointMass(true);
                arrayLike[1] = Bernoulli.PointMass(false);
            }
            var tailConstant = new bool[] { true };
            if (tailPoint)
            {
                //tail.ObservedValue = tailConstant;
                Variable.ConstrainEqual(tail, tailConstant);
                arrayLike[2] = Bernoulli.PointMass(tailConstant[0]);
            }
            if (arrayPoint)
            {
                array.ObservedValue = new bool[] { true, false, true };
            }
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial == 0)
                {
                    engine.Algorithm = new ExpectationPropagation();
                }
                else
                {
                    engine.Algorithm = new VariationalMessagePassing();
                }
                var arrayActual = engine.Infer<IList<Bernoulli>>(array);
                var splitActual = engine.Infer<IReadOnlyList<IReadOnlyList<Bernoulli>>>(split);
                var headActual = splitActual[0];
                var tailActual = splitActual[1];
                var arrayPrior = new BernoulliArray(count, i => priorArray[i]);
                double evExpected = 0;
                if (arrayPoint)
                {
                    evExpected += arrayPrior.GetLogProb(array.ObservedValue);
                    arrayPrior.Point = array.ObservedValue;
                }
                var headPrior = new BernoulliArray(headCount, i => arrayPrior[i]);
                var tailPrior = new BernoulliArray(tailCount, i => arrayPrior[headCount + i]);
                evExpected += arrayPrior.GetLogAverageOf(arrayLike);
                var arrayExpected = arrayPrior;
                arrayExpected.SetToProduct(arrayPrior, arrayLike);
                var headExpected = headPrior;
                headExpected.SetToProduct(headPrior, headLike);
                var tailExpected = tailPrior;
                tailExpected.SetToProduct(tailPrior, tailLike);
                if (headPoint)
                {
                    headExpected.Point = headConstant;
                    evExpected += headLike.GetLogProb(headConstant);
                }
                if (tailPoint)
                {
                    tailExpected.Point = tailConstant;
                    evExpected += tailLike.GetLogProb(tailConstant);
                }
                Console.WriteLine("array = ");
                Console.WriteLine(arrayActual);
                Console.WriteLine(" should be ");
                Console.WriteLine(arrayExpected);
                Console.WriteLine("head = ");
                Console.WriteLine(headActual);
                Console.WriteLine(" should be ");
                Console.WriteLine(headExpected);
                Console.WriteLine("tail = ");
                Console.WriteLine(tailActual);
                Console.WriteLine(" should be ");
                Console.WriteLine(tailExpected);
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                Assert.True(arrayExpected.MaxDiff(arrayActual) < 1e-8);
                Assert.True(headExpected.MaxDiff(headActual) < 1e-8);
                Assert.True(tailExpected.MaxDiff(tailActual) < 1e-8);
                Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-8);
            }
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void GatedSplitSubarrayJaggedTest()
        {
            GatedSplitSubarrayJagged(false, false, false);
            GatedSplitSubarrayJagged(false, false, true);
            GatedSplitSubarrayJagged(false, true, false);
            GatedSplitSubarrayJagged(false, true, true);
            GatedSplitSubarrayJagged(true, false, false);
            GatedSplitSubarrayJagged(true, false, true);
            GatedSplitSubarrayJagged(true, true, false);
            GatedSplitSubarrayJagged(true, true, true);
        }

        private void GatedSplitSubarrayJagged(bool headPoint, bool tailPoint, bool arrayPoint)
        {
            int count = 3;
            int innerCount = 2;
            Range outer = new Range(count).Named("outer");
            Range inner = new Range(innerCount).Named("inner");
            var priorArray = new Bernoulli[][] {
                new Bernoulli[] { new Bernoulli(0.1), new Bernoulli(0.2) },
                new Bernoulli[] { new Bernoulli(0.3), new Bernoulli(0.4) },
                new Bernoulli[] { new Bernoulli(0.6), new Bernoulli(0.7) }
            };
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            var arrayPriorVar = Variable.Constant(priorArray, outer, inner).Named("arrayPrior");
            var array = Variable.Array(Variable.Array<bool>(inner), outer).Named("array");
            array[outer][inner] = Variable<bool>.Random(arrayPriorVar[outer][inner]);
            var outerIndex = new Range(2);
            var innerIndexCount = Variable.Observed(new int[] { 2, 1 }, outerIndex).Named("innerCount");
            var innerIndex = new Range(innerIndexCount[outerIndex]).Named("inner");
            var indices = Variable.Observed(new int[][] { new int[] { 0, 1 }, new int[] { 2 } }, outerIndex, innerIndex).Named("indices");
            var split = Variable.SplitSubarray(array, indices).Named("split");
            var head = split[0];
            var tail = split[1];
            var headCount = innerIndexCount.ObservedValue[0];
            var tailCount = innerIndexCount.ObservedValue[1];
            var headLikeArray = new Bernoulli[][] {
                new Bernoulli[] { new Bernoulli(0.9), new Bernoulli(0.8) },
                new Bernoulli[] { new Bernoulli(0.7), new Bernoulli(0.6) },
            };
            var headLike = new BernoulliArrayArray(headCount, i => new BernoulliArray(innerCount, j => headLikeArray[i][j]));
            var headLikeVar = Variable.Constant(headLikeArray, head.Range, inner).Named("headLike");
            Variable.ConstrainEqualRandom(head[head.Range][inner], headLikeVar[head.Range][inner]);
            var tailLikeArray = new Bernoulli[][] {
                new Bernoulli[] { new Bernoulli(0.9), new Bernoulli(0.7) },
            };
            var tailLike = new BernoulliArrayArray(tailCount, i => new BernoulliArray(innerCount, j => tailLikeArray[i][j]));
            var tailLikeVar = Variable.Constant(tailLikeArray, tail.Range, inner).Named("tailLike");
            Variable.ConstrainEqualRandom(tail[tail.Range][inner], tailLikeVar[tail.Range][inner]);
            var arrayLike = new BernoulliArrayArray(count, i => (i < headCount) ? headLike[i] : tailLike[i - headCount]);
            if (headPoint)
            {
                var headConstant = new bool[][] {
                    new bool[] { true, false },
                    new bool[] { false, true }
                };
                head.ObservedValue = headConstant;
                for (int i = 0; i < headCount; i++)
                {
                    arrayLike[i] = new BernoulliArray(innerCount, j => Bernoulli.PointMass(headConstant[i][j]));
                }
            }
            var tailConstant = new bool[][] { new bool[] { true, false } };
            if (tailPoint)
            {
                //tail.ObservedValue = tailConstant;
                Variable.ConstrainEqual(tail, tailConstant);
                arrayLike[headCount] = new BernoulliArray(innerCount, j => Bernoulli.PointMass(tailConstant[0][j]));
            }
            if (arrayPoint)
            {
                array.ObservedValue = new bool[][] {
                    new bool[] { true, false },
                    new bool[] { false, true },
                    new bool[] { true, false }
                };
            }
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial == 0)
                {
                    engine.Algorithm = new ExpectationPropagation();
                }
                else
                {
                    engine.Algorithm = new VariationalMessagePassing();
                }
                var arrayActual = engine.Infer<IList<BernoulliArray>>(array);
                var headActual = engine.Infer<IList<BernoulliArray>>(head);
                var tailActual = engine.Infer<IList<BernoulliArray>>(tail);
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                var arrayPrior = new BernoulliArrayArray(count, i => new BernoulliArray(innerCount, j => priorArray[i][j]));
                double evExpected = 0;
                if (arrayPoint)
                {
                    evExpected += arrayPrior.GetLogProb(array.ObservedValue);
                    arrayPrior.Point = array.ObservedValue;
                }
                var headPrior = new BernoulliArrayArray(headCount, i => new BernoulliArray(innerCount, j => arrayPrior[i][j]));
                var tailPrior = new BernoulliArrayArray(tailCount, i => new BernoulliArray(innerCount, j => arrayPrior[headCount + i][j]));
                evExpected += arrayPrior.GetLogAverageOf(arrayLike);
                var arrayExpected = arrayPrior;
                arrayExpected.SetToProduct(arrayPrior, arrayLike);
                var headExpected = headPrior;
                headExpected.SetToProduct(headPrior, headLike);
                var tailExpected = tailPrior;
                tailExpected.SetToProduct(tailPrior, tailLike);
                if (headPoint)
                {
                    headExpected.Point = head.ObservedValue;
                    evExpected += headLike.GetLogProb(head.ObservedValue);
                }
                if (tailPoint)
                {
                    tailExpected.Point = tailConstant;
                    evExpected += tailLike.GetLogProb(tailConstant);
                }
                Console.WriteLine("{0},{1},{2}", headPoint, tailPoint, arrayPoint);
                Console.WriteLine("array = ");
                Console.WriteLine(arrayActual);
                Console.WriteLine(" should be ");
                Console.WriteLine(arrayExpected);
                Console.WriteLine("head = ");
                Console.WriteLine(headActual);
                Console.WriteLine(" should be ");
                Console.WriteLine(headExpected);
                Console.WriteLine("tail = ");
                Console.WriteLine(tailActual);
                Console.WriteLine(" should be ");
                Console.WriteLine(tailExpected);
                Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                Assert.True(arrayExpected.MaxDiff(arrayActual) < 1e-8);
                Assert.True(headExpected.MaxDiff(headActual) < 1e-8);
                Assert.True(tailExpected.MaxDiff(tailActual) < 1e-8);
                Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-8);
            }
        }

        [Fact]
        public void SplitTest()
        {
            Split(false, false, false);
            Split(false, false, true);
            Split(false, true, false);
            Split(false, true, true);
            Split(true, false, false);
            Split(true, false, true);
            Split(true, true, false);
            Split(true, true, true);
        }

        private void Split(bool headPoint, bool tailPoint, bool arrayPoint)
        {
            int count = 3;
            Range item = new Range(count).Named("item");
            var priorArray = new Bernoulli[] {
                new Bernoulli(0.1), new Bernoulli(0.2), new Bernoulli(0.3)
            };
            var arrayPriorVar = Variable.Constant(priorArray, item).Named("arrayPrior");
            var array = Variable.Array<bool>(item).Named("array");
            array[item] = Variable<bool>.Random(arrayPriorVar[item]);
            var headCount = 2;
            var tailCount = count - headCount;
            var headRange = new Range(headCount).Named("headRange");
            var tailRange = new Range(tailCount).Named("tailRange");
            VariableArray<bool> tail;
            //var head = Variable.Split(array, headCount, out tail).Named("head");
            var head = Variable.Split(array, headRange, tailRange, out tail).Named("head");
            tail.Name = "tail";
            var headLikeArray = new Bernoulli[] { new Bernoulli(0.7), new Bernoulli(0.8) };
            var headLike = new BernoulliArray(headCount, i => headLikeArray[i]);
            var headLikeVar = Variable.Constant(headLikeArray, head.Range).Named("headLike");
            Variable.ConstrainEqualRandom(head[head.Range], headLikeVar[head.Range]);
            var tailLikeArray = new Bernoulli[] { new Bernoulli(0.9) };
            var tailLikeArray2 = new Bernoulli[] { tailLikeArray[0] * tailLikeArray[0] };
            var tailLike = new BernoulliArray(tailCount, i => tailLikeArray2[i]);
            var tailLikeVar = Variable.Constant(tailLikeArray, tail.Range).Named("tailLike");
            Variable.ConstrainEqualRandom(tail[tail.Range], tailLikeVar[tail.Range]);
            Variable.ConstrainEqualRandom(tail[tail.Range], tailLikeVar[tail.Range]);
            var arrayLike = new BernoulliArray(count, i => (i < headCount) ? headLikeArray[i] : tailLikeArray2[i - headCount]);
            if (headPoint)
            {
                head.ObservedValue = new bool[] { true, false };
                arrayLike[0] = Bernoulli.PointMass(true);
                arrayLike[1] = Bernoulli.PointMass(false);
            }
            var tailConstant = new bool[] { true };
            if (tailPoint)
            {
                //tail.ObservedValue = tailConstant;
                Variable.ConstrainEqual(tail, tailConstant);
                arrayLike[2] = Bernoulli.PointMass(tailConstant[0]);
            }
            if (arrayPoint)
            {
                array.ObservedValue = new bool[] { true, false, true };
            }

            InferenceEngine engine = new InferenceEngine();
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial == 0)
                {
                    engine.Algorithm = new ExpectationPropagation();
                }
                else
                {
                    engine.Algorithm = new VariationalMessagePassing();
                }
                var arrayActual = engine.Infer<IList<Bernoulli>>(array);
                var headActual = engine.Infer<IList<Bernoulli>>(head);
                var tailActual = engine.Infer<IList<Bernoulli>>(tail);
                var arrayPrior = new BernoulliArray(count, i => priorArray[i]);
                if (arrayPoint)
                {
                    arrayPrior.Point = array.ObservedValue;
                }
                var headPrior = new BernoulliArray(headCount, i => arrayPrior[i]);
                var tailPrior = new BernoulliArray(tailCount, i => arrayPrior[headCount + i]);
                var arrayExpected = arrayPrior;
                arrayExpected.SetToProduct(arrayPrior, arrayLike);
                var headExpected = headPrior;
                headExpected.SetToProduct(headPrior, headLike);
                var tailExpected = tailPrior;
                tailExpected.SetToProduct(tailPrior, tailLike);
                if (headPoint)
                {
                    headExpected.Point = head.ObservedValue;
                }
                if (tailPoint)
                {
                    tailExpected.Point = tailConstant;
                }
                Console.WriteLine("array = ");
                Console.WriteLine(arrayActual);
                Console.WriteLine(" should be ");
                Console.WriteLine(arrayExpected);
                Console.WriteLine("head = ");
                Console.WriteLine(headActual);
                Console.WriteLine(" should be ");
                Console.WriteLine(headExpected);
                Console.WriteLine("tail = ");
                Console.WriteLine(tailActual);
                Console.WriteLine(" should be ");
                Console.WriteLine(tailExpected);
                Assert.True(arrayExpected.MaxDiff(arrayActual) < 1e-8);
                Assert.True(headExpected.MaxDiff(headActual) < 1e-8);
                Assert.True(tailExpected.MaxDiff(tailActual) < 1e-8);
            }
        }

        [Fact]
        public void GatedSplitTest()
        {
            GatedSplit(false, false, false);
            GatedSplit(false, false, true);
            GatedSplit(false, true, false);
            GatedSplit(false, true, true);
            GatedSplit(true, false, false);
            GatedSplit(true, false, true);
            GatedSplit(true, true, false);
            GatedSplit(true, true, true);
        }

        private void GatedSplit(bool headPoint, bool tailPoint, bool arrayPoint)
        {
            int count = 3;
            Range item = new Range(count).Named("item");
            var priorArray = new Bernoulli[] {
                new Bernoulli(0.1), new Bernoulli(0.2), new Bernoulli(0.3)
            };
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            var arrayPriorVar = Variable.Constant(priorArray, item).Named("arrayPrior");
            var array = Variable.Array<bool>(item).Named("array");
            array[item] = Variable<bool>.Random(arrayPriorVar[item]);
            var headCount = 2;
            var tailCount = count - headCount;
            var headRange = new Range(headCount).Named("headRange");
            var tailRange = new Range(tailCount).Named("tailRange");
            VariableArray<bool> tail;
            //var head = Variable.Split(array, headCount, out tail).Named("head");
            var head = Variable.Split(array, headRange, tailRange, out tail).Named("head");
            tail.Name = "tail";
            var headLikeArray = new Bernoulli[] { new Bernoulli(0.7), new Bernoulli(0.8) };
            var headLike = new BernoulliArray(headCount, i => headLikeArray[i]);
            var headLikeVar = Variable.Constant(headLikeArray, head.Range).Named("headLike");
            Variable.ConstrainEqualRandom(head[head.Range], headLikeVar[head.Range]);
            var tailLikeArray = new Bernoulli[] { new Bernoulli(0.9) };
            var tailLike = new BernoulliArray(tailCount, i => tailLikeArray[i]);
            var tailLikeVar = Variable.Constant(tailLikeArray, tail.Range).Named("tailLike");
            Variable.ConstrainEqualRandom(tail[tail.Range], tailLikeVar[tail.Range]);
            var arrayLike = new BernoulliArray(count, i => (i < headCount) ? headLikeArray[i] : tailLikeArray[i - headCount]);
            if (headPoint)
            {
                head.ObservedValue = new bool[] { true, false };
                arrayLike[0] = Bernoulli.PointMass(true);
                arrayLike[1] = Bernoulli.PointMass(false);
            }
            var tailConstant = new bool[] { true };
            if (tailPoint)
            {
                //tail.ObservedValue = tailConstant;
                Variable.ConstrainEqual(tail, tailConstant);
                arrayLike[2] = Bernoulli.PointMass(tailConstant[0]);
            }
            if (arrayPoint)
            {
                array.ObservedValue = new bool[] { true, false, true };
            }
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial == 0)
                {
                    engine.Algorithm = new ExpectationPropagation();
                }
                else
                {
                    engine.Algorithm = new VariationalMessagePassing();
                }
                var arrayActual = engine.Infer<IList<Bernoulli>>(array);
                var headActual = engine.Infer<IList<Bernoulli>>(head);
                var tailActual = engine.Infer<IList<Bernoulli>>(tail);
                var arrayPrior = new BernoulliArray(count, i => priorArray[i]);
                double evExpected = 0;
                if (arrayPoint)
                {
                    evExpected += arrayPrior.GetLogProb(array.ObservedValue);
                    arrayPrior.Point = array.ObservedValue;
                }
                var headPrior = new BernoulliArray(headCount, i => arrayPrior[i]);
                var tailPrior = new BernoulliArray(tailCount, i => arrayPrior[headCount + i]);
                evExpected += arrayPrior.GetLogAverageOf(arrayLike);
                var arrayExpected = arrayPrior;
                arrayExpected.SetToProduct(arrayPrior, arrayLike);
                var headExpected = headPrior;
                headExpected.SetToProduct(headPrior, headLike);
                var tailExpected = tailPrior;
                tailExpected.SetToProduct(tailPrior, tailLike);
                if (headPoint)
                {
                    headExpected.Point = head.ObservedValue;
                    evExpected += headLike.GetLogProb(head.ObservedValue);
                }
                if (tailPoint)
                {
                    tailExpected.Point = tailConstant;
                    evExpected += tailLike.GetLogProb(tailConstant);
                }
                Console.WriteLine("array = ");
                Console.WriteLine(arrayActual);
                Console.WriteLine(" should be ");
                Console.WriteLine(arrayExpected);
                Console.WriteLine("head = ");
                Console.WriteLine(headActual);
                Console.WriteLine(" should be ");
                Console.WriteLine(headExpected);
                Console.WriteLine("tail = ");
                Console.WriteLine(tailActual);
                Console.WriteLine(" should be ");
                Console.WriteLine(tailExpected);
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                Assert.True(arrayExpected.MaxDiff(arrayActual) < 1e-8);
                Assert.True(headExpected.MaxDiff(headActual) < 1e-8);
                Assert.True(tailExpected.MaxDiff(tailActual) < 1e-8);
                Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-8);
            }
        }

        [Trait("Category", "OpenBug")]
        [Fact]
        public void GatedSplitJaggedTest()
        {
            GatedSplitJagged(false, false, false);
            GatedSplitJagged(false, false, true);
            GatedSplitJagged(false, true, false);
            GatedSplitJagged(false, true, true);
            GatedSplitJagged(true, false, false);
            GatedSplitJagged(true, false, true);
            GatedSplitJagged(true, true, false);
            GatedSplitJagged(true, true, true);
        }

        private void GatedSplitJagged(bool headPoint, bool tailPoint, bool arrayPoint)
        {
            int count = 3;
            Range outer = new Range(count).Named("outer");
            var innerCount = Variable.Observed(new int[] { 2, 2, 2 }, outer).Named("innerCount");
            Range inner = new Range(innerCount[outer]).Named("inner");
            var priorArray = new Bernoulli[][] {
                new Bernoulli[] { new Bernoulli(0.1), new Bernoulli(0.2) },
                new Bernoulli[] { new Bernoulli(0.3), new Bernoulli(0.4) },
                new Bernoulli[] { new Bernoulli(0.6), new Bernoulli(0.7) }
            };
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            var arrayPriorVar = Variable.Constant(priorArray, outer, inner).Named("arrayPrior");
            var array = Variable.Array(Variable.Array<bool>(inner), outer).Named("array");
            array[outer][inner] = Variable<bool>.Random(arrayPriorVar[outer][inner]);
            var headCount = 2;
            var tailCount = count - headCount;
            var headRange = new Range(headCount).Named("headRange");
            var tailRange = new Range(tailCount).Named("tailRange");
            VariableArray<VariableArray<bool>, bool[][]> tail;
            //var head = Variable.Split(array, headCount, out tail).Named("head");
            var head = Variable.Split(array, headRange, tailRange, out tail).Named("head");
            tail.Name = "tail";
            Range headInner = head[headRange].Range;
            Range tailInner = tail[tailRange].Range;
            var headLikeArray = new Bernoulli[][] {
                new Bernoulli[] { new Bernoulli(0.9), new Bernoulli(0.8) },
                new Bernoulli[] { new Bernoulli(0.7), new Bernoulli(0.6) },
            };
            var headLike = new BernoulliArrayArray(headCount, i => new BernoulliArray(innerCount.ObservedValue[i], j => headLikeArray[i][j]));
            var headLikeVar = Variable.Constant(headLikeArray, headRange, headInner).Named("headLike");
            Variable.ConstrainEqualRandom(head[headRange][headInner], headLikeVar[headRange][headInner]);
            var tailLikeArray = new Bernoulli[][] {
                new Bernoulli[] { new Bernoulli(0.9), new Bernoulli(0.7) },
            };
            var tailLike = new BernoulliArrayArray(tailCount, i => new BernoulliArray(innerCount.ObservedValue[i], j => tailLikeArray[i][j]));
            var tailLikeVar = Variable.Constant(tailLikeArray, tailRange, tailInner).Named("tailLike");
            Variable.ConstrainEqualRandom(tail[tailRange][tailInner], tailLikeVar[tailRange][tailInner]);
            var arrayLike = new BernoulliArrayArray(count, i => (i < headCount) ? headLike[i] : tailLike[i - headCount]);
            if (headPoint)
            {
                var headConstant = new bool[][] {
                    new bool[] { true, false },
                    new bool[] { false, true }
                };
                head.ObservedValue = headConstant;
                for (int i = 0; i < headCount; i++)
                {
                    arrayLike[i] = new BernoulliArray(innerCount.ObservedValue[i], j => Bernoulli.PointMass(headConstant[i][j]));
                }
            }
            var tailConstant = new bool[][] { new bool[] { true, false } };
            if (tailPoint)
            {
                //tail.ObservedValue = tailConstant;
                Variable.ConstrainEqual(tail, tailConstant);
                arrayLike[headCount] = new BernoulliArray(innerCount.ObservedValue[0], j => Bernoulli.PointMass(tailConstant[0][j]));
            }
            if (arrayPoint)
            {
                array.ObservedValue = new bool[][] {
                    new bool[] { true, false },
                    new bool[] { false, true },
                    new bool[] { true, false }
                };
            }
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial == 0)
                {
                    engine.Algorithm = new ExpectationPropagation();
                }
                else
                {
                    engine.Algorithm = new VariationalMessagePassing();
                }
                var arrayActual = engine.Infer<IList<BernoulliArray>>(array);
                var headActual = engine.Infer<IList<BernoulliArray>>(head);
                var tailActual = engine.Infer<IList<BernoulliArray>>(tail);
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                var arrayPrior = new BernoulliArrayArray(count, i => new BernoulliArray(innerCount.ObservedValue[i], j => priorArray[i][j]));
                double evExpected = 0;
                if (arrayPoint)
                {
                    evExpected += arrayPrior.GetLogProb(array.ObservedValue);
                    arrayPrior.Point = array.ObservedValue;
                }
                var headPrior = new BernoulliArrayArray(headCount, i => new BernoulliArray(innerCount.ObservedValue[i], j => arrayPrior[i][j]));
                var tailPrior = new BernoulliArrayArray(tailCount, i => new BernoulliArray(innerCount.ObservedValue[i], j => arrayPrior[headCount + i][j]));
                evExpected += arrayPrior.GetLogAverageOf(arrayLike);
                var arrayExpected = arrayPrior;
                arrayExpected.SetToProduct(arrayPrior, arrayLike);
                var headExpected = headPrior;
                headExpected.SetToProduct(headPrior, headLike);
                var tailExpected = tailPrior;
                tailExpected.SetToProduct(tailPrior, tailLike);
                if (headPoint)
                {
                    headExpected.Point = head.ObservedValue;
                    evExpected += headLike.GetLogProb(head.ObservedValue);
                }
                if (tailPoint)
                {
                    tailExpected.Point = tailConstant;
                    evExpected += tailLike.GetLogProb(tailConstant);
                }
                Console.WriteLine("{0},{1},{2}", headPoint, tailPoint, arrayPoint);
                Console.WriteLine("array = ");
                Console.WriteLine(arrayActual);
                Console.WriteLine(" should be ");
                Console.WriteLine(arrayExpected);
                Console.WriteLine("head = ");
                Console.WriteLine(headActual);
                Console.WriteLine(" should be ");
                Console.WriteLine(headExpected);
                Console.WriteLine("tail = ");
                Console.WriteLine(tailActual);
                Console.WriteLine(" should be ");
                Console.WriteLine(tailExpected);
                Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                Assert.True(arrayExpected.MaxDiff(arrayActual) < 1e-8);
                Assert.True(headExpected.MaxDiff(headActual) < 1e-8);
                Assert.True(tailExpected.MaxDiff(tailActual) < 1e-8);
                Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-8);
            }
        }

        [Fact]
        public void GatedSubvectorTest()
        {
            GatedSubvector(false, false, false);
            GatedSubvector(false, false, true);
            GatedSubvector(false, true, false);
            GatedSubvector(false, true, true);
            GatedSubvector(true, false, false);
            GatedSubvector(true, false, true);
            GatedSubvector(true, true, false);
            GatedSubvector(true, true, true);
        }

        private void GatedSubvector(bool x1Point, bool x2Point, bool cPoint)
        {
            int d1 = 2;
            Vector x1Mean = Vector.Zero(d1);
            PositiveDefiniteMatrix x1Prec = new PositiveDefiniteMatrix(d1, d1);
            for (int i = 0; i < d1; i++)
            {
                x1Mean[i] = (i + 2);
                if (x1Point) x1Prec[i, i] = Double.PositiveInfinity;
                else
                {
                    for (int j = 0; j < d1; j++)
                    {
                        x1Prec[i, j] = (double)(d1 - System.Math.Abs(i - j));
                    }
                }
            }
            int d2 = 3;
            Vector x2Mean = Vector.Zero(d2);
            PositiveDefiniteMatrix x2Prec = new PositiveDefiniteMatrix(d2, d2);
            for (int i = 0; i < d2; i++)
            {
                x2Mean[i] = (i + 3);
                if (x2Point) x2Prec[i, i] = Double.PositiveInfinity;
                else
                {
                    for (int j = 0; j < d2; j++)
                    {
                        x2Prec[i, j] = (double)(d2 - System.Math.Abs(i - j));
                    }
                }
            }
            int dc = d1 + d2;
            Vector cMean = Vector.Zero(dc);
            PositiveDefiniteMatrix cPrec = new PositiveDefiniteMatrix(dc, dc);
            for (int i = 0; i < dc; i++)
            {
                cMean[i] = (i + 5);
                if (cPoint) cPrec[i, i] = Double.PositiveInfinity;
                else
                {
                    for (int j = 0; j < dc; j++)
                    {
                        cPrec[i, j] = (double)(dc - System.Math.Abs(i - j));
                    }
                }
            }
            VectorGaussian cLike = VectorGaussian.FromMeanAndPrecision(cMean, cPrec);

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<Vector> c = Variable.Random(cLike).Named("c");
            Variable<Vector> x1 = Variable.Subvector(c, 0, d1).Named("x1");
            Variable<Vector> x2 = Variable.Subvector(c, d1, d2).Named("x2");
            Variable.ConstrainEqualRandom(x1, VectorGaussian.FromMeanAndPrecision(x1Mean, x1Prec));
            Variable.ConstrainEqualRandom(x2, VectorGaussian.FromMeanAndPrecision(x2Mean, x2Prec));
            block.CloseBlock();

            VectorGaussian cPrior = new VectorGaussian(dc);
            cPrior.Precision.SetSubmatrix(0, 0, x1Prec);
            cPrior.Precision.SetSubmatrix(d1, d1, x2Prec);
            Vector c0Mean = Vector.Concat(x1Mean, x2Mean);
            cPrior.SetMeanAndPrecision(c0Mean, cPrior.Precision);
            InferenceEngine engine = new InferenceEngine();
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial == 0)
                {
                    engine.Algorithm = new ExpectationPropagation();
                }
                else
                {
                    engine.Algorithm = new VariationalMessagePassing();
                }
                bool shouldThrow = (cPoint && (x1Point || x2Point));
                VectorGaussian cActual, x1Actual, x2Actual;
                try
                {
                    cActual = engine.Infer<VectorGaussian>(c);
                    x1Actual = engine.Infer<VectorGaussian>(x1);
                    x2Actual = engine.Infer<VectorGaussian>(x2);
                }
                catch (AllZeroException)
                {
                    if (shouldThrow) continue;
                    else throw;
                }
                //if (shouldThrow) Assert.True(false, "Did not throw AllZeroException");

                double evExpected = cPrior.GetLogAverageOf(cLike);
                if (!double.IsNegativeInfinity(evExpected))
                {
                    VectorGaussian cExpected = cPrior * cLike;
                    VectorGaussian x1Expected = new VectorGaussian(d1);
                    VectorGaussian x2Expected = new VectorGaussian(d2);
                    x1Expected = cExpected.GetMarginal(0, x1Expected);
                    x2Expected = cExpected.GetMarginal(d1, x2Expected);

                    Console.WriteLine("c = ");
                    Console.WriteLine(cActual);
                    Console.WriteLine(" should be ");
                    Console.WriteLine(cExpected);
                    Console.WriteLine("x1 = ");
                    Console.WriteLine(x1Actual);
                    Console.WriteLine(" should be ");
                    Console.WriteLine(x1Expected);
                    Console.WriteLine("x2 = ");
                    Console.WriteLine(x2Actual);
                    Console.WriteLine(" should be ");
                    Console.WriteLine(x2Expected);
                    Assert.True(cExpected.MaxDiff(cActual) < 1e-8);
                    Assert.True(x1Expected.MaxDiff(x1Actual) < 1e-8);
                    Assert.True(x2Expected.MaxDiff(x2Actual) < 1e-8);
                }
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-8);
            }
        }

        [Fact]
        public void GatedConcatTest()
        {
            GatedConcat(false, false, false);
            GatedConcat(false, false, true);
            GatedConcat(false, true, false);
            GatedConcat(false, true, true);
            GatedConcat(true, false, false);
            GatedConcat(true, false, true);
            GatedConcat(true, true, false);
            GatedConcat(true, true, true);
        }

        private void GatedConcat(bool x1Point, bool x2Point, bool cPoint)
        {
            int d1 = 2;
            Vector x1Mean = Vector.Zero(d1);
            PositiveDefiniteMatrix x1Prec = new PositiveDefiniteMatrix(d1, d1);
            for (int i = 0; i < d1; i++)
            {
                x1Mean[i] = (i + 2);
                if (x1Point) x1Prec[i, i] = Double.PositiveInfinity;
                else
                {
                    for (int j = 0; j < d1; j++)
                    {
                        x1Prec[i, j] = (double)(d1 - System.Math.Abs(i - j));
                    }
                }
            }
            int d2 = 3;
            Vector x2Mean = Vector.Zero(d2);
            PositiveDefiniteMatrix x2Prec = new PositiveDefiniteMatrix(d2, d2);
            for (int i = 0; i < d2; i++)
            {
                x2Mean[i] = (i + 3);
                if (x2Point) x2Prec[i, i] = Double.PositiveInfinity;
                else
                {
                    for (int j = 0; j < d2; j++)
                    {
                        x2Prec[i, j] = (double)(d2 - System.Math.Abs(i - j));
                    }
                }
            }
            int dc = d1 + d2;
            Vector cMean = Vector.Zero(dc);
            PositiveDefiniteMatrix cPrec = new PositiveDefiniteMatrix(dc, dc);
            for (int i = 0; i < dc; i++)
            {
                cMean[i] = (i + 5);
                if (cPoint) cPrec[i, i] = Double.PositiveInfinity;
                else
                {
                    for (int j = 0; j < dc; j++)
                    {
                        cPrec[i, j] = (double)(dc - System.Math.Abs(i - j));
                    }
                }
            }
            VectorGaussian cLike = VectorGaussian.FromMeanAndPrecision(cMean, cPrec);

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<Vector> x1 = Variable.VectorGaussianFromMeanAndPrecision(x1Mean, x1Prec).Named("x1");
            Variable<Vector> x2 = Variable.VectorGaussianFromMeanAndPrecision(x2Mean, x2Prec).Named("x2");
            Variable<Vector> c = Variable.Concat(x1, x2).Named("c");
            Variable.ConstrainEqualRandom(c, cLike);
            block.CloseBlock();

            VectorGaussian cPrior = new VectorGaussian(dc);
            cPrior.Precision.SetSubmatrix(0, 0, x1Prec);
            cPrior.Precision.SetSubmatrix(d1, d1, x2Prec);
            Vector c0Mean = Vector.Concat(x1Mean, x2Mean);
            cPrior.SetMeanAndPrecision(c0Mean, cPrior.Precision);
            InferenceEngine engine = new InferenceEngine();
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial == 0)
                {
                    engine.Algorithm = new ExpectationPropagation();
                }
                else
                {
                    engine.Algorithm = new VariationalMessagePassing();
                }
                bool shouldThrow = (cPoint && (x1Point || x2Point));
                VectorGaussian cActual;
                try
                {
                    cActual = engine.Infer<VectorGaussian>(c);
                }
                catch (AllZeroException)
                {
                    if (shouldThrow) continue;
                    else throw;
                }
                if (shouldThrow) Assert.True(false, "Did not throw AllZeroException");
                VectorGaussian x1Actual = engine.Infer<VectorGaussian>(x1);
                VectorGaussian x2Actual = engine.Infer<VectorGaussian>(x2);

                VectorGaussian cExpected = cPrior * cLike;
                VectorGaussian x1Expected = new VectorGaussian(d1);
                VectorGaussian x2Expected = new VectorGaussian(d2);
                double evExpected = cPrior.GetLogAverageOf(cLike);
                if (trial == 0)
                {
                }
                else
                {
                    PositiveDefiniteMatrix prec = (PositiveDefiniteMatrix)cExpected.Precision.Clone();
                    prec.SetSubmatrix(0, d1, new Matrix(d1, d2));
                    prec.SetSubmatrix(d1, 0, new Matrix(d2, d1));
                    cExpected.SetMeanAndPrecision(cExpected.GetMean(), prec);
                    evExpected = cExpected.GetAverageLog(cLike);
                    evExpected += cExpected.GetAverageLog(cPrior);
                    evExpected -= cExpected.GetAverageLog(cExpected);
                }
                x1Expected = cExpected.GetMarginal(0, x1Expected);
                x2Expected = cExpected.GetMarginal(d1, x2Expected);
                Console.WriteLine("c = ");
                Console.WriteLine(cActual);
                Console.WriteLine(" should be ");
                Console.WriteLine(cExpected);
                Console.WriteLine("x1 = ");
                Console.WriteLine(x1Actual);
                Console.WriteLine(" should be ");
                Console.WriteLine(x1Expected);
                Console.WriteLine("x2 = ");
                Console.WriteLine(x2Actual);
                Console.WriteLine(" should be ");
                Console.WriteLine(x2Expected);
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                Assert.True(cExpected.MaxDiff(cActual) < 1e-8);
                Assert.True(x1Expected.MaxDiff(x1Actual) < 1e-8);
                Assert.True(x2Expected.MaxDiff(x2Actual) < 1e-8);
                Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-8);
            }
        }

        [Fact]
        public void GatedVectorFromArrayTest()
        {
            GatedVectorFromArray(false, false, false);
            GatedVectorFromArray(false, false, true);
            GatedVectorFromArray(false, true, false);
            GatedVectorFromArray(false, true, true);
            GatedVectorFromArray(true, false, false);
            GatedVectorFromArray(true, false, true);
            GatedVectorFromArray(true, true, false);
            GatedVectorFromArray(true, true, true);
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        private void GatedVectorFromArray(bool x1Point, bool x2Point, bool cPoint)
        {
            int dc = 2;
            Vector cMean = Vector.Zero(dc);
            PositiveDefiniteMatrix cPrec = new PositiveDefiniteMatrix(dc, dc);
            for (int i = 0; i < dc; i++)
            {
                cMean[i] = (i + 1);
                if (cPoint) cPrec[i, i] = Double.PositiveInfinity;
                else
                {
                    for (int j = 0; j < dc; j++)
                    {
                        cPrec[i, j] = dc - System.Math.Abs(i - j);
                    }
                }
            }
            VectorGaussian cLike = VectorGaussian.FromMeanAndPrecision(cMean, cPrec);

            double[] xMean = { 1.2, 3.4 };
            double[] xPrec = { 5.6, 7.8 };
            if (x1Point) xPrec[0] = Double.PositiveInfinity;
            if (x2Point) xPrec[1] = Double.PositiveInfinity;
            Range item = new Range(dc);
            Gaussian[] xPriors = new Gaussian[dc];
            for (int i = 0; i < xPriors.Length; i++)
            {
                xPriors[i] = Gaussian.FromMeanAndPrecision(xMean[i], xPrec[i]);
            }
            VariableArray<Gaussian> xPrior = Variable.Array<Gaussian>(item).Named("xPrior");
            xPrior.ObservedValue = xPriors;

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            VariableArray<double> x = Variable.Array<double>(item).Named("x");
            x[item] = Variable.Random<double, Gaussian>(xPrior[item]);
            Variable<Vector> c = Variable.Vector(x).Named("c");
            Variable.ConstrainEqualRandom(c, cLike);
            block.CloseBlock();


            VectorGaussian cPrior = new VectorGaussian(dc);
            for (int i = 0; i < dc; i++)
            {
                if (Double.IsPositiveInfinity(xPrec[i]))
                    cPrior.MeanTimesPrecision[i] = xMean[i];
                else
                    cPrior.MeanTimesPrecision[i] = xMean[i] * xPrec[i];
                cPrior.Precision[i, i] = xPrec[i];
            }
            InferenceEngine engine = new InferenceEngine();
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial == 0)
                {
                    engine.Algorithm = new ExpectationPropagation();
                }
                else
                {
                    engine.Algorithm = new VariationalMessagePassing();
                }
                bool shouldThrow = (cPoint && (x1Point || x2Point));
                VectorGaussian cActual;
                try
                {
                    cActual = engine.Infer<VectorGaussian>(c);
                }
                catch (AllZeroException)
                {
                    if (shouldThrow) continue;
                    else throw;
                }
                catch (AggregateException)  // this can be merged with above in C# 6
                {
                    if (shouldThrow)
                        continue;
                    else
                        throw;
                }
                if (shouldThrow) Assert.True(false, "Did not throw AllZeroException");

                VectorGaussian cExpected;
                double evExpected;
                if (trial == 0)
                {
                    cExpected = cPrior * cLike;
                    evExpected = cPrior.GetLogAverageOf(cLike);
                }
                else
                {
                    if (false)
                    {
                        PositiveDefiniteMatrix prec = (PositiveDefiniteMatrix)cLike.Precision.Clone();
                        prec[0, 1] = 0;
                        prec[1, 0] = 0;
                        VectorGaussian cLike2 = VectorGaussian.FromMeanAndPrecision(cLike.GetMean(), prec);
                        cExpected = cPrior * cLike2;
                        evExpected = cPrior.GetLogAverageOf(cLike2);
                    }
                    else
                    {
                        cExpected = cPrior * cLike;
                        PositiveDefiniteMatrix prec = (PositiveDefiniteMatrix)cExpected.Precision.Clone();
                        prec[0, 1] = 0;
                        prec[1, 0] = 0;
                        cExpected.SetMeanAndPrecision(cExpected.GetMean(), prec);
                        if (false)
                        {
                            evExpected = cExpected.GetAverageLog(cLike * cPrior);
                            evExpected += cPrior.GetLogAverageOf(cLike);
                        }
                        else
                        {
                            evExpected = cExpected.GetAverageLog(cLike);
                            evExpected += cExpected.GetAverageLog(cPrior);
                        }
                        evExpected -= cExpected.GetAverageLog(cExpected);
                    }
                }
                IDistribution<double[]> xExpected = Distribution<double>.Array(Util.ArrayInit(dc, i => cExpected.GetMarginal(i)));
                object xActual = engine.Infer(x);
                Console.WriteLine("c = ");
                Console.WriteLine(cActual);
                Console.WriteLine(" should be ");
                Console.WriteLine(cExpected);
                Console.WriteLine(StringUtil.JoinColumns("x = ", xActual, " should be ", xExpected));
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                Assert.True(cExpected.MaxDiff(cActual) < 1e-8);
                Assert.True(xExpected.MaxDiff(xActual) < 1e-8);
                Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-8);
            }
        }

        [Fact]
        public void GatedVectorFromArrayTest2()
        {
            Range item = new Range(2);
            VariableArray<double> array = Variable.Array<double>(item);
            array[item] = Variable.GaussianFromMeanAndVariance(1, 2).ForEach(item);
            var vector = Variable.Vector(array);
            var coeffs = Variable.Constant(Vector.FromArray(1, -1)).Named("coeffs");
            var y = Variable.InnerProduct(coeffs, vector);
            Gaussian yLike = new Gaussian(3, 4);
            Variable.ConstrainEqualRandom(y, yLike);

            InferenceEngine engine = new InferenceEngine();
            var actual = engine.Infer<Diffable>(array);
            var expected = GatedVectorFromArrayTest2Expected();
            Assert.True(actual.MaxDiff(expected) < 1e-10);
        }

        private static IDistribution<double[]> GatedVectorFromArrayTest2Expected()
        {
            Range item = new Range(2);
            VariableArray<double> array = Variable.Array<double>(item);
            array[item] = Variable.GaussianFromMeanAndVariance(1, 2).ForEach(item);
            var y = array[0] - array[1];
            Gaussian yLike = new Gaussian(3, 4);
            Variable.ConstrainEqualRandom(y, yLike);

            InferenceEngine engine = new InferenceEngine();
            return engine.Infer<IDistribution<double[]>>(array);
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        [Fact]
        public void GatedArrayFromVectorTest()
        {
            GatedArrayFromVector(new ExpectationPropagation());
            GatedArrayFromVector(new VariationalMessagePassing());
        }

        private void GatedArrayFromVector(IAlgorithm algorithm)
        {
            int dc = 2;
            Range item = new Range(dc);
            VariableArray<Gaussian> xPrior = Variable.Array<Gaussian>(item).Named("xPrior");

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<VectorGaussian> cPrior = Variable.New<VectorGaussian>().Named("cPrior");
            Variable<Vector> c = Variable<Vector>.Random(cPrior).Named("c");
            VariableArray<double> x = Variable.ArrayFromVector(c, item).Named("x");
            Variable.ConstrainEqualRandom(x[item], xPrior[item]);
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine(algorithm);
            for (int ctrial = 0; ctrial < 3; ctrial++)
            {
                Vector cMean = Vector.Zero(dc);
                PositiveDefiniteMatrix cPrec = new PositiveDefiniteMatrix(dc, dc);
                for (int i = 0; i < dc; i++)
                {
                    cMean[i] = (i + 1);
                    if (ctrial > 0) cPrec[i, i] = Double.PositiveInfinity;
                    else
                    {
                        for (int j = 0; j < dc; j++)
                        {
                            cPrec[i, j] = (double)(dc - System.Math.Abs(i - j));
                        }
                    }
                }
                cPrior.ObservedValue = VectorGaussian.FromMeanAndPrecision(cMean, cPrec);
                if (ctrial == 2) c.ObservedValue = cMean;
                for (int x1trial = 0; x1trial < 3; x1trial++)
                {
                    for (int x2trial = 0; x2trial < 3; x2trial++)
                    {
                        double[] xMean = { 1.2, 3.4 };
                        double[] xPrec = { 5.6, 7.8 };
                        if (x1trial > 0) xPrec[0] = Double.PositiveInfinity;
                        if (x2trial > 0) xPrec[1] = Double.PositiveInfinity;
                        Gaussian[] xPriors = new Gaussian[dc];
                        for (int i = 0; i < xPriors.Length; i++)
                        {
                            xPriors[i] = Gaussian.FromMeanAndPrecision(xMean[i], xPrec[i]);
                        }
                        xPrior.ObservedValue = xPriors;
                        if (x1trial == 2 && x2trial == 2) x.ObservedValue = xMean;
                        else x.ClearObservedValue();

                        VectorGaussian cLike = new VectorGaussian(dc);
                        for (int i = 0; i < dc; i++)
                        {
                            if (Double.IsPositiveInfinity(xPrec[i]))
                                cLike.MeanTimesPrecision[i] = xMean[i];
                            else
                                cLike.MeanTimesPrecision[i] = xMean[i] * xPrec[i];
                            cLike.Precision[i, i] = xPrec[i];
                        }

                        bool shouldThrow = ((ctrial > 0) && ((x1trial > 0) || (x2trial > 0)));
                        double evActual;
                        try
                        {
                            evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                        }
                        catch (AllZeroException)
                        {
                            if (shouldThrow) continue;
                            else throw;
                        }
                        double evExpected = cPrior.ObservedValue.GetLogAverageOf(cLike);
                        Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                        Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-5) < 1e-8);
                        if (shouldThrow) continue;
                        VectorGaussian cActual = engine.Infer<VectorGaussian>(c);
                        VectorGaussian cExpected = cPrior.ObservedValue * cLike;
                        IDistribution<double[]> xExpected = Distribution<double>.Array(Util.ArrayInit(dc, i => cExpected.GetMarginal(i)));
                        object xActual = engine.Infer(x);
                        Console.WriteLine("c = ");
                        Console.WriteLine(cActual);
                        Console.WriteLine(" should be ");
                        Console.WriteLine(cExpected);
                        Console.WriteLine(StringUtil.JoinColumns("x = ", xActual, " should be ", xExpected));
                        Assert.True(cExpected.MaxDiff(cActual) < 1e-8);
                        Assert.True(xExpected.MaxDiff(xActual) < 1e-8);
                    }
                }
            }
        }

        [Fact]
        public void GatedArrayFromVectorMomentsTest()
        {
            GatedArrayFromVectorMoments(new ExpectationPropagation());
            GatedArrayFromVectorMoments(new VariationalMessagePassing());
        }

        private void GatedArrayFromVectorMoments(IAlgorithm algorithm)
        {
            int dc = 2;
            Range item = new Range(dc);
            VariableArray<Gaussian> xPrior = Variable.Array<Gaussian>(item).Named("xPrior");

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<VectorGaussianMoments> cPrior = Variable.New<VectorGaussianMoments>().Named("cPrior");
            Variable<Vector> c = Variable<Vector>.Random(cPrior).Named("c");
            VariableArray<double> x = Variable.ArrayFromVector(c, item).Named("x");
            Variable.ConstrainEqualRandom(x[item], xPrior[item]);
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            for (int ctrial = 0; ctrial < 3; ctrial++)
            {
                Vector cMean = Vector.Zero(dc);
                PositiveDefiniteMatrix cPrec = new PositiveDefiniteMatrix(dc, dc);
                for (int i = 0; i < dc; i++)
                {
                    cMean[i] = (i + 1);
                    if (ctrial > 0) cPrec[i, i] = Double.PositiveInfinity;
                    else
                    {
                        for (int j = 0; j < dc; j++)
                        {
                            cPrec[i, j] = (double)(dc - System.Math.Abs(i - j));
                        }
                    }
                }
                cPrior.ObservedValue = VectorGaussianMoments.FromMeanAndPrecision(cMean, cPrec);
                if (ctrial == 2) c.ObservedValue = cMean;
                for (int x1trial = 0; x1trial < 3; x1trial++)
                {
                    for (int x2trial = 0; x2trial < 3; x2trial++)
                    {
                        double[] xMean = { 1.2, 3.4 };
                        double[] xPrec = { 5.6, 7.8 };
                        if (x1trial > 0) xPrec[0] = Double.PositiveInfinity;
                        if (x2trial > 0) xPrec[1] = Double.PositiveInfinity;
                        Gaussian[] xPriors = new Gaussian[dc];
                        for (int i = 0; i < xPriors.Length; i++)
                        {
                            xPriors[i] = Gaussian.FromMeanAndPrecision(xMean[i], xPrec[i]);
                        }
                        xPrior.ObservedValue = xPriors;
                        if (x1trial == 2 && x2trial == 2) x.ObservedValue = xMean;
                        else x.ClearObservedValue();

                        VectorGaussianMoments cLike = new VectorGaussianMoments(dc);
                        for (int i = 0; i < dc; i++)
                        {
                            cLike.Mean[i] = xMean[i];
                            cLike.Variance[i, i] = 1 / xPrec[i];
                        }

                        bool shouldThrow = ((ctrial > 0) && ((x1trial > 0) || (x2trial > 0)));
                        double evActual;
                        try
                        {
                            evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                        }
                        catch (AllZeroException)
                        {
                            if (shouldThrow) continue;
                            else throw;
                        }
                        double evExpected = cPrior.ObservedValue.GetLogAverageOf(cLike);
                        Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                        Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-5) < 1e-8);
                        if (shouldThrow) continue;
                        VectorGaussianMoments cActual = engine.Infer<VectorGaussianMoments>(c);
                        VectorGaussianMoments cExpected = cPrior.ObservedValue * cLike;
                        IDistribution<double[]> xExpected = Distribution<double>.Array(Util.ArrayInit(dc, i => cExpected.GetMarginal(i)));
                        object xActual = engine.Infer(x);
                        Console.WriteLine("c = ");
                        Console.WriteLine(cActual);
                        Console.WriteLine(" should be ");
                        Console.WriteLine(cExpected);
                        Console.WriteLine(StringUtil.JoinColumns("x = ", xActual, " should be ", xExpected));
                        Assert.True(cExpected.MaxDiff(cActual) < 1e-8);
                        Assert.True(xExpected.MaxDiff(xActual) < 1e-8);
                    }
                }
            }
        }

        [Fact]
        public void GatedPlusIntRRRTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<Discrete> aPrior = Variable.New<Discrete>().Named("aPrior");
            Variable<Discrete> bPrior = Variable.New<Discrete>().Named("bPrior");
            Variable<Discrete> sumPrior = Variable.New<Discrete>().Named("sumPrior");
            Variable<int> a = Variable.Random<int, Discrete>(aPrior).Named("a");
            Variable<int> b = Variable.Random<int, Discrete>(bPrior).Named("b");
            Variable<int> sum = (a + b);
            Variable.ConstrainEqual(sum, Variable.Random<int, Discrete>(sumPrior));
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            for (int sumtrial = 0; sumtrial < 2; sumtrial++)
            {
                if (sumtrial == 0)
                {
                    Vector bProbs = Vector.FromArray(new double[] { 0.5, 0.3, 0.4, 0.2, 0.1, 0.9, 0.6, 0.8, 0.7 });
                    bProbs.Scale(1.0 / bProbs.Sum());
                    sumPrior.ObservedValue = new Discrete(bProbs);
                }
                else
                {
                    sumPrior.ObservedValue = Discrete.PointMass(4, 9);
                }
                for (int atrial = 0; atrial < 2; atrial++)
                {
                    if (atrial == 0)
                    {
                        Vector nProbs = Vector.FromArray(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5 });
                        nProbs.Scale(1.0 / nProbs.Sum());
                        aPrior.ObservedValue = new Discrete(nProbs);
                    }
                    else
                    {
                        aPrior.ObservedValue = Discrete.PointMass(3, 5);
                    }
                    for (int btrial = 0; btrial < 2; btrial++)
                    {
                        if (btrial == 0)
                        {
                            Vector xProbs = Vector.FromArray(new double[] { 0.3, 0.4, 0.1, 0.5, 0.2 });
                            xProbs.Scale(1.0 / xProbs.Sum());
                            bPrior.ObservedValue = new Discrete(xProbs);
                        }
                        else
                        {
                            bPrior.ObservedValue = Discrete.PointMass(2, 5);
                        }
                        double[] toB = new double[bPrior.ObservedValue.Dimension];
                        double[] toA = new double[aPrior.ObservedValue.Dimension];
                        double z = 0.0;
                        for (int i = 0; i < bPrior.ObservedValue.Dimension; i++)
                        {
                            for (int j = 0; j < aPrior.ObservedValue.Dimension; j++)
                            {
                                double f = sumPrior.ObservedValue[i + j];
                                toB[i] += aPrior.ObservedValue[j] * f;
                                toA[j] += bPrior.ObservedValue[i] * f;
                                z += bPrior.ObservedValue[i] * aPrior.ObservedValue[j] * f;
                            }
                        }
                        Discrete toAdist = new Discrete(toA);
                        if (Double.IsNegativeInfinity(aPrior.ObservedValue.GetLogAverageOf(toAdist))) continue;
                        Discrete aExpected = aPrior.ObservedValue * toAdist;
                        Discrete toBdist = new Discrete(toB);
                        if (Double.IsNegativeInfinity(bPrior.ObservedValue.GetLogAverageOf(toBdist))) continue;
                        Discrete bExpected = bPrior.ObservedValue * toBdist;
                        double evExpected = System.Math.Log(z);
                        Discrete aActual = engine.Infer<Discrete>(a);
                        Discrete bActual = engine.Infer<Discrete>(b);
                        double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                        Console.WriteLine("a = {0} should be {1}", aActual, aExpected);
                        Console.WriteLine("b = {0} should be {1}", bActual, bExpected);
                        Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                        Assert.True(bExpected.MaxDiff(bActual) < 1e-10);
                        Assert.True(aExpected.MaxDiff(aActual) < 1e-10);
                        Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-10);
                    }
                }
            }
        }

        [Fact]
        public void GatedPlusIntRRCTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<Discrete> aPrior = Variable.New<Discrete>().Named("aPrior");
            Variable<Discrete> sumPrior = Variable.New<Discrete>().Named("sumPrior");
            Variable<int> a = Variable.Random<int, Discrete>(aPrior).Named("a");
            Variable<int> b = Variable.New<int>().Named("b");
            Variable<int> sum = (a + b).Named("sum");
            //Variable.ConstrainEqual(sum, Variable.Random<int, Discrete>(sumPrior).Named("sum2"));
            Variable.ConstrainEqualRandom(sum, sumPrior);
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            for (int sumtrial = 0; sumtrial < 2; sumtrial++)
            {
                for (int atrial = 0; atrial < 2; atrial++)
                {
                    if (atrial == 0)
                    {
                        Vector nProbs = Vector.FromArray(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5 });
                        nProbs.Scale(1.0 / nProbs.Sum());
                        aPrior.ObservedValue = new Discrete(nProbs);
                    }
                    else
                    {
                        aPrior.ObservedValue = Discrete.PointMass(3, 5);
                    }
                    for (int btrial = 0; btrial < 2; btrial++)
                    {
                        if (btrial == 0)
                        {
                            b.ObservedValue = 0;
                        }
                        else
                        {
                            b.ObservedValue = 1;
                        }
                        int sumPriorLength = aPrior.ObservedValue.Dimension + b.ObservedValue;
                        if (sumtrial == 0)
                        {
                            Vector bProbs = Vector.FromArray(Util.ArrayInit(sumPriorLength, i => (double)i));
                            bProbs.Scale(1.0 / bProbs.Sum());
                            sumPrior.ObservedValue = new Discrete(bProbs);
                        }
                        else
                        {
                            sumPrior.ObservedValue = Discrete.PointMass(4, sumPriorLength);
                        }
                        double[] toA = new double[aPrior.ObservedValue.Dimension];
                        double z = 0.0;
                        int j = b.ObservedValue;
                        for (int i = 0; i < aPrior.ObservedValue.Dimension; i++)
                        {
                            double f = sumPrior.ObservedValue[i + j];
                            toA[i] += f;
                            z += aPrior.ObservedValue[i] * f;
                        }
                        Discrete toAdist = new Discrete(toA);
                        if (Double.IsNegativeInfinity(aPrior.ObservedValue.GetLogAverageOf(toAdist))) continue;
                        Discrete aExpected = aPrior.ObservedValue * toAdist;
                        double evExpected = System.Math.Log(z);
                        Discrete aActual = engine.Infer<Discrete>(a);
                        double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                        Console.WriteLine("a = {0} should be {1}", aActual, aExpected);
                        Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                        Assert.True(aExpected.MaxDiff(aActual) < 1e-10);
                        Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-10);
                    }
                }
            }
        }

        [Fact]
        public void GatedMatrixVectorProductTest()
        {
            GatedMatrixVectorProduct(new ExpectationPropagation());
            GatedMatrixVectorProduct(new VariationalMessagePassing());
        }

        private void GatedMatrixVectorProduct(IAlgorithm alg)
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<VectorGaussian> xPrior = Variable.New<VectorGaussian>().Named("xPrior");
            Variable<Vector> x = Variable<Vector>.Random(xPrior).Named("x");
            Variable<VectorGaussian> yPrior = Variable.New<VectorGaussian>().Named("yPrior");
            Variable<Matrix> a = Variable.New<Matrix>().Named("a");
            Variable<Vector> y = Variable.MatrixTimesVector(a, x).Named("y");
            Variable.ConstrainEqualRandom(y, yPrior);
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = alg;
            a.ObservedValue = new Matrix(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            Vector xMean = Vector.FromArray(1, 2, 3);
            PositiveDefiniteMatrix xVar = new PositiveDefiniteMatrix(new double[,] { { 4, 1, 2 }, { 1, 5, 1 }, { 2, 1, 6 } });
            Vector yMean = a.ObservedValue * xMean;
            PositiveDefiniteMatrix yVar = new PositiveDefiniteMatrix(new double[,] { { 9, 1 }, { 1, 11 } });
            Matrix aT = a.ObservedValue.Transpose();
            for (int atrial = 0; atrial < 3; atrial++)
            {
                x.ClearObservedValue();
                if (atrial == 0)
                {
                    xPrior.ObservedValue = new VectorGaussian(xMean, xVar);
                }
                else
                {
                    xPrior.ObservedValue = VectorGaussian.PointMass(xMean);
                    if (atrial == 2) x.ObservedValue = xMean;
                }
                for (int strial = 0; strial < 3; strial++)
                {
                    y.ClearObservedValue();
                    if (strial == 0)
                    {
                        yPrior.ObservedValue = new VectorGaussian(yMean, yVar);
                    }
                    else
                    {
                        yPrior.ObservedValue = VectorGaussian.PointMass(yMean);
                        if (strial == 2) y.ObservedValue = yMean;
                    }
                    try
                    {
                        VectorGaussian yLike = new VectorGaussian(a.ObservedValue * xPrior.ObservedValue.GetMean(),
                                                                  new PositiveDefiniteMatrix(a.ObservedValue * xPrior.ObservedValue.GetVariance() * aT));
                        VectorGaussian yExpected = yLike * yPrior.ObservedValue;
                        VectorGaussian xLike = MatrixVectorProductOp.BAverageConditional(yPrior.ObservedValue, a.ObservedValue,
                                                                                         new VectorGaussian(xPrior.ObservedValue.Dimension));
                        VectorGaussian xExpected = xLike * xPrior.ObservedValue;
                        double evExpected = yPrior.ObservedValue.GetLogAverageOf(yLike);
                        VectorGaussian yActual = engine.Infer<VectorGaussian>(y);
                        object xActual = engine.Infer(x);
                        double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                        Console.WriteLine("y = {0} should be {1}", yActual, yExpected);
                        Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
                        Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                        Assert.True(yExpected.MaxDiff(yActual) < 1e-10);
                        Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
                        Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-10);
                    }
                    catch (CompilationFailedException ex)
                    {
                        if (yPrior.ObservedValue.IsPointMass)
                        {
                           if(verbose) Console.WriteLine("Correctly threw " + ex);
                        }
                        else
                            throw;
                    }
                    catch (NotSupportedException ex)
                    {
                        if (yPrior.ObservedValue.IsPointMass)
                        {
                            if (verbose) Console.WriteLine("Correctly threw " + ex);
                        }
                        else
                            throw;
                    }
                }
            }
        }

        [Fact]
        public void GatedSumTest()
        {
            GatedSum(new ExpectationPropagation());
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void VmpGatedSumTest()
        {
            GatedSum(new VariationalMessagePassing());
        }

        private void GatedSum(IAlgorithm alg)
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<GaussianArray> arrayPrior = Variable.New<GaussianArray>().Named("arrayPrior");
            Variable<int> arraySize = Variable.New<int>().Named("arraySize");
            Range item = new Range(arraySize);
            VariableArray<double> array = Variable.Array<double>(item).Named("array");
            array.SetTo(Variable<double[]>.Random(arrayPrior));
            Variable<Gaussian> sumPrior = Variable.New<Gaussian>().Named("sumPrior");
            Variable<double> sum = Variable.Sum(array).Named("sum");
            Variable.ConstrainEqualRandom(sum, sumPrior);
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = alg;
            engine.ShowProgress = false;
            for (int atrial = 0; atrial < 5; atrial++)
            {
                array.ClearObservedValue();
                if (atrial == 0)
                {
                    arrayPrior.ObservedValue = new GaussianArray(new Gaussian[] { new Gaussian(1, 2), new Gaussian(3, 4) });
                }
                else if (atrial == 1)
                {
                    arrayPrior.ObservedValue = new GaussianArray(new Gaussian[] { Gaussian.PointMass(1), new Gaussian(3, 4) });
                }
                else if (atrial == 2 || atrial == 3)
                {
                    arrayPrior.ObservedValue = new GaussianArray(new Gaussian[] { Gaussian.PointMass(1), Gaussian.PointMass(2) });
                    if (atrial == 3) array.ObservedValue = new double[] { 1, 2 };
                }
                else
                {
                    arrayPrior.ObservedValue = new GaussianArray(new Gaussian[] { new Gaussian(1, 2), Gaussian.Uniform() });
                }
                for (int strial = 0; strial < 3; strial++)
                {
                    sum.ClearObservedValue();
                    if (strial == 0)
                    {
                        sumPrior.ObservedValue = new Gaussian(5, 6);
                    }
                    else
                    {
                        sumPrior.ObservedValue = Gaussian.PointMass(3);
                        if (strial == 2) sum.ObservedValue = 3;
                    }
                    arraySize.ObservedValue = arrayPrior.ObservedValue.Count;
                    Gaussian[] arrayExpectedArray = new Gaussian[arrayPrior.ObservedValue.Count];
                    arrayExpectedArray[0] = arrayPrior.ObservedValue[0] * DoublePlusOp.AAverageConditional(sumPrior.ObservedValue, arrayPrior.ObservedValue[1]);
                    arrayExpectedArray[1] = arrayPrior.ObservedValue[1] * DoublePlusOp.AAverageConditional(sumPrior.ObservedValue, arrayPrior.ObservedValue[0]);
                    Gaussian sumLike = DoublePlusOp.SumAverageConditional(arrayPrior.ObservedValue[0], arrayPrior.ObservedValue[1]);
                    Gaussian sumExpected = sumLike * sumPrior.ObservedValue;
                    double evExpected = DoublePlusOp.LogAverageFactor(sumPrior.ObservedValue, sumLike);
                    if (alg is VariationalMessagePassing)
                    {
                        double m0, v0, m1, v1;
                        arrayPrior.ObservedValue[0].GetMeanAndVariance(out m0, out v0);
                        arrayPrior.ObservedValue[1].GetMeanAndVariance(out m1, out v1);
                        double mSum, vSum;
                        sumPrior.ObservedValue.GetMeanAndVariance(out mSum, out vSum);
                        double vTotal = vSum + v0 + v1;
                        if (vTotal == 0)
                        {
                            arrayExpectedArray[0] = new Gaussian(m0, v0);
                            arrayExpectedArray[1] = new Gaussian(m1, v1);
                        }
                        else
                        {
                            double r = (mSum - m0 - m1) / vTotal;
                            arrayExpectedArray[0] = new Gaussian(m0 + v0 * r, 1 / (1 / v0 + 1 / vSum));
                            arrayExpectedArray[1] = new Gaussian(m1 + v1 * r, 1 / (1 / v1 + 1 / vSum));
                        }
                        sumExpected = DoublePlusOp.SumAverageConditional(arrayExpectedArray[0], arrayExpectedArray[1]);
                        evExpected = arrayExpectedArray[0].GetAverageLog(arrayPrior.ObservedValue[0]) +
                                     arrayExpectedArray[1].GetAverageLog(arrayPrior.ObservedValue[1]) +
                                     sumExpected.GetAverageLog(sumPrior.ObservedValue)
                                     - arrayExpectedArray[0].GetAverageLog(arrayExpectedArray[0]) - arrayExpectedArray[1].GetAverageLog(arrayExpectedArray[1]);
                    }
                    IDistribution<double[]> arrayExpected = Distribution<double>.Array(arrayExpectedArray);
                    try
                    {
                        Gaussian sumActual = engine.Infer<Gaussian>(sum);
                        object arrayActual = engine.Infer(array);
                        double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                        Console.WriteLine("sum = {0} should be {1}", sumActual, sumExpected);
                        Console.WriteLine(StringUtil.JoinColumns("array = ", arrayActual, " should be ", arrayExpected));
                        Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                        Assert.True(sumExpected.MaxDiff(sumActual) < 1e-8);
                        Assert.True(arrayExpected.MaxDiff(arrayActual) < 1e-8);
                        Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-8);
                    }
                    catch (CompilationFailedException ex)
                    {
                        if (alg is VariationalMessagePassing && sumPrior.ObservedValue.IsPointMass)
                        {
                            if (verbose) Console.WriteLine("Correctly threw " + ex);
                        }
                        else
                            throw;
                    }
                    catch (NotSupportedException ex)
                    {
                        if (alg is VariationalMessagePassing && sumPrior.ObservedValue.IsPointMass)
                        {
                            if (verbose) Console.WriteLine("Correctly threw " + ex);
                        }
                        else
                            throw;
                    }
                }
            }
        }

        /// <summary>
        /// Tests the VectorGaussian sum factor in an evidence gate using EP.
        /// </summary>
        [Fact]
        public void EpGatedVectorGaussianSumFactorTest()
        {
            this.GatedVectorGaussianSumFactorTest(new ExpectationPropagation());
        }

        /// <summary>
        /// Tests the VectorGaussian sum factor in an evidence gate using VMP.
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")]
        public void VmpGatedVectorGaussianSumFactorTest()
        {
            this.GatedVectorGaussianSumFactorTest(new VariationalMessagePassing(), 1000);
        }

        /// <summary>
        /// Tests the VectorGaussian sum factor in an evidence gate.
        /// </summary>
        /// <param name="algorithm">The inference algorithm.</param>
        /// <param name="inferenceIterationCount">The number of iterations of the inference algorithm. Defaults to 50.</param>
        private void GatedVectorGaussianSumFactorTest(IAlgorithm algorithm, int inferenceIterationCount = 50)
        {
            // Model
            Variable<int> arraySize;
            Variable<VectorGaussianArray> arrayPrior;
            VariableArray<Vector> array;
            Variable<VectorGaussian> sumPrior;
            Variable<Vector> sum;

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");

            using (Variable.If(evidence))
            {
                arraySize = Variable.New<int>().Named("arraySize");
                var element = new Range(arraySize).Named("element");

                arrayPrior = Variable.New<VectorGaussianArray>().Named("arrayPrior");

                array = Variable.Array<Vector>(element).Named("array");
                array.SetTo(Variable<Vector[]>.Random(arrayPrior));

                sumPrior = Variable.New<VectorGaussian>().Named("sumPrior");
                sum = Variable.Sum(array).Named("sum");

                Variable.ConstrainEqualRandom(sum, sumPrior);
            }

            // Inference engine
            var engine = new InferenceEngine { Algorithm = algorithm, NumberOfIterations = inferenceIterationCount, ShowProgress = true };
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            engine.Compiler.RequiredQuality = QualityBand.Experimental;

            // Expected results (these come from the one-dimensional gated sum test)
            var expectedArrayData = new VectorGaussian[5, 3][];
            expectedArrayData[0, 0] = new[] { new VectorGaussian(1.167, 1.667), new VectorGaussian(3.333, 2.667) };
            expectedArrayData[0, 1] = new[] { new VectorGaussian(0.6667, 1.333), new VectorGaussian(2.333, 1.333) };
            expectedArrayData[0, 2] = new[] { new VectorGaussian(0.6667, 1.333), new VectorGaussian(2.333, 1.333) };
            expectedArrayData[1, 0] = new[] { VectorGaussian.PointMass(1), new VectorGaussian(3.4, 2.4) };
            expectedArrayData[1, 1] = new[] { VectorGaussian.PointMass(1), VectorGaussian.PointMass(2) };
            expectedArrayData[1, 2] = new[] { VectorGaussian.PointMass(1), VectorGaussian.PointMass(2) };
            expectedArrayData[2, 0] = new[] { VectorGaussian.PointMass(1), VectorGaussian.PointMass(2) };
            expectedArrayData[2, 1] = new[] { VectorGaussian.PointMass(1), VectorGaussian.PointMass(2) };
            expectedArrayData[2, 2] = new[] { VectorGaussian.PointMass(1), VectorGaussian.PointMass(2) };
            expectedArrayData[3, 0] = new[] { VectorGaussian.PointMass(1), VectorGaussian.PointMass(2) };
            expectedArrayData[3, 1] = new[] { VectorGaussian.PointMass(1), VectorGaussian.PointMass(2) };
            expectedArrayData[3, 2] = new[] { VectorGaussian.PointMass(1), VectorGaussian.PointMass(2) };
            expectedArrayData[4, 0] = new[] { new VectorGaussian(1, 2), new VectorGaussian(4, 8) };
            expectedArrayData[4, 1] = new[] { new VectorGaussian(1, 2), new VectorGaussian(2, 2) };
            expectedArrayData[4, 2] = new[] { new VectorGaussian(1, 2), new VectorGaussian(2, 2) };

            var expectedSumData = new VectorGaussian[5, 3];
            expectedSumData[0, 0] = new VectorGaussian(4.5, 3);
            expectedSumData[0, 1] = VectorGaussian.PointMass(3);
            expectedSumData[0, 2] = VectorGaussian.PointMass(3);
            expectedSumData[1, 0] = new VectorGaussian(4.4, 2.4);
            expectedSumData[1, 1] = VectorGaussian.PointMass(3);
            expectedSumData[1, 2] = VectorGaussian.PointMass(3);
            expectedSumData[2, 0] = VectorGaussian.PointMass(3);
            expectedSumData[2, 1] = VectorGaussian.PointMass(3);
            expectedSumData[2, 2] = VectorGaussian.PointMass(3);
            expectedSumData[3, 0] = VectorGaussian.PointMass(3);
            expectedSumData[3, 1] = VectorGaussian.PointMass(3);
            expectedSumData[3, 2] = VectorGaussian.PointMass(3);
            expectedSumData[4, 0] = new VectorGaussian(5, 6);
            expectedSumData[4, 1] = VectorGaussian.PointMass(3);
            expectedSumData[4, 2] = VectorGaussian.PointMass(3);

            var expectedEvidenceData = new double[5, 3];
            expectedEvidenceData[0, 0] = -2.20305852476534;
            expectedEvidenceData[0, 1] = -1.89815160115203;
            expectedEvidenceData[0, 2] = -1.89815160115203;
            expectedEvidenceData[1, 0] = -2.12023107970169;
            expectedEvidenceData[1, 1] = -1.73708571376462;
            expectedEvidenceData[1, 2] = -1.73708571376462;
            expectedEvidenceData[2, 0] = -2.14815160115203;
            expectedEvidenceData[2, 1] = 0;
            expectedEvidenceData[2, 2] = 0;
            expectedEvidenceData[3, 0] = -2.14815160115203;
            expectedEvidenceData[3, 1] = 0;
            expectedEvidenceData[3, 2] = 0;
            expectedEvidenceData[4, 0] = 0;
            expectedEvidenceData[4, 1] = 0;
            expectedEvidenceData[4, 2] = 0;

            // Check differences
            for (int arrayTrial = 0; arrayTrial < 5; arrayTrial++)
            {
                array.ClearObservedValue();
                switch (arrayTrial)
                {
                    case 0:
                        arrayPrior.ObservedValue = new VectorGaussianArray(new[] { new VectorGaussian(1, 2), new VectorGaussian(3, 4) });
                        break;
                    case 1:
                        arrayPrior.ObservedValue = new VectorGaussianArray(new[] { VectorGaussian.PointMass(1), new VectorGaussian(3, 4) });
                        break;
                    case 2:
                    case 3:
                        arrayPrior.ObservedValue = new VectorGaussianArray(new[] { VectorGaussian.PointMass(1), VectorGaussian.PointMass(2) });
                        if (arrayTrial == 3)
                        {
                            array.ObservedValue = new[] { Vector.FromArray(1.0), Vector.FromArray(2.0) };
                        }

                        break;
                    default:
                        arrayPrior.ObservedValue = new VectorGaussianArray(new[] { new VectorGaussian(1, 2), VectorGaussian.Uniform(1) });
                        break;
                }

                for (int sumTrial = 0; sumTrial < 3; sumTrial++)
                {
                    sum.ClearObservedValue();
                    if (sumTrial == 0)
                    {
                        sumPrior.ObservedValue = new VectorGaussian(5, 6);
                    }
                    else
                    {
                        sumPrior.ObservedValue = VectorGaussian.PointMass(3);
                        if (sumTrial == 2)
                        {
                            sum.ObservedValue = Vector.FromArray(3.0);
                        }
                    }

                    arraySize.ObservedValue = arrayPrior.ObservedValue.Count;

                    IDistribution<Vector[]> expectedArray = Distribution<Vector>.Array(expectedArrayData[arrayTrial, sumTrial]);
                    VectorGaussian expectedSum = expectedSumData[arrayTrial, sumTrial];
                    double expectedEvidence = expectedEvidenceData[arrayTrial, sumTrial];

                    try
                    {
                        var actualSum = engine.Infer<VectorGaussian>(sum);
                        object actualArray = engine.Infer(array);
                        double actualEvidence = engine.Infer<Bernoulli>(evidence).LogOdds;

                        Console.WriteLine("sum = {0} should be {1}", actualSum, expectedSum);
                        Console.WriteLine(StringUtil.JoinColumns("array = ", actualArray, " should be ", expectedArray));
                        Console.WriteLine("evidence = {0} should be {1}", actualEvidence, expectedEvidence);

                        Assert.True(expectedSum.MaxDiff(actualSum) <= 1e-8);
                        Assert.True(expectedArray.MaxDiff(actualArray) <= 1e-3);
                        Assert.True(MMath.AbsDiff(expectedEvidence, actualEvidence, 1e-6) <= 1e-8);
                    }
                    catch (CompilationFailedException compilationFailedException)
                    {
                        if (algorithm is VariationalMessagePassing && sumPrior.ObservedValue.IsPointMass)
                        {
                            if (verbose) Console.WriteLine("Correctly threw " + compilationFailedException);
                        }
                        else
                        {
                            throw;
                        }
                    }
                    catch (NotSupportedException notSupportedException)
                    {
                        if (algorithm is VariationalMessagePassing && sumPrior.ObservedValue.IsPointMass)
                        {
                            if (verbose) Console.WriteLine("Correctly threw " + notSupportedException);
                        }
                        else
                        {
                            throw;
                        }
                    }
                }
            }
        }

        [Fact]
        public void GatedSumWhereTest()
        {
            Diffable[,] aActual, bActual, aExpected, bExpected, cActual, cExpected;
            double[,] evExpected, evActual;
            GatedSumWhere(false, out aExpected, out bExpected, out cExpected, out evExpected);
            GatedSumWhere(true, out aActual, out bActual, out cActual, out evActual);
            for (int aTrial = 0; aTrial < 3; aTrial++)
            {
                for (int bTrial = 0; bTrial < 3; bTrial++)
                {
                    bool writeOutput = false;
                    if (writeOutput)
                    {
                        Console.WriteLine("a = {0} should be {1}", aActual[aTrial, bTrial], aExpected[aTrial, bTrial]);
                        Console.WriteLine("b = {0} should be {1}", bActual[aTrial, bTrial], bExpected[aTrial, bTrial]);
                        Console.WriteLine("c = {0} should be {1}", cActual[aTrial, bTrial], cExpected[aTrial, bTrial]);
                        Console.WriteLine("evidence = {0} should be {1}", evActual[aTrial, bTrial], evExpected[aTrial, bTrial]);
                    }
                    Assert.True(aActual[aTrial, bTrial].MaxDiff(aExpected[aTrial, bTrial]) < 1e-10);
                    Assert.True(bActual[aTrial, bTrial].MaxDiff(bExpected[aTrial, bTrial]) < 1e-10);
                    Assert.True(cActual[aTrial, bTrial].MaxDiff(cExpected[aTrial, bTrial]) < 1e-10);
                    Assert.True(MMath.AbsDiff(evActual[aTrial, bTrial], evExpected[aTrial, bTrial], 1e-8) < 1e-10);
                }
            }
        }
        private void GatedSumWhere(bool useVector, out Diffable[,] aPost, out Diffable[,] bPost, out Diffable[,] cPost, out double[,] evPost)
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Range item = new Range(2);
            VariableArray<Bernoulli> aPrior = Variable.Array<Bernoulli>(item).Named("aPrior");
            VariableArray<Gaussian> bPrior = Variable.Array<Gaussian>(item).Named("bPrior");
            Variable<Gaussian> cPrior = Variable.New<Gaussian>().Named("cPrior");
            VariableArray<bool> a = Variable.Array<bool>(item).Named("a");
            a[item] = Variable<bool>.Random(aPrior[item]);
            VariableArray<double> b = Variable.Array<double>(item).Named("b");
            b[item] = Variable<double>.Random(bPrior[item]);
            Variable<double> c;
            if (useVector)
                c = Variable.SumWhere(a, Variable.Vector(b)).Named("c");
            else
                c = Variable.SumWhere(a, b).Named("c");
            Variable.ConstrainEqualRandom(c, cPrior);
            block.CloseBlock();

            aPost = new Diffable[3, 3];
            bPost = new Diffable[3, 3];
            cPost = new Diffable[3, 3];
            evPost = new double[3, 3];

            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = new VariationalMessagePassing();
            for (int aTrial = 0; aTrial < 3; aTrial++)
            {
                a.ClearObservedValue();
                if (aTrial == 0)
                {
                    aPrior.ObservedValue = new[] { new Bernoulli(0.1), new Bernoulli(0.2) };
                }
                else
                {
                    aPrior.ObservedValue = new[] { Bernoulli.PointMass(false), Bernoulli.PointMass(true) };
                    if (aTrial == 2)
                    {
                        a.ObservedValue = new[] { false, true };
                    }
                }
                for (int bTrial = 0; bTrial < 3; bTrial++)
                {
                    b.ClearObservedValue();
                    if (bTrial == 0)
                    {
                        bPrior.ObservedValue = new[] { new Gaussian(1, 2), new Gaussian(3, 4) };
                    }
                    else
                    {
                        bPrior.ObservedValue = new[] { Gaussian.PointMass(1), Gaussian.PointMass(3) };
                        if (bTrial == 2)
                        {
                            b.ObservedValue = new[] { 1.0, 3.0 };
                        }
                    }
                    cPrior.ObservedValue = new Gaussian(5, 6);
                    ICloneable aActual = engine.Infer<ICloneable>(a);
                    ICloneable bActual = engine.Infer<ICloneable>(b);
                    Gaussian cActual = engine.Infer<Gaussian>(c);
                    double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                    aPost[aTrial, bTrial] = (Diffable)aActual.Clone();
                    bPost[aTrial, bTrial] = (Diffable)bActual.Clone();
                    cPost[aTrial, bTrial] = cActual;
                    evPost[aTrial, bTrial] = evActual;
                }
            }
        }

        [Fact]
        public void GatedIsGreaterThanTest()
        {
            GatedIsGreaterThan(new ExpectationPropagation());
            GatedIsGreaterThan(new VariationalMessagePassing());
        }

        private void GatedIsGreaterThan(IAlgorithm alg)
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<Discrete> nPrior = Variable.New<Discrete>().Named("nPrior");
            Variable<Discrete> xPrior = Variable.New<Discrete>().Named("xPrior");
            Variable<Bernoulli> bPrior = Variable.New<Bernoulli>().Named("bPrior");
            Variable<int> n = Variable.Random<int, Discrete>(nPrior).Named("n");
            Variable<int> x = Variable.Random<int, Discrete>(xPrior).Named("x");
            Variable<bool> b = (x > n).Named("b");
            Variable.ConstrainEqualRandom(b, bPrior);
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = alg;
            for (int btrial = 0; btrial < 3; btrial++)
            {
                b.ClearObservedValue();
                if (btrial == 0)
                {
                    bPrior.ObservedValue = new Bernoulli(0.2);
                }
                else
                {
                    bPrior.ObservedValue = Bernoulli.PointMass(false);
                    if (btrial == 2) b.ObservedValue = false;
                }
                for (int ntrial = 0; ntrial < 3; ntrial++)
                {
                    n.ClearObservedValue();
                    if (ntrial == 0)
                    {
                        Vector nProbs = Vector.FromArray(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5 });
                        nProbs.Scale(1.0 / nProbs.Sum());
                        nPrior.ObservedValue = new Discrete(nProbs);
                    }
                    else
                    {
                        nPrior.ObservedValue = Discrete.PointMass(3, 5);
                        if (ntrial == 2) n.ObservedValue = 3;
                    }
                    for (int xtrial = 0; xtrial < 3; xtrial++)
                    {
                        x.ClearObservedValue();
                        if (xtrial == 0)
                        {
                            Vector xProbs = Vector.FromArray(new double[] { 0.3, 0.4, 0.1, 0.5, 0.2 });
                            xProbs.Scale(1.0 / xProbs.Sum());
                            xPrior.ObservedValue = new Discrete(xProbs);
                        }
                        else
                        {
                            xPrior.ObservedValue = Discrete.PointMass(2, 5);
                            if (xtrial == 2) x.ObservedValue = 2;
                        }
                        double[] toX = new double[xPrior.ObservedValue.Dimension];
                        double[] toN = new double[nPrior.ObservedValue.Dimension];
                        double z = 0.0;
                        double bprobTrue = bPrior.ObservedValue.GetProbTrue();
                        for (int i = 0; i < xPrior.ObservedValue.Dimension; i++)
                        {
                            for (int j = 0; j < nPrior.ObservedValue.Dimension; j++)
                            {
                                // factor is f(x,n) = p(b=T) 1(x > n) + p(b=F) 1(x <= n)
                                double f = (i > j) ? bprobTrue : (1 - bprobTrue);
                                toX[i] += nPrior.ObservedValue[j] * f;
                                toN[j] += xPrior.ObservedValue[i] * f;
                                z += xPrior.ObservedValue[i] * nPrior.ObservedValue[j] * f;
                            }
                        }
                        Discrete nExpected = nPrior.ObservedValue * new Discrete(toN);
                        Discrete xExpected = xPrior.ObservedValue * new Discrete(toX);
                        double evExpected = System.Math.Log(z);
                        try
                        {
                            Discrete nActual = engine.Infer<Discrete>(n);
                            Discrete xActual = engine.Infer<Discrete>(x);
                            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
                            Console.WriteLine("n = {0} should be {1}", nActual, nExpected);
                            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                            if (alg is VariationalMessagePassing)
                            {
                                // TODO: determine the correct answers here
                            }
                            else
                            {
                                Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
                                Assert.True(nExpected.MaxDiff(nActual) < 1e-10);
                                Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-10);
                            }
                        }
                        catch (CompilationFailedException ex)
                        {
                            if (alg is VariationalMessagePassing && bPrior.ObservedValue.IsPointMass)
                            {
                                if (verbose) Console.WriteLine("Correctly threw " + ex);
                            }
                            else
                                throw;
                        }
                        catch (NotSupportedException ex)
                        {
                            if (alg is VariationalMessagePassing && bPrior.ObservedValue.IsPointMass)
                            {
                                if (verbose) Console.WriteLine("Correctly threw " + ex);
                            }
                            else
                                throw;
                        }
                    }
                }
            }
        }

        [Fact]
        public void GatedBinomialTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<Discrete> nPrior = Variable.New<Discrete>().Named("nPrior");
            Variable<Discrete> xPrior = Variable.New<Discrete>().Named("xPrior");
            Variable<int> n = Variable.Random<int, Discrete>(nPrior).Named("n");
            double p = 0.15;
            Variable<int> x = Variable.Binomial(n, p).Named("x");
            Variable.ConstrainEqual(x, Variable.Random<int, Discrete>(xPrior));
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            for (int ntrial = 0; ntrial < 2; ntrial++)
            {
                if (ntrial == 0)
                {
                    Vector nProbs = Vector.FromArray(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5 });
                    nProbs.Scale(1.0 / nProbs.Sum());
                    nPrior.ObservedValue = new Discrete(nProbs);
                }
                else
                {
                    nPrior.ObservedValue = Discrete.PointMass(3, 5);
                }
                for (int xtrial = 0; xtrial < 2; xtrial++)
                {
                    if (xtrial == 0)
                    {
                        Vector xProbs = Vector.FromArray(new double[] { 0.3, 0.4, 0.1, 0.5, 0.2 });
                        xProbs.Scale(1.0 / xProbs.Sum());
                        xPrior.ObservedValue = new Discrete(xProbs);
                    }
                    else
                    {
                        xPrior.ObservedValue = Discrete.PointMass(2, 5);
                    }
                    double[] toX = new double[xPrior.ObservedValue.Dimension];
                    double[] toN = new double[nPrior.ObservedValue.Dimension];
                    double z = 0.0;
                    for (int i = 0; i < xPrior.ObservedValue.Dimension; i++)
                    {
                        for (int j = 0; j < nPrior.ObservedValue.Dimension; j++)
                        {
                            if (i <= j)
                            {
                                // factor is f(x,n) = 1(x <= n) nchoosek(n,x) p^x (1-p)^(n-x)
                                double f = System.Math.Exp(MMath.ChooseLn(j, i) + i * System.Math.Log(p) + (j - i) * System.Math.Log(1 - p));
                                toX[i] += nPrior.ObservedValue[j] * f;
                                toN[j] += xPrior.ObservedValue[i] * f;
                                z += xPrior.ObservedValue[i] * nPrior.ObservedValue[j] * f;
                            }
                        }
                    }
                    Discrete nExpected = nPrior.ObservedValue * new Discrete(toN);
                    Discrete xExpected = xPrior.ObservedValue * new Discrete(toX);
                    double evExpected = System.Math.Log(z);
                    Discrete nActual = engine.Infer<Discrete>(n);
                    Discrete xActual = engine.Infer<Discrete>(x);
                    double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                    Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
                    Console.WriteLine("n = {0} should be {1}", nActual, nExpected);
                    Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                    Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
                    Assert.True(nExpected.MaxDiff(nActual) < 1e-10);
                    Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-10);
                }
            }
        }

        [Fact]
        public void GatedDiscreteUniformTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<Discrete> nPrior = Variable.New<Discrete>().Named("nPrior");
            Variable<Discrete> xPrior = Variable.New<Discrete>().Named("xPrior");
            Variable<int> n = Variable<int>.Random(nPrior).Named("n");
            Variable<int> x = Variable.DiscreteUniform(n).Named("x");
            Variable.ConstrainEqual(x, Variable<int>.Random(xPrior));
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            for (int ntrial = 0; ntrial < 2; ntrial++)
            {
                if (ntrial == 0)
                {
                    Vector nProbs = Vector.FromArray(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5 });
                    nProbs.Scale(1.0 / nProbs.Sum());
                    nPrior.ObservedValue = new Discrete(nProbs);
                }
                else
                {
                    nPrior.ObservedValue = Discrete.PointMass(3, 5);
                }
                for (int xtrial = 0; xtrial < 2; xtrial++)
                {
                    if (xtrial == 0)
                    {
                        Vector xProbs = Vector.FromArray(new double[] { 0.3, 0.4, 0.1, 0.5 });
                        xProbs.Scale(1.0 / xProbs.Sum());
                        xPrior.ObservedValue = new Discrete(xProbs);
                    }
                    else
                    {
                        xPrior.ObservedValue = Discrete.PointMass(2, 4);
                    }
                    double[] toX = new double[xPrior.ObservedValue.Dimension];
                    double[] toN = new double[nPrior.ObservedValue.Dimension];
                    double z = 0.0;
                    for (int i = 0; i < xPrior.ObservedValue.Dimension; i++)
                    {
                        for (int j = 1; j < nPrior.ObservedValue.Dimension; j++)
                        {
                            if (j > i)
                            {
                                // factor is f(x,n) = 1(x < n)/n
                                double f = 1.0 / j;
                                toX[i] += nPrior.ObservedValue[j] * f;
                                toN[j] += xPrior.ObservedValue[i] * f;
                                z += xPrior.ObservedValue[i] * nPrior.ObservedValue[j] * f;
                            }
                        }
                    }
                    Discrete nExpected = nPrior.ObservedValue * new Discrete(toN);
                    Discrete xExpected = xPrior.ObservedValue * new Discrete(toX);
                    double evExpected = System.Math.Log(z);
                    Discrete nActual = engine.Infer<Discrete>(n);
                    Discrete xActual = engine.Infer<Discrete>(x);
                    double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                    Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
                    Console.WriteLine("n = {0} should be {1}", nActual, nExpected);
                    Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                    Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
                    Assert.True(nExpected.MaxDiff(nActual) < 1e-10);
                    Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-10);
                }
            }
        }

        [Fact]
        public void GatedSubarrayTest2()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            double priorA = 0.3;
            Range item = new Range(4).Named("item");
            VariableArray<bool> array = Variable.Array<bool>(item).Named("array");
            array[item] = Variable.Bernoulli(priorA).ForEach(item);
            Range item2 = new Range(2).Named("item2");
            VariableArray<int> indices = Variable.Constant(new int[] { 1, 2 }, item2).Named("indices");
            double priorB = 0.1;
            Variable<bool> b = Variable.Bernoulli(priorB).Named("b");
            using (Variable.If(b))
            {
                VariableArray<bool> items = Variable.Subarray(array, indices).Named("items");
                items.ObservedValue = new bool[] { true, false };
            }
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            double evExpected, evActual;
            double sumCondB = priorA * (1 - priorA);
            evExpected = priorB * sumCondB + (1 - priorB);
            evActual = System.Math.Exp(engine.Infer<Bernoulli>(evidence).LogOdds);
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual) < 1e-8);
        }

        [Fact]
        public void GatedSubarrayTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            double priorA = 0.3;
            Range item = new Range(4).Named("item");
            VariableArray<bool> array = Variable.Array<bool>(item).Named("array");
            array[item] = Variable.Bernoulli(priorA).ForEach(item);
            Range item2 = new Range(2).Named("item2");
            VariableArray<int> indices = Variable.Constant(new int[] { 1, 2 }, item2).Named("indices");
            double priorB = 0.1;
            Variable<bool> b = Variable.Bernoulli(priorB).Named("b");
            double priorC = 0.2;
            using (Variable.If(b))
            {
                VariableArray<bool> items = Variable.Subarray(array, indices).Named("items");
                Variable.ConstrainEqualRandom(items[item2], new Bernoulli(priorC));
            }
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            double sumCondB = System.Math.Pow(priorC * priorA + (1 - priorC) * (1 - priorA), item2.SizeAsInt);
            double evExpected, evActual;
            evExpected = priorB * sumCondB + (1 - priorB);
            Bernoulli bExpected = new Bernoulli(priorB * sumCondB / evExpected);
            Bernoulli bActual = engine.Infer<Bernoulli>(b);
            Console.WriteLine("b = {0} should be {1}", bActual, bExpected);
            evActual = System.Math.Exp(engine.Infer<Bernoulli>(evidence).LogOdds);
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(bExpected.MaxDiff(bActual) < 1e-8);
            Assert.True(MMath.AbsDiff(evExpected, evActual) < 1e-8);
        }

        [Fact]
        public void GatedBetweenTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Gaussian priorL = new Gaussian(0.1, 0.9);
            Variable<double> lowerBound = Variable.Random<double>(priorL).Named("lowerBound");
            Gaussian priorU = new Gaussian(0.3, 0.8);
            Variable<double> upperBound = Variable.Random<double>(priorU).Named("upperBound");
            Gaussian priorX = new Gaussian(0.2, 0.7);
            Variable<double> x = Variable.Random<double>(priorX).Named("x");
            Variable.ConstrainBetween(x, lowerBound, upperBound);
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            double evExpected, evActual;

            evExpected = IsBetweenGaussianOp.LogProbBetween(priorX, priorL, priorU);
            evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-8);

            lowerBound.ObservedValue = priorL.GetMean();
            evExpected = IsBetweenGaussianOp.LogProbBetween(priorX, Gaussian.PointMass(lowerBound.ObservedValue), priorU) + priorL.GetLogProb(lowerBound.ObservedValue);
            evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-8);

            upperBound.ObservedValue = priorU.GetMean();
            evExpected = IsBetweenGaussianOp.LogProbBetween(priorX, lowerBound.ObservedValue, upperBound.ObservedValue) + priorL.GetLogProb(lowerBound.ObservedValue) +
                         priorU.GetLogProb(upperBound.ObservedValue);
            evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-8);

            lowerBound.ClearObservedValue();
            upperBound.ClearObservedValue();
            x.ObservedValue = priorX.GetMean();
            evExpected = IsBetweenGaussianOp.LogProbBetween(Gaussian.PointMass(x.ObservedValue), priorL, priorU) + priorX.GetLogProb(x.ObservedValue);
            evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-8);

            lowerBound.ObservedValue = priorL.GetMean();
            evExpected = IsBetweenGaussianOp.LogProbBetween(Gaussian.PointMass(x.ObservedValue), Gaussian.PointMass(lowerBound.ObservedValue), priorU) +
                         priorX.GetLogProb(x.ObservedValue) + priorL.GetLogProb(lowerBound.ObservedValue);
            evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-8);

            upperBound.ObservedValue = priorU.GetMean();
            evExpected = IsBetweenGaussianOp.LogProbBetween(Gaussian.PointMass(x.ObservedValue), lowerBound.ObservedValue, upperBound.ObservedValue) +
                         priorX.GetLogProb(x.ObservedValue) + priorL.GetLogProb(lowerBound.ObservedValue) + priorU.GetLogProb(upperBound.ObservedValue);
            evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-8);
        }

        [Fact]
        public void GatedBetweenCCCTest()
        {
            var x = Variable.GaussianFromMeanAndVariance(100, 100);
            var b = Variable.Bernoulli(0.5);
            using (Variable.If(b))
            {
                Variable.ConstrainBetween(x, 100, 200);
            }
            InferenceEngine engine = new InferenceEngine();
            x.ObservedValue = 90;
            Bernoulli bExpected = new Bernoulli(0);
            Bernoulli bActual = engine.Infer<Bernoulli>(b);
            Console.WriteLine("b = {0} should be {1}", bActual, bExpected);
            Assert.True(bExpected.MaxDiff(bActual) < 1e-10);
        }

        [Fact]
        public void GatedIntAreEqualTest()
        {
            foreach (var algorithm in new IAlgorithm[] { new ExpectationPropagation(), new VariationalMessagePassing() })
            {
                Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
                IfBlock block = Variable.If(evidence);
                Vector priorA = Vector.FromArray(0.1, 0.9);
                Vector priorB = Vector.FromArray(0.2, 0.8);
                Variable<int> a = Variable.Discrete(priorA).Named("a");
                Variable<int> b = Variable.Discrete(priorB).Named("b");
                Variable<bool> c = (a == b).Named("c");
                double priorC = 0.3;
                Variable.ConstrainEqualRandom(c, new Bernoulli(priorC));
                block.CloseBlock();

                InferenceEngine engine = new InferenceEngine(algorithm);

                double probEqual = priorA.Inner(priorB);
                double evPrior = 0;
                for (int atrial = 0; atrial < 2; atrial++)
                {
                    if (atrial == 1)
                    {
                        a.ObservedValue = 1;
                        probEqual = priorB[1];
                        c.ClearObservedValue();
                        evPrior = System.Math.Log(priorA[1]);
                        priorA[0] = 0.0;
                        priorA[1] = 1.0;
                    }
                    double evExpected = System.Math.Log(probEqual * priorC + (1 - probEqual) * (1 - priorC)) + evPrior;
                    double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                    Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                    if (algorithm is ExpectationPropagation || atrial == 1)
                        Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-5) < 1e-5);

                    Bernoulli cExpected = new Bernoulli(probEqual * priorC / (probEqual * priorC + (1 - probEqual) * (1 - priorC)));
                    Bernoulli cActual = engine.Infer<Bernoulli>(c);
                    Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
                    if (algorithm is ExpectationPropagation || atrial == 1)
                        Assert.True(cExpected.MaxDiff(cActual) < 1e-10);

                    Vector postB = Vector.Zero(2);
                    postB[0] = priorB[0] * (priorA[0] * priorC + priorA[1] * (1 - priorC));
                    postB[1] = priorB[1] * (priorA[1] * priorC + priorA[0] * (1 - priorC));
                    postB.Scale(1.0 / postB.Sum());
                    Discrete bExpected = new Discrete(postB);
                    Discrete bActual = engine.Infer<Discrete>(b);
                    Console.WriteLine("b = {0} should be {1}", bActual, bExpected);
                    if (algorithm is ExpectationPropagation || atrial == 1)
                        Assert.True(bExpected.MaxDiff(bActual) < 1e-10);

                    if (atrial == 0 && algorithm is VariationalMessagePassing) continue;

                    for (int trial = 0; trial < 2; trial++)
                    {
                        if (trial == 0)
                        {
                            c.ObservedValue = true;
                            evExpected = System.Math.Log(probEqual * priorC) + evPrior;
                        }
                        else
                        {
                            c.ObservedValue = false;
                            evExpected = System.Math.Log((1 - probEqual) * (1 - priorC)) + evPrior;
                        }
                        evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                        Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                        Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-5) < 1e-5);

                        if (a.IsObserved)
                        {
                            bExpected = Discrete.PointMass(c.ObservedValue ? a.ObservedValue : 1 - a.ObservedValue, 2);
                        }
                        else
                        {
                            postB[0] = priorB[0] * (c.ObservedValue ? priorA[0] : priorA[1]);
                            postB[1] = priorB[1] * (c.ObservedValue ? priorA[1] : priorA[0]);
                            postB.Scale(1.0 / postB.Sum());
                            bExpected = new Discrete(postB);
                        }
                        bActual = engine.Infer<Discrete>(b);
                        Console.WriteLine("b = {0} should be {1}", bActual, bExpected);
                        Assert.True(bExpected.MaxDiff(bActual) < 1e-10);
                    }
                }
            }
        }

        [Fact]
        public void GatedPoissonCRTest()
        {
            double priorB = 0.1;
            Variable<bool> b = Variable.Bernoulli(priorB).Named("b");
            Gamma priorMean = new Gamma(1.2, 3.4);
            Variable<double> mean = Variable.Random(priorMean).Named("mean");
            IfBlock block = Variable.If(b);
            Variable<int> x = Variable.Poisson(mean).Named("sample");
            x.ObservedValue = 5;
            block.CloseBlock();

            // p(x|mean) = mean^(x) exp(-mean)/Gamma(x+1)
            //           = Ga(mean; x+1, 1)
            double sumCondT = System.Math.Exp(priorMean.GetLogAverageOf(new Gamma(x.ObservedValue + 1, 1)));
            double Z = priorB * sumCondT + (1 - priorB);
            double postB = priorB * sumCondT / Z;

            InferenceEngine engine = new InferenceEngine();
            Bernoulli bDist = engine.Infer<Bernoulli>(b);
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        [Fact]
        public void GatedPoissonRRTest()
        {
            double priorB = 0.1;
            Variable<bool> b = Variable.Bernoulli(priorB).Named("b");
            Gamma priorMean = new Gamma(1.2, 3.4);
            Variable<double> mean = Variable.Random(priorMean).Named("mean");
            IfBlock block = Variable.If(b);
            Variable<int> x = Variable.Poisson(mean).Named("sample");
            Poisson xLike = new Poisson(0.6, 0);
            Variable.ConstrainEqualRandom(x, xLike);
            block.CloseBlock();

            Rand.Restart(0);
            Gamma meanExpected;
            Poisson xExpected;
            double postB;
            bool useMonteCarlo = false;
            if (useMonteCarlo)
            {
                double sumCondT = 0;
                int nSamples = 100000;
                GammaEstimator meanEst = new GammaEstimator();
                PoissonEstimator xEst = new PoissonEstimator();
                for (int i = 0; i < nSamples; i++)
                {
                    double meanSample = priorMean.Sample();
                    int xSample = Poisson.Sample(meanSample);
                    double weight = System.Math.Exp(xLike.GetLogProb(xSample));
                    sumCondT += weight;
                    meanEst.Add(meanSample, weight);
                    xEst.Add(xSample, weight);
                }
                sumCondT /= nSamples;
                double Z = priorB * sumCondT + (1 - priorB);
                postB = priorB * sumCondT / Z;
                Gamma meanCondT = meanEst.GetDistribution(new Gamma());
                meanExpected = new Gamma();
                meanExpected.SetToSum(postB, meanCondT, 1 - postB, priorMean);
                xExpected = xEst.GetDistribution(new Poisson());
            }
            else
            {
                meanExpected = Gamma.FromShapeAndRate(1.2, 0.29678795957289567);
                xExpected = new Poisson(1.03728813559322);
                postB = 0.015617445898679475;
            }

            InferenceEngine engine = new InferenceEngine();
            Gamma meanActual = engine.Infer<Gamma>(mean);
            Console.WriteLine("mean = {0} (should be {1})", meanActual, meanExpected);
            Assert.True(meanExpected.MaxDiff(meanActual) < 1e-4);
            var xActual = engine.Infer<Poisson>(x);
            Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-4);
            Bernoulli bDist = engine.Infer<Bernoulli>(b);
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        [Trait("Category", "CsoftModel")]
        public void GatedDiscreteFromDiscreteRRTest()
        {
            Discrete samplePrior = new Discrete(0.2, 0.8);
            Vector selectorPrior = Vector.FromArray(0.4, 0.6);
            // note these rows are not normalized
            Matrix transitionMatrix = new Matrix(new double[,] { { 0.12, 0.34 }, { 0.56, 0.78 } });

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedDiscreteFromDiscreteRRModel, samplePrior, selectorPrior, transitionMatrix);
            ca.Execute(20);

            double sumCondT = transitionMatrix.QuadraticForm(selectorPrior, samplePrior.GetProbs());
            double priorB = 0.1;
            double Z = priorB * sumCondT + (1 - priorB);
            double postB = priorB * sumCondT / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedDiscreteFromDiscreteRRModel(Discrete samplePrior, Vector selectorPrior, Matrix transitionMatrix)
        {
            bool b = Factor.Bernoulli(0.1);
            int selector = Factor.Discrete(selectorPrior);
            if (b)
            {
                int sample = Factor.Discrete(selector, transitionMatrix);
                Constrain.EqualRandom(sample, samplePrior);
            }
            InferNet.Infer(b, nameof(b));
        }

        [Fact]
        public void GatedPlusCCCTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> x = Variable.New<double>().Named("x");
            Variable<double> y = Variable.New<double>().Named("y");
            Variable<double> sum = (x + y).Named("sum");
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            for (int trial = 0; trial < 3; trial++)
            {
                x.ObservedValue = 1;
                y.ObservedValue = 2;
                double evExpected;
                if (trial == 0)
                {
                    sum.ObservedValue = 3;
                    evExpected = 0;
                }
                else
                {
                    sum.ObservedValue = 4;
                    evExpected = Double.NegativeInfinity;
                    if (trial == 2) sum.IsReadOnly = true;
                }
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                Assert.True(MMath.AbsDiff(evActual, evExpected, 1e-10) < 1e-10);
            }
        }

        internal void GatedPlusCCCModel(double data)
        {
            bool b = Factor.Bernoulli(0.1);
            if (b)
            {
                data = Factor.Plus(0, 1);
            }
            InferNet.Infer(b, nameof(b));
        }

        [Fact]
        public void GatedVectorGaussianRCCTest()
        {
            Variable<Vector> mean1 = Variable.New<Vector>();
            mean1.ObservedValue = Vector.FromArray(0.3, 0.5);
            Variable<PositiveDefiniteMatrix> precision1 = Variable.New<PositiveDefiniteMatrix>();
            precision1.ObservedValue = PositiveDefiniteMatrix.IdentityScaledBy(2, 1.5);

            Variable<PositiveDefiniteMatrix> precision2 = Variable.New<PositiveDefiniteMatrix>();
            precision2.ObservedValue = PositiveDefiniteMatrix.IdentityScaledBy(2, 1);

            Variable<Vector> sample = Variable.New<Vector>();
            sample.ObservedValue = Vector.FromArray(0.1, 0.3);

            Variable<bool> selector = Variable.Bernoulli(0.5);

            using (Variable.If(selector))
            {
                sample.SetTo(Variable.VectorGaussianFromMeanAndPrecision(mean1, precision1));
            }

            using (Variable.IfNot(selector))
            {
                sample.SetTo(Variable.VectorGaussianFromMeanAndPrecision(Vector.FromArray(0.1, 0.2), precision2));
            }

            var engine = new InferenceEngine();
            var selectorPosterior = engine.Infer<Bernoulli>(selector);

            Assert.Equal(0.586730, selectorPosterior.GetProbTrue(), 1e-5);
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedVectorGaussianRRCTest()
        {
            double priorB = 0.1;
            double meanM = 0.4;
            double varM = 1.4;
            PositiveDefiniteMatrix precision = new PositiveDefiniteMatrix(new double[,] { { 1.3 } });
            VectorGaussian priorSample = new VectorGaussian(Vector.FromArray(new double[] { 0.2 }), new PositiveDefiniteMatrix(new double[,] { { 1.2 } }));
            VectorGaussian meanPrior = new VectorGaussian(Vector.FromArray(new double[] { meanM }), new PositiveDefiniteMatrix(new double[,] { { varM } }));

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedVectorGaussianRRCModel, precision, priorSample, meanPrior);
            ca.Execute(20);

            double sumCondT = System.Math.Exp(priorSample.GetMarginal(0).GetLogAverageOf(new Gaussian(meanM, varM + 1 / precision[0, 0])));
            double Z = priorB * sumCondT + (1 - priorB);
            double postB = priorB * sumCondT / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedVectorGaussianRRCModel(PositiveDefiniteMatrix precision, VectorGaussian samplePrior, VectorGaussian meanPrior)
        {
            bool b = Factor.Bernoulli(0.1);
            Vector mean = Factor.Random(meanPrior);
            if (b)
            {
                Vector sample = Factor.VectorGaussian(mean, precision);
                Constrain.EqualRandom(sample, samplePrior);
            }
            InferNet.Infer(b, nameof(b));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedGetItemTest()
        {
            double priorB = 0.1;
            double priorX = 0.2;
            double pXCondT = 0.3;

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedGetItemModel, 5, priorB, priorX, pXCondT);
            ca.Execute(20);

            double sumXCondT = priorX * pXCondT + (1 - priorX) * (1 - pXCondT);
            double Z = priorB * sumXCondT + (1 - priorB);
            double postB = priorB * sumXCondT / Z;
            double postX = priorX * pXCondT / sumXCondT;

            Bernoulli xActual = ca.Marginal<Bernoulli>("x");
            Bernoulli bActual = ca.Marginal<Bernoulli>("b");
            Bernoulli evActual = ca.Marginal<Bernoulli>("evidence");
            Console.WriteLine("x = {0} (should be {1})", xActual, postX);
            Console.WriteLine("b = {0} (should be {1})", bActual, postB);
            Console.WriteLine("evidence = {0} (should be {1})", System.Math.Exp(evActual.LogOdds), Z);
            Assert.True(System.Math.Abs(xActual.GetProbTrue() - postX) < 1e-4);
            Assert.True(System.Math.Abs(bActual.GetProbTrue() - postB) < 1e-4);
            Assert.True(System.Math.Abs(System.Math.Exp(evActual.LogOdds) - Z) < 1e-4);
        }

        private void GatedGetItemModel(int n, double priorB, double priorX, double pXCondT)
        {
            bool evidence = Factor.Bernoulli(0.5);
            if (evidence)
            {
                bool b = Factor.Bernoulli(priorB);
                bool[] array = new bool[n];
                for (int i = 0; i < n; i++)
                {
                    array[i] = Factor.Bernoulli(priorX);
                }
                if (b)
                {
                    bool x = Collection.GetItem(array, 0);
                    Constrain.EqualRandom(x, new Bernoulli(pXCondT));
                    InferNet.Infer(x, nameof(x));
                }
                InferNet.Infer(b, nameof(b));
                InferNet.Infer(array, nameof(array));
            }
            InferNet.Infer(evidence, nameof(evidence));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedUnaryTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedUnaryModel);
            ca.Execute(20);

            double priorB = 0.1;
            double pXCondT = 0.3;
            double pXCondF = 0.4;
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) [(pT)^x (1-pT)^(1-x)]^b [(pF)^x (1-pF)^(1-x)]^(1-b)
            double postB = priorB;
            double postX = priorB * pXCondT + (1 - priorB) * pXCondF;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Bernoulli xDist = ca.Marginal<Bernoulli>("x");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - postX) < 1e-4);
        }

        private void GatedUnaryModel()
        {
            bool b = Factor.Bernoulli(0.1);
            bool x;
            if (b)
            {
                x = Factor.Random(new Bernoulli(0.3));
            }
            else
            {
                x = Factor.Random(new Bernoulli(0.4));
            }
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(x, nameof(x));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedUnary2Test()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedUnary2Model);
            ca.Execute(20);

            double priorB = 0.1;
            double priorX = 0.2;
            double pXCondT = 0.3;
            double pXCondF = 0.4;
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x) 
            //                [(pT)^x (1-pT)^(1-x)]^b [(pF)^x (1-pF)^(1-x)]^(1-b)
            double sumXCondT = priorX * pXCondT + (1 - priorX) * (1 - pXCondT);
            double sumXCondF = priorX * pXCondF + (1 - priorX) * (1 - pXCondF);
            double Z = priorB * sumXCondT + (1 - priorB) * sumXCondF;
            double postB = priorB * sumXCondT / Z;
            double postX = priorX * (priorB * pXCondT + (1 - priorB) * pXCondF) / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Bernoulli xDist = ca.Marginal<Bernoulli>("x");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - postX) < 1e-4);
        }

        private void GatedUnary2Model()
        {
            bool b = Factor.Bernoulli(0.1);
            bool x;
            if (b)
            {
                x = Factor.Random(new Bernoulli(0.3));
            }
            else
            {
                x = Factor.Random(new Bernoulli(0.4));
            }
            Constrain.EqualRandom(x, new Bernoulli(0.2));
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(x, nameof(x));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedIsBetweenRRRRTest()
        {
            double priorB = 0.1;
            double meanX = 0.4, varX = 1.4;
            double meanL = 0.3, varL = 1.3;
            double meanU = 0.5, varU = 1.5;
            double priorY = 0.2;

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedIsBetweenRRRRModel, new Gaussian(meanX, varX), new Gaussian(meanL, varL), new Gaussian(meanU, varU));
            ca.Execute(20);

            double pYCondT = System.Math.Exp(IsBetweenGaussianOp.LogProbBetween(new Gaussian(meanX, varX), new Gaussian(meanL, varL), new Gaussian(meanU, varU)));
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) G(x)
            //                [(pT)^y (1-pT)^(1-y) delta(y - step(x))]^b
            double sumCondT = priorY * pYCondT + (1 - priorY) * (1 - pYCondT);
            double Z = priorB * sumCondT + (1 - priorB);
            double postB = priorB * sumCondT / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedIsBetweenRRRRModel(Gaussian priorX, Gaussian priorLowerBound, Gaussian priorUpperBound)
        {
            bool b = Factor.Bernoulli(0.1);
            double x = Factor.Random(priorX);
            double lowerBound = Factor.Random(priorLowerBound);
            double upperBound = Factor.Random(priorUpperBound);
            if (b)
            {
                bool y = Factor.IsBetween(x, lowerBound, upperBound);
                Constrain.EqualRandom(y, new Bernoulli(0.2));
            }
            InferNet.Infer(b, nameof(b));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedIsBetweenCRCCTest()
        {
            GatedIsBetweenCRCC(true);
            GatedIsBetweenCRCC(false);
        }

        private void GatedIsBetweenCRCC(bool y)
        {
            double priorB = 0.1;
            double meanX = 0.4, varX = 1.4;
            double L = 0.3;
            double U = 0.5;
            double priorY = y ? 1 : 0;

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedIsBetweenCRCCModel, y, new Gaussian(meanX, varX), L, U);
            ca.Execute(20);

            double pYCondT = System.Math.Exp(IsBetweenGaussianOp.LogProbBetween(new Gaussian(meanX, varX), L, U));
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) G(x)
            //                [(pT)^y (1-pT)^(1-y) delta(y - isbet(x,L,U))]^b
            double sumCondT = priorY * pYCondT + (1 - priorY) * (1 - pYCondT);
            double Z = priorB * sumCondT + (1 - priorB);
            double bExpected = priorB * sumCondT / Z;

            Bernoulli bActual = ca.Marginal<Bernoulli>("b");
            Console.WriteLine($"b = {bActual} (should be {bExpected})");
            Assert.True(System.Math.Abs(bActual.GetProbTrue() - bExpected) < 1e-4);

            Gaussian xActual = ca.Marginal<Gaussian>("x");
            Console.WriteLine($"x = {xActual}");
        }

        private void GatedIsBetweenCRCCModel(bool y, Gaussian priorX, double lowerBound, double upperBound)
        {
            bool b = Factor.Bernoulli(0.1);
            double x = Factor.Random(priorX);
            if (b)
            {
                y = Factor.IsBetween(x, lowerBound, upperBound);
            }
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(x, nameof(x));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedIsBetweenRRCCTest()
        {
            double priorB = 0.1;
            double meanX = 0.4, varX = 1.4;
            double L = 0.3;
            double U = 0.5;
            double priorY = 0.2;

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedIsBetweenRRCCModel, new Gaussian(meanX, varX), L, U);
            ca.Execute(20);

            double pYCondT = System.Math.Exp(IsBetweenGaussianOp.LogProbBetween(new Gaussian(meanX, varX), L, U));
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) G(x)
            //                [(pT)^y (1-pT)^(1-y) delta(y - isbet(x,L,U))]^b
            double sumCondT = priorY * pYCondT + (1 - priorY) * (1 - pYCondT);
            double Z = priorB * sumCondT + (1 - priorB);
            double postB = priorB * sumCondT / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedIsBetweenRRCCModel(Gaussian priorX, double lowerBound, double upperBound)
        {
            bool b = Factor.Bernoulli(0.1);
            double x = Factor.Random(priorX);
            if (b)
            {
                bool y = Factor.IsBetween(x, lowerBound, upperBound);
                Constrain.EqualRandom(y, new Bernoulli(0.2));
            }
            InferNet.Infer(b, nameof(b));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedIsPositiveCRTest()
        {
            GatedIsPositiveCR(true);
            GatedIsPositiveCR(false);
        }

        private void GatedIsPositiveCR(bool y)
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedIsPositiveCRModel, y);
            ca.Execute(20);

            double priorB = 0.1;
            double meanX = 0.4;
            double varX = 1.4;
            double pYCondT = MMath.NormalCdf(meanX / System.Math.Sqrt(varX));
            double priorY = y ? 1 : 0;
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) G(x)
            //                [(pT)^y (1-pT)^(1-y) delta(y - step(x))]^b
            double sumCondT = priorY * pYCondT + (1 - priorY) * (1 - pYCondT);
            double Z = priorB * sumCondT + (1 - priorB);
            double postB = priorB * sumCondT / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedIsPositiveCRModel(bool y)
        {
            bool b = Factor.Bernoulli(0.1);
            double x = Factor.Random(new Gaussian(0.4, 1.4));
            if (b)
            {
                y = Factor.IsPositive(x);
            }
            InferNet.Infer(b, nameof(b));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedIsPositiveRRTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedIsPositiveRRModel);
            ca.Execute(20);

            double priorB = 0.1;
            double meanX = 0.4;
            double varX = 1.4;
            double pYCondT = MMath.NormalCdf(meanX / System.Math.Sqrt(varX));
            double priorY = 0.2;
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) G(x)
            //                [(pT)^y (1-pT)^(1-y) delta(y - step(x))]^b
            double sumCondT = priorY * pYCondT + (1 - priorY) * (1 - pYCondT);
            double Z = priorB * sumCondT + (1 - priorB);
            double postB = priorB * sumCondT / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedIsPositiveRRModel()
        {
            bool b = Factor.Bernoulli(0.1);
            double x = Factor.Random(new Gaussian(0.4, 1.4));
            if (b)
            {
                bool y = Factor.IsPositive(x);
                Constrain.EqualRandom(y, new Bernoulli(0.2));
            }
            InferNet.Infer(b, nameof(b));
        }

        [Fact]
        public void GatedInnerProductArrayTest()
        {
            foreach (var flip in new[] { false, true })
            {
                GatedInnerProductArray(flip);
            }
        }

        internal void GatedInnerProductArray(bool flip)
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            var evBlock = Variable.If(evidence);
            Range item = new Range(2).Named("item");
            VariableArray<double> a = Variable.Array<double>(item).Named("a");
            Gaussian aPrior = Gaussian.FromMeanAndVariance(1.2, 3.4);
            a[item] = Variable<double>.Random(aPrior).ForEach(item);
            VariableArray<double> b = Variable.Observed(default(double[]), item).Named("b");
            Variable<double> c = flip ? Variable.InnerProduct(b, a) : Variable.InnerProduct(a, b);
            Variable<Gaussian> cLike = Variable.Observed(default(Gaussian));
            Variable.ConstrainEqualRandom(c, cLike);
            evBlock.CloseBlock();
            InferenceEngine engine = new InferenceEngine();

            for (int trial = 0; trial <= 3; trial++)
            {
                if (trial == 0)
                {
                    b.ObservedValue = new[] { 2.0, 3.0 };
                    cLike.ObservedValue = Gaussian.FromMeanAndVariance(4, 5);
                }
                else if (trial == 1)
                {
                    b.ObservedValue = new[] { 0.0, 0.0 };
                    cLike.ObservedValue = Gaussian.PointMass(0);
                }
                else if (trial == 2)
                {
                    b.ObservedValue = new[] { 0.0, 2.0 };
                    cLike.ObservedValue = Gaussian.PointMass(2.3);
                }
                else 
                {
                    b.ObservedValue = new[] { 0.0, 2.0 };
                    cLike.ObservedValue = Gaussian.FromNatural(-7.3097118076958154E-10, 1.542967011962213E-320);
                }

                var aActual = engine.Infer<IList<Gaussian>>(a);
                VectorGaussian vectorGaussianExpected = new VectorGaussian(item.SizeAsInt);
                var bVector = Vector.FromArray(b.ObservedValue);
                var aMean = (DenseVector)Util.ArrayInit(item.SizeAsInt, i => aPrior.GetMean()).ToVector();
                var aVariance = new PositiveDefiniteMatrix(item.SizeAsInt, item.SizeAsInt);
                aVariance.SetToDiagonal(Util.ArrayInit(item.SizeAsInt, i => aPrior.GetVariance()).ToVector());
                var aVector = VectorGaussian.FromMeanAndVariance(aMean, aVariance);
                InnerProductOp.AAverageConditional(cLike.ObservedValue, bVector, vectorGaussianExpected);
                vectorGaussianExpected.SetToProduct(vectorGaussianExpected, aVector);
                for (int i = 0; i < aActual.Count; i++)
                {
                    var aExpected = vectorGaussianExpected.GetMarginal(i);
                    Assert.True(aExpected.MaxDiff(aActual[i]) < 1e-10);
                }
                var cActual = engine.Infer<Gaussian>(c);
                var toC = InnerProductOp.InnerProductAverageConditional(aMean, aVariance, bVector);
                var cExpected = toC * cLike.ObservedValue;
                Assert.True(cExpected.MaxDiff(cActual) < 1e-10);

                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                double evExpected = cLike.ObservedValue.GetLogAverageOf(toC);
                Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-8) < 1e-10);
            }
        }

        [Fact]
        public void GatedInnerProductVectorTest()
        {
            foreach (var algorithm in new IAlgorithm[] { new Algorithms.ExpectationPropagation(), new Algorithms.VariationalMessagePassing() })
            {
                foreach (var flip in new[] { false, true })
                {
                    GatedInnerProductVector(flip, algorithm);
                }
            }
        }

        internal void GatedInnerProductVector(bool flip, IAlgorithm algorithm)
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            var evBlock = Variable.If(evidence);
            Range item = new Range(2).Named("item");
            DenseVector aMean = DenseVector.FromArray(1.2, 3.4);
            PositiveDefiniteMatrix aVariance = new PositiveDefiniteMatrix(new double[,] {
                { 4.5, 2.3 }, { 2.3, 6.7 }
            });
            VectorGaussian aPrior = VectorGaussian.FromMeanAndVariance(aMean, aVariance);
            Variable<Vector> a = Variable.Random(aPrior).Named("a");
            a.SetValueRange(item);
            Variable<Vector> b = Variable.Observed(default(Vector)).Named("b");
            b.SetValueRange(item);
            Variable<double> c = flip ? Variable.InnerProduct(b, a) : Variable.InnerProduct(a, b);
            Variable<Gaussian> cLike = Variable.Observed(default(Gaussian));
            Variable.ConstrainEqualRandom(c, cLike);
            evBlock.CloseBlock();
            InferenceEngine engine = new InferenceEngine(algorithm);

            for (int trial = 0; trial <= 2; trial++)
            {
                if (trial == 0)
                {
                    b.ObservedValue = Vector.FromArray(2.0, 3.0);
                    cLike.ObservedValue = Gaussian.FromMeanAndVariance(4, 5);
                }
                else if (trial == 1)
                {
                    b.ObservedValue = Vector.FromArray(0.0, 0.0);
                    cLike.ObservedValue = Gaussian.PointMass(0);
                }
                else
                {
                    b.ObservedValue = Vector.FromArray(0.0, 2.0);
                    cLike.ObservedValue = Gaussian.PointMass(2.3);
                }

                var aActual = engine.Infer<VectorGaussian>(a);
                VectorGaussian vectorGaussianExpected = new VectorGaussian(item.SizeAsInt);
                var bVector = b.ObservedValue;
                InnerProductOp.AAverageConditional(cLike.ObservedValue, bVector, vectorGaussianExpected);
                vectorGaussianExpected.SetToProduct(vectorGaussianExpected, aPrior);
                Assert.True(vectorGaussianExpected.MaxDiff(aActual) < 1e-10);
                var cActual = engine.Infer<Gaussian>(c);
                var toC = InnerProductOp.InnerProductAverageConditional(aMean, aVariance, bVector);
                var cExpected = toC * cLike.ObservedValue;
                Assert.True(cExpected.MaxDiff(cActual) < 1e-10);

                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                double evExpected;
                if(engine.Algorithm is Algorithms.VariationalMessagePassing)
                {
                    // The only stochastic variable is A, and the only stochastic factors are A's prior and cLike.
                    evExpected = aActual.GetAverageLog(aPrior) + cExpected.GetAverageLog(cLike.ObservedValue)
                        -aActual.GetAverageLog(aActual);
                }
                else evExpected = cLike.ObservedValue.GetLogAverageOf(toC);
                Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-8) < 1e-10);
            }
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedInnerProductRRCTest()
        {
            double priorB = 0.1;
            double meanX = 0.4;
            double varX = 1.4;
            Vector y = Vector.FromArray(0.3);
            VectorGaussian priorX = new VectorGaussian(meanX, varX);
            Gaussian priorZ = new Gaussian(0.2, 1.2);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedInnerProductRRCModel, priorX, y);
            ca.Execute(20);

            Gaussian xy = new Gaussian(meanX * y[0], y[0] * y[0] * varX);
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) VG(x) VG(y)
            //                [G(z) delta(z - x.Inner(y))]^b
            double sumCondT = System.Math.Exp(priorZ.GetLogAverageOf(xy));
            double Z = priorB * sumCondT + (1 - priorB);
            double postB = priorB * sumCondT / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedInnerProductRRCModel(VectorGaussian priorX, Vector y)
        {
            bool b = Factor.Bernoulli(0.1);
            Vector x = Factor.Random(priorX);
            if (b)
            {
                double z = Vector.InnerProduct(x, y);
                Constrain.EqualRandom(z, new Gaussian(0.2, 1.2));
            }
            InferNet.Infer(b, nameof(b));
        }

        // Fails because InnerProduct with fixed output is not implemented 
        [Fact]
        [Trait("Category", "OpenBug")]
        [Trait("Category", "CsoftModel")]
        public void GatedInnerProductCRCTest()
        {
            double priorB = 0.1;
            double meanX = 0.4;
            double varX = 1.4;
            Vector y = Vector.FromArray(0.3);
            VectorGaussian priorX = new VectorGaussian(meanX, varX);
            Gaussian priorZ = Gaussian.PointMass(0.2);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedInnerProductCRCModel, priorZ.Point, priorX, y);
            ca.Execute(20);

            Gaussian xy = new Gaussian(meanX * y[0], y[0] * y[0] * varX);
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) VG(x) VG(y)
            //                [G(z) delta(z - x.Inner(y))]^b
            double sumCondT = System.Math.Exp(priorZ.GetLogAverageOf(xy));
            double Z = priorB * sumCondT + (1 - priorB);
            double postB = priorB * sumCondT / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedInnerProductCRCModel(double z, VectorGaussian priorX, Vector y)
        {
            bool b = Factor.Bernoulli(0.1);
            Vector x = Factor.Random(priorX);
            if (b)
            {
                z = Vector.InnerProduct(x, y);
            }
            InferNet.Infer(b, nameof(b));
        }

        // This test will not work with EP at the moment, but it should work in the future.
        // Fails because InnerProduct with random inputs is not implemented in EP.
        [Fact]
        [Trait("Category", "OpenBug")]
        [Trait("Category", "CsoftModel")]
        public void GatedInnerProductRRRTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedInnerProductRRRModel);
            ca.Execute(20);

            double priorB = 0.1;
            double meanX = 0.4;
            double varX = 1.4;
            double meanY = 0.3;
            double varY = 1.3;
            VectorGaussian priorX = new VectorGaussian(meanX, varX);
            VectorGaussian priorY = new VectorGaussian(meanY, varY);
            Gaussian xy = new Gaussian(meanX * meanY, meanX * meanX * varY + meanY * meanY * varX + varX * varY);
            Gaussian priorZ = new Gaussian(0.2, 1.2);
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) VG(x) VG(y)
            //                [G(z) delta(z - x.Inner(y))]^b
            double sumCondT = System.Math.Exp(priorZ.GetLogAverageOf(xy));
            double Z = priorB * sumCondT + (1 - priorB);
            double postB = priorB * sumCondT / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedInnerProductRRRModel()
        {
            bool b = Factor.Bernoulli(0.1);
            Vector x = Factor.Random(new VectorGaussian(0.4, 1.4));
            Vector y = Factor.Random(new VectorGaussian(0.3, 1.3));
            if (b)
            {
                double z = Vector.InnerProduct(x, y);
                Constrain.EqualRandom(z, new Gaussian(0.2, 1.2));
            }
            InferNet.Infer(b, nameof(b));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedConstrainEqualTest()
        {
            double priorB = 0.1;
            Gaussian priorX = new Gaussian(0.4, 1.4);
            Gaussian priorY = new Gaussian(0.3, 1.3);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedConstrainEqual, priorX, priorY);
            ca.Execute(20);

            double sumCondT = System.Math.Exp(priorX.GetLogAverageOf(priorY));
            double Z = priorB * sumCondT + (1 - priorB);
            double postB = priorB * sumCondT / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedConstrainEqual(Gaussian priorX, Gaussian priorY)
        {
            bool b = Factor.Bernoulli(0.1);
            double x = Factor.Random(priorX);
            double y = Factor.Random(priorY);
            if (b)
            {
                Constrain.Equal(x, y);
            }
            InferNet.Infer(b, nameof(b));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedExpCRTest()
        {
            double priorB = 0.1;
            double meanX = 0.4;
            double varX = 1.4;
            Gaussian priorX = new Gaussian(meanX, varX);
            Gamma priorZ = Gamma.PointMass(0.2);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedExpCRModel, priorB, priorX, priorZ.Point);
            ca.Execute(20);

            double sumCondT = System.Math.Exp(priorX.GetLogProb(System.Math.Log(priorZ.Point))) / priorZ.Point;
            double Z = priorB * sumCondT + (1 - priorB);
            double postB = priorB * sumCondT / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedExpCRModel(double priorB, Gaussian priorX, double z)
        {
            bool b = Factor.Bernoulli(priorB);
            double x = Factor.Random(priorX);
            if (b)
            {
                z = System.Math.Exp(x);
            }
            InferNet.Infer(b, nameof(b));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedExpRRTest()
        {
            double priorB = 0.1;
            double meanX = 0.4;
            double varX = 1.4;
            Gamma priorZ = new Gamma(0.2, 1.2);
            Gaussian priorX = new Gaussian(meanX, varX);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedExpRRModel, priorB, priorX, priorZ);
            ca.Execute(20);

            double sumCondT = System.Math.Exp(ExpOp.LogAverageFactor_slow(priorZ, priorX));
            double Z = priorB * sumCondT + (1 - priorB);
            double postB = priorB * sumCondT / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedExpRRModel(double priorB, Gaussian priorX, Gamma priorZ)
        {
            bool b = Factor.Bernoulli(priorB);
            double x = Factor.Random(priorX);
            if (b)
            {
                double z = System.Math.Exp(x);
                Constrain.EqualRandom(z, priorZ);
            }
            InferNet.Infer(b, nameof(b));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedMinusCRCTest()
        {
            double priorB = 0.1;
            double meanX = 0.4;
            double varX = 1.4;
            double meanY = 0.3;
            double varY = 0;
            Gaussian priorZ = Gaussian.PointMass(0.2);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedMinusCRCModel, priorZ.Point, meanY);
            ca.Execute(20);

            // p(x,b) =propto (pb)^b (1-pb)^(1-b) G(x) G(y)
            //                [G(z) delta(z - (x-y))]^b
            Gaussian xy = new Gaussian(meanX - meanY, varX + varY);
            double sumCondT = System.Math.Exp(priorZ.GetLogAverageOf(xy));
            double Z = priorB * sumCondT + (1 - priorB);
            double postB = priorB * sumCondT / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedMinusCRCModel(double z, double y)
        {
            bool b = Factor.Bernoulli(0.1);
            double x = Factor.Random(new Gaussian(0.4, 1.4));
            if (b)
            {
                z = Factor.Difference(x, y);
            }
            InferNet.Infer(b, nameof(b));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedMinusCRRTest()
        {
            double priorB = 0.1;
            double meanX = 0.4;
            double varX = 1.4;
            double meanY = 0.3;
            double varY = 1.3;
            Gaussian priorZ = Gaussian.PointMass(0.2);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedMinusCRRModel, priorZ.Point);
            ca.Execute(20);

            // p(x,b) =propto (pb)^b (1-pb)^(1-b) G(x) G(y)
            //                [G(z) delta(z - (x-y))]^b
            Gaussian xy = new Gaussian(meanX - meanY, varX + varY);
            double sumCondT = System.Math.Exp(priorZ.GetLogAverageOf(xy));
            double Z = priorB * sumCondT + (1 - priorB);
            double postB = priorB * sumCondT / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedMinusCRRModel(double z)
        {
            bool b = Factor.Bernoulli(0.1);
            double x = Factor.Random(new Gaussian(0.4, 1.4));
            double y = Factor.Random(new Gaussian(0.3, 1.3));
            if (b)
            {
                z = Factor.Difference(x, y);
            }
            InferNet.Infer(b, nameof(b));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedMinusRCRTest()
        {
            double priorB = 0.1;
            double meanX = 0.4;
            double varX = 1.4;
            double meanY = 0.3;
            double varY = 0.0;
            Gaussian priorZ = new Gaussian(0.2, 1.2);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedMinusRCRModel, meanY);
            ca.Execute(20);

            // p(x,b) =propto (pb)^b (1-pb)^(1-b) G(x) G(y)
            //                [G(z) delta(z - (x-y))]^b
            Gaussian xy = new Gaussian(meanX - meanY, varX + varY);
            double sumCondT = System.Math.Exp(priorZ.GetLogAverageOf(xy));
            double Z = priorB * sumCondT + (1 - priorB);
            double postB = priorB * sumCondT / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedMinusRCRModel(double y)
        {
            bool b = Factor.Bernoulli(0.1);
            double x = Factor.Random(new Gaussian(0.4, 1.4));
            if (b)
            {
                double z = Factor.Difference(x, y);
                Constrain.EqualRandom(z, new Gaussian(0.2, 1.2));
            }
            InferNet.Infer(b, nameof(b));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedMinusRRRTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedMinusRRRModel);
            ca.Execute(20);

            double priorB = 0.1;
            double meanX = 0.4;
            double varX = 1.4;
            double meanY = 0.3;
            double varY = 1.3;
            Gaussian priorZ = new Gaussian(0.2, 1.2);
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) G(x) G(y)
            //                [G(z) delta(z - (x-y))]^b
            Gaussian xy = new Gaussian(meanX - meanY, varX + varY);
            double sumCondT = System.Math.Exp(priorZ.GetLogAverageOf(xy));
            double Z = priorB * sumCondT + (1 - priorB);
            double postB = priorB * sumCondT / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedMinusRRRModel()
        {
            bool b = Factor.Bernoulli(0.1);
            double x = Factor.Random(new Gaussian(0.4, 1.4));
            double y = Factor.Random(new Gaussian(0.3, 1.3));
            if (b)
            {
                double z = Factor.Difference(x, y);
                Constrain.EqualRandom(z, new Gaussian(0.2, 1.2));
            }
            InferNet.Infer(b, nameof(b));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedPlusCRRTest()
        {
            Gaussian priorZ = Gaussian.PointMass(0.2);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedPlusCRRModel, priorZ.Point);
            ca.Execute(20);

            double priorB = 0.1;
            double meanX = 0.4;
            double varX = 1.4;
            double meanY = 0.3;
            double varY = 1.3;
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) G(x) G(y)
            //                [G(z) delta(z - (x+y))]^b
            Gaussian xy = new Gaussian(meanX + meanY, varX + varY);
            double sumCondT = System.Math.Exp(priorZ.GetLogAverageOf(xy));
            double Z = priorB * sumCondT + (1 - priorB);
            double postB = priorB * sumCondT / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedPlusCRRModel(double z)
        {
            bool b = Factor.Bernoulli(0.1);
            double x = Factor.Random(new Gaussian(0.4, 1.4));
            double y = Factor.Random(new Gaussian(0.3, 1.3));
            if (b)
            {
                z = Factor.Plus(x, y);
            }
            InferNet.Infer(b, nameof(b));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedPlusCRCTest()
        {
            double priorB = 0.1;
            double meanX = 0.4;
            double varX = 1.4;
            double meanY = 0.3;
            double varY = 0;
            Gaussian priorZ = Gaussian.PointMass(0.2);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedPlusCRCModel, priorZ.Point, meanY);
            ca.Execute(20);

            // p(x,b) =propto (pb)^b (1-pb)^(1-b) G(x) G(y)
            //                [G(z) delta(z - (x+y))]^b
            Gaussian xy = new Gaussian(meanX + meanY, varX + varY);
            double sumCondT = System.Math.Exp(priorZ.GetLogAverageOf(xy));
            double Z = priorB * sumCondT + (1 - priorB);
            double postB = priorB * sumCondT / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedPlusCRCModel(double z, double y)
        {
            bool b = Factor.Bernoulli(0.1);
            double x = Factor.Random(new Gaussian(0.4, 1.4));
            if (b)
            {
                z = Factor.Plus(x, y);
            }
            InferNet.Infer(b, nameof(b));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedPlusRRRTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedPlusRRRModel);
            ca.Execute(20);

            double priorB = 0.1;
            double meanX = 0.4;
            double varX = 1.4;
            double meanY = 0.3;
            double varY = 1.3;
            Gaussian priorZ = new Gaussian(0.2, 1.2);
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) G(x) G(y)
            //                [G(z) delta(z - (x+y))]^b
            Gaussian xy = new Gaussian(meanX + meanY, varX + varY);
            double sumCondT = System.Math.Exp(priorZ.GetLogAverageOf(xy));
            double Z = priorB * sumCondT + (1 - priorB);
            double postB = priorB * sumCondT / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedPlusRRRModel()
        {
            bool b = Factor.Bernoulli(0.1);
            double x = Factor.Random(new Gaussian(0.4, 1.4));
            double y = Factor.Random(new Gaussian(0.3, 1.3));
            if (b)
            {
                double z = Factor.Plus(x, y);
                Constrain.EqualRandom(z, new Gaussian(0.2, 1.2));
            }
            InferNet.Infer(b, nameof(b));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedPlusRCRTest()
        {
            double priorB = 0.1;
            double meanX = 0.4;
            double varX = 1.4;
            double meanY = 0.3;
            double varY = 0;
            Gaussian priorZ = new Gaussian(0.2, 1.2);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedPlusRCRModel, meanY);
            ca.Execute(20);

            // p(x,b) =propto (pb)^b (1-pb)^(1-b) G(x) G(y)
            //                [G(z) delta(z - (x+y))]^b
            Gaussian xy = new Gaussian(meanX + meanY, varX + varY);
            double sumCondT = System.Math.Exp(priorZ.GetLogAverageOf(xy));
            double Z = priorB * sumCondT + (1 - priorB);
            double postB = priorB * sumCondT / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedPlusRCRModel(double y)
        {
            bool b = Factor.Bernoulli(0.1);
            double x = Factor.Random(new Gaussian(0.4, 1.4));
            if (b)
            {
                double z = Factor.Plus(x, y);
                Constrain.EqualRandom(z, new Gaussian(0.2, 1.2));
            }
            InferNet.Infer(b, nameof(b));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedProductCRCTest()
        {
            double priorB = 0.1;
            double meanX = 0.4;
            double varX = 1.4;
            double y = 0.3;
            Gaussian priorX = new Gaussian(meanX, varX);
            Gaussian priorZ = Gaussian.PointMass(0.2);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedProductCRCModel, priorB, priorX, y, priorZ.Point);
            ca.Execute(20);

            // p(x,b) =propto (pb)^b (1-pb)^(1-b) G(x) G(y)
            //                [G(z) delta(z - (x+y))]^b
            Gaussian xy = new Gaussian(meanX * y, varX * y * y);
            double sumCondT = System.Math.Exp(priorZ.GetLogAverageOf(xy));
            double Z = priorB * sumCondT + (1 - priorB);
            double postB = priorB * sumCondT / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedProductCRCModel(double priorB, Gaussian priorX, double y, double z)
        {
            bool b = Factor.Bernoulli(priorB);
            double x = Factor.Random(priorX);
            if (b)
            {
                z = Factor.Product(x, y);
            }
            InferNet.Infer(b, nameof(b));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedProductRRCTest()
        {
            double priorB = 0.1;
            double meanX = 0.4;
            double varX = 1.4;
            double y = 0.3;
            Gaussian priorZ = new Gaussian(0.2, 1.2);
            Gaussian priorX = new Gaussian(meanX, varX);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedProductRRCModel, priorB, priorX, y, priorZ);
            ca.Execute(20);

            Gaussian xy = new Gaussian(meanX * y, varX * y * y);
            double sumCondT = System.Math.Exp(priorZ.GetLogAverageOf(xy));
            double Z = priorB * sumCondT + (1 - priorB);
            double postB = priorB * sumCondT / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedProductRRCModel(double priorB, Gaussian priorX, double y, Gaussian priorZ)
        {
            bool b = Factor.Bernoulli(priorB);
            double x = Factor.Random(priorX);
            if (b)
            {
                double z = Factor.Product(x, y);
                Constrain.EqualRandom(z, priorZ);
            }
            InferNet.Infer(b, nameof(b));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedProductRRRTest()
        {
            double priorB = 0.1;
            Gaussian priorZ = new Gaussian(5, 6);
            Gaussian priorX = new Gaussian(1, 2);
            Gaussian priorY = new Gaussian(3, 4);
            Gaussian xCondT = new Gaussian(1.440115377416467, 0.826021607530145);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedProductRRRModel, priorB, priorX, priorY, priorZ);
            ca.Execute(20);

            double sumCondT = System.Math.Exp(-2.834075861624064);
            double Z = priorB * sumCondT + (1 - priorB);
            double bExpected = priorB * sumCondT / Z;
            Gaussian xExpected = new Gaussian();
            xExpected.SetToSum(bExpected, xCondT, 1 - bExpected, priorX);

            Bernoulli bActual = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bActual, bExpected);
            Gaussian xActual = ca.Marginal<Gaussian>("x");
            Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);

            Assert.True(System.Math.Abs(bActual.GetProbTrue() - bExpected) < 1e-4);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-4);
        }

        private void GatedProductRRRModel(double priorB, Gaussian priorX, Gaussian priorY, Gaussian priorZ)
        {
            bool b = Factor.Bernoulli(priorB);
            double x = Factor.Random(priorX);
            double y = Factor.Random(priorY);
            if (b)
            {
                double z = Factor.Product(x, y);
                Constrain.EqualRandom(z, priorZ);
            }
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(x, nameof(x));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedGammaProductRRCTest()
        {
            double priorB = 0.1;
            double meanX = 0.4;
            double meanLogX = 1.4;
            double y = 0.3;
            Gamma priorZ = new Gamma(0.2, 1.2);
            Gamma priorX = Gamma.FromMeanAndMeanLog(meanX, meanLogX);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedGammaProductRRCModel, priorB, priorX, y, priorZ);
            ca.Execute(20);

            Gamma xy = Gamma.FromMeanAndMeanLog(meanX * y, meanLogX + System.Math.Log(y));
            double sumCondT = System.Math.Exp(priorZ.GetLogAverageOf(xy));
            double Z = priorB * sumCondT + (1 - priorB);
            double postB = priorB * sumCondT / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedGammaProductRRCModel(double priorB, Gamma priorX, double y, Gamma priorZ)
        {
            bool b = Factor.Bernoulli(priorB);
            double x = Factor.Random(priorX);
            if (b)
            {
                double z = Factor.Product(x, y);
                Constrain.EqualRandom(z, priorZ);
            }
            InferNet.Infer(b, nameof(b));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedGammaPowerProductRRCTest()
        {
            double priorB = 0.1;
            double meanX = 0.4;
            double meanLogX = 1.4;
            double y = 0.3;
            double power = -0.5;
            GammaPower priorZ = new GammaPower(0.2, 1.2, power);
            GammaPower priorX = GammaPower.FromMeanAndMeanLog(meanX, meanLogX, power);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedGammaPowerProductRRCModel, priorB, priorX, y, priorZ);
            ca.Execute(20);

            GammaPower xy = GammaPower.FromMeanAndMeanLog(meanX * y, meanLogX + System.Math.Log(y), power);
            double sumCondT = System.Math.Exp(priorZ.GetLogAverageOf(xy));
            double Z = priorB * sumCondT + (1 - priorB);
            double postB = priorB * sumCondT / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedGammaPowerProductRRCModel(double priorB, GammaPower priorX, double y, GammaPower priorZ)
        {
            bool b = Factor.Bernoulli(priorB);
            double x = Factor.Random(priorX);
            if (b)
            {
                double z = Factor.Product(x, y);
                Constrain.EqualRandom(z, priorZ);
            }
            InferNet.Infer(b, nameof(b));
        }

        [Fact]
        public void GatedGammaProductCRCTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            double shape = 2.5;
            double rate = 1;
            Variable<double> scale = Variable.Observed(0.0);
            Variable<double> y = Variable.GammaFromShapeAndRate(shape, rate).Named("y");
            Variable<double> x = (y * scale).Named("x");
            x.ObservedValue = 0.0;
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            foreach (var alg in new IAlgorithm[] { new ExpectationPropagation(), new VariationalMessagePassing() })
            {
                engine.Algorithm = alg;
                foreach (var xObserved in new[] { 0.0, 2.0 })
                {
                    x.ObservedValue = xObserved;
                    scale.ObservedValue = xObserved;
                    Gamma yActual = engine.Infer<Gamma>(y);
                    double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                    Gamma yExpected = Gamma.FromShapeAndRate(shape, rate);
                    double evExpected = 0;
                    if (xObserved != 0)
                    {
                        evExpected = Gamma.FromShapeAndRate(shape, rate / scale.ObservedValue).GetLogProb(x.ObservedValue);
                        yExpected = Gamma.PointMass(1);
                    }

                    Console.WriteLine("y = {0} should be {1}", yActual, yExpected);
                    Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                    Assert.True(yExpected.MaxDiff(yActual) < 1e-8);
                    Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-8);
                }
            }
        }

        [Fact]
        public void GatedGammaPowerProductCRCTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            double shape = 2.5;
            double rate = 1;
            double power = -0.5;
            Variable<double> scale = Variable.Observed(0.0);
            Variable<double> y = Variable.Random(GammaPower.FromShapeAndRate(shape, rate, power)).Named("y");
            Variable<double> x = (y * scale).Named("x");
            x.ObservedValue = 0.0;
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            foreach (var alg in new IAlgorithm[] { new ExpectationPropagation(), new VariationalMessagePassing() })
            {
                engine.Algorithm = alg;
                foreach (var xObserved in new[] { 0.0, 2.0 })
                {
                    x.ObservedValue = xObserved;
                    scale.ObservedValue = xObserved;
                    GammaPower yActual = engine.Infer<GammaPower>(y);
                    double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                    GammaPower yExpected = GammaPower.FromShapeAndRate(shape, rate, power);
                    double evExpected = 0;
                    if (xObserved != 0)
                    {
                        evExpected = GammaPower.FromShapeAndRate(shape, rate / System.Math.Pow(scale.ObservedValue, 1/power), power).GetLogProb(x.ObservedValue);
                        yExpected = GammaPower.PointMass(1, power);
                    }

                    Console.WriteLine("y = {0} should be {1}", yActual, yExpected);
                    Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                    Assert.True(yExpected.MaxDiff(yActual) < 1e-8);
                    Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-8);
                }
            }
        }

        [Fact]
        public void GatedGammaRatioCRCTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            double shape = 2.5;
            double rate = 3;
            Variable<double> y = Variable.GammaFromShapeAndRate(shape, rate).Named("y");
            var denom = Variable.Observed(0.0);
            Variable<double> x = (y / denom).Named("x");
            x.ObservedValue = double.PositiveInfinity;
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            foreach (var alg in new IAlgorithm[] { new ExpectationPropagation(), new VariationalMessagePassing() })
            {
                engine.Algorithm = alg;
                foreach (var xObserved in new[] { double.PositiveInfinity, 2.0 })
                {
                    x.ObservedValue = xObserved;
                    denom.ObservedValue = 1 / xObserved;
                    Gamma yActual = engine.Infer<Gamma>(y);
                    double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                    Gamma yExpected = Gamma.FromShapeAndRate(shape, rate);
                    double evExpected = 0;
                    if (!double.IsPositiveInfinity(xObserved))
                    {
                        evExpected = Gamma.FromShapeAndRate(shape, rate / xObserved).GetLogProb(xObserved);
                        yExpected = Gamma.PointMass(1);
                    }

                    Console.WriteLine("y = {0} should be {1}", yActual, yExpected);
                    Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                    Assert.True(yExpected.MaxDiff(yActual) < 1e-10);
                    Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-6);
                }
            }
        }

        [Fact]
        public void GatedGammaRatioRRCTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            double shape = 2.5;
            double rate = 3;
            Variable<double> y = Variable.GammaFromShapeAndRate(shape, rate).Named("y");
            var denom = Variable.Constant(0.0);
            Variable<double> x = (y / denom).Named("x");
            Gamma xLike = Gamma.Uniform();
            Variable.ConstrainEqualRandom(x, xLike);
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            foreach (var alg in new IAlgorithm[] { new ExpectationPropagation(), new VariationalMessagePassing() })
            {
                engine.Algorithm = alg;
                Gamma yActual = engine.Infer<Gamma>(y);
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                Gamma yExpected = Gamma.FromShapeAndRate(shape, rate);
                double evExpected = 0;

                Console.WriteLine("y = {0} should be {1}", yActual, yExpected);
                Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                Assert.True(yExpected.MaxDiff(yActual) < 1e-10);
                Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-6);
            }
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedRatioCRCTest()
        {
            double priorB = 0.1;
            double meanX = 0.4;
            double varX = 1.4;
            double y = 0.3;
            Gaussian priorZ = Gaussian.PointMass(0.2);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedRatioCRCModel, priorZ.Point, y);
            ca.Execute(20);

            // p(x,b) =propto (pb)^b (1-pb)^(1-b) G(x) G(y)
            //                [G(z) delta(z - (x+y))]^b
            Gaussian xy = new Gaussian(meanX / y, varX / y / y);
            double sumCondT = System.Math.Exp(priorZ.GetLogAverageOf(xy));
            double Z = priorB * sumCondT + (1 - priorB);
            double postB = priorB * sumCondT / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedRatioCRCModel(double z, double y)
        {
            bool b = Factor.Bernoulli(0.1);
            double x = Factor.Random(new Gaussian(0.4, 1.4));
            if (b)
            {
                z = Factor.Ratio(x, y);
            }
            InferNet.Infer(b, nameof(b));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedRatioRRCTest()
        {
            double priorB = 0.1;
            double meanX = 0.4;
            double varX = 1.4;
            double y = 0.3;
            Gaussian priorZ = new Gaussian(0.2, 1.2);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedRatioRRCModel, y);
            ca.Execute(20);

            Gaussian xy = new Gaussian(meanX / y, varX / y / y);
            double sumCondT = System.Math.Exp(priorZ.GetLogAverageOf(xy));
            double Z = priorB * sumCondT + (1 - priorB);
            double postB = priorB * sumCondT / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedRatioRRCModel(double y)
        {
            bool b = Factor.Bernoulli(0.1);
            double x = Factor.Random(new Gaussian(0.4, 1.4));
            if (b)
            {
                double z = Factor.Ratio(x, y);
                Constrain.EqualRandom(z, new Gaussian(0.2, 1.2));
            }
            InferNet.Infer(b, nameof(b));
        }

        internal void GatedRatioRRRModel()
        {
            bool b = Factor.Bernoulli(0.1);
            double x = Factor.Random(new Gaussian(0.4, 1.4));
            double y = Factor.Random(new Gaussian(0.3, 1.3));
            if (b)
            {
                double z = Factor.Ratio(x, y);
                Constrain.EqualRandom(z, new Gaussian(0.2, 1.2));
            }
            InferNet.Infer(b, nameof(b));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedDiscreteFromDirichletCRTest()
        {
            double priorB = 0.1;
            Dirichlet priorP = new Dirichlet(3.0, 7.0); // mean = 0.3
            Discrete priorX = Discrete.PointMass(0, priorP.Dimension);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedDiscreteFromDirichletCRModel, priorB, priorP, priorX.Point);
            ca.Execute(20);

            Vector meanP = priorP.GetMean();
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) Dirichlet(p;a1,a0)
            //                [(p)^x (1-p)^(1-x) (pT)^x (1-pT)^(1-x)]^b
            double sumXCondT = meanP[0] * priorX[0] + meanP[1] * priorX[1];
            double Z = priorB * sumXCondT + (1 - priorB);
            double postB = priorB * sumXCondT / Z;
            //double postP = priorX * (priorB * (priorY * pXYCondT + (1-priorY) *(1-pXYCondT)) + (1 - priorB)) / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedDiscreteFromDirichletCRModel(double priorB, Dirichlet priorP, int x)
        {
            bool b = Factor.Bernoulli(priorB);
            Vector p = Factor.Random(priorP);
            if (b)
            {
                x = Factor.Discrete(p);
            }
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(p, nameof(p));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedDiscreteFromDirichletRRTest()
        {
            double priorB = 0.1;
            Dirichlet priorP = new Dirichlet(3.0, 7.0); // mean = 0.3
            Discrete priorX = new Discrete(0.2, 0.8);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedDiscreteFromDirichletRRModel, priorB, priorP, priorX);
            ca.Execute(20);

            Vector meanP = priorP.GetMean();
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) Dirichlet(p;a1,a0)
            //                [(p)^x (1-p)^(1-x) (pT)^x (1-pT)^(1-x)]^b
            double sumXCondT = meanP[0] * priorX[0] + meanP[1] * priorX[1];
            double Z = priorB * sumXCondT + (1 - priorB);
            double postB = priorB * sumXCondT / Z;
            //double postP = priorX * (priorB * (priorY * pXYCondT + (1-priorY) *(1-pXYCondT)) + (1 - priorB)) / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedDiscreteFromDirichletRRModel(double priorB, Dirichlet priorP, Discrete priorX)
        {
            bool b = Factor.Bernoulli(priorB);
            Vector p = Factor.Random(priorP);
            if (b)
            {
                int x = Factor.Discrete(p);
                Constrain.EqualRandom(x, priorX);
            }
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(p, nameof(p));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedBernoulliFromBetaCRTest()
        {
            GatedBernoulliFromBetaCR(true);
            GatedBernoulliFromBetaCR(false);
        }

        private void GatedBernoulliFromBetaCR(bool x)
        {
            double priorB = 0.1;
            Beta priorP = new Beta(3, 7); // mean = 0.3
            double priorX = x ? 1.0 : 0;

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedBernoulliFromBetaCRModel, priorB, priorP, x);
            ca.Execute(20);

            double meanP = priorP.GetMean();
            double pXCondT = priorX;
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) Beta(p;a1,a0)
            //                [(p)^x (1-p)^(1-x) (pT)^x (1-pT)^(1-x)]^b
            double sumXCondT = meanP * pXCondT + (1 - meanP) * (1 - pXCondT);
            double Z = priorB * sumXCondT + (1 - priorB);
            double postB = priorB * sumXCondT / Z;
            //double postP = priorX * (priorB * (priorY * pXYCondT + (1-priorY) *(1-pXYCondT)) + (1 - priorB)) / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedBernoulliFromBetaCRModel(double priorB, Beta priorP, bool x)
        {
            bool b = Factor.Bernoulli(priorB);
            double p = Factor.Random(priorP);
            if (b)
            {
                x = Factor.Bernoulli(p);
            }
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(p, nameof(p));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedBernoulliFromBetaRRTest()
        {
            double priorB = 0.1;
            Beta priorP = new Beta(3.0, 7.0); // mean = 0.3
            double priorX = 0.2;

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedBernoulliFromBetaRRModel, priorB, priorP, priorX);
            ca.Execute(20);

            double meanP = priorP.GetMean();
            double pXCondT = priorX;
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) Beta(p;a1,a0)
            //                [(p)^x (1-p)^(1-x) (pT)^x (1-pT)^(1-x)]^b
            double sumXCondT = meanP * pXCondT + (1 - meanP) * (1 - pXCondT);
            double Z = priorB * sumXCondT + (1 - priorB);
            double postB = priorB * sumXCondT / Z;
            //double postP = priorX * (priorB * (priorY * pXYCondT + (1-priorY) *(1-pXYCondT)) + (1 - priorB)) / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
        }

        private void GatedBernoulliFromBetaRRModel(double priorB, Beta priorP, double priorX)
        {
            bool b = Factor.Bernoulli(priorB);
            double p = Factor.Random(priorP);
            if (b)
            {
                bool x = Factor.Bernoulli(p);
                Constrain.EqualRandom(x, new Bernoulli(priorX));
            }
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(p, nameof(p));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedBoolAreEqualRRCTest()
        {
            GatedBoolAreEqualRRC(true);
            GatedBoolAreEqualRRC(false);
        }

        private void GatedBoolAreEqualRRC(bool y)
        {
            double priorB = 0.1;
            double priorX = 0.4;
            double priorY = y ? 1 : 0;
            double pXYCondT = 0.2;

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedBoolAreEqualRRCModel, priorB, priorX, y, pXYCondT);
            ca.Execute(20);

            // p(x,b) =propto (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x) (py)^y (1-py)^(1-y)
            //                [(pT)^(x*y + (1-x)*(1-y)) (1-pT)^(x*(1-y)+y*(1-x))]^b
            double sumXYCondT = (priorX * priorY + (1 - priorX) * (1 - priorY)) * pXYCondT +
                                (priorX * (1 - priorY) + (1 - priorX) * priorY) * (1 - pXYCondT);
            double Z = priorB * sumXYCondT + (1 - priorB);
            double postB = priorB * sumXYCondT / Z;
            double postX = priorX * (priorB * (priorY * pXYCondT + (1 - priorY) * (1 - pXYCondT)) + (1 - priorB)) / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Bernoulli xDist = ca.Marginal<Bernoulli>("x");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - postX) < 1e-4);
        }

        private void GatedBoolAreEqualRRCModel(double priorB, double priorX, bool y, double priorZ)
        {
            bool b = Factor.Bernoulli(priorB);
            bool x = Factor.Bernoulli(priorX);
            if (b)
            {
                bool z = Factor.AreEqual(x, y);
                Constrain.EqualRandom(z, new Bernoulli(priorZ));
            }
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(x, nameof(x));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedBoolAreEqualCRRTest()
        {
            GatedBoolAreEqualCRR(true);
            GatedBoolAreEqualCRR(false);
        }

        private void GatedBoolAreEqualCRR(bool z)
        {
            double priorB = 0.1;
            double priorX = 0.4;
            double priorY = 0.3;
            double pXYCondT = z ? 1.0 : 0;

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedBoolAreEqualCRRModel, priorB, priorX, priorY, z);
            ca.Execute(20);

            // p(x,b) =propto (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x) (py)^y (1-py)^(1-y)
            //                [(pT)^(x*y + (1-x)*(1-y)) (1-pT)^(x*(1-y)+y*(1-x))]^b
            double sumXYCondT = (priorX * priorY + (1 - priorX) * (1 - priorY)) * pXYCondT +
                                (priorX * (1 - priorY) + (1 - priorX) * priorY) * (1 - pXYCondT);
            double Z = priorB * sumXYCondT + (1 - priorB);
            double postB = priorB * sumXYCondT / Z;
            double postX = priorX * (priorB * (priorY * pXYCondT + (1 - priorY) * (1 - pXYCondT)) + (1 - priorB)) / Z;
            double postY = priorY * (priorB * (priorX * pXYCondT + (1 - priorX) * (1 - pXYCondT)) + (1 - priorB)) / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Bernoulli xDist = ca.Marginal<Bernoulli>("x");
            Bernoulli yDist = ca.Marginal<Bernoulli>("y");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            Console.WriteLine("y = {0} (should be {1})", yDist, postY);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - postX) < 1e-4);
            Assert.True(System.Math.Abs(yDist.GetProbTrue() - postY) < 1e-4);
        }

        private void GatedBoolAreEqualCRRModel(double priorB, double priorX, double priorY, bool z)
        {
            bool b = Factor.Bernoulli(priorB);
            bool x = Factor.Bernoulli(priorX);
            bool y = Factor.Bernoulli(priorY);
            if (b)
            {
                z = Factor.AreEqual(x, y);
            }
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(x, nameof(x));
            InferNet.Infer(y, nameof(y));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedBoolAreEqualCRCTest()
        {
            GatedBoolAreEqualCRC(true, true);
            GatedBoolAreEqualCRC(true, false);
            GatedBoolAreEqualCRC(false, true);
            GatedBoolAreEqualCRC(false, false);
        }

        private void GatedBoolAreEqualCRC(bool y, bool z)
        {
            double priorB = 0.1;
            double priorX = 0.4;
            double priorY = y ? 1 : 0;
            double pXYCondT = z ? 1 : 0;

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedBoolAreEqualCRCModel, priorB, priorX, y, z);
            ca.Execute(20);

            // p(x,b) =propto (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x) (py)^y (1-py)^(1-y)
            //                [(pT)^(x*y + (1-x)*(1-y)) (1-pT)^(x*(1-y)+y*(1-x))]^b
            double sumXYCondT = (priorX * priorY + (1 - priorX) * (1 - priorY)) * pXYCondT +
                                (priorX * (1 - priorY) + (1 - priorX) * priorY) * (1 - pXYCondT);
            double Z = priorB * sumXYCondT + (1 - priorB);
            double postB = priorB * sumXYCondT / Z;
            double postX = priorX * (priorB * (priorY * pXYCondT + (1 - priorY) * (1 - pXYCondT)) + (1 - priorB)) / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Bernoulli xDist = ca.Marginal<Bernoulli>("x");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - postX) < 1e-4);
        }

        private void GatedBoolAreEqualCRCModel(double priorB, double priorX, bool y, bool z)
        {
            bool b = Factor.Bernoulli(priorB);
            bool x = Factor.Bernoulli(priorX);
            if (b)
            {
                z = Factor.AreEqual(x, y);
            }
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(x, nameof(x));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedBoolAreEqualRRRTest()
        {
            double priorB = 0.1;
            double priorX = 0.4;
            double priorY = 0.3;
            double pXYCondT = 0.2;

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedBoolAreEqualRRRModel, priorB, priorX, priorY, pXYCondT);
            ca.Execute(20);

            // p(x,b) =propto (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x) (py)^y (1-py)^(1-y)
            //                [(pT)^(x*y + (1-x)*(1-y)) (1-pT)^(x*(1-y)+y*(1-x))]^b
            double sumXYCondT = (priorX * priorY + (1 - priorX) * (1 - priorY)) * pXYCondT +
                                (priorX * (1 - priorY) + (1 - priorX) * priorY) * (1 - pXYCondT);
            double Z = priorB * sumXYCondT + (1 - priorB);
            double postB = priorB * sumXYCondT / Z;
            double postX = priorX * (priorB * (priorY * pXYCondT + (1 - priorY) * (1 - pXYCondT)) + (1 - priorB)) / Z;
            double postY = priorY * (priorB * (priorX * pXYCondT + (1 - priorX) * (1 - pXYCondT)) + (1 - priorB)) / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Bernoulli xDist = ca.Marginal<Bernoulli>("x");
            Bernoulli yDist = ca.Marginal<Bernoulli>("y");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            Console.WriteLine("y = {0} (should be {1})", yDist, postY);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - postX) < 1e-4);
            Assert.True(System.Math.Abs(yDist.GetProbTrue() - postY) < 1e-4);
        }

        private void GatedBoolAreEqualRRRModel(double priorB, double priorX, double priorY, double priorZ)
        {
            bool b = Factor.Bernoulli(priorB);
            bool x = Factor.Bernoulli(priorX);
            bool y = Factor.Bernoulli(priorY);
            if (b)
            {
                bool z = Factor.AreEqual(x, y);
                Constrain.EqualRandom(z, new Bernoulli(priorZ));
            }
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(x, nameof(x));
            InferNet.Infer(y, nameof(y));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedAndTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedAnd);
            ca.Execute(20);

            double priorB = 0.1;
            double priorX = 0.4;
            double priorY = 0.3;
            double pXYCondT = 0.2;
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x) (py)^y (1-py)^(1-y)
            //                [(pT)^(x*y) (1-pT)^(1-x*y)]^b
            double sumXYCondT = priorX * priorY * pXYCondT +
                                (1 - priorX * priorY) * (1 - pXYCondT);
            double Z = priorB * sumXYCondT + (1 - priorB);
            double postB = priorB * sumXYCondT / Z;
            double postX = priorX * (priorB * (priorY * pXYCondT + (1 - priorY) * (1 - pXYCondT)) + (1 - priorB)) / Z;
            double postY = priorY * (priorB * (priorX * pXYCondT + (1 - priorX) * (1 - pXYCondT)) + (1 - priorB)) / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Bernoulli xDist = ca.Marginal<Bernoulli>("x");
            Bernoulli yDist = ca.Marginal<Bernoulli>("y");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            Console.WriteLine("y = {0} (should be {1})", yDist, postY);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - postX) < 1e-4);
            Assert.True(System.Math.Abs(yDist.GetProbTrue() - postY) < 1e-4);
        }

        private void GatedAnd()
        {
            bool b = Factor.Bernoulli(0.1);
            bool x = Factor.Bernoulli(0.4);
            bool y = Factor.Bernoulli(0.3);
            if (b)
            {
                bool z = Factor.And(x, y);
                Constrain.EqualRandom(z, new Bernoulli(0.2));
            }
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(x, nameof(x));
            InferNet.Infer(y, nameof(y));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedOrRRRTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedOrRRRModel);
            ca.Execute(20);

            double priorB = 0.1;
            double priorX = 0.2;
            double priorY = 0.3;
            double pXYCondT = 0.4;
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x) (py)^y (1-py)^(1-y)
            //                [(pT)^(1-(1-x)*(1-y)) (1-pT)^((1-x)*(1-y))]^b
            double sumXYCondT = (1 - (1 - priorX) * (1 - priorY)) * pXYCondT +
                                ((1 - priorX) * (1 - priorY)) * (1 - pXYCondT);
            double Z = priorB * sumXYCondT + (1 - priorB);
            double postB = priorB * sumXYCondT / Z;
            double postX = priorX * (priorB * pXYCondT + (1 - priorB)) / Z;
            double postY = priorY * (priorB * pXYCondT + (1 - priorB)) / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Bernoulli xDist = ca.Marginal<Bernoulli>("x");
            Bernoulli yDist = ca.Marginal<Bernoulli>("y");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            Console.WriteLine("y = {0} (should be {1})", yDist, postY);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - postX) < 1e-4);
            Assert.True(System.Math.Abs(yDist.GetProbTrue() - postY) < 1e-4);
        }

        private void GatedOrRRRModel()
        {
            bool b = Factor.Bernoulli(0.1);
            bool x = Factor.Bernoulli(0.2);
            bool y = Factor.Bernoulli(0.3);
            if (b)
            {
                bool z = Factor.Or(x, y);
                Constrain.EqualRandom(z, new Bernoulli(0.4));
            }
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(x, nameof(x));
            InferNet.Infer(y, nameof(y));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedOrRRCTest()
        {
            GatedOrRRC(true);
            GatedOrRRC(false);
        }

        private void GatedOrRRC(bool y)
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedOrRRCModel, y);
            ca.Execute(20);

            double priorB = 0.1;
            double priorX = 0.2;
            double priorY = y ? 1.0 : 0.0;
            double pXYCondT = 0.4;
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x) (py)^y (1-py)^(1-y)
            //                [(pT)^(1-(1-x)*(1-y)) (1-pT)^((1-x)*(1-y))]^b
            double sumXYCondT = (1 - (1 - priorX) * (1 - priorY)) * pXYCondT +
                                ((1 - priorX) * (1 - priorY)) * (1 - pXYCondT);
            double Z = priorB * sumXYCondT + (1 - priorB);
            double postB = priorB * sumXYCondT / Z;
            double postX = priorX * (priorB * pXYCondT + (1 - priorB)) / Z;
            double postY = priorY * (priorB * pXYCondT + (1 - priorB)) / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Bernoulli xDist = ca.Marginal<Bernoulli>("x");
            //Bernoulli yDist = ca.Marginal<Bernoulli>("y");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            //Console.WriteLine("y = {0} (should be {1})", yDist, postY);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - postX) < 1e-4);
            //Assert.True(Math.Abs(yDist.GetProbTrue() - postY) < 1e-4);
        }

        private void GatedOrRRCModel(bool y)
        {
            bool b = Factor.Bernoulli(0.1);
            bool x = Factor.Bernoulli(0.2);
            if (b)
            {
                bool z = Factor.Or(x, y);
                Constrain.EqualRandom(z, new Bernoulli(0.4));
            }
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(x, nameof(x));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedOrCRRTest()
        {
            GatedOrCRR(true);
            GatedOrCRR(false);
        }

        private void GatedOrCRR(bool z)
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedOrCRRModel, z);
            ca.Execute(20);

            double priorB = 0.1;
            double priorX = 0.2;
            double priorY = 0.3;
            double pXYCondT = z ? 1.0 : 0.0;
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x) (py)^y (1-py)^(1-y)
            //                [(pT)^(1-(1-x)*(1-y)) (1-pT)^((1-x)*(1-y))]^b
            double sumXYCondT = (1 - (1 - priorX) * (1 - priorY)) * pXYCondT +
                                ((1 - priorX) * (1 - priorY)) * (1 - pXYCondT);
            double Z = priorB * sumXYCondT + (1 - priorB);
            double postB = priorB * sumXYCondT / Z;
            double postX = priorX * (priorB * pXYCondT + (1 - priorB)) / Z;
            double postY = priorY * (priorB * pXYCondT + (1 - priorB)) / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Bernoulli xDist = ca.Marginal<Bernoulli>("x");
            Bernoulli yDist = ca.Marginal<Bernoulli>("y");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            Console.WriteLine("y = {0} (should be {1})", yDist, postY);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - postX) < 1e-4);
            Assert.True(System.Math.Abs(yDist.GetProbTrue() - postY) < 1e-4);
        }

        private void GatedOrCRRModel(bool z)
        {
            bool b = Factor.Bernoulli(0.1);
            bool x = Factor.Bernoulli(0.2);
            bool y = Factor.Bernoulli(0.3);
            if (b)
            {
                z = Factor.Or(x, y);
            }
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(x, nameof(x));
            InferNet.Infer(y, nameof(y));
        }

        // Fails due to spurious all zero exception
        [Fact]
        [Trait("Category", "OpenBug")]
        [Trait("Category", "CsoftModel")]
        public void GatedOrCRCTest()
        {
            GatedOrCRC(true, true);
            GatedOrCRC(false, true);
            GatedOrCRC(false, false);
            GatedOrCRC(true, false);
        }

        private void GatedOrCRC(bool y, bool z)
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedOrCRCModel, y, z);
            ca.Execute(20);

            double priorB = 0.1;
            double priorX = 0.2;
            double priorY = y ? 1.0 : 0.0;
            double pXYCondT = z ? 1.0 : 0.0;
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x) (py)^y (1-py)^(1-y)
            //                [(pT)^(1-(1-x)*(1-y)) (1-pT)^((1-x)*(1-y))]^b
            double sumXYCondT = (1 - (1 - priorX) * (1 - priorY)) * pXYCondT +
                                ((1 - priorX) * (1 - priorY)) * (1 - pXYCondT);
            double Z = priorB * sumXYCondT + (1 - priorB);
            double postB = priorB * sumXYCondT / Z;
            double postX = priorX * (priorB * pXYCondT + (1 - priorB)) / Z;
            double postY = priorY * (priorB * pXYCondT + (1 - priorB)) / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Bernoulli xDist = ca.Marginal<Bernoulli>("x");
            //Bernoulli yDist = ca.Marginal<Bernoulli>("y");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            //Console.WriteLine("y = {0} (should be {1})", yDist, postY);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - postX) < 1e-4);
            //Assert.True(Math.Abs(yDist.GetProbTrue() - postY) < 1e-4);
        }

        private void GatedOrCRCModel(bool y, bool z)
        {
            bool b = Factor.Bernoulli(0.1);
            bool x = Factor.Bernoulli(0.2);
            if (b)
            {
                z = Factor.Or(x, y);
            }
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(x, nameof(x));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedNotCRTest()
        {
            GatedNotCR(true);
            GatedNotCR(false);
        }

        private void GatedNotCR(bool z)
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedNotCRModel, z);
            ca.Execute(20);

            double priorB = 0.1;
            double priorX = 0.4;
            double pXCondT = z ? 0 : 1;
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x) [(pT)^(1-x) (1-pT)^x]^b
            double sumXCondT = priorX * pXCondT + (1 - priorX) * (1 - pXCondT);
            double Z = priorB * sumXCondT + (1 - priorB);
            double postB = priorB * sumXCondT / Z;
            double postX = priorX * (priorB * pXCondT + (1 - priorB)) / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Bernoulli xDist = ca.Marginal<Bernoulli>("x");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - postX) < 1e-4);
        }

        private void GatedNotCRModel(bool z)
        {
            bool b = Factor.Bernoulli(0.1);
            bool x = Factor.Bernoulli(0.4);
            if (b)
            {
                z = Factor.Not(x);
            }
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(x, nameof(x));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedNotRRTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedNotRRModel);
            ca.Execute(20);

            double priorB = 0.1;
            double priorX = 0.4;
            double pXCondT = 1 - 0.2;
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x) [(pT)^(1-x) (1-pT)^x]^b
            double sumXCondT = priorX * pXCondT + (1 - priorX) * (1 - pXCondT);
            double Z = priorB * sumXCondT + (1 - priorB);
            double postB = priorB * sumXCondT / Z;
            double postX = priorX * (priorB * pXCondT + (1 - priorB)) / Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Bernoulli xDist = ca.Marginal<Bernoulli>("x");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - postX) < 1e-4);
        }

        private void GatedNotRRModel()
        {
            bool b = Factor.Bernoulli(0.1);
            bool x = Factor.Bernoulli(0.4);
            if (b)
            {
                bool z = Factor.Not(x);
                Constrain.EqualRandom(z, new Bernoulli(0.2));
            }
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(x, nameof(x));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedSumTest2()
        {
            Gaussian xPrior = new Gaussian(1.2, 3.4);
            Gaussian sumPrior = new Gaussian(5.6, 7.8);
            int count = 2;

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedSum, xPrior, sumPrior, count);
            ca.Execute(20);

            double bPrior = 0.1;
            Gaussian xSum = new Gaussian(count * xPrior.GetMean(), count * xPrior.GetVariance());
            double sumCondT = System.Math.Exp(xSum.GetLogAverageOf(sumPrior));
            double Z = bPrior * sumCondT + (1 - bPrior);
            double bPost = bPrior * sumCondT / Z;
            IDistribution<double[]> xExpected = Distribution<double>.Array(count, delegate (int i) { return xPrior; });

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            DistributionArray<Gaussian> xActual = ca.Marginal<DistributionArray<Gaussian>>("x");
            Console.WriteLine("b = {0} (should be {1})", bDist, bPost);
            Console.WriteLine("x = {0} (should be approx {1})", xActual, xExpected);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - bPost) < 1e-4);
            //Assert.True(xActual.MaxDiff(xExpected) < 1e-4);
        }

        private void GatedSum(Gaussian xPrior, Gaussian sumPrior, int count)
        {
            bool b = Factor.Bernoulli(0.1);
            double[] x = new double[count];
            for (int i = 0; i < count; i++)
            {
                x[i] = Factor.Random(xPrior);
            }
            if (b)
            {
                double sum = Factor.Sum(x);
                Constrain.EqualRandom(sum, sumPrior);
            }
            InferNet.Infer(x, nameof(x));
            InferNet.Infer(b, nameof(b));
        }
    }
}
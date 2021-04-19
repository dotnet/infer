// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Xunit;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Models;

namespace Microsoft.ML.Probabilistic.Tests
{
#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

    using Assert = Xunit.Assert;
    using GaussianArray = DistributionStructArray<Gaussian, double>;
    using GaussianArrayArray = DistributionRefArray<DistributionStructArray<Gaussian, double>, double[]>;
    using GaussianArray2DArray = DistributionRefArray2D<DistributionStructArray<Gaussian, double>, double[]>;
    using DirichletArray2D = DistributionRefArray2D<Dirichlet, Vector>;
    using Microsoft.ML.Probabilistic.Utilities;
    using Microsoft.ML.Probabilistic.Algorithms;
    using Microsoft.ML.Probabilistic.Compiler;
    using Range = Microsoft.ML.Probabilistic.Models.Range;

    /// <summary>
    /// Gibbs sampling tests
    /// </summary>
    public class GibbsTests
    {
        // This test demonstrates the difference in inferring the root versus children, even if they are copies.
        [Fact]
        public void GibbsCopyTest()
        {
            Rand.Restart(12347);
            Bernoulli xPrior = new Bernoulli(0.1);
            Bernoulli xLike = new Bernoulli(0.2);
            Variable<bool> x = Variable.Bernoulli(0.1).Named("x");
            Variable.ConstrainEqualRandom(x, xLike);
            Variable<bool> y = Variable.Copy(x).Named("y");
            Variable<bool> z = Variable.Copy(y).Named("z");
            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.OptimiseForVariables = new Variable[] {x, z};
            Bernoulli xActual = engine.Infer<Bernoulli>(x);
            Bernoulli zActual = engine.Infer<Bernoulli>(z);
            Bernoulli xExpected = xPrior*xLike;
            Bernoulli zExpected = xExpected;
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Console.WriteLine("z = {0} should be {1}", zActual, zExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
            Assert.True(zExpected.MaxDiff(zActual) < 2e-1);
        }

        [Fact]
        public void GibbsIfExitDeterministicTest()
        {
            double xPrior = 0.4;
            Variable<bool> x = Variable.Bernoulli(xPrior).Named("x");
            Variable<bool> y = Variable.New<bool>().Named("y");
            using (Variable.If(x))
            {
                y.SetTo(true);
            }
            using (Variable.IfNot(x))
            {
                y.SetTo(false);
            }

            Rand.Restart(12347);
            GibbsSampling gs = new GibbsSampling();
            InferenceEngine engine = new InferenceEngine(gs);
            //engine.NumberOfIterations = 5000;
            //engine.Group(x, y);
            // y cannot be the root because GateExit cannot receive a distribution from cases
            //engine.Group(y, x);
            Bernoulli xActual = engine.Infer<Bernoulli>(x);
            Bernoulli yActual = engine.Infer<Bernoulli>(y);
            Bernoulli xExpected = new Bernoulli(xPrior);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Console.WriteLine("y = {0} should be {1}", yActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-2);
            Assert.True(xExpected.MaxDiff(yActual) < 6e-1);
        }

        [Fact]
        public void GibbsIfExitDeterministicTest2()
        {
            double xPrior = 0.4;
            double yLike = 0.7;
            Variable<bool> x = Variable.Bernoulli(xPrior).Named("x");
            Variable<bool> y = Variable.New<bool>().Named("y");
            using (Variable.If(x))
            {
                y.SetTo(true);
            }
            using (Variable.IfNot(x))
            {
                y.SetTo(false);
            }
            Variable.ConstrainEqualRandom(y, new Bernoulli(yLike));

            Rand.Restart(12347);
            GibbsSampling gs = new GibbsSampling();
            InferenceEngine engine = new InferenceEngine(gs);
            //engine.NumberOfIterations = 5000;
            //engine.Group(x, y);
            // y cannot be the root because GateExit cannot receive a distribution from cases
            //engine.Group(y, x);
            Bernoulli xActual = engine.Infer<Bernoulli>(x);
            Bernoulli yActual = engine.Infer<Bernoulli>(y);
            double Z = xPrior*yLike + (1 - xPrior)*(1 - yLike);
            Bernoulli xExpected = new Bernoulli(xPrior*yLike/Z);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Console.WriteLine("y = {0} should be {1}", yActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-2);
            Assert.True(xExpected.MaxDiff(yActual) < 6e-1);
        }

        [Fact]
        public void GibbsIfRandomIfRandomConditionTest()
        {
            double aPrior = 0.9;
            double bPrior = 0.1;
            double xPrior = 0.4;
            double xCondTT = 0.2;

            Variable<bool> a = Variable.Bernoulli(aPrior).Named("a");
            Variable<bool> b = Variable.Bernoulli(bPrior).Named("b");
            Variable<bool> x = Variable.Bernoulli(xPrior).Named("x");
            using (Variable.If(a))
            {
                using (Variable.If(b))
                {
                    Variable.ConstrainEqualRandom(x, new Bernoulli(xCondTT));
                }
            }

            Rand.Restart(12347);
            GibbsSampling gs = new GibbsSampling();
            InferenceEngine ie = new InferenceEngine(gs);
            ie.NumberOfIterations = 10000;
            ie.ShowProgress = false;
            Bernoulli aDist = ie.Infer<Bernoulli>(a);
            Bernoulli bDist = ie.Infer<Bernoulli>(b);
            Bernoulli xDist = ie.Infer<Bernoulli>(x);

            double sumCondTT = xPrior*xCondTT + (1 - xPrior)*(1 - xCondTT);
            double sumCondT = bPrior*sumCondTT + (1 - bPrior);
            double Z = aPrior*bPrior*sumCondTT + (1 - aPrior)*bPrior + aPrior*(1 - bPrior) + (1 - aPrior)*(1 - bPrior);
            Bernoulli aExpected = new Bernoulli(aPrior*(bPrior*sumCondTT + (1 - bPrior))/Z);
            Bernoulli bExpected = new Bernoulli(bPrior*(aPrior*sumCondTT + (1 - aPrior))/Z);
            Bernoulli xExpected = new Bernoulli(xPrior*(aPrior*bPrior*xCondTT + (1 - aPrior)*bPrior + aPrior*(1 - bPrior) + (1 - aPrior)*(1 - bPrior))/Z);

            double aError = aExpected.MaxDiff(aDist);
            double bError = bExpected.MaxDiff(bDist);
            double xError = xExpected.MaxDiff(xDist);
            Console.WriteLine("a = {0} should be {1} (error = {2})", aDist, aExpected, aError.ToString("g3"));
            Console.WriteLine("b = {0} should be {1} (error = {2})", bDist, bExpected, bError.ToString("g3"));
            Console.WriteLine("x = {0} should be {1} (error = {2})", xDist, xExpected, xError.ToString("g3"));
            Assert.True(aError < 5e-2);
            Assert.True(bError < 1e-2);
            Assert.True(xError < 5e-3);
        }

        [Fact]
        public void GibbsNestedIfExitTest()
        {
            double aPrior = 0.4;
            double bPrior = 0.6;
            Variable<bool> a = Variable.Bernoulli(aPrior).Named("a");
            Variable<bool> b = Variable.Bernoulli(bPrior).Named("b");
            Range outcomeRange = new Range(3).Named("outcomeRange");
            Variable<int> outcome = Variable.New<int>().Named("outcome");
            outcome.SetValueRange(outcomeRange);
            using (Variable.If(a))
            {
                outcome.SetTo(0);
            }
            using (Variable.IfNot(a))
            {
                using (Variable.If(b))
                {
                    outcome.SetTo(2);
                }
                using (Variable.IfNot(b))
                {
                    outcome.SetTo(1);
                }
            }
            Discrete outcomeLike = new Discrete(1, 2, 3);
            Variable.ConstrainEqualRandom(outcome, outcomeLike);

            double sumT = outcomeLike[0];
            double sumFT = outcomeLike[2];
            double sumFF = outcomeLike[1];
            double sumF = bPrior*sumFT + (1 - bPrior)*sumFF;
            double Z = aPrior*sumT + (1 - aPrior)*sumF;
            double aPost = aPrior*sumT/Z;
            double bPost = bPrior*(aPrior*sumT + (1 - aPrior)*sumFT)/Z;
            Bernoulli aExpected = new Bernoulli(aPost);
            Bernoulli bExpected = new Bernoulli(bPost);
            Discrete outcomeExpected = new Discrete(aPrior*sumT, (1 - aPrior)*(1 - bPrior)*sumFF, (1 - aPrior)*bPrior*sumFT);

            Rand.Restart(12347);
            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.NumberOfIterations = 10000;
            //engine.Group(white_delta, blackWins, whiteWins, outcome);
            Bernoulli aActual = engine.Infer<Bernoulli>(a);
            Bernoulli bActual = engine.Infer<Bernoulli>(b);
            Discrete outcomeActual = engine.Infer<Discrete>(outcome);
            Console.WriteLine("a = {0} should be {1}", aActual, aExpected);
            Console.WriteLine("b = {0} should be {1}", bActual, bExpected);
            Console.WriteLine("outcome = {0} should be {1}", outcomeActual, outcomeExpected);
            Assert.True(aExpected.MaxDiff(aActual) < 1e-2);
            Assert.True(bExpected.MaxDiff(bActual) < 1e-2);
            Assert.True(outcomeExpected.MaxDiff(outcomeActual) < 1e-2);
            if (false)
            {
                var samples = engine.Infer<IList<int>>(outcome, QueryTypes.Samples);
                Console.WriteLine(StringUtil.VerboseToString(samples));
            }
        }

        [Fact]
        public void GibbsNestedIfExitTest2()
        {
            Variable<double> white_delta = Variable.TruncatedGaussian(0, 1, double.NegativeInfinity, double.PositiveInfinity).Named("white_delta");
            Variable<bool> whiteWins = (white_delta > 0).Named("whiteWins");
            Variable<bool> blackWins = (!whiteWins).Named("blackWins");
            Range outcomeRange = new Range(3).Named("outcomeRange");
            Variable<int> outcome = Variable.New<int>().Named("outcome");
            outcome.SetValueRange(outcomeRange);
            using (Variable.If(blackWins))
            {
                outcome.SetTo(0);
            }
            using (Variable.IfNot(blackWins))
            {
                using (Variable.If(whiteWins))
                {
                    outcome.SetTo(2);
                }
                using (Variable.IfNot(whiteWins))
                {
                    outcome.SetTo(1);
                }
            }

            Rand.Restart(12347);
            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.NumberOfIterations = 10000;
            engine.ShowProgress = false;
            //engine.Group(white_delta, blackWins, whiteWins, outcome);
            Discrete outcomeActual = engine.Infer<Discrete>(outcome);
            Discrete outcomeExpected = new Discrete(0.5, 0, 0.5);
            Console.WriteLine("outcome = {0} should be {1}", outcomeActual, outcomeExpected);
            Assert.True(outcomeExpected.MaxDiff(outcomeActual) < 1e-2);
            if (false)
            {
                var samples = engine.Infer<IList<int>>(outcome, QueryTypes.Samples);
                Console.WriteLine(StringUtil.VerboseToString(samples));
            }
        }

        internal void GibbsTruncatedGaussianTest()
        {
            Variable<double> x = Variable.TruncatedGaussian(0, 1, Double.NegativeInfinity, double.PositiveInfinity).Named("x");
            //Variable.ConstrainPositive(x);
            Variable.ConstrainTrue(x > 0);
            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            TruncatedGaussian xActual = engine.Infer<TruncatedGaussian>(x);
            Gaussian xPrior = new Gaussian(0, 1);
            Gaussian xExpected = xPrior*IsPositiveOp.XAverageConditional(true, xPrior);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual.ToGaussian()) < 1e-10);
        }

        internal void GibbsTruncatedGaussianTest2()
        {
            Variable<double> x = Variable.TruncatedGaussian(0, 1, Double.NegativeInfinity, double.PositiveInfinity).Named("x");
            Variable.ConstrainPositive(x - 0.5);
            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            TruncatedGaussian xActual = engine.Infer<TruncatedGaussian>(x);
            Gaussian xPrior = new Gaussian(-0.5, 1);
            Gaussian xExpected = xPrior*IsPositiveOp.XAverageConditional(true, xPrior);
            xExpected.SetMeanAndVariance(xExpected.GetMean() + 0.5, xExpected.GetVariance());
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual.ToGaussian()) < 1e-10);
        }

        internal void GibbsTruncatedGaussianTest3()
        {
            Variable<double> x = Variable.TruncatedGaussian(0, 1, Double.NegativeInfinity, double.PositiveInfinity).Named("x");
            Variable.ConstrainBetween(x, 2, 3);
            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            TruncatedGaussian xActual = engine.Infer<TruncatedGaussian>(x);
            Gaussian xPrior = new Gaussian(0, 1);
            Gaussian xExpected = xPrior*IsBetweenGaussianOp.XAverageConditional(true, xPrior, 2, 3);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual.ToGaussian()) < 1e-10);
        }

        [Fact]
        public void GibbsTruncatedGaussianTest4()
        {
            Variable<double> x = Variable.TruncatedGaussian(0, 1, Double.NegativeInfinity, double.PositiveInfinity).Named("x");
            Variable<double> lowerBound = Variable.TruncatedGaussian(0, 1, Double.NegativeInfinity, double.PositiveInfinity).Named("lowerBound");
            Variable.ConstrainBetween(x, lowerBound, 3);

            Gaussian xExpected = new Gaussian(0.5559, 0.6639);
            if (false)
            {
                Gaussian xPrior = new Gaussian(0, 1);
                Gaussian lowerBoundPrior = new Gaussian(0, 1);
                GaussianEstimator xEst = new GaussianEstimator();
                for (int i = 0; i < 10000000; i++)
                {
                    double xs = xPrior.Sample();
                    double lbs = lowerBoundPrior.Sample();
                    if (lbs < xs && xs < 3) xEst.Add(xs);
                }
                xExpected = xEst.GetDistribution(new Gaussian());
            }

            Rand.Restart(12347);
            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.NumberOfIterations = 100000;
            engine.ShowProgress = false;
            TruncatedGaussian xActual = engine.Infer<TruncatedGaussian>(x);
            Console.WriteLine(xActual);
            IList<double> xSamples = engine.Infer<IList<double>>(x, QueryTypes.Samples);
            GaussianEstimator est = new GaussianEstimator();
            foreach (double sample in xSamples) est.Add(sample);
            Gaussian xPost = est.GetDistribution(new Gaussian());
            Console.WriteLine("x samples = {0} should be {1}", xPost, xExpected);
            Assert.True(xExpected.MaxDiff(xPost) < 3e-2);
        }

        // Same as GibbsTruncatedGaussianTest4 but with a poor initialization that causes Gibbs to crash
        [Fact]
        [Trait("Category", "OpenBug")]
        public void GibbsTruncatedGaussianTest5()
        {
            Variable<double> x = Variable.TruncatedGaussian(0, 1, Double.NegativeInfinity, double.PositiveInfinity).Named("x");
            Variable<double> lowerBound = Variable.TruncatedGaussian(4, 1e-2, Double.NegativeInfinity, double.PositiveInfinity).Named("lowerBound");
            Variable.ConstrainBetween(x, lowerBound, 3);
            //lowerBound.InitialiseTo(TruncatedGaussian.PointMass(4));

            Gaussian xExpected = new Gaussian(0.5559, 0.6639);
            if (false)
            {
                Gaussian xPrior = new Gaussian(0, 1);
                Gaussian lowerBoundPrior = new Gaussian(0, 1);
                GaussianEstimator xEst = new GaussianEstimator();
                for (int i = 0; i < 10000000; i++)
                {
                    double xs = xPrior.Sample();
                    double lbs = lowerBoundPrior.Sample();
                    if (lbs < xs && xs < 3) xEst.Add(xs);
                }
                xExpected = xEst.GetDistribution(new Gaussian());
            }

            Rand.Restart(12347);
            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.NumberOfIterations = 100000;
            engine.ShowProgress = false;
            TruncatedGaussian xActual = engine.Infer<TruncatedGaussian>(x);
            Console.WriteLine(xActual);
            IList<double> xSamples = engine.Infer<IList<double>>(x, QueryTypes.Samples);
            GaussianEstimator est = new GaussianEstimator();
            foreach (double sample in xSamples) est.Add(sample);
            Gaussian xPost = est.GetDistribution(new Gaussian());
            Console.WriteLine("x samples = {0} should be {1}", xPost, xExpected);
            Assert.True(xExpected.MaxDiff(xPost) < 2e-2);
        }

        // Fails because MessageTransform tries to get an operator that it doesn't actually need.
        [Fact]
        [Trait("Category", "OpenBug")]
        public void GibbsRatio()
        {
            Rand.Restart(12347);
            double shape = 10;
            double rate = 10;
            Variable<double> x = Variable.GammaFromShapeAndRate(shape, rate).Named("x");
            Variable<double> y = 1.0/x;
            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.ShowProgress = false;
            engine.NumberOfIterations = 10000;
            IList<double> ySamples = engine.Infer<IList<double>>(y, QueryTypes.Samples);
            GaussianEstimator yEst = new GaussianEstimator();
            foreach (double sample in ySamples)
            {
                yEst.Add(sample);
            }
            Gaussian yActual = yEst.GetDistribution(new Gaussian());
            double yMean = rate/(shape - 1);
            Gaussian yExpected = Gaussian.FromMeanAndVariance(yMean, yMean*yMean/(shape - 2));
            Console.WriteLine("y = {0} should be {1}", yActual, yExpected);
            Assert.True(yExpected.MaxDiff(yActual) < 1e-1);
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void GibbsLogisticBinomial()
        {
            // This test is expected to fail for Gibbs sampling, as the factor
            // is non-conjugate
            int n = 10;
            int k = 3;

            Variable<double> w = Variable.GaussianFromMeanAndVariance(1.2, 3.4).Named("w");
            Variable<double> p = Variable.Logistic(w).Named("p");
            Variable<int> y = Variable.Binomial(n, p).Named("y");
            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = new GibbsSampling();
            y.ObservedValue = k;
            Gaussian wActual = engine.Infer<Gaussian>(w);
        }

        [Fact]
        public void GibbsNotEqual()
        {
            Rand.Restart(12347);
            Variable<bool> x = Variable.Bernoulli(0.2).Named("x");
            Variable<bool> y = Variable.Bernoulli(0.7).Named("y");
            var z = (x != y);
            z.ObservedValue = true;
            z.Name = "z";
            InferenceEngine ie = new InferenceEngine();
            ie.NumberOfIterations = 5000;
            ie.ShowProgress = false;
            ie.Algorithm = new GibbsSampling();
            ie.Group(x, y);
            Bernoulli xExpected = new Bernoulli(0.2)*new Bernoulli(0.3);
            Bernoulli xActual = ie.Infer<Bernoulli>(x);
            Bernoulli yExpected = new Bernoulli(0.8)*new Bernoulli(0.7);
            Bernoulli yActual = ie.Infer<Bernoulli>(y);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Console.WriteLine("y = {0} should be {1}", yActual, yExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
            Assert.True(yExpected.MaxDiff(yActual) < 1e-1);
        }

        [Fact]
        public void GibbsEqual()
        {
            Variable<bool> x = Variable.Bernoulli(0.2).Named("x");
            Variable<bool> y = Variable.Bernoulli(0.7).Named("y");
            var z = (x == y);
            z.ObservedValue = false;
            z.Name = "z";
            InferenceEngine ie = new InferenceEngine();
            ie.Algorithm = new GibbsSampling();
            ie.Group(x, y);
            Bernoulli xExpected = new Bernoulli(0.0967741935483872);
            Bernoulli xActual = ie.Infer<Bernoulli>(x);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
        }

        [Fact]
        public void GibbsEqualDeterministic()
        {
            Variable<bool> x = Variable.Bernoulli(0.2).Named("x");
            Variable<bool> y = Variable.Bernoulli(0.7).Named("y");
            Variable<bool> z = (!y).Named("z");
            Variable.ConstrainEqual(x, z);
            InferenceEngine ie = new InferenceEngine();
            ie.Algorithm = new GibbsSampling();
            //ie.Group(x, z);
            Bernoulli xExpected = new Bernoulli(0.2)*new Bernoulli(0.3);
            Bernoulli xActual = ie.Infer<Bernoulli>(x);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
        }

        [Fact]
        public void GibbsEqual2Error()
        {
            try
            {
                Variable<bool> x = Variable.Bernoulli(0.2).Named("x");
                Variable<bool> y = Variable.Bernoulli(0.7).Named("y");
                Variable<bool> z = Variable.Bernoulli(0.4).Named("z");
                Variable.ConstrainEqual(x, y);
                Variable.ConstrainEqual(y, z);
                InferenceEngine ie = new InferenceEngine();
                ie.Algorithm = new GibbsSampling();
                ie.Group(x, y, z);
                Bernoulli xExpected = new Bernoulli(0.2)*new Bernoulli(0.7)*new Bernoulli(0.4);
                Bernoulli xActual = ie.Infer<Bernoulli>(x);
                Bernoulli yActual = ie.Infer<Bernoulli>(y);
                Bernoulli zActual = ie.Infer<Bernoulli>(z);
                Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
                Console.WriteLine("y = {0} should be {1}", yActual, xExpected);
                Console.WriteLine("z = {0} should be {1}", zActual, xExpected);
                Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
                Assert.True(xExpected.MaxDiff(yActual) < 1e-10);
                Assert.True(xExpected.MaxDiff(zActual) < 1e-10);
                Assert.True(false, "Did not throw exception");
            }
            catch (CompilationFailedException ex)
            {
                Console.WriteLine("Correctly threw exception: " + ex);
            }
        }

        [Fact]
        public void GibbsSimpleGaussianRange()
        {
            // The equivalent WinBugs model and data is:
            // model {
            //    m ~ dnorm(0.0,0.01)
            //    p ~ dgamma(1.0,1.0)
            //    for (i in 1:N) {
            //      x[i] ~ dnorm(m,p) 
            //    }
            // }
            // list(x = c(5.0,7.0), N = 2)

            Rand.Restart(12347);
            int burnIn = 1000;
            int numIters = 20000;
            int thin = 10;

            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100).Named("mean");
            Variable<double> prec = Variable.GammaFromShapeAndScale(1, 1).Named("precision");
            Range i = new Range(2).Named("i");
            VariableArray<double> data = Variable.Array<double>(i).Named("data");
            data[i] = Variable.GaussianFromMeanAndPrecision(mean, prec).ForEach(i);
            data.ObservedValue = new double[] {5.0, 7.0};

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = burnIn;
            gs.Thin = thin;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = numIters;
            engine.ShowProgress = false;
            var meanMarg = engine.Infer<Gaussian>(mean);
            var precMarg = engine.Infer<Gamma>(prec);
            double meanActual = meanMarg.GetMean();
            double precActual = precMarg.GetMean();
            // Following are from BUGS - 100000 iterations
            double meanExpected = 5.9;
            double precExpected = 0.7523;
            Assert.True(System.Math.Abs(meanActual - meanExpected) < 0.05);
            Assert.True(System.Math.Abs(precActual - precExpected) < 0.05);
        }

        [Fact]
        public void GibbsSimpleGaussian()
        {
            // The equivalent WinBugs model is:
            // model {
            //    m ~ dnorm(1.0,0.5)
            //    p ~ dgamma(2.0,0.25)
            //    x ~ dnorm(m, p)
            // }

            Rand.Restart(12347);
            int burnIn = 1000;
            int numIters = 300000;
            int thin = 10;

            Variable<double> mean = Variable.GaussianFromMeanAndVariance(1, 2).Named("mean");
            Variable<double> precision = Variable.GammaFromShapeAndScale(2, 4).Named("precision");
            Variable<double> x = Variable.GaussianFromMeanAndPrecision(mean, precision).Named("x");
            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = burnIn;
            gs.Thin = thin;
            mean.InitialiseTo(new Gaussian(1, 2));
            precision.InitialiseTo(new Gamma(2, 4));
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = numIters;
            engine.ShowProgress = false;
            Gaussian xActual = engine.Infer<Gaussian>(x);
            Gaussian meanActual = engine.Infer<Gaussian>(mean);
            Gamma precisionActual = engine.Infer<Gamma>(precision);
            Gaussian meanExpected = new Gaussian(1, 2);
            Gamma precisionExpected = new Gamma(2, 4);
            Gaussian xExpected = GaussianOp.SampleAverageConditional_slow(Gaussian.Uniform(), meanExpected, precisionExpected);
            Console.WriteLine("mean = {0} should be {1}", meanActual, meanExpected);
            Assert.True(meanExpected.MaxDiff(meanActual) < 3e-2);
            Assert.True(precisionExpected.MaxDiff(precisionActual) < 2e-2);
            Assert.True(xExpected.MaxDiff(xActual) < 3e-2);
        }

        [Fact]
        public void GibbsSimpleGaussianDuplicated()
        {
            // This just duplicates the previous test across a range
            Rand.Restart(12347);
            int burnIn = 1000;
            int numIters = 100000;
            int thin = 10;
            Range i = new Range(2);
            VariableArray<double> mean = Variable.Array<double>(i).Named("mean");
            VariableArray<double> precision = Variable.Array<double>(i).Named("precision");
            VariableArray<double> x = Variable.Array<double>(i).Named("x");
            mean[i] = Variable.GaussianFromMeanAndVariance(1, 2).ForEach(i);
            precision[i] = Variable.GammaFromShapeAndScale(2, 4).ForEach(i);
            x[i] = Variable.GaussianFromMeanAndPrecision(mean[i], precision[i]);
            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = burnIn;
            gs.Thin = thin;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = numIters;
            engine.ShowProgress = false;
            x.AddAttribute(QueryTypes.Marginal);
            x.AddAttribute(QueryTypes.Samples);
            x.AddAttribute(QueryTypes.Conditionals);
            IList<Gaussian> xActual = engine.Infer<IList<Gaussian>>(x);
            IList<double[]> samples = engine.Infer<IList<double[]>>(x, QueryTypes.Samples);
            IList<GaussianArray> conditionals = engine.Infer<IList<GaussianArray>>(x, QueryTypes.Conditionals);
            Assert.NotNull(samples);
            Assert.NotNull(conditionals);
            Assert.NotEqual(samples[0], samples[1]);
            Assert.NotEqual(conditionals[0], conditionals[1]);

            Gaussian meanExpected = new Gaussian(1, 2);
            Gamma precisionExpected = new Gamma(2, 4);
            Gaussian xExpected = GaussianOp.SampleAverageConditional_slow(Gaussian.Uniform(), meanExpected, precisionExpected);
            IList<Gaussian> meanActual = engine.Infer<IList<Gaussian>>(mean);
            IList<Gamma> precisionActual = engine.Infer<IList<Gamma>>(precision);
            for (int j = 0; j < xActual.Count; j++)
            {
                Assert.True(meanExpected.MaxDiff(meanActual[j]) < 1e-1);
                Assert.True(precisionExpected.MaxDiff(precisionActual[j]) < 1e-1);
                Assert.True(xExpected.MaxDiff(xActual[j]) < 1e-1);
                GaussianEstimator est = new GaussianEstimator();
                foreach (double[] sample in samples)
                {
                    est.Add(sample[j]);
                }
                Gaussian xActual2 = est.GetDistribution(new Gaussian());
                Assert.True(xExpected.MaxDiff(xActual2) < 1e-1);
            }
        }

        [Fact]
        public void GibbsProfiling()
        {
            // This just duplicates the previous test across a range
            Rand.Restart(12347);
            int burnIn = 1000;
            int numIters = 10000;
            int thin = 10;
            Range i = new Range(2);
            VariableArray<double> mean = Variable.Array<double>(i).Named("mean");
            VariableArray<double> precision = Variable.Array<double>(i).Named("precision");
            VariableArray<double> x = Variable.Array<double>(i).Named("x");
            mean[i] = Variable.GaussianFromMeanAndVariance(1, 2).ForEach(i);
            precision[i] = Variable.GammaFromShapeAndScale(2, 4).ForEach(i);
            x[i] = Variable.GaussianFromMeanAndPrecision(mean[i], precision[i]);
            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = burnIn;
            gs.Thin = thin;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = numIters;
            engine.ShowProgress = false;
            var xPost = engine.Infer<Gaussian[]>(x);
        }

        [Fact]
        public void GibbsDiscreteFromDirichlet()
        {
            Rand.Restart(12347);
            int burnIn = 100;
            int numIters = 10000;
            int thin = 10;
            Vector alpha = Vector.FromArray(2.0, 5.0, 3.0);
            Variable<Vector> dir = Variable.Dirichlet(alpha).Named("dir");
            Variable<int> pi = Variable.Discrete(dir).Named("pi");
            pi.AddAttribute(QueryTypes.Marginal);
            pi.AddAttribute(QueryTypes.Samples);
            pi.AddAttribute(QueryTypes.Conditionals);
            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = burnIn;
            gs.Thin = thin;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = numIters;
            engine.ShowProgress = false;
            var piMarg = engine.Infer<Discrete>(pi);
            var piSamples = engine.Infer(pi, QueryTypes.Samples);
            var piConditionals = engine.Infer(pi, QueryTypes.Conditionals);
            Assert.True(System.Math.Abs(piMarg[0] - 0.2) < 0.01);
            Assert.True(System.Math.Abs(piMarg[1] - 0.5) < 0.01);
            Assert.True(System.Math.Abs(piMarg[2] - 0.3) < 0.01);
        }

        [Fact]
        public void GibbsTwoCoins()
        {
            // The equivalent WinBugs model is:
            // model {
            //    firstCoin ~ dbern(0.5)
            //    secondCoin ~ dbern(0.5)
            //    bothHeads <- firstCoin * secondCoin
            // }
            Rand.Restart(12347);
            int burnIn = 100;
            int numIters = 20000;
            int thin = 10;
            Variable<bool> firstCoin = Variable.Bernoulli(0.5).Named("firstCoin");
            Variable<bool> secondCoin = Variable.Bernoulli(0.5).Named("secondCoin");
            Variable<bool> bothHeads = (firstCoin & secondCoin).Named("bothHeads");

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = burnIn;
            gs.Thin = thin;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = numIters;
            engine.ShowProgress = false;
            Bernoulli bothHeadsExpected = new Bernoulli(0.25);
            Bernoulli bothHeadsActual = engine.Infer<Bernoulli>(bothHeads);
            Console.WriteLine("bothHeads = {0} should be {1}", bothHeadsActual, bothHeadsExpected);
            Assert.True(bothHeadsExpected.MaxDiff(bothHeadsActual) < 1e-2);
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void GibbsTwoCoinsObserved()
        {
            Rand.Restart(12);
            int burnIn = 100;
            int numIters = 10000;
            int thin = 10;
            Variable<bool> firstCoin = Variable.Bernoulli(0.5).Named("firstCoin");
            Variable<bool> secondCoin = Variable.Bernoulli(0.5).Named("secondCoin");
            Variable<bool> bothHeads = (firstCoin & secondCoin).Named("bothHeads");

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = burnIn;
            gs.Thin = thin;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = numIters;
            engine.ShowProgress = false;
            for (int trial = 0; trial < 20; trial++)
            {
                Bernoulli firstCoinExpected;
                if (trial%2 == 0)
                {
                    bothHeads.ObservedValue = false;
                    firstCoinExpected = new Bernoulli(1.0/3);
                }
                else
                {
                    bothHeads.ObservedValue = true;
                    firstCoinExpected = new Bernoulli(1.0);
                }
                var firstCoinActual = engine.Infer<Bernoulli>(firstCoin);
                Console.WriteLine("firstCoin = {0} should be {1}", firstCoinActual, firstCoinExpected);
                Assert.True(firstCoinExpected.MaxDiff(firstCoinActual) < 5e-2);
            }
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void GibbsTwoCoinsConstrained()
        {
            Rand.Restart(12);
            int burnIn = 100;
            int numIters = 10000;
            int thin = 10;
            Variable<bool> firstCoin = Variable.Bernoulli(0.5).Named("firstCoin");
            Variable<bool> secondCoin = Variable.Bernoulli(0.5).Named("secondCoin");
            Variable<bool> bothHeads = (firstCoin & secondCoin).Named("bothHeads");
            Variable.ConstrainFalse(bothHeads);

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = burnIn;
            gs.Thin = thin;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = numIters;
            engine.ShowProgress = false;
            for (int trial = 0; trial < 20; trial++)
            {
                Bernoulli firstCoinExpected;
                if (trial%2 == 0)
                {
                    bothHeads.ObservedValue = false;
                    firstCoinExpected = new Bernoulli(1.0/3);
                }
                else
                {
                    bothHeads.ObservedValue = true;
                    firstCoinExpected = new Bernoulli(1.0);
                }
                var firstCoinActual = engine.Infer<Bernoulli>(firstCoin);
                Console.WriteLine("firstCoin = {0} should be {1}", firstCoinActual, firstCoinExpected);
                Assert.True(firstCoinExpected.MaxDiff(firstCoinActual) < 5e-2);
            }
        }

        [Fact]
        public void GibbsGatedConstraint()
        {
            Rand.Restart(12347);
            double bPrior = 0.3;
            double cPrior = 0.4;
            Variable<bool> b = Variable.Bernoulli(bPrior).Named("b");
            Variable<bool> c = Variable.Bernoulli(cPrior).Named("c");
            using (Variable.If(c))
            {
                Variable.ConstrainTrue(b);
            }

            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.ShowProgress = false;
            double Z = cPrior*bPrior + (1 - cPrior);
            Bernoulli bExpected = new Bernoulli(bPrior/Z);
            Bernoulli cExpected = new Bernoulli(cPrior*bPrior/Z);
            var bActual = engine.Infer<Bernoulli>(b);
            var cActual = engine.Infer<Bernoulli>(c);
            Console.WriteLine("b = {0} should be {1}", bActual, bExpected);
            Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
            Assert.True(bExpected.MaxDiff(bActual) < 3e-2);
            Assert.True(cExpected.MaxDiff(cActual) < 3e-2);
        }

        // It should be possible to get this to work, since y will be collapsed
        [Fact]
        public void GibbsEvidenceTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            double xPrior = 0.1;
            Variable<bool> x = Variable.Bernoulli(xPrior).Named("x");
            Variable<bool> y = !x;
            y.Name = "y";
            double yLike = 0.2;
            Variable.ConstrainEqualRandom(y, new Bernoulli(yLike));
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.NumberOfIterations = 10000;
            engine.ShowProgress = false;
            double evActual = System.Math.Exp(engine.Infer<Bernoulli>(evidence).LogOdds);
            double evExpected = xPrior*(1 - yLike) + (1 - xPrior)*yLike;
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-4) < 1e-2);
        }

        // Fails with the wrong evidence value
        [Fact]
        public void GibbsEvidenceError()
        {
            try
            {
                Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
                IfBlock block = Variable.If(evidence);
                Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 1).Named("x");
                Variable<double> y = Variable.GaussianFromMeanAndVariance(x, 1).Named("y");
                Gaussian yLike = new Gaussian(2, 3);
                Variable.ConstrainEqualRandom(y, yLike);
                block.CloseBlock();

                InferenceEngine engine = new InferenceEngine(new GibbsSampling());
                //engine.NumberOfIterations = 100000;
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                double evExpected = yLike.GetLogAverageOf(new Gaussian(0, 2));
                Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-2);
                Assert.True(false, "Did not throw exception");
            }
            catch (CompilationFailedException ex)
            {
                Console.WriteLine("Correctly threw exception: " + ex);
            }
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void GibbsSumTest()
        {
            Rand.Restart(12347);
            Variable<double> a = Variable.GaussianFromMeanAndVariance(1, 2).Named("a");
            Variable<double> b = Variable.GaussianFromMeanAndVariance(3, 4).Named("b");
            var c = (a + b).Named("c");
            c.ObservedValue = 5;

            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = new GibbsSampling();
            Gaussian aActual = engine.Infer<Gaussian>(a);
            engine.Algorithm = new ExpectationPropagation();
            Gaussian aExpected = engine.Infer<Gaussian>(a);
            Console.WriteLine("a = {0} should be {1}", aActual, aExpected);
            Assert.True(aExpected.MaxDiff(aActual) < 1e-2);
        }

        [Fact]
        public void GibbsSumPlusNoise()
        {
            Rand.Restart(12347);
            Variable<double> x = Variable.GaussianFromMeanAndVariance(1, 2).Named("x");
            Variable<double> y = (x + 3).Named("y");
            Variable<double> z = Variable.GaussianFromMeanAndVariance(y, 4).Named("z");
            Gaussian zLike = new Gaussian(5, 6);
            Variable.ConstrainEqualRandom(z, zLike);

            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.Group(y, z);
            Gaussian xActual = engine.Infer<Gaussian>(x);
            Gaussian xExpected = new Gaussian(1.167, 1.667); // (new InferenceEngine()).Infer<Gaussian>(x);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 2e-2);
        }

        [Fact]
        public void GibbsGaussianChain()
        {
            Rand.Restart(12347);
            Variable<double> x = Variable.GaussianFromMeanAndVariance(1, 2).Named("x");
            Variable<double> y = Variable.GaussianFromMeanAndVariance(x, 3).Named("y");
            Variable<double> z = Variable.GaussianFromMeanAndVariance(y, 4).Named("z");
            Gaussian zLike = new Gaussian(5, 6);
            Variable.ConstrainEqualRandom(z, zLike);

            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.Group(x, y, z);
            //engine.Group(y, z);
            Gaussian xActual = engine.Infer<Gaussian>(x);
            Gaussian yActual = engine.Infer<Gaussian>(y);
            Gaussian zActual = engine.Infer<Gaussian>(z);
            InferenceEngine engineEP = new InferenceEngine();
            Gaussian xExpected = engineEP.Infer<Gaussian>(x);
            Gaussian yExpected = engineEP.Infer<Gaussian>(y);
            Gaussian zExpected = engineEP.Infer<Gaussian>(z);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Console.WriteLine("y = {0} should be {1}", yActual, yExpected);
            Console.WriteLine("z = {0} should be {1}", zActual, zExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
            Assert.True(yExpected.MaxDiff(yActual) < 1e-2);
            Assert.True(zExpected.MaxDiff(zActual) < 1e-2);
        }

        [Fact]
        public void GibbsDisconnectedGroupError()
        {
            try
            {
                Variable<double> x = Variable.GaussianFromMeanAndVariance(1, 2).Named("x");
                Variable<double> y = Variable.GaussianFromMeanAndVariance(x, 3).Named("y");
                Variable<double> z = Variable.GaussianFromMeanAndVariance(y, 4).Named("z");
                Gaussian zLike = new Gaussian(5, 6);
                Variable.ConstrainEqualRandom(z, zLike);

                InferenceEngine engine = new InferenceEngine(new GibbsSampling());
                engine.Group(x, z);
                Gaussian xActual = engine.Infer<Gaussian>(x);
                Gaussian xExpected = new Gaussian(1.533, 1.733); //(new InferenceEngine()).Infer<Gaussian>(x);
                Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
                Assert.True(xExpected.MaxDiff(xActual) < 1e-1);
                Assert.True(false, "Did not throw exception");
            }
            catch (CompilationFailedException ex)
            {
                Console.WriteLine("Correctly threw exception: " + ex);
            }
        }

        [Fact]
        public void GibbsNotChain()
        {
            double xPrior = 0.1;
            Variable<bool> x = Variable.Bernoulli(xPrior).Named("x");
            Variable<bool> y = (!x).Named("y");
            Variable<bool> z = (!y).Named("z");
            double zLike = 0.2;
            Variable.ConstrainEqualRandom(z, new Bernoulli(zLike));

            Rand.Restart(12347);
            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            //engine.Group(x, y, z);
            engine.Group(y, z); // forces y to be the root
            Bernoulli xActual = engine.Infer<Bernoulli>(x);
            Bernoulli xExpected = (new Bernoulli(xPrior))*(new Bernoulli(zLike));
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-1);
        }

        [Fact]
        public void GibbsInvalidRootError()
        {
            try
            {
                double xPrior = 0.1;
                Variable<bool> x = Variable.Bernoulli(xPrior).Named("x");
                Variable<bool> y = (!x).Named("y");
                Variable<bool> z = (!y).Named("z");
                double zLike = 0.2;
                Variable.ConstrainEqualRandom(z, new Bernoulli(zLike));

                InferenceEngine engine = new InferenceEngine(new GibbsSampling());
                engine.Group(x, y);
                engine.Group(y, z);
                Bernoulli xActual = engine.Infer<Bernoulli>(x);
                Bernoulli xExpected = (new Bernoulli(xPrior))*(new Bernoulli(zLike));
                Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
                Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
                Assert.True(false, "Did not throw exception");
            }
            catch (CompilationFailedException ex)
            {
                Console.WriteLine("Correctly threw exception: " + ex);
            }
        }

        [Fact]
        public void GibbsExitDerivedTest()
        {
            Rand.Restart(12347);
            double xPrior = 0.1;
            Variable<bool> x = Variable.Bernoulli(xPrior).Named("x");
            double cPrior = 0.2;
            Variable<bool> c = Variable.Bernoulli(cPrior).Named("c");
            Variable<bool> y = Variable.New<bool>().Named("y");
            using (Variable.If(c))
            {
                y.SetTo(!x);
            }
            using (Variable.IfNot(c))
            {
                y.SetTo(!x);
            }
            double yLike = 0.3;
            Variable.ConstrainEqualRandom(y, new Bernoulli(yLike));

            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            Bernoulli xActual = engine.Infer<Bernoulli>(x);
            Bernoulli xExpected = (new Bernoulli(xPrior))*(new Bernoulli(1 - yLike));
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-2);
            Bernoulli cActual = engine.Infer<Bernoulli>(c);
            Bernoulli cExpected = new Bernoulli(cPrior);
            Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
            Assert.True(cExpected.MaxDiff(cActual) < 1e-2);
        }

        [Fact]
        public void GibbsExitDerivedTest2()
        {
            Rand.Restart(12347);
            double xPrior = 0.1;
            Variable<bool> x = Variable.Bernoulli(xPrior).Named("x");
            double cPrior = 0.2;
            Variable<bool> c = Variable.Bernoulli(cPrior).Named("c");
            Variable<bool> y = Variable.New<bool>().Named("y");
            using (Variable.If(c))
            {
                y.SetTo(x);
            }
            using (Variable.IfNot(c))
            {
                y.SetTo(!x);
            }
            double yLike = 0.3;
            Variable.ConstrainEqualRandom(y, new Bernoulli(yLike));

            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.NumberOfIterations = 10000;
            // p(x,y,c) = p0(x) p0(c) p0(y) delta(y==x)^c delta(y==!x)^c
            double sumT = xPrior*yLike + (1 - xPrior)*(1 - yLike);
            double sumF = xPrior*(1 - yLike) + (1 - xPrior)*yLike;
            double z = cPrior*sumT + (1 - cPrior)*sumF;
            Bernoulli cActual = engine.Infer<Bernoulli>(c);
            Bernoulli cExpected = new Bernoulli(cPrior*sumT/z);
            Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
            Assert.True(cExpected.MaxDiff(cActual) < 1e-2);
            Bernoulli xActual = engine.Infer<Bernoulli>(x);
            Bernoulli xExpected = new Bernoulli(xPrior*(yLike*cPrior + (1 - yLike)*(1 - cPrior))/z);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-2);
        }

        [Fact]
        public void GibbsProductOfGaussiansWithExplicitGroups()
        {
            // The equivalent WinBugs model is:
            // model {
            //    a ~ dnorm(1.0,0.5)
            //    b ~ dnorm(2.0,0.8)
            //    c <- a*b
            // }
            Rand.Restart(12347);
            int burnIn = 1000;
            int numIters = 20000;
            int thin = 1;

            double ma = 1, va = 2;
            double mb = 2, vb = 1/0.8;
            Variable<double> a = Variable.GaussianFromMeanAndVariance(ma, va).Named("a");
            Variable<double> b = Variable.GaussianFromMeanAndVariance(mb, vb).Named("b");
            Variable<double> c = (a*b).Named("c");
            c.AddAttribute(QueryTypes.Marginal);
            c.AddAttribute(QueryTypes.Samples);
            c.AddAttribute(QueryTypes.Conditionals);

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = burnIn;
            gs.Thin = thin;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.ShowProgress = false;
            engine.NumberOfIterations = numIters;
            engine.Group(a, c);
            engine.Group(b, c);
            var cActual = engine.Infer<Gaussian>(c);
            IList<double> samples = engine.Infer<IList<double>>(c, QueryTypes.Samples);
            IList<Gaussian> conditionals = engine.Infer<IList<Gaussian>>(c, QueryTypes.Conditionals);
            Gaussian cExpected = new Gaussian(ma*mb, va*vb + va*mb*mb + vb*ma*ma);
            Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
            Assert.True(cExpected.MaxDiff(cActual) < 2e-3);
            GaussianEstimator est = new GaussianEstimator();
            foreach (double sample in samples)
            {
                est.Add(sample);
            }
            Gaussian cActual2 = est.GetDistribution(new Gaussian());
            Assert.True(cExpected.MaxDiff(cActual2) < 2e-3);
            GaussianEstimator est2 = new GaussianEstimator();
            foreach (Gaussian conditional in conditionals)
            {
                est2.Add(conditional);
            }
            Gaussian cActual3 = est2.GetDistribution(new Gaussian());
            Assert.True(cExpected.MaxDiff(cActual3) < 2e-3);
            Assert.NotNull(samples);
            Assert.NotNull(conditionals);
            Assert.NotEqual(samples[0], samples[1]);
            Assert.NotEqual(conditionals[0], conditionals[1]);
        }

        [Fact]
        public void GibbsProductOfGaussiansWithBadRoot()
        {
            try
            {
                // The equivalent WinBugs model is:
                // model {
                //    a ~ dnorm(1.0,0.5)
                //    b ~ dnorm(2.0,0.8)
                //    c <- a*b
                // }
                Rand.Restart(12347);
                int burnIn = 1000;
                int numIters = 20000;
                int thin = 10;

                Variable<double> a = Variable.GaussianFromMeanAndPrecision(1, .5).Named("a");
                Variable<double> b = Variable.GaussianFromMeanAndPrecision(2, .8).Named("b");
                Variable<double> c = (a*b).Named("c");

                GibbsSampling gs = new GibbsSampling();
                gs.BurnIn = burnIn;
                gs.Thin = thin;
                InferenceEngine engine = new InferenceEngine(gs);
                engine.NumberOfIterations = numIters;
                engine.Group(c, a); // Downstream variable is root - not allowed except for GateExitRandom
                engine.Group(c, b); // Downstream variable is root - not allowed except for GateExitRandom
                var marg = engine.Infer<Gaussian>(c);
                Assert.True(false, "An exception should be thrown for the bad root");
            }
            catch (Exception e)
            {
                // There should be some message about the root
                Assert.Contains("root", e.Message.ToLower());
            }
        }

        [Fact]
        public void GibbsProductOfGaussians()
        {
            // The equivalent WinBugs model is:
            // model {
            //    a ~ dnorm(1.0,0.5)
            //    b ~ dnorm(2.0,0.8)
            //    c <- a*b
            // }
            Rand.Restart(12347);
            int burnIn = 1000;
            int numIters = 20000;
            int thin = 1;

            double ma = 1, va = 2;
            double mb = 2, vb = 1/0.8;
            Variable<double> a = Variable.GaussianFromMeanAndVariance(ma, va).Named("a");
            Variable<double> b = Variable.GaussianFromMeanAndVariance(mb, vb).Named("b");
            Variable<double> c = (a*b).Named("c");
            c.AddAttribute(QueryTypes.Marginal);
            c.AddAttribute(QueryTypes.Samples);
            c.AddAttribute(QueryTypes.Conditionals);

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = burnIn;
            gs.Thin = thin;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = numIters;
            engine.ShowProgress = false;
            var cActual = engine.Infer<Gaussian>(c);
            IList<double> samples = engine.Infer<IList<double>>(c, QueryTypes.Samples);
            IList<Gaussian> conditionals = engine.Infer<IList<Gaussian>>(c, QueryTypes.Conditionals);
            Gaussian cExpected = new Gaussian(ma*mb, va*vb + va*mb*mb + vb*ma*ma);
            Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
            Assert.True(cExpected.MaxDiff(cActual) < 2e-3);
            GaussianEstimator est = new GaussianEstimator();
            foreach (double sample in samples)
            {
                est.Add(sample);
            }
            Gaussian cActual2 = est.GetDistribution(new Gaussian());
            Assert.True(cActual.MaxDiff(cActual2) < 1e-8);
            GaussianEstimator est2 = new GaussianEstimator();
            foreach (Gaussian conditional in conditionals)
            {
                est2.Add(conditional);
            }
            Gaussian cActual3 = est2.GetDistribution(new Gaussian());
            Assert.True(cActual.MaxDiff(cActual3) < 1e-8);
            Assert.NotNull(samples);
            Assert.NotNull(conditionals);
            Assert.NotEqual(samples[0], samples[1]);
            Assert.NotEqual(conditionals[0], conditionals[1]);
        }

        // Tests out UsesEqualsDef array arguments for Gibbs sampling
        [Fact]
        public void GibbsScaledGaussianMean()
        {
            Rand.Restart(12347);
            int burnIn = 10;
            int numIters = 1000;
            int thin = 10;

            // Sample data from standard Gaussian
            double[] data = new double[100];
            for (int i = 0; i < data.Length; i++)
                data[i] = 2.0*Rand.Normal(0, 1);

            // Create mean random variables
            Variable<double> m = Variable.GaussianFromMeanAndVariance(0, 100).Named("m");
            Range dataRange = new Range(data.Length).Named("n");
            VariableArray<double> x = Variable.Array<double>(dataRange).Named("x");
            x[dataRange] = Variable.GaussianFromMeanAndPrecision(m, 1.0).ForEach(dataRange);
            VariableArray<double> y = Variable.Array<double>(dataRange).Named("y");
            y[dataRange] = x[dataRange]*2.0;
            y.ObservedValue = data;

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = burnIn;
            gs.Thin = thin;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = numIters;

            // Retrieve the posterior distribution
            var mMarg = engine.Infer<Gaussian>(m);
            double mEst = mMarg.GetMean();
            Assert.True((mEst*mEst) < 10.0*mMarg.GetVariance());
        }

        // Tests out UsesEqualsDef array arguments for Gibbs sampling
        [Fact]
        public void GibbsScaledGaussian()
        {
            Rand.Restart(12347);
            int burnIn = 10;
            int numIters = 100;
            int thin = 10;

            // Sample data from standard Gaussian
            double[] data = new double[100];
            for (int i = 0; i < data.Length; i++)
                data[i] = 2.0*Rand.Normal(0, 1);

            // Create mean and precision random variables
            Variable<double> m = Variable.GaussianFromMeanAndVariance(0, 100).Named("m");
            Variable<double> p = Variable.GammaFromShapeAndScale(1, 1).Named("p");

            Range dataRange = new Range(data.Length).Named("n");
            VariableArray<double> x = Variable.Array<double>(dataRange).Named("x");
            x[dataRange] = Variable.GaussianFromMeanAndPrecision(m, p).ForEach(dataRange);
            VariableArray<double> y = Variable.Array<double>(dataRange).Named("y");
            y[dataRange] = x[dataRange]*2.0;
            y.ObservedValue = data;

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = burnIn;
            gs.Thin = thin;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = numIters;

            // Retrieve the posterior distributions
            var mMarg = engine.Infer<Gaussian>(m);
            var pMarg = engine.Infer<Gamma>(p);
            double mEst = mMarg.GetMean();
            Assert.True((mEst*mEst) < 10.0*mMarg.GetVariance());

            double pEst = pMarg.GetMean();
            Assert.True((pEst - 1.0)*(pEst - 1.0) < 10.0*pMarg.GetVariance());
        }

        [Fact]
        public void GibbsIfTest()
        {
            double pPrior = 0.3;
            double xPriorT = 0.4;
            double xPriorF = 0.2;
            var p = Variable.Bernoulli(pPrior).Named("p");
            Variable<bool> x = Variable.New<bool>().Named("x");
            using (Variable.If(p))
                x.SetTo(Variable.Bernoulli(xPriorT));
            using (Variable.IfNot(p))
                x.SetTo(Variable.Bernoulli(xPriorF));

            Rand.Restart(12347);
            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.NumberOfIterations = 10000;
            engine.ShowProgress = false;
            Bernoulli xExpected = new Bernoulli(xPriorT*pPrior + xPriorF*(1 - pPrior));
            Bernoulli xActual = engine.Infer<Bernoulli>(x);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-1);
        }

        [Fact]
        public void GibbsGateExitConstraintTest()
        {
            double priorB = 0.2;
            double priorX = 0.8;
            double pXCondT = 0.3;
            double pXCondF = 0.4;

            Variable<bool> b = Variable.Bernoulli(priorB).Named("b");
            Variable<bool> x = Variable.New<bool>().Named("x");
            using (Variable.If(b))
            {
                x.SetTo(Variable.Bernoulli(pXCondT));
            }
            using (Variable.IfNot(b))
            {
                x.SetTo(Variable.Bernoulli(pXCondF));
            }
            Variable.ConstrainEqualRandom(x, new Bernoulli(priorX));
            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.NumberOfIterations = 1000000;
            engine.ShowProgress = false;
            double tolerance = 1e-2;

            // p(x,b) =propto (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x)
            //                [(pT)^x (1-pT)^(1-x) (pT2)^x (1-pT2)^(1-x)]^b [(pF)^x (1-pF)^(1-x)]^(1-b)
            double sumXCondT = priorX*pXCondT + (1 - priorX)*(1 - pXCondT);
            double sumXCondF = priorX*pXCondF + (1 - priorX)*(1 - pXCondF);
            double Z = priorB*sumXCondT + (1 - priorB)*sumXCondF;
            double postB = priorB*sumXCondT/Z;
            double postX = priorX*(priorB*pXCondT + (1 - priorB)*pXCondF)/Z;

            Bernoulli bActual = engine.Infer<Bernoulli>(b);
            Bernoulli xActual = engine.Infer<Bernoulli>(x);
            Console.WriteLine("b = {0} (should be {1})", bActual, postB);
            Console.WriteLine("x = {0} (should be {1})", xActual, postX);
            Assert.True(System.Math.Abs(bActual.GetProbTrue() - postB) < tolerance);
            Assert.True(System.Math.Abs(xActual.GetProbTrue() - postX) < tolerance);
        }

        [Fact]
        public void GibbsGateExitConstraintTest2()
        {
            Rand.Restart(0);
            double priorB = 0.1;
            double priorX = 0.2;
            double pXCondT = 0.3;
            double pXCondT2 = 0.6;
            double pXCondF = 0.4;

            Variable<bool> b = Variable.Bernoulli(priorB).Named("b");
            Variable<bool> x = Variable.New<bool>().Named("x");
            using (Variable.If(b))
            {
                x.SetTo(Variable.Bernoulli(pXCondT));
                Variable.ConstrainEqualRandom(x, new Bernoulli(pXCondT2));
            }
            using (Variable.IfNot(b))
            {
                x.SetTo(Variable.Bernoulli(pXCondF));
            }
            Variable.ConstrainEqualRandom(x, new Bernoulli(priorX));
            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.ShowProgress = false;
            engine.NumberOfIterations = 1000000;
            double tolerance = 1e-3;
            b.AddAttribute(QueryTypes.Marginal);
            x.AddAttribute(QueryTypes.Marginal);

            // p(x,b) =propto (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x)
            //                [(pT)^x (1-pT)^(1-x) (pT2)^x (1-pT2)^(1-x)]^b [(pF)^x (1-pF)^(1-x)]^(1-b)
            double sumXCondT = priorX*pXCondT*pXCondT2 + (1 - priorX)*(1 - pXCondT)*(1 - pXCondT2);
            double sumXCondF = priorX*pXCondF + (1 - priorX)*(1 - pXCondF);
            double Z = priorB*sumXCondT + (1 - priorB)*sumXCondF;
            double postB = priorB*sumXCondT/Z;
            double postX = priorX*(priorB*pXCondT*pXCondT2 + (1 - priorB)*pXCondF)/Z;

            Bernoulli bActual = engine.Infer<Bernoulli>(b);
            Bernoulli xActual = engine.Infer<Bernoulli>(x);
            Console.WriteLine("b = {0} (should be {1})", bActual, postB);
            Console.WriteLine("x = {0} (should be {1})", xActual, postX);
            Assert.True(System.Math.Abs(bActual.GetProbTrue() - postB) < tolerance);
            Assert.True(System.Math.Abs(xActual.GetProbTrue() - postX) < tolerance);
        }

        [Fact]
        public void GibbsGateExitConstraintTest3()
        {
            Rand.Restart(12347);
            double priorB = 0.1;
            double priorX = 0.2;
            double pXCondT = 0.9;
            double pXCondT2 = 0.7;
            double pXCondT3 = 0.25;
            double pXCondF = 0.4;

            Variable<bool> b = Variable.Bernoulli(priorB).Named("b");
            Variable<bool> x = Variable.New<bool>().Named("x");
            using (Variable.If(b))
            {
                x.SetTo(Variable.Bernoulli(pXCondT));
                Variable<bool> xCopy = Variable.Copy(x).Named("xCopy");
                Variable.ConstrainEqualRandom(xCopy, new Bernoulli(pXCondT2));
                Variable.ConstrainEqualRandom(xCopy, new Bernoulli(pXCondT3));
            }
            using (Variable.IfNot(b))
            {
                x.SetTo(Variable.Bernoulli(pXCondF));
            }
            Variable.ConstrainEqualRandom(x, new Bernoulli(priorX));
            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.ShowProgress = false;
            engine.NumberOfIterations = 1000000;
            double tolerance = 1e-3;
            b.AddAttribute(QueryTypes.Marginal);
            x.AddAttribute(QueryTypes.Marginal);

            // p(x,b) =propto (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x)
            //                [(pT)^x (1-pT)^(1-x) (pT2)^x (1-pT2)^(1-x)]^b [(pF)^x (1-pF)^(1-x)]^(1-b)
            double sumXCondT = priorX*pXCondT*pXCondT2*pXCondT3 + (1 - priorX)*(1 - pXCondT)*(1 - pXCondT2)*(1 - pXCondT3);
            double sumXCondF = priorX*pXCondF + (1 - priorX)*(1 - pXCondF);
            double Z = priorB*sumXCondT + (1 - priorB)*sumXCondF;
            double postB = priorB*sumXCondT/Z;
            double postX = priorX*(priorB*pXCondT*pXCondT2*pXCondT3 + (1 - priorB)*pXCondF)/Z;

            Bernoulli bActual = engine.Infer<Bernoulli>(b);
            Bernoulli xActual = engine.Infer<Bernoulli>(x);
            Console.WriteLine("b = {0} (should be {1})", bActual, postB);
            Console.WriteLine("x = {0} (should be {1})", xActual, postX);
            Assert.True(System.Math.Abs(bActual.GetProbTrue() - postB) < tolerance);
            Assert.True(System.Math.Abs(xActual.GetProbTrue() - postX) < tolerance);
        }

        [Fact]
        public void GibbsClutterTest()
        {
            Rand.Restart(12347);
            var mean = Variable.GaussianFromMeanAndVariance(0.0, 100.0).Named("mean");
            double prec = 1.0;
            double noiseMean = 0.0;
            double noisePrec = 0.1;
            var b1 = Variable.Bernoulli(0.5).Named("b1");
            var x1 = Variable.New<double>().Named("x1");
            using (Variable.If(b1))
            {
                x1.SetTo(Variable.GaussianFromMeanAndPrecision(mean, prec));
            }
            using (Variable.IfNot(b1))
            {
                x1.SetTo(Variable.GaussianFromMeanAndPrecision(noiseMean, noisePrec));
            }
            x1.ObservedValue = 0.1;

            var b2 = Variable.Bernoulli(0.5).Named("b2");
            var x2 = Variable.New<double>().Named("x2");
            using (Variable.If(b2))
            {
                x2.SetTo(Variable.GaussianFromMeanAndPrecision(mean, prec));
            }
            using (Variable.IfNot(b2))
            {
                x2.SetTo(Variable.GaussianFromMeanAndPrecision(noiseMean, noisePrec));
            }
            x2.ObservedValue = 2.3;

            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.ShowProgress = false;
            engine.NumberOfIterations = 100000;
            engine.OptimiseForVariables = new List<IVariable>() {mean};
            Gaussian meanActual = engine.Infer<Gaussian>(mean);
            Gaussian meanExpected = new Gaussian(0.637152, 51.5467);
            Console.WriteLine("mean = {0} should be {1}", meanActual, meanExpected);
            Assert.True(meanExpected.MaxDiff(meanActual) < 5e-4);
        }

        [Fact]
        public void GibbsBernoulliMixtureTest()
        {
            Rand.Restart(12347);
            int N = 100, K = 2;
            Range n = new Range(N).Named("n");
            Range k = new Range(K).Named("k");
            VariableArray<double> p = Variable.Array<double>(k).Named("p");
            p[k] = Variable.Beta(1, 1).ForEach(k);
            VariableArray<bool> x = Variable.Array<bool>(n).Named("x");
            VariableArray<int> c = Variable.Array<int>(n).Named("c");
            using (Variable.ForEach(n))
            {
                c[n] = Variable.Discrete(k, 0.5, 0.5);
                using (Variable.Switch(c[n]))
                {
                    x[n] = Variable.Bernoulli(p[c[n]]);
                }
            }
            InferenceEngine engine = new InferenceEngine(new GibbsSampling() {});
            bool[] data = new bool[N];
            for (int i = 0; i < N; i++)
                data[i] = (i%2) == 0 ? true : false;
            x.ObservedValue = data;
            Discrete[] cInit = new Discrete[N];
            for (int j = 0; j < N; j++)
            {
                double r = Rand.Double();
                cInit[j] = new Discrete(r, 1 - r);
            }
            c.InitialiseTo(Distribution<int>.Array(cInit));
            c.AddAttribute(QueryTypes.Marginal);
            p.AddAttribute(QueryTypes.Marginal);
            c.AddAttribute(QueryTypes.Conditionals);
            p.AddAttribute(QueryTypes.Conditionals);
            var cPost = engine.Infer<Discrete[]>(c);
            var pPost = engine.Infer<Beta[]>(p);
            var pSamplesPost = engine.Infer<IList<DistributionStructArray<Beta, double>>>(p, QueryTypes.Conditionals);
            for (int i = 0; i < 20; i++)
                Console.WriteLine(StringUtil.CollectionToString(pSamplesPost[i], " "));

            // Not much we can assert about this
        }

        [Fact]
        public void GibbsBernoulliArrayMixtureTest()
        {
            Rand.Restart(12347);
            int N = 10, D = 2, K = 2;
            Range n = new Range(N).Named("n");
            Range k = new Range(K).Named("k");
            Range d = new Range(D).Named("d");
            VariableArray2D<double> p = Variable.Array<double>(k, d).Named("p");
            p[k, d] = Variable.Beta(1, 1).ForEach(k, d);
            VariableArray2D<bool> x = Variable.Array<bool>(n, d).Named("x");
            VariableArray<int> c = Variable.Array<int>(n).Named("c");
            using (Variable.ForEach(n))
            {
                c[n] = Variable.Discrete(k, 0.5, 0.5);
                using (Variable.Switch(c[n]))
                {
                    x[n, d] = Variable.Bernoulli(p[c[n], d]);
                }
            }
            InferenceEngine engine = new InferenceEngine(new GibbsSampling() {BurnIn = 2000});
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;
            bool[,] data = new bool[N,D];
            int N1 = N/2;
            int i = 0;
            for (; i < N1; i++)
            {
                data[i, 0] = true;
                data[i, 1] = false;
            }
            for (; i < N; i++)
            {
                data[i, 0] = false;
                data[i, 1] = true;
            }
            x.ObservedValue = data;
            Discrete[] cInit = new Discrete[N];
            for (int j = 0; j < N; j++)
            {
                double r = Rand.Double();
                cInit[j] = new Discrete(r, 1 - r);
            }
            c.InitialiseTo(Distribution<int>.Array(cInit));
            var cPost = engine.Infer<Discrete[]>(c);
            var pPost = engine.Infer<Beta[,]>(p);
            // Not much we can assert about this
        }

        // This is not a test - more of an example, really
        internal void GibbsAllAlgorithms()
        {
            Rand.Restart(12347);

            // The model
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100).Named("mean");
            Variable<double> precision = Variable.GammaFromShapeAndRate(1, 1).Named("precision");
            Range i = new Range(2).Named("i");
            VariableArray<double> data = Variable.Array<double>(i).Named("data");
            data[i] = Variable.GaussianFromMeanAndPrecision(mean, precision).ForEach(i);

            // The observations
            data.ObservedValue = new double[] {5.0, 7.0};

            Console.WriteLine("Running Variational Message Passing");
            InferenceEngine engineVMP = new InferenceEngine(new VariationalMessagePassing());
            Gaussian meanVMP = engineVMP.Infer<Gaussian>(mean);
            Gamma precVMP = engineVMP.Infer<Gamma>(precision);

            Console.WriteLine("Running Expectation Propagation");
            InferenceEngine engineEP = new InferenceEngine(new ExpectationPropagation());
            Gaussian meanEP = engineEP.Infer<Gaussian>(mean);
            Gamma precEP = engineEP.Infer<Gamma>(precision);

            Console.WriteLine("Running Gibbs Sampling");
            InferenceEngine engineGS = new InferenceEngine(new GibbsSampling() {BurnIn = 100, Thin = 10});
            engineGS.NumberOfIterations = 1000;
            Gaussian meanGS = engineGS.Infer<Gaussian>(mean);
            Gamma precGS = engineGS.Infer<Gamma>(precision);

            Console.WriteLine("\t\tEP\tVMP\tGS");
            Console.WriteLine(
                String.Format("Mean mean:\t{0:0.00}\t{1:0.00}\t{2:0.00}",
                              meanEP.GetMean(), meanVMP.GetMean(), meanGS.GetMean()));
            Console.WriteLine(
                string.Format("Mean sdev:\t{0:0.00}\t{1:0.00}\t{2:0.00}",
                              System.Math.Sqrt(meanEP.GetVariance()), System.Math.Sqrt(meanVMP.GetVariance()), System.Math.Sqrt(meanGS.GetVariance())));
            Console.WriteLine(
                String.Format("Prec mean:\t{0:0.00}\t{1:0.00}\t{2:0.00}",
                              precEP.GetMean(), precVMP.GetMean(), precGS.GetMean()));
            Console.WriteLine(
                string.Format("Prec sdev:\t{0:0.00}\t{1:0.00}\t{2:0.00}",
                              System.Math.Sqrt(precEP.GetVariance()), System.Math.Sqrt(precVMP.GetVariance()), System.Math.Sqrt(precGS.GetVariance())));
        }

        //-----------------------------------------------------------------------
        // Following tests are adapted from Infer.NET 1
        //-----------------------------------------------------------------------

        [Fact]
        public void GibbsGaussian1rrr()
        {
            Rand.Restart(12347);
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(7, 0.3333333333333333).Named("mean");
            Variable<double> precision = Variable.GammaFromShapeAndRate(3.0, 1.0).Named("precision");
            Variable<double> x = Variable.GaussianFromMeanAndPrecision(mean, precision).Named("x");
            Variable.ConstrainEqualRandom(x, Gaussian.FromMeanAndVariance(3, 0.5));

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = 1000;
            gs.Thin = 10;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;

            var xMarginal = engine.Infer<Gaussian>(x);
            Assert.True(MMath.AbsDiff(4.03993028599997, xMarginal.GetMean(), 0.1) < 0.1);
            Assert.True(MMath.AbsDiff(0.54, xMarginal.GetVariance(), 0.1) < 0.1);

            var meanMarginal = engine.Infer<Gaussian>(mean);
            Assert.True(MMath.AbsDiff(6.304764059999949, meanMarginal.GetMean(), 0.1) < 0.1);
            Assert.True(MMath.AbsDiff(0.3524947510873381, meanMarginal.GetVariance(), 0.1) < 0.1);

            var precMarginal = engine.Infer<Gamma>(precision);
            Assert.True(MMath.AbsDiff(1.207642644600002, precMarginal.GetMean(), 0.1) < 0.1);
            Assert.True(MMath.AbsDiff(1.066889976363217, precMarginal.GetVariance(), 0.1) < 0.1);
        }

        [Fact]
        public void GibbsGaussian1crr()
        {
            Rand.Restart(12347);
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(7, 0.3333333333333333).Named("mean");
            Variable<double> precision = Variable.GammaFromShapeAndRate(3, 1).Named("precision");
            Variable<double> x = Variable.GaussianFromMeanAndPrecision(mean, precision).Named("x");
            x.ObservedValue = 3.0;
            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = 1000;
            gs.Thin = 10;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;

            var meanMarginal = engine.Infer<Gaussian>(mean);
            Assert.True(MMath.AbsDiff(6.407071739999963, meanMarginal.GetMean(), 0.1) < 0.1);
            Assert.True(MMath.AbsDiff(0.3768844030014321, meanMarginal.GetVariance(), 0.1) < 0.1);

            var precMarginal = engine.Infer<Gamma>(precision);
            Assert.True(MMath.AbsDiff(0.5516397604400063, precMarginal.GetMean(), 0.1) < 0.1);
            Assert.True(MMath.AbsDiff(0.1328864249406182, precMarginal.GetVariance(), 0.1) < 0.1);
        }

        [Fact]
        public void GibbsGaussian1rcr()
        {
            Rand.Restart(12347);
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(7, 0.3333333333333333).Named("mean");
            Variable<double> precision = Variable.GammaFromShapeAndRate(3.0, 1.0).Named("precision");
            Variable<double> x = Variable.GaussianFromMeanAndPrecision(mean, precision).Named("x");
            Variable.ConstrainEqualRandom(x, Gaussian.FromMeanAndVariance(3, 0.5));
            mean.ObservedValue = 7.0;
            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = 1000;
            gs.Thin = 10;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;

            var xMarginal = engine.Infer<Gaussian>(x);
            Assert.True(MMath.AbsDiff(3.946624517999992, xMarginal.GetMean(), 0.1) < 0.1);
            Assert.True(MMath.AbsDiff(0.6111880684856339, xMarginal.GetVariance(), 0.1) < 0.1);

            var precMarginal = engine.Infer<Gamma>(precision);
            Assert.True(MMath.AbsDiff(0.704465172599999, precMarginal.GetMean(), 0.1) < 0.1);
            Assert.True(MMath.AbsDiff(0.2988971440860863, precMarginal.GetVariance(), 0.1) < 0.1);
        }

        [Fact]
        public void GibbsGaussian1ccr()
        {
            Rand.Restart(12347);
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(7, 0.3333333333333333).Named("mean");
            Variable<double> precision = Variable.GammaFromShapeAndRate(3.0, 1.0).Named("precision");
            Variable<double> x = Variable.GaussianFromMeanAndPrecision(mean, precision).Named("x");
            x.ObservedValue = 3.0;
            mean.ObservedValue = 7.0;

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = 1000;
            gs.Thin = 10;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;

            var precMarginal = engine.Infer<Gamma>(precision);
            Assert.True(MMath.AbsDiff(0.3881633502, precMarginal.GetMean(), 0.1) < 0.1);
            Assert.True(MMath.AbsDiff(0.04303535914668142, precMarginal.GetVariance(), 0.1) < 0.1);
        }

        [Fact]
        public void GibbsGaussian1rrc()
        {
            Rand.Restart(12347);
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(7, 0.3333333333333333).Named("mean");
            Variable<double> precision = Variable.GammaFromShapeAndRate(3.0, 1.0).Named("precision");
            Variable<double> x = Variable.GaussianFromMeanAndPrecision(mean, precision).Named("x");
            Variable.ConstrainEqualRandom(x, Gaussian.FromMeanAndVariance(3, 0.5));
            precision.ObservedValue = 2.0;

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = 1000;
            gs.Thin = 10;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;

            var xMarginal = engine.Infer<Gaussian>(x);
            Assert.True(MMath.AbsDiff(4.495638980000003, xMarginal.GetMean(), 0.1) < 0.1);
            Assert.True(MMath.AbsDiff(0.3100127476995136, xMarginal.GetVariance(), 0.1) < 0.1);

            var meanMarginal = engine.Infer<Gaussian>(mean);
            Assert.True(MMath.AbsDiff(5.994344560000011, meanMarginal.GetMean(), 0.1) < 0.1);
            Assert.True(MMath.AbsDiff(0.2498059743578944, meanMarginal.GetVariance(), 0.1) < 0.1);
        }

        [Fact]
        public void GibbsGaussian1crc()
        {
            Rand.Restart(12347);
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(7, 0.3333333333333333).Named("mean");
            Variable<double> precision = Variable.GammaFromShapeAndRate(3.0, 1.0).Named("precision");
            Variable<double> x = Variable.GaussianFromMeanAndPrecision(mean, precision).Named("x");
            x.ObservedValue = 3.0;
            precision.ObservedValue = 2.0;

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = 1000;
            gs.Thin = 10;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;

            var meanMarginal = engine.Infer<Gaussian>(mean);
            Assert.True(MMath.AbsDiff(5.397419000000007, meanMarginal.GetMean(), 0.1) < 0.1);
            Assert.True(MMath.AbsDiff(0.1984488342156844, meanMarginal.GetVariance(), 0.1) < 0.1);
        }

        [Fact]
        public void GibbsGaussian1rcc()
        {
            Rand.Restart(12347);
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(7, 0.3333333333333333).Named("mean");
            Variable<double> precision = Variable.GammaFromShapeAndRate(3.0, 1.0).Named("precision");
            Variable<double> x = Variable.GaussianFromMeanAndPrecision(mean, precision).Named("x");
            Variable.ConstrainEqualRandom(x, Gaussian.FromMeanAndVariance(3, 0.5));
            mean.ObservedValue = 7.0;
            precision.ObservedValue = 2.0;

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = 1000;
            gs.Thin = 10;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;

            var xMarginal = engine.Infer<Gaussian>(x);
            Assert.True(MMath.AbsDiff(4.997117940000017, xMarginal.GetMean(), 0.1) < 0.1);
            Assert.True(MMath.AbsDiff(0.2480602044742462, xMarginal.GetVariance(), 0.1) < 0.1);
        }

        [Fact]
        public void GibbsGaussian1prrr()
        {
            Rand.Restart(12347);
            int P = 3;
            Range i = new Range(P);
            VariableArray<double> mean = Variable.Array<double>(i).Named("mean");
            VariableArray<double> precision = Variable.Array<double>(i).Named("precision");
            VariableArray<double> x = Variable.Array<double>(i).Named("x");
            mean[i] = Variable.GaussianFromMeanAndVariance(7, 0.3333333333333333).ForEach(i);
            precision[i] = Variable.GammaFromShapeAndRate(3.0, 1.0).ForEach(i);
            x[i] = Variable.GaussianFromMeanAndPrecision(mean[i], precision[i]);
            Variable.ConstrainEqualRandom(x[i], Gaussian.FromMeanAndVariance(3, 0.5));

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = 1000;
            gs.Thin = 10;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;

            var xMarginal = engine.Infer<Gaussian[]>(x);
            for (int p = 0; p < P; p++)
            {
                Assert.True(MMath.AbsDiff(4.03993028599997, xMarginal[p].GetMean(), 0.1) < 0.1);
                Assert.True(MMath.AbsDiff(0.5442163936456329, xMarginal[p].GetVariance(), 0.1) < 0.15);
            }

            var meanMarginal = engine.Infer<Gaussian[]>(mean);
            for (int p = 0; p < P; p++)
            {
                Assert.True(MMath.AbsDiff(6.304764059999949, meanMarginal[p].GetMean(), 0.1) < 0.1);
                Assert.True(MMath.AbsDiff(0.3524947510873381, meanMarginal[p].GetVariance(), 0.1) < 0.1);
            }

            var precMarginal = engine.Infer<Gamma[]>(precision);
            for (int p = 0; p < P; p++)
            {
                Assert.True(MMath.AbsDiff(1.207642644600002, precMarginal[p].GetMean(), 0.1) < 0.1);
                Assert.True(MMath.AbsDiff(1.066889976363217, precMarginal[p].GetVariance(), 0.1) < 0.1);
            }
        }

        [Fact]
        public void GibbsGaussian1pcrr()
        {
            Rand.Restart(12347);
            int P = 3;
            Range i = new Range(P);
            VariableArray<double> mean = Variable.Array<double>(i).Named("mean");
            VariableArray<double> precision = Variable.Array<double>(i).Named("precision");
            VariableArray<double> x = Variable.Array<double>(i).Named("x");

            mean[i] = Variable.GaussianFromMeanAndVariance(7, 0.3333333333333333).ForEach(i);
            precision[i] = Variable.GammaFromShapeAndRate(3.0, 1.0).ForEach(i);
            x[i] = Variable.GaussianFromMeanAndPrecision(mean[i], precision[i]);
            x.ObservedValue = new double[] {3.0, 3.0, 3.0};

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = 1000;
            gs.Thin = 10;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;

            var meanMarginal = engine.Infer<Gaussian[]>(mean);
            for (int p = 0; p < P; p++)
            {
                Assert.True(MMath.AbsDiff(6.407071739999963, meanMarginal[p].GetMean(), 0.1) < 0.1);
                Assert.True(MMath.AbsDiff(0.3768844030014321, meanMarginal[p].GetVariance(), 0.1) < 0.1);
            }

            var precMarginal = engine.Infer<Gamma[]>(precision);
            for (int p = 0; p < P; p++)
            {
                Assert.True(MMath.AbsDiff(0.5516397604400063, precMarginal[p].GetMean(), 0.1) < 0.1);
                Assert.True(MMath.AbsDiff(0.1328864249406182, precMarginal[p].GetVariance(), 0.1) < 0.1);
            }
        }

        [Fact]
        public void GibbsGaussian1prcr()
        {
            Rand.Restart(12347);
            int P = 3;
            Range i = new Range(P);
            VariableArray<double> mean = Variable.Array<double>(i).Named("mean");
            VariableArray<double> precision = Variable.Array<double>(i).Named("precision");
            VariableArray<double> x = Variable.Array<double>(i).Named("x");

            mean[i] = Variable.GaussianFromMeanAndVariance(7, 0.3333333333333333).ForEach(i);
            precision[i] = Variable.GammaFromShapeAndRate(3.0, 1.0).ForEach(i);
            x[i] = Variable.GaussianFromMeanAndPrecision(mean[i], precision[i]);
            Variable.ConstrainEqualRandom(x[i], Gaussian.FromMeanAndVariance(3, 0.5));
            mean.ObservedValue = new double[] {7.0, 7.0, 7.0};

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = 1000;
            gs.Thin = 10;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;

            var xMarginal = engine.Infer<Gaussian[]>(x);
            for (int p = 0; p < P; p++)
            {
                Assert.True(MMath.AbsDiff(3.946624517999992, xMarginal[p].GetMean(), 0.1) < 0.1);
                Assert.True(MMath.AbsDiff(0.6111880684856339, xMarginal[p].GetVariance(), 0.1) < 0.1);
            }

            var precMarginal = engine.Infer<Gamma[]>(precision);
            for (int p = 0; p < P; p++)
            {
                Assert.True(MMath.AbsDiff(0.704465172599999, precMarginal[p].GetMean(), 0.1) < 0.1);
                Assert.True(MMath.AbsDiff(0.2988971440860863, precMarginal[p].GetVariance(), 0.1) < 0.1);
            }
        }

        [Fact]
        public void GibbsGaussian1pccr()
        {
            Rand.Restart(12347);
            int P = 3;
            Range i = new Range(P);
            VariableArray<double> mean = Variable.Array<double>(i).Named("mean");
            VariableArray<double> precision = Variable.Array<double>(i).Named("precision");
            VariableArray<double> x = Variable.Array<double>(i).Named("x");

            mean[i] = Variable.GaussianFromMeanAndVariance(7, 0.3333333333333333).ForEach(i);
            precision[i] = Variable.GammaFromShapeAndRate(3.0, 1.0).ForEach(i);
            x[i] = Variable.GaussianFromMeanAndPrecision(mean[i], precision[i]);
            x.ObservedValue = new double[] {3.0, 3.0, 3.0};
            mean.ObservedValue = new double[] {7.0, 7.0, 7.0};

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = 1000;
            gs.Thin = 10;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;

            var precMarginal = engine.Infer<Gamma[]>(precision);
            for (int p = 0; p < P; p++)
            {
                Assert.True(MMath.AbsDiff(0.3881633502, precMarginal[p].GetMean(), 0.1) < 0.1);
                Assert.True(MMath.AbsDiff(0.04303535914668142, precMarginal[p].GetVariance(), 0.1) < 0.1);
            }
        }

        [Fact]
        public void GibbsGaussian1prrc()
        {
            Rand.Restart(12347);
            int P = 3;
            Range i = new Range(P);
            VariableArray<double> mean = Variable.Array<double>(i).Named("mean");
            VariableArray<double> precision = Variable.Array<double>(i).Named("precision");
            VariableArray<double> x = Variable.Array<double>(i).Named("x");

            mean[i] = Variable.GaussianFromMeanAndVariance(7, 0.3333333333333333).ForEach(i);
            precision[i] = Variable.GammaFromShapeAndRate(3.0, 1.0).ForEach(i);
            x[i] = Variable.GaussianFromMeanAndPrecision(mean[i], precision[i]);
            Variable.ConstrainEqualRandom(x[i], Gaussian.FromMeanAndVariance(3, 0.5));
            precision.ObservedValue = new double[] {2.0, 2.0, 2.0};

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = 1000;
            gs.Thin = 10;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;

            var xMarginal = engine.Infer<Gaussian[]>(x);
            for (int p = 0; p < P; p++)
            {
                Assert.True(MMath.AbsDiff(4.495638980000003, xMarginal[p].GetMean(), 0.1) < 0.1);
                Assert.True(MMath.AbsDiff(0.3100127476995136, xMarginal[p].GetVariance(), 0.1) < 0.1);
            }

            var meanMarginal = engine.Infer<Gaussian[]>(mean);
            for (int p = 0; p < P; p++)
            {
                Assert.True(MMath.AbsDiff(5.994344560000011, meanMarginal[p].GetMean(), 0.1) < 0.1);
                Assert.True(MMath.AbsDiff(0.2498059743578944, meanMarginal[p].GetVariance(), 0.1) < 0.1);
            }
        }

        [Fact]
        public void GibbsGaussian1pcrc()
        {
            Rand.Restart(12347);
            int P = 3;
            Range i = new Range(P);
            VariableArray<double> mean = Variable.Array<double>(i).Named("mean");
            VariableArray<double> precision = Variable.Array<double>(i).Named("precision");
            VariableArray<double> x = Variable.Array<double>(i).Named("x");

            mean[i] = Variable.GaussianFromMeanAndVariance(7, 0.3333333333333333).ForEach(i);
            precision[i] = Variable.GammaFromShapeAndRate(3.0, 1.0).ForEach(i);
            x[i] = Variable.GaussianFromMeanAndPrecision(mean[i], precision[i]);
            x.ObservedValue = new double[] {3.0, 3.0, 3.0};
            precision.ObservedValue = new double[] {2.0, 2.0, 2.0};

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = 1000;
            gs.Thin = 10;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;

            var meanMarginal = engine.Infer<Gaussian[]>(mean);
            for (int p = 0; p < P; p++)
            {
                Assert.True(MMath.AbsDiff(5.397419000000007, meanMarginal[p].GetMean(), 0.1) < 0.1);
                Assert.True(MMath.AbsDiff(0.1984488342156844, meanMarginal[p].GetVariance(), 0.1) < 0.1);
            }
        }

        [Fact]
        public void GibbsGaussian1prcc()
        {
            Rand.Restart(12347);
            int P = 3;
            Range i = new Range(P);
            VariableArray<double> mean = Variable.Array<double>(i).Named("mean");
            VariableArray<double> precision = Variable.Array<double>(i).Named("precision");
            VariableArray<double> x = Variable.Array<double>(i).Named("x");

            mean[i] = Variable.GaussianFromMeanAndVariance(7, 0.3333333333333333).ForEach(i);
            precision[i] = Variable.GammaFromShapeAndRate(3.0, 1.0).ForEach(i);
            x[i] = Variable.GaussianFromMeanAndPrecision(mean[i], precision[i]);
            Variable.ConstrainEqualRandom(x[i], Gaussian.FromMeanAndVariance(3, 0.5));
            mean.ObservedValue = new double[] {7.0, 7.0, 7.0};
            precision.ObservedValue = new double[] {2.0, 2.0, 2.0};

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = 1000;
            gs.Thin = 10;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;

            var xMarginal = engine.Infer<Gaussian[]>(x);
            for (int p = 0; p < P; p++)
            {
                Assert.True(MMath.AbsDiff(4.997117940000017, xMarginal[p].GetMean(), 0.1) < 0.1);
                Assert.True(MMath.AbsDiff(0.2480602044742462, xMarginal[p].GetVariance(), 0.1) < 0.1);
            }
        }

        [Fact]
        public void GibbsGaussian2rrr()
        {
            Rand.Restart(12347);
            Variable<Vector> mean = Variable.VectorGaussianFromMeanAndVariance(
                Vector.FromArray(7.400000000000001, 9.4),
                new PositiveDefiniteMatrix(new double[,] {{0.5999999999999999, -0.4}, {-0.4, 0.6}})).Named("mean");
            Variable<PositiveDefiniteMatrix> precision = Variable.Random<PositiveDefiniteMatrix>(
                Wishart.FromShapeAndRate(4.5, new PositiveDefiniteMatrix(new double[,] {{4.5, -3}, {-3, 3}}))).Named("precision");
            Variable<Vector> x = Variable.VectorGaussianFromMeanAndPrecision(mean, precision).Named("x");
            Variable.ConstrainEqualRandom(
                x,
                VectorGaussian.FromMeanAndVariance(
                    Vector.FromArray(1.666666666666666, 2.666666666666667),
                    new PositiveDefiniteMatrix(new double[,] {{0.6666666666666666, -0.3333333333333333}, {-0.3333333333333333, 0.6666666666666666}})));

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = 1000;
            gs.Thin = 10;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;

            var xMarginal = engine.Infer<VectorGaussian>(x);
            Assert.True(Vector.FromArray(1.853720757423829, 3.046769437199992).MaxDiff(xMarginal.GetMean(), 0.1) < 0.1);
            // TODO: Investigate
            Assert.True(new PositiveDefiniteMatrix(new double[,] {{0.66, -0.31}, {-0.31, 0.66}}).MaxDiff(xMarginal.GetVariance(), 0.1) < 0.1);

            var meanMarginal = engine.Infer<VectorGaussian>(mean);
            Assert.True(Vector.FromArray(7.320509300000029, 9.138372700000055).MaxDiff(meanMarginal.GetMean(), 0.1) < 0.1);
            Assert.True(
                new PositiveDefiniteMatrix(new double[,] {{0.5899341223559563, -0.3870740576772629}, {-0.3870740576772629, 0.5945957492697019}}).MaxDiff(
                    meanMarginal.GetVariance(), 0.1) < 0.1);

            var precMarginal = engine.Infer<Wishart>(precision);
            Assert.True(
                new PositiveDefiniteMatrix(new double[,] {{0.4696463033999955, -0.2806245454862432}, {-0.2806245454862432, 0.4338032545999959}}).MaxDiff(
                    precMarginal.GetMean(), 0.1) < 0.1);
        }

        [Fact]
        public void GibbsGaussian2crr()
        {
            Rand.Restart(12347);
            Variable<Vector> mean = Variable.VectorGaussianFromMeanAndVariance(
                Vector.FromArray(7.400000000000001, 9.4), new PositiveDefiniteMatrix(new double[,] {{0.5999999999999999, -0.4}, {-0.4, 0.6}})).Named("mean");
            Variable<PositiveDefiniteMatrix> precision = Variable.Random<PositiveDefiniteMatrix>(
                Wishart.FromShapeAndRate(4.5, new PositiveDefiniteMatrix(new double[,] {{4.5, -3}, {-3, 3}}))).Named("precision");
            Variable<Vector> x = Variable.VectorGaussianFromMeanAndPrecision(mean, precision).Named("x");
            x.ObservedValue = Vector.FromArray(1.666666666666666, 2.666666666666667);

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = 1000;
            gs.Thin = 10;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;

            var meanMarginal = engine.Infer<VectorGaussian>(mean);
            Assert.True(Vector.FromArray(7.331047820000016, 9.141518279999941).MaxDiff(meanMarginal.GetMean(), 0.1) < 0.1);
            Assert.True(
                new PositiveDefiniteMatrix(new double[,] {{0.5848910647545453, -0.3863803274506987}, {-0.3863803274506987, 0.5927857329404954}}).MaxDiff(
                    meanMarginal.GetVariance(), 0.1) < 0.1);

            var precMarginal = engine.Infer<Wishart>(precision);
            Assert.True(
                new PositiveDefiniteMatrix(new double[,] {{0.468112459199998, -0.2928359974098008}, {-0.2928359974098008, 0.4070885105999972}}).MaxDiff(
                    precMarginal.GetMean(), 0.1) < 0.1);
        }

        [Fact]
        public void GibbsGaussian2rcr()
        {
            Rand.Restart(12347);
            Variable<Vector> mean = Variable.VectorGaussianFromMeanAndVariance(
                Vector.FromArray(7.400000000000001, 9.4), new PositiveDefiniteMatrix(new double[,] {{0.5999999999999999, -0.4}, {-0.4, 0.6}})).Named("mean");
            Variable<PositiveDefiniteMatrix> precision = Variable.Random<PositiveDefiniteMatrix>(
                Wishart.FromShapeAndRate(4.5, new PositiveDefiniteMatrix(new double[,] {{4.5, -3}, {-3, 3}}))).Named("precision");
            Variable<Vector> x = Variable.VectorGaussianFromMeanAndPrecision(mean, precision).Named("x");
            Variable.ConstrainEqualRandom(
                x,
                VectorGaussian.FromMeanAndVariance(
                    Vector.FromArray(1.666666666666666, 2.666666666666667),
                    new PositiveDefiniteMatrix(new double[,] {{0.6666666666666666, -0.3333333333333333}, {-0.3333333333333333, 0.6666666666666666}})));
            mean.ObservedValue = Vector.FromArray(7.4, 9.4);

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = 1000;
            gs.Thin = 10;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;

            var xMarginal = engine.Infer<VectorGaussian>(x);
            Assert.True(Vector.FromArray(1.835194393030008, 3.036958674348004).MaxDiff(xMarginal.GetMean(), 0.1) < 0.1);
            Assert.True(
                new PositiveDefiniteMatrix(new double[,] {{0.6546096499201396, -0.3165004368759269}, {-0.3165004368759269, 0.6704558698104106}}).MaxDiff(
                    xMarginal.GetVariance(), 0.1) < 0.1);

            var precMarginal = engine.Infer<Wishart>(precision);
            Assert.True(
                new PositiveDefiniteMatrix(new double[,] {{0.4705598439999973, -0.2899506201464017}, {-0.2899506201464017, 0.410642632399999}}).MaxDiff(
                    precMarginal.GetMean(), 0.1) < 0.1);
        }

        [Fact]
        public void GibbsGaussian2ccr()
        {
            Rand.Restart(12347);
            Variable<Vector> mean = Variable.VectorGaussianFromMeanAndVariance(
                Vector.FromArray(7.400000000000001, 9.4), new PositiveDefiniteMatrix(new double[,] {{0.5999999999999999, -0.4}, {-0.4, 0.6}})).Named("mean");
            Variable<PositiveDefiniteMatrix> precision = Variable.Random<PositiveDefiniteMatrix>(
                Wishart.FromShapeAndRate(4.5, new PositiveDefiniteMatrix(new double[,] {{4.5, -3}, {-3, 3}}))).Named("precision");
            Variable<Vector> x = Variable.VectorGaussianFromMeanAndPrecision(mean, precision).Named("x");
            x.ObservedValue = Vector.FromArray(1.666666666666666, 2.666666666666667);
            mean.ObservedValue = Vector.FromArray(7.4, 9.4);

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = 1000;
            gs.Thin = 10;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;

            var precMarginal = engine.Infer<Wishart>(precision);
            Assert.True(
                new PositiveDefiniteMatrix(new double[,] {{0.4720626760000004, -0.2991924347768026}, {-0.2991924347768026, 0.3843250634000018}}).MaxDiff(
                    precMarginal.GetMean(), 0.1) < 0.1);
        }

        [Fact]
        public void GibbsGaussian2rrc()
        {
            Rand.Restart(12347);
            Variable<Vector> mean = Variable.VectorGaussianFromMeanAndVariance(
                Vector.FromArray(7.400000000000001, 9.4), new PositiveDefiniteMatrix(new double[,] {{0.5999999999999999, -0.4}, {-0.4, 0.6}})).Named("mean");
            Variable<PositiveDefiniteMatrix> precision = Variable.Random<PositiveDefiniteMatrix>(
                Wishart.FromShapeAndRate(4.5, new PositiveDefiniteMatrix(new double[,] {{4.5, -3}, {-3, 3}}))).Named("precision");
            Variable<Vector> x = Variable.VectorGaussianFromMeanAndPrecision(mean, precision).Named("x");
            Variable.ConstrainEqualRandom(
                x,
                VectorGaussian.FromMeanAndVariance(
                    Vector.FromArray(1.666666666666666, 2.666666666666667),
                    new PositiveDefiniteMatrix(new double[,] {{0.6666666666666666, -0.3333333333333333}, {-0.3333333333333333, 0.6666666666666666}})));
            precision.ObservedValue = new PositiveDefiniteMatrix(new double[,] {{2, 2}, {2, 3}});

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = 1000;
            gs.Thin = 10;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;

            var xMarginal = engine.Infer<VectorGaussian>(x);
            Assert.True(Vector.FromArray(3.783132303999977, 5.977692540000003).MaxDiff(xMarginal.GetMean(), 0.1) < 0.1);
            Assert.True(
                new PositiveDefiniteMatrix(new double[,] {{0.4978722877734079, -0.2848076259943319}, {-0.2848076259943319, 0.4458780131686158}}).MaxDiff(
                    xMarginal.GetVariance(), 0.1) < 0.1);

            var meanMarginal = engine.Infer<VectorGaussian>(mean);
            Assert.True(Vector.FromArray(6.365791720000041, 7.169097460000024).MaxDiff(meanMarginal.GetMean(), 0.1) < 0.1);
            Assert.True(
                new PositiveDefiniteMatrix(new double[,] {{0.4690469522384849, -0.3064125998530273}, {-0.3064125998530273, 0.4387578736790226}}).MaxDiff(
                    meanMarginal.GetVariance(), 0.1) < 0.1);
        }

        [Fact]
        public void GibbsGaussian2crc()
        {
            Rand.Restart(12347);
            Variable<Vector> mean = Variable.VectorGaussianFromMeanAndVariance(
                Vector.FromArray(7.400000000000001, 9.4), new PositiveDefiniteMatrix(new double[,] {{0.5999999999999999, -0.4}, {-0.4, 0.6}})).Named("mean");
            Variable<PositiveDefiniteMatrix> precision = Variable.Random<PositiveDefiniteMatrix>(
                Wishart.FromShapeAndRate(4.5, new PositiveDefiniteMatrix(new double[,] {{4.5, -3}, {-3, 3}}))).Named("precision");
            Variable<Vector> x = Variable.VectorGaussianFromMeanAndPrecision(mean, precision).Named("x");
            x.ObservedValue = Vector.FromArray(1.666666666666666, 2.666666666666667);
            precision.ObservedValue = new PositiveDefiniteMatrix(new double[,] {{2, 2}, {2, 3}});

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = 1000;
            gs.Thin = 10;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;

            var meanMarginal = engine.Infer<VectorGaussian>(mean);
            Assert.True(Vector.FromArray(5.759463460000001, 5.212455740000014).MaxDiff(meanMarginal.GetMean(), 0.1) < 0.1);
            Assert.True(
                new PositiveDefiniteMatrix(new double[,] {{0.4291474471337715, -0.2860789439761391}, {-0.2860789439761391, 0.356087414789347}}).MaxDiff(
                    meanMarginal.GetVariance(), 0.1) < 0.1);
        }

        [Fact]
        public void GibbsGaussian2rcc()
        {
            Rand.Restart(12347);
            Variable<Vector> mean = Variable.VectorGaussianFromMeanAndVariance(
                Vector.FromArray(7.400000000000001, 9.4), new PositiveDefiniteMatrix(new double[,] {{0.5999999999999999, -0.4}, {-0.4, 0.6}})).Named("mean");
            Variable<PositiveDefiniteMatrix> precision = Variable.Random<PositiveDefiniteMatrix>(
                Wishart.FromShapeAndRate(4.5, new PositiveDefiniteMatrix(new double[,] {{4.5, -3}, {-3, 3}}))).Named("precision");
            Variable<Vector> x = Variable.VectorGaussianFromMeanAndPrecision(mean, precision).Named("x");
            Variable.ConstrainEqualRandom(
                x,
                VectorGaussian.FromMeanAndVariance(
                    Vector.FromArray(1.666666666666666, 2.666666666666667),
                    new PositiveDefiniteMatrix(new double[,] {{0.6666666666666666, -0.3333333333333333}, {-0.3333333333333333, 0.6666666666666666}})));
            mean.ObservedValue = Vector.FromArray(7.4, 9.4);
            precision.ObservedValue = new PositiveDefiniteMatrix(new double[,] {{2, 2}, {2, 3}});

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = 1000;
            gs.Thin = 10;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;

            var xMarginal = engine.Infer<VectorGaussian>(x);
            Assert.True(Vector.FromArray(4.360657580000011, 7.379968300000039).MaxDiff(xMarginal.GetMean(), 0.1) < 0.1);
            Assert.True(
                new PositiveDefiniteMatrix(new double[,] {{0.4551355217589777, -0.27328615229776}, {-0.27328615229776, 0.3625638447720047}}).MaxDiff(xMarginal.GetVariance(),
                                                                                                                                                     0.1) < 0.1);
        }

        [Fact]
        public void GibbsGaussian2prrr()
        {
            Rand.Restart(12347);
            Range p = new Range(3);
            var mean = Variable.Array<Vector>(p).Named("mean");
            var precision = Variable.Array<PositiveDefiniteMatrix>(p).Named("precision");
            var x = Variable.Array<Vector>(p).Named("x");
            mean[p] = Variable.VectorGaussianFromMeanAndVariance(
                Vector.FromArray(7.400000000000001, 9.4),
                new PositiveDefiniteMatrix(new double[,] {{0.5999999999999999, -0.4}, {-0.4, 0.6}})).ForEach(p);
            precision[p] = Variable.Random<PositiveDefiniteMatrix>(
                Wishart.FromShapeAndRate(4.5, new PositiveDefiniteMatrix(new double[,] {{4.5, -3}, {-3, 3}}))).ForEach(p);
            x[p] = Variable.VectorGaussianFromMeanAndPrecision(mean[p], precision[p]);
            Variable.ConstrainEqualRandom(
                x[p],
                VectorGaussian.FromMeanAndVariance(
                    Vector.FromArray(1.666666666666666, 2.666666666666667),
                    new PositiveDefiniteMatrix(new double[,] {{0.6666666666666666, -0.3333333333333333}, {-0.3333333333333333, 0.6666666666666666}})));

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = 1000;
            gs.Thin = 10;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;

            var xMarginal = engine.Infer<VectorGaussian[]>(x);
            var xMargExpectedMean = Vector.FromArray(1.853720757423829, 3.046769437199992);
            // TODO: Investigate
            var xMargExpectedVar = new PositiveDefiniteMatrix(new double[,] {{0.66, -0.31}, {-0.31, 0.66}});
            for (int i = 0; i < p.SizeAsInt; i++)
            {
                Assert.True(xMargExpectedMean.MaxDiff(xMarginal[i].GetMean(), 0.1) < 0.1);
                Assert.True(xMargExpectedVar.MaxDiff(xMarginal[i].GetVariance(), 0.1) < 0.1);
            }

            var meanMarginal = engine.Infer<VectorGaussian[]>(mean);
            var meanMargExpectedMean = Vector.FromArray(7.320509300000029, 9.138372700000055);
            var meanMargExpectedVar = new PositiveDefiniteMatrix(new double[,] {{0.5899341223559563, -0.3870740576772629}, {-0.3870740576772629, 0.5945957492697019}});
            for (int i = 0; i < p.SizeAsInt; i++)
            {
                Assert.True(meanMargExpectedMean.MaxDiff(meanMarginal[i].GetMean(), 0.1) < 0.1);
                Assert.True(meanMargExpectedVar.MaxDiff(meanMarginal[i].GetVariance(), 0.1) < 0.1);
            }

            var precMarginal = engine.Infer<Wishart[]>(precision);
            var precMargExpectedMean = new PositiveDefiniteMatrix(new double[,] {{0.4696463033999955, -0.2806245454862432}, {-0.2806245454862432, 0.4338032545999959}});
            for (int i = 0; i < p.SizeAsInt; i++)
                Assert.True(precMargExpectedMean.MaxDiff(precMarginal[i].GetMean(), 0.1) < 0.1);
        }

        [Fact]
        public void GibbsGaussian2pcrr()
        {
            Rand.Restart(12347);
            Range p = new Range(3);
            var mean = Variable.Array<Vector>(p).Named("mean");
            var precision = Variable.Array<PositiveDefiniteMatrix>(p).Named("precision");
            var x = Variable.Array<Vector>(p).Named("x");
            mean[p] = Variable.VectorGaussianFromMeanAndVariance(
                Vector.FromArray(7.400000000000001, 9.4),
                new PositiveDefiniteMatrix(new double[,] {{0.5999999999999999, -0.4}, {-0.4, 0.6}})).ForEach(p);
            precision[p] = Variable.Random<PositiveDefiniteMatrix>(
                Wishart.FromShapeAndRate(4.5, new PositiveDefiniteMatrix(new double[,] {{4.5, -3}, {-3, 3}}))).ForEach(p);
            x[p] = Variable.VectorGaussianFromMeanAndPrecision(mean[p], precision[p]);
            var xobs = Vector.FromArray(1.666666666666666, 2.666666666666667);
            x.ObservedValue = new Vector[] {xobs, xobs, xobs};

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = 1000;
            gs.Thin = 10;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;

            var meanMarginal = engine.Infer<VectorGaussian[]>(mean);
            var meanMargExpectedMean = Vector.FromArray(7.331047820000016, 9.141518279999941);
            var meanMargExpectedVar = new PositiveDefiniteMatrix(new double[,] {{0.5848910647545453, -0.3863803274506987}, {-0.3863803274506987, 0.5927857329404954}});
            for (int i = 0; i < p.SizeAsInt; i++)
            {
                Assert.True(meanMargExpectedMean.MaxDiff(meanMarginal[i].GetMean(), 0.1) < 0.1);
                Assert.True(meanMargExpectedVar.MaxDiff(meanMarginal[i].GetVariance(), 0.1) < 0.1);
            }

            var precMarginal = engine.Infer<Wishart[]>(precision);
            var precMargExpectedMean = new PositiveDefiniteMatrix(new double[,] {{0.468112459199998, -0.2928359974098008}, {-0.2928359974098008, 0.4070885105999972}});
            for (int i = 0; i < p.SizeAsInt; i++)
                Assert.True(precMargExpectedMean.MaxDiff(precMarginal[i].GetMean(), 0.1) < 0.1);
        }

        [Fact]
        public void GibbsGaussian2prcr()
        {
            Rand.Restart(12347);
            Range p = new Range(3);
            var mean = Variable.Array<Vector>(p).Named("mean");
            var precision = Variable.Array<PositiveDefiniteMatrix>(p).Named("precision");
            var x = Variable.Array<Vector>(p).Named("x");
            mean[p] = Variable.VectorGaussianFromMeanAndVariance(
                Vector.FromArray(7.400000000000001, 9.4),
                new PositiveDefiniteMatrix(new double[,] {{0.5999999999999999, -0.4}, {-0.4, 0.6}})).ForEach(p);
            precision[p] = Variable.Random<PositiveDefiniteMatrix>(
                Wishart.FromShapeAndRate(4.5, new PositiveDefiniteMatrix(new double[,] {{4.5, -3}, {-3, 3}}))).ForEach(p);
            x[p] = Variable.VectorGaussianFromMeanAndPrecision(mean[p], precision[p]);
            Variable.ConstrainEqualRandom(
                x[p],
                VectorGaussian.FromMeanAndVariance(
                    Vector.FromArray(1.666666666666666, 2.666666666666667),
                    new PositiveDefiniteMatrix(new double[,] {{0.6666666666666666, -0.3333333333333333}, {-0.3333333333333333, 0.6666666666666666}})));
            var meanobs = Vector.FromArray(7.4, 9.4);
            mean.ObservedValue = new Vector[] {meanobs, meanobs, meanobs};

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = 1000;
            gs.Thin = 10;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;

            var xMarginal = engine.Infer<VectorGaussian[]>(x);
            var xMargExpectedMean = Vector.FromArray(1.835194393030008, 3.036958674348004);
            var xMargExpectedVar = new PositiveDefiniteMatrix(new double[,] {{0.6546096499201396, -0.3165004368759269}, {-0.3165004368759269, 0.6704558698104106}});
            for (int i = 0; i < p.SizeAsInt; i++)
            {
                Assert.True(xMargExpectedMean.MaxDiff(xMarginal[i].GetMean(), 0.1) < 0.1);
                Assert.True(xMargExpectedVar.MaxDiff(xMarginal[i].GetVariance(), 0.1) < 0.1);
            }

            var precMarginal = engine.Infer<Wishart[]>(precision);
            var precMargExpectedMean = new PositiveDefiniteMatrix(new double[,] {{0.4705598439999973, -0.2899506201464017}, {-0.2899506201464017, 0.410642632399999}});
            for (int i = 0; i < p.SizeAsInt; i++)
                Assert.True(precMargExpectedMean.MaxDiff(precMarginal[i].GetMean(), 0.1) < 0.1);
        }

        [Fact]
        public void GibbsGaussian2pccr()
        {
            Rand.Restart(12347);
            Range p = new Range(3);
            var mean = Variable.Array<Vector>(p).Named("mean");
            var precision = Variable.Array<PositiveDefiniteMatrix>(p).Named("precision");
            var x = Variable.Array<Vector>(p).Named("x");
            mean[p] = Variable.VectorGaussianFromMeanAndVariance(
                Vector.FromArray(7.400000000000001, 9.4),
                new PositiveDefiniteMatrix(new double[,] {{0.5999999999999999, -0.4}, {-0.4, 0.6}})).ForEach(p);
            precision[p] = Variable.Random<PositiveDefiniteMatrix>(
                Wishart.FromShapeAndRate(4.5, new PositiveDefiniteMatrix(new double[,] {{4.5, -3}, {-3, 3}}))).ForEach(p);
            x[p] = Variable.VectorGaussianFromMeanAndPrecision(mean[p], precision[p]);
            var xobs = Vector.FromArray(1.666666666666666, 2.666666666666667);
            x.ObservedValue = new Vector[] {xobs, xobs, xobs};
            var meanobs = Vector.FromArray(7.4, 9.4);
            mean.ObservedValue = new Vector[] {meanobs, meanobs, meanobs};

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = 1000;
            gs.Thin = 10;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;

            var precMarginal = engine.Infer<Wishart[]>(precision);
            var precMargExpectedMean = new PositiveDefiniteMatrix(new double[,] {{0.4720626760000004, -0.2991924347768026}, {-0.2991924347768026, 0.3843250634000018}});
            for (int i = 0; i < p.SizeAsInt; i++)
                Assert.True(precMargExpectedMean.MaxDiff(precMarginal[i].GetMean(), 0.1) < 0.1);
        }

        [Fact]
        public void GibbsGaussian2prrc()
        {
            Rand.Restart(12347);
            Range p = new Range(3);
            var mean = Variable.Array<Vector>(p).Named("mean");
            var precision = Variable.Array<PositiveDefiniteMatrix>(p).Named("precision");
            var x = Variable.Array<Vector>(p).Named("x");
            mean[p] = Variable.VectorGaussianFromMeanAndVariance(
                Vector.FromArray(7.400000000000001, 9.4),
                new PositiveDefiniteMatrix(new double[,] {{0.5999999999999999, -0.4}, {-0.4, 0.6}})).ForEach(p);
            precision[p] = Variable.Random<PositiveDefiniteMatrix>(
                Wishart.FromShapeAndRate(4.5, new PositiveDefiniteMatrix(new double[,] {{4.5, -3}, {-3, 3}}))).ForEach(p);
            x[p] = Variable.VectorGaussianFromMeanAndPrecision(mean[p], precision[p]);
            Variable.ConstrainEqualRandom(
                x[p],
                VectorGaussian.FromMeanAndVariance(
                    Vector.FromArray(1.666666666666666, 2.666666666666667),
                    new PositiveDefiniteMatrix(new double[,] {{0.6666666666666666, -0.3333333333333333}, {-0.3333333333333333, 0.6666666666666666}})));
            var precObs = new PositiveDefiniteMatrix(new double[,] {{2, 2}, {2, 3}});
            precision.ObservedValue = new PositiveDefiniteMatrix[] {precObs, precObs, precObs};

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = 1000;
            gs.Thin = 10;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;

            var xMarginal = engine.Infer<VectorGaussian[]>(x);
            var xMargExpectedMean = Vector.FromArray(3.783132303999977, 5.977692540000003);
            var xMargExpectedVar = new PositiveDefiniteMatrix(new double[,] {{0.4978722877734079, -0.2848076259943319}, {-0.2848076259943319, 0.4458780131686158}});
            for (int i = 0; i < p.SizeAsInt; i++)
            {
                Assert.True(xMargExpectedMean.MaxDiff(xMarginal[i].GetMean(), 0.1) < 0.1);
                Assert.True(xMargExpectedVar.MaxDiff(xMarginal[i].GetVariance(), 0.1) < 0.1);
            }

            var meanMarginal = engine.Infer<VectorGaussian[]>(mean);
            var meanMargExpectedMean = Vector.FromArray(6.365791720000041, 7.169097460000024);
            var meanMargExpectedVar = new PositiveDefiniteMatrix(new double[,] {{0.4690469522384849, -0.3064125998530273}, {-0.3064125998530273, 0.4387578736790226}});
            for (int i = 0; i < p.SizeAsInt; i++)
            {
                Assert.True(meanMargExpectedMean.MaxDiff(meanMarginal[i].GetMean(), 0.1) < 0.1);
                Assert.True(meanMargExpectedVar.MaxDiff(meanMarginal[i].GetVariance(), 0.1) < 0.1);
            }
        }

        [Fact]
        public void GibbsGaussian2pcrc()
        {
            Rand.Restart(12347);
            Range p = new Range(3);
            var mean = Variable.Array<Vector>(p).Named("mean");
            var precision = Variable.Array<PositiveDefiniteMatrix>(p).Named("precision");
            var x = Variable.Array<Vector>(p).Named("x");
            mean[p] = Variable.VectorGaussianFromMeanAndVariance(
                Vector.FromArray(7.400000000000001, 9.4),
                new PositiveDefiniteMatrix(new double[,] {{0.5999999999999999, -0.4}, {-0.4, 0.6}})).ForEach(p);
            precision[p] = Variable.Random<PositiveDefiniteMatrix>(
                Wishart.FromShapeAndRate(4.5, new PositiveDefiniteMatrix(new double[,] {{4.5, -3}, {-3, 3}}))).ForEach(p);
            x[p] = Variable.VectorGaussianFromMeanAndPrecision(mean[p], precision[p]);
            var xobs = Vector.FromArray(1.666666666666666, 2.666666666666667);
            x.ObservedValue = new Vector[] {xobs, xobs, xobs};
            var precObs = new PositiveDefiniteMatrix(new double[,] {{2, 2}, {2, 3}});
            precision.ObservedValue = new PositiveDefiniteMatrix[] {precObs, precObs, precObs};

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = 1000;
            gs.Thin = 10;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;

            var meanMarginal = engine.Infer<VectorGaussian[]>(mean);
            var meanMargExpectedMean = Vector.FromArray(5.759463460000001, 5.212455740000014);
            var meanMargExpectedVar = new PositiveDefiniteMatrix(new double[,] {{0.4291474471337715, -0.2860789439761391}, {-0.2860789439761391, 0.356087414789347}});
            for (int i = 0; i < p.SizeAsInt; i++)
            {
                Assert.True(meanMargExpectedMean.MaxDiff(meanMarginal[i].GetMean(), 0.1) < 0.1);
                Assert.True(meanMargExpectedVar.MaxDiff(meanMarginal[i].GetVariance(), 0.1) < 0.1);
            }
        }

        [Fact]
        public void GibbsGaussian2prcc()
        {
            Rand.Restart(12347);
            Range p = new Range(3);
            var mean = Variable.Array<Vector>(p).Named("mean");
            var precision = Variable.Array<PositiveDefiniteMatrix>(p).Named("precision");
            var x = Variable.Array<Vector>(p).Named("x");
            mean[p] = Variable.VectorGaussianFromMeanAndVariance(
                Vector.FromArray(7.400000000000001, 9.4),
                new PositiveDefiniteMatrix(new double[,] {{0.5999999999999999, -0.4}, {-0.4, 0.6}})).ForEach(p);
            precision[p] = Variable.Random<PositiveDefiniteMatrix>(
                Wishart.FromShapeAndRate(4.5, new PositiveDefiniteMatrix(new double[,] {{4.5, -3}, {-3, 3}}))).ForEach(p);
            x[p] = Variable.VectorGaussianFromMeanAndPrecision(mean[p], precision[p]);
            Variable.ConstrainEqualRandom(
                x[p],
                VectorGaussian.FromMeanAndVariance(
                    Vector.FromArray(1.666666666666666, 2.666666666666667),
                    new PositiveDefiniteMatrix(new double[,] {{0.6666666666666666, -0.3333333333333333}, {-0.3333333333333333, 0.6666666666666666}})));
            var meanobs = Vector.FromArray(7.4, 9.4);
            mean.ObservedValue = new Vector[] {meanobs, meanobs, meanobs};
            var precObs = new PositiveDefiniteMatrix(new double[,] {{2, 2}, {2, 3}});
            precision.ObservedValue = new PositiveDefiniteMatrix[] {precObs, precObs, precObs};

            GibbsSampling gs = new GibbsSampling();
            gs.BurnIn = 1000;
            gs.Thin = 10;
            InferenceEngine engine = new InferenceEngine(gs);
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;

            var xMarginal = engine.Infer<VectorGaussian[]>(x);
            var xMargExpectedMean = Vector.FromArray(4.360657580000011, 7.379968300000039);
            var xMargExpectedVar = new PositiveDefiniteMatrix(new double[,] {{0.4551355217589777, -0.27328615229776}, {-0.27328615229776, 0.3625638447720047}});
            for (int i = 0; i < p.SizeAsInt; i++)
            {
                Assert.True(xMargExpectedMean.MaxDiff(xMarginal[i].GetMean(), 0.1) < 0.1);
                Assert.True(xMargExpectedVar.MaxDiff(xMarginal[i].GetVariance(), 0.1) < 0.1);
            }
        }

        [Fact]
        public void GibbsTable2rr()
        {
            Rand.Restart(12347);
            var x1 = Variable.Discrete(Vector.FromArray(2, 8)).Named("x1");
            var x2 = Variable.New<int>().Named("x2");
            using (Variable.Case(x1, 0))
                x2.SetTo(Variable.Discrete(Vector.FromArray(5, 4, 3)));

            using (Variable.Case(x1, 1))
                x2.SetTo(Variable.Discrete(Vector.FromArray(2, 1, 0)));

            Variable.ConstrainEqualRandom(x2, new Discrete(Vector.FromArray(1, 3, 6)));

            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;
            var x1Actual = engine.Infer<Discrete>(x1);
            var x2Actual = engine.Infer<Discrete>(x2);
            Discrete x1Expected = new Discrete(0.3043, 0.6957);
            Discrete x2Expected = new Discrete(0.3217, 0.5217, 0.1565);
            Console.WriteLine("x1 = {0} should be {1}", x1Actual, x1Expected);
            Console.WriteLine("x2 = {0} should be {1}", x2Actual, x2Expected);
            // EP and VMP both show x1Marg = 0.3043 0.6957, x2Marg = 0.3217 0.5217 0.1565
            Assert.True(x1Expected.MaxDiff(x1Actual) < 0.02);
            Assert.True(x2Expected.MaxDiff(x2Actual) < 0.02);
        }

        [Fact]
        public void GibbsTable2cr()
        {
            var x1 = Variable.Discrete(Vector.FromArray(0, 1)).Named("x1");
            var x2 = Variable.New<int>().Named("x2");
            using (Variable.Case(x1, 0))
                x2.SetTo(Variable.Discrete(Vector.FromArray(5, 4, 3)));

            using (Variable.Case(x1, 1))
                x2.SetTo(Variable.Discrete(Vector.FromArray(2, 1, 0)));

            Variable.ConstrainEqualRandom(x2, new Discrete(Vector.FromArray(1, 3, 6)));

            Rand.Restart(12347);
            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;
            var x1Marg = engine.Infer<Discrete>(x1);
            var x2Marg = engine.Infer<Discrete>(x2);
            Assert.True(x1Marg.GetProbs().MaxDiff(Vector.FromArray(0, 1), 0.001) < 0.02);
            Assert.True(x2Marg.GetProbs().MaxDiff(Vector.FromArray(0.4, 0.6, 0.0), 0.001) < 0.02);
        }

        [Fact]
        public void GibbsTable2rc()
        {
            // If 0 is used rather than 1e-10 below then with the specifierd seed
            // this test fails. With some other seeds it can pass.
            // The problem is that, randomly, case 1 is the first case input to
            // ExitAverageConditional, which will then output Discrete(2,1,0) as the
            // forward message to x2. This then causes zero probability exception because of
            // the constraint on x2 (Discrete(0,0,1)), and so the x2 marginal calculation fails.
            Rand.Restart(1234);
            var x1 = Variable.Discrete(Vector.FromArray(2, 8)).Named("x1");
            var x2 = Variable.New<int>().Named("x2");
            using (Variable.Case(x1, 0))
                x2.SetTo(Variable.Discrete(Vector.FromArray(5, 4, 3)));

            using (Variable.Case(x1, 1))
                x2.SetTo(Variable.Discrete(Vector.FromArray(2, 1, 0)));

            Variable.ConstrainEqualRandom(x2, new Discrete(Vector.FromArray(1e-10, 1e-10, 1)));

            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;
            var x1Marg = engine.Infer<Discrete>(x1);
            var x2Marg = engine.Infer<Discrete>(x2);
            Assert.True(x1Marg.GetProbs().MaxDiff(Vector.FromArray(1, 0), 0.001) < 0.02);
            Assert.True(x2Marg.GetProbs().MaxDiff(Vector.FromArray(0, 0, 1), 0.001) < 0.02);
        }

        // Fails with UseParallelLoops because Parallel.For does not guarantee a consistent partitioning into threads
        [Trait("Category", "OpenBug")]
        [Fact]
        public void GibbsTable2prr_Repeated()
        {
            var result = GibbsTable2prr_Helper(true);
            var result2 = GibbsTable2prr_Helper(true);
            Assert.Equal(result, result2);
        }

        [Fact]
        public void GibbsTable2prr()
        {
            GibbsTable2prr_Helper();
        }
        private IList<Discrete> GibbsTable2prr_Helper(bool useParallelForLoops = false)
        {
            Rand.Restart(12347);

            Range p = new Range(3);
            var x1 = Variable.Array<int>(p).Named("x1");
            var x2 = Variable.Array<int>(p).Named("x2");
            using (Variable.ForEach(p))
            {
                x1[p] = Variable.Discrete(Vector.FromArray(2, 8));
                using (Variable.Case(x1[p], 0))
                    x2[p] = Variable.Discrete(Vector.FromArray(5, 4, 3));

                using (Variable.Case(x1[p], 1))
                    x2[p] = Variable.Discrete(Vector.FromArray(2, 1, 0));

                Variable.ConstrainEqualRandom(x2[p], new Discrete(Vector.FromArray(1, 3, 6)));
            }
            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            if (useParallelForLoops)
                engine.Compiler.UseParallelForLoops = true;
            engine.NumberOfIterations = 50000;
            engine.ShowProgress = false;
            double threshold = 0.05;
            var x1Marg = engine.Infer<IList<Discrete>>(x1);
            var x2Marg = engine.Infer<IList<Discrete>>(x2);
            for (int i = 0; i < p.SizeAsInt; i++)
            {
                Vector x1Actual = x1Marg[i].GetProbs();
                Vector x2Actual = x2Marg[i].GetProbs();
                Vector x1Expected = Vector.FromArray(0.3043, 0.6957);
                Vector x2Expected = Vector.FromArray(0.3217, 0.5217, 0.1565);
                var error1 = x1Actual.MaxDiff(x1Expected, 0.001);
                var error2 = x2Actual.MaxDiff(x2Expected, 0.001);
                //Console.WriteLine("x1 = {0} should be {1} (error={2})", x1Actual, x1Expected, error1.ToString("g2"));
                Console.WriteLine("x2[{0}] = {1} should be {2} (error={3})", i, x2Actual, x2Expected, error2.ToString("g2"));
                Assert.True(error1 < threshold);
                Assert.True(error2 < threshold);
            }
            return x2Marg;
        }

        [Fact]
        public void GibbsTable2pcr()
        {
            Rand.Restart(12347);

            Range p = new Range(3);
            var x1 = Variable.Array<int>(p).Named("x1");
            var x2 = Variable.Array<int>(p).Named("x2");
            using (Variable.ForEach(p))
            {
                x1[p] = Variable.Discrete(Vector.FromArray(0, 1));
                using (Variable.Case(x1[p], 0))
                    x2[p] = Variable.Discrete(Vector.FromArray(5, 4, 3));

                using (Variable.Case(x1[p], 1))
                    x2[p] = Variable.Discrete(Vector.FromArray(2, 1, 0));

                Variable.ConstrainEqualRandom(x2[p], new Discrete(Vector.FromArray(1, 3, 6)));
            }
            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;
            var x1Marg = engine.Infer<IList<Discrete>>(x1);
            var x2Marg = engine.Infer<IList<Discrete>>(x2);
            var x1Expected = new Discrete(Vector.FromArray(0, 1));
            var x2Expected = new Discrete(Vector.FromArray(0.4, 0.6, 0.0));
            for (int i = 0; i < p.SizeAsInt; i++)
            {
                Assert.True(x1Marg[i].MaxDiff(x1Expected) < 1e-8);
                Assert.True(x2Marg[i].MaxDiff(x2Expected) < 0.04);
            }
        }

        [Fact]
        public void GibbsTable2prc()
        {
            // See comments for GibbsTable2rc
            Rand.Restart(12347);

            Range p = new Range(3);
            var x1 = Variable.Array<int>(p).Named("x1");
            var x2 = Variable.Array<int>(p).Named("x2");
            using (Variable.ForEach(p))
            {
                x1[p] = Variable.Discrete(Vector.FromArray(2, 8));
                using (Variable.Case(x1[p], 0))
                    x2[p] = Variable.Discrete(Vector.FromArray(5, 4, 3));

                using (Variable.Case(x1[p], 1))
                    x2[p] = Variable.Discrete(Vector.FromArray(2, 1, 0));

                Variable.ConstrainEqualRandom(x2[p], new Discrete(Vector.FromArray(1e-10, 1e-10, 1)));
            }
            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;
            var x1Marg = engine.Infer<Discrete[]>(x1);
            var x2Marg = engine.Infer<Discrete[]>(x2);
            for (int i = 0; i < p.SizeAsInt; i++)
            {
                Assert.True(x1Marg[i].GetProbs().MaxDiff(Vector.FromArray(1, 0), 0.001) < 0.02);
                Assert.True(x2Marg[i].GetProbs().MaxDiff(Vector.FromArray(0, 0, 1), 0.001) < 0.02);
            }
        }

        [Fact]
        public void GibbsCategorical1rr()
        {
            var p = Variable.Dirichlet(Vector.FromArray(0.01, 1, 3, 6)).Named("p");
            var x = Variable.New<int>().Named("x");
            x = Variable.Discrete(p);
            Variable.ConstrainEqualRandom(x, new Discrete(8, 6, 4, 2));
            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;
            var pPost = engine.Infer<Dirichlet>(p);
            var xPost = engine.Infer<Discrete>(x);
            Assert.True(xPost.GetProbs().MaxDiff(Vector.FromArray(0.00292, 0.20222, 0.39716, 0.3977), 0.1) < 0.1);
            Assert.True(pPost.GetMean().MaxDiff(Vector.FromArray(0.001194980747175012, 0.109866856459003, 0.3081557670599985, 0.5807822242000037), 0.1) < 0.1);
        }

        [Fact]
        public void GibbsCategorical1cr()
        {
            var p = Variable.Dirichlet(Vector.FromArray(0.01, 1, 3, 6)).Named("p");
            var x = Variable.New<int>().Named("x");
            x = Variable.Discrete(p);
            Variable.ConstrainEqualRandom(x, new Discrete(1, 0, 0, 0));
            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;
            var pPost = engine.Infer<Dirichlet>(p);
            var xPost = engine.Infer<Discrete>(x);
            var pExpected = new Dirichlet(Vector.FromArray(1.01, 1, 3, 6));
            Assert.True(xPost.GetProbs().MaxDiff(Vector.FromArray(1, 0, 0, 0), 0.1) < 0.1);
            Assert.True(pPost.MaxDiff(pExpected) < 0.1);
        }

        [Fact]
        public void GibbsCategorical1rc()
        {
            var p = Variable.Observed(Vector.FromArray(0.000999000999000999, 0.0999000999000999, 0.2997002997002997, 0.5994005994005994)).Named("p");
            var x = Variable.New<int>().Named("x");
            x = Variable.Discrete(p);
            Variable.ConstrainEqualRandom(x, new Discrete(8, 6, 4, 2));
            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;
            var xPost = engine.Infer<Discrete>(x);
            Assert.True(xPost.GetProbs().MaxDiff(Vector.FromArray(0.00266, 0.1995, 0.3989, 0.3989), 0.1) < 0.1);
        }

        [Fact]
        public void GibbsCategorical1cc()
        {
            // If we use 0 rather than 1e-10 below, this test can randomly fail.
            // This is because the Gamma distribution sampler is quite likely to
            // give a zero sample when the argument is small - for example 0.000999000999000999 as
            // below (the reason for this is that the Gamma boost calculation reaches machine precision).
            // This leads to the Dirichlet sampler giving a 0 in the probability vector, which, combined
            // with the constraint below, throws up an exception in the marginal calculation for x.
            Rand.Restart(12347);
            var p = Variable.Random(Dirichlet.PointMass(0.000999000999000999, 0.0999000999000999, 0.2997002997002997, 0.5994005994005994)).Named("p");
            var x = Variable.New<int>().Named("x");
            x = Variable.Discrete(p);
            Variable.ConstrainEqualRandom(x, new Discrete(1, 1e-10, 1e-10, 1e-10));
            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.NumberOfIterations = 100000;
            engine.ShowProgress = false;
            var pPost = engine.Infer<Dirichlet>(p);
            var xPost = engine.Infer<Discrete>(x);
            Assert.True(xPost.GetProbs().MaxDiff(Vector.FromArray(1, 0, 0, 0), 0.1) < 0.1);
            Assert.True(pPost.GetMean().MaxDiff(Vector.FromArray(0.000999000999000999, 0.0999000999000999, 0.2997002997002997, 0.5994005994005994), 0.1) < 0.1);
        }

        [Fact]
        public void GibbsCategorical1prr()
        {
            Rand.Restart(12347);
            Range r = new Range(3);
            var p = Variable.Array<Vector>(r).Named("p");
            var x = Variable.Array<int>(r).Named("x");
            using (Variable.ForEach(r))
            {
                p[r] = Variable.Dirichlet(Vector.FromArray(0.01, 1, 3, 6));
                x[r] = Variable.Discrete(p[r]);
                Variable.ConstrainEqualRandom(x[r], new Discrete(8, 6, 4, 2));
            }
            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;
            var pPost = engine.Infer<Dirichlet[]>(p);
            var xPost = engine.Infer<Discrete[]>(x);
            Assert.True(xPost[1].GetProbs().MaxDiff(Vector.FromArray(0.00266, 0.1995, 0.3989, 0.3989), 0.1) < 0.1);
            Assert.True(pPost[1].GetMean().MaxDiff(Vector.FromArray(0.00115, 0.1089, 0.3087, 0.5812), 0.1) < 0.1);
        }

        [Fact]
        public void GibbsCategorical1pcr()
        {
            Rand.Restart(12347);
            Range r = new Range(3);
            var p = Variable.Array<Vector>(r).Named("p");
            var x = Variable.Array<int>(r).Named("x");
            using (Variable.ForEach(r))
            {
                p[r] = Variable.Dirichlet(Vector.FromArray(0.01, 1, 3, 6));
                x[r] = Variable.Discrete(p[r]);
                Variable.ConstrainEqualRandom(x[r], new Discrete(1, 0, 0, 0));
            }
            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;
            var pPost = engine.Infer<Dirichlet[]>(p);
            var xPost = engine.Infer<Discrete[]>(x);
            Assert.True(xPost[1].GetProbs().MaxDiff(Vector.FromArray(1, 0, 0, 0), 0.1) < 0.1);
            Assert.True(pPost[1].GetMean().MaxDiff(Vector.FromArray(0.09173, 0.09083, 0.2725, 0.545), 0.1) < 0.1);
        }

        [Fact]
        public void GibbsCategorical1prc()
        {
            Rand.Restart(12347);
            Range r = new Range(3);
            var p = Variable.Array<Vector>(r).Named("p");
            var x = Variable.Array<int>(r).Named("x");
            using (Variable.ForEach(r))
            {
                p[r] = Variable.Random(Dirichlet.PointMass(0.000999000999000999, 0.0999000999000999, 0.2997002997002997, 0.5994005994005994));
                x[r] = Variable.Discrete(p[r]);
                Variable.ConstrainEqualRandom(x[r], new Discrete(8, 6, 4, 2));
            }
            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;
            var pPost = engine.Infer<Dirichlet[]>(p);
            var xPost = engine.Infer<Discrete[]>(x);
            Assert.True(xPost[1].GetProbs().MaxDiff(Vector.FromArray(0.00266, 0.1995, 0.3989, 0.3989), 0.1) < 0.1);
            Assert.True(pPost[1].GetMean().MaxDiff(Vector.FromArray(0.000999000999000999, 0.0999000999000999, 0.2997002997002997, 0.5994005994005994), 0.1) < 0.1);
        }

        [Fact]
        public void GibbsCategorical1prc2()
        {
            Rand.Restart(12347);
            Range r = new Range(3);
            var p = Variable.Array<Vector>(r).Named("p");
            var x = Variable.Array<int>(r).Named("x");
            using (Variable.ForEach(r))
            {
                p[r] = Variable<Vector>.Random(Variable.Observed(Dirichlet.PointMass(0.000999000999000999, 0.0999000999000999, 0.2997002997002997, 0.5994005994005994)));
                x[r] = Variable.Discrete(p[r]);
                Variable.ConstrainEqualRandom(x[r], new Discrete(8, 6, 4, 2));
            }
            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.NumberOfIterations = 20000;
            engine.ShowProgress = false;
            var pPost = engine.Infer<Dirichlet[]>(p);
            var xPost = engine.Infer<Discrete[]>(x);
            Assert.True(xPost[1].GetProbs().MaxDiff(Vector.FromArray(0.00266, 0.1995, 0.3989, 0.3989), 0.1) < 0.1);
            Assert.True(pPost[1].GetMean().MaxDiff(Vector.FromArray(0.000999000999000999, 0.0999000999000999, 0.2997002997002997, 0.5994005994005994), 0.1) < 0.1);
        }

        [Fact]
        public void GibbsGrid2()
        {
            Rand.Restart(12347);
            var V1 = Variable.Discrete(5.440981145757824, 1.331264592607048).Named("V1");
            var V2 = Variable.Discrete(5.533625857693537, 1.238619880671335).Named("V2");
            var V3 = Variable.Discrete(4.011332082664074, 2.760913655700799).Named("V3");
            var V4 = Variable.Discrete(4.026520561985976, 2.745725176378898).Named("V4");

            using (Variable.Case(V1, 0))
            {
                Variable.ConstrainEqualRandom(V2, new Discrete(2.184051455636825, 1));
                Variable.ConstrainEqualRandom(V3, new Discrete(1.766430150272333, 1));
            }
            using (Variable.Case(V1, 1))
            {
                Variable.ConstrainEqualRandom(V2, new Discrete(1, 2.184051455636825));
                Variable.ConstrainEqualRandom(V3, new Discrete(1, 1.766430150272333));
            }
            using (Variable.Case(V2, 0))
                Variable.ConstrainEqualRandom(V3, new Discrete(0.4396772729776542, 1));
            using (Variable.Case(V2, 1))
                Variable.ConstrainEqualRandom(V3, new Discrete(1, 0.4396772729776542));
            using (Variable.Case(V3, 0))
                Variable.ConstrainEqualRandom(V4, new Discrete(0.7667405112631087, 1));
            using (Variable.Case(V3, 1))
                Variable.ConstrainEqualRandom(V4, new Discrete(1, 0.7667405112631087));

            Variable.ConstrainEqualRandom(V1, new Discrete(0.3048982951619423, 0.1105463137004335));
            Variable.ConstrainEqualRandom(V2, new Discrete(2.68139556179517, 0.5953325831358508));
            Variable.ConstrainEqualRandom(V3, new Discrete(1.387311309282806, 1.263716538231863));
            Variable.ConstrainEqualRandom(V4, new Discrete(1.021698193905576, 0.3664312110740297));

            var engine = new InferenceEngine(new GibbsSampling());
            var V1Marginal = engine.Infer<Discrete>(V1);
            var V2Marginal = engine.Infer<Discrete>(V2);
            var V3Marginal = engine.Infer<Discrete>(V3);
            var V4Marginal = engine.Infer<Discrete>(V3);

            double tol = 1e-2;
            Assert.True(V1Marginal.GetProbs().MaxDiff(Vector.FromArray(0.9511, 0.04885)) < tol);
            Assert.True(V2Marginal.GetProbs().MaxDiff(Vector.FromArray(0.9652, 0.03478)) < tol);
            Assert.True(V3Marginal.GetProbs().MaxDiff(Vector.FromArray(0.5122, 0.4878)) < tol);
            Assert.True(V4Marginal.GetProbs().MaxDiff(Vector.FromArray(0.5122, 0.4878)) < tol);
        }

        [Fact]
        public void GibbsGrid3()
        {
            Rand.Restart(12347);
            var V1 = Variable.Discrete(391.4075758053536, 232.4236617342759).Named("V1");
            var V2 = Variable.Discrete(216.3423120851904, 407.488925454439).Named("V2");
            var V3 = Variable.Discrete(462.2812823010871, 161.5499552385424).Named("V3");
            var V4 = Variable.Discrete(85.97700420942128, 537.8542333302082).Named("V4");
            var V5 = Variable.Discrete(242.1637095180639, 381.6675280215655).Named("V5");
            var V6 = Variable.Discrete(394.8542984803361, 228.9769390592934).Named("V6");
            var V7 = Variable.Discrete(71.2208893953617, 552.6103481442677).Named("V7");
            var V8 = Variable.Discrete(404.2674177270012, 219.5638198126283).Named("V8");
            var V9 = Variable.Discrete(531.8842079363141, 91.94702960331553).Named("V9");

            using (Variable.Case(V1, 0))
            {
                Variable.ConstrainEqualRandom(V2, new Discrete(0.3878463337902373, 1));
                Variable.ConstrainEqualRandom(V4, new Discrete(0.3054753430397658, 1));
            }
            using (Variable.Case(V1, 1))
            {
                Variable.ConstrainEqualRandom(V2, new Discrete(1, 0.3878463337902373));
                Variable.ConstrainEqualRandom(V4, new Discrete(1, 0.3054753430397658));
            }
            using (Variable.Case(V2, 0))
            {
                Variable.ConstrainEqualRandom(V3, new Discrete(0.6876816989147311, 1));
                Variable.ConstrainEqualRandom(V5, new Discrete(0.3478781780656773, 1));
            }
            using (Variable.Case(V2, 1))
            {
                Variable.ConstrainEqualRandom(V3, new Discrete(1, 0.6876816989147311));
                Variable.ConstrainEqualRandom(V3, new Discrete(1, 0.3478781780656773));
            }
            using (Variable.Case(V3, 0))
                Variable.ConstrainEqualRandom(V6, new Discrete(1.057326795579609, 1));
            using (Variable.Case(V3, 1))
                Variable.ConstrainEqualRandom(V6, new Discrete(1, 1.057326795579609));
            using (Variable.Case(V4, 0))
            {
                Variable.ConstrainEqualRandom(V5, new Discrete(4.360034344112384, 1));
                Variable.ConstrainEqualRandom(V7, new Discrete(0.9596111451759485, 1));
            }
            using (Variable.Case(V4, 1))
            {
                Variable.ConstrainEqualRandom(V5, new Discrete(1, 4.360034344112384));
                Variable.ConstrainEqualRandom(V7, new Discrete(1, 0.9596111451759485));
            }
            using (Variable.Case(V5, 0))
            {
                Variable.ConstrainEqualRandom(V6, new Discrete(0.2960231987112843, 1));
                Variable.ConstrainEqualRandom(V8, new Discrete(0.323568686574075, 1));
            }
            using (Variable.Case(V5, 1))
            {
                Variable.ConstrainEqualRandom(V6, new Discrete(1, 0.2960231987112843));
                Variable.ConstrainEqualRandom(V8, new Discrete(1, 0.323568686574075));
            }
            using (Variable.Case(V6, 0))
                Variable.ConstrainEqualRandom(V9, new Discrete(0.7702026452870254, 1));
            using (Variable.Case(V6, 1))
                Variable.ConstrainEqualRandom(V9, new Discrete(1, 0.7702026452870254));
            using (Variable.Case(V7, 0))
                Variable.ConstrainEqualRandom(V8, new Discrete(0.2594276182310469, 1));
            using (Variable.Case(V7, 1))
                Variable.ConstrainEqualRandom(V8, new Discrete(1, 0.2594276182310469));
            using (Variable.Case(V8, 0))
                Variable.ConstrainEqualRandom(V9, new Discrete(2.594685839479944, 1));
            using (Variable.Case(V8, 1))
                Variable.ConstrainEqualRandom(V9, new Discrete(1, 2.594685839479944));

            Variable.ConstrainEqualRandom(V1, new Discrete(1.137285667651903, 1.927969764756977));
            Variable.ConstrainEqualRandom(V2, new Discrete(0.3110444768257816, 0.6309017211601474));
            Variable.ConstrainEqualRandom(V3, new Discrete(0.7691725496103468, 0.2972588210000597));
            Variable.ConstrainEqualRandom(V4, new Discrete(0.2672857419909021, 2.537596859815947));
            Variable.ConstrainEqualRandom(V5, new Discrete(1.01130835788106, 0.5245860436528574));
            Variable.ConstrainEqualRandom(V6, new Discrete(2.238327181965948, 1.260648171829282));
            Variable.ConstrainEqualRandom(V7, new Discrete(0.3716660021547728, 3.817461605685822));
            Variable.ConstrainEqualRandom(V8, new Discrete(1.335762158963715, 4.388190949151847));
            Variable.ConstrainEqualRandom(V9, new Discrete(3.120608493764948, 0.5045246474713611));

            var engine = new InferenceEngine(new GibbsSampling());
            // tol = 1e-2
            //engine.NumberOfIterations = 1172; // old scheduler
            //engine.NumberOfIterations = 212; // new scheduler
            // tol = 1e-3
            //engine.NumberOfIterations = 24000; // old scheduler
            //engine.NumberOfIterations = 200000; // new scheduler
            var V1Marginal = engine.Infer<Discrete>(V1);
            var V2Marginal = engine.Infer<Discrete>(V2);
            var V3Marginal = engine.Infer<Discrete>(V3);
            var V4Marginal = engine.Infer<Discrete>(V4);
            var V5Marginal = engine.Infer<Discrete>(V5);
            var V6Marginal = engine.Infer<Discrete>(V6);
            var V7Marginal = engine.Infer<Discrete>(V7);
            var V8Marginal = engine.Infer<Discrete>(V8);
            var V9Marginal = engine.Infer<Discrete>(V9);

            double tol = 1e-2;
            Assert.True(V1Marginal.GetProbs().MaxDiff(Vector.FromArray(0.8536, 0.1464)) < tol);
            Assert.True(V2Marginal.GetProbs().MaxDiff(Vector.FromArray(0.1096, 0.8904)) < tol);
            Assert.True(V3Marginal.GetProbs().MaxDiff(Vector.FromArray(0.9558, 0.04424)) < tol);
            Assert.True(V4Marginal.GetProbs().MaxDiff(Vector.FromArray(0.006783, 0.9932)) < tol);
            Assert.True(V5Marginal.GetProbs().MaxDiff(Vector.FromArray(0.07721, 0.9228)) < tol);
            Assert.True(V6Marginal.GetProbs().MaxDiff(Vector.FromArray(0.858, 0.142)) < tol);
            Assert.True(V7Marginal.GetProbs().MaxDiff(Vector.FromArray(0.007205, 0.9928)) < tol);
            Assert.True(V8Marginal.GetProbs().MaxDiff(Vector.FromArray(0.9142, 0.08582)) < tol);
            Assert.True(V9Marginal.GetProbs().MaxDiff(Vector.FromArray(0.9811, 0.01887)) < tol);
        }

        public class GibbsChildFromTwoParents
        {
            // Model variables
            public Variable<int> NumCases = Variable.New<int>();
            public VariableArray<int> s1;
            public VariableArray<int> h1;
            public VariableArray<int> o1;
            public Variable<Vector> pt_s1;
            public Variable<Vector> pt_h1;
            public VariableArray2D<Vector> cpt_o1;
            public Variable<Dirichlet> pt_s1_prior = null;
            public Variable<Dirichlet> pt_h1_prior = null;
            public Variable<DirichletArray2D> cpt_o1_prior = null;
            public Dirichlet pt_s1_posterior = null;
            public Dirichlet pt_h1_posterior = null;
            public DirichletArray2D cpt_o1_posterior = null;

            //Inference engine
            public InferenceEngine InfEngine = new InferenceEngine();

            public void CreateModel(int range_o, int range_h, int range_s)
            {
                NumCases = Variable.New<int>().Named("NumCases");
                Range n = new Range(NumCases).Named("n");
                Range r_s1 = new Range(range_s).Named("r_s1");
                Range r_h1 = new Range(range_h).Named("r_h1");
                Range r_o1 = new Range(range_o).Named("r_o1");
                pt_s1_prior = Variable.New<Dirichlet>().Named("pt_s1_prior");
                pt_h1_prior = Variable.New<Dirichlet>().Named("pt_h1_prior");
                cpt_o1_prior = Variable.New<DirichletArray2D>().Named("cpt_o1_prior");

                pt_s1 = Variable<Vector>.Random(pt_s1_prior).Named("pt_s1");
                pt_h1 = Variable<Vector>.Random(pt_h1_prior).Named("pt_h1");
                cpt_o1 = Variable.Array<Vector>(r_s1, r_h1).Named("cpt_o1");
                cpt_o1.SetTo(Variable<Vector[,]>.Random(cpt_o1_prior));
                pt_s1.SetValueRange(r_s1);
                pt_h1.SetValueRange(r_h1);
                cpt_o1.SetValueRange(r_o1);
                s1 = Variable.Array<int>(n).Named("s1");
                s1[n] = Variable.Discrete(pt_s1).ForEach(n);
                h1 = Variable.Array<int>(n).Named("h1");
                h1[n] = Variable.Discrete(pt_h1).ForEach(n);
                o1 = AddChildFromTwoParents(s1, h1, cpt_o1).Named("o1");
            }

            public void InferParameters(int[] o1_data)
            {
                GibbsSampling g = new GibbsSampling();
                g.DefaultNumberOfIterations = 1000;
                InfEngine.Algorithm = g;
                o1.ObservedValue = o1_data;

                NumCases.ObservedValue = o1_data.Length;
                int num_s1 = s1.GetValueRange().SizeAsInt;
                int num_h1 = h1.GetValueRange().SizeAsInt;
                int num_o1 = o1.GetValueRange().SizeAsInt;

                Dirichlet[,] cpt_O1_prior = new Dirichlet[num_s1,num_h1];

                for (int i = 0; i < num_s1; i++)
                    for (int j = 0; j < num_h1; j++)
                        cpt_O1_prior[i, j] = Dirichlet.Uniform(num_o1);

                pt_s1_prior.ObservedValue = Dirichlet.Uniform(num_s1);
                pt_h1_prior.ObservedValue = Dirichlet.Uniform(num_h1);
                cpt_o1_prior.ObservedValue = (DirichletArray2D) Distribution<Vector>.Array(cpt_O1_prior);

                // Run the inference
                pt_s1_posterior = InfEngine.Infer<Dirichlet>(pt_s1);
                pt_h1_posterior = InfEngine.Infer<Dirichlet>(pt_h1);
                cpt_o1_posterior = InfEngine.Infer<DirichletArray2D>(cpt_o1);

                Console.WriteLine("S1");
                Console.WriteLine(pt_s1_posterior);
                Console.WriteLine("H1");
                Console.WriteLine(pt_h1_posterior);
                Console.WriteLine("O1");
                Console.WriteLine(cpt_o1_posterior);

                Console.WriteLine("===========================================");
                //Console.ReadLine();
            }

            // Model code for adding a child from a single parent
            public static VariableArray<int> AddChildFromOneParent(VariableArray<int> parent, VariableArray<Vector> cpt)
            {
                var d = parent.Range;
                // data range
                var child = Variable.Array<int>(d);
                using (Variable.ForEach(d))
                using (Variable.Switch(parent[d]))
                    child[d] = Variable.Discrete(cpt[parent[d]]);
                return child;
            }

            // Model code for adding a child from two parents
            public static VariableArray<int> AddChildFromTwoParents(
                VariableArray<int> parent1, VariableArray<int> parent2, VariableArray2D<Vector> cpt)
            {
                var d = parent1.Range;
                // data range
                var child = Variable.Array<int>(d);
                using (Variable.ForEach(d))
                using (Variable.Switch(parent1[d]))
                using (Variable.Switch(parent2[d]))
                    child[d] = Variable.Discrete(cpt[parent1[d], parent2[d]]);
                return child;
            }
        }

        [Fact]
        public void GibbsChildFromTwoParentsTest()
        {
            int[] o1_data = new int[] {3, 4, 0, 4, 0, 0, 0, 0, 3, 0, 2, 1, 3, 0, 0, 0, 1, 0, 3, 0};

            var model = new GibbsChildFromTwoParents();
            model.CreateModel(5, 4, 2);
            model.InferParameters(o1_data);

            // No asserts - just make sure it runs without error
        }

        /// <summary>
        /// Failing test transcribed from Microsoft.ML.Probabilistic.Fun
        /// </summary>
        [Fact]
        public void MontyHallFun()
        {
            var _range1 = (new Range(3)).Named("_range1");
            var _range2 = (new Range(3)).Named("_range2");
            var var1 = Variable.DiscreteUniform(_range1).Named("var1");
            var var3 = Variable.New<int>().Named("var3");
            var var1Eq0 = (var1 == 0).Named("var1Eq0");
            var var1Eq1 = (var1 == 1).Named("var1Eq1");
            using (Variable.If(var1Eq0))
                var3.SetTo(Variable.Discrete(_range2, new[] {0.000000, 0.500000, 0.500000}).Named("d1"));
            using (Variable.IfNot(var1Eq0))
            {
                using (Variable.If(var1Eq1))
                    var3.SetTo(2);
                using (Variable.IfNot(var1Eq1))
                    var3.SetTo(1);
            }

            var new_guess0 = Variable.New<int>().Named("new_guess0");
            var var3Eq0 = (var3 == 0).Named("var3Eq0");
            var var3Eq1 = (var3 == 1).Named("var3Eq1");

            using (Variable.If(var3Eq0))
                new_guess0.SetTo(Variable.Discrete(_range1, new[] {0.000000, 0.500000, 0.500000}).Named("d2"));

            using (Variable.IfNot(var3Eq0))
            {
                using (Variable.If(var3Eq1))
                    new_guess0.SetTo(2);
                using (Variable.IfNot(var3Eq1))
                    new_guess0.SetTo(1);
            }

            var e = new InferenceEngine(new GibbsSampling());
            var b = e.Infer<Discrete>(new_guess0);
        }
    }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif
}
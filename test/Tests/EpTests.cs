// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Text;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Xunit;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Factors.Attributes;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Distributions.Kernels;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Compiler.Transforms;
using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Serialization;
using Microsoft.ML.Probabilistic.Compiler;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

    public class EpTests
    {
        public static double[] linspace(double min, double max, int count)
        {
            if (count < 2)
                throw new ArgumentException("count < 2");
            double inc = (max - min) / (count - 1);
            return Util.ArrayInit(count, i => (min + i * inc));
        }

        [Fact]
        public void CutForwardWhenTest()
        {
            Range item = new Range(2);
            Variable<double> x = Variable.GaussianFromMeanAndPrecision(0, 1);
            using (Variable.ForEach(item))
            {
                var isForwardLoop = item.IsIncreasing();
                var xCut = Variable.CutForwardWhen(x, isForwardLoop);
                Variable.ConstrainPositive(xCut);
            }

            InferenceEngine engine = new InferenceEngine();
            var xWithoutBackwardPass = engine.Infer<Gaussian>(x);
            // Without a backward pass, isForwardLoop is always true so the posterior equals the prior.
            Assert.Equal(new Gaussian(0, 1), xWithoutBackwardPass);
            item.AddAttribute(new Sequential() { BackwardPass = true });
            var xWithBackwardPass = engine.Infer<Gaussian>(x);
            Assert.NotEqual(new Gaussian(0, 1), xWithBackwardPass);
        }

        [Fact]
        public void IntegralTest()
        {
            Gaussian dist = new Gaussian(0, 1);
            double lowerBoundPoint = 0.1;
            var lowerBound = Variable.GaussianFromMeanAndVariance(lowerBoundPoint, 0.0);
            lowerBound.Name = nameof(lowerBound);
            lowerBound.AddAttribute(QueryTypes.Marginal);
            lowerBound.AddAttribute(QueryTypes.MarginalDividedByPrior);
            double upperBoundPoint = 0.5;
            var upperBound = Variable.GaussianFromMeanAndVariance(upperBoundPoint, 0.0);
            upperBound.Name = nameof(upperBound);
            upperBound.AddAttribute(QueryTypes.Marginal);
            upperBound.AddAttribute(QueryTypes.MarginalDividedByPrior);
            var integral = Variable<double>.Factor(Factor.Integral, lowerBound, upperBound, Variable.Constant((Func<double,double>)Predictability), Variable.Constant((ITruncatableDistribution<double>)dist));
            integral.Name = nameof(integral);
            Variable.ConstrainEqualRandom(integral, Gaussian.FromNatural(1, 0));

            InferenceEngine engine = new InferenceEngine();
            var actual = engine.Infer(integral);
            Console.WriteLine(actual);
            var lowerMsg = engine.Infer<Gaussian>(lowerBound, QueryTypes.MarginalDividedByPrior);
            lowerMsg.GetDerivatives(lowerBoundPoint, out double lowerBoundDerivative, out _);
            var upperMsg = engine.Infer<Gaussian>(upperBound, QueryTypes.MarginalDividedByPrior);
            upperMsg.GetDerivatives(upperBoundPoint, out double upperBoundDerivative, out _);

            lowerBound.ObservedValue = lowerBoundPoint;
            upperBound.ObservedValue = upperBoundPoint;
            double f = engine.Infer<Gaussian>(integral).GetMean();
            double delta = 1e-4;
            lowerBound.ObservedValue += delta;
            double fdl = engine.Infer<Gaussian>(integral).GetMean();
            lowerBound.ObservedValue = lowerBoundPoint;
            upperBound.ObservedValue += delta;
            double fdr = engine.Infer<Gaussian>(integral).GetMean();
            double lowerBoundDerivativeExpected = (fdl - f) / delta;
            double upperBoundDerivativeExpected = (fdr - f) / delta;
            Assert.Equal(lowerBoundDerivativeExpected, lowerBoundDerivative, 5 * delta);
            Assert.Equal(upperBoundDerivativeExpected, upperBoundDerivative, 5 * delta);
        }

        public static double Predictability(double skillDifference)
        {
            return MMath.NormalCdf(System.Math.Abs(skillDifference) / MMath.Sqrt2) * 100;
        }

        [Fact]
        public void IntegralTest2()
        {
            var evidence = Variable.Bernoulli(0.5);
            evidence.Name = nameof(evidence);
            var evBlock = Variable.If(evidence);
            Gaussian dist = new Gaussian(0, 1);
            var skillDifference = Variable.Random(dist);
            skillDifference.Name = nameof(skillDifference);
            var lowerPoint = Variable.Observed(0.1);
            var lowerBound = Variable.GaussianFromMeanAndVariance(lowerPoint, 0.0);
            lowerBound.Name = nameof(lowerBound);
            lowerBound.AddAttribute(QueryTypes.Marginal);
            lowerBound.AddAttribute(QueryTypes.MarginalDividedByPrior);
            var upperPoint = Variable.Observed(0.5);
            var upperBound = Variable.GaussianFromMeanAndVariance(upperPoint, 0.0);
            upperBound.Name = nameof(upperBound);
            upperBound.AddAttribute(QueryTypes.Marginal);
            upperBound.AddAttribute(QueryTypes.MarginalDividedByPrior);
            Variable.ConstrainBetween(skillDifference, lowerBound, upperBound);
            // Add Predictability as a factor
            var abs = Abs(skillDifference);
            abs.Name = nameof(abs);
            var noisy = Variable.GaussianFromMeanAndPrecision(abs / MMath.Sqrt2, 1);
            Variable.ConstrainPositive(noisy);
            IncrementLogLikelihood(System.Math.Log(100));
            evBlock.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            var ev = engine.Infer<Bernoulli>(evidence).LogOdds;
            var lowerMsg = engine.Infer<Gaussian>(lowerBound, QueryTypes.MarginalDividedByPrior);
            lowerMsg.GetDerivatives(lowerPoint.ObservedValue, out double lowerBoundDerivative, out _);
            double likelihood = System.Math.Exp(ev);

            var upperMsg = engine.Infer<Gaussian>(upperBound, QueryTypes.MarginalDividedByPrior);
            upperMsg.GetDerivatives(upperPoint.ObservedValue, out double upperBoundDerivative, out double _);

            double f = engine.Infer<Bernoulli>(evidence).LogOdds;
            double delta = 1e-4;
            double lowerOld = lowerPoint.ObservedValue;
            lowerPoint.ObservedValue += delta;
            double fdl = engine.Infer<Bernoulli>(evidence).LogOdds;
            lowerPoint.ObservedValue = lowerOld;
            upperPoint.ObservedValue += delta;
            double fdr = engine.Infer<Bernoulli>(evidence).LogOdds;
            double lowerBoundDerivativeExpected = (fdl - f) / delta;
            double upperBoundDerivativeExpected = (fdr - f) / delta;
            Assert.Equal(lowerBoundDerivativeExpected, lowerBoundDerivative, 5 * delta);
            Assert.Equal(upperBoundDerivativeExpected, upperBoundDerivative, 5 * delta);

            void IncrementLogLikelihood(double increment)
            {
                Variable.ConstrainEqualRandom(Variable.Constant(increment), Gaussian.FromNatural(1, 0));
            }
        }

        public static Variable<double> Abs(Variable<double> x)
        {
            Variable<double> y = Variable.New<double>();
            var positive = (x >= 0);
            using(Variable.If(positive))
            {
                y.SetTo(Variable.Copy(x));
            }
            using(Variable.IfNot(positive))
            {
                y.SetTo(-x);
            }
            return y;
        }

        [Fact]
        public void ProbBetweenTest()
        {
            Gaussian dist = new Gaussian(0, 1);
            double dpPoint = 0.1;
            var dp = Variable.GaussianFromMeanAndVariance(dpPoint, 0.0);
            dp.Name = nameof(dp);
            dp.AddAttribute(QueryTypes.MarginalDividedByPrior);
            var left = Variable<double>.Factor(Factor.Quantile, (CanGetQuantile<double>)dist, 0.5 - dp);
            left.Name = nameof(left);
            left.AddAttribute(QueryTypes.Marginal);
            left.AddAttribute(QueryTypes.MarginalDividedByPrior);
            var right = Variable<double>.Factor(Factor.Quantile, (CanGetQuantile<double>)dist, 0.5 + dp);
            right.Name = nameof(right);
            right.AddAttribute(QueryTypes.Marginal);
            right.AddAttribute(QueryTypes.MarginalDividedByPrior);
            var probBetween = Variable<double>.Factor(Factor.ProbBetween, (CanGetProbLessThan<double>)dist, left, right);
            var waitTime = 10.0 / probBetween;
            waitTime.Name = nameof(waitTime);
            Variable.ConstrainEqualRandom(waitTime, Gaussian.FromNatural(1, 0));

            InferenceEngine engine = new InferenceEngine();
            var actual = engine.Infer(waitTime);
            Console.WriteLine(actual);
            var dpMsg = engine.Infer<Gaussian>(dp, QueryTypes.MarginalDividedByPrior);
            dpMsg.GetDerivatives(dpPoint, out double dpDerivative, out _);

            if (false)
            {
                left.ObservedValue = engine.Infer<Gaussian>(left).Point;
                right.ObservedValue = engine.Infer<Gaussian>(right).Point;
                double f = engine.Infer<Gaussian>(waitTime).GetMean();
                double delta = 1e-4;
                double leftOld = left.ObservedValue;
                left.ObservedValue += delta;
                double fdl = engine.Infer<Gaussian>(waitTime).GetMean();
                left.ObservedValue = leftOld;
                right.ObservedValue += delta;
                double fdr = engine.Infer<Gaussian>(waitTime).GetMean();
                double leftMsgExpected = (fdl - f) / delta;
                double rightMsgExpected = (fdr - f) / delta;
                Console.WriteLine($"leftMsgExpected = {leftMsgExpected} rightMsgExpected = {rightMsgExpected}");
            }
            else
            {
                dp.ObservedValue = dpPoint;
                double f = engine.Infer<Gaussian>(waitTime).GetMean();
                double delta = 1e-4;
                dp.ObservedValue += delta;
                double fd = engine.Infer<Gaussian>(waitTime).GetMean();
                double dpDerivativeExpected = (fd - f) / delta;
                Assert.Equal(dpDerivativeExpected, dpDerivative, 11 * delta * System.Math.Abs(dpDerivativeExpected));
            }
        }

        [Fact]
        public void BetaSubtractionTest()
        {
            Variable<double> a = Variable.Beta(1, 1);
            Variable<double> b = Variable.Beta(1, 1);
            var c = a - b;

            InferenceEngine engine = new InferenceEngine();
            engine.Infer(c);
        }

        /// <summary>
        /// Demonstrate that EP can solve a linear program
        /// </summary>
        [Fact]
        public void LinearProgrammingTest()
        {
            // Could repeat inference incrementally while decreasing precision
            double precision = 1e-20;
            // maximize x+y 
            // subject to x + 2*y = 1
            // x >= 0
            // y >= 0
            Variable<double> x = Variable.GaussianFromMeanAndPrecision(1.0 / precision / precision, precision);
            Variable<double> y = Variable.GaussianFromMeanAndPrecision(1.0 / precision / precision, precision);
            Variable.ConstrainEqual(x + 2 * y, 1);
            Variable.ConstrainPositive(x);
            Variable.ConstrainPositive(y);

            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine($"x = {engine.Infer(x)}");
            Console.WriteLine($"y = {engine.Infer(y)}");
        }

        [Fact]
        public void TruncatedGammaPowerTest()
        {
            Assert.True(PowerOp.PowAverageConditional(new TruncatedGamma(2.333, 0.02547, 1, double.PositiveInfinity), 1.1209480955953663, GammaPower.Uniform(-1)).IsProper());
            Assert.True(PowerOp.PowAverageConditional(new TruncatedGamma(5.196e+48, 5.567e-50, 1, double.PositiveInfinity), 0.0016132617913803061, GammaPower.Uniform(-1)).IsProper());
            Assert.True(PowerOp.PowAverageConditional(new TruncatedGamma(23.14, 0.06354, 1, double.PositiveInfinity), 1.5543122344752203E-15, GammaPower.Uniform(-1)).IsProper());
        }

        [Fact]
        public void TruncatedGammaPower_ReturnsGammaShapeGreaterThan1()
        {
            Variable<TruncatedGamma> xPriorVar = Variable.Observed(default(TruncatedGamma)).Named("xPrior");
            Variable<double> x = Variable<double>.Random(xPriorVar).Named("x");
            Variable<double> power = Variable.Observed(0.5).Named("power");
            var y = x ^ power;
            y.Name = nameof(y);
            Variable<Gamma> yLikeVar = Variable.Observed(default(Gamma)).Named("yLike");
            Variable.ConstrainEqualRandom(y, yLikeVar);
            y.SetMarginalPrototype(yLikeVar);
            InferenceEngine engine = new InferenceEngine();

            Rand.Restart(0);
            foreach (var powerValue in linspace(1, 10, 10))
            {
                TruncatedGamma xPrior = new TruncatedGamma(Gamma.FromShapeAndRate(3, 3), 1, double.PositiveInfinity);
                xPriorVar.ObservedValue = xPrior;
                Gamma yLike = Gamma.Uniform();
                yLikeVar.ObservedValue = yLike;
                power.ObservedValue = powerValue;
                var xActual = engine.Infer<TruncatedGamma>(x);
                var yActual = engine.Infer<Gamma>(y);

                // Importance sampling
                GammaEstimator xEstimator = new GammaEstimator();
                GammaEstimator yEstimator = new GammaEstimator();
                MeanVarianceAccumulator yExpectedInverse = new MeanVarianceAccumulator();
                int nSamples = 100000;
                for (int i = 0; i < nSamples; i++)
                {
                    double xSample = xPrior.Sample();
                    double ySample = System.Math.Pow(xSample, power.ObservedValue);
                    double logWeight = yLike.GetLogProb(ySample);
                    double weight = System.Math.Exp(logWeight);
                    xEstimator.Add(xSample, weight);
                    yEstimator.Add(ySample, weight);
                    yExpectedInverse.Add(1 / ySample, weight);
                }
                Gamma xExpected = xEstimator.GetDistribution(new Gamma());
                Gamma yExpected = yEstimator.GetDistribution(yLike);
                double yActualMeanInverse = yActual.GetMeanPower(-1);
                double meanInverseError = MMath.AbsDiff(yExpectedInverse.Mean, yActualMeanInverse, 1e-8);
                Trace.WriteLine($"power = {powerValue}:");
                Trace.WriteLine($"  x = {xActual} should be {xExpected}");
                Trace.WriteLine($"  y = {yActual}[E^-1={yActualMeanInverse}] should be {yExpected}[E^-1={yExpectedInverse.Mean}], E^-1 error = {meanInverseError}");
                Assert.True(yActual.Shape > 1);
                Assert.True(MMath.AbsDiff(yExpected.GetMean(), yActual.GetMean(), 1e-8) < 1);
                Assert.True(meanInverseError < 1e-2);
            }
        }

        [Fact]
        public void TruncatedGammaPower_ReturnsGammaPowerShapeGreaterThan1()
        {
            var result = PowerOp.PowAverageConditional(new TruncatedGamma(0.4, 0.5, 1, double.PositiveInfinity), 0, GammaPower.PointMass(0, -1));
            Assert.True(result.IsPointMass);
            Assert.Equal(1.0, result.Point);

            Variable<TruncatedGamma> xPriorVar = Variable.Observed(default(TruncatedGamma)).Named("xPrior");
            Variable<double> x = Variable<double>.Random(xPriorVar).Named("x");
            Variable<double> power = Variable.Observed(0.5).Named("power");
            var y = x ^ power;
            y.Name = nameof(y);
            Variable<GammaPower> yLikeVar = Variable.Observed(default(GammaPower)).Named("yLike");
            Variable.ConstrainEqualRandom(y, yLikeVar);
            y.SetMarginalPrototype(yLikeVar);
            InferenceEngine engine = new InferenceEngine();

            Rand.Restart(0);
            foreach (var powerValue in linspace(1, 10, 10))
            {
                TruncatedGamma xPrior = new TruncatedGamma(Gamma.FromShapeAndRate(3, 3), 1, double.PositiveInfinity);
                xPriorVar.ObservedValue = xPrior;
                GammaPower yLike = GammaPower.Uniform(-1);
                //GammaPower yLike = GammaPower.FromShapeAndRate(1, 0.5, -1);
                yLikeVar.ObservedValue = yLike;
                power.ObservedValue = powerValue;
                var xActual = engine.Infer<TruncatedGamma>(x);
                var yActual = engine.Infer<GammaPower>(y);

                // Importance sampling
                GammaEstimator xEstimator = new GammaEstimator();
                GammaPowerEstimator yEstimator = new GammaPowerEstimator(yLike.Power);
                MeanVarianceAccumulator yExpectedInverse = new MeanVarianceAccumulator();
                MeanVarianceAccumulator yMva = new MeanVarianceAccumulator();
                int nSamples = 100000;
                for (int i = 0; i < nSamples; i++)
                {
                    double xSample = xPrior.Sample();
                    double ySample = System.Math.Pow(xSample, power.ObservedValue);
                    double logWeight = yLike.GetLogProb(ySample);
                    double weight = System.Math.Exp(logWeight);
                    xEstimator.Add(xSample, weight);
                    yEstimator.Add(ySample, weight);
                    yExpectedInverse.Add(1 / ySample, weight);
                    yMva.Add(ySample, weight);
                }
                Gamma xExpected = xEstimator.GetDistribution(new Gamma());
                GammaPower yExpected = yEstimator.GetDistribution(yLike);
                yExpected = GammaPower.FromMeanAndVariance(yMva.Mean, yMva.Variance, yLike.Power);
                double yActualMeanInverse = yActual.GetMeanPower(-1);
                double meanInverseError = MMath.AbsDiff(yExpectedInverse.Mean, yActualMeanInverse, 1e-8);
                Trace.WriteLine($"power = {powerValue}:");
                Trace.WriteLine($"  x = {xActual} should be {xExpected}");
                Trace.WriteLine($"  y = {yActual}[E^-1={yActual.GetMeanPower(-1)}] should be {yExpected}[E^-1={yExpectedInverse.Mean}], error = {meanInverseError}");
                Assert.True(yActual.Shape > 1);
                Assert.True(MMath.AbsDiff(yExpected.GetMean(), yActual.GetMean(), 1e-8) < 1);
                Assert.True(meanInverseError < 2e-2);
            }
        }

        [Fact]
        public void GammaPowerPower_ReturnsShapeGreaterThan2()
        {
            Variable<GammaPower> xPriorVar = Variable.Observed(default(GammaPower)).Named("xPrior");
            Variable<double> x = Variable<double>.Random(xPriorVar).Named("x");
            Variable<double> power = Variable.Observed(0.5).Named("power");
            var y = x ^ power;
            y.Name = nameof(y);
            Variable<GammaPower> yLikeVar = Variable.Observed(default(GammaPower)).Named("yLike");
            Variable.ConstrainEqualRandom(y, yLikeVar);
            y.SetMarginalPrototype(yLikeVar);
            InferenceEngine engine = new InferenceEngine();

            foreach (var powerValue in linspace(1, 10, 10))
            {
                GammaPower xPrior = new GammaPower(3, 3, 1);
                xPriorVar.ObservedValue = xPrior;
                GammaPower yLike = GammaPower.Uniform(-1);
                yLikeVar.ObservedValue = yLike;
                power.ObservedValue = powerValue;
                var xActual = engine.Infer<GammaPower>(x);
                var yActual = engine.Infer<GammaPower>(y);

                // Importance sampling
                GammaEstimator xEstimator = new GammaEstimator();
                GammaEstimator yEstimator = new GammaEstimator();
                MeanVarianceAccumulator yExpectedInverse = new MeanVarianceAccumulator();
                int nSamples = 1000000;
                for (int i = 0; i < nSamples; i++)
                {
                    double xSample = xPrior.Sample();
                    double ySample = System.Math.Pow(xSample, power.ObservedValue);
                    double logWeight = yLike.GetLogProb(ySample);
                    double weight = System.Math.Exp(logWeight);
                    xEstimator.Add(xSample, weight);
                    yEstimator.Add(ySample, weight);
                    yExpectedInverse.Add(1 / ySample, weight);
                }
                Gamma xExpected = xEstimator.GetDistribution(new Gamma());
                Gamma yExpected = yEstimator.GetDistribution(new Gamma());
                double yActualMeanInverse = yActual.GetMeanPower(-1);
                double yExpectedMeanInverse = xPrior.GetMeanPower(-powerValue);
                double meanInverseError = MMath.AbsDiff(yExpectedInverse.Mean, yActualMeanInverse, 1e-8);
                Trace.WriteLine($"power = {powerValue}:");
                Trace.WriteLine($"  x = {xActual} should be {xExpected}");
                Trace.WriteLine($"  y = {yActual}[E^-1={yActualMeanInverse}] should be {yExpected}[E^-1={yExpectedInverse.Mean} should be {yExpectedMeanInverse}], error = {meanInverseError}");
                Assert.True(yActual.Shape > 2);
                Assert.True(MMath.AbsDiff(yExpected.GetMean(), yActual.GetMean(), 1e-8) < 1);
                // E^1 doesn't match because yActual is projected onto a Gamma.
                //Assert.True(meanInverseError < 10);
            }
        }

        [Fact]
        public void GammaPowerPowerTest()
        {
            Assert.False(double.IsNaN(PowerOp.GammaPowerFromDifferentPower(new GammaPower(1.333, 1.5, 1), 0.01).Shape));
            for (int i = 1; i <= 10; i++)
            {
                // TODO: make this work
                //Assert.True(PowerOp.XAverageConditional(new GammaPower(7, 0.1111, -1), new GammaPower(16.19, 0.06154, 1), 2.2204460492503136E-10/i, GammaPower.Uniform(1)).IsProper());
            }
            Assert.True(PowerOp.XAverageConditional(new GammaPower(7, 0.1111, -1), new GammaPower(16.19, 0.06154, 1), MMath.Ulp(1), GammaPower.Uniform(1)).IsProper());
            Assert.True(PowerOp.PowAverageConditional(GammaPower.FromShapeAndRate(9.0744065303642287, 8.7298765698182414, 1), 1.6327904641199278, GammaPower.Uniform(-1)).IsProper());
            Assert.False(PowerOp.XAverageConditional(GammaPower.Uniform(-1), GammaPower.FromShapeAndRate(1, 1, 1), 4.0552419045546273, GammaPower.Uniform(1)).IsPointMass);

            Variable<GammaPower> xPriorVar = Variable.Observed(default(GammaPower)).Named("xPrior");
            Variable<double> x = Variable<double>.Random(xPriorVar).Named("x");
            Variable<double> power = Variable.Observed(0.5).Named("power");
            var y = x ^ power;
            y.Name = nameof(y);
            Variable<GammaPower> yLikeVar = Variable.Observed(default(GammaPower)).Named("yLike");
            Variable.ConstrainEqualRandom(y, yLikeVar);
            y.SetMarginalPrototype(yLikeVar);
            InferenceEngine engine = new InferenceEngine();

            foreach (var (xPower, yPower) in new[]
            {
                (-1, -1),
                (-1, -0.5),
                (1, 1),
                (2, 2)
            })
            {
                var xPrior = GammaPower.FromShapeAndRate(2, 3, xPower);
                xPriorVar.ObservedValue = xPrior;
                GammaPower yLike = GammaPower.FromShapeAndRate(1, 0.5, yPower);
                yLikeVar.ObservedValue = yLike;

                // Importance sampling
                GammaPowerEstimator xEstimator = new GammaPowerEstimator(xPrior.Power);
                GammaPowerEstimator yEstimator = new GammaPowerEstimator(yLike.Power);
                MeanVarianceAccumulator yMva = new MeanVarianceAccumulator();
                int nSamples = 1000000;
                Rand.Restart(0);
                for (int i = 0; i < nSamples; i++)
                {
                    double xSample = xPrior.Sample();
                    double ySample = System.Math.Pow(xSample, power.ObservedValue);
                    if (double.IsNaN(ySample) || double.IsInfinity(ySample)) throw new Exception();
                    double logWeight = yLike.GetLogProb(ySample);
                    if (double.IsNaN(logWeight)) throw new Exception();
                    double weight = System.Math.Exp(logWeight);
                    xEstimator.Add(xSample, weight);
                    yEstimator.Add(ySample, weight);
                    yMva.Add(ySample, weight);
                }
                GammaPower xExpected = xEstimator.GetDistribution(xPrior);
                GammaPower yExpected = yEstimator.GetDistribution(yLike);
                if (yLike.Power == -1)
                    yExpected = GammaPower.FromMeanAndVariance(yMva.Mean, yMva.Variance, yLike.Power);

                var xActual = engine.Infer<GammaPower>(x);
                double xError = MMath.AbsDiff(xActual.GetMean(), xExpected.GetMean(), 1e-6);
                Trace.WriteLine($"x = {xActual} should be {xExpected}, error = {xError}");
                var yActual = engine.Infer<GammaPower>(y);
                double yError = MMath.AbsDiff(yActual.GetMean(), yExpected.GetMean(), 1e-6);
                Trace.WriteLine($"y = {yActual} should be {yExpected}, error = {yError}");
                double tolerance = 1e-1;
                Assert.True(xError < tolerance);
                Assert.True(yError < tolerance);
            }
        }

        [Fact]
        public void VectorTimesScalarTest()
        {
            Vector aMean = Vector.FromArray(2.2, 3.3);
            PositiveDefiniteMatrix aVariance = new PositiveDefiniteMatrix(new double[,] { { 4, 1 }, { 1, 4 } });
            VectorGaussian aPrior = new VectorGaussian(aMean, aVariance);
            double bMean = 5.5;
            double bVariance = 6.6;
            Gaussian bPrior = new Gaussian(bMean, bVariance);
            Variable<Vector> a = Variable.Random(aPrior).Named("a");
            Variable<double> b = Variable.Random(bPrior).Named("b");
            b.AddAttribute(new PointEstimate());
            b.InitialiseTo(Gaussian.PointMass(bMean));
            Variable<Vector> x = Variable.VectorTimesScalar(a, b).Named("x");
            Vector xMean = Vector.FromArray(7.7, 8.8);
            PositiveDefiniteMatrix xVariance = new PositiveDefiniteMatrix(new double[,] { { 9, 1 }, { 1, 9 } });
            VectorGaussian xLike = new VectorGaussian(xMean, xVariance);
            Variable.ConstrainEqualRandom(x, xLike);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.InitialisationAffectsSchedule = true;
            var bActual = engine.Infer<Gaussian>(b).Point;

            // compute expected value
            double[] bs = linspace(2, 3, 100);
            double argmax = 0;
            double max = double.NegativeInfinity;
            foreach (double bPoint in bs)
            {
                double logProb = ProductVectorGaussianOp_PointB.LogAverageFactor(xLike, aPrior, Gaussian.PointMass(bPoint)) + bPrior.GetLogProb(bPoint);
                if (logProb > max)
                {
                    max = logProb;
                    argmax = bPoint;
                }
            }
            double bExpected = argmax;
            Console.WriteLine($"b = {bActual} should be {bExpected}");
            Assert.Equal(bExpected, bActual, 0.1);
        }

        internal void BaseOffsetTest2()
        {
            double baseSkillPriorVariance = 0.45580040754755607;
            double baseSkillWeight = 3.0889215616628265;
            double skillPriorMean = 4.4812609769076293;
            double skillOffsetPriorVariance = 2.5342192437224029;
            Gaussian messageToScaledBase = DoublePlusOp.AAverageConditional(Gaussian.PointMass(21), new Gaussian(skillPriorMean, skillOffsetPriorVariance));
            Gaussian messageToBase = GaussianProductOp.AAverageConditional(messageToScaledBase, baseSkillWeight);
            Gaussian basePosterior = new Gaussian(0, baseSkillPriorVariance) * messageToBase;
            Console.WriteLine(basePosterior);
            Variable<double> baseSkill = Variable.GaussianFromMeanAndVariance(0, baseSkillPriorVariance);
            Variable<double> offset = Variable.GaussianFromMeanAndVariance(skillPriorMean, skillOffsetPriorVariance);
            Variable<double> skill = baseSkill * baseSkillWeight + offset;
            Gaussian[] messages = new Gaussian[]
            {
                new Gaussian(26.56, 15.14),
                new Gaussian(24.34, 15.12),
                new Gaussian(23.12, 15.1),
                new Gaussian(22.39, 15.08),
                new Gaussian(21.99, 15.06),
                new Gaussian(21.73, 15.04),
                new Gaussian(21.52, 15.02),
                new Gaussian(21.39, 15.01),
                new Gaussian(21.3, 14.99),
                new Gaussian(21.23, 14.98),
                new Gaussian(21.18, 14.97),
                new Gaussian(21.14, 14.96),
                new Gaussian(21.11, 14.95),
                new Gaussian(21.08, 14.94),
                new Gaussian(21.06, 14.93),
                new Gaussian(21.04, 14.93),
                new Gaussian(21.03, 14.92),
                new Gaussian(21.02, 14.91),
                new Gaussian(21.01, 14.91),
                new Gaussian(21.01, 14.9),
                new Gaussian(21, 14.9)
            };
            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;
            for (int i = 0; i < messages.Length; i++)
            {
                Gaussian message = messages[i];
                Variable.ConstrainEqualRandom(skill, message);
                Console.WriteLine($"{i} {engine.Infer(baseSkill)}");
            }
        }

        internal void BaseOffsetTest()
        {
            int N = 100;
            Range time = new Range(N);
            var basePrecision = 1;
            Variable<double> changePrecision;
            bool coupled = true;
            bool usePrevious = false;
            bool useCompoundPrior = false;
            if (useCompoundPrior)
            {
                var changePrecisionRate = Variable.GammaFromShapeAndRate(3, 3);
                changePrecision = Variable.GammaFromShapeAndRate(3, changePrecisionRate);
            }
            else
            {
                changePrecision = Variable.GammaFromShapeAndRate(3, 1e-2);
            }
            changePrecision.AddAttribute(new PointEstimate());
            var obsPrecision = 1000;
            var change1 = Variable.Array<double>(time).Named("change1");
            var state1 = Variable.Array<double>(time).Named("state1");
            var obs1 = Variable.Array<double>(time).Named("obs1");
            var state2 = Variable.Array<double>(time).Named("state2");
            var change2 = Variable.Array<double>(time).Named("change2");
            var obs2 = Variable.Array<double>(time).Named("obs2");
            using (var block = Variable.ForEach(time))
            {
                var t = block.Index;
                using (Variable.If(t == 0))
                {
                    change1[t] = Variable.GaussianFromMeanAndPrecision(0, basePrecision);
                    state1[t] = change1[t] + 0;
                    change2[t] = Variable.GaussianFromMeanAndPrecision(0, basePrecision);
                    state2[t] = change2[t] + 0;
                }
                using (Variable.If(t > 0))
                {
                    change1[t] = Variable.GaussianFromMeanAndPrecision(0, changePrecision);
                    change2[t] = Variable.GaussianFromMeanAndPrecision(0, changePrecision);
                    if (coupled)
                    {
                        state1[t] = state1[t - 1] + change1[t] + (usePrevious ? change2[t - 1] : change2[t]);
                        state2[t] = state2[t - 1] + change2[t] + change1[t];
                    }
                    else
                    {
                        state1[t] = state1[t - 1] + change1[t];
                        state2[t] = state2[t - 1] + change2[t];
                    }
                }
                obs1[t] = Variable.GaussianFromMeanAndPrecision(state1[t], obsPrecision);
                obs2[t] = Variable.GaussianFromMeanAndPrecision(state2[t], obsPrecision);
            }

            var changePrecisionTrue = 1e1;
            changePrecision.InitialiseTo(Gamma.PointMass(changePrecisionTrue));

            InferenceEngine engine = new InferenceEngine();
            int trialCount = 10;
            int overestimateCount = 0;
            for (int trial = 0; trial < trialCount; trial++)
            {
                // sample data from the model
                double[] state1True = new double[N];
                double[] change1True = new double[N];
                double[] obs1True = new double[N];
                double[] state2True = new double[N];
                double[] change2True = new double[N];
                double[] obs2True = new double[N];
                Gaussian changePrior = Gaussian.FromMeanAndPrecision(0, changePrecisionTrue);
                for (int i = 0; i < N; i++)
                {
                    if (i == 0)
                    {
                        change1True[i] = Gaussian.FromMeanAndPrecision(0, 1).Sample();
                        state1True[i] = Gaussian.FromMeanAndPrecision(0, 1).Sample();
                        change2True[i] = Gaussian.FromMeanAndPrecision(0, 1).Sample();
                        state2True[i] = Gaussian.FromMeanAndPrecision(0, 1).Sample();
                    }
                    else
                    {
                        change1True[i] = changePrior.Sample();
                        change2True[i] = changePrior.Sample();
                        if (coupled)
                        {
                            state1True[i] = state1True[i - 1] + change1True[i] + (usePrevious ? change2True[i - 1] : change2True[i]);
                            state2True[i] = state2True[i - 1] + change2True[i] + change1True[i];
                        }
                        else
                        {
                            state1True[i] = state1True[i - 1] + change1True[i];
                            state2True[i] = state2True[i - 1] + change2True[i];
                        }
                    }
                    obs1True[i] = Gaussian.FromMeanAndPrecision(state1True[i], obsPrecision).Sample();
                    obs2True[i] = Gaussian.FromMeanAndPrecision(state2True[i], obsPrecision).Sample();
                }
                obs1.ObservedValue = obs1True;
                obs2.ObservedValue = obs2True;

                bool showState = false;
                if (showState)
                {
                    var state1Actual = engine.Infer<IList<Gaussian>>(state1);
                    for (int i = 0; i < N; i++)
                    {
                        Trace.WriteLine($"{i} {state1True[i]} {state1Actual[i]}");
                    }
                }
                var changePrecisionActual = engine.Infer<Gamma>(changePrecision);
                var acc = new MeanVarianceAccumulator();
                for (int i = 1; i < N; i++)
                {
                    double diff = obs1True[i] - obs1True[i - 1];
                    acc.Add(diff);
                    diff = obs2True[i] - obs2True[i - 1];
                    acc.Add(diff);
                }
                var empiricalEstimate = 1 / acc.Variance;
                if (coupled) empiricalEstimate *= 2;
                Trace.WriteLine($"changePrecision = {changePrecisionActual} should be {changePrecisionTrue}, empirical = {empiricalEstimate}");
                if (changePrecisionActual.GetMean() > changePrecisionTrue) overestimateCount++;
            }
            double overestimatePercent = 100.0 * overestimateCount / trialCount;
            Trace.WriteLine($"{overestimatePercent}% overestimated");
        }

        /// <summary>
        /// Test a model where EP fails due to improper message.
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")]
        public void ProductIsBetweenTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            var evBlock = Variable.If(evidence);
            Variable<double> x = Variable.GaussianFromMeanAndVariance(2, 1).Named("x");
            Variable<double> y = Variable.GaussianFromMeanAndVariance(3, 1).Named("y");
            Variable<double> product = (x * y).Named("product");
            Variable<bool> b = Variable.Bernoulli(0.5).Named("b");
            using (Variable.If(b))
            {
                Variable.ConstrainBetween(product, -1, 0);
            }
            using (Variable.IfNot(b))
            {
                Variable.ConstrainBetween(product, double.NegativeInfinity, double.PositiveInfinity);
            }
            evBlock.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.GivePriorityTo(typeof(GaussianProductOp_Slow));
            Console.WriteLine(engine.Infer(evidence));
        }

        /// <summary>
        /// Test a model where EP requires damping to converge.
        /// Also see FactorizedRegression2.
        /// </summary>
        internal void RegressionConvergenceTest()
        {
            Gaussian prior = new Gaussian(0, 1);
            Variable<double>[][] perf = Util.ArrayInit(2, t => Util.ArrayInit(2, i => Variable.Random(prior).Named($"perf{t}{i}")));
            double prec = 100;
            double pm = 3;
            double em = 2;
            double[,] data = { { 1, 1 }, { 1, 1 } };
            for (int t = 0; t < 2; t++)
            {
                int t2 = 1 - t;
                for (int i = 0; i < 2; i++)
                {
                    // works if any 1 variable is a point mass
                    //if(t==0 && i==0)
                    //    perf[t][i].AddAttribute(new PointEstimate());
                    //var enemySum = perf[t2][0] + perf[t2][1];
                    //var mean = perf[t][i] * pm + enemySum * em;
                    Variable<double>[][] perf2 = Util.ArrayInit(2, tt => Util.ArrayInit(2, ii =>
                        Variable<double>.Factor(Damp.Backward<double>, perf[tt][ii], 0.5)
                        ));
                    var mean = perf2[t][i] * pm + perf2[t2][0] * em + perf2[t2][1] * em;
                    Variable.ConstrainEqual(data[t, i], Variable.GaussianFromMeanAndPrecision(mean, prec));
                    // initialise on the correct answer (doesn't seem to help)
                    perf[t][i].InitialiseTo(Gaussian.PointMass(0.1428));
                }
            }

            // Solution using PointEstimate (VMP is similar):
            /* [0] [0] Gaussian.PointMass(0.1428)
                   [1] Gaussian.PointMass(0.1428)
               [1] [0] Gaussian.PointMass(0.1428)
                   [1] Gaussian.PointMass(0.1428)
             */

            InferenceEngine engine = new InferenceEngine();
            engine.OptimiseForVariables = perf.SelectMany(a => a.Select(v => (IVariable)v)).ToList();
            //engine.Compiler.TraceAllMessages = true;
            //engine.Algorithm = new VariationalMessagePassing();
            engine.Compiler.FreeMemory = false;
            engine.Compiler.GivePriorityTo(typeof(ReplicateOp_NoDivide));
            for (int iter = 1; iter < 1000; iter++)
            {
                engine.NumberOfIterations = iter;
                var perfActual = Util.ArrayInit(2, t => Util.ArrayInit(2, i => engine.Infer<Gaussian>(perf[t][i])));
                Console.WriteLine(StringUtil.ArrayToString(perfActual));
            }
        }

        /// <summary>
        /// Test approximate inference in a model with correlated latent variables
        /// </summary>
        internal void CorrelatedLatentTest()
        {
            Rand.Restart(0);
            int n = 100;
            Range item = new Range(n).Named("item");
            var innerCount = Variable.Array<int>(item).Named("innerCount");
            Range inner = new Range(innerCount[item]).Named("inner");
            var hMean = Variable.GaussianFromMeanAndPrecision(1, 0.1).Named("hMean");
            hMean.AddAttribute(new PointEstimate());
            var hPrecision = Variable.GammaFromShapeAndRate(1, 1).Named("hPrecision");
            hPrecision.AddAttribute(new PointEstimate());
            var xPrecision = Variable.GammaFromShapeAndRate(1, 1).Named("xPrecision");
            xPrecision.AddAttribute(new PointEstimate());
            var yPrecision = Variable.GammaFromShapeAndRate(1, 1).Named("yPrecision");
            yPrecision.AddAttribute(new PointEstimate());
            var h2Precision = Variable.GammaFromShapeAndRate(1, 1).Named("h2Precision");
            h2Precision.AddAttribute(new PointEstimate());
            var h = Variable.Array<double>(item).Named("h");
            var h2 = Variable.Array<double>(item).Named("h2");
            var x = Variable.Array(Variable.Array<double>(inner), item).Named("x");
            var y = Variable.Array(Variable.Array<double>(inner), item).Named("y");
            bool useFakeModel = true;
            bool useCorrection = true;
            using (Variable.ForEach(item))
            {
                h[item] = Variable.GaussianFromMeanAndPrecision(hMean, hPrecision);
                h2[item] = Variable.GaussianFromMeanAndPrecision(h[item], h2Precision);
                using (Variable.ForEach(inner))
                {
                    x[item][inner] = Variable.GaussianFromMeanAndPrecision(h[item], xPrecision);
                    if (useFakeModel)
                    {
                        var fakeH = Variable.GaussianFromMeanAndPrecision(h[item], h2Precision / Variable.Double(innerCount[item]));
                        if (!useCorrection)
                            fakeH = Variable.Copy(h[item]);
                        y[item][inner] = Variable.GaussianFromMeanAndPrecision(fakeH, yPrecision);
                    }
                    else
                    {
                        y[item][inner] = Variable.GaussianFromMeanAndPrecision(h2[item], yPrecision);
                    }
                }
            }
            if (useFakeModel)
            {
                using (Variable.Repeat(-1))
                {
                    var hDenom = Variable.Array<double>(item).Named("hDenom");
                    using (Variable.ForEach(item))
                    {
                        hDenom[item] = Variable.GaussianFromMeanAndPrecision(Variable.Cut(hMean), Variable.Cut(hPrecision));
                        using (Variable.ForEach(inner))
                        {
                            var fakeHDenom = Variable.GaussianFromMeanAndPrecision(hDenom[item], h2Precision / Variable.Double(innerCount[item]));
                            if (!useCorrection)
                                fakeHDenom = Variable.Copy(hDenom[item]);
                            var y2 = Variable.GaussianFromMeanAndPrecision(fakeHDenom, yPrecision);
                            Variable.ConstrainEqual(y[item][inner], y2);
                        }
                    }
                }
            }
            double hMeanTrue = 0;
            double hPrecisionTrue = 1;
            double yPrecisionTrue = 1.0;
            double xPrecisionTrue = 1.0e-1;
            double h2PrecisionTrue = 1e4;
            //hMean.ObservedValue = hMeanTrue;
            //yPrecision.ObservedValue = yPrecisionTrue;
            //yPrecision.ObservedValue = 1e-2;
            //hPrecision.ObservedValue = hPrecisionTrue;
            //h2Precision.ObservedValue = h2PrecisionTrue;

            // generate data from the model
            var hPrior = new Gaussian(hMeanTrue, 1 / hPrecisionTrue);
            var hSample = Util.ArrayInit(n, i => hPrior.Sample());
            var h2Sample = Util.ArrayInit(n, i => Gaussian.Sample(hSample[i], h2PrecisionTrue));
            innerCount.ObservedValue = Util.ArrayInit(n, i => Rand.Int(100, 200));
            //innerCount.ObservedValue = Util.ArrayInit(n, i => 100);
            var xData = Util.ArrayInit(n, i => Util.ArrayInit(innerCount.ObservedValue[i], j => Gaussian.Sample(hSample[i], xPrecisionTrue)));
            var yData = Util.ArrayInit(n, i => Util.ArrayInit(innerCount.ObservedValue[i], j => Gaussian.Sample(h2Sample[i], yPrecisionTrue)));
            x.ObservedValue = xData;
            y.ObservedValue = yData;

            var engine = new InferenceEngine();
            engine.Compiler.GivePriorityTo(typeof(GaussianOp_PointPrecision));
            for (int iter = 1; iter < 100; iter++)
            {
                engine.NumberOfIterations = iter;
                Console.WriteLine("{0} xPrec={1:g4} yPrec={2:g4} hMean={3:g4} hPrec={4:g4} h2Prec={5:g4}", iter,
                    engine.Infer<Gamma>(xPrecision).Point,
                    engine.Infer<Gamma>(yPrecision).Point,
                    engine.Infer<Gaussian>(hMean).Point,
                    engine.Infer<Gamma>(hPrecision).Point,
                    engine.Infer<Gamma>(h2Precision).Point);
            }

            var hActual = engine.Infer<IList<Gaussian>>(h);
            double hError = 0;
            for (int i = 0; i < n; i++)
            {
                hError += hActual[i].GetMean() - hSample[i];
            }
            hError /= n;
            Console.WriteLine($"hError = {hError}");

            var xPrecisionActual = engine.Infer<Gamma>(xPrecision);
            Console.WriteLine("xPrecision = {0} should be {1}", xPrecisionActual, xPrecisionTrue);

            var yPrecisionActual = engine.Infer<Gamma>(yPrecision);
            Console.WriteLine("yPrecision = {0} should be {1}", yPrecisionActual, yPrecisionTrue);

            var hMeanActual = engine.Infer<Gaussian>(hMean);
            Console.WriteLine("hMean = {0} should be {1}", hMeanActual, hMeanTrue);

            var hPrecisionActual = engine.Infer<Gamma>(hPrecision);
            Console.WriteLine("hPrecision = {0} should be {1}", hPrecisionActual, hPrecisionTrue);

            var h2PrecisionActual = engine.Infer<Gamma>(h2Precision);
            Console.WriteLine("h2Precision = {0} should be {1}", h2PrecisionActual, h2PrecisionTrue);

            if (useFakeModel)
                Console.WriteLine("effective y variance = {0} (should be {1})",
                    1 / yPrecisionActual.GetMean() + innerCount.ObservedValue[0] / h2PrecisionActual.GetMean(),
                    1 / yPrecisionTrue + innerCount.ObservedValue[0] / h2PrecisionTrue);
        }

        /// <summary>
        /// Test using a repeat block to construct a discriminative model.
        /// </summary>
        internal void DiscriminativeTest()
        {
            Rand.Restart(0);
            int n = 1000;
            Range item = new Range(n).Named("item");
            double hMean = 0;
            double hVariance = 1;
            var xPrecision = Variable.GammaFromShapeAndRate(1, 1).Named("xPrecision");
            xPrecision.AddAttribute(new PointEstimate());
            var yPrecision = Variable.GammaFromShapeAndRate(1, 1).Named("yPrecision");
            var h = Variable.Array<double>(item).Named("h");
            var x = Variable.Array<double>(item).Named("x");
            var y = Variable.Array<double>(item).Named("y");
            using (Variable.ForEach(item))
            {
                h[item] = Variable.GaussianFromMeanAndVariance(hMean, hVariance);
                x[item] = Variable.GaussianFromMeanAndPrecision(h[item], xPrecision);
                y[item] = Variable.GaussianFromMeanAndPrecision(h[item], yPrecision);

                if (true)
                {
                    using (Variable.Repeat(-1))
                    {
                        var h2 = Variable.GaussianFromMeanAndVariance(hMean, hVariance);
                        var x2 = Variable.GaussianFromMeanAndPrecision(h2, xPrecision);
                        Variable.ConstrainEqual(x[item], x2);
                    }
                }
            }
            double yPrecisionTrue = 1.0;
            double xPrecisionTrue = 1.0;
            yPrecision.ObservedValue = yPrecisionTrue;

            // generate data from the model
            var hPrior = new Gaussian(hMean, hVariance);
            var hSample = Util.ArrayInit(n, i => hPrior.Sample());
            // When xMultiplier != 1, we have model mismatch so we want the learned xPrecision to decrease.
            double xMultiplier = 5;
            var xData = Util.ArrayInit(n, i => Gaussian.Sample(xMultiplier * hSample[i], xPrecisionTrue));
            var yData = Util.ArrayInit(n, i => Gaussian.Sample(hSample[i], yPrecisionTrue));
            x.ObservedValue = xData;
            y.ObservedValue = yData;

            // N(x; a*h, vx) N(h; mh, vh) = N(h; mh + k*(x - a*mh), (1-ka)vh)
            // where k = vh*a/(a^2*vh + vx)
            // if x = a*x' then k(x - a*mh) = a*k(x' - mh)
            // a*k = vh/(vh + vx/a^2)
            // thus vx' = vx/a^2
            // p(h | x', vx', a=1) = p(h | x, vx, a)
            // we want mh + k'*(x - mh) = mh + k*(x - a*mh)
            // since mh=0, k'*x = k*x  so k'=k
            // k' = vh/(vh + vx')
            // vx'/vh + 1 = vx/vh/a + a
            // vx' = vx/a + (a-1)*vh
            // without the denominator, the learned xPrecision is much too high

            var engine = new InferenceEngine();
            engine.Compiler.GivePriorityTo(typeof(GaussianOp_PointPrecision));
            for (int iter = 1; iter < 100; iter++)
            {
                engine.NumberOfIterations = iter;
                Console.WriteLine("{0} {1}", iter, engine.Infer(xPrecision));
            }

            var xPrecisionActual = engine.Infer<Gamma>(xPrecision);
            double xPrecisionExpected = 1 / (xPrecisionTrue / xMultiplier + (xMultiplier - 1) * hVariance);
            Console.WriteLine("xPrecision = {0} should be {1}", xPrecisionActual, xPrecisionExpected);
        }

        /// <summary>
        /// Test different ways of representing a model
        /// </summary>
        internal void FactorAnalysisTest()
        {
            Range player = new Range(1000);
            var basePrecisionRate = Variable.GammaFromShapeAndRate(1, 1);
            var basePrecision = Variable.GammaFromShapeAndRate(1, basePrecisionRate);
            basePrecision.AddAttribute(new PointEstimate());
            //basePrecision.InitialiseTo(Gamma.PointMass(0.01));
            basePrecision.Name = nameof(basePrecision);
            var baseSkill = Variable.Array<double>(player);
            baseSkill.Name = nameof(baseSkill);
            baseSkill[player] = Variable.GaussianFromMeanAndPrecision(0, basePrecision).ForEach(player);
            Range mode = new Range(2);
            var offsetPrecisionRate = Variable.GammaFromShapeAndRate(1, 1);
            var offsetPrecision = Variable.Array<double>(mode);
            offsetPrecision.Name = nameof(offsetPrecision);
            offsetPrecision[mode] = Variable.GammaFromShapeAndRate(1, offsetPrecisionRate).ForEach(mode);
            offsetPrecision.AddAttribute(new PointEstimate());
            var offset = Variable.Array(Variable.Array<double>(mode), player);
            offset[player][mode] = Variable.GaussianFromMeanAndPrecision(0, offsetPrecision[mode]).ForEach(player);
            var skill = Variable.Array(Variable.Array<double>(mode), player);
            skill[player][mode] = baseSkill[player] + offset[player][mode];

            // sample from model
            Rand.Restart(0);
            double trueBasePrecision = 5;
            Gaussian basePrior = Gaussian.FromMeanAndPrecision(0, trueBasePrecision);
            double[] trueBaseSkills = Util.ArrayInit(player.SizeAsInt, i => basePrior.Sample());
            double[] trueOffsetPrecisions = Util.ArrayInit(mode.SizeAsInt, j => 1.0);
            double[][] trueOffsets = Util.ArrayInit(player.SizeAsInt, i =>
                Util.ArrayInit(mode.SizeAsInt, j =>
                    Gaussian.FromMeanAndPrecision(0, trueOffsetPrecisions[j]).Sample()));
            double[][] trueSkills = Util.ArrayInit(player.SizeAsInt, i =>
                Util.ArrayInit(mode.SizeAsInt, j =>
                    trueBaseSkills[i] + trueOffsets[i][j]));

            bool gaussianLikelihood = true;
            if (gaussianLikelihood)
            {
                var data = Variable.Observed(default(Gaussian[][]), player, mode);
                double dataPrecision = 100;
                bool redundantAddition = false;
                if (redundantAddition)
                {
                    int gameCount = 100;
                    Range game = new Range(gameCount);
                    using (Variable.ForEach(game))
                    {
                        var skillInGame = Variable.Array(Variable.Array<double>(mode), player);
                        skillInGame[player][mode] = baseSkill[player] + offset[player][mode];
                        Variable.ConstrainEqualRandom(skillInGame[player][mode], data[player][mode]);
                    }
                    dataPrecision /= gameCount;
                }
                else
                {
                    Variable.ConstrainEqualRandom(skill[player][mode], data[player][mode]);
                }

                data.ObservedValue = Util.ArrayInit(player.SizeAsInt, i =>
                    Util.ArrayInit(mode.SizeAsInt, j =>
                        Gaussian.FromMeanAndPrecision(trueSkills[i][j], dataPrecision)));
            }
            else
            {
                var gameCount = 10000;
                Range game = new Range(gameCount);
                var observedModes = Variable.Observed(Util.ArrayInit(gameCount, i => Rand.Int(mode.SizeAsInt)), game);
                List<int> observedWinner = new List<int>();
                List<int> observedLoser = new List<int>();
                var allPlayers = Enumerable.Range(0, player.SizeAsInt).ToArray();
                for (int i = 0; i < gameCount; i++)
                {
                    var modeOfGame = observedModes.ObservedValue[i];
                    var players = Rand.SampleWithoutReplacement(allPlayers, 2).ToList();
                    if (trueSkills[players[0]][modeOfGame] > trueSkills[players[1]][modeOfGame])
                    {
                        observedWinner.Add(players[0]);
                        observedLoser.Add(players[1]);
                    }
                    else
                    {
                        observedWinner.Add(players[1]);
                        observedLoser.Add(players[0]);
                    }
                }
                var winner = Variable.Observed(observedWinner, game);
                var loser = Variable.Observed(observedLoser, game);
                using (Variable.ForEach(game))
                {
                    var modeOfGame = observedModes[game];
                    Variable.ConstrainTrue(skill[winner[game]][modeOfGame] > skill[loser[game]][modeOfGame]);
                }
            }

            InferenceEngine engine = new InferenceEngine();
            engine.NumberOfIterations = 100;
            //engine.Compiler.GivePriorityTo(typeof(VariablePointOp_Mean<>));
            //engine.Compiler.GivePriorityTo(typeof(GammaFromShapeAndRateOp_Laplace));
            var baseSkillActual = engine.Infer<IList<Gaussian>>(baseSkill);
            for (int i = 0; i < 5; i++)
            {
                Trace.WriteLine($"baseSkill[{i}] = {baseSkillActual[i]} should be {trueBaseSkills[i]}");
            }
            Trace.WriteLine($"basePrecision = {engine.Infer(basePrecision)} should be {trueBasePrecision}");
            Trace.WriteLine(StringUtil.JoinColumns(engine.Infer(offsetPrecision), " should be ", StringUtil.VerboseToString(trueOffsetPrecisions)));
        }

        /// <summary>
        /// Test different ways of representing a model
        /// </summary>
        internal void FactorAnalysisTest2()
        {
            Range player = new Range(1000);
            player.Name = nameof(player);
            var baseVariance = Variable.Exp(Variable.GaussianFromMeanAndPrecision(-2, 1e-2));
            baseVariance.AddAttribute(new PointEstimate());
            bool initialiseSmall = false;
            if (initialiseSmall)
            {
                baseVariance.InitialiseTo(Gamma.PointMass(0.01));
            }
            else
            {
                // If baseVariance is initialised smaller than offsetVariance, baseVariance will get stuck at zero.
                baseVariance.InitialiseTo(Gamma.PointMass(1));
            }
            baseVariance.Name = nameof(baseVariance);
            var baseSkill = Variable.Array<double>(player);
            baseSkill.Name = nameof(baseSkill);
            baseSkill[player] = Variable.GaussianFromMeanAndVariance(0, baseVariance).ForEach(player);
            Range mode = new Range(2);
            mode.Name = nameof(mode);
            var offsetVariance = Variable.Array<double>(mode);
            offsetVariance.Name = nameof(offsetVariance);
            offsetVariance[mode] = Variable.Exp(Variable.GaussianFromMeanAndPrecision(-2, 1e-2).ForEach(mode));
            offsetVariance.AddAttribute(new PointEstimate());
            offsetVariance[mode].InitialiseTo(Gamma.PointMass(0.1));
            var offset = Variable.Array(Variable.Array<double>(mode), player);
            offset.Name = nameof(offset);
            offset[player][mode] = Variable.GaussianFromMeanAndVariance(0, offsetVariance[mode]).ForEach(player);
            var skill = Variable.Array(Variable.Array<double>(mode), player);
            skill.Name = nameof(skill);
            // equivalent to: Variable.GaussianFromMeanAndVariance(baseSkill[player], offsetVariance[mode])
            skill[player][mode] = baseSkill[player] + offset[player][mode];

            // sample from model
            Rand.Restart(0);
            double trueBaseVariance = 5;
            Gaussian basePrior = Gaussian.FromMeanAndVariance(0, trueBaseVariance);
            double[] trueBaseSkills = Util.ArrayInit(player.SizeAsInt, i => basePrior.Sample());
            double[] trueOffsetVariances = Util.ArrayInit(mode.SizeAsInt, j => 1.0);
            double[][] trueOffsets = Util.ArrayInit(player.SizeAsInt, i =>
                Util.ArrayInit(mode.SizeAsInt, j =>
                    Gaussian.FromMeanAndVariance(0, trueOffsetVariances[j]).Sample()));
            double[][] trueSkills = Util.ArrayInit(player.SizeAsInt, i =>
                Util.ArrayInit(mode.SizeAsInt, j =>
                    trueBaseSkills[i] + trueOffsets[i][j]));

            int teamSize = 1;
            bool redundantAddition = false;
            bool gaussianLikelihood = true;
            bool sequential = false;
            if (gaussianLikelihood)
            {
                var data = Variable.Observed(default(Gaussian[][]), player, mode);
                // If dataPrecision is smaller than true offsetVariance, learned offsetVariance goes to zero.
                double dataPrecision = 10;
                if (redundantAddition)
                {
                    int gameCount = 100;
                    Range game = new Range(gameCount);
                    using (Variable.ForEach(game))
                    {
                        var skillInGame = Variable.Array(Variable.Array<double>(mode), player);
                        skillInGame[player][mode] = baseSkill[player] + offset[player][mode];
                        Variable.ConstrainEqualRandom(skillInGame[player][mode], data[player][mode]);
                    }
                    dataPrecision /= gameCount;
                }
                else
                {
                    Variable.ConstrainEqualRandom(skill[player][mode], data[player][mode]);
                }

                data.ObservedValue = Util.ArrayInit(player.SizeAsInt, i =>
                    Util.ArrayInit(mode.SizeAsInt, j =>
                        Gaussian.FromMeanAndPrecision(trueSkills[i][j], dataPrecision)));
            }
            else
            {
                var gameCount = 20000;
                // With too few games and teamSize=1, redundantAddition causes learned all offsetVariances to be zero, or
                // one learned offsetVariance is zero, and the others are inflated to compensate.
                // More games just slows down convergence.
                // Sequential does not affect this.
                // The estimates have similar accuracy, but posterior variance is lower.
                // Because estimates tend to be smoothed versions of the truth, low posterior variance leads to smaller variance estimates.
                //if (redundantAddition) gameCount *= 10;
                Range game = new Range(gameCount);
                if (teamSize > 2 || (teamSize == 2 && redundantAddition))
                {
                    // teamSize>2 causes estimates to go to infinity if games are not sequential
                    // teamSize=2 with redundantAddition causes stuck state if games are not sequential
                    game.AddAttribute(new Sequential());
                    sequential = true;
                }
                var observedModes = Variable.Observed(Util.ArrayInit(gameCount, i => Rand.Int(mode.SizeAsInt)), game);
                observedModes.Name = nameof(observedModes);
                var allPlayers = Enumerable.Range(0, player.SizeAsInt).ToArray();
                if (teamSize == 1)
                {
                    // works for redundantAddition=false and true without Sequential
                    // teamSize=1, redundantAddition=true Sequential gives baseVariance=9 (others are fine)
                    // schedule is ruining the results?
                    List<int> observedWinner = new List<int>();
                    List<int> observedLoser = new List<int>();
                    for (int i = 0; i < gameCount; i++)
                    {
                        var modeOfGame = observedModes.ObservedValue[i];
                        var players = Rand.SampleWithoutReplacement(allPlayers, 2).ToList();
                        var team0Performance = trueSkills[players[0]][modeOfGame] + Rand.Normal();
                        var team1Performance = trueSkills[players[1]][modeOfGame] + Rand.Normal();
                        if (team0Performance > team1Performance)
                        {
                            observedWinner.Add(players[0]);
                            observedLoser.Add(players[1]);
                        }
                        else
                        {
                            observedWinner.Add(players[1]);
                            observedLoser.Add(players[0]);
                        }
                    }
                    var winner = Variable.Observed(observedWinner, game);
                    winner.Name = nameof(winner);
                    var loser = Variable.Observed(observedLoser, game);
                    loser.Name = nameof(loser);
                    using (Variable.ForEach(game))
                    {
                        var modeOfGame = observedModes[game];
                        Variable<double> winnerSkill;
                        Variable<double> loserSkill;
                        if (redundantAddition)
                        {
                            winnerSkill = baseSkill[winner[game]] + offset[winner[game]][modeOfGame];
                            loserSkill = baseSkill[loser[game]] + offset[loser[game]][modeOfGame];
                        }
                        else
                        {
                            winnerSkill = skill[winner[game]][modeOfGame];
                            loserSkill = skill[loser[game]][modeOfGame];
                        }
                        var winnerPerformance = Variable.GaussianFromMeanAndPrecision(winnerSkill, 1);
                        winnerPerformance.Name = nameof(winnerPerformance);
                        var loserPerformance = Variable.GaussianFromMeanAndPrecision(loserSkill, 1);
                        loserPerformance.Name = nameof(loserPerformance);
                        Variable.ConstrainTrue(winnerPerformance > loserPerformance);

                    }
                }
                else
                {
                    // teamSize=1, redundantAddition=true gives baseVariance=9 (others are fine)
                    // teamSize=2, redundantAddition=true gives baseVariance=14 (others are fine)
                    // teamSize=3, redundantAddition=true gives baseVariance=18 (others are fine)
                    // teamSize=4, redundantAddition=true gives baseVariance=22 (others are fine)
                    List<int[]> observedWinner = new List<int[]>();
                    List<int[]> observedLoser = new List<int[]>();
                    for (int i = 0; i < gameCount; i++)
                    {
                        var modeOfGame = observedModes.ObservedValue[i];
                        var players = Rand.SampleWithoutReplacement(allPlayers, 2 * teamSize).ToList();
                        var team0 = Collection.Split(players, teamSize, out int[] team1);
                        var team0Performance = team0.Select(p => trueSkills[p][modeOfGame] + Rand.Normal()).Sum();
                        var team1Performance = team1.Select(p => trueSkills[p][modeOfGame] + Rand.Normal()).Sum();
                        if (team0Performance > team1Performance)
                        {
                            observedWinner.Add(team0);
                            observedLoser.Add(team1);
                        }
                        else
                        {
                            observedWinner.Add(team1);
                            observedLoser.Add(team0);
                        }
                    }
                    Range playerOnTeam = new Range(teamSize);
                    playerOnTeam.Name = nameof(playerOnTeam);
                    var winner = Variable.Observed(observedWinner, game, playerOnTeam);
                    winner.Name = nameof(winner);
                    var loser = Variable.Observed(observedLoser, game, playerOnTeam);
                    loser.Name = nameof(loser);
                    using (Variable.ForEach(game))
                    {
                        var modeOfGame = observedModes[game];
                        VariableArray<double> winnerSkills = Variable.Array<double>(playerOnTeam);
                        winnerSkills.Name = nameof(winnerSkills);
                        VariableArray<double> loserSkills = Variable.Array<double>(playerOnTeam);
                        loserSkills.Name = nameof(loserSkills);
                        if (redundantAddition)
                        {
                            winnerSkills[playerOnTeam] = baseSkill[winner[game][playerOnTeam]] + offset[winner[game][playerOnTeam]][modeOfGame];
                            loserSkills[playerOnTeam] = baseSkill[loser[game][playerOnTeam]] + offset[loser[game][playerOnTeam]][modeOfGame];
                        }
                        else
                        {
                            winnerSkills[playerOnTeam] = skill[winner[game][playerOnTeam]][modeOfGame];
                            loserSkills[playerOnTeam] = skill[loser[game][playerOnTeam]][modeOfGame];
                        }
                        var winnerPerformances = Variable.Array<double>(playerOnTeam);
                        winnerPerformances.Name = nameof(winnerPerformances);
                        winnerPerformances[playerOnTeam] = Variable.GaussianFromMeanAndPrecision(winnerSkills[playerOnTeam], 1);
                        var winnerPerformance = Variable.Sum(winnerPerformances);
                        winnerPerformance.Name = nameof(winnerPerformance);
                        var loserPerformances = Variable.Array<double>(playerOnTeam);
                        loserPerformances.Name = nameof(loserPerformances);
                        loserPerformances[playerOnTeam] = Variable.GaussianFromMeanAndPrecision(loserSkills[playerOnTeam], 1);
                        var loserPerformance = Variable.Sum(loserPerformances);
                        loserPerformance.Name = nameof(loserPerformance);
                        Variable.ConstrainTrue(winnerPerformance > loserPerformance);
                    }
                }
            }

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.UseParallelForLoops = !sequential;
            //engine.Compiler.GivePriorityTo(typeof(VariablePointOp_Mean<>));
            //engine.Compiler.GivePriorityTo(typeof(GammaFromShapeAndRateOp_Laplace));
            engine.Compiler.GivePriorityTo(typeof(GaussianOp_PointPrecision));
            engine.Compiler.GivePriorityTo(typeof(GaussianFromMeanAndVarianceOp_PointVariance));
            for (int i = 0; i < 1000; i++)
            {
                engine.NumberOfIterations = i + 1;
                Trace.WriteLine($"baseVariance = {engine.Infer(baseVariance)} should be {trueBaseVariance}");
                Trace.WriteLine(StringUtil.JoinColumns(engine.Infer(offsetVariance), " should be ", StringUtil.VerboseToString(trueOffsetVariances)));
            }
            var baseSkillActual = engine.Infer<IReadOnlyList<Gaussian>>(baseSkill);
            var offsetSkillActual = engine.Infer<IReadOnlyList<IReadOnlyList<Gaussian>>>(offset);
            for (int i = 0; i < 5; i++)
            {
                // baseSkills are not always accurate since they are inferred to be between the mode skills.
                Trace.WriteLine($"baseSkill[{i}] = {baseSkillActual[i]} should be {trueBaseSkills[i]}");
                for (int j = 0; j < mode.SizeAsInt; j++)
                {
                    Trace.WriteLine($"offset[{i}][{j}] = {offsetSkillActual[i][j]} should be {trueOffsets[i][j]}");
                }
            }
            Trace.WriteLine($"baseVariance = {engine.Infer(baseVariance)} should be {trueBaseVariance}");
            Trace.WriteLine(StringUtil.JoinColumns(engine.Infer(offsetVariance), " should be ", StringUtil.VerboseToString(trueOffsetVariances)));
        }

        /// <summary>
        /// Demonstrates how improper distributions can arise with compound gamma priors.
        /// </summary>
        [Fact]
        public void CompoundGammaTest()
        {
            var rateRate = Variable.GammaFromShapeAndRate(1, 1).Named("rateRate");
            var rate = Variable.GammaFromShapeAndRate(1, rateRate).Named("rate");
            var prec = Variable.GammaFromShapeAndRate(1, rate).Named("prec");
            var w1 = Variable.GaussianFromMeanAndPrecision(0, prec).Named("w1");
            var w2 = Variable.GaussianFromMeanAndPrecision(0, prec).Named("w2");
            if (false)
            {
                var sum = w1 + w2;
                Variable.ConstrainEqual(sum, 0.0);
            }
            else
            {
                Variable.ConstrainEqual(w1, w2); // this wants prec to be large
            }
            Variable.ConstrainEqualRandom(rate, Gamma.FromShapeAndRate(20, 0));

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.GivePriorityTo(typeof(GammaFromShapeAndRateOp_Laplace));
            engine.ShowProgress = false;
            for (int iter = 1; iter < 100; iter++)
            {
                engine.NumberOfIterations = iter;
                // prec is inferred small and rate is large
                Console.WriteLine("{0}: {1} {2}", iter, engine.Infer(prec), engine.Infer(rate));
            }
        }

        /// <summary>
        /// Demonstrates how improper distributions can arise with compound gamma priors.
        /// </summary>
        [Fact]
        public void CompoundGammaTest2()
        {
            var rateRate = Variable.GammaFromShapeAndRate(1, 1).Named("rateRate");
            var rate = Variable.GammaFromShapeAndRate(1, rateRate).Named("rate");
            var prec1 = Variable.GammaFromShapeAndRate(1, rate).Named("prec1");
            var prec2 = Variable.GammaFromShapeAndRate(1, rate).Named("prec2");
            Variable.ConstrainEqual(prec1, prec2);  // this wants rate to be large
            Variable.ConstrainEqualRandom(rate, Gamma.FromShapeAndRate(20, 0));

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.GivePriorityTo(typeof(GammaFromShapeAndRateOp_Laplace));
            engine.ShowProgress = false;
            for (int iter = 1; iter < 100; iter++)
            {
                engine.NumberOfIterations = iter;
                Console.WriteLine("{0}: {1} {2}", iter, engine.Infer(prec1), engine.Infer(rate));
            }
        }

        [Fact]
        public void LogExpTest()
        {
            var evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> x = Variable.GaussianFromMeanAndPrecision(1, 1).Named("x");
            var ExpX = Variable.Exp(x).Named("ExpX");
            var y = Variable.Log(ExpX).Named("y");
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            y.ObservedValue = 2;
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            double evExpected = Gaussian.FromMeanAndPrecision(1, 1).GetLogProb(y.ObservedValue);
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evActual, evExpected, 1e-8) < 1e-8);
        }

        [Fact]
        public void BugsRatsSequential()
        {
            // Height data
            double[,] RatsHeightData = new double[,]
            {
             {151, 199, 246, 283, 320},
             {145, 199, 249, 293, 354},
             {147, 214, 263, 312, 328},
             {155, 200, 237, 272, 297},
             {135, 188, 230, 280, 323},
             {159, 210, 252, 298, 331},
             {141, 189, 231, 275, 305},
             {159, 201, 248, 297, 338},
             {177, 236, 285, 350, 376},
             {134, 182, 220, 260, 296},
             {160, 208, 261, 313, 352},
             {143, 188, 220, 273, 314},
             {154, 200, 244, 289, 325},
             {171, 221, 270, 326, 358},
             {163, 216, 242, 281, 312},
             {160, 207, 248, 288, 324},
             {142, 187, 234, 280, 316},
             {156, 203, 243, 283, 317},
             {157, 212, 259, 307, 336},
             {152, 203, 246, 286, 321},
             {154, 205, 253, 298, 334},
             {139, 190, 225, 267, 302},
             {146, 191, 229, 272, 302},
             {157, 211, 250, 285, 323},
             {132, 185, 237, 286, 331},
             {160, 207, 257, 303, 345},
             {169, 216, 261, 295, 333},
             {157, 205, 248, 289, 316},
             {137, 180, 219, 258, 291},
             {153, 200, 244, 286, 324}
            };

            // x data
            double[] RatsXData = { 8.0, 15.0, 22.0, 29.0, 36.0 };

            Rand.Restart(12347);

            // The model
            Range N = new Range(RatsHeightData.GetLength(0)).Named("N");
            Range T = new Range(RatsHeightData.GetLength(1)).Named("T");

            Variable<double> alphaC = Variable.GaussianFromMeanAndPrecision(0.0, 1e-4).Named("alphaC");
            Variable<double> alphaTau = Variable.GammaFromShapeAndRate(1e-3, 1e-3).Named("alphaTau");
            VariableArray<double> alpha = Variable.Array<double>(N).Named("alpha");
            alpha[N] = Variable.GaussianFromMeanAndPrecision(alphaC, alphaTau).ForEach(N);

            Variable<double> betaC = Variable.GaussianFromMeanAndPrecision(0.0, 1e-4).Named("betaC");
            Variable<double> betaTau = Variable.GammaFromShapeAndRate(1e-3, 1e-3).Named("betaTau");
            VariableArray<double> beta = Variable.Array<double>(N).Named("beta");
            beta[N] = Variable.GaussianFromMeanAndPrecision(betaC, betaTau).ForEach(N);

            Variable<double> tauC = Variable.GammaFromShapeAndRate(1e-3, 1e-3).Named("tauC");
            VariableArray<double> x = Variable.Observed<double>(RatsXData, T).Named("x");
            Variable<double> xbar = Variable.Sum(x) / T.SizeAsInt;
            VariableArray2D<double> y = Variable.Observed<double>(RatsHeightData, N, T).Named("y");
            y[N, T] = Variable.GaussianFromMeanAndPrecision(alpha[N] + (beta[N] * (x[T] - xbar)), tauC);
            Variable<double> alpha0 = (alphaC - betaC * xbar).Named("alpha0");

            if (false)
            {
                // Initialise with the mean of the prior (needed for Gibbs to converge quickly)
                alphaC.InitialiseTo(Gaussian.PointMass(0.0));
                tauC.InitialiseTo(Gamma.PointMass(1.0));
                alphaTau.InitialiseTo(Gamma.PointMass(1.0));
                betaTau.InitialiseTo(Gamma.PointMass(1.0));
            }

            // Inference engine
            InferenceEngine ie = new InferenceEngine();
            N.AddAttribute(new Sequential());
            Gaussian betaCMarg = ie.Infer<Gaussian>(betaC);
            Gaussian alpha0Marg = ie.Infer<Gaussian>(alpha0);
            Gamma tauCMarg = ie.Infer<Gamma>(tauC);

            // Inference
            Console.WriteLine("alpha0 = {0}[sd={1}]", alpha0Marg, System.Math.Sqrt(alpha0Marg.GetVariance()).ToString("g4"));
            Console.WriteLine("betaC = {0}[sd={1}]", betaCMarg, System.Math.Sqrt(betaCMarg.GetVariance()).ToString("g4"));
            Console.WriteLine("tauC = {0}", tauCMarg);
        }

        [Fact]
        public void BinomialPoissonTest()
        {
            Variable<int> nObjects = Variable.Poisson(10).Named("nObjects");
            double probDetect = 0.9;
            Variable<int> nDetected = Variable.Binomial(nObjects, probDetect).Named("nDetected");
            //nDetected.AddAttribute(new MarginalPrototype(new Poisson()));
            double meanSpurious = 2;
            Variable<int> nSpurious = Variable.Poisson(meanSpurious).Named("nSpurious");
            Variable<int> nObserved = (nDetected + nSpurious).Named("nObserved");
            nObserved.ObservedValue = 7;

            InferenceEngine engine = new InferenceEngine();
            Poisson nObjectsActual = engine.Infer<Poisson>(nObjects);

            // compute exact mean
            double totalProb = 0;
            double mean = 0;
            int maxObjects = 100;
            int x = nObserved.ObservedValue;
            Poisson nPrior = new Poisson(10);
            Poisson spuriousPrior = new Poisson(meanSpurious);
            for (int n = 0; n < maxObjects; n++)
            {
                double probN = 0;
                int max_d = System.Math.Min(n, x);
                Binomial binom = new Binomial(n, probDetect);
                for (int d = 0; d <= max_d; d++)
                {
                    double probD = System.Math.Exp(binom.GetLogProb(d) + spuriousPrior.GetLogProb(x - d));
                    probN += probD;
                }
                probN *= System.Math.Exp(nPrior.GetLogProb(n));
                totalProb += probN;
                mean += n * probN;
            }
            mean /= totalProb;

            Poisson nObjectsExpected = new Poisson(mean);
            Console.WriteLine("nObjects = {0} should be {1}", nObjectsActual, nObjectsExpected);
            Assert.True(nObjectsExpected.MaxDiff(nObjectsActual) < 1e-4);
        }
        [Fact]
        public void BinomialPoissonTest2()
        {
            Variable<int> prevObjects = Variable.Poisson(10).Named("prevObjects");
            double probSurvive = 0.8;
            Variable<int> nSurvived = Variable.Binomial(prevObjects, probSurvive).Named("nSurvived");
            double meanBirth = 3;
            Variable<int> nBirth = Variable.Poisson(meanBirth).Named("nBirth");
            Variable<int> nObjects = (nSurvived + nBirth).Named("nObjects");
            double probDetect = 0.9;
            Variable<int> nDetected = Variable.Binomial(nObjects, probDetect).Named("nDetected");
            //nDetected.AddAttribute(new MarginalPrototype(new Poisson()));
            double meanSpurious = 2;
            Variable<int> nSpurious = Variable.Poisson(meanSpurious).Named("nSpurious");
            Variable<int> nObserved = (nDetected + nSpurious).Named("nObserved");
            nObserved.ObservedValue = 7;

            InferenceEngine engine = new InferenceEngine();
            Poisson nObjectsActual = engine.Infer<Poisson>(nObjects);

            // compute exact mean
            double totalProb = 0;
            double mean = 0;
            int maxObjects = 100;
            int x = nObserved.ObservedValue;
            Poisson nPrior = new Poisson(10);
            Poisson spuriousPrior = new Poisson(meanSpurious);
            for (int n = 0; n < maxObjects; n++)
            {
                double probN = 0;
                int max_d = System.Math.Min(n, x);
                Binomial binom = new Binomial(n, probDetect);
                for (int d = 0; d <= max_d; d++)
                {
                    double probD = System.Math.Exp(binom.GetLogProb(d) + spuriousPrior.GetLogProb(x - d));
                    probN += probD;
                }
                probN *= System.Math.Exp(nPrior.GetLogProb(n));
                totalProb += probN;
                mean += n * probN;
            }
            mean /= totalProb;

            Poisson nObjectsExpected = new Poisson(mean);
            Console.WriteLine("nObjects = {0} should be {1}", nObjectsActual, nObjectsExpected);
            //Assert.True(nObjectsExpected.MaxDiff(nObjectsActual) < 1e-4);
        }

        [Fact]
        public void RegressionLearningNoiseTest()
        {
            double[] xobs =
                {
                    -0.2,
                                0.3,
                                0.8,
                                1.3,
                                1.8,
                                2.3,
                                2.8,
                                3.3,
                                3.8,
                                4.3,
                                4.8,
                    5.3
                };
            double[] yobs =
                {
                    5.074476033,
                                6.600718815,
                                4.884130877,
                                4.417261879,
                                3.381936761,
                                3.97316699,
                                3.990442347,
                                4.120380425,
                                6.295349392,
                                2.835300298,
                                2.842412922,
                    3.007296355
                };
            int n = xobs.Length;

            Range index = new Range(n).Named("index");
            //index.AddAttribute(new Sequential());

            VariableArray<double> y = Variable.Array<double>(index).Named("y");
            VariableArray<double> x = Variable.Array<double>(index).Named("x");
            VariableArray<double> mu = Variable.Array<double>(index).Named("mu");

            Variable<double> beta0 = Variable.GaussianFromMeanAndVariance(0.0, 10).Named("beta0");
            Variable<double> beta1 = Variable.GaussianFromMeanAndVariance(0.0, 10).Named("beta1");
            Variable<double> tau = Variable.GammaFromShapeAndRate(1, 1).Named("tau");
            mu[index] = beta0 + beta1 * x[index];
            y[index] = Variable.GaussianFromMeanAndPrecision(mu[index], tau);

            x.ObservedValue = xobs;
            y.ObservedValue = yobs;

            var engine = new InferenceEngine();
            //engine.Compiler.UnrollLoops = true;
            //beta0.InitialiseTo(Gaussian.FromMeanAndVariance(0, 10));
            //tau.InitialiseTo(Gamma.FromShapeAndRate(1, 1));
            //mu.InitialiseTo(Distribution<double>.Array(Util.ArrayInit(n, i => new Gaussian(0, 1))));

            Console.WriteLine("beta0=" + engine.Infer(beta0));
            Console.WriteLine("beta1=" + engine.Infer(beta1));
            Console.WriteLine("tau=" + engine.Infer(tau));
        }

        internal void IsBetweenErrorBoundTest()
        {
            Gaussian prior = new Gaussian(0, 1);
            double lowerBound = -1;
            double upperBound = 1;
            Bernoulli isBetween = Bernoulli.PointMass(true);
            Gaussian lowerMsg = Gaussian.Uniform();
            Gaussian upperMsg = Gaussian.Uniform();
            for (int iter = 0; iter < 10; iter++)
            {
                Trace.WriteLine($"iter {iter}: lowerMsg = {lowerMsg} upperMsg = {upperMsg}");
                IsBetweenErrorBound(prior, lowerBound, upperBound, lowerMsg, upperMsg);
                //lowerMsg = DoubleIsBetweenOp.XAverageConditional(isBetween, prior * upperMsg, lowerBound, double.PositiveInfinity);
                //lowerMsg = new Gaussian(0.6947, 0.2002+ (9-iter)*0.01);
                //lowerMsg = new Gaussian(0.6947, (prior * upperMsg).GetVariance());
                //lowerMsg = new Gaussian(0.3503, 1 + (9 - iter) * 1);
                upperMsg = IsBetweenGaussianOp.XAverageConditional(isBetween, prior * lowerMsg, double.NegativeInfinity, upperBound);
                //upperMsg = new Gaussian(-0.3503, 1 + (9 - iter) * 1);
            }
        }

        internal double IsBetweenErrorBound(Gaussian prior, double lowerBound, double upperBound, Gaussian lowerMsg, Gaussian upperMsg)
        {
            // msgs are scaled by normalizer
            double Z = System.Math.Exp(IsBetweenGaussianOp.LogProbBetween(prior, lowerBound, upperBound));
            double lowerLogNormalizer = lowerMsg.GetLogNormalizer();
            double lowerPriorLogScale = prior.GetLogAverageOf(lowerMsg) + lowerLogNormalizer;
            double upperLogNormalizer = upperMsg.GetLogNormalizer();
            double upperPriorLogScale = prior.GetLogAverageOf(upperMsg) + upperLogNormalizer;
            Gaussian upperPrior = prior * upperMsg;
            double lowerLogZ = IsBetweenGaussianOp.LogProbBetween(upperPrior, lowerBound, double.PositiveInfinity) + upperPriorLogScale;
            Gaussian lowerPrior = prior * lowerMsg;
            double upperLogZ = IsBetweenGaussianOp.LogProbBetween(lowerPrior, double.NegativeInfinity, upperBound) + lowerPriorLogScale;
            double qLogScale = lowerPriorLogScale + lowerPrior.GetLogAverageOf(upperMsg) + upperLogNormalizer;
            double qLogScale2 = upperPriorLogScale + upperPrior.GetLogAverageOf(lowerMsg) + lowerLogNormalizer;
            double Zt = System.Math.Exp(lowerLogZ + upperLogZ - qLogScale);
            Trace.WriteLine($"Z = {Z} Zt = {Zt} qLogScale = {qLogScale} {qLogScale2}");
            double error = System.Math.Pow(Z - Zt, 2);
            double lowerDenomLogScale = upperPrior.GetLogAverageOfPower(lowerMsg, -1) - lowerLogNormalizer + upperPriorLogScale;
            Gaussian lowerDenom = upperPrior / lowerMsg;
            double lowerZ2;
            if (lowerDenom.IsProper())
                lowerZ2 = System.Math.Exp(IsBetweenGaussianOp.LogProbBetween(lowerDenom, lowerBound, double.PositiveInfinity) + lowerDenomLogScale);
            else
                lowerZ2 = double.PositiveInfinity;
            double upperDenomLogScale = lowerPrior.GetLogAverageOfPower(upperMsg, -1) - upperLogNormalizer + lowerPriorLogScale;
            Gaussian upperDenom = lowerPrior / upperMsg;
            double upperZ2;
            if (upperDenom.IsProper())
                upperZ2 = System.Math.Exp(IsBetweenGaussianOp.LogProbBetween(upperDenom, double.NegativeInfinity, upperBound) + upperDenomLogScale);
            else
                upperZ2 = double.PositiveInfinity;
            Trace.WriteLine($"lowerDenom = {lowerDenom} upperDenom = {upperDenom}");
            double errorBoundLower = lowerZ2 - System.Math.Exp(2 * lowerLogZ - qLogScale);
            double errorBoundUpper = upperZ2 - System.Math.Exp(2 * upperLogZ - qLogScale);
            double errorBound = errorBoundLower * errorBoundUpper;
            Trace.WriteLine($"error = {error} errorBound = {errorBound}");
            return errorBound;
        }

        // example of inference failure due to deterministic loops
        internal void DecodingTest()
        {
            Variable<bool>[] m = new Variable<bool>[4];
            for (int i = 0; i < m.Length; i++)
            {
                m[i] = Variable.Bernoulli(0.3);
            }
            Variable<bool> s = gf2sum4vectors(m);
            s.ObservedValue = true;
            InferenceEngine engine = new InferenceEngine();
            //engine.Algorithm = new VariationalMessagePassing();
            //engine.NumberOfIterations = 15;
            for (int i = 0; i < m.Length; i++)
            {
                Console.WriteLine(engine.Infer(m[i]));
            }
        }
        private Variable<bool> gf2sum4vectors(Variable<bool>[] m)
        {
            Variable<bool>[] partialSums = new Variable<bool>[m.Length];
            for (int i = 0; i < m.Length; i++)
            {
                if (i == 0)
                    partialSums[i] = m[0];
                else
                    partialSums[i] = gf2sum(partialSums[i - 1], m[i]);
            }

            return partialSums[m.Length - 1];
        }
        private Variable<bool> gf2sum(Variable<bool> a, Variable<bool> b)
        {
            //return (a != b);
            return ((a & (!b)) | (b & (!a)));
        }

        internal void WishartFromBinaryData()
        {
            Rand.Restart(0);
            const double PROBIT_TO_LOGIT = 1.59576912;
            int dim = 2;
            int count = 10000;
            //count = 1000;
            double precShape = 1;
            Vector zero = Vector.Zero(dim);
            var identity = Variable.Observed(PositiveDefiniteMatrix.Identity(dim)).Named("identity");
            var rate = Variable.GammaFromShapeAndRate(1, 1).Named("rate");
            //rate.ObservedValue = 0.35;
            rate.ObservedValue = 10;
            var prec = Variable.WishartFromShapeAndRate(precShape, Variable.MatrixTimesScalar(identity, rate)).Named("prec");
            Range row = new Range(count).Named("row");
            Range col = new Range(dim).Named("col");
            var x = Variable.Array<Vector>(row);
            var y = Variable.Array(Variable.Array<double>(col), row).Named("y");
            var b = Variable.Array(Variable.Array<bool>(col), row).Named("b");
            double yPrec = 1;
            bool useGibbs = false;
            bool useLogit = false;
            double noiseVariance = useLogit ? 1.0 : 1 / yPrec;
            using (Variable.ForEach(row))
            {
                x[row] = Variable.VectorGaussianFromMeanAndPrecision(zero, prec);
                if (!useGibbs && !useLogit)
                {
                    var noisyAffinity = Variable.VectorGaussianFromMeanAndPrecision(x[row], Variable.MatrixTimesScalar(identity, yPrec));
                    b[row].SetTo(Variable<bool[]>.Factor(EpTests.IsPositive, noisyAffinity));
                }
                else if (useLogit)
                {
                    using (var colBlock = Variable.ForEach(col))
                    {
                        var xitem = Variable.GetItem(x[row], colBlock.Index);
                        //y[row][col] = Variable.GaussianFromMeanAndPrecision(xitem, yPrec);
                        //b[row][col] = Variable.Bernoulli(Variable.Logistic(y[row][col]));
                        var scaledItem = xitem * PROBIT_TO_LOGIT;
                        if (useGibbs)
                        {
                            // use a Gaussian scale mixture to approximate a logistic distribution
                            double logisticVariance = System.Math.PI * System.Math.PI / 3;
                            double shape = 4.5;
                            Gamma precPrior = Gamma.FromShapeAndRate(shape, (shape - 1) * logisticVariance);
                            Variable<double> auxNoisePrecision = Variable.Random(precPrior).Named("auxNoisePrecision");
                            Variable<double> noisyAffinity = Variable.GaussianFromMeanAndPrecision(scaledItem, auxNoisePrecision);
                            b[row][col] = noisyAffinity > 0;
                            noisyAffinity.AddAttribute(new MarginalPrototype(new TruncatedGaussian()));
                        }
                        else
                        {
                            b[row][col] = Variable.BernoulliFromLogOdds(scaledItem);
                        }
                    }
                }
                else
                {
                    using (var colBlock = Variable.ForEach(col))
                    {
                        var xitem = Variable.GetItem(x[row], colBlock.Index);
                        Variable<double> noisyAffinity = Variable.GaussianFromMeanAndPrecision(xitem, yPrec);
                        b[row][col] = noisyAffinity > 0;
                        if (useGibbs)
                        {
                            noisyAffinity.AddAttribute(new MarginalPrototype(new TruncatedGaussian()));
                        }
                    }
                }
            }

            // generate data
            PositiveDefiniteMatrix varianceTrue = new PositiveDefiniteMatrix(dim, dim);
            varianceTrue.SetToIdentityScaledBy(1e-1);
            Vector traits = Vector.FromArray(Util.ArrayInit(dim, j => 2.0));
            varianceTrue.SetToSumWithOuter(varianceTrue, 1.0, traits, traits);
            Console.WriteLine(varianceTrue);
            Console.WriteLine();
            Console.WriteLine(NormalizedVariance(varianceTrue, noiseVariance));
            VectorGaussian xPrior = VectorGaussian.FromMeanAndPrecision(zero, varianceTrue.Inverse());
            Vector[] xTrue = Util.ArrayInit(count, i => xPrior.Sample());
            double[][] yData = Util.ArrayInit(count, i => Util.ArrayInit(dim, j => Gaussian.Sample(xTrue[i][j], yPrec)));
            bool[][] bData;
            if (useLogit)
                bData = Util.ArrayInit(count, i => Util.ArrayInit(dim, j => Bernoulli.Sample(MMath.Logistic(PROBIT_TO_LOGIT * xTrue[i][j]))));
            else
                bData = Util.ArrayInit(count, i => Util.ArrayInit(dim, j => yData[i][j] > 0));
            b.ObservedValue = bData;

            InferenceEngine engine = new InferenceEngine();
            engine.NumberOfIterations = 10;
            engine.Compiler.GivePriorityTo(typeof(LogisticOp));
            //engine.ShowProgress = false;
            //engine.Algorithm = new VariationalMessagePassing();
            if (false)
            {
                prec.AddAttribute(new PointEstimate());
                engine.Compiler.GivePriorityTo(typeof(VariablePointOp_Mean<>));
            }
            if (useGibbs)
            {
                var gibbs = new GibbsSampling();
                engine.Algorithm = gibbs;
                engine.NumberOfIterations = 1000;
                prec.InitialiseTo(Wishart.PointMass(varianceTrue.Inverse()));
            }
            Console.WriteLine("rate = {0}", engine.Infer(rate));
            Wishart precActual = engine.Infer<Wishart>(prec);
            PositiveDefiniteMatrix precMean = precActual.GetMean();
            Console.WriteLine(precMean);
            Console.WriteLine();
            PositiveDefiniteMatrix varianceEst = precMean.Inverse();
            Console.WriteLine(NormalizedVariance(varianceEst, noiseVariance));
        }

        private Matrix NormalizedVariance(Matrix variance, double noiseVariance)
        {
            Matrix result = new Matrix(variance.Rows, variance.Cols);
            result.SetToIdentityScaledBy(noiseVariance);
            result.SetToSum(result, variance);
            Vector diag = result.Diagonal();
            diag.SetToFunction(diag, v => 1.0 / System.Math.Sqrt(v));
            result.ScaleRows(diag);
            result.ScaleCols(diag);
            return result;
        }

        // This test shows that the posterior covariance inferred by EP is restricted by only receiving factorized messages
        internal void VectorIsPositiveTest()
        {
            int dim = 2;
            Vector zero = Vector.Zero(dim);
            PositiveDefiniteMatrix varianceTrue = new PositiveDefiniteMatrix(dim, dim);
            varianceTrue.SetToIdentityScaledBy(1e-1);
            Vector traits = Vector.FromArray(Util.ArrayInit(dim, j => 2.0));
            varianceTrue.SetToSumWithOuter(varianceTrue, 1.0, traits, traits);
            Console.WriteLine(varianceTrue);
            PositiveDefiniteMatrix prec = varianceTrue.Inverse();

            VectorGaussian xPrior = VectorGaussian.FromMeanAndPrecision(zero, prec);
            Console.WriteLine(xPrior.Precision);
            bool[] isPositive = Util.ArrayInit(dim, j => true);
            //bool[] isPositive = Util.ArrayInit(dim, j => (j == 0)); // mixed
            VectorGaussianEstimator est = new VectorGaussianEstimator(dim);
            for (int i = 0; i < 1000000; i++)
            {
                Vector xSample = xPrior.Sample();
                bool match = true;
                for (int j = 0; j < dim; j++)
                {
                    if (isPositive[j] != (xSample[j] > 0))
                    {
                        match = false;
                        break;
                    }
                }
                if (match)
                    est.Add(xSample);
            }
            VectorGaussian xPost = est.GetDistribution(new VectorGaussian(dim));
            Console.WriteLine(xPost);
            Console.WriteLine();
            Console.WriteLine(xPost.Precision);

            Range col = new Range(dim);
            var b = Variable.Array<bool>(col);
            Variable<Vector> x;
            LowerTriangularMatrix chol = null;
            bool useRotation = false;
            if (useRotation)
            {
                // rotated coordinates
                x = Variable.VectorGaussianFromMeanAndPrecision(zero, PositiveDefiniteMatrix.Identity(dim));
                chol = new LowerTriangularMatrix(dim, dim);
                chol.SetToCholesky(prec.Inverse());
                var cholRows = Variable.Observed(Util.ArrayInit(dim, i => chol.RowVector(i)), col);
                using (var colBlock = Variable.ForEach(col))
                {
                    var xitem = Variable.InnerProduct(x, cholRows[col]);
                    b[col] = (xitem > 0);
                }
            }
            else
            {
                x = Variable.VectorGaussianFromMeanAndPrecision(zero, prec);
                if (false)
                {
                    b.SetTo(Variable<bool[]>.Factor(EpTests.IsPositive, x));
                }
                else
                {
                    using (var colBlock = Variable.ForEach(col))
                    {
                        var xitem = Variable.GetItem(x, colBlock.Index);
                        b[col] = (xitem > 0);
                    }
                }
            }
            b.ObservedValue = isPositive;

            InferenceEngine engine = new InferenceEngine();
            var xPost2 = engine.Infer<VectorGaussian>(x);
            if (useRotation)
                xPost2 = MatrixVectorProductOp.ProductAverageConditional(chol, xPost2.GetMean(), xPost2.GetVariance(), new VectorGaussian(dim));
            Console.WriteLine(xPost2);
            Console.WriteLine();
            // the off-diagonal of EP's precision estimate always matches the prior
            Console.WriteLine(xPost2.Precision);

            // finite differences
            var ep = new VectorIsPositiveEP(dim);
            double delta = 1e-6;
            Vector mean = Vector.Zero(dim);
            mean[0] = delta;
            double Z00 = ep.GetLogEvidence(isPositive, zero, prec);
            double Z10 = ep.GetLogEvidence(isPositive, mean, prec);
            mean[0] = 0;
            mean[1] = delta;
            double Z01 = ep.GetLogEvidence(isPositive, mean, prec);
            Vector dm = Vector.FromArray(new double[] { (Z10 - Z00) / delta, (Z01 - Z00) / delta });
            dm.PredivideBy(prec);
            Console.WriteLine("post m = {0}", dm);

            mean[0] = delta;
            mean[1] = delta;
            double Z11 = ep.GetLogEvidence(isPositive, mean, prec);
            mean[0] = -delta;
            mean[1] = 0;
            double Zm10 = ep.GetLogEvidence(isPositive, mean, prec);
            mean[0] = 0;
            mean[1] = -delta;
            double Z0m1 = ep.GetLogEvidence(isPositive, mean, prec);
            double ddm00 = (Z10 - 2 * Z00 + Zm10) / (delta * delta);
            double ddm01 = (Z11 - Z10 - Z01 + Z00) / (delta * delta);
            double ddm11 = (Z01 - 2 * Z00 + Z0m1) / (delta * delta);
            Matrix ddm = new Matrix(new double[,] { { ddm00, ddm01 }, { ddm01, ddm11 } });
            PositiveDefiniteMatrix V = new PositiveDefiniteMatrix(dim, dim);
            V.SetToSum(varianceTrue, varianceTrue * ddm * varianceTrue);
            Console.WriteLine("post prec = {0}", V.Inverse());
        }

        [Hidden]
        public static bool[] IsPositive(Vector vector)
        {
            return Util.ArrayInit(vector.Count, i => vector[i] > 0);
        }

        public class VectorIsPositiveEP
        {
            readonly IGeneratedAlgorithm gen;
            public int NumberOfIterations = 100;

            public VectorIsPositiveEP(int dim)
            {
                Vector zero = Vector.Zero(dim);
                Range col = new Range(dim);
                var mean = Variable.Observed(zero).Named("mean");
                var prec = Variable.Observed(PositiveDefiniteMatrix.Identity(dim)).Named("prec");
                var evidence = Variable.Bernoulli(0.5).Named("evidence");
                var block = Variable.If(evidence);
                var x = Variable.VectorGaussianFromMeanAndPrecision(mean, prec).Named("x");
                var b = Variable.Array<bool>(col).Named("b");
                if (true)
                {
                    using (var colBlock = Variable.ForEach(col))
                    {
                        var xitem = Variable.GetItem(x, colBlock.Index);
                        b[col] = (xitem > 0);
                    }
                }
                else
                {
                    b[0] = (Variable.GetItem(x, 0) > 0);
                    b[1] = Variable.Bernoulli(0.5);
                }
                block.CloseBlock();
                b.ObservedValue = new bool[dim];

                InferenceEngine engine = new InferenceEngine();
                gen = engine.GetCompiledInferenceAlgorithm(x, evidence);
            }

            public VectorGaussian GetVectorPosterior(bool[] isPositive, PositiveDefiniteMatrix prec)
            {
                gen.SetObservedValue("b", isPositive);
                gen.SetObservedValue("prec", prec);
                gen.Execute(NumberOfIterations);
                return gen.Marginal<VectorGaussian>("x");
            }

            public double GetLogEvidence(bool[] isPositive, Vector mean, PositiveDefiniteMatrix prec)
            {
                gen.SetObservedValue("b", isPositive);
                gen.SetObservedValue("prec", prec);
                gen.SetObservedValue("mean", mean);
                gen.Execute(NumberOfIterations);
                return gen.Marginal<Bernoulli>("evidence").LogOdds;
            }
        }

        [FactorMethod(typeof(EpTests), "IsPositive")]
        [Quality(QualityBand.Experimental)]
        public static class VectorIsPositiveOp
        {
            public static int SampleCount = 1000;

            public static VectorIsPositiveEP ep;

            public static VectorGaussian VectorAverageConditional(bool[] isPositive, VectorGaussian vector, VectorGaussian result)
            {
                // use Monte Carlo to estimate the posterior
                int dim = vector.Dimension;
                VectorGaussianEstimator est = new VectorGaussianEstimator(dim);
                for (int i = 0; i < SampleCount; i++)
                {
                    Vector xSample = vector.Sample();
                    bool match = true;
                    for (int j = 0; j < dim; j++)
                    {
                        if (isPositive[j] != (xSample[j] > 0))
                        {
                            match = false;
                            break;
                        }
                    }
                    if (match)
                        est.Add(xSample);
                }
                est.GetDistribution(result);
                if (true)
                {
                    if (ep == null)
                        ep = new VectorIsPositiveEP(dim);
                    var vectorPost = ep.GetVectorPosterior(isPositive, vector.Precision);
                    //result.SetMeanAndPrecision(result.GetMean(), vectorPost.Precision);
                    result.SetMeanAndPrecision(vectorPost.GetMean(), result.Precision);
                }
                if (false)
                {
                    PositiveDefiniteMatrix variance = result.Precision.Inverse();
                    LowerTriangularMatrix chol = new LowerTriangularMatrix(dim, dim);
                    // chol*chol' = vector.Precision
                    chol.SetToCholesky(vector.Precision);
                    // inv(chol')*inv(chol) = priorVariance
                    Matrix uVariance = chol.Transpose() * variance * chol;
                    uVariance.SetToDiagonal(uVariance.Diagonal());
                    var invchol = chol.Inverse();
                    variance.SetToProduct(invchol.Transpose(), uVariance * invchol);
                    result.SetMeanAndVariance(result.GetMean(), variance);
                }
                result.SetToRatio(result, vector);
                return result;
            }
        }

        [Fact]
        public void WishartCCRTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> shape = Variable.Observed(2.5).Named("shape");
            PositiveDefiniteMatrix rateRate = new PositiveDefiniteMatrix(new double[,] { { 2, 1 }, { 1, 3 } });
            int dim = rateRate.Rows;
            var ratePrior = Wishart.FromShapeAndRate(3, rateRate);
            var rate = Variable<PositiveDefiniteMatrix>.Random(ratePrior).Named("rate");
            var x = Variable.WishartFromShapeAndRate(shape, rate).Named("x");
            x.ObservedValue = new PositiveDefiniteMatrix(new double[,] { { 5, 2 }, { 2, 7 } });
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            //engine.Algorithm = new VariationalMessagePassing();
            engine.Compiler.GivePriorityTo(typeof(WishartFromShapeAndRateOp));
            var rateActual = engine.Infer<Wishart>(rate);
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            var rateExpected = Wishart.FromShapeAndRate(5.5, x.ObservedValue + rateRate);
            double evExpected = -9.04651537762959;

            if (false)
            {
                // importance sampling
                double totalWeight = 0;
                int numIter = 100000;
                WishartEstimator est = new WishartEstimator(dim);
                for (int iter = 0; iter < numIter; iter++)
                {
                    var rateSample = ratePrior.Sample();
                    double logWeight = Wishart.FromShapeAndRate(shape.ObservedValue, rateSample).GetLogProb(x.ObservedValue);
                    double weight = System.Math.Exp(logWeight);
                    est.Add(rateSample, weight);
                    totalWeight += weight;
                }
                rateExpected = est.GetDistribution(new Wishart(dim));
                evExpected = System.Math.Log(totalWeight / numIter);
            }
            Console.WriteLine(StringUtil.JoinColumns("rate = ", rateActual, " should be ", rateExpected));
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(rateExpected.MaxDiff(rateActual) < 1e-10);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-6);
        }

        [Fact]
        public void WishartRCRTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> shape = Variable.Observed(2.5).Named("shape");
            int dim = 2;
            PositiveDefiniteMatrix rateRate;
            if (dim == 2)
            {
                rateRate = new PositiveDefiniteMatrix(new double[,] {
          { 2, 1 }, { 1, 3 }
        });
            }
            else
            {
                rateRate = new PositiveDefiniteMatrix(new double[,] { { 2 } });
            }
            var ratePrior = Wishart.FromShapeAndRate(3, rateRate);
            var rate = Variable<PositiveDefiniteMatrix>.Random(ratePrior).Named("rate");
            var x = Variable.WishartFromShapeAndRate(shape, rate).Named("x");
            PositiveDefiniteMatrix xRate;
            if (dim == 2)
            {
                xRate = new PositiveDefiniteMatrix(new double[,] {
          { 5, 2 }, { 2, 7 }
        });
            }
            else
            {
                xRate = new PositiveDefiniteMatrix(new double[,] { { 5 } });
            }
            Wishart xPrior = Wishart.FromShapeAndRate(4, xRate);
            Variable.ConstrainEqualRandom(x, xPrior);
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            //engine.Algorithm = new VariationalMessagePassing();
            engine.Compiler.GivePriorityTo(typeof(WishartFromShapeAndRateOp));
            var rateActual = engine.Infer<Wishart>(rate);
            var xActual = engine.Infer<Wishart>(x);
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            var rateExpected = Wishart.FromShapeAndRate(5.5, xRate + rateRate);
            var xExpected = rateExpected;
            double evExpected = -9.04651537762959;
            if (dim == 1)
            {
                rateExpected = Wishart.FromShapeAndScale(5.048, new PositiveDefiniteMatrix(new double[,] { { 0.3949 } }));
                xExpected = Wishart.FromShapeAndScale(5.04, new PositiveDefiniteMatrix(new double[,] { { 0.1584 } }));
                evExpected = -1.06;
            }
            else if (dim == 2)
            {
                rateExpected = Wishart.FromShapeAndScale(5, new PositiveDefiniteMatrix(new double[,] { { 0.4578, -0.1131 }, { -0.1131, 0.3431 } }));
                xExpected = Wishart.FromShapeAndScale(4.5, new PositiveDefiniteMatrix(new double[,] { { 0.1626, -0.02872 }, { -0.02872, 0.1339 } }));
                evExpected = -3.26;
            }

            if (false)
            {
                // importance sampling
                double totalWeight = 0;
                int numIter = 100000;
                WishartEstimator estRate = new WishartEstimator(dim);
                WishartEstimator estX = new WishartEstimator(dim);
                for (int iter = 0; iter < numIter; iter++)
                {
                    var rateSample = ratePrior.Sample();
                    var xDist = Wishart.FromShapeAndRate(shape.ObservedValue, rateSample);
                    double logWeight = xDist.GetLogAverageOf(xPrior);
                    double weight = System.Math.Exp(logWeight);
                    estRate.Add(rateSample, weight);
                    var xPost = xDist * xPrior;
                    estX.Add(xPost, weight);
                    totalWeight += weight;
                }
                rateExpected = estRate.GetDistribution(new Wishart(dim));
                xExpected = estX.GetDistribution(new Wishart(dim));
                evExpected = System.Math.Log(totalWeight / numIter);
            }
            Console.WriteLine(StringUtil.JoinColumns("rate = ", rateExpected));
            Console.WriteLine(StringUtil.JoinColumns("x = ", xExpected));
            Console.WriteLine(engine.Algorithm.ShortName);
            Console.WriteLine(StringUtil.JoinColumns("rate = ", rateActual));
            Console.WriteLine(StringUtil.JoinColumns("x = ", xActual));
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(rateExpected.MaxDiff(rateActual) < 2e-1);
            Assert.True(xExpected.MaxDiff(xActual) < 3e-1);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-2);
        }

        [Fact]
        public void GammaRatioCRRTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> shape = Variable.Observed(2.5).Named("shape");
            Gamma ratePrior = Gamma.FromShapeAndRate(3, 4);
            Variable<double> rate = Variable<double>.Random(ratePrior).Named("rate");
            Variable<double> y = Variable.GammaFromShapeAndRate(shape, 1).Named("y");
            Variable<double> x = (y / rate).Named("x");
            x.ObservedValue = 1;
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            Gamma rateActual = engine.Infer<Gamma>(rate);
            Gamma yActual = engine.Infer<Gamma>(y);
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Gamma rateExpected = new Gamma(5.5, 0.2);
            Gamma yExpected = new Gamma(5.5, 0.2);
            double evExpected = -1.71304151844203;

            if (false)
            {
                // importance sampling
                double rateMean = 0;
                double rateMeanLog = 0;
                double totalWeight = 0;
                Gamma yPrior = Gamma.FromShapeAndRate(shape.ObservedValue, 1);
                GammaEstimator rateEst = new GammaEstimator();
                GammaEstimator yEst = new GammaEstimator();
                int numIter = 1000000;
                for (int iter = 0; iter < numIter; iter++)
                {
                    double rateSample = ratePrior.Sample();
                    double logWeight = Gamma.FromShapeAndRate(shape.ObservedValue, rateSample).GetLogProb(x.ObservedValue);
                    double weight = System.Math.Exp(logWeight);
                    totalWeight += weight;
                    rateEst.Add(rateSample, weight);
                    rateMean += rateSample * weight;
                    rateMeanLog += weight * System.Math.Log(rateSample);
                    // f(y) p(y) delta(x - y/r) p(r) dy
                    // = f(y'r) p(y=y'r) delta(x - y') p(r) r dy'
                    // = f(xr) p(y=xr) p(r) r 
                    //double ySample = rateSample * x.ObservedValue;
                    //double weight2 = Math.Exp(yPrior.GetLogProb(ySample))*rateSample;
                    double ySample = yPrior.Sample();
                    double weight2 = System.Math.Exp(GammaRatioOp.RatioAverageConditional(ySample, ratePrior).GetLogProb(x.ObservedValue));
                    yEst.Add(ySample, weight2);
                }
                rateMean /= totalWeight;
                rateMeanLog /= totalWeight;
                rateExpected = Gamma.FromMeanAndMeanLog(rateMean, rateMeanLog);
                evExpected = System.Math.Log(totalWeight / numIter);
                yExpected = yEst.GetDistribution(new Gamma());
            }
            Console.WriteLine("rate = {0} should be {1}", rateActual, rateExpected);
            Console.WriteLine("y = {0} should be {1}", yActual, yExpected);
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(rateExpected.MaxDiff(rateActual) < 1e-10);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-6);
        }

        [Fact]
        public void GammaProductCRRTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> shape = Variable.Observed(2.5).Named("shape");
            Gamma scalePrior = Gamma.FromShapeAndRate(3, 4);
            Variable<double> scale = Variable<double>.Random(scalePrior).Named("scale");
            Variable<double> y = Variable.GammaFromShapeAndRate(shape, 1).Named("y");
            Variable<double> x = (y * scale).Named("x");
            x.ObservedValue = 1;
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            Gamma scaleActual = engine.Infer<Gamma>(scale);
            Gamma yActual = engine.Infer<Gamma>(y);
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Gamma scaleExpected = new Gamma(4.168, 0.15);
            Gamma yExpected = new Gamma(3.999, 0.5001);
            double evExpected = -0.940203050521001;

            if (false)
            {
                // importance sampling
                double totalWeight = 0;
                Gamma yPrior = Gamma.FromShapeAndRate(shape.ObservedValue, 1);
                GammaEstimator scaleEst = new GammaEstimator();
                GammaEstimator yEst = new GammaEstimator();
                int numIter = 1000000;
                for (int iter = 0; iter < numIter; iter++)
                {
                    double scaleSample = scalePrior.Sample();
                    double logWeight = Gamma.FromShapeAndScale(shape.ObservedValue, scaleSample).GetLogProb(x.ObservedValue);
                    double weight = System.Math.Exp(logWeight);
                    totalWeight += weight;
                    scaleEst.Add(scaleSample, weight);
                    double ySample = yPrior.Sample();
                    double weight2 = System.Math.Exp(GammaProductOp.ProductAverageConditional(ySample, scalePrior).GetLogProb(x.ObservedValue));
                    yEst.Add(ySample, weight2);
                }
                scaleExpected = scaleEst.GetDistribution(new Gamma());
                evExpected = System.Math.Log(totalWeight / numIter);
                yExpected = yEst.GetDistribution(new Gamma());
            }
            Console.WriteLine("scale = {0} should be {1}", scaleActual, scaleExpected);
            Console.WriteLine("y = {0} should be {1}", yActual, yExpected);
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(scaleExpected.MaxDiff(scaleActual) < 2e-2);
            Assert.True(yExpected.MaxDiff(yActual) < 0.5);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-1);
        }

        [Fact]
        [Trait("Category", "ModifiesGlobals")]
        public void GammaProductRRRTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<Gamma> bPriorVar = Variable.Observed(default(Gamma)).Named("bPrior");
            Variable<double> b = Variable<double>.Random(bPriorVar).Named("b");
            Variable<Gamma> aPriorVar = Variable.Observed(default(Gamma)).Named("aPrior");
            Variable<double> a = Variable<double>.Random(aPriorVar).Named("a");
            Variable<double> product = (a * b).Named("product");
            Variable<Gamma> productPriorVar = Variable.Observed(default(Gamma)).Named("productPrior");
            Variable.ConstrainEqualRandom(product, productPriorVar);
            block.CloseBlock();
            InferenceEngine engine = new InferenceEngine();

            var groundTruthArray = new[]
            {
                ((Gamma.FromShapeAndRate(1, 2), Gamma.FromShapeAndRate(10, 10), Gamma.FromShapeAndRate(101, double.MaxValue)),
                 (Gamma.FromShapeAndRate(7.7197672020942445, 1.0990849072459638E+307), Gamma.FromShapeAndRate(8.9943945066991926, 9.9943035391783432), Gamma.PointMass(5.6185210110227856E-307), 0.79888352306712)),
                ((Gamma.FromShapeAndRate(1, 2), Gamma.FromShapeAndRate(10, 10), Gamma.FromShapeAndRate(101, double.PositiveInfinity)),
                 (Gamma.PointMass(0), Gamma.PointMass(0), Gamma.FromShapeAndRate(101, double.PositiveInfinity), double.NegativeInfinity)),
                ((Gamma.FromShapeAndRate(1, 1), Gamma.FromShapeAndRate(1, 1), Gamma.Uniform()),
                 (Gamma.FromShapeAndRate(1, 1), Gamma.FromShapeAndRate(1, 1), new Gamma(0.3332, 3), 0)),
                ((Gamma.FromShapeAndRate(1, 1), Gamma.FromShapeAndRate(1, 1), Gamma.FromShapeAndRate(30, 1)),
                 (Gamma.FromShapeAndRate(9.31900740051435, 1.7964933711493432), Gamma.FromShapeAndRate(9.2434278607621678, 1.7794585918328409), Gamma.FromShapeAndRate(26.815751741761741, 1.0848182432245914), -10.6738675986344)),
                ((Gamma.FromShapeAndRate(3, 4), Gamma.FromShapeAndRate(2.5, 1), Gamma.FromShapeAndRate(5, 6)),
                 (new Gamma(3.335, 0.1678), new Gamma(3.021, 0.5778), new Gamma(5.619, 0.1411), -0.919181055678219)),
                ((Gamma.FromShapeAndRate(3, 4), Gamma.FromShapeAndRate(2.5, 1), Gamma.Uniform()),
                 (Gamma.FromShapeAndRate(3, 4), Gamma.FromShapeAndRate(2.5, 1), new Gamma(1.154, 1.625), 0.0)),
            };

            using (TestUtils.TemporarilyAllowGammaImproperProducts)
            {
                foreach (var groundTruth in groundTruthArray)
                {
                    var (bPrior, aPrior, productPrior) = groundTruth.Item1;
                    var (bExpected, aExpected, productExpected, evExpected) = groundTruth.Item2;
                    bPriorVar.ObservedValue = bPrior;
                    aPriorVar.ObservedValue = aPrior;
                    productPriorVar.ObservedValue = productPrior;

                    Gamma bActual = engine.Infer<Gamma>(b);
                    Gamma aActual = engine.Infer<Gamma>(a);
                    Gamma productActual = engine.Infer<Gamma>(product);
                    double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;

                    if (false)
                    {
                        // importance sampling
                        double totalWeight = 0;
                        GammaEstimator bEst = new GammaEstimator();
                        GammaEstimator aEst = new GammaEstimator();
                        GammaEstimator productEst = new GammaEstimator();
                        int numIter = 10_000_000;
                        double bScale = 1;
                        double aScale = 1;
                        double logBScale = System.Math.Log(bScale);
                        double logAScale = System.Math.Log(aScale);
                        MeanVarianceAccumulator mvaLogA = new MeanVarianceAccumulator();
                        MeanVarianceAccumulator mvaLogB = new MeanVarianceAccumulator();
                        for (int iter = 0; iter < numIter; iter++)
                        {
                            if (iter % 1_000_000 == 0) Trace.WriteLine($"iter = {iter}");
                            double logWeight = 0;
                            double bSample = bPrior.Sample();
                            double aSample = aPrior.Sample();
                            if (bScale != 1)
                            {
                                logWeight -= bPrior.GetLogProb(bSample) - logBScale;
                                bSample *= bScale;
                                logWeight += bPrior.GetLogProb(bSample);
                            }
                            if (aScale != 1)
                            {
                                logWeight -= aPrior.GetLogProb(aSample) - logAScale;
                                aSample *= aScale;
                                logWeight += aPrior.GetLogProb(aSample);
                            }
                            double productSample = aSample * bSample;
                            logWeight += productPrior.GetLogProb(productSample);
                            double weight = System.Math.Exp(logWeight);
                            totalWeight += weight;
                            bEst.Add(bSample, weight);
                            aEst.Add(aSample, weight);
                            productEst.Add(productSample, weight);
                            mvaLogA.Add(System.Math.Log(aSample), weight);
                            mvaLogB.Add(System.Math.Log(bSample), weight);
                        }
                        Trace.WriteLine($"{nameof(totalWeight)} = {totalWeight}");
                        if (totalWeight > 0)
                        {
                            bExpected = bEst.GetDistribution(new Gamma());
                            if (bExpected.IsPointMass) bExpected = Gamma.FromMeanAndMeanLog(bEst.mva.Mean, mvaLogB.Mean);
                            evExpected = System.Math.Log(totalWeight / numIter);
                            aExpected = aEst.GetDistribution(new Gamma());
                            if (aExpected.IsPointMass) aExpected = Gamma.FromMeanAndMeanLog(aEst.mva.Mean, mvaLogA.Mean);
                            productExpected = productEst.GetDistribution(new Gamma());
                            Trace.WriteLine($"{Quoter.Quote(bExpected)}, {Quoter.Quote(aExpected)}, {Quoter.Quote(productExpected)}, {evExpected}");
                        }
                    }
                    double bError = MomentDiff(bExpected, bActual);
                    double aError = MomentDiff(aExpected, aActual);
                    double productError = MomentDiff(productExpected, productActual);
                    double evError = MMath.AbsDiff(evExpected, evActual, 1e-6);
                    bool trace = false;
                    if (trace)
                    {
                        Trace.WriteLine($"b = {bActual} should be {bExpected}, error = {bError}");
                        Trace.WriteLine($"a = {aActual}[variance={aActual.GetVariance()}] should be {aExpected}[variance={aExpected.GetVariance()}], error = {aError}");
                        Trace.WriteLine($"product = {productActual} should be {productExpected}, error = {productError}");
                        Trace.WriteLine($"evidence = {evActual} should be {evExpected}, error = {evError}");
                    }
                    Assert.True(bError < 3e-2);
                    Assert.True(aError < 1);
                    Assert.True(productError < 3e-2);
                    Assert.True(evError < 5e-2);
                }
            }
        }

        [Fact]
        [Trait("Category", "ModifiesGlobals")]
        public void GammaPowerProductRRRTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<GammaPower> bPriorVar = Variable.Observed(default(GammaPower)).Named("bPrior");
            Variable<double> b = Variable<double>.Random(bPriorVar).Named("b");
            Variable<GammaPower> aPriorVar = Variable.Observed(default(GammaPower)).Named("aPrior");
            Variable<double> a = Variable<double>.Random(aPriorVar).Named("a");
            Variable<double> product = (a * b).Named("product");
            Variable<GammaPower> productPriorVar = Variable.Observed(default(GammaPower)).Named("productPrior");
            Variable.ConstrainEqualRandom(product, productPriorVar);
            block.CloseBlock();
            InferenceEngine engine = new InferenceEngine();

            // TODO: use test cases from GammaProductRRRTest instead of duplicating them
            var groundTruthArray = new[]
            {
                ((GammaPower.FromShapeAndRate(0.83228652924877289, 0.31928405884349487, -1), GammaPower.FromShapeAndRate(1.7184321234630087, 0.709692740551586, -1), GammaPower.FromShapeAndRate(491, 1583.0722891566263, -1)),
                 (GammaPower.FromShapeAndRate(3.1727695744145481, 10.454478169320565, -1.0), GammaPower.FromShapeAndRate(2.469020042117986, 2.5421356314915293, -1.0), GammaPower.FromShapeAndRate(495.57371802470414, 1592.4685605878328, -1.0), -3.57744782716672)),
                ((GammaPower.FromShapeAndRate(1, 2, 1), GammaPower.FromShapeAndRate(10, 10, 1), GammaPower.FromShapeAndRate(101, double.MaxValue, 1)),
                 (GammaPower.FromShapeAndRate(7.7197672020942445, 1.0990849072459638E+307, 1), GammaPower.FromShapeAndRate(8.9943945066991926, 9.9943035391783432, 1), GammaPower.PointMass(5.6185210110227856E-307, 1), 0.79888352306712)),
                ((GammaPower.FromShapeAndRate(1, 2, 1), GammaPower.FromShapeAndRate(10, 10, 1), GammaPower.FromShapeAndRate(101, double.PositiveInfinity, 1)),
                 (GammaPower.PointMass(0, 1.0), GammaPower.PointMass(0, 1.0), GammaPower.FromShapeAndRate(101, double.PositiveInfinity, 1), double.NegativeInfinity)),
                ((GammaPower.FromShapeAndRate(2.25, 0.625, -1), GammaPower.FromShapeAndRate(100000002, 100000001, -1), GammaPower.PointMass(5, -1)),
                 (GammaPower.FromShapeAndRate(99999999.000000119, 19999999.375000019, -1.0), GammaPower.FromShapeAndRate(100000000.0, 100000001.125, -1.0), GammaPower.PointMass(5, -1), -6.5380532346178)),
                ((GammaPower.FromShapeAndRate(2.25, 0.625, -1), GammaPower.FromShapeAndRate(100000002, 100000001, -1), GammaPower.PointMass(0, -1)),
                 (GammaPower.PointMass(0, -1.0), GammaPower.PointMass(0, -1.0), GammaPower.PointMass(0, -1), double.NegativeInfinity)),
                ((GammaPower.FromShapeAndRate(1, 1, 1), GammaPower.FromShapeAndRate(1, 1, 1), GammaPower.Uniform(1)),
                 (GammaPower.FromShapeAndRate(1, 1, 1), GammaPower.FromShapeAndRate(1, 1, 1), new GammaPower(0.3332, 3, 1), 0)),
                ((GammaPower.FromShapeAndRate(1, 1, 1), GammaPower.FromShapeAndRate(1, 1, 1), GammaPower.FromShapeAndRate(30, 1, 1)),
                 (GammaPower.FromShapeAndRate(9.31900740051435, 1.7964933711493432, 1.0), GammaPower.FromShapeAndRate(9.2434278607621678, 1.7794585918328409, 1.0), GammaPower.FromShapeAndRate(26.815751741761741, 1.0848182432245914, 1.0), -10.6738675986344)),
                ((GammaPower.FromShapeAndRate(3, 1, -1), GammaPower.FromShapeAndRate(4, 1, -1), GammaPower.Uniform(-1)),
                 (GammaPower.FromShapeAndRate(3, 1, -1), GammaPower.FromShapeAndRate(4, 1, -1), GammaPower.FromShapeAndRate(2.508893738311099, 0.25145071304806094, -1.0), 0)),
                ((GammaPower.FromShapeAndRate(1, 1, -1), GammaPower.FromShapeAndRate(1, 1, -1), GammaPower.Uniform(-1)),
                 (GammaPower.FromShapeAndRate(1, 1, -1), GammaPower.FromShapeAndRate(1, 1, -1), new GammaPower(0.3332, 3, -1), 0)),
                ((GammaPower.FromShapeAndRate(1, 1, -1), GammaPower.FromShapeAndRate(1, 1, -1), GammaPower.FromShapeAndRate(30, 1, -1)),
                 (GammaPower.FromShapeAndRate(11.057594449558747, 2.0731054100295871, -1.0), GammaPower.FromShapeAndRate(11.213079710986863, 2.1031756133562678, -1.0), GammaPower.FromShapeAndRate(28.815751741667615, 1.0848182432207041, -1.0), -4.22210057295786)),
                ((GammaPower.FromShapeAndRate(1, 1, 2), GammaPower.FromShapeAndRate(1, 1, 2), GammaPower.Uniform(2)),
                 (GammaPower.FromShapeAndRate(1, 1, 2), GammaPower.FromShapeAndRate(1, 1, 2), GammaPower.FromShapeAndRate(0.16538410345846666, 0.219449497990138, 2.0), 0)),
                ((GammaPower.FromShapeAndRate(1, 1, 2), GammaPower.FromShapeAndRate(1, 1, 2), GammaPower.FromShapeAndRate(30, 1, 2)),
                 (GammaPower.FromShapeAndRate(8.72865708291647, 1.71734403810018, 2.0), GammaPower.FromShapeAndRate(8.5298603954575931, 1.6767026737490067, 2.0), GammaPower.FromShapeAndRate(25.831187278202215, 1.0852321896648485, 2.0), -14.5369973268808)),
            };

            using (TestUtils.TemporarilyAllowGammaImproperProducts)
            {
                foreach (var groundTruth in groundTruthArray)
                {
                    var (bPrior, aPrior, productPrior) = groundTruth.Item1;
                    var (bExpected, aExpected, productExpected, evExpected) = groundTruth.Item2;
                    bPriorVar.ObservedValue = bPrior;
                    aPriorVar.ObservedValue = aPrior;
                    productPriorVar.ObservedValue = productPrior;

                    GammaPower bActual = engine.Infer<GammaPower>(b);
                    GammaPower aActual = engine.Infer<GammaPower>(a);
                    GammaPower productActual = engine.Infer<GammaPower>(product);
                    double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;

                    if (false)
                    {
                        // importance sampling
                        Rand.Restart(0);
                        double totalWeight = 0;
                        GammaPowerEstimator bEstimator = new GammaPowerEstimator(bPrior.Power);
                        GammaPowerEstimator aEstimator = new GammaPowerEstimator(aPrior.Power);
                        GammaPowerEstimator productEstimator = new GammaPowerEstimator(productPrior.Power);
                        MeanVarianceAccumulator bMva = new MeanVarianceAccumulator();
                        MeanVarianceAccumulator aMva = new MeanVarianceAccumulator();
                        MeanVarianceAccumulator productMva = new MeanVarianceAccumulator();
                        int numIter = 10_000_000;
                        for (int iter = 0; iter < numIter; iter++)
                        {
                            if (iter % 1000000 == 0) Trace.WriteLine($"iter = {iter}");
                            double bSample = bPrior.Sample();
                            double aSample = aPrior.Sample();
                            if (productPrior.Rate > 1e100)
                            {
                                bSample = 0;
                                aSample = 0;
                            }
                            double productSample = aSample * bSample;
                            double logWeight = productPrior.GetLogProb(productSample);
                            double weight = System.Math.Exp(logWeight);
                            totalWeight += weight;
                            bEstimator.Add(bSample, weight);
                            aEstimator.Add(aSample, weight);
                            productEstimator.Add(productSample, weight);
                            bMva.Add(bSample, weight);
                            aMva.Add(aSample, weight);
                            productMva.Add(productSample, weight);
                        }
                        Trace.WriteLine($"totalWeight = {totalWeight}");
                        if (totalWeight > 0)
                        {
                            evExpected = System.Math.Log(totalWeight / numIter);
                            bExpected = bEstimator.GetDistribution(bPrior);
                            aExpected = aEstimator.GetDistribution(aPrior);
                            productExpected = productEstimator.GetDistribution(productPrior);
                            bExpected = GammaPower.FromMeanAndVariance(bMva.Mean, bMva.Variance, bPrior.Power);
                            aExpected = GammaPower.FromMeanAndVariance(aMva.Mean, aMva.Variance, aPrior.Power);
                            productExpected = GammaPower.FromMeanAndVariance(productMva.Mean, productMva.Variance, productPrior.Power);
                            Trace.WriteLine($"{Quoter.Quote(bExpected)}, {Quoter.Quote(aExpected)}, {Quoter.Quote(productExpected)}, {evExpected}");
                        }
                    }
                    double bError = MomentDiff(bExpected, bActual);
                    double aError = MomentDiff(aExpected, aActual);
                    double productError = MomentDiff(productExpected, productActual);
                    double evError = MMath.AbsDiff(evExpected, evActual, 1e-6);
                    bool trace = false;
                    if (trace)
                    {
                        Trace.WriteLine($"b = {bActual} should be {bExpected}, error = {bError}");
                        Trace.WriteLine($"a = {aActual}[variance={aActual.GetVariance()}] should be {aExpected}[variance={aExpected.GetVariance()}], error = {aError}");
                        Trace.WriteLine($"product = {productActual} should be {productExpected}, error = {productError}");
                        Trace.WriteLine($"evidence = {evActual} should be {evExpected}, error = {evError}");
                    }
                    Assert.True(bError < 10);
                    Assert.True(aError < 2.1);
                    Assert.True(productError < 9);
                    Assert.True(evError < 3e-2);
                }
            }
        }

        internal static void TestLogEvidence()
        {
            LogEvidenceScale(new GammaPower(100, 5.0 / 100, -1), new GammaPower(100, 2.0 / 100, -1), new GammaPower(100, 3.0 / 100, -1), 0.2);
        }

        internal static void LogEvidenceShift(GammaPower sum, GammaPower a, GammaPower b)
        {
            double logz100 = PlusGammaOp.LogAverageFactor(GammaPower.FromShapeAndRate(sum.Shape - 1, sum.Rate, sum.Power), GammaPower.FromShapeAndRate(a.Shape, a.Rate, a.Power), GammaPower.FromShapeAndRate(b.Shape, b.Rate, b.Power));
            double logz010 = PlusGammaOp.LogAverageFactor(GammaPower.FromShapeAndRate(sum.Shape, sum.Rate, sum.Power), GammaPower.FromShapeAndRate(a.Shape - 1, a.Rate, a.Power), GammaPower.FromShapeAndRate(b.Shape, b.Rate, b.Power));
            double logz001 = PlusGammaOp.LogAverageFactor(GammaPower.FromShapeAndRate(sum.Shape, sum.Rate, sum.Power), GammaPower.FromShapeAndRate(a.Shape, a.Rate, a.Power), GammaPower.FromShapeAndRate(b.Shape - 1, b.Rate, b.Power));
            double lhs = logz100 + System.Math.Log(sum.Rate / (sum.Shape - 1));
            double rhs1 = logz010 + System.Math.Log(a.Rate / (a.Shape - 1));
            double rhs2 = logz001 + System.Math.Log(b.Rate / (b.Shape - 1));
            Trace.WriteLine($"lhs = {lhs} rhs = {MMath.LogSumExp(rhs1, rhs2)}");
        }

        internal static void LogEvidenceScale(GammaPower sum, GammaPower a, GammaPower b, double scale)
        {
            double logZ = LogEvidenceBrute(sum, a, b);
            double logZ2 = System.Math.Log(scale) + LogEvidenceBrute(GammaPower.FromShapeAndRate(sum.Shape, scale * sum.Rate, sum.Power), GammaPower.FromShapeAndRate(a.Shape, scale * a.Rate, a.Power), GammaPower.FromShapeAndRate(b.Shape, scale * b.Rate, b.Power));
            Trace.WriteLine($"logZ = {logZ} {logZ2}");
        }

        internal static double LogEvidenceBrute(GammaPower sumPrior, GammaPower aPrior, GammaPower bPrior)
        {
            bool trace = false;
            double totalWeight = 0;
            int numIter = 1000000;
            for (int iter = 0; iter < numIter; iter++)
            {
                if (trace && iter % 1000000 == 0) Trace.WriteLine($"iter = {iter}");
                double bSample = bPrior.Sample();
                double aSample = aPrior.Sample();
                if (sumPrior.Rate > 1e100)
                {
                    bSample = 0;
                    aSample = 0;
                }
                double sumSample = aSample + bSample;
                double logWeight = sumPrior.GetLogProb(sumSample);
                double weight = System.Math.Exp(logWeight);
                totalWeight += weight;
            }
            if (trace) Trace.WriteLine($"totalWeight = {totalWeight}");
            return System.Math.Log(totalWeight / numIter);
        }

        internal static double LogEvidenceIncrementBShape(GammaPower sum, GammaPower a, GammaPower b)
        {
            const double threshold = 0;
            if (b.Shape > threshold)
            {
                //return PlusGammaOp.LogAverageFactor(sum, a, b);
                return LogEvidenceBrute(sum, a, b);
            }
            double logz100 = LogEvidenceIncrementBShape(GammaPower.FromShapeAndRate(sum.Shape - 1, sum.Rate, sum.Power), GammaPower.FromShapeAndRate(a.Shape, a.Rate, a.Power), GammaPower.FromShapeAndRate(b.Shape + 1, b.Rate, b.Power));
            double logz010 = LogEvidenceIncrementBShape(GammaPower.FromShapeAndRate(sum.Shape, sum.Rate, sum.Power), GammaPower.FromShapeAndRate(a.Shape - 1, a.Rate, a.Power), GammaPower.FromShapeAndRate(b.Shape + 1, b.Rate, b.Power));
            double lhs = logz100 + System.Math.Log(sum.Rate / (sum.Shape - 1));
            double rhs1 = logz010 + System.Math.Log(a.Rate / (a.Shape - 1));
            double rhs2 = System.Math.Log(b.Rate / b.Shape);
            return MMath.LogDifferenceOfExp(lhs, rhs1) - rhs2;
        }

        internal void Test()
        {
            var new_position = 1;
            var p_var = new PositiveDefiniteMatrix(new double[,] { { 1 } });
            var position_prior = Variable.New<VectorGaussian>().Named("position_prior").Attrib(new DoNotInfer());
            position_prior.ObservedValue = VectorGaussian.PointMass(new_position);
            Variable<Vector> p_mean = Variable<Vector>.Random<VectorGaussian>(position_prior);
            var position = Variable.VectorGaussianFromMeanAndVariance(p_mean, p_var).Named("position");

            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(position));
        }

        [Fact]
        public void GammaPowerSumRRRTest()
        {
            //Assert.True(PlusGammaOp.AAverageConditional(GammaPower.FromShapeAndRate(299, 2135, -1), GammaPower.FromShapeAndRate(2.01, 10, -1), GammaPower.FromShapeAndRate(12, 22, -1), GammaPower.Uniform(-1)).Shape > 2);
            //Assert.True(PlusGammaOp.AAverageConditional(GammaPower.Uniform(-1), GammaPower.FromShapeAndRate(2.0095439611576689, 43.241375394505766, -1), GammaPower.FromShapeAndRate(12, 11, -1), GammaPower.Uniform(-1)).IsUniform());
            //Assert.False(double.IsNaN(PlusGammaOp.BAverageConditional(new GammaPower(287, 0.002132, -1), new GammaPower(1.943, 1.714, -1), new GammaPower(12, 0.09091, -1), GammaPower.Uniform(-1)).Shape));

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<GammaPower> bPriorVar = Variable.Observed(default(GammaPower)).Named("bPrior");
            Variable<double> b = Variable<double>.Random(bPriorVar).Named("b");
            Variable<GammaPower> aPriorVar = Variable.Observed(default(GammaPower)).Named("aPrior");
            Variable<double> a = Variable<double>.Random(aPriorVar).Named("a");
            Variable<double> sum = (a + b).Named("sum");
            Variable<GammaPower> sumPriorVar = Variable.Observed(default(GammaPower)).Named("sumPrior");
            Variable.ConstrainEqualRandom(sum, sumPriorVar);
            block.CloseBlock();
            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;

            var groundTruthArray = new[]
            {
                ((GammaPower.FromShapeAndRate(1001.0936734710671, 36.21011524652728, 1), GammaPower.FromShapeAndRate(0.38162412924733724, 0.32671422227025443, 1), GammaPower.FromShapeAndRate(252.56213972271735, 9.512615685427761, 1)),
                 (GammaPower.FromShapeAndRate(1222.7684713384558, 44.756548562749032, 1.0), GammaPower.FromShapeAndRate(0.47701962589739461, 1.2067745679220236, 1.0), GammaPower.FromShapeAndRate(943.45467449193779, 34.040414940129914, 1.0), -2.0988360229430723)),
                ((GammaPower.FromShapeAndRate(1103.2845932433772, 97.372616083185363, 1), GammaPower.Uniform(1), GammaPower.FromShapeAndRate(4544, 408.2010204790879, 1)),
                 (GammaPower.FromShapeAndRate(2675.5634867292201, 243.72445043333758, 1.0), GammaPower.FromShapeAndRate(1.5517181262728859, 6.5283566500919736, 1.0), GammaPower.FromShapeAndRate(5321.2880411411688, 474.45795019477413, 1.0), 0)),
                ((GammaPower.FromShapeAndRate(1, 2, 1), GammaPower.FromShapeAndRate(10, 10, 1), GammaPower.FromShapeAndRate(101, double.MaxValue, 1)),
                 (GammaPower.PointMass(0, 1.0), GammaPower.PointMass(0, 1), GammaPower.PointMass(5.6183114927306835E-307, 1), -3824)),
                ((GammaPower.FromShapeAndRate(0.83228652924877289, 0.31928405884349487, -1), GammaPower.FromShapeAndRate(1.7184321234630087, 0.709692740551586, -1), GammaPower.FromShapeAndRate(491, 1583.0722891566263, -1)),
                 (GammaPower.FromShapeAndRate(5.6062357530254419, 8.7330355320375, -1.0), GammaPower.FromShapeAndRate(3.7704064465114597, 3.6618414405426956, -1.0), GammaPower.FromShapeAndRate(493.79911104976264, 1585.67297686381, -1.0), -2.62514943790608)),
                ((new GammaPower(12, 0.09091, -1), new GammaPower(1.943, 1.714, -1), new GammaPower(287, 0.002132, -1)),
                 (GammaPower.FromShapeAndRate(23.445316648707465, 25.094880573396285, -1.0), GammaPower.FromShapeAndRate(6.291922598211336, 2.6711637040924909, -1.0), GammaPower.FromShapeAndRate(297.59289156399706, 481.31323394825631, -1.0), -0.517002984399292)),
                ((GammaPower.FromShapeAndRate(12, 22, -1), GammaPower.FromShapeAndRate(2.01, 10, -1), GammaPower.FromShapeAndRate(299, 2135, -1)),
                 (GammaPower.FromShapeAndRate(12.4019151884055, 23.487535138993064, -1.0), GammaPower.FromShapeAndRate(47.605465737960976, 236.41203334327037, -1.0), GammaPower.FromShapeAndRate(303.94717779788243, 2160.7976040127091, -1.0), -2.26178042225837)),
                ((GammaPower.FromShapeAndRate(1, 1, -1), GammaPower.FromShapeAndRate(1, 1, -1), GammaPower.FromShapeAndRate(30, 1, -1)),
                 (GammaPower.FromShapeAndRate(28.334615735207226, 2.0631059498231852, -1.0), GammaPower.FromShapeAndRate(28.023157852553162, 2.037420901598074, -1.0), GammaPower.FromShapeAndRate(53.15319932389427, 7.8527322627767493, -1.0), -41.9658555081493)),
                 //(GammaPower.FromShapeAndRate(19.904821842480409, 1.5012846571136531, -1.0), GammaPower.FromShapeAndRate(19.929465176191822, 1.5037327360414059, -1.0), GammaPower.FromShapeAndRate(32.566439029580131, 5.009814326700412, -1.0), -38.551827978313334)),
                ((GammaPower.FromShapeAndRate(1, 2, 1), GammaPower.FromShapeAndRate(10, 10, 1), GammaPower.FromShapeAndRate(101, double.PositiveInfinity, 1)),
                 (GammaPower.PointMass(0, 1.0), GammaPower.PointMass(0, 1), GammaPower.FromShapeAndRate(101, double.PositiveInfinity, 1), double.NegativeInfinity)),
                ((GammaPower.FromShapeAndRate(2.25, 0.625, -1), GammaPower.FromShapeAndRate(100000002, 100000001, -1), GammaPower.PointMass(0, -1)),
                 (GammaPower.PointMass(0, -1.0), GammaPower.PointMass(0, -1.0), GammaPower.PointMass(0, -1), double.NegativeInfinity)),
                ((GammaPower.FromShapeAndRate(2.25, 0.625, -1), GammaPower.FromShapeAndRate(100000002, 100000001, -1), GammaPower.PointMass(5, -1)),
                 (GammaPower.FromShapeAndRate(1599999864.8654146, 6399999443.0866585, -1.0), GammaPower.FromShapeAndRate(488689405.117356, 488689405.88170129, -1.0), GammaPower.FromShapeAndRate(double.PositiveInfinity, 5.0, -1.0), -4.80649551611576)),

                ((GammaPower.FromShapeAndRate(1, 1, 1), GammaPower.FromShapeAndRate(1, 1, 1), GammaPower.Uniform(1)),
                 (GammaPower.FromShapeAndRate(1, 1, 1), GammaPower.FromShapeAndRate(1, 1, 1), new GammaPower(2, 1, 1), 0)),
                ((GammaPower.FromShapeAndRate(1, 1, 1), GammaPower.FromShapeAndRate(1, 1, 1), GammaPower.FromShapeAndRate(10, 1, 1)),
                 (GammaPower.FromShapeAndRate(2.2, 0.8, 1), GammaPower.FromShapeAndRate(2.2, 0.8, 1), GammaPower.FromShapeAndRate(11, 2, 1), -5.32133409609914)),
                ((GammaPower.FromShapeAndRate(3, 1, -1), GammaPower.FromShapeAndRate(4, 1, -1), GammaPower.Uniform(-1)),
                 (GammaPower.FromShapeAndRate(3, 1, -1), GammaPower.FromShapeAndRate(4, 1, -1), GammaPower.FromShapeAndRate(4.311275674659143, 2.7596322350392035, -1.0), 0)),
                ((GammaPower.FromShapeAndRate(3, 1, -1), GammaPower.FromShapeAndRate(4, 1, -1), GammaPower.FromShapeAndRate(10, 1, -1)),
                 (new GammaPower(10.17, 0.6812, -1), new GammaPower(10.7, 0.7072, -1), new GammaPower(17.04, 0.2038, -1), -5.80097480415528)),
                ((GammaPower.FromShapeAndRate(2, 1, -1), GammaPower.FromShapeAndRate(2, 1, -1), GammaPower.Uniform(-1)),
                 (GammaPower.FromShapeAndRate(2, 1, -1), GammaPower.FromShapeAndRate(2, 1, -1), GammaPower.FromShapeAndRate(2, 2, -1), 0)),
                ((GammaPower.FromShapeAndRate(1, 1, 2), GammaPower.FromShapeAndRate(1, 1, 2), GammaPower.Uniform(2)),
                 (GammaPower.FromShapeAndRate(1, 1, 2), GammaPower.FromShapeAndRate(1, 1, 2), GammaPower.FromShapeAndRate(1.8739663250181251, 1.1602914853106219, 2.0), 0)),
                ((GammaPower.FromShapeAndRate(1, 1, 2), GammaPower.FromShapeAndRate(1, 1, 2), GammaPower.FromShapeAndRate(30, 1, 2)),
                 (GammaPower.FromShapeAndRate(3.6006762919673134, 0.45026963024042149, 2.0), GammaPower.FromShapeAndRate(4.2874318525802124, 0.45391732702230442, 2.0), GammaPower.FromShapeAndRate(46.19738982803915, 3.3722594889688562, 2.0), -23.47535770480025)),
            };

            bool trace = false;
            double aErrorMax = 0;
            double bErrorMax = 0;
            double sumErrorMax = 0;
            double evErrorMax = 0;

            //using (TestUtils.TemporarilyAllowGammaImproperProducts)
            {
                foreach (var groundTruth in groundTruthArray)
                {
                    var (bPrior, aPrior, sumPrior) = groundTruth.Item1;
                    var (bExpected, aExpected, sumExpected, evExpected) = groundTruth.Item2;
                    bPriorVar.ObservedValue = bPrior;
                    aPriorVar.ObservedValue = aPrior;
                    sumPriorVar.ObservedValue = sumPrior;

                    GammaPower bActual = engine.Infer<GammaPower>(b);
                    GammaPower aActual = engine.Infer<GammaPower>(a);
                    GammaPower sumActual = engine.Infer<GammaPower>(sum);
                    double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;

                    double logZ = LogEvidenceIncrementBShape(sumPrior, aPrior, bPrior);
                    if (trace) Trace.WriteLine($"LogZ = {logZ}");

                    if (false)
                    {
                        // importance sampling
                        Rand.Restart(0);
                        double totalWeight = 0;
                        GammaPowerEstimator bEstimator = new GammaPowerEstimator(bPrior.Power);
                        GammaPowerEstimator aEstimator = new GammaPowerEstimator(aPrior.Power);
                        GammaPowerEstimator sumEstimator = new GammaPowerEstimator(sumPrior.Power);
                        MeanVarianceAccumulator bMva = new MeanVarianceAccumulator();
                        MeanVarianceAccumulator aMva = new MeanVarianceAccumulator();
                        MeanVarianceAccumulator sumMva = new MeanVarianceAccumulator();
                        int numIter = 10_000_000;
                        double tailProbability = 1.0 / numIter;
                        // If sum cannot be more than x, then b cannot be more than x.
                        double sumUpperBound = sumPrior.GetQuantile(1 - tailProbability);
                        double evidenceMultiplier = GammaPowerProbBetween(bPrior, 0, sumUpperBound) * GammaPowerProbBetween(aPrior, 0, sumUpperBound);
                        Trace.WriteLine($"sumUpperBound = {sumUpperBound} evidenceMultiplier = {evidenceMultiplier}");
                        for (int iter = 0; iter < numIter; iter++)
                        {
                            if (iter % 1000000 == 0) Trace.WriteLine($"iter = {iter}");
                            double logWeight = 0;
                            double bSample = Sample(bPrior, 0, sumUpperBound);
                            double aSample = Sample(aPrior, 0, sumUpperBound);
                            double bScale = 1;
                            double aScale = 1;
                            if (bScale != 1)
                            {
                                logWeight -= bPrior.GetLogProb(bSample) - System.Math.Log(bScale);
                                bSample *= bScale;
                                logWeight += bPrior.GetLogProb(bSample);
                            }
                            if (aScale != 1)
                            {
                                logWeight -= aPrior.GetLogProb(aSample) - System.Math.Log(aScale);
                                aSample *= aScale;
                                logWeight += aPrior.GetLogProb(aSample);
                            }
                            double sumSample = aSample + bSample;
                            logWeight += sumPrior.GetLogProb(sumSample);
                            double weight = System.Math.Exp(logWeight);
                            totalWeight += weight;
                            bEstimator.Add(bSample, weight);
                            aEstimator.Add(aSample, weight);
                            sumEstimator.Add(sumSample, weight);
                            bMva.Add(bSample, weight);
                            aMva.Add(aSample, weight);
                            sumMva.Add(sumSample, weight);
                        }
                        Trace.WriteLine($"totalWeight = {totalWeight}");
                        if (totalWeight > 0)
                        {
                            evExpected = System.Math.Log(evidenceMultiplier * totalWeight / numIter);
                            bExpected = bEstimator.GetDistribution(bPrior);
                            aExpected = aEstimator.GetDistribution(aPrior);
                            sumExpected = sumEstimator.GetDistribution(sumPrior);
                            //bExpected = GammaPower.FromMeanAndVariance(bMva.Mean, bMva.Variance, bPrior.Power);
                            //aExpected = GammaPower.FromMeanAndVariance(aMva.Mean, aMva.Variance, aPrior.Power);
                            //sumExpected = GammaPower.FromMeanAndVariance(sumMva.Mean, sumMva.Variance, sumPrior.Power);
                            Trace.WriteLine($"{Quoter.Quote(bExpected)}, {Quoter.Quote(aExpected)}, {Quoter.Quote(sumExpected)}, {evExpected}");
                        }
                    }
                    else if (trace) Trace.WriteLine($"{Quoter.Quote(bActual)}, {Quoter.Quote(aActual)}, {Quoter.Quote(sumActual)}, {evActual}");
                    double bError = MomentDiff(bExpected, bActual);
                    double aError = MomentDiff(aExpected, aActual);
                    double sumError = MomentDiff(sumExpected, sumActual);
                    double evError = MMath.AbsDiff(evExpected, evActual, 1e-6);
                    if (trace)
                    {
                        Trace.WriteLine($"b = {bActual} should be {bExpected}, error = {bError}");
                        Trace.WriteLine($"a = {aActual}[variance={aActual.GetVariance()}] should be {aExpected}[variance={aExpected.GetVariance()}], error = {aError}");
                        Trace.WriteLine($"sum = {sumActual} should be {sumExpected}, error = {sumError}");
                        Trace.WriteLine($"evidence = {evActual} should be {evExpected}, error = {evError}");
                    }
                    Assert.True(bError < 100);
                    Assert.True(aError < 1000);
                    Assert.True(sumError < 1.1);
                    Assert.True(evError < 2);
                    aErrorMax = System.Math.Max(aErrorMax, aError);
                    bErrorMax = System.Math.Max(bErrorMax, bError);
                    sumErrorMax = System.Math.Max(sumErrorMax, sumError);
                    evErrorMax = System.Math.Max(evErrorMax, evError);
                }
            }
            Trace.WriteLine($"A evidence error = {aErrorMax}");
            Trace.WriteLine($"B evidence error = {bErrorMax}");
            Trace.WriteLine($"sum evidence error = {sumErrorMax}");
            Trace.WriteLine($"max evidence error = {evErrorMax}");
        }

        /// <summary>
        /// Computes the probability that a GammaPower sample lands in an interval.
        /// </summary>
        /// <param name="gammaPower"></param>
        /// <param name="lowerBound"></param>
        /// <param name="upperBound"></param>
        /// <returns></returns>
        public static double GammaPowerProbBetween(GammaPower gammaPower, double lowerBound, double upperBound)
        {
            double unpowerLowerBound = System.Math.Pow((gammaPower.Power < 0) ? upperBound : lowerBound, 1 / gammaPower.Power);
            double unpowerUpperBound = System.Math.Pow((gammaPower.Power < 0) ? lowerBound : upperBound, 1 / gammaPower.Power);
            return MMath.GammaProbBetween(gammaPower.Shape, gammaPower.Rate, unpowerLowerBound, unpowerUpperBound);
        }

        /// <summary>
        /// Samples from a truncated GammaPower distribution.
        /// </summary>
        /// <param name="gammaPower"></param>
        /// <param name="lowerBound"></param>
        /// <param name="upperBound"></param>
        /// <returns></returns>
        public static double Sample(GammaPower gammaPower, double lowerBound, double upperBound)
        {
            double unpowerLowerBound = System.Math.Pow((gammaPower.Power < 0) ? upperBound : lowerBound, 1 / gammaPower.Power);
            double unpowerUpperBound = System.Math.Pow((gammaPower.Power < 0) ? lowerBound : upperBound, 1 / gammaPower.Power);
            return System.Math.Pow(new TruncatedGamma(Gamma.FromShapeAndRate(gammaPower.Shape, gammaPower.Rate), unpowerLowerBound, unpowerUpperBound).Sample(), gammaPower.Power);
        }

        const double rel = 1e-8;

        public static double MomentDiff(CanGetMeanAndVarianceOut<double,double> expected, CanGetMeanAndVarianceOut<double, double> actual)
        {
            expected.GetMeanAndVariance(out double meanExpected, out double varianceExpected);
            actual.GetMeanAndVariance(out double meanActual, out double varianceActual);
            return System.Math.Max(MMath.AbsDiff(meanExpected, meanActual, rel), MMath.AbsDiff(varianceExpected, varianceActual, rel));
        }

        [Fact]
        public void GammaCCRTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> shape = Variable.Observed(2.5).Named("shape");
            Gamma ratePrior = Gamma.FromShapeAndRate(3, 4);
            Variable<double> rate = Variable<double>.Random(ratePrior).Named("rate");
            Variable<double> x = Variable.GammaFromShapeAndRate(shape, rate).Named("x");
            x.ObservedValue = 1;
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            Gamma rateActual = engine.Infer<Gamma>(rate);
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Gamma rateExpected = new Gamma(5.5, 0.2);
            double evExpected = -1.71304151844203;

            if (false)
            {
                // importance sampling
                double rateMean = 0;
                double rateMeanLog = 0;
                double totalWeight = 0;
                int numIter = 100000;
                for (int iter = 0; iter < numIter; iter++)
                {
                    double rateSample = ratePrior.Sample();
                    double logWeight = Gamma.FromShapeAndRate(shape.ObservedValue, rateSample).GetLogProb(x.ObservedValue);
                    double weight = System.Math.Exp(logWeight);
                    totalWeight += weight;
                    rateMean += rateSample * weight;
                    rateMeanLog += weight * System.Math.Log(rateSample);
                }
                rateMean /= totalWeight;
                rateMeanLog /= totalWeight;
                rateExpected = Gamma.FromMeanAndMeanLog(rateMean, rateMeanLog);
                evExpected = System.Math.Log(totalWeight / numIter);
            }
            Console.WriteLine("rate = {0} should be {1}", rateActual, rateExpected);
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(rateExpected.MaxDiff(rateActual) < 1e-10);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-6);
        }

        [Fact]
        public void GammaRCRTest()
        {
            GammaRCR(false);
            GammaRCR(true);
        }
        private void GammaRCR(bool difficultCase)
        {
            Gamma xPrior = Gamma.FromShapeAndRate(5, 6);
            Gamma ratePrior = Gamma.FromShapeAndRate(3, 4);
            Variable<double> shape = Variable.Observed(2.5).Named("shape");
            Gamma xExpected = new Gamma(6.298, 0.1455);
            Gamma rateExpected = new Gamma(5.325, 0.2112);
            double evExpected = -1.89509023271149;
            if (difficultCase)
            {
                // these settings induce a very difficult integral over rate
                xPrior = new Gamma(4.5, 387.6);
                ratePrior = new Gamma(3, 1);
                shape.ObservedValue = 3.0;
                xExpected = Gamma.FromShapeAndRate(4.5, 0.018707535303819141);
                rateExpected = Gamma.FromShapeAndRate(3, 24.875503791382645);
                evExpected = -22.5625239057597;
                //GammaFromShapeAndRateOp.ForceProper = false;
            }

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> rate = Variable<double>.Random(ratePrior).Named("rate");
            Variable<double> x = Variable.GammaFromShapeAndRate(shape, rate).Named("x");
            Variable.ConstrainEqualRandom(x, xPrior);
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            Gamma xActual = engine.Infer<Gamma>(x);
            Gamma rateActual = engine.Infer<Gamma>(rate);
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;

            if (false)
            {
                // importance sampling
                double rateMean = 0;
                double rateMeanLog = 0;
                double xMean = 0;
                double xMeanLog = 0;
                double totalWeight = 0;
                int numIter = 100000;
                for (int iter = 0; iter < numIter; iter++)
                {
                    double rateSample = ratePrior.Sample();
                    Gamma xDist = Gamma.FromShapeAndRate(shape.ObservedValue, rateSample);
                    double logWeight = xDist.GetLogAverageOf(xPrior);
                    double weight = System.Math.Exp(logWeight);
                    totalWeight += weight;
                    rateMean += rateSample * weight;
                    rateMeanLog += weight * System.Math.Log(rateSample);
                    Gamma xPost = xDist * xPrior;
                    xMean += xPost.GetMean() * weight;
                    xMeanLog += xPost.GetMeanLog() * weight;
                }
                rateMean /= totalWeight;
                rateMeanLog /= totalWeight;
                rateExpected = Gamma.FromMeanAndMeanLog(rateMean, rateMeanLog);
                xMean /= totalWeight;
                xMeanLog /= totalWeight;
                xExpected = Gamma.FromMeanAndMeanLog(xMean, xMeanLog);
                evExpected = System.Math.Log(totalWeight / numIter);
            }
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Console.WriteLine("rate = {0} should be {1}", rateActual, rateExpected);
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-3);
            Assert.True(rateExpected.MaxDiff(rateActual) < 1e-3);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-6);

            if (true)
            {
                engine.Compiler.GivePriorityTo(typeof(GammaFromShapeAndRateOp_Laplace));
                xActual = engine.Infer<Gamma>(x);
                rateActual = engine.Infer<Gamma>(rate);
                evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
                Console.WriteLine("rate = {0} should be {1}", rateActual, rateExpected);
                Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                Assert.True(xExpected.MaxDiff(xActual) < 1e-1);
                Assert.True(rateExpected.MaxDiff(rateActual) < 0.5);
                Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-2);
            }
        }

        [Fact]
        public void GammaRatioRRRTest()
        {
            GammaRatioRRR(false);
            GammaRatioRRR(true);
        }
        private void GammaRatioRRR(bool difficultCase)
        {
            Gamma xPrior = Gamma.FromShapeAndRate(5, 6);
            Gamma ratePrior = Gamma.FromShapeAndRate(3, 4);
            Variable<double> shape = Variable.Observed(2.5).Named("shape");
            Gamma xExpected = new Gamma(6.298, 0.1455);
            Gamma yExpected = new Gamma(3.298, 0.3036);
            Gamma rateExpected = new Gamma(5.325, 0.2112);
            double evExpected = -1.89509023271149;
            if (difficultCase)
            {
                // these settings induce a very difficult integral over rate
                xPrior = new Gamma(4.5, 387.6);
                ratePrior = new Gamma(3, 1);
                shape.ObservedValue = 3.0;
                xExpected = Gamma.FromShapeAndRate(4.5, 0.018707535303819141);
                yExpected = new Gamma(5.242, 1.161);
                rateExpected = Gamma.FromShapeAndRate(3, 24.875503791382645);
                evExpected = -22.5625239057597;
                //GammaFromShapeAndRateOp.ForceProper = false;
            }

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> rate = Variable<double>.Random(ratePrior).Named("rate");
            Variable<double> y = Variable.GammaFromShapeAndRate(shape, 1).Named("y");
            Variable<double> x = (y / rate).Named("x");
            Variable.ConstrainEqualRandom(x, xPrior);
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            Gamma xActual = engine.Infer<Gamma>(x);
            Gamma yActual = engine.Infer<Gamma>(y);
            Gamma rateActual = engine.Infer<Gamma>(rate);
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;

            if (false)
            {
                // importance sampling
                double rateMean = 0;
                double rateMeanLog = 0;
                GammaEstimator xEst = new GammaEstimator();
                GammaEstimator yEst = new GammaEstimator();
                Gamma yPrior = Gamma.FromShapeAndRate(shape.ObservedValue, 1);
                double totalWeight = 0;
                int numIter = 1000000;
                for (int iter = 0; iter < numIter; iter++)
                {
                    double rateSample = ratePrior.Sample();
                    double ySample = yPrior.Sample();
                    double xSample = ySample / rateSample;
                    double logWeight = xPrior.GetLogProb(xSample);
                    double weight = System.Math.Exp(logWeight);
                    totalWeight += weight;
                    rateMean += rateSample * weight;
                    rateMeanLog += weight * System.Math.Log(rateSample);
                    xEst.Add(xSample, weight);
                    yEst.Add(ySample, weight);
                }
                rateMean /= totalWeight;
                rateMeanLog /= totalWeight;
                rateExpected = Gamma.FromMeanAndMeanLog(rateMean, rateMeanLog);
                xExpected = xEst.GetDistribution(new Gamma());
                yExpected = yEst.GetDistribution(new Gamma());
                evExpected = System.Math.Log(totalWeight / numIter);
            }
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Console.WriteLine("y = {0} should be {1}", yActual, yExpected);
            Console.WriteLine("rate = {0} should be {1}", rateActual, rateExpected);
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            if (difficultCase)
            {
                Assert.True(xExpected.MaxDiff(xActual) < 1e-1);
                Assert.True(yExpected.MaxDiff(yActual) < 0.5);
                Assert.True(rateExpected.MaxDiff(rateActual) < 0.5);
                Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-2);
            }
            else
            {
                Assert.True(xExpected.MaxDiff(xActual) < 1e-3);
                Assert.True(yExpected.MaxDiff(yActual) < 3e-2);
                Assert.True(rateExpected.MaxDiff(rateActual) < 1e-3);
                Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-3);
            }

            if (true)
            {
                engine.Compiler.GivePriorityTo(typeof(GammaRatioOp_Laplace));
                xActual = engine.Infer<Gamma>(x);
                yActual = engine.Infer<Gamma>(y);
                rateActual = engine.Infer<Gamma>(rate);
                evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
                Console.WriteLine("y = {0} should be {1}", yActual, yExpected);
                Console.WriteLine("rate = {0} should be {1}", rateActual, rateExpected);
                Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                Assert.True(xExpected.MaxDiff(xActual) < 1e-1);
                if (difficultCase)
                    Assert.True(yExpected.MaxDiff(yActual) < 0.5);
                else
                    Assert.True(yExpected.MaxDiff(yActual) < 3e-2);
                Assert.True(rateExpected.MaxDiff(rateActual) < 0.5);
                Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-2);
            }
        }

        [Fact]
        public void ConstrainTrueReplicateTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            double xPrior = 0.1;
            Variable<bool> x = Variable.Bernoulli(xPrior).Named("x");
            Range item = new Range(10).Named("item");
            VariableArray<Bernoulli> like = Variable.Array<Bernoulli>(item).Named("like");
            using (Variable.ForEach(item))
            {
                Variable.ConstrainEqualRandom(x, like[item]);
            }
            block.CloseBlock();
            like.ObservedValue = Util.ArrayInit(item.SizeAsInt, i => (i == 0) ? Bernoulli.PointMass(true) : Bernoulli.Uniform());
            InferenceEngine engine = new InferenceEngine();
            Bernoulli xActual = engine.Infer<Bernoulli>(x);
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            double evExpected = System.Math.Log(xPrior) + (item.SizeAsInt - 1) * System.Math.Log(0.5);
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-10);
        }

        [Fact]
        public void BernoulliFromLogOdds_DifficultyAbilityTest()
        {
            InferenceEngine engine = new InferenceEngine();
            Rand.Restart(0);

            int nQuestions = 100;
            int nSubjects = 40;
            int nChoices = 4;
            Gaussian abilityPrior = new Gaussian(0, 1);
            Gaussian difficultyPrior = new Gaussian(0, 1);

            double[] trueAbility, trueDifficulty;
            int[] trueTrueAnswer;
            int[][] data = SampleDifficultyAbility(nSubjects, nQuestions, nChoices, abilityPrior, difficultyPrior,
                out trueAbility, out trueDifficulty, out trueTrueAnswer);

            Range question = new Range(nQuestions).Named("question");
            Range subject = new Range(nSubjects).Named("subject");
            Range choice = new Range(nChoices).Named("choice");
            var response = Variable.Array(Variable.Array<int>(question), subject).Named("response");
            response.ObservedValue = data;

            var ability = Variable.Array<double>(subject).Named("ability");
            ability[subject] = Variable.Random(abilityPrior).ForEach(subject);
            var difficulty = Variable.Array<double>(question).Named("difficulty");
            difficulty[question] = Variable.Random(difficultyPrior).ForEach(question);
            var trueAnswer = Variable.Array<int>(question).Named("trueAnswer");
            trueAnswer[question] = Variable.DiscreteUniform(choice).ForEach(question);

            using (Variable.ForEach(subject))
            {
                using (Variable.ForEach(question))
                {
                    var advantage = (ability[subject] - difficulty[question]).Named("advantage");
                    var correct = Variable.BernoulliFromLogOdds(advantage).Named("correct");
                    using (Variable.If(correct))
                        response[subject][question] = trueAnswer[question];
                    using (Variable.IfNot(correct))
                        response[subject][question] = Variable.DiscreteUniform(choice);
                }
            }

            engine.NumberOfIterations = 5;
            subject.AddAttribute(new Sequential());  // needed to get stable convergence
            question.AddAttribute(new Sequential());  // needed to get stable convergence
            var trueAnswerPosterior = engine.Infer<IList<Discrete>>(trueAnswer);
            int numCorrect = 0;
            for (int q = 0; q < nQuestions; q++)
            {
                int bestGuess = trueAnswerPosterior[q].GetMode();
                if (bestGuess == trueTrueAnswer[q])
                    numCorrect++;
            }
            double pctCorrect = 100.0 * numCorrect / nQuestions;
            Console.WriteLine("{0}% TrueAnswers correct", pctCorrect.ToString("f0"));
            var difficultyPosterior = engine.Infer<IList<Gaussian>>(difficulty);
            for (int q = 0; q < System.Math.Min(nQuestions, 4); q++)
            {
                Console.WriteLine("difficulty[{0}] = {1} (sampled from {2})", q, difficultyPosterior[q], trueDifficulty[q].ToString("g2"));
            }
            var abilityPosterior = engine.Infer<IList<Gaussian>>(ability);
            for (int s = 0; s < System.Math.Min(nSubjects, 4); s++)
            {
                Console.WriteLine("ability[{0}] = {1} (sampled from {2})", s, abilityPosterior[s], trueAbility[s].ToString("g2"));
            }
        }

        public int[][] SampleDifficultyAbility(int nSubjects, int nQuestions, int nChoices, Gaussian abilityPrior, Gaussian difficultyPrior,
            out double[] ability, out double[] difficulty, out int[] trueAnswer)
        {
            ability = Util.ArrayInit(nSubjects, s => abilityPrior.Sample());
            difficulty = Util.ArrayInit(nQuestions, q => difficultyPrior.Sample());
            trueAnswer = Util.ArrayInit(nQuestions, q => Rand.Int(nChoices));
            int[][] response = new int[nSubjects][];
            for (int s = 0; s < nSubjects; s++)
            {
                response[s] = new int[nQuestions];
                for (int q = 0; q < nQuestions; q++)
                {
                    double advantage = ability[s] - difficulty[q];
                    bool correct = Bernoulli.Sample(MMath.Logistic(advantage));
                    if (correct)
                        response[s][q] = trueAnswer[q];
                    else
                        response[s][q] = Rand.Int(nChoices);
                }
            }
            return response;
        }

        [Fact]
        public void BernoulliFromLogOddsCRTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> w = Variable.GaussianFromMeanAndPrecision(1.2, 0.4).Named("w");
            Variable<bool> y = Variable.BernoulliFromLogOdds(w).Named("y");
            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine();

            for (int trial = 0; trial < 2; trial++)
            {
                Gaussian wExpected;
                double evExpected;
                if (trial == 0)
                {
                    y.ObservedValue = true;
                    wExpected = new Gaussian(1.735711683643876, 1.897040876799618);
                    evExpected = System.Math.Log(0.697305276585867);
                }
                else
                {
                    y.ObservedValue = false;
                    wExpected = new Gaussian(-0.034096780812067, 1.704896988818977);
                    evExpected = System.Math.Log(0.302694723413305);
                }
                Gaussian wActual = ie.Infer<Gaussian>(w);
                double evActual = ie.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("w = {0} should be {1}", wActual, wExpected);
                Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                Assert.True(wExpected.MaxDiff(wActual) < 1e-4);
                Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-4) < 1e-4);
            }
        }

        [Fact]
        public void BernoulliFromLogOddsRRTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> w = Variable.GaussianFromMeanAndPrecision(1.2, 0.4).Named("w");
            Variable<bool> y = Variable.BernoulliFromLogOdds(w).Named("y");
            double yLike = 0.3;
            Variable.ConstrainEqualRandom(y, new Bernoulli(yLike));
            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine();

            // these are the true values, not the expected results of EP 
            // EP is not exact on this model
            Bernoulli yExpected = new Bernoulli(0.496800207892833);
            Bernoulli yActual = ie.Infer<Bernoulli>(y);
            Gaussian wExpected = new Gaussian(0.845144432260138, 2.583377542742149);
            double evExpected = System.Math.Log(0.421077889365311);
            Gaussian wActual = ie.Infer<Gaussian>(w);
            double evActual = ie.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("y = {0} should be {1}", yActual, yExpected);
            Console.WriteLine("w = {0} should be {1}", wActual, wExpected);
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            //Assert.True(yExpected.MaxDiff(yActual) < 1e-4);
            //Assert.True(wExpected.MaxDiff(wActual) < 1e-4);
            //Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-8) < 1e-4);
        }

        [Fact]
        public void BernoulliLogisticCRTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> w = Variable.GaussianFromMeanAndPrecision(1.2, 0.4).Named("w");
            Variable<bool> y = Variable.Bernoulli(Variable.Logistic(w).Named("p")).Named("y");
            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine();

            for (int trial = 0; trial < 2; trial++)
            {
                Gaussian wExpected;
                double evExpected;
                if (trial == 0)
                {
                    y.ObservedValue = true;
                    wExpected = new Gaussian(1.735711683643876, 1.897040876799618);
                    evExpected = System.Math.Log(0.697305276585867);
                }
                else
                {
                    y.ObservedValue = false;
                    wExpected = new Gaussian(-0.034096780812067, 1.704896988818977);
                    evExpected = System.Math.Log(0.302694723413305);
                }
                Gaussian wActual = ie.Infer<Gaussian>(w);
                double evActual = ie.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("w = {0} should be {1}", wActual, wExpected);
                Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                Assert.True(wExpected.MaxDiff(wActual) < 1e-4);
                Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-4) < 1e-4);
            }
        }

        private void BernoulliLogisticModel(Range item, out VariableArray<bool> y, out Variable<double> w, out Variable<bool> evidence)
        {
            evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            w = Variable.GaussianFromMeanAndPrecision(1.2, 0.4).Named("w");
            y = Variable.Array<bool>(item).Named("y");
            Variable<double> p = Variable.Logistic(w).Named("p");
            using (Variable.ForEach(item))
            {
                y[item] = Variable.Bernoulli(p);
            }
            block.CloseBlock();
        }
        private void BernoulliLogisticModel2(Range item, out VariableArray<bool> y, out Variable<double> w, out Variable<bool> evidence)
        {
            evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            w = Variable.GaussianFromMeanAndPrecision(1.2, 0.4).Named("w");
            y = Variable.Array<bool>(item).Named("y");
            using (Variable.ForEach(item))
            {
                Variable<double> p = Variable.Logistic(w).Named("p");
                y[item] = Variable.Bernoulli(p);
            }
            block.CloseBlock();
        }

        [Fact]
        public void BernoulliLogisticCRTest2()
        {
            Range item = new Range(5).Named("item");
            VariableArray<bool> y, y2;
            Variable<double> w, w2;
            Variable<bool> evidence, evidence2;
            BernoulliLogisticModel(item, out y, out w, out evidence);
            BernoulliLogisticModel2(item, out y2, out w2, out evidence2);

            y.ObservedValue = Util.ArrayInit(item.SizeAsInt, i => (i % 2 == 0));
            y2.ObservedValue = y.ObservedValue;

            InferenceEngine engine = new InferenceEngine();
            Gaussian wExpected = engine.Infer<Gaussian>(w);
            Gaussian wActual = engine.Infer<Gaussian>(w2);
            double evExpected = engine.Infer<Bernoulli>(evidence).LogOdds;
            double evActual = engine.Infer<Bernoulli>(evidence2).LogOdds;
            Console.WriteLine("w = {0} should be {1}", wActual, wExpected);
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(wExpected.MaxDiff(wActual) < 1e-6);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-10) < 1e-6);
        }

        [Fact]
        public void BernoulliLogisticRRTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> w = Variable.GaussianFromMeanAndPrecision(1.2, 0.4).Named("w");
            Variable<bool> y = Variable.Bernoulli(Variable.Logistic(w).Named("p")).Named("y");
            double yLike = 0.3;
            Variable.ConstrainEqualRandom(y, new Bernoulli(yLike));
            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine();
            //Beta.AllowImproperSum = true;

            // these are the true values, not the expected results of EP 
            // EP is not exact on this model
            // y = Bernoulli(0.4958) should be Bernoulli(0.4968)
            // w = Gaussian(0.8299, 2.282) should be Gaussian(0.8451, 2.583)
            // evidence = -0.864132543423833 should be -0.864937452043154
            Bernoulli yExpected = new Bernoulli(0.496800207892833);
            Bernoulli yActual = ie.Infer<Bernoulli>(y);
            Gaussian wExpected = new Gaussian(0.845144432260138, 2.583377542742149);
            double evExpected = System.Math.Log(0.421077889365311);
            Gaussian wActual = ie.Infer<Gaussian>(w);
            double evActual = ie.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("y = {0} should be {1}", yActual, yExpected);
            Console.WriteLine("w = {0} should be {1}", wActual, wExpected);
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            //Assert.True(yExpected.MaxDiff(yActual) < 1e-4);
            //Assert.True(wExpected.MaxDiff(wActual) < 1e-4);
            //Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-8) < 1e-4);
        }

        /// <summary>
        /// Test convergence rate of EP on a difficult case
        /// </summary>
        internal void LogisticRRTest2()
        {
            Variable<double> w = Variable.GaussianFromMeanAndPrecision(0, 1.0 / 6).Named("w");
            Variable<double> p = Variable.Logistic(w).Named("p");
            Variable<Beta> pLike = Variable.New<Beta>().Named("pLike");
            Variable.ConstrainEqualRandom(p, pLike);
            InferenceEngine engine = new InferenceEngine();
            pLike.ObservedValue = new Beta(65, 33);
            for (int iter = 1; iter < 20; iter++)
            {
                engine.NumberOfIterations = iter;
                var wPost = engine.Infer(w);
                Console.WriteLine("{0}: {1}", iter, wPost);
            }
        }

        [Fact]
        public void LogisticRRTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> w = Variable.GaussianFromMeanAndPrecision(1.2, 0.4).Named("w");
            Variable<double> p = Variable.Logistic(w).Named("p");
            Variable<Beta> pLike = Variable.New<Beta>().Named("pLike");
            Variable.ConstrainEqualRandom(p, pLike);
            block.CloseBlock();
            InferenceEngine engine = new InferenceEngine();
            //engine.NumberOfIterations = 50;
            //Beta.AllowImproperSum = true;

            for (int trial = 0; trial < 5; trial++)
            {
                double pMeanExpected, pVarianceExpected, evExpected;
                Gaussian wExpected;
                double pMaxDiff, wMaxDiff, evMaxDiff;
                if (trial == 0)
                {
                    pLike.ObservedValue = new Beta(900, 100);
                    pMeanExpected = 0.900398348541759;
                    pVarianceExpected = 8.937152379237305e-005;
                    wExpected = new Gaussian(2.206120388297770, 0.011173774909080);
                    evExpected = System.Math.Log(2.301146908438258);
                    pMaxDiff = 1.4;
                    wMaxDiff = 0.7;
                    evMaxDiff = 8e-4;
                }
                else if (trial == 1)
                {
                    pLike.ObservedValue = new Beta(1.9, 1.1);
                    pMeanExpected = 0.760897924850939;
                    pVarianceExpected = 0.039965142298523;
                    wExpected = new Gaussian(1.547755187872655, 1.862487573678543);
                    evExpected = System.Math.Log(1.293468509455230);
                    pMaxDiff = 0.026;
                    wMaxDiff = 1e-14;
                    evMaxDiff = 1e-12;
                }
                else if (trial == 2)
                {
                    pLike.ObservedValue = new Beta(1.5, 1.0);
                    pMeanExpected = 0.749761013751125;
                    pVarianceExpected = 0.046475109167811;
                    wExpected = new Gaussian(1.512798732811160, 2.131541336692143);
                    evExpected = System.Math.Log(1.224934926942292);
                    pMaxDiff = 0.12;
                    wMaxDiff = 0.0015;
                    evMaxDiff = 0.0025;
                }
                else if (trial == 3)
                {
                    pLike.ObservedValue = new Beta(0.2, 1.8);
                    pMeanExpected = 0.365045037670852;
                    pVarianceExpected = 0.070103215459281;
                    wExpected = new Gaussian(-0.799999972065338, 2.499999742966089);
                    evExpected = System.Math.Log(0.199292845117683);
                    pMaxDiff = 0.07;
                    wMaxDiff = 1e-7;
                    evMaxDiff = 1e-8;
                }
                else
                {
                    pLike.ObservedValue = new Beta(0.9, 0.1);
                    pMeanExpected = 0.900249510880496;
                    pVarianceExpected = 0.024074272963431;
                    wExpected = new Gaussian(3.200623777202209, 3.061252056444702);
                    evExpected = System.Math.Log(0.885586119114385);
                    pMaxDiff = 0.85;
                    wMaxDiff = 1e-10;
                    evMaxDiff = 1e-11;
                }
                Beta pExpected = Beta.FromMeanAndVariance(pMeanExpected, pVarianceExpected);
                Beta pActual = engine.Infer<Beta>(p);
                Gaussian wActual = engine.Infer<Gaussian>(w);
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("p = {0} should be {1}", pActual, pExpected);
                Console.WriteLine("w = {0} should be {1}", wActual, wExpected);
                Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                Assert.True(pExpected.MaxDiff(pActual) < pMaxDiff);
                Assert.True(wExpected.MaxDiff(wActual) < wMaxDiff);
                Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-8) < evMaxDiff);
            }
        }

        internal void LogisticIsPositiveGibbsSampler()
        {
            // Gibbs sampling on the approximate model
            double wMean = 1.2;
            Gaussian wExpected = new Gaussian(1.735, 1.897);
            double logisticVariance = System.Math.PI * System.Math.PI / 3;
            // Gamma approximation of precision distribution
            double shape = 4.5; // or 7.175/2
            double rate = (shape - 1) * logisticVariance;
            Gaussian wPrior = Gaussian.FromMeanAndPrecision(wMean, 0.4);
            Gamma precPrior = Gamma.FromShapeAndRate(shape, rate);
            GaussianEstimator wEst = new GaussianEstimator();
            int niter = (int)1e6;
            int burnin = 100;
            double x = 0;
            double prec = 0;
            for (int iter = 0; iter < niter; iter++)
            {
                // sample w given (x,prec)
                Gaussian wPost = wPrior * GaussianOp.MeanAverageConditional(x, prec);
                double w = wPost.Sample();
                // sample prec given (w,x)
                Gamma precPost = precPrior * GaussianOp.PrecisionAverageConditional(x, w);
                prec = precPost.Sample();
                // sample x given (w,prec)
                Gaussian xPrior = Gaussian.FromMeanAndPrecision(w, prec);
                Gaussian xPost = xPrior * IsPositiveOp.XAverageConditional(true, xPrior);
                double oldx = x;
                x = xPost.Sample();
                if (x < 0)
                    x = oldx; // rejected
                else
                {
                    // compute importance weights
                    double oldweight = xPrior.GetLogProb(oldx) - xPost.GetLogProb(oldx);
                    double newweight = xPrior.GetLogProb(x) - xPost.GetLogProb(x);
                    // acceptance ratio
                    double paccept = System.Math.Exp(newweight - oldweight);
                    if (paccept < 1 && Rand.Double() > paccept)
                    {
                        x = oldx; // rejected
                    }
                }
                if (iter > burnin)
                {
                    wEst.Add(wPost);
                }
            }
            Console.WriteLine("w = {0} should be {1}", wEst.GetDistribution(new Gaussian()), wExpected);
        }
        internal void LogisticIsPositiveRejectionSampler()
        {
            // Rejection sampling on the approximate models
            // Gives E[w] = 1.736
            // Why is EP inaccurate on this model?
            double wMean = 1.2;
            Gaussian wExpected = new Gaussian(1.735, 1.897);
            double logisticVariance = System.Math.PI * System.Math.PI / 3;
            // Gamma approximation of precision distribution
            double shape = 4.5; // or 7.175/2
            double rate = (shape - 1) * logisticVariance;
            // log-normal approximation of precision distribution
            double varLogPrec = System.Math.Log(7.0 / 5);
            double meanLogPrec = 0.5 * varLogPrec - System.Math.Log(logisticVariance);
            Gaussian wPrior = Gaussian.FromMeanAndPrecision(wMean, 0.4);
            GaussianEstimator wEst = new GaussianEstimator();
            int niter = (int)1e6;
            int numAccepted = 0;
            for (int iter = 0; iter < niter; iter++)
            {
                double w = Gaussian.Sample(wMean, 0.4);
                double prec = Gamma.Sample(shape, 1 / rate);
                //double prec_sample = Math.Exp(Gaussian.Sample(meanLogPrec, 1.0/varLogPrec));
                double x = Gaussian.Sample(w, prec);
                if (x >= 0)
                {
                    Gaussian wPost = wPrior * GaussianOp.MeanAverageConditional(x, prec);
                    wEst.Add(wPost);
                    numAccepted++;
                }
            }
            Console.WriteLine("accepted {0} samples", numAccepted);
            Console.WriteLine("w = {0} should be {1}", wEst.GetDistribution(new Gaussian()), wExpected);
            //Console.WriteLine("rejection prec_mean = {0} should be {1}", prec_mean, shape/rate);
            //Console.WriteLine("rejection x_mean = {0}", x_mean);
        }
        internal void LogisticIsPositiveTest()
        {
            // exact w = Gaussian(1.735, 1.897)
            // EP approx w (prec=Gamma) = Gaussian(1.74, 1.922)
            // EP approx w (prec=Exp(N)) = Gaussian(1.742, 1.921)
            double wMean = 1.2;
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> w = Variable.GaussianFromMeanAndPrecision(wMean, 0.4).Named("w");
            Variable<double> prec;
            double logisticVariance = System.Math.PI * System.Math.PI / 3;
            if (true)
            {
                // Student T approximation of a standard logistic distribution.
                double shape = 4.5; // or 7.175/2
                double rate = (shape - 1) * logisticVariance;
                prec = Variable.GammaFromShapeAndRate(shape, rate).Named("prec");
            }
            else
            {
                // log-normal approximation of precision distribution
                double varLogPrec = System.Math.Log(7.0 / 5);
                double meanLogPrec = 0.5 * varLogPrec - System.Math.Log(logisticVariance);
                // On paper, this should be better than the Student T approximation.
                prec = Variable.Exp(Variable.GaussianFromMeanAndPrecision(meanLogPrec, 1.0 / varLogPrec)).Named("prec");
            }
            Variable<double> x = Variable.GaussianFromMeanAndPrecision(w, prec).Named("x");
            Variable.ConstrainPositive(x);
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            Gaussian wExpected = new Gaussian(1.735711683643876, 1.897040876799618);
            //engine.NumberOfIterations = 5;
            //engine.Algorithm = new VariationalMessagePassing();
            //engine.Algorithm = new GibbsSampling();
            Console.WriteLine("w = {0} should be {1}", engine.Infer(w), wExpected);
            //Console.WriteLine("x = {0}", engine.Infer(x));
            double evExpected = 0.697305276585867;
            double evActual = System.Math.Exp(engine.Infer<Bernoulli>(evidence).LogOdds);
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
        }

        internal void GaussianChildIsPositiveTest()
        {
            int n = 5;
            var x = new Variable<double>[n];
            x[0] = Variable.GaussianFromMeanAndPrecision(0, 1);
            for (int i = 1; i < n; i++)
            {
                x[i] = Variable.GaussianFromMeanAndPrecision(x[i - 1], 1);
            }
            Variable.ConstrainPositive(x[n - 1]);
            InferenceEngine engine = new InferenceEngine();
            for (int i = 0; i < n; i++)
            {
                Console.WriteLine("x[{0}] = {1}", i, engine.Infer(x[i]));
            }

            // rejection sampler
            GaussianEstimator[] ests = Util.ArrayInit(n, i => new GaussianEstimator());
            double[] xSample = new double[n];
            for (int iter = 0; iter < 100000; iter++)
            {
                for (int i = 0; i < n; i++)
                {
                    Gaussian prior = (i == 0) ? new Gaussian(0, 1) : new Gaussian(xSample[i - 1], 1);
                    //Gaussian like = (i==n-1) ? Gaussian.Uniform() : new Gaussian(xSample[i+1], 1);
                    xSample[i] = prior.Sample();
                }
                if (xSample[n - 1] > 0)
                {
                    for (int i = 0; i < n; i++)
                    {
                        ests[i].Add(xSample[i]);
                    }
                }
            }
            Console.WriteLine("exact:");
            for (int i = 0; i < n; i++)
            {
                Console.WriteLine("x[{0}] = {1}", i, ests[i].GetDistribution(new Gaussian()));
            }
        }


        [Fact]
        public void GaussianBoxTest()
        {
            int d = 2;
            Variable<Vector> w = Variable.VectorGaussianFromMeanAndVariance(Vector.Zero(d), PositiveDefiniteMatrix.Identity(d)).Named("w");
            Variable<int> n = Variable.New<int>().Named("n");
            Range item = new Range(n).Named("item");
            VariableArray<Vector> x = Variable.Array<Vector>(item).Named("x");
            VariableArray<double> b = Variable.Array<double>(item).Named("b");
            using (Variable.ForEach(item))
            {
                Variable<double> h = Variable.InnerProduct(w, x[item]) + b[item];
                //Variable.ConstrainPositive(h);
                // change the scaling here to stress-test the Logistic operator
                Variable<double> p = Variable.Logistic(100 * h);
                // change 2 to 1.5 here to test power EP
                Variable.ConstrainEqualRandom(p, new Beta(2, 1));
            }
            List<double> shifts = new List<double>();
            List<Vector> vectors = new List<Vector>();
            Box(vectors, shifts, 0);
            for (int i = 0; i < 4; i++)
            {
                Box(vectors, shifts, i * System.Math.PI / 10);
            }
            x.ObservedValue = vectors.ToArray();
            n.ObservedValue = vectors.Count;
            b.ObservedValue = shifts.ToArray();
            //w.AddAttribute(new DivideMessages(false));
            //w.AddAttribute(new TraceMessages());
            item.AddAttribute(new Sequential());

            InferenceEngine engine = new InferenceEngine();
            VectorGaussian xActual = engine.Infer<VectorGaussian>(w);
            if (false)
            {
                VectorGaussian xExpected = VectorGaussian.FromNatural(Vector.FromArray(18.2586, 18.2586), new PositiveDefiniteMatrix(new double[,] { { 11.566, 0 }, { 0, 11.566 } }));
                Console.WriteLine(StringUtil.JoinColumns("x = ", xActual, " should be ", xExpected));
                Assert.True(xExpected.MaxDiff(xActual) < 1e-3);
            }
            else
            {
                Console.WriteLine(xActual);
            }
        }
        private void Box(List<Vector> vectors, List<double> shifts, double angle)
        {
            Matrix rot = new Matrix(2, 2);
            rot[0, 0] = System.Math.Cos(angle);
            rot[1, 1] = rot[0, 0];
            rot[0, 1] = System.Math.Sin(angle);
            rot[1, 0] = -rot[0, 1];
            Vector center = Vector.FromArray(2, 2);
            vectors.Add(rot * Vector.FromArray(1, 0));
            shifts.Add(1 - vectors[vectors.Count - 1].Inner(center));
            vectors.Add(rot * Vector.FromArray(-1, 0));
            shifts.Add(1 - vectors[vectors.Count - 1].Inner(center));
            vectors.Add(rot * Vector.FromArray(0, 1));
            shifts.Add(1 - vectors[vectors.Count - 1].Inner(center));
            vectors.Add(rot * Vector.FromArray(0, -1));
            shifts.Add(1 - vectors[vectors.Count - 1].Inner(center));
        }

        [Fact]
        public void ProbitFactorTest()
        {
            double a = 0.01;
            double p = 0.7;
            Variable<Gaussian> xPrior = Variable.New<Gaussian>();
            Variable<double> x = Variable<double>.Random(xPrior);
            Variable<double> y = Variable.GaussianFromMeanAndVariance(x, a);
            Variable.ConstrainEqualRandom(Variable.IsPositive(y), new Bernoulli(p));
            InferenceEngine engine = new InferenceEngine();
            for (int i = 0; i < 10; i++)
            {
                double vx = 1;
                double mx = -i;
                double z = mx / System.Math.Sqrt(vx + a);
                double LogAverageFactor = System.Math.Log((2 * p - 1) * MMath.NormalCdf(z) + (1 - p));
                double alpha = (2 * p - 1) * System.Math.Exp(Gaussian.GetLogProb(z, 0, 1) - LogAverageFactor - 0.5 * System.Math.Log(vx + a));
                double beta = alpha * (alpha + mx / (vx + a));
                double msg_m = mx + alpha / beta;
                double msg_v = 1 / beta - vx;
                Gaussian msgToX = new Gaussian(msg_m, msg_v);
                Console.WriteLine("msgToX = {0}", msgToX);
                Gaussian xExpected = Gaussian.FromMeanAndVariance(mx, vx) * msgToX;
                if (FactorManager.IsDefaultOperator(typeof(IsPositiveOp_Proper)))
                {
                    xExpected = Gaussian.FromMeanAndVariance(xExpected.GetMean(), System.Math.Min(vx, xExpected.GetVariance()));
                }

                xPrior.ObservedValue = new Gaussian(mx, vx);
                Gaussian xActual = engine.Infer<Gaussian>(x);
                Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);
                Assert.True(xExpected.MaxDiff(xActual) < 1e-8);
            }
        }

        // demonstrates why you must divide messages in a Plus factor
        internal void PlusTest()
        {
            var x = Variable.GaussianFromMeanAndVariance(1, 2).Named("x");
            var y = Variable.GaussianFromMeanAndVariance(3, 4).Named("y");
            var z = x + y;
            z.Name = "z";
            Variable.ConstrainEqualRandom(z, new Gaussian(5, 6));

            InferenceEngine engine = new InferenceEngine();
            Gaussian xActual = engine.Infer<Gaussian>(x);
            Gaussian yActual = engine.Infer<Gaussian>(y);
            Gaussian zActual = engine.Infer<Gaussian>(z);
            Gaussian zPost2 = new Gaussian(xActual.GetMean() + yActual.GetMean(), xActual.GetVariance() + yActual.GetVariance());
            Console.WriteLine(zActual);
            // answers are different due to factorization
            Console.WriteLine(zPost2);
        }

        // requires a non-parallel schedule.
        [Fact]
        public void PlusScheduleTest()
        {
            int n = 40;
            int h = n / 2;
            bool[] data = new bool[n];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (i > h);
            }
            Range item = new Range(data.Length);
            VariableArray<bool> x = Variable.Observed(data, item).Named("x");
            Variable<double> bias = Variable.GaussianFromMeanAndVariance(0, 1).Named("bias");
            Variable<double> bias2 = Variable.GaussianFromMeanAndVariance(0, 1).Named("bias2");
            Variable<double> bias3 = Variable.GaussianFromMeanAndVariance(0, 1).Named("bias3");
            using (Variable.ForEach(item))
            {
                // moving this line outside the loop fixes EP.  VMP is invariant to the placement of this line.
                Variable<double> w = bias + bias2 + bias3;
                w.Name = "w";
                Variable<double> wNoisy = Variable.GaussianFromMeanAndPrecision(w, 0.1).Named("wNoisy");
                x[item] = (wNoisy > 0);
            }
            InferenceEngine engine = new InferenceEngine();//new VariationalMessagePassing());
            engine.NumberOfIterations = 1000;
            //engine.ModelName = "PlusTest";
            Gaussian w1Actual = engine.Infer<Gaussian>(bias);
            Gaussian w1Expected = new Gaussian(-0.05951, 0.2913);
            Console.WriteLine("w1 = {0} should be {1}", w1Actual, w1Expected);
            Assert.True(w1Expected.MaxDiff(w1Actual) < 2e-4);
        }
        // this version works with EP
        //[Fact]
        internal void PlusScheduleTestUnrolled()
        {
            int n = 40;
            int h = n / 2;
            bool[] data = new bool[n];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (i > h);
            }
            Variable<bool>[] x = new Variable<bool>[n];
            Variable<double> bias = Variable.GaussianFromMeanAndVariance(0, 1).Named("bias");
            Variable<double> bias2 = Variable.GaussianFromMeanAndVariance(0, 1).Named("bias2");
            Variable<double> bias3 = Variable.GaussianFromMeanAndVariance(0, 1).Named("bias3");
            for (int i = 0; i < n; i++)
            {
                // moving this line outside the loop fixes EP.  VMP is invariant to the placement of this line.
                Variable<double> w = bias + bias2 + bias3;
                w.Name = "w" + i;
                Variable<double> wNoisy = Variable.GaussianFromMeanAndPrecision(w, 0.1).Named("wNoisy" + i);
                x[i] = (wNoisy > 0).Named("x" + i);
                x[i].ObservedValue = data[i];
            }
            InferenceEngine engine = new InferenceEngine();//new VariationalMessagePassing());
            engine.NumberOfIterations = 1000;
            //engine.ModelName = "PlusTestUnrolled";
            Gaussian w1Actual = engine.Infer<Gaussian>(bias);
            Gaussian w1Expected = new Gaussian(-0.05951, 0.2913);
            Console.WriteLine("w1 = {0} should be {1}", w1Actual, w1Expected);
        }

        [Fact]
        public void VectorGaussianFromMeanAndPrecisionTest()
        {
            VectorGaussianFromMeanAndPrecision(false);
            VectorGaussianFromMeanAndPrecision(true);
        }
        private void VectorGaussianFromMeanAndPrecision(bool computeEvidence)
        {
            Rand.Restart(0);
            // Sample data from standard Gaussian
            Vector[] data = new Vector[3];
            int dim = 3;
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = Vector.FromArray(Util.ArrayInit(dim, d => Rand.Normal(0, 1.0 + d)));
            }

            var evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = null;
            if (computeEvidence)
                block = Variable.If(evidence);

            var meanVariance = PositiveDefiniteMatrix.IdentityScaledBy(dim, 1e1);
            var mean = Variable.VectorGaussianFromMeanAndVariance(Vector.Zero(dim), meanVariance).Named("mean");
            double shape = 4;
            PositiveDefiniteMatrix precRate = PositiveDefiniteMatrix.IdentityScaledBy(dim, shape);
            var precision = Variable.WishartFromShapeAndRate(shape, precRate).Named("precision");

            Range dataRange = new Range(data.Length).Named("n");
            VariableArray<Vector> x = Variable.Array<Vector>(dataRange).Named("x");
            x[dataRange] = Variable.VectorGaussianFromMeanAndPrecision(mean, precision).ForEach(dataRange);
            x.ObservedValue = data;

            if (computeEvidence)
                block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;
            VectorGaussian meanExpected = VectorGaussian.Uniform(dim);
            Wishart precExpected = Wishart.Uniform(dim);
            double evExpected = 0;
            if (computeEvidence)
            {
                // importance sampling
                VectorGaussian meanPrior = VectorGaussian.FromMeanAndVariance(Vector.Zero(dim), meanVariance);
                Wishart precPrior = Wishart.FromShapeAndRate(shape, precRate);
                int nsamples = 100000;
                WishartEstimator precEst = new WishartEstimator(dim);
                VectorGaussianEstimator meanEst = new VectorGaussianEstimator(dim);
                double totalWeight = 0;
                for (int iter = 0; iter < nsamples; iter++)
                {
                    Vector meanSample = meanPrior.Sample();
                    PositiveDefiniteMatrix precSample = precPrior.Sample();
                    VectorGaussian xDist = VectorGaussian.FromMeanAndPrecision(meanSample, precSample);
                    double logWeight = 0;
                    for (int i = 0; i < data.Length; i++)
                    {
                        logWeight += xDist.GetLogProb(data[i]);
                    }
                    double weight = System.Math.Exp(logWeight);
                    totalWeight += weight;
                    precEst.Add(precSample, weight);
                    meanEst.Add(meanSample, weight);
                }
                meanExpected = meanEst.GetDistribution(new VectorGaussian(dim));
                precExpected = precEst.GetDistribution(new Wishart(dim));
                evExpected = System.Math.Log(totalWeight / nsamples);
                Console.WriteLine("importance sampling");
                Console.WriteLine(StringUtil.JoinColumns("mean = ", meanExpected));
                Console.WriteLine(StringUtil.JoinColumns("prec = ", precExpected));
                Console.WriteLine("evidence = {0}", evExpected);
            }

            for (int iter = computeEvidence ? 1 : 0; iter < 3; iter++)
            {
                if (iter == 0)
                    engine.Algorithm = new GibbsSampling();
                else if (iter == 1)
                    engine.Algorithm = new ExpectationPropagation();
                else if (iter == 2)
                    engine.Algorithm = new VariationalMessagePassing();
                var meanActual = engine.Infer<VectorGaussian>(mean);
                var precActual = engine.Infer<Wishart>(precision);
                // Retrieve the posterior distributions
                Console.WriteLine(engine.Algorithm.ShortName);
                Console.WriteLine(StringUtil.JoinColumns("mean = ", meanActual));
                Console.WriteLine(StringUtil.JoinColumns("prec = ", precActual));
                if (computeEvidence)
                {
                    double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                    Console.WriteLine("evidence = {0}", evActual);
                }
                if (iter == 0)
                {
                    meanExpected = meanActual;
                    precExpected = precActual;
                }
                else if (iter == 1)
                {
                    // && !computeEvidence)
                    Assert.True(meanExpected.MaxDiff(meanActual) < 2);
                    Assert.True(precExpected.MaxDiff(precActual) < 1);
                }
            }
        }

        internal void GaussianFromMeanAndVarianceTest3()
        {
            Rand.Restart(0);
            // Sample data from standard Gaussian
            double[] data = new double[40];
            for (int i = 0; i < data.Length; i++)
                data[i] = Rand.Normal(0, 1);

            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100).Named("mean");
            Variable<double> variance = Variable.GammaFromShapeAndScale(1, 1).Named("variance");

            Range dataRange = new Range(data.Length).Named("n");
            VariableArray<double> x = Variable.Array<double>(dataRange).Named("x");
            x[dataRange] = Variable.GaussianFromMeanAndVariance(mean, variance).ForEach(dataRange);
            x.ObservedValue = data;
            variance.InitialiseTo(Gamma.PointMass(1));

            GaussianEstimator est = new GaussianEstimator();
            for (int i = 0; i < data.Length; i++)
            {
                est.Add(data[i]);
            }
            Gaussian g = est.GetDistribution(Gaussian.Uniform());
            Gaussian meanExpected = (new Gaussian(g.GetMean(), g.GetVariance() / data.Length)) * (new Gaussian(0, 100));

            InferenceEngine engine = new InferenceEngine();
            Gaussian meanActual = engine.Infer<Gaussian>(mean);
            Gamma varActual = engine.Infer<Gamma>(variance);
            // Retrieve the posterior distributions
            Console.WriteLine("mean = {0} should be near {1}", meanActual, meanExpected);
            Console.WriteLine("variance = {0}[mode={1}] should be near {2}", varActual, varActual.GetMode().ToString("g4"), g.GetVariance().ToString("g4"));
        }

        /// <summary>
        /// Test PointEstimate with damping
        /// </summary>
        [Fact]
        [Trait("Category", "ModifiesGlobals")]
        public void LearningAGaussianEPPointTest()
        {
            LearningAGaussianEPPoint(false, false);
            LearningAGaussianEPPoint(true, false);
            LearningAGaussianEPPoint(false, true);
            LearningAGaussianEPPoint(true, true);
            using (TestUtils.TemporarilyUseMeanPointGamma)
            {
                LearningAGaussianEPPoint(false, false, true);
                LearningAGaussianEPPoint(true, false, true);
            }
        }

        private void LearningAGaussianEPPoint(bool meanIsPointEstimate, bool useEM, bool useMean = false)
        {
            // Create mean and precision random variables
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100).Named("mean");
            Variable<double> precision;
            bool useTruncatedGamma = false;
            if (useTruncatedGamma)
            {
                Variable<double> precisionT = Variable.Random(new TruncatedGamma(1, 1, 0, double.PositiveInfinity)).Named("precision");
                precisionT.AddAttribute(new PointEstimate());
                precisionT.InitialiseTo(Gamma.PointMass(1.0));
                precision = Variable.Copy(precisionT);
                precision.AddAttribute(new MarginalPrototype(new Gamma()));
            }
            else
            {
                precision = Variable.GammaFromShapeAndScale(1, 1).Named("precision");
                precision.AddAttribute(new PointEstimate());
                precision.InitialiseTo(Gamma.PointMass(1.0));
            }
            Variable<int> xCount = Variable.New<int>().Named("xCount");
            Range item = new Range(xCount).Named("item");
            item.AddAttribute(new Sequential());
            VariableArray<double> x = Variable.Array<double>(item).Named("x");
            using (Variable.ForEach(item))
            {
                var precisionDamped = Variable<double>.Factor(Damp.Forward<double>, precision, useEM ? 1.0 : 0.5);
                x[item] = Variable.GaussianFromMeanAndPrecision(mean, precisionDamped);
            }

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.InitialisationAffectsSchedule = true;
            engine.ShowProgress = false;
            //engine.Compiler.GivePriorityTo(typeof(PointEstimatorForwardOp_Mean<>));
            //SecantBufferData<Gamma>.debug = true;
            if (useEM)
            {
                engine.Compiler.GivePriorityTo(typeof(GaussianOp_EM));
            }
            //PointEstimatorForwardOp_SecantGamma.UseMean = !useEM;
            precision.InitialiseTo(Gamma.PointMass(1.0));
            // this is for PointEstimatorForwardOp_Mode
            double[] precisionExpectedMode = new double[] { 0.7733324320834765, 0.7925625949603335 };
            // this is for PointEstimatorForwardOp_Mean
            // Gamma(5.04, 0.1875)[mean=0.9451][mode=0.7576]
            // Gamma(5.045, 0.192)[mean=0.9686][mode=0.7766]
            //double[] precisionExpectedMean = new double[] { 0.7576, 0.7766 };
            double[] precisionExpectedMean = new double[] { 0.94513928278798, 0.9686429274597133 };
            if (meanIsPointEstimate)
            {
                mean.AddAttribute(new PointEstimate());
                mean.InitialiseTo(Gaussian.PointMass(0));
                // Gamma(6, 0.1761)[mean=1.057][mode=0.8805]
                precisionExpectedMode = new double[] { 0.8591349795002973, 0.880501882579348 };
                precisionExpectedMean = new double[] { 1.030961977426691, 1.056602354669931 };
            }
            Rand.Restart(0);
            for (int trial = 0; trial < 2; trial++)
            {
                // Sample data from standard Gaussian
                double[] data = new double[10];
                for (int i = 0; i < data.Length; i++)
                    data[i] = Rand.Normal(0, 1);
                x.ObservedValue = data;
                xCount.ObservedValue = data.Length;

                for (int iter = 1; iter <= 100; iter++)
                {
                    engine.NumberOfIterations = iter;
                    var precisionActual2 = engine.Infer<Gamma>(precision);
                    //Console.WriteLine("{0}: {1}[mode={2:g4}]", iter, precisionActual2, precisionActual2.GetMode());
                }

                // Retrieve the posterior distributions
                Console.WriteLine("mean = {0}", engine.Infer(mean));
                Gamma precisionActual = engine.Infer<Gamma>(precision);
                Console.WriteLine("prec = {0}[mode={1:g16}]", precisionActual, precisionActual.GetMode());
                if (useMean)
                    Assert.True(MMath.AbsDiff(precisionExpectedMean[trial], precisionActual.GetMode(), 1e-10) < 1e-10);
                else
                    Assert.True(MMath.AbsDiff(precisionExpectedMode[trial], precisionActual.GetMode(), 1e-10) < 1e-10);
            }
        }

        /// <summary>
        /// Tests a case where Rprop converges slowly.
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")]
        public void PointEstimateTest()
        {
            Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 1).Named("x");
            Variable<double> y = Variable.GaussianFromMeanAndVariance(0, 1).Named("y");
            Variable.ConstrainEqualRandom(x, new Gaussian(-0.03945, 0.03732));
            Variable.ConstrainEqualRandom(y, new Gaussian(-0.08798, 0.07365));
            var xNoisy = Variable.GaussianFromMeanAndVariance(x, 1e-8);
            xNoisy.Name = nameof(xNoisy);
            Variable.ConstrainTrue(xNoisy < y);
            //Variable.ConstrainTrue(xNoisy > 0);
            //x.InitialiseTo(Gaussian.PointMass(1));
            //y.InitialiseTo(Gaussian.PointMass(10));
            x.InitialiseTo(Gaussian.PointMass(-0.0588738));
            y.InitialiseTo(Gaussian.PointMass(-0.0584));
            x.AddAttribute(new PointEstimate());
            y.AddAttribute(new PointEstimate());

            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;
            //engine.Compiler.UseExistingSourceFiles = true;
            // TODO: throw if InitialisationAffectsSchedule is not true
            engine.Compiler.InitialisationAffectsSchedule = true;
            engine.Compiler.GivePriorityTo(typeof(GaussianFromMeanAndVarianceOp_PointVariance));
            double xPrevious = double.NaN, yPrevious = double.NaN;
            List<string> lines = new List<string>();
            for (int iter = 1; iter <= 20000; iter++)
            {
                engine.NumberOfIterations = iter;
                double xActual = engine.Infer<Gaussian>(x).Point;
                double yActual = engine.Infer<Gaussian>(y).Point;
                //Trace.WriteLine($"{iter} {xActual} {yActual}");
                string line = $"{xActual}, {yActual}";
                lines.Add(line);
                if (MMath.AbsDiff(xActual, xPrevious) < 1e-8 && MMath.AbsDiff(yActual, yPrevious) < 1e-8)
                {
                    Trace.WriteLine($"Converged after {iter} iterations");
                    break;
                }
                xPrevious = xActual;
                yPrevious = yActual;
                if (iter == 10000) throw new Exception("Did not converge fast enough");
            }
            //System.IO.File.WriteAllLines("points.csv", lines);
        }

        [Fact]
        public void VectorGaussianEvidenceTest()
        {
            double Sigma = 10.0;
            var evidenceA = Variable.Bernoulli(0.5).Named("evidenceA");
            var blockA = Variable.If(evidenceA);
            var root = Variable.VectorGaussianFromMeanAndVariance(Vector.Zero(3), PositiveDefiniteMatrix.IdentityScaledBy(3, 0.2 * Sigma));
            var a = Variable.VectorGaussianFromMeanAndVariance(root, PositiveDefiniteMatrix.IdentityScaledBy(3, 0.5 * Sigma));
            a.ObservedValue = Vector.FromArray(5.0, 6.0, 0.0);
            root.Name = "root";
            a.Name = "a";
            blockA.CloseBlock();

            var evidenceB = Variable.Bernoulli(0.5).Named("evidenceB");
            var blockB = Variable.If(evidenceB);
            var root1 = Variable.GaussianFromMeanAndVariance(0.0, 0.2 * Sigma);
            var root2 = Variable.GaussianFromMeanAndVariance(0.0, 0.2 * Sigma);
            var root3 = Variable.GaussianFromMeanAndVariance(0.0, 0.2 * Sigma);
            var a1 = Variable.GaussianFromMeanAndVariance(root1, 0.5 * Sigma);
            var a2 = Variable.GaussianFromMeanAndVariance(root2, 0.5 * Sigma);
            var a3 = Variable.GaussianFromMeanAndVariance(root3, 0.5 * Sigma);
            a1.ObservedValue = 5.0;
            a2.ObservedValue = 6.0;
            a3.ObservedValue = 0.0;
            blockB.CloseBlock();

            var ie = new InferenceEngine();
            if (false)
            {
                Console.WriteLine("root = {0}", ie.Infer(root));
                Console.WriteLine("root1 = {0}", ie.Infer(root1));
                Console.WriteLine("root2 = {0}", ie.Infer(root2));
                Console.WriteLine("root3 = {0}", ie.Infer(root3));
            }
            double evA = ie.Infer<Bernoulli>(evidenceA).LogOdds;
            Console.WriteLine("Evidence A = {0}", evA);
            double evB = ie.Infer<Bernoulli>(evidenceB).LogOdds;
            Console.WriteLine("Evidence B = {0}", evB);
            Assert.True(MMath.AbsDiff(evA, evB, 1e-6) < 1e-10);
        }

        [Fact]
        public void UnobservedPoissonTest()
        {
            double a = 1, b = 1;
            var mean = Variable.GammaFromShapeAndRate(a, b);
            var x1 = Variable.Poisson(mean);
            var x2 = Variable.Poisson(mean);
            var x3 = Variable.Poisson(mean);
            x1.ObservedValue = 5;
            Variable.ConstrainEqualRandom(x3, Poisson.PointMass(x1.ObservedValue));
            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            Gamma meanExpected = Gamma.FromShapeAndRate(a + 2 * x1.ObservedValue, b + 2);
            Gamma meanActual = engine.Infer<Gamma>(mean);
            Console.WriteLine("mean = {0} should be {1}", meanActual, meanExpected);
            Assert.True(meanExpected.MaxDiff(meanActual) < 1e-10);
            Poisson x2Actual = engine.Infer<Poisson>(x2);
            Poisson x2Expected = new Poisson(meanExpected.GetMean());
            Console.WriteLine("x2 = {0} should be {1}", x2Actual, x2Expected);
            Assert.True(x2Expected.MaxDiff(x2Actual) < 1e-10);
        }

        // It is not clear whether inference should fail here or not.
        //[Fact]
        //[ExpectedException(typeof(ArgumentException))]
        internal void InfiniteVarianceError()
        {
            // Create mean and precision random variables
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100).Named("mean");
            // This prior for the precision leads to infinite variance on x.
            // An exception should be thrown in this case.
            Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1).Named("precision");

            for (int i = 0; i < 2; i++)
            {
                Variable<double> x = Variable.GaussianFromMeanAndPrecision(mean, precision).Named("x" + i);
                Variable.ConstrainPositive(x);
            }

            InferenceEngine engine = new InferenceEngine();
            // Retrieve the posterior distributions
            Console.WriteLine("mean=" + engine.Infer(mean));
            Console.WriteLine("prec=" + engine.Infer(precision));
        }

        internal void ProbitTracking()
        {
            bool[] data = new bool[] { true, true, false };
            int n = data.Length;
            Variable<double> mu = Variable.GaussianFromMeanAndVariance(0, 1).Named("mu");
            Variable<double>[] f = new Variable<double>[n];
            Variable<bool>[] y = new Variable<bool>[n];
            f[0] = Variable.GaussianFromMeanAndVariance(mu, 1).Named("f0");
            double phi = 0.85;
            for (int i = 1; i < n; i++)
            {
                f[i] = Variable.GaussianFromMeanAndVariance(phi * f[i - 1] + (1 - phi) * mu, 1).Named("f" + i);
            }
            for (int i = 0; i < n; i++)
            {
                y[i] = (f[i] > 0);
                y[i].ObservedValue = data[i];
            }
            InferenceEngine engine = new InferenceEngine();
            for (int i = 0; i < n; i++)
            {
                Gaussian fActual = engine.Infer<Gaussian>(f[i]);
                Console.WriteLine("f[{0}] = {1}", i, fActual);
            }
        }

        internal void StudentTrackingTest()
        {
            //(new EpTests()).LogisticBernoulliTracking();
            double[] data = new double[] { 1, 2, 3 };
            //double[] data = (new EpTests()).StudentTrackingData(3);
            for (int i = 0; i < data.Length; i++)
            {
                Console.WriteLine("data[{0}] = {1}", i, data[i]);
            }
            // The results from these 3 models should be similar
            Console.WriteLine("VectorStudent:");
            (new EpTests()).VectorStudentTracking(data);
            Console.WriteLine("Student:");
            (new EpTests()).StudentTracking(data);
            Console.WriteLine("DiscreteStudent:");
            (new EpTests()).DiscreteStudentTracking(data);
        }

        public double[] StudentTrackingData(int n)
        {
            double[] data = new double[n];
            double mu = Gaussian.Sample(0, 1);
            double phi = 0.85;
            double f = 0;
            for (int i = 0; i < n; i++)
            {
                if (i == 0)
                {
                    f = Gaussian.Sample(mu, 1);
                }
                else
                {
                    f = Gaussian.Sample(phi * f + (1 - phi) * mu, 1);
                }
                double prec = Gamma.Sample(3.0 / 2, 2.0 / 3);
                data[i] = Gaussian.Sample(f, prec);
            }
            return data;
        }

        private void StudentTracking(double[] data)
        {
            int n = data.Length;
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock evBlock = Variable.If(evidence);
            Variable<double> mu = Variable.GaussianFromMeanAndVariance(0, 1).Named("mu");
            Variable<double>[] f = new Variable<double>[n];
            Variable<double>[] y = new Variable<double>[n];
            double phi = 0.85;
            for (int i = 0; i < n; i++)
            {
                if (i == 0)
                {
                    f[0] = Variable.GaussianFromMeanAndVariance(mu, 1).Named("f0");
                }
                else
                {
                    f[i] = Variable.GaussianFromMeanAndVariance(phi * f[i - 1] + (1 - phi) * mu, 1).Named("f" + i);
                }
            }
            for (int i = 0; i < n; i++)
            {
                Variable<double> prec = Variable.GammaFromShapeAndScale(3.0 / 2, 2.0 / 3);
                //Variable<double> prec = Variable.Constant(1.0);
                y[i] = Variable.GaussianFromMeanAndPrecision(f[i], prec);
                y[i].ObservedValue = data[i];
            }
            evBlock.CloseBlock();
            InferenceEngine engine = new InferenceEngine();
            if (false)
            {
                engine.Algorithm = new GibbsSampling();
                engine.NumberOfIterations = 200000;
                engine.ShowProgress = false;
            }
            for (int i = 0; i < n; i++)
            {
                Gaussian fActual = engine.Infer<Gaussian>(f[i]);
                Console.WriteLine("f[{0}] = {1}", i, fActual);
            }
            Console.WriteLine("evidence = {0}", engine.Infer<Bernoulli>(evidence).LogOdds);
        }

        [Fact]
        public void VectorStudentTrackingTest()
        {
            VectorStudentTracking(new double[] { 1, 2, 3 });
        }


        private void VectorStudentTracking(double[] data)
        {
            int n = data.Length;
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock evBlock = Variable.If(evidence);
            Variable<Vector>[] fmu = new Variable<Vector>[n];
            Variable<double>[] y = new Variable<double>[n];
            double phi = 0.85;
            double muVar = 1;
            // The implicit equality constraint here prevents GibbsSampling from giving correct results.
            Variable<PositiveDefiniteMatrix> Aprec = Variable.Constant(new PositiveDefiniteMatrix(new double[,] { { 1, 0 }, { 0, 1e-100 } }).Inverse()).Named("Aprec");
            var A = new Matrix(new double[,] { { phi, 1 - phi }, { 0, 1 } });
            for (int i = 0; i < n; i++)
            {
                if (i == 0)
                {
                    fmu[0] = Variable.VectorGaussianFromMeanAndVariance(
                        Vector.Zero(2),
                        new PositiveDefiniteMatrix(new double[,] { { muVar + 1, muVar }, { muVar, muVar } })).Named("fmu0");
                }
                else
                {
                    fmu[i] = Variable.VectorGaussianFromMeanAndPrecision(
                        Variable.MatrixTimesVector(A, fmu[i - 1]),
                        Aprec).Named("fmu" + i);
                }
            }
            Variable<Vector> pickf = Variable.Constant(Vector.FromArray(new double[] { 1, 0 })).Named("pickf");
            for (int i = 0; i < n; i++)
            {
                Variable<double> f = Variable.InnerProduct(pickf, fmu[i]).Named("f" + i);
                Variable<double> prec = Variable.GammaFromShapeAndScale(3.0 / 2, 2.0 / 3);
                //Variable<double> prec = Variable.Constant(1.0);
                y[i] = Variable.GaussianFromMeanAndPrecision(f, prec);
                y[i].ObservedValue = data[i];
            }
            evBlock.CloseBlock();
            InferenceEngine engine = new InferenceEngine();
            for (int i = 0; i < n; i++)
            {
                VectorGaussian d = engine.Infer<VectorGaussian>(fmu[i]);
                Gaussian g = d.GetMarginal(0);
                Console.WriteLine("f[{0}] = {1}", i, g);
            }
            Console.WriteLine("evidence = {0}", engine.Infer<Bernoulli>(evidence).LogOdds);
        }

        private Matrix GetTransitionMatrix(double phi, double[] nodes, double mu)
        {
            Matrix transition = new Matrix(nodes.Length, nodes.Length);
            for (int j = 0; j < nodes.Length; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < nodes.Length; k++)
                {
                    transition[j, k] = System.Math.Exp(Gaussian.GetLogProb(nodes[k], phi * nodes[j] + (1 - phi) * mu, 1));
                    sum += transition[j, k];
                }
                for (int k = 0; k < nodes.Length; k++)
                {
                    transition[j, k] /= sum;
                }
            }
            return transition;
        }
        private Vector GetPrior(double[] nodes, double mu)
        {
            Vector prior = Vector.Zero(nodes.Length);
            for (int k = 0; k < nodes.Length; k++)
            {
                prior[k] = System.Math.Exp(Gaussian.GetLogProb(nodes[k], mu, 1));
            }
            return prior;
        }

        private void DiscreteStudentTracking(double[] data)
        {
            int numNodes = 100;
            double inc = 10.0 / numNodes;
            double[] nodes = new double[numNodes];
            for (int i = 0; i < nodes.Length; i++)
            {
                nodes[i] = -5 + i * inc;
            }
            Range nodeRange = new Range(nodes.Length);

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock evBlock = Variable.If(evidence);
            int n = data.Length;
            Variable<int>[] f = new Variable<int>[n];
            double phi = 0.85;
            Variable<double> muVar = Variable.GaussianFromMeanAndVariance(0, 1).Named("mu");
            Variable<Matrix> transitionMatrix = Variable.New<Matrix>().Named("transitionMatrix");
            Variable<Vector> prior = Variable.New<Vector>().Named("prior");
            for (int i = 0; i < n; i++)
            {
                if (i == 0)
                {
                    f[0] = Variable.Discrete(nodeRange, prior).Named("f0");
                }
                else
                {
                    f[i] = Variable<int>.Factor(Factor.Discrete, f[i - 1], transitionMatrix).Named("f" + i);
                    f[i].AddAttribute(new MarginalPrototype(Discrete.Uniform(nodes.Length)));
                    f[i].AddAttribute(new ValueRange(nodeRange));
                }
            }
            for (int i = 0; i < n; i++)
            {
                double[] probs = new double[nodes.Length];
                for (int k = 0; k < nodes.Length; k++)
                {
                    //probs[k] = Math.Exp(Gaussian.GetLogProb(data[i], nodes[k], 1));
                    probs[k] = System.Math.Exp(GaussianOp.TPdfLn(data[i] - nodes[k], 3, 4));
                }
#if false
            //Variable<bool> y = Variable<bool>.Factor(Factor.BernoulliFromDiscrete, f[i], probs);
#else
                VariableArray<double> probsVar = new VariableArray<double>(nodeRange).Named("Probs_" + i);
                probsVar.ObservedValue = probs;
                Variable<bool> y = Variable.New<bool>();
                using (Variable.Switch(f[i]))
                    y.SetTo(Variable.Bernoulli(probsVar[f[i]]));
#endif
                y.ObservedValue = true;
            }
            evBlock.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;

            DiscreteEstimator[] ests = new DiscreteEstimator[data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                ests[i] = new DiscreteEstimator(nodes.Length);
            }
            double ev = Double.NegativeInfinity;
            for (int j = 0; j < nodes.Length; j++)
            {
                double mu = nodes[j];
                muVar.ObservedValue = mu;
                transitionMatrix.ObservedValue = GetTransitionMatrix(phi, nodes, mu);
                prior.ObservedValue = GetPrior(nodes, mu);
                double scale = engine.Infer<Bernoulli>(evidence).LogOdds;
                ev = MMath.LogSumExp(ev, scale);
                for (int i = 0; i < data.Length; i++)
                {
                    Discrete fPost = engine.Infer<Discrete>(f[i]);
                    ests[i].Add(fPost, System.Math.Exp(scale));
                }
            }
            ev += System.Math.Log(inc);
            for (int i = 0; i < data.Length; i++)
            {
                Discrete fPost = ests[i].GetDistribution(null);
                // compute mean and variance
                double mean = 0;
                for (int k = 0; k < nodes.Length; k++)
                {
                    mean += nodes[k] * fPost[k];
                }
                double var = 0;
                for (int k = 0; k < nodes.Length; k++)
                {
                    double diff = nodes[k] - mean;
                    var += diff * diff * fPost[k];
                }
                Gaussian g = new Gaussian(mean, var);
                Console.WriteLine("f[{0}] = {1}", i, g);
            }
            Console.WriteLine("evidence = {0}", ev);
        }

        //[Fact]
        [Trait("Category", "ModifiesGlobals")]
        internal void NoisyCoinTest2()
        {
            SharedVariable<double> pShared = SharedVariable<double>.Random(new Beta(1, 1));
            int n = 100;
            double noise = 0.2;
            Model model = new Model(n);
            Variable<double> p = pShared.GetCopyFor(model);
            Variable<bool> flip = Variable.Bernoulli(p);
            Variable<bool> noisyFlip = Variable.New<bool>().Named("noisyFlip");
            using (Variable.If(flip))
                noisyFlip.SetTo(Variable.Bernoulli(1 - noise));
            using (Variable.IfNot(flip))
                noisyFlip.SetTo(Variable.Bernoulli(noise));

            Variable.ConstrainFalse(noisyFlip);

            using (TestUtils.TemporarilyAllowBetaImproperSums)
            {
                InferenceEngine engine = new InferenceEngine();
                engine.ShowProgress = false;
                for (int iter = 0; iter < 20; iter++)
                {
                    for (int batch = 0; batch < model.BatchCount; batch++)
                    {
                        model.InferShared(engine, batch);
                    }
                }
                Beta pActual = pShared.Marginal<Beta>();
                Console.WriteLine(pActual);
            }
        }

        /// <summary>
        /// not sure if this is needed as a test
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")]
        [Trait("Category", "ModifiesGlobals")]
        public void NoisyCoinTest()
        {
            Variable<double> p = Variable.Beta(1, 1).Named("p");
            Variable<int> n = Variable.New<int>().Named("n");
            Range item = new Range(n).Named("item");
            VariableArray<bool> flips = Variable.Array<bool>(item);
            flips[item] = Variable.Bernoulli(p).ForEach(item);
            VariableArray<bool> noisyFlips = Variable.Array<bool>(item);
            double noise = 0.2;
            using (Variable.ForEach(item))
            {
                using (Variable.If(flips[item]))
                    noisyFlips[item] = Variable.Bernoulli(1 - noise);
                using (Variable.IfNot(flips[item]))
                    noisyFlips[item] = Variable.Bernoulli(noise);
            }
            InferenceEngine engine = new InferenceEngine();
            using (TestUtils.TemporarilyAllowBetaImproperSums)
            {
                for (int i = 100; i <= 100; i++)
                {
                    n.ObservedValue = i;
                    noisyFlips.ObservedValue = new bool[i];
                    Beta pActual = engine.Infer<Beta>(p);
                    Console.WriteLine(pActual);
                }
            }
        }

        [Fact]
        public void BernoulliBetaTest()
        {
            double pTrue = 0.7;
            bool[] data = Util.ArrayInit(100, i => (Rand.Double() < pTrue));
            Variable<double> p = Variable.Beta(1, 1).Named("p");
            Range item = new Range(data.Length);
            VariableArray<bool> x = Variable.Array<bool>(item).Named("x");
            x[item] = Variable.Bernoulli(p).ForEach(item);
            x.ObservedValue = data;
            InferenceEngine engine = new InferenceEngine();
            Beta pActual = engine.Infer<Beta>(p);
            Console.WriteLine("p = {0}", pActual);
        }

        [Fact]
        public void BernoulliBetaTest2()
        {
            Variable<bool>[][] b = new Variable<bool>[2][];
            Variable<double>[] p = new Variable<double>[b.Length];
            for (int i = 0; i < b.Length; i++)
            {
                b[i] = new Variable<bool>[2];
                p[i] = Variable.Beta(1, 1).Named("p[" + i + "]");
                for (int j = 0; j < b[i].Length; j++)
                {
                    b[i][j] = Variable.Bernoulli(p[i]).Named("b[" + i + "][" + j + "]");
                }
            }
            Variable<bool> x = b[0][0] | b[0][1];
            Variable<bool> y = b[1][0] | !b[1][1];
            Variable.ConstrainTrue(x & y);

            InferenceEngine engine = new InferenceEngine();
            Beta dist1 = engine.Infer<Beta>(p[0]);
            Beta dist2 = engine.Infer<Beta>(p[1]);
            Console.WriteLine(dist1);
            Console.WriteLine(dist2);
            Assert.True(System.Math.Abs(dist1.GetMean() - 0.6) < 1e-2);
            Assert.True(System.Math.Abs(dist2.GetMean() - 0.5) < 1e-2);
        }

#if false
    [Fact]
    public void DiscretePlusTest()
    {
        int n = 4;
        Variable<bool>[] X = new Variable<bool>[n];
        Variable<int>[] Sum = new Variable<int>[n];
        for (int i = 0; i < n; i++) {
            X[i] = Variable.Bernoulli((double)(i + 1) / n).Named("x" + i);
            Variable prev;
            if (i == 0) {
                prev = Variable.Constant<int>(0);
            } else {
                prev = Sum[i - 1];
            }
            //Sum[i] = prev + X[i];
        }

        InferenceEngine engine = new InferenceEngine();

        Discrete[] sumDist = new Discrete[n];
        for (int i = 0; i < n; i++) {
            sumDist[i] = engine.Infer<Discrete>(Sum[i]);
            //Console.WriteLine(sDist);
        }
        // n=4
        Assert.True(sumDist[0].MaxDiff(new Discrete(3.0 / 4, 1.0 / 4, 0, 0, 0)) < 1e-4);
        Assert.True(sumDist[1].MaxDiff(new Discrete(3.0 / 8, 1.0 / 2, 1.0 / 8, 0, 0)) < 1e-4);
        Assert.True(sumDist[2].MaxDiff(new Discrete(3.0 / 32, 13.0 / 32, 13.0 / 32, 3.0 / 32, 0)) < 1e-4);
        Assert.True(sumDist[3].MaxDiff(new Discrete(0, 3.0 / 32, 13.0 / 32, 13.0 / 32, 3.0 / 32)) < 1e-4);
    }
#endif



        [Fact]
        public void ExpFactorTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            var dPriorVar = Variable.Observed(default(Gaussian)).Named("dPrior");
            Variable<double> d = Variable<double>.Random(dPriorVar).Named("d");
            Variable<double> exp = Variable.Exp(d).Named("exp");
            var expPriorVar = Variable.Observed(default(GammaPower)).Named("expPrior");
            Variable.ConstrainEqualRandom(exp, expPriorVar);
            exp.SetMarginalPrototype(expPriorVar);
            block.CloseBlock();
            InferenceEngine engine = new InferenceEngine();

            var groundTruthArray = new[]
            {
                ((Gaussian.FromMeanAndPrecision(1, 2), GammaPower.Uniform(-1)),
                 (Gaussian.FromNatural(1.9998965132582331, 2.0001832427396851), GammaPower.FromShapeAndRate(2.8220804368812047, 6.3586593528303261, -1.0), 0)),
                ((Gaussian.FromMeanAndPrecision(1, 1000), GammaPower.Uniform(2)),
                 (Gaussian.FromNatural(999.81974838904966, 999.83231543664101), GammaPower.FromShapeAndRate(3999.2421377863202, 2425.3749492614797, 2.0), 0)),
                ((Gaussian.FromMeanAndPrecision(1, 1000), GammaPower.FromShapeAndRate(1, 1, 2)),
                 (Gaussian.FromNatural(999.65208985784875, 1000.9840655225279), GammaPower.FromShapeAndRate(4003.8192381348445, 2429.7519152549512, 2.0), -2.84118821044094)),
                ((Gaussian.FromMeanAndPrecision(1, 1000), GammaPower.FromShapeAndRate(1, 1, -1)),
                 (Gaussian.FromNatural(999.12830752800949, 1000.7521969038098), GammaPower.FromShapeAndRate(1001.5790335640826, 2716.8030087447901, -1.0), -2.36674585661261)),
                ((Gaussian.FromMeanAndPrecision(1, 2), GammaPower.FromShapeAndRate(10, 10, -1)),
                 (Gaussian.FromNatural(1.3463798358521841, 10.918025124120254), GammaPower.FromShapeAndRate(11.479613174194139, 12.42489966931152, -1.0), -1.59084475637629)),
            };

            foreach (var groundTruth in groundTruthArray)
            {
                var (dPrior, expPrior) = groundTruth.Item1;
                var (dExpected, expExpected, evExpected) = groundTruth.Item2;
                dPriorVar.ObservedValue = dPrior;
                expPriorVar.ObservedValue = expPrior;

                var dActual = engine.Infer<Gaussian>(d);
                var expActual = engine.Infer<GammaPower>(exp);
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;

                if (false)
                {
                    // importance sampling
                    double totalWeight = 0;
                    var dEst = new GaussianEstimator();
                    GammaPowerEstimator expEst = new GammaPowerEstimator(expPrior.Power);
                    int numIter = 10_000_000;
                    MeanVarianceAccumulator mvaExp = new MeanVarianceAccumulator();
                    for (int iter = 0; iter < numIter; iter++)
                    {
                        if (iter % 1_000_000 == 0) Trace.WriteLine($"iter = {iter}");
                        double logWeight = 0;
                        double dSample = dPrior.Sample();
                        double expSample = System.Math.Exp(dSample);
                        logWeight += expPrior.GetLogProb(expSample);
                        double weight = System.Math.Exp(logWeight);
                        totalWeight += weight;
                        dEst.Add(dSample, weight);
                        expEst.Add(expSample, weight);
                        mvaExp.Add(expSample, weight);
                    }
                    Trace.WriteLine($"{nameof(totalWeight)} = {totalWeight}");
                    if (totalWeight > 0)
                    {
                        dExpected = dEst.GetDistribution(new Gaussian());
                        evExpected = System.Math.Log(totalWeight / numIter);
                        expExpected = expEst.GetDistribution(new GammaPower());
                        expExpected = GammaPower.FromMeanAndMeanLog(mvaExp.Mean, dExpected.GetMean(), expPrior.Power);
                        Trace.WriteLine($"{Quoter.Quote(dExpected)}, {Quoter.Quote(expExpected)}, {evExpected}");
                    }
                }
                double dError = MomentDiff(dExpected, dActual);
                double expError = MomentDiff(expExpected, expActual);
                double evError = MMath.AbsDiff(evExpected, evActual, 1e-6);
                bool trace = true;
                if (trace)
                {
                    Trace.WriteLine($"d = {dActual} should be {dExpected}, error = {dError}");
                    Trace.WriteLine($"exp = {expActual} should be {expExpected}, error = {expError}");
                    Trace.WriteLine($"evidence = {evActual} should be {evExpected}, error = {evError}");
                }
                Assert.True(dError < 0.002);
                Assert.True(expError < 0.1);
                Assert.True(evError < 5e-4);
            }
        }

        [Fact]
        public void ExpFactorTest1()
        {
            // Generate the data
            Rand.Restart(123);
            int nData = 100;
            double gaussMean = 5.5;
            double gaussPrec = 1.0;
            Gaussian g = new Gaussian(gaussMean, gaussPrec);
            double[] yarr = new double[nData];
            for (int i = 0; i < nData; i++)
            {
                double x = g.Sample();
                double fx = System.Math.Exp(x);
                yarr[i] = Gaussian.Sample(0.0, fx);
            }

            Range r = new Range(nData);
            Variable<double> vMean = Variable.Random<double>(Gaussian.FromMeanAndVariance(0.0, 100.0)).Named("mean");
            VariableArray<double> vX = Variable.Array<double>(r).Named("x");
            VariableArray<double> vFX = Variable.Array<double>(r).Named("fx");
            VariableArray<double> vY = Variable.Array<double>(r).Named("y");
            vX[r] = Variable.GaussianFromMeanAndPrecision(vMean, gaussPrec).ForEach(r);
            vFX[r] = Variable.Exp(vX[r]);
            vY[r] = Variable.GaussianFromMeanAndPrecision(0.0, vFX[r]);
            vY.ObservedValue = yarr;

            Gaussian meanExpected = new Gaussian(gaussMean, 4.0 / nData);
            if (true)
            {
                InferenceEngine ie = new InferenceEngine();
                Gaussian meanActual = (Gaussian)ie.Infer(vMean);
                Console.WriteLine("mean = {0} should be {1}", meanActual, meanExpected);
                Assert.True(meanExpected.MaxDiff(meanActual) < 0.51);
            }
            if (true)
            {
                InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
                Gaussian meanActual = (Gaussian)ie.Infer(vMean);
                Console.WriteLine("VMP mean = {0} should be {1}", meanActual, meanExpected);
                Assert.True(meanExpected.MaxDiff(meanActual) < 416);
            }
        }

        [Fact]
        public void ExpFactorTest2()
        {
            Rand.Restart(0);
            // Generate the data
            int nData = 100;
            double gaussMean = 2.3;
            double gaussPrec = 1.0;
            double gaussMean1 = -1.0;
            Gaussian g = new Gaussian(gaussMean, gaussPrec);
            double[] yarr = new double[nData];
            for (int i = 0; i < nData; i++)
            {
                double x = g.Sample();
                double fx = System.Math.Exp(x);
                yarr[i] = Gaussian.Sample(gaussMean1, fx);
            }

            Range r = new Range(nData);
            Variable<double> vMean = Variable.Random<double>(Gaussian.FromMeanAndVariance(0.0, 100.0)).Named("mean");
            Variable<double> vMean1 = Variable.Random<double>(Gaussian.FromMeanAndVariance(0.0, 100.0)).Named("mean1");
            VariableArray<double> vX = Variable.Array<double>(r).Named("x");
            VariableArray<double> vFX = Variable.Array<double>(r).Named("fx");
            VariableArray<double> vY = Variable.Array<double>(r).Named("y");
            vX[r] = Variable.GaussianFromMeanAndPrecision(vMean, gaussPrec).ForEach(r);
            vFX[r] = Variable.Exp(vX[r]);
            vY[r] = Variable.GaussianFromMeanAndPrecision(vMean1, vFX[r]);
            vY.ObservedValue = yarr;

            //Gamma[] ginit = new Gamma[nData];
            //for (int i=0; i < nData; i++)
            //   ginit[i] = Gamma.FromShapeAndRate(1.0, 0.5);
            //vFX.InitialiseTo(Distribution<double>.Array(ginit));

            InferenceEngine ie = new InferenceEngine();
            //ie.NumberOfIterations = 10;
            Gaussian meanPost = (Gaussian)ie.Infer(vMean);
            Gaussian meanPost1 = (Gaussian)ie.Infer(vMean1);
            Console.WriteLine("True Mean: {0}, Est: {1}", gaussMean, meanPost);
            Console.WriteLine("True Mean1: {0}, Est: {1}", gaussMean1, meanPost1);
            Assert.True(System.Math.Abs(gaussMean - meanPost.GetMean()) < 0.5);
        }

        [Fact]
        public void LogFactorTest()
        {
            var x = Variable.GammaFromShapeAndRate(1, 1).Named("x");
            var y = Variable.Log(x).Named("y");
            var ie = new InferenceEngine();
            Gaussian yPost = ie.Infer<Gaussian>(y);
            Gaussian truth = Gaussian.FromMeanAndVariance(-0.5775281, 1.644557);
            Console.WriteLine("Log(Gamma(1,1))=" + yPost + " should be " + truth);
        }

        internal void LogisticAuxVarTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> w = Variable.GaussianFromMeanAndPrecision(1.2, 0.4).Named("w");
            int C = 2;
            Variable<double>[] exponentialVar = new Variable<double>[C];
            Variable<double>[] logExponentialVar = new Variable<double>[C];
            for (int c = 0; c < C; c++)
            {
                exponentialVar[c] = Variable.GammaFromShapeAndRate(1, 1).Named("e" + c);
                logExponentialVar[c] = Variable.Log(exponentialVar[c]).Named("x" + c);
            }
            var logisticVar = logExponentialVar[1] - logExponentialVar[0];
            logisticVar.Named("l");
            //var logisticVar = Variable.GaussianFromMeanAndPrecision(0, 1).Named("l");
            var s = w - logisticVar;
            s.Named("s");
            Variable<bool> y = Variable.IsPositive(s).Named("y");
            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine();
            ie.ShowFactorGraph = true;

            for (int trial = 0; trial < 2; trial++)
            {
                Gaussian wExpected;
                double evExpected;
                if (trial == 0)
                {
                    y.ObservedValue = true;
                    wExpected = new Gaussian(1.735711683643876, 1.897040876799618);
                    evExpected = System.Math.Log(0.697305276585867);
                }
                else
                {
                    y.ObservedValue = false;
                    wExpected = new Gaussian(-0.034096780812067, 1.704896988818977);
                    evExpected = System.Math.Log(0.302694723413305);
                }
                Console.WriteLine(ie.Infer(exponentialVar[0]));
                Gaussian wActual = ie.Infer<Gaussian>(w);
                double evActual = ie.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("w = {0} should be {1}", wActual, wExpected);
                Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                // Assert.True(wExpected.MaxDiff(wActual) < 1e-4);
                // Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-4) < 1e-4);
            }
        }

        internal void GammaExpTest()
        {
            double meanTrue = 2;
            double precisionTrue = 3;
            int n = 10000;
            double[] data = Util.ArrayInit(n, i => Gamma.Sample(1, System.Math.Exp(-Gaussian.Sample(meanTrue, precisionTrue))));
            Range item = new Range(n).Named("item");
            var obs = Variable.Observed(data, item);
            var mean = Variable.GaussianFromMeanAndPrecision(0, 1e-4);
            mean.AddAttribute(new PointEstimate());
            var precision = Variable.GammaFromShapeAndRate(1, 1);
            precision.AddAttribute(new PointEstimate());
            using (Variable.ForEach(item))
            {
                var state = Variable.GaussianFromMeanAndPrecision(mean, precision);
                var exp = Variable.Exp(state);
                obs[item] = Variable.GammaFromShapeAndRate(1, exp);
            }

            InferenceEngine engine = new InferenceEngine();
            var meanActual = engine.Infer<Gaussian>(mean);
            Console.WriteLine($"mean = {meanActual} should be {meanTrue}");
            var precisionActual = engine.Infer<Gamma>(precision);
            Console.WriteLine($"precision = {precisionActual} should be {precisionTrue}");
        }

        internal void QuadratureParameterTuning()
        {
            ExpOp.QuadratureNodeCount = 11;
            ExpOp.QuadratureIterations = 3;
            ExpOp.QuadratureShift = false;
            PoissonTracker();
        }

        [Fact]
        public void PoissonTracker()
        {
            // compare this against the matlab code in tex/dynep/matlab/test_poisstrack.m
            int[] data = new int[] { 0, 1, 2 };
            //data = new int[] { 0 };
            double transitionPrecision = 10;
            //int[] data = new int[] { 1, 1, 3, 2, 1, 2, 3, 1, 0, 0, 1, 0, 1, 2, 0, 1, 2, 2, 0, 0, 0, 0, 3, 0, 1, 1, 0, 2, 2, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 3, 1, 0, 0, 1, 0, 1, 1, 1, 0, 3, 1, 1, 2, 1, 5, 1, 0, 3, 1, 6, 2, 0, 1, 0 };
            //double transitionPrecision = 100;
            int n = data.Length;
            Variable<double>[] state = new Variable<double>[n];
            Variable<int>[] observation = new Variable<int>[n];
            for (int i = 0; i < n; i++)
            {
                if (i == 0)
                {
                    state[i] = Variable.GaussianFromMeanAndPrecision(0, 0.01);
                }
                else
                {
                    state[i] = Variable.GaussianFromMeanAndPrecision(state[i - 1], transitionPrecision);
                }
                state[i].Name = "state[" + i + "]";
                observation[i] = Variable.Poisson(Variable.Exp(state[i]).Named("exp(state[" + i + "])")).Named("obs[" + i + "]");
                observation[i].ObservedValue = data[i];
            }

            // data={0}:
            // EP = Gaussian(-7.875, 36.55)
            // VMP = Gaussian(-4.151, 8.641e-08) (step=1)
            // VMP = Gaussian(-7.954, 8.364)  (step=0.5)
            // VMP = Gaussian(-7.409, 1.032)  (step=0.75)
            // VMP = Gaussian(-8.047, 11.05)  (step=0.25)
            // VMP = Gaussian(-8.047, 11.05)  (step=rand/2)
            InferenceEngine engine = new InferenceEngine();
            if (false)
            {
                // test convergence of VMP
                engine.ShowProgress = false;
                for (int i = 0; i < 40; i++)
                {
                    engine.Algorithm = new VariationalMessagePassing();
                    engine.NumberOfIterations = 10 * (i + 1);
                    Console.WriteLine(engine.Infer(state[0]));
                }
            }
            if (false)
            {
                // demonstrate VMP for the summer school lecture
                engine.Algorithm = new VariationalMessagePassing();
                engine.NumberOfIterations = 1000;
                // save results
                Matrix results = new Matrix(n, 2);
                for (int i = 0; i < n; i++)
                {
                    Gaussian dist = engine.Infer<Gaussian>(state[i]);
                    results[i, 0] = dist.GetMean();
                    results[i, 1] = dist.GetVariance();
                }
                using (MatlabWriter writer = new MatlabWriter("poisstrack_vmp.mat"))
                {
                    writer.Write("vmp", results);
                }
                return;
                // VMP step=rand/2:
                // Gaussian(-0.1317, 0.09151)
                // Gaussian(-0.04003, 0.04766)
                // Gaussian(0.05, 0.09009)
                // full data, VMP step=rand/2 niter=1000:
                // last state = Gaussian(0.3295, 0.009862)
                // full data, EP:
                // last state = Gaussian(0.3038, 0.07934)
                // exact:
                // last state = Gaussian(0.303757761723916, 0.079964538422247)
            }
            // these are the exact marginals, not the EP marginals, so the answers are expected to differ
            // somewhat.
            Gaussian[] trueDists =
                {
            Gaussian.FromMeanAndVariance(-0.2851,0.4403),
            Gaussian.FromMeanAndVariance(-0.1937,0.4067),
            Gaussian.FromMeanAndVariance(-0.1033,0.4308)
        };
            // result from matlab, using a different number of quadrature nodes.
            Gaussian[] stateExpected =
                {
            Gaussian.FromMeanAndVariance(-0.282473672726657,0.411613840944542),
            Gaussian.FromMeanAndVariance(-0.190973934884677,0.377430434558053),
            Gaussian.FromMeanAndVariance(-0.100461109190121,0.401863830253411)
        };
            double maxError = 0;
            for (int i = 0; i < n; i++)
            {
                Gaussian dist = engine.Infer<Gaussian>(state[i]);
                Console.WriteLine(dist);
                maxError = System.Math.Max(maxError, dist.MaxDiff(stateExpected[i]));
            }
            Console.WriteLine("maxError = {0}", maxError);
            Assert.True(maxError < 1e-3);
        }

        // Fails with improper message exception.  Nothing seems to help except for ForceProper.
        [Fact]
        public void DifficultyAbilityTest()
        {
            Rand.Restart(0);

            int nQuestions = 10;
            int nSubjects = 100;
            int nChoices = 5;
            Gaussian abilityPrior = new Gaussian(0, 1);
            Gaussian difficultyPrior = new Gaussian(0, 1);
            Gamma discriminationPrior = Gamma.FromShapeAndScale(10, 1);
            discriminationPrior = Gamma.PointMass(10.0);

            double[] trueAbility, trueDifficulty, trueDiscrimination;
            int[] trueTrueAnswer;
            int[][] data = Sample(nSubjects, nQuestions, nChoices, abilityPrior, difficultyPrior, discriminationPrior,
              out trueAbility, out trueDifficulty, out trueDiscrimination, out trueTrueAnswer);

            Range question = new Range(nQuestions).Named("question");
            Range subject = new Range(nSubjects).Named("subject");
            Range choice = new Range(nChoices).Named("choice");

            var difficulty = Variable.Array<double>(question).Named("difficulty");
            difficulty[question] = Variable.Random(difficultyPrior).ForEach(question);
            var ability = Variable.Array<double>(subject).Named("ability");
            ability[subject] = Variable.Random(abilityPrior).ForEach(subject);
            var trueAnswer = Variable.Array<int>(question).Named("trueAnswer");
            trueAnswer[question] = Variable.DiscreteUniform(nChoices).ForEach(question);
            var discrimination = Variable.Array<double>(question).Named("discrimination");
            discrimination[question] = Variable.Random(discriminationPrior).ForEach(question);

            if (false)
            {
                var response = Variable.Array(Variable.Array<int>(question), subject).Named("response");
                response.ObservedValue = data;
                using (Variable.ForEach(subject))
                {
                    using (Variable.ForEach(question))
                    {
                        var advantage = (ability[subject] - difficulty[question]).Named("advantage");
                        var advantageNoisy = Variable.GaussianFromMeanAndPrecision(advantage, discrimination[question]).Named("advantageNoisy");
                        var correct = (advantageNoisy > 0).Named("correct");
                        if (true)
                        {
                            using (Variable.If(correct))
                                response[subject][question] = trueAnswer[question];
                            using (Variable.IfNot(correct))
                                response[subject][question] = Variable.DiscreteUniform(nChoices);
                        }
                        else
                        {
                            // this is equivalent to above
                            using (Variable.If(correct))
                            {
                                Variable.ConstrainEqual(response[subject][question], trueAnswer[question]);
                                // this contributes a weight of nChoices to this branch
                                Variable.ConstrainEqualRandom(Variable.Constant(0.0), new TruncatedGaussian(0, 1e10, -0.5 / nChoices, 0.5 / nChoices));
                            }
                        }
                    }
                }
            }
            else
            {
                int nObserved = nSubjects * nQuestions;
                Range obs = new Range(nObserved).Named("obs");
                var subjectOfObs = Variable.Array<int>(obs).Named("subjectOfObs");
                subjectOfObs.ObservedValue = Util.ArrayInit(nObserved, o => o / nQuestions);
                var questionOfObs = Variable.Array<int>(obs).Named("questionOfObs");
                questionOfObs.ObservedValue = Util.ArrayInit(nObserved, o => o % nQuestions);
                var response = Variable.Array<int>(obs).Named("response");
                response.ObservedValue = Util.ArrayInit(nObserved, o => data[subjectOfObs.ObservedValue[o]][questionOfObs.ObservedValue[o]]);
                using (Variable.ForEach(obs))
                {
                    var q = questionOfObs[obs];
                    var advantage = (ability[subjectOfObs[obs]] - difficulty[q]).Named("advantage");
                    var advantageNoisy = Variable.GaussianFromMeanAndPrecision(advantage, discrimination[q]).Named("advantageNoisy");
                    var correct = (advantageNoisy > 0).Named("correct");
                    using (Variable.If(correct))
                        response[obs] = trueAnswer[q];
                    using (Variable.IfNot(correct))
                        response[obs] = Variable.DiscreteUniform(nChoices);
                }
                obs.AddAttribute(new Sequential());
            }

            InferenceEngine engine = new InferenceEngine();
            //subject.AddAttribute(new Sequential());
            //question.AddAttribute(new Sequential());
            //Factors.IsPositiveOp.ForceProper = true;
            var trueAnswerPosterior = engine.Infer<IList<Discrete>>(trueAnswer);
            int numCorrect = 0;
            for (int q = 0; q < nQuestions; q++)
            {
                int bestGuess = trueAnswerPosterior[q].GetMode();
                if (bestGuess == trueTrueAnswer[q])
                    numCorrect++;
            }
            double pctCorrect = 100.0 * numCorrect / nQuestions;
            Console.WriteLine("{0}% TrueAnswers correct", pctCorrect.ToString("f"));
            var difficultyPosterior = engine.Infer<IList<Gaussian>>(difficulty);
            for (int q = 0; q < System.Math.Min(nQuestions, 4); q++)
            {
                Console.WriteLine("difficulty[{0}] = {1} (sampled from {2})", q, difficultyPosterior[q], trueDifficulty[q]);
            }
            var discriminationPosterior = engine.Infer<IList<Gamma>>(discrimination);
            for (int q = 0; q < System.Math.Min(nQuestions, 4); q++)
            {
                Console.WriteLine("discrimination[{0}] = {1} (sampled from {2})", q, discriminationPosterior[q], trueDiscrimination[q]);
            }
            var abilityPosterior = engine.Infer<IList<Gaussian>>(ability);
            for (int s = 0; s < System.Math.Min(nSubjects, 4); s++)
            {
                Console.WriteLine("ability[{0}] = {1} (sampled from {2})", s, abilityPosterior[s], trueAbility[s]);
            }
        }

        public int[][] Sample(int nSubjects, int nQuestions, int nChoices, Gaussian abilityPrior, Gaussian difficultyPrior, Gamma discriminationPrior,
          out double[] ability, out double[] difficulty, out double[] discrimination, out int[] trueAnswer)
        {
            ability = Util.ArrayInit(nSubjects, s => abilityPrior.Sample());
            difficulty = Util.ArrayInit(nQuestions, q => difficultyPrior.Sample());
            discrimination = Util.ArrayInit(nQuestions, q => discriminationPrior.Sample());
            trueAnswer = Util.ArrayInit(nQuestions, q => Rand.Int(nChoices));
            int[][] response = new int[nSubjects][];
            for (int s = 0; s < nSubjects; s++)
            {
                response[s] = new int[nQuestions];
                for (int q = 0; q < nQuestions; q++)
                {
                    double advantage = ability[s] - difficulty[q];
                    double noise = Gaussian.Sample(0, discrimination[q]);
                    bool correct = (advantage > noise);
                    if (correct)
                        response[s][q] = trueAnswer[q];
                    else
                        response[s][q] = Rand.Int(nChoices);
                }
            }
            return response;
        }

        internal void GatedMulticlass()
        {
            MultiStatePRHiddenVariable c = new MultiStatePRHiddenVariable();
            c.Run();
        }

        // from Vincent Tan
        public class MultiStatePRHiddenVariable
        {
            // Some global variables
            public Variable<int>[] nItem;
            public VariableArray<Vector>[] xValues;
            public InferenceEngine engine;

            /// <summary>
            /// This class performs cross-validation to compute the test error on the MultiState Probit Regression model.
            /// Specify the number of classes, features and the number of runs (numRuns-fold cross-validation). 
            /// </summary>        
            public void Run()
            {
                // Reading Data and Checking Missing Data

                // Get the raw data
                int nClass = 4; // Number of Classes              
                int nFeatures = 0; // Number of Features without augmented ones                  

                Vector[][] xData = GetData(new int[] { 10, 10, 10, 10 });

                nFeatures = xData[0][0].Count;
                Console.WriteLine("Number of Features including appended ones = " + nFeatures);

                // Get the number of samples in each class
                int[] nInstances = new int[nClass];
                for (int c = 0; c < nClass; c++)
                {
                    nInstances[c] = xData[c].Length;
                    Console.WriteLine("Number of Training Examples in class " + (c + 1) + " = " + nInstances[c]);
                }

                // Number of instances in each class as ranges
                nItem = new Variable<int>[nClass];
                // Put these in ranges
                Range[] item = new Range[nClass];
                // Put the data instances in this VariableArray
                xValues = new VariableArray<Vector>[nClass];


                // The MultiState Probit Regression Model

                // All the variables and the model      
                Variable<Vector>[] weights = new Variable<Vector>[nClass];
                Variable<Vector>[] veights = new Variable<Vector>[nClass];
                Variable<VectorGaussian>[] weightsPrior = new Variable<VectorGaussian>[nClass];
                Variable<VectorGaussian>[] veightsPrior = new Variable<VectorGaussian>[nClass];
                Variable<double>[] score = new Variable<double>[nClass];
                Variable<double>[] noisyScore = new Variable<double>[nClass];
                Variable<Vector>[] hiddenPrior = new Variable<Vector>[nClass];
                VariableArray<bool>[] hiddenVar = new VariableArray<bool>[nClass];
                VariableArray<double>[] prHidden = new VariableArray<double>[nClass];

                // Need to fix noise prec otherwise EP will return improper message exception                              
                // The data is not linearly separable
                double prec = .01;

                // Initialize the weights          
                for (int c = 0; c < nClass; c++)
                {
                    weightsPrior[c] = Variable.New<VectorGaussian>().Named("weightsPrior_" + c);
                    weights[c] = Variable.Random<Vector, VectorGaussian>(weightsPrior[c]).Named("weights_" + c);

                    veightsPrior[c] = Variable.New<VectorGaussian>().Named("veightsPrior_" + c);
                    veights[c] = Variable.Random<Vector, VectorGaussian>(veightsPrior[c]).Named("veights_" + c);
                }

                // Describe the model using the following loop
                for (int c = 0; c < nClass; c++)
                {
                    nItem[c] = Variable.New<int>().Named("nItem_" + c);
                    item[c] = new Range(nItem[c]).Named("item_" + c);

                    xValues[c] = Variable.Array<Vector>(item[c]).Named("xValues_" + c);
                    prHidden[c] = Variable.Array<double>(item[c]).Named("prHidden_" + c);
                    prHidden[c][item[c]] = Variable.Beta(1, 1 - .01 * c).ForEach(item[c]); // symmetry breaking prior
                    hiddenVar[c] = Variable.Array<bool>(item[c]).Named("hiddenVar_" + c);
                    hiddenVar[c][item[c]] = Variable.Bernoulli(prHidden[c][item[c]]);

                    using (Variable.ForEach(item[c]))
                    {
                        using (Variable.If(hiddenVar[c][item[c]]))
                        {
                            score = ComputeClassScores(weights, xValues[c][item[c]]);
                            noisyScore = AddNoiseToScore(score, prec);
                            ConstrainArgMax(c, noisyScore);
                        }
                        using (Variable.IfNot(hiddenVar[c][item[c]]))
                        {
                            score = ComputeClassScores(veights, xValues[c][item[c]]);
                            noisyScore = AddNoiseToScore(score, prec);
                            ConstrainArgMax(c, noisyScore);
                        }

                    }
                }

                // Inference and Display of Results
                for (int c = 0; c < nClass; c++)
                {
                    // Set the prior on the weights                 
                    weightsPrior[c].ObservedValue = new VectorGaussian(
                      Vector.Zero(nFeatures),
                      PositiveDefiniteMatrix.IdentityScaledBy(nFeatures, 1));

                    // Set the prior on the weights                 
                    veightsPrior[c].ObservedValue = new VectorGaussian(
                      Vector.Zero(nFeatures),
                      PositiveDefiniteMatrix.IdentityScaledBy(nFeatures, 1));

                    // Set the observed data
                    xValues[c].ObservedValue = xData[c];
                    nItem[c].ObservedValue = xData[c].Length;
                }

                // Instantiate the Inference engine
                engine = new InferenceEngine(new ExpectationPropagation());
                engine.NumberOfIterations = 100;

                // Infer the weights
                VectorGaussian[] wInferred = new VectorGaussian[nClass];
                VectorGaussian[] vInferred = new VectorGaussian[nClass];
                DistributionArray<Bernoulli>[] hiddenMarg = new DistributionArray<Bernoulli>[nClass];

                for (int c = 0; c < nClass; c++)
                {
                    wInferred[c] = (VectorGaussian)engine.Infer(weights[c]);
                    Console.WriteLine(wInferred[c]);

                    vInferred[c] = (VectorGaussian)engine.Infer(veights[c]);
                    Console.WriteLine(vInferred[c]);

                    hiddenMarg[c] = (DistributionArray<Bernoulli>)engine.Infer(hiddenVar[c]);
                    Console.WriteLine(hiddenMarg[c]);
                }
            }

            // Create some artificial training data 
            private Vector[][] GetData(int[] numInstances)
            {
                int nClass = numInstances.Length;
                Vector[][] xData = new Vector[nClass][];
                double precision = 100;
                double x1, x2, x3;

                double[][] means = new double[][]
                    {
                        new double[] {-1.0, -1.0},
                                new double[] { -1.0, 1.0 },
                                new double[] { 1.0, 1.0},
                        new double[] {1.0, -1.0}
                    };

                for (int c = 0; c < nClass; c++)
                {
                    xData[c] = new Vector[numInstances[c]];
                    for (int i = 0; i < xData[c].Length; i++)
                    {
                        if (i < xData[c].Length / 2)
                        {
                            x1 = Gaussian.Sample(means[c][0], precision);
                            x2 = Gaussian.Sample(means[c][1], precision);
                            x3 = Gaussian.Sample(1, precision);
                        }
                        else
                        {
                            x1 = Gaussian.Sample(means[c][0], precision);
                            x2 = Gaussian.Sample(means[c][1], precision);
                            x3 = Gaussian.Sample(-1, precision);
                        }
                        xData[c][i] = Vector.FromArray(new double[] { x1, x2, x3, 1 });
                        //xData[c][i] = Vector.FromArray(new double[] { x1, x2 });
                    }
                }

                return xData;
            }

            /// <summary>
            /// Method to add noise to the inner product score.
            /// This is to ensure that EP updates are ok.
            /// </summary>
            /// <param name="score">The scores.</param>
            /// <param name="prec">Precision of the noise to be added.</param>         
            private Variable<double>[] AddNoiseToScore(Variable<double>[] score, double prec)
            {
                int nClass = score.Length;
                Variable<double>[] noisyScore = new Variable<double>[nClass];
                for (int c = 0; c < score.Length; c++)
                {
                    noisyScore[c] = Variable.GaussianFromMeanAndPrecision(score[c], prec);
                }

                return noisyScore;
            }

            /// <summary>
            /// Method to take inner product of weights with data values.         
            /// </summary>
            /// <param name="w">The weights stores in a dot net array of length nClass.</param>
            /// <param name="xValues">The data values as a Vector.</param>
            private Variable<double>[] ComputeClassScores(Variable<Vector>[] w, Variable<Vector> xValues)
            {
                int nClass = w.Length;
                Variable<double>[] score = new Variable<double>[nClass];
                for (int c = 0; c < nClass; c++)
                {
                    score[c] = Variable.InnerProduct(w[c], xValues);
                }
                return score;
            }

            /// <summary>
            /// A Factor to constrain argmax by doing pairwise comparisons between values.            
            /// </summary>
            /// <param name="argmax">The index that maximizes the second variable.</param>
            /// <param name="score">A variable double array of score values.</param>
            private void ConstrainArgMax(int argmax, Variable<double>[] score)
            {
                for (int c = 0; c < score.Length; c++)
                {
                    if (c != argmax)
                        Variable.ConstrainPositive(score[argmax] - score[c]);
                }
            }

            /// <summary>
            /// A Factor to constrain a given int to be the maximum.
            /// </summary>
            /// <param name="ytrain">The index to be the maximum.</param>
            /// <param name="score">A variable double array of score values.</param>
            /// <param name="nClass">Total number of classes.</param>
            private void ConstrainMaximum(Variable<int> ytrain, Variable<double>[] score, int nClass)
            {
                for (int c = 0; c < nClass; c++)
                {
                    using (Variable.Case(ytrain, c))
                    {
                        ConstrainArgMax(c, score);
                    }
                }
            }
        }
    }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

}

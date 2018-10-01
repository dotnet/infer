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
using Assert = Xunit.Assert;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Serialization;

namespace Microsoft.ML.Probabilistic.Tests
{
#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

    public class EpTests
    {
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
            Variable<Vector> x = Variable.VectorTimesScalar(a, b).Named("x");
            Vector xMean = Vector.FromArray(7.7, 8.8);
            PositiveDefiniteMatrix xVariance = new PositiveDefiniteMatrix(new double[,] { { 9, 1 }, { 1, 9 } });
            VectorGaussian xLike = new VectorGaussian(xMean, xVariance);
            Variable.ConstrainEqualRandom(x, xLike);

            InferenceEngine engine = new InferenceEngine();
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
            Assert.Equal(bExpected, bActual, 1);
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
            double xMultiplier = 5;
            var xData = Util.ArrayInit(n, i => Gaussian.Sample(xMultiplier * hSample[i], xPrecisionTrue));
            var yData = Util.ArrayInit(n, i => Gaussian.Sample(hSample[i], yPrecisionTrue));
            x.ObservedValue = xData;
            y.ObservedValue = yData;

            // N(x; ah, vx) N(h; mh, vh) = N(h; mh + k*(x - a*mh), (1-ka)vh)
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
            var baseSkill = Variable.Array<double>(player);
            baseSkill[player] = Variable.GaussianFromMeanAndPrecision(0, basePrecision).ForEach(player);
            Range mode = new Range(2);
            var offsetPrecisionRate = Variable.GammaFromShapeAndRate(1, 1);
            var offsetPrecision = Variable.Array<double>(mode);
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
                Range game = new Range(100);
                using (Variable.ForEach(game))
                {
                    var skillInGame = Variable.Array(Variable.Array<double>(mode), player);
                    skillInGame[player][mode] = baseSkill[player] + offset[player][mode];
                    Variable.ConstrainEqualRandom(skillInGame[player][mode], data[player][mode]);
                }
                //Variable.ConstrainEqualRandom(skill[player][mode], data[player][mode]);

                data.ObservedValue = Util.ArrayInit(player.SizeAsInt, i =>
                    Util.ArrayInit(mode.SizeAsInt, j =>
                        Gaussian.FromMeanAndPrecision(trueSkills[i][j], 1)));
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
            engine.Compiler.GivePriorityTo(typeof(VariablePointOp_Mean<>));
            engine.Compiler.GivePriorityTo(typeof(GammaFromShapeAndRateOp_Laplace));
            var baseSkillActual = engine.Infer<IList<Gaussian>>(baseSkill);
            for (int i = 0; i < 3; i++)
            {
                Console.WriteLine("baseSkill[{0}] = {1} should be {2}", i, baseSkillActual[i], trueBaseSkills[i]);
            }
            Console.WriteLine("basePrecision = {0} should be {1}", engine.Infer(basePrecision), trueBasePrecision);
            Console.WriteLine(StringUtil.JoinColumns(engine.Infer(offsetPrecision), " should be ", StringUtil.VerboseToString(trueOffsetPrecisions)));
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

        public Gaussian[] IndexOfMaximumObservedIndexExplicit(int N, int index, out double logEvidence)
        {
            var ev = Variable.Bernoulli(0.5).Named("ev");
            var block = Variable.If(ev);
            var x = Enumerable.Range(0, N).Select(o => Variable.GaussianFromMeanAndPrecision(0, 1)).ToArray();
            for (int i = 0; i < N; i++)
            {
                if (i != index)
                    Variable.ConstrainPositive(x[index] - x[i]);
            }
            block.CloseBlock();
            var ie = new InferenceEngine();
            //ie.NumberOfIterations = 2;
            var toInfer = x.Select(o => (IVariable)o).ToList();
            toInfer.Add(ev);
            var ca = ie.GetCompiledInferenceAlgorithm(toInfer.ToArray());
            ca.Execute(10);
            logEvidence = ca.Marginal<Bernoulli>(ev.NameInGeneratedCode).LogOdds;
            return x.Select(o => ca.Marginal<Gaussian>(o.NameInGeneratedCode)).ToArray();
        }

        public Gaussian[] IndexOfMaximumExplicit(Discrete y, out double logEvidence)
        {
            int N = y.Dimension;
            var ev = Variable.Bernoulli(0.5).Named("ev");
            var block = Variable.If(ev);
            var x = Enumerable.Range(0, N).Select(o => Variable.GaussianFromMeanAndPrecision(0, 1)).ToArray();
            var yVar = Variable<int>.Random(y).Named("y");
            for (int index = 0; index < N; index++)
            {
                using (Variable.Case(yVar, index))
                {
                    for (int i = 0; i < N; i++)
                    {
                        if (i != index)
                            Variable.ConstrainPositive(x[index] - x[i]);
                    }
                }
            }
            block.CloseBlock();
            var ie = new InferenceEngine();
            //ie.NumberOfIterations = 2;
            var toInfer = x.Select(o => (IVariable)o).ToList();
            toInfer.Add(ev);
            var ca = ie.GetCompiledInferenceAlgorithm(toInfer.ToArray());
            ca.Execute(10);
            logEvidence = ca.Marginal<Bernoulli>(ev.NameInGeneratedCode).LogOdds;
            return x.Select(o => ca.Marginal<Gaussian>(o.NameInGeneratedCode)).ToArray();
        }

        public Gaussian[] IndexOfMaximumObservedIndexFactor(int N, int index, out double logEvidence)
        {
            var n = new Range(N);
            var ev = Variable.Bernoulli(0.5);
            var block = Variable.If(ev);
            var x = Variable.Array<double>(n);
            x[n] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(n);
            var p = IndexOfMaximum(x);
            p.ObservedValue = index;
            block.CloseBlock();
            var ie = new InferenceEngine();
            //ie.NumberOfIterations = 2;
            logEvidence = ie.Infer<Bernoulli>(ev).LogOdds;
            return ie.Infer<Gaussian[]>(x);
        }

        [Fact]
        public void IndexOfMaximumObservedIndexTest()
        {
            int N = 3;
            int index = 0;
            double expEv, facEv;
            var exp = IndexOfMaximumObservedIndexExplicit(N, index, out expEv);
            var fac = IndexOfMaximumObservedIndexFactor(N, index, out facEv);
            for (int i = 0; i < N; i++)
            {
                Console.WriteLine("exp: " + exp[i] + " fac: " + fac[i]);
                Assert.True(exp[i].MaxDiff(fac[i]) < 1e-8);
            }
            Assert.True(MMath.AbsDiff(expEv, facEv) < 1e-8);
        }


        public Gaussian[] IndexOfMaximumFactorGate(Discrete y, out double logEvidence)
        {
            var ev = Variable.Bernoulli(0.5).Named("ev");
            int N = y.Dimension;
            var n = new Range(N).Named("n");
            var block = Variable.If(ev);
            var x = Variable.Array<double>(n).Named("x");
            x[n] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(n);
            var yVar = Variable<int>.Random(y).Named("y");
            for (int index = 0; index < N; index++)
            {
                using (Variable.Case(yVar, index))
                {
                    //var temp = Variable.Observed<int>(index).Named("temp"+index) ;
                    //temp.SetTo(Variable<int>.Factor(MMath.IndexOfMaximumDouble, x).Named("fac"+index));
                    var temp = IndexOfMaximum(x).Named("temp" + index);
                    temp.ObservedValue = index;
                }
            }
            block.CloseBlock();
            var ie = new InferenceEngine();
            ie.ModelName = "FactorGate";
            ie.OptimiseForVariables = new List<IVariable>() { x, ev };
            logEvidence = ie.Infer<Bernoulli>(ev).LogOdds;
            return ie.Infer<Gaussian[]>(x);
        }

        public Gaussian[] IndexOfMaximumFactorGate2(Discrete y, out double logEvidence)
        {
            var ev = Variable.Bernoulli(0.5).Named("ev");
            int N = y.Dimension;
            var n = new Range(N).Named("n");
            var block = Variable.If(ev);
            var x = Variable.Array<double>(n).Named("x");
            x[n] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(n);
            var yVar = Variable<int>.Random(y).Named("y");
            var indices = Variable.Observed(new int[] { 0, 1, 2 }, n);
            yVar.SetValueRange(n);
            using (Variable.Switch(yVar))
            {
                indices[yVar] = IndexOfMaximum(x).Named("temp");
            }
            block.CloseBlock();
            var ie = new InferenceEngine();
            //ie.NumberOfIterations = 2;
            logEvidence = ie.Infer<Bernoulli>(ev).LogOdds;
            return ie.Infer<Gaussian[]>(x);
        }

        // This factor is not included as a Variable.IndexOfMaximum shortcut 
        // because you generally get better results with the expanded version, when part of a larger model.
        // Eventually the factor should be deprecated.
        public static Variable<int> IndexOfMaximum(VariableArray<double> array)
        {
            var p = Variable<int>.Factor(MMath.IndexOfMaximumDouble, array).Named("p");
            p.SetValueRange(array.Range);
            return p;
        }

        public Gaussian[] IndexOfMaximumFactorCA(Discrete y, out double logEvidence)
        {
            int N = y.Dimension;
            var n = new Range(N);
            var ev = Variable.Bernoulli(0.5).Named("ev");
            var block = Variable.If(ev);
            var x = Variable.Array<double>(n).Named("x");
            x[n] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(n);
            var p = IndexOfMaximum(x);
            Variable.ConstrainEqualRandom(p, y);
            block.CloseBlock();
            var ie = new InferenceEngine();
            ie.ModelName = "IndexOfMaximumCA";
            //ie.NumberOfIterations = 2;
            var toinfer = new List<IVariable>();
            toinfer.Add(x);
            toinfer.Add(ev);
            ie.OptimiseForVariables = toinfer;

            var ca = ie.GetCompiledInferenceAlgorithm(x, ev);
            ca.Reset();
            Gaussian[] xPost = null;
            logEvidence = 0;
            for (int i = 0; i < 10; i++)
            {
                ca.Update(1);
                logEvidence = ca.Marginal<Bernoulli>(ev.NameInGeneratedCode).LogOdds;
                xPost = ca.Marginal<Gaussian[]>(x.NameInGeneratedCode);
            }
            return xPost;
        }

        public Gaussian[] IndexOfMaximumFactorIE(Discrete y, out double logEvidence)
        {
            int N = y.Dimension;
            var n = new Range(N);
            var ev = Variable.Bernoulli(0.5).Named("ev");
            var block = Variable.If(ev);
            var x = Variable.Array<double>(n).Named("x");
            x[n] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(n);
            var p = IndexOfMaximum(x).Named("p");
            Variable.ConstrainEqualRandom(p, y);
            block.CloseBlock();
            var ie = new InferenceEngine();
            ie.ModelName = "IndexOfMaximumIE";
            var toinfer = new List<IVariable>();
            toinfer.Add(x);
            toinfer.Add(ev);
            ie.OptimiseForVariables = toinfer;
            logEvidence = ie.Infer<Bernoulli>(ev).LogOdds;
            return ie.Infer<Gaussian[]>(x);
        }

        [Fact]
        public void IndexOfMaximumTest()
        {
            var y = new Discrete(0.1, 0.4, 0.5);
            double expEv, gateEv, facEv;
            Console.WriteLine("explicit");
            var exp = IndexOfMaximumExplicit(y, out expEv);
            Console.WriteLine("gate");
            var gate = IndexOfMaximumFactorGate(y, out gateEv);
            Console.WriteLine("compiled alg");
            var facCA = IndexOfMaximumFactorCA(y, out facEv);
            Console.WriteLine("engine");
            var facIE = IndexOfMaximumFactorIE(y, out facEv);
            for (int i = 0; i < y.Dimension; i++)
            {
                Console.WriteLine("exp: " + exp[i] + " facCA: " + facCA[i] + " fac: " + facIE[i] + " gate: " + gate[i]);
                Assert.True(exp[i].MaxDiff(facCA[i]) < 1e-8);
                Assert.True(exp[i].MaxDiff(gate[i]) < 1e-8);
                Assert.True(exp[i].MaxDiff(facIE[i]) < 1e-8);
            }
            Assert.True(MMath.AbsDiff(expEv, facEv) < 1e-8);
            Assert.True(MMath.AbsDiff(expEv, gateEv) < 1e-8);
        }

        [Fact]
        public void IndexOfMaximumFastTest()
        {
            int n = 5;
            Range item = new Range(n).Named("item");
            var priors = Variable<Gaussian>.Array(item);
            priors.ObservedValue = Util.ArrayInit(n, i => Gaussian.FromMeanAndVariance(i * 0.5, i));
            var x = Variable.Array<double>(item).Named("x");
            x[item] = Variable<double>.Random(priors[item]);
            var y = Variable<int>.Factor(MMath.IndexOfMaximumDouble, x);
            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;
            string format = "f4";
            var yActual = engine.Infer<Discrete>(y);
            Console.WriteLine("Quadratic: {0}", yActual.ToString(format));

            // Monte Carlo estimate
            Rand.Restart(0);
            DiscreteEstimator est = new DiscreteEstimator(n);
            for (int iter = 0; iter < 100000; iter++)
            {
                double[] samples = Util.ArrayInit(n, i => priors.ObservedValue[i].Sample());
                int argmax = MMath.IndexOfMaximumDouble(samples);
                est.Add(argmax);
            }
            var yExpected = est.GetDistribution(Discrete.Uniform(n));
            Console.WriteLine("Sampling:  {0}", yExpected.ToString(format));
            Assert.True(yExpected.MaxDiff(yActual) < 1e-2);

            engine.Compiler.GivePriorityTo(typeof(IndexOfMaximumOp_Fast));
            yActual = engine.Infer<Discrete>(y);
            Console.WriteLine("Linear:    {0}", yActual.ToString(format));
            Assert.True(yExpected.MaxDiff(yActual) < 1e-2);

            if (false)
            {
                var yPost2 = IndexOfMaximumOp_Fast.IndexOfMaximumDoubleAverageConditional(priors.ObservedValue, Discrete.Uniform(n));
                Console.WriteLine(yPost2);
                var yPost3 = IndexOfMaximumOp_Fast.IndexOfMaximumDoubleAverageConditional2(priors.ObservedValue, Discrete.Uniform(n));
                Console.WriteLine(yPost3);
            }
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

        [Fact]
        [Trait("Category", "OpenBug")]
        public void VarianceGammaTimesGaussianMomentsTest()
        {
            double a = 1457.651420562946;
            double m = -55.3161989949422;
            double v = 117.31286883057773;
            double mu, vu;
            GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianMoments5(a, m, v, out mu, out vu);
            Assert.False(double.IsNaN(mu));
            Assert.False(double.IsNaN(vu));
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void VarianceGammaTimesGaussianIntegralTest()
        {
            double logZ, mu, vu;
            GaussianFromMeanAndVarianceOp.LaplacianTimesGaussianMoments(-100, 1, out logZ, out mu, out vu);
            Console.WriteLine(logZ);
            //GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianMoments5(1, 1000, 1, out mu, out vu);
            Console.WriteLine(mu);
            Console.WriteLine(vu);
            double[][] vgmoments = GaussianFromMeanAndVarianceOp.NormalVGMomentRatios(10, 1, -6, 1);
            for (int i = 0; i < 10; i++)
            {
                double f = MMath.NormalCdfMomentRatio(i, -6) * MMath.Gamma(i + 1);
                Console.WriteLine("R({0}) = {1}, Zp = {2}", i, f, vgmoments[0][i]);
            }

            // true values computed by tex/factors/matlab/test_variance_gamma2.m
            double scale = 2 * System.Math.Exp(-Gaussian.GetLogProb(2 / System.Math.Sqrt(3), 0, 1));
            Assert.True(MMath.AbsDiff(0.117700554409044, GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(1, 2, 3) / scale, 1e-20) < 1e-5);
            Assert.True(MMath.AbsDiff(0.112933034747473, GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(2, 2, 3) / scale, 1e-20) < 1e-5);
            Assert.True(MMath.AbsDiff(0.117331854901251, GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(1.1, 2, 3) / scale, 1e-20) < 1e-5);
            Assert.True(MMath.AbsDiff(0.115563913123152, GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(1.5, 2, 3) / scale, 1e-20) < 2e-5);

            scale = 2 * System.Math.Exp(-Gaussian.GetLogProb(20 / System.Math.Sqrt(0.3), 0, 1));
            Assert.True(MMath.AbsDiff(1.197359429038085e-009, GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(1, 20, 0.3) / scale, 1e-20) < 1e-5);
            Assert.True(MMath.AbsDiff(1.239267009054433e-008, GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(2, 20, 0.3) / scale, 1e-20) < 1e-5);
            Assert.True(MMath.AbsDiff(1.586340098271600e-009, GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(1.1, 20, 0.3) / scale, 1e-20) < 1e-4);
            Assert.True(MMath.AbsDiff(4.319412089896069e-009, GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(1.5, 20, 0.3) / scale, 1e-20) < 1e-4);

            scale = 2 * System.Math.Exp(-Gaussian.GetLogProb(40 / System.Math.Sqrt(0.3), 0, 1));
            scale = 2 * System.Math.Exp(40 - 0.3 / 2);
            Assert.True(MMath.AbsDiff(2.467941724509690e-018, GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(1, 40, 0.3) / scale, 1e-20) < 1e-5);
            Assert.True(MMath.AbsDiff(5.022261409377230e-017, GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(2, 40, 0.3) / scale, 1e-20) < 1e-5);
            Assert.True(MMath.AbsDiff(3.502361147666615e-018, GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(1.1, 40, 0.3) / scale, 1e-20) < 1e-4);
            Assert.True(MMath.AbsDiff(1.252310352551344e-017, GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(1.5, 40, 0.3) / scale, 1e-20) < 1e-4);

            Assert.True(GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(1.138, 33.4, 0.187) > 0);
            Assert.True(GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(1.138, -33.4, 0.187) > 0);
            Assert.True(GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(1.138, 58.25, 0.187) > 0);
            Assert.True(GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(1.138, 50, 0.187) > 0);
            Assert.True(GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(1.138, 100, 0.187) > 0);
        }
        internal void HeteroscedasticGPR()
        {
            // This model is based on the paper "Most Likely Heteroscedastic Gaussian Process Regression" by Kersting et al, ICML 2007
            // Silverman's motorcycle benchmark dataset
            double[] inputs = new double[]
                {
                2.4, 2.6, 3.2, 3.6, 4, 6.2, 6.6, 6.8, 7.8, 8.199999999999999, 8.800000000000001, 8.800000000000001,
                9.6, 10, 10.2, 10.6, 11, 11.4, 13.2, 13.6, 13.8, 14.6, 14.6, 14.6, 14.6, 14.6, 14.6, 14.8, 15.4, 15.4,
                15.4, 15.4, 15.6, 15.6, 15.8, 15.8, 16, 16, 16.2, 16.2, 16.2, 16.4, 16.4, 16.6, 16.8, 16.8, 16.8, 17.6,
                17.6, 17.6, 17.6, 17.8, 17.8, 18.6, 18.6, 19.2, 19.4, 19.4, 19.6, 20.2, 20.4, 21.2, 21.4, 21.8, 22, 23.2,
                23.4, 24, 24.2, 24.2, 24.6, 25, 25, 25.4, 25.4, 25.6, 26, 26.2, 26.2, 26.4, 27, 27.2, 27.2, 27.2, 27.6,
                28.2, 28.4, 28.4, 28.6, 29.4, 30.2, 31, 31.2, 32, 32, 32.8, 33.4, 33.8, 34.4, 34.8, 35.2, 35.2, 35.4, 35.6,
                35.6, 36.2, 36.2, 38, 38, 39.2, 39.4, 40, 40.4, 41.6, 41.6, 42.4, 42.8, 42.8, 43, 44, 44.4, 45, 46.6, 47.8,
                    47.8, 48.8, 50.6, 52, 53.2, 55, 55, 55.4, 57.6
                };
            double[] outputs = new double[]
                {
                0, -1.3, -2.7, 0, -2.7, -2.7, -2.7, -1.3, -2.7, -2.7, -1.3, -2.7, -2.7, -2.7, -5.4,
                -2.7, -5.4, 0, -2.7, -2.7, 0, -13.3, -5.4, -5.4, -9.300000000000001, -16, -22.8, -2.7, -22.8, -32.1, -53.5,
                -54.9, -40.2, -21.5, -21.5, -50.8, -42.9, -26.8, -21.5, -50.8, -61.7, -5.4, -80.40000000000001, -59, -71,
                -91.09999999999999, -77.7, -37.5, -85.59999999999999, -123.1, -101.9, -99.09999999999999, -104.4, -112.5,
                -50.8, -123.1, -85.59999999999999, -72.3, -127.2, -123.1, -117.9, -134, -101.9, -108.4, -123.1, -123.1, -128.5,
                -112.5, -95.09999999999999, -81.8, -53.5, -64.40000000000001, -57.6, -72.3, -44.3, -26.8, -5.4, -107.1, -21.5,
                -65.59999999999999, -16, -45.6, -24.2, 9.5, 4, 12, -21.5, 37.5, 46.9, -17.4, 36.2, 75, 8.1, 54.9, 48.2, 46.9,
                16, 45.6, 1.3, 75, -16, -54.9, 69.59999999999999, 34.8, 32.1, -37.5, 22.8, 46.9, 10.7, 5.4, -1.3, -21.5, -13.3,
                    30.8, -10.7, 29.4, 0, -10.7, 14.7, -1.3, 0, 10.7, 10.7, -26.8, -14.7, -13.3, 0, 10.7, -14.7, -2.7, 10.7, -2.7, 10.7
                };
            Range j = new Range(inputs.Length);
            Vector[] inputsVec = Util.ArrayInit(inputs.Length, i => Vector.FromArray(inputs[i]));
            VariableArray<Vector> x = Variable.Observed(inputsVec, j).Named("x");
            VariableArray<double> y = Variable.Observed(outputs, j).Named("y");
            // Set up the GP prior, which will be filled in later
            Variable<SparseGP> prior = Variable.New<SparseGP>().Named("prior");
            Variable<SparseGP> prior2 = Variable.New<SparseGP>().Named("prior2");

            // The sparse GP variable - a distribution over functions
            Variable<IFunction> f = Variable<IFunction>.Random(prior).Named("f");
            Variable<IFunction> r = Variable<IFunction>.Random(prior2).Named("r");

            Variable<double> mean = Variable.FunctionEvaluate(f, x[j]).Named("mean");
            Variable<double> logVariance = Variable.FunctionEvaluate(r, x[j]).Named("logVariance");
            Variable<double> variance = Variable.Exp(logVariance);
            y[j] = Variable.GaussianFromMeanAndVariance(mean, variance);


            InferenceEngine engine = new InferenceEngine();
            GaussianProcess gp = new GaussianProcess(new ConstantFunction(0), new SquaredExponential(0));
            GaussianProcess gp2 = new GaussianProcess(new ConstantFunction(0), new SquaredExponential(0));
            // Fill in the sparse GP prior
            //Vector[] basis = Util.ArrayInit(120, i => Vector.FromArray(0.5*i));
            Vector[] basis = Util.ArrayInit(60, i => Vector.FromArray(1.0 * i));
            prior.ObservedValue = new SparseGP(new SparseGPFixed(gp, basis));
            prior2.ObservedValue = new SparseGP(new SparseGPFixed(gp2, basis));
            // Infer the posterior Sparse GP
            SparseGP sgp = engine.Infer<SparseGP>(f);

            // Check that training set is classified correctly
            Console.WriteLine();
            Console.WriteLine("Predictions on training set:");
            for (int i = 0; i < outputs.Length; i++)
            {
                Gaussian post = sgp.Marginal(inputsVec[i]);
                //double postMean = post.GetMean();
                Console.WriteLine("f({0}) = {1}", inputs[i], post);
            }
            //TODO: change path for cross platform using
            using (MatlabWriter writer = new MatlabWriter(@"..\..\HGPR.mat"))
            {
                int n = outputs.Length;
                double[] m = new double[n];
                double[] s = new double[n];
                for (int i = 0; i < n; i++)
                {
                    Gaussian post = sgp.Marginal(inputsVec[i]);
                    double mi, vi;
                    post.GetMeanAndVariance(out mi, out vi);
                    m[i] = mi;
                    s[i] = System.Math.Sqrt(vi);
                }
                writer.Write("mean", m);
                writer.Write("std", s);
            }
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

        public static bool[] IsPositive(Vector vector)
        {
            return Util.ArrayInit(vector.Count, i => vector[i] > 0);
        }

        public class VectorIsPositiveEP
        {
            IGeneratedAlgorithm gen;
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
        public void GammaProductRRRTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> shape = Variable.Observed(2.5).Named("shape");
            Gamma scalePrior = Gamma.FromShapeAndRate(3, 4);
            Variable<double> scale = Variable<double>.Random(scalePrior).Named("scale");
            Variable<double> y = Variable.GammaFromShapeAndRate(shape, 1).Named("y");
            Variable<double> x = (y * scale).Named("x");
            Gamma xPrior = Gamma.FromShapeAndRate(5, 6);
            Variable.ConstrainEqualRandom(x, xPrior);
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            Gamma scaleActual = engine.Infer<Gamma>(scale);
            Gamma yActual = engine.Infer<Gamma>(y);
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Gamma scaleExpected = new Gamma(3.335, 0.1678);
            Gamma yExpected = new Gamma(3.021, 0.5778);
            double evExpected = -0.919181055678219;

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
                    double ySample = yPrior.Sample();
                    double logWeight = xPrior.GetLogProb(ySample * scaleSample);
                    double weight = System.Math.Exp(logWeight);
                    totalWeight += weight;
                    scaleEst.Add(scaleSample, weight);
                    yEst.Add(ySample, weight);
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

        /// <summary>
        /// Test modified EP updates
        /// </summary>
        internal void StudentIsPositiveTest()
        {
            double shape = 1;
            Gamma precPrior = Gamma.FromShapeAndRate(shape, shape);
            // mean=-1 causes improper messages
            double mean = -1;
            double evExpected;
            Gaussian xExpected = StudentIsPositiveExact(mean, precPrior, out evExpected);

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> prec = Variable.Random(precPrior).Named("prec");
            Variable<double> x = Variable.GaussianFromMeanAndPrecision(mean, prec).Named("x");
            Variable.ConstrainPositive(x);
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            x.AddAttribute(new TraceMessages());
            //x.InitialiseTo(Gaussian.FromMeanAndVariance(-3.719, 4.836));
            //GaussianOp.ForceProper = false;
            //GaussianOp.modified = true;
            //engine.Compiler.GivePriorityTo(typeof(GaussianOp_Laplace));
            //engine.Compiler.GivePriorityTo(typeof(GaussianOp_Slow));
            GaussianOp_Laplace.modified = true;
            GaussianOp_Laplace.modified2 = true;
            Console.WriteLine("x = {0} should be {1}", engine.Infer(x), xExpected);
            double evActual = System.Math.Exp(engine.Infer<Bernoulli>(evidence).LogOdds);
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
        }

        /// <summary>
        /// Test a difficult case
        /// </summary>
        internal void StudentIsPositiveTest5()
        {
            // depending on the exact setting of priors, the messages will alternate between proper and improper
            Gamma precPrior = new Gamma(5, 0.2);
            // mean=-1 causes improper messages
            var mean = Variable.Random(new Gaussian(-0.9, 0.25)).Named("mean");
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> prec = Variable.Random(precPrior).Named("prec");
            Variable<double> x = Variable.GaussianFromMeanAndPrecision(mean, prec).Named("x");
            Variable<bool> y = Variable.IsPositive(x);
            Variable.ConstrainEqualRandom(y, new Bernoulli(0.8889));
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            x.AddAttribute(new TraceMessages());
            Console.WriteLine(engine.Infer(x));
            //Console.WriteLine("x = {0} should be {1}", engine.Infer(x), xExpected);
            double evActual = System.Math.Exp(engine.Infer<Bernoulli>(evidence).LogOdds);
            //Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
        }

        internal void StudentIsPositiveTest4()
        {
            double shape = 1;
            Gamma precPrior = Gamma.FromShapeAndRate(shape, shape);
            // mean=-1 causes improper messages
            double mean = -1;
            Gaussian meanPrior = Gaussian.PointMass(mean);
            double evExpected;
            Gaussian xExpected = StudentIsPositiveExact(mean, precPrior, out evExpected);

            GaussianOp.ForceProper = false;
            GaussianOp_Laplace.modified = true;
            GaussianOp_Laplace.modified2 = true;
            Gaussian xF = Gaussian.Uniform();
            Gaussian xB = Gaussian.Uniform();
            Gamma q = GaussianOp_Laplace.QInit();
            double r0 = 0.38;
            r0 = 0.1;
            for (int iter = 0; iter < 20; iter++)
            {
                q = GaussianOp_Laplace.Q(xB, meanPrior, precPrior, q);
                //xF = GaussianOp_Laplace.SampleAverageConditional(xB, meanPrior, precPrior, q);
                xF = Gaussian.FromMeanAndPrecision(mean, r0);
                xB = IsPositiveOp.XAverageConditional(true, xF);
                Console.WriteLine("xF = {0} xB = {1}", xF, xB);
            }
            Console.WriteLine("x = {0} should be {1}", xF * xB, xExpected);

            double[] precs = linspace(1e-3, 5, 100);
            double[] evTrue = new double[precs.Length];
            double[] evApprox = new double[precs.Length];
            double[] evApprox2 = new double[precs.Length];
            //r0 = q.GetMean();
            double sum = 0, sum2 = 0;
            for (int i = 0; i < precs.Length; i++)
            {
                double r = precs[i];
                Gaussian xFt = Gaussian.FromMeanAndPrecision(mean, r);
                evTrue[i] = IsPositiveOp.LogAverageFactor(true, xFt) + precPrior.GetLogProb(r);
                evApprox[i] = IsPositiveOp.LogAverageFactor(true, xF) + precPrior.GetLogProb(r) + xB.GetLogAverageOf(xFt) - xB.GetLogAverageOf(xF);
                evApprox2[i] = IsPositiveOp.LogAverageFactor(true, xF) + precPrior.GetLogProb(r0) + q.GetLogProb(r) - q.GetLogProb(r0);
                sum += System.Math.Exp(evApprox[i]);
                sum2 += System.Math.Exp(evApprox2[i]);
            }
            Console.WriteLine("r0 = {0}: {1} {2} {3}", r0, sum, sum2, q.GetVariance() + System.Math.Pow(r0 - q.GetMean(), 2));
            //TODO: change path for cross platform using
            using (var writer = new MatlabWriter(@"..\..\..\Tests\student.mat"))
            {
                writer.Write("z", evTrue);
                writer.Write("z2", evApprox);
                writer.Write("z3", evApprox2);
                writer.Write("precs", precs);
            }
        }

        private Gaussian StudentIsPositiveExact(double mean, Gamma precPrior, out double evidence)
        {
            // importance sampling for true answer
            GaussianEstimator est = new GaussianEstimator();
            int nSamples = 1000000;
            evidence = 0;
            for (int iter = 0; iter < nSamples; iter++)
            {
                double precSample = precPrior.Sample();
                Gaussian xPrior = Gaussian.FromMeanAndPrecision(mean, precSample);
                double logWeight = IsPositiveOp.LogAverageFactor(true, xPrior);
                evidence += System.Math.Exp(logWeight);
                double xSample = xPrior.Sample();
                if (xSample > 0)
                    est.Add(xSample);
            }
            evidence /= nSamples;
            return est.GetDistribution(new Gaussian());
        }

        public static double[] linspace(double min, double max, int count)
        {
            if (count < 2)
                throw new ArgumentException("count < 2");
            double inc = (max - min) / (count - 1);
            return Util.ArrayInit(count, i => (min + i * inc));
        }

        internal void StudentIsPositiveTest2()
        {
            GaussianOp.ForceProper = false;
            double shape = 1;
            double mean = -1;
            Gamma precPrior = Gamma.FromShapeAndRate(shape, shape);
            Gaussian meanPrior = Gaussian.PointMass(mean);
            double evExpected;
            Gaussian xExpected = StudentIsPositiveExact(mean, precPrior, out evExpected);

            Gaussian xF2 = Gaussian.FromMeanAndVariance(-1, 1);
            // the energy has a stationary point here (min in both dimensions), even though xF0 is improper
            Gaussian xB0 = new Gaussian(2, 1);
            xF2 = Gaussian.FromMeanAndVariance(-4.552, 6.484);
            //xB0 = new Gaussian(1.832, 0.9502);
            //xB0 = new Gaussian(1.792, 1.558);
            //xB0 = new Gaussian(1.71, 1.558);
            //xB0 = new Gaussian(1.792, 1.5);
            Gaussian xF0 = GaussianOp_Slow.SampleAverageConditional(xB0, meanPrior, precPrior);
            //Console.WriteLine("xB0 = {0} xF0 = {1}", xB0, xF0);
            //Console.WriteLine(xF0*xB0);
            //Console.WriteLine(xF2*xB0);

            xF2 = new Gaussian(0.8651, 1.173);
            xB0 = new Gaussian(-4, 2);
            xB0 = new Gaussian(7, 7);
            if (false)
            {
                xF2 = new Gaussian(mean, 1);
                double[] xs = linspace(0, 100, 1000);
                double[] logTrue = Util.ArrayInit(xs.Length, i => GaussianOp.LogAverageFactor(xs[i], mean, precPrior));
                Normalize(logTrue);
                xF2 = FindxF4(xs, logTrue, xF2);
                xF2 = Gaussian.FromNatural(-0.85, 0);
                xB0 = IsPositiveOp.XAverageConditional(true, xF2);
                Console.WriteLine("xF = {0} xB = {1}", xF2, xB0);
                Console.WriteLine("x = {0} should be {1}", xF2 * xB0, xExpected);
                Console.WriteLine("proj[T*xB] = {0}", GaussianOp_Slow.SampleAverageConditional(xB0, meanPrior, precPrior) * xB0);
                double ev = System.Math.Exp(IsPositiveOp.LogAverageFactor(true, xF2) + GaussianOp_Slow.LogAverageFactor(xB0, meanPrior, precPrior) - xF2.GetLogAverageOf(xB0));
                Console.WriteLine("evidence = {0} should be {1}", ev, evExpected);
                return;
            }
            if (false)
            {
                xF2 = new Gaussian(mean, 1);
                xF2 = FindxF3(xExpected, evExpected, meanPrior, precPrior, xF2);
                xB0 = IsPositiveOp.XAverageConditional(true, xF2);
                Console.WriteLine("xF = {0} xB = {1}", xF2, xB0);
                Console.WriteLine("x = {0} should be {1}", xF2 * xB0, xExpected);
                //double ev = Math.Exp(IsPositiveOp.LogAverageFactor(true, xF2) + GaussianOp.LogAverageFactor_slow(xB0, meanPrior, precPrior) - xF2.GetLogAverageOf(xB0));
                //Console.WriteLine("evidence = {0} should be {1}", ev, evExpected);
                return;
            }
            if (false)
            {
                xF2 = new Gaussian(-2, 10);
                xF2 = FindxF2(meanPrior, precPrior, xF2);
                xB0 = IsPositiveOp.XAverageConditional(true, xF2);
                xF0 = GaussianOp_Slow.SampleAverageConditional(xB0, meanPrior, precPrior);
                Console.WriteLine("xB = {0}", xB0);
                Console.WriteLine("xF = {0} should be {1}", xF0, xF2);
                return;
            }
            if (false)
            {
                xF2 = new Gaussian(-3998, 4000);
                xF2 = new Gaussian(0.8651, 1.173);
                xB0 = new Gaussian(-4, 2);
                xB0 = new Gaussian(2000, 1e-5);
                xB0 = FindxB(xB0, meanPrior, precPrior, xF2);
                xF0 = GaussianOp_Slow.SampleAverageConditional(xB0, meanPrior, precPrior);
                Console.WriteLine("xB = {0}", xB0);
                Console.WriteLine("xF = {0} should be {1}", xF0, xF2);
                return;
            }
            if (false)
            {
                //xF2 = new Gaussian(-7, 10);
                //xF2 = new Gaussian(-50, 52);
                xB0 = new Gaussian(-1.966, 5.506e-08);
                //xF2 = new Gaussian(-3998, 4000);
                xF0 = FindxF(xB0, meanPrior, precPrior, xF2);
                Gaussian xB2 = IsPositiveOp.XAverageConditional(true, xF0);
                Console.WriteLine("xF = {0}", xF0);
                Console.WriteLine("xB = {0} should be {1}", xB2, xB0);
                return;
            }
            if (true)
            {
                xF0 = new Gaussian(-3.397e+08, 5.64e+08);
                xF0 = new Gaussian(-2.373e+04, 2.8e+04);
                xB0 = new Gaussian(2.359, 1.392);
                xF0 = Gaussian.FromNatural(-0.84, 0);
                //xF0 = Gaussian.FromNatural(-0.7, 0);
                for (int iter = 0; iter < 10; iter++)
                {
                    xB0 = FindxB(xB0, meanPrior, precPrior, xF0);
                    Gaussian xFt = GaussianOp_Slow.SampleAverageConditional(xB0, meanPrior, precPrior);
                    Console.WriteLine("xB = {0}", xB0);
                    Console.WriteLine("xF = {0} should be {1}", xFt, xF0);
                    xF0 = FindxF0(xB0, meanPrior, precPrior, xF0);
                    Gaussian xBt = IsPositiveOp.XAverageConditional(true, xF0);
                    Console.WriteLine("xF = {0}", xF0);
                    Console.WriteLine("xB = {0} should be {1}", xBt, xB0);
                }
                Console.WriteLine("x = {0} should be {1}", xF0 * xB0, xExpected);
                double ev = System.Math.Exp(IsPositiveOp.LogAverageFactor(true, xF0) + GaussianOp_Slow.LogAverageFactor(xB0, meanPrior, precPrior) - xF0.GetLogAverageOf(xB0));
                Console.WriteLine("evidence = {0} should be {1}", ev, evExpected);
                return;
            }

            //var precs = linspace(1e-6, 1e-5, 200);
            var precs = linspace(xB0.Precision / 11, xB0.Precision, 100);
            //var precs = linspace(xF0.Precision/20, xF0.Precision/3, 100);
            precs = linspace(1e-9, 1e-5, 100);
            //precs = new double[] { xB0.Precision };
            var ms = linspace(xB0.GetMean() - 1, xB0.GetMean() + 1, 100);
            //var ms = linspace(xF0.GetMean()-1, xF0.GetMean()+1, 100);
            //precs = linspace(1.0/10, 1.0/8, 200);
            ms = linspace(2000, 4000, 100);
            //ms = new double[] { xB0.GetMean() };
            Matrix result = new Matrix(precs.Length, ms.Length);
            Matrix result2 = new Matrix(precs.Length, ms.Length);
            //ms = new double[] { 0.7 };
            for (int j = 0; j < ms.Length; j++)
            {
                double maxZ = double.NegativeInfinity;
                double minZ = double.PositiveInfinity;
                Gaussian maxxF = Gaussian.Uniform();
                Gaussian minxF = Gaussian.Uniform();
                Gaussian maxxB = Gaussian.Uniform();
                Gaussian minxB = Gaussian.Uniform();
                Vector v = Vector.Zero(3);
                for (int i = 0; i < precs.Length; i++)
                {
                    Gaussian xF = Gaussian.FromMeanAndPrecision(ms[j], precs[i]);
                    xF = xF2;
                    Gaussian xB = IsPositiveOp.XAverageConditional(true, xF);
                    xB = Gaussian.FromMeanAndPrecision(ms[j], precs[i]);
                    //xB = xB0;
                    v[0] = IsPositiveOp.LogAverageFactor(true, xF);
                    v[1] = GaussianOp.LogAverageFactor_slow(xB, meanPrior, precPrior);
                    //v[1] = GaussianOp_Slow.LogAverageFactor(xB, meanPrior, precPrior);
                    v[2] = -xF.GetLogAverageOf(xB);
                    double logZ = v.Sum();
                    double Z = logZ;
                    if (Z > maxZ)
                    {
                        maxZ = Z;
                        maxxF = xF;
                        maxxB = xB;
                    }
                    if (Z < minZ)
                    {
                        minZ = Z;
                        minxF = xF;
                        minxB = xB;
                    }
                    result[i, j] = Z;
                    result2[i, j] = IsPositiveOp.LogAverageFactor(true, xF) + xF0.GetLogAverageOf(xB) - xF.GetLogAverageOf(xB);
                    //Gaussian xF3 = GaussianOp.SampleAverageConditional_slower(xB, meanPrior, precPrior);
                    //result[i, j] = Math.Pow(xF3.Precision - xF.Precision, 2);
                    //result2[i, j] = Math.Pow((xF2*xB).Precision - (xF*xB).Precision, 2);
                    //result2[i, j] = -xF.GetLogAverageOf(xB);
                    //Gaussian xF2 = GaussianOp.SampleAverageConditional_slow(xB, Gaussian.PointMass(0), precPrior);
                    Gaussian xMarginal = xF * xB;
                    //Console.WriteLine("xF = {0} Z = {1} x = {2}", xF, Z.ToString("g4"), xMarginal);
                }
                double delta = v[1] - v[2];
                //Console.WriteLine("xF = {0} xB = {1} maxZ = {2} x = {3}", maxxF, maxxB, maxZ.ToString("g4"), maxxF*maxxB);
                //Console.WriteLine("xF = {0} maxZ = {1} delta = {2}", maxxF, maxZ.ToString("g4"), delta.ToString("g4"));
                Console.WriteLine("xF = {0} xB = {1} minZ = {2} x = {3}", minxF, minxB, minZ.ToString("g4"), minxF * minxB);
            }
            //TODO: change path for cross platform using
            using (var writer = new MatlabWriter(@"..\..\..\Tests\student.mat"))
            {
                writer.Write("z", result);
                writer.Write("z2", result2);
                writer.Write("precs", precs);
                writer.Write("ms", ms);
            }
        }

        public static Gaussian FindxF4(double[] xs, double[] logTrue, Gaussian xF)
        {
            double[] logApprox = new double[xs.Length];
            Func<Vector, double> func = delegate (Vector x2)
            {
                Gaussian xFt = Gaussian.FromMeanAndPrecision(x2[0], System.Math.Exp(x2[1]));
                for (int i = 0; i < xs.Length; i++)
                {
                    logApprox[i] = xFt.GetLogProb(xs[i]);
                }
                Normalize(logApprox);
                double sum = 0;
                for (int i = 0; i < xs.Length; i++)
                {
                    sum += System.Math.Abs(System.Math.Exp(logApprox[i]) - System.Math.Exp(logTrue[i]));
                    //sum += Math.Pow(Math.Exp(logApprox[i]) - Math.Exp(logTrue[i]), 2);
                    //sum += Math.Pow(Math.Exp(logApprox[i]/2) - Math.Exp(logTrue[i]/2), 2);
                    //sum += Math.Exp(logApprox[i])*(logApprox[i] - logTrue[i]);
                    //sum += Math.Exp(logTrue[i])*(logTrue[i] - logApprox[i]);
                }
                return sum;
            };

            double m = xF.GetMean();
            double p = xF.Precision;
            Vector x = Vector.FromArray(m, System.Math.Log(p));
            Minimize2(func, x);
            return Gaussian.FromMeanAndPrecision(x[0], System.Math.Exp(x[1]));
        }
        private static void Normalize(double[] logProb)
        {
            double logSum = MMath.LogSumExp(logProb);
            for (int i = 0; i < logProb.Length; i++)
            {
                logProb[i] -= logSum;
            }
        }

        public static Gaussian FindxF3(Gaussian xExpected, double evExpected, Gaussian meanPrior, Gamma precPrior, Gaussian xF)
        {
            Func<Vector, double> func = delegate (Vector x2)
            {
                Gaussian xFt = Gaussian.FromMeanAndPrecision(x2[0], System.Math.Exp(x2[1]));
                Gaussian xB = IsPositiveOp.XAverageConditional(true, xFt);
                Gaussian xM = xFt * xB;
                //return KlDiv(xExpected, xM);
                return KlDiv(xM, xExpected);
                //Gaussian xF2 = GaussianOp.SampleAverageConditional_slow(xB, meanPrior, precPrior);
                //Gaussian xF2 = GaussianOp_Slow.SampleAverageConditional(xB, meanPrior, precPrior);
                //Gaussian xM2 = xF2*xB;
                //double ev1 = IsPositiveOp.LogAverageFactor(true, xFt);
                //double ev2 = GaussianOp.LogAverageFactor_slow(xB, meanPrior, precPrior) - xFt.GetLogAverageOf(xB);
                //double ev = ev1 + ev2;
                //return xExpected.MaxDiff(xM);
                //return Math.Pow(xExpected.GetMean() - xM.GetMean(), 2) + Math.Pow(ev - Math.Log(evExpected), 2);
                //return 100*Math.Pow(xM.GetMean() - xM2.GetMean(), 2) -ev;
                //return 100*Math.Pow(ev2, 2) + Math.Pow(ev - Math.Log(evExpected), 2);
                //return 100*Math.Pow(ev2, 2) + Math.Pow(xM2.GetMean() - xM.GetMean(), 2);
            };

            double m = xF.GetMean();
            double p = xF.Precision;
            Vector x = Vector.FromArray(m, System.Math.Log(p));
            Minimize2(func, x);
            return Gaussian.FromMeanAndPrecision(x[0], System.Math.Exp(x[1]));
        }

        public static double KlDiv(Gaussian p, Gaussian q)
        {
            // E[log p] = -0.5*log(vp) - 0.5*vp/vp
            // E[log q] = -0.5*log(vq) - 0.5*((mp-mq)^2 + vp)/vq
            double delta = p.GetMean() - q.GetMean();
            return 0.5 * System.Math.Log(p.Precision / q.Precision) - 0.5 + 0.5 * (delta * delta + p.GetVariance()) * q.Precision;
        }
        public static double MeanError(Gaussian p, Gaussian q)
        {
            double delta = p.GetMean() - q.GetMean();
            return 0.5 * delta * delta;
        }

        public static Gaussian FindxF2(Gaussian meanPrior, Gamma precPrior, Gaussian xF)
        {
            Func<Vector, double> func = delegate (Vector x2)
            {
                Gaussian xFt = Gaussian.FromMeanAndPrecision(x2[0], System.Math.Exp(x2[1]));
                Gaussian xB = IsPositiveOp.XAverageConditional(true, xFt);
                Gaussian xF2 = GaussianOp_Slow.SampleAverageConditional(xB, meanPrior, precPrior);
                return xFt.MaxDiff(xF2);
            };

            double m = xF.GetMean();
            double p = xF.Precision;
            Vector x = Vector.FromArray(m, System.Math.Log(p));
            Minimize2(func, x);
            return Gaussian.FromMeanAndPrecision(x[0], System.Math.Exp(x[1]));
        }

        public static Gaussian FindxB(Gaussian xB, Gaussian meanPrior, Gamma precPrior, Gaussian xF)
        {
            Gaussian xB3 = IsPositiveOp.XAverageConditional(true, xF);
            Func<Vector, double> func = delegate (Vector x2)
            {
                Gaussian xB2 = Gaussian.FromMeanAndPrecision(x2[0], System.Math.Exp(x2[1]));
                //Gaussian xF2 = GaussianOp.SampleAverageConditional_slow(xB2, meanPrior, precPrior);
                Gaussian xF2 = GaussianOp_Slow.SampleAverageConditional(xB2, meanPrior, precPrior);
                //Assert.True(xF2.MaxDiff(xF3) < 1e-10);
                //return Math.Pow((xF*xB2).GetMean() - (xF2*xB2).GetMean(), 2) + Math.Pow((xF*xB2).GetVariance() - (xF2*xB2).GetVariance(), 2);
                //return KlDiv(xF2*xB2, xF*xB2) + KlDiv(xF*xB3, xF*xB2);
                //return KlDiv(xF2*xB2, xF*xB2) + Math.Pow((xF*xB3).GetMean() - (xF*xB2).GetMean(),2);
                return MeanError(xF2 * xB2, xF * xB2) + KlDiv(xF * xB3, xF * xB2);
                //return xF.MaxDiff(xF2);
                //Gaussian q = new Gaussian(0, 0.1);
                //return Math.Pow((xF*q).GetMean() - (xF2*q).GetMean(), 2) + Math.Pow((xF*q).GetVariance() - (xF2*q).GetVariance(), 2);
            };

            double m = xB.GetMean();
            double p = xB.Precision;
            Vector x = Vector.FromArray(m, System.Math.Log(p));
            Minimize2(func, x);
            return Gaussian.FromMeanAndPrecision(x[0], System.Math.Exp(x[1]));
        }

        public static Gaussian FindxF(Gaussian xB, Gaussian meanPrior, Gamma precPrior, Gaussian xF)
        {
            Gaussian xF3 = GaussianOp_Slow.SampleAverageConditional(xB, meanPrior, precPrior);
            Func<Vector, double> func = delegate (Vector x2)
            {
                Gaussian xF2 = Gaussian.FromMeanAndPrecision(x2[0], System.Math.Exp(x2[1]));
                Gaussian xB2 = IsPositiveOp.XAverageConditional(true, xF2);
                //return (xF2*xB2).MaxDiff(xF2*xB) + (xF3*xB).MaxDiff(xF2*xB);
                //return KlDiv(xF2*xB2, xF2*xB) + KlDiv(xF3*xB, xF2*xB);
                //return KlDiv(xF3*xB, xF2*xB) + Math.Pow((xF2*xB2).GetMean() - (xF2*xB).GetMean(),2);
                return KlDiv(xF2 * xB2, xF2 * xB) + MeanError(xF3 * xB, xF2 * xB);
            };

            double m = xF.GetMean();
            double p = xF.Precision;
            Vector x = Vector.FromArray(m, System.Math.Log(p));
            Minimize2(func, x);
            //MinimizePowell(func, x);
            return Gaussian.FromMeanAndPrecision(x[0], System.Math.Exp(x[1]));
        }

        public static Gaussian FindxF0(Gaussian xB, Gaussian meanPrior, Gamma precPrior, Gaussian xF)
        {
            Gaussian xF3 = GaussianOp_Slow.SampleAverageConditional(xB, meanPrior, precPrior);
            Func<double, double> func = delegate (double tau2)
            {
                Gaussian xF2 = Gaussian.FromNatural(tau2, 0);
                if (tau2 >= 0)
                    return double.PositiveInfinity;
                Gaussian xB2 = IsPositiveOp.XAverageConditional(true, xF2);
                //return (xF2*xB2).MaxDiff(xF2*xB) + (xF3*xB).MaxDiff(xF2*xB);
                //return KlDiv(xF2*xB2, xF2*xB) + KlDiv(xF3*xB, xF2*xB);
                //return KlDiv(xF3*xB, xF2*xB) + Math.Pow((xF2*xB2).GetMean() - (xF2*xB).GetMean(), 2);
                return KlDiv(xF2 * xB2, xF2 * xB) + MeanError(xF3 * xB, xF2 * xB);
            };

            double tau = xF.MeanTimesPrecision;
            double fmin;
            tau = Minimize(func, tau, out fmin);
            //MinimizePowell(func, x);
            return Gaussian.FromNatural(tau, 0);
        }

        internal static void Minimize2a(Func<Vector, double> func, Vector x)
        {
            Vector temp = Vector.Copy(x);
            Func<double, double> func1 = delegate (double x1)
            {
                temp[1] = x1;
                return func(temp);
            };
            Func<double, double> func0 = delegate (double x0)
            {
                temp[0] = x0;
                double fmin0;
                Minimize(func1, x[1], out fmin0);
                return fmin0;
            };
            double fTol = 1e-15;
            double delta = 100;
            int maxIter = 100;
            int iter;
            double fmin = func(x);
            for (iter = 0; iter < maxIter; iter++)
            {
                double oldmin = fmin;
                double newx = MinimizeBrent(func0, x[0] - 2 * delta, x[0], x[0] + 2 * delta, out fmin);
                if (fmin > oldmin)
                    throw new Exception("objective increased");
                delta = newx - x[0];
                x[0] = newx;
                temp[0] = newx;
                x[1] = Minimize(func1, x[1], out fmin);
                Console.WriteLine("x = {0} f = {1}", x, fmin);
                if (MMath.AbsDiff(fmin, oldmin, 1e-14) < fTol)
                    break;
            }
            if (iter == maxIter)
                throw new Exception("exceeded maximum number of iterations");
        }
        private static void Minimize2(Func<Vector, double> func, Vector x)
        {
            Vector temp = Vector.Copy(x);
            Func<double, double> func1 = delegate (double x1)
            {
                temp[1] = x1;
                return func(temp);
            };
            Func<double, double> func0 = delegate (double x0)
            {
                temp[0] = x0;
                double fmin0;
                Minimize(func1, x[1], out fmin0);
                return fmin0;
            };
            double xTol = 1e-10;
            double delta = 1;
            int maxIter = 100;
            int iter;
            double fmin = func(x);
            for (iter = 0; iter < maxIter; iter++)
            {
                double oldmin = fmin;
                Console.WriteLine("x={0} f={1} delta={2}", x, fmin, delta);
                if (delta < xTol)
                    break;
                bool changed = false;
                double f1 = func0(x[0] - delta);
                if (f1 < fmin)
                {
                    x[0] -= delta;
                    changed = true;
                }
                else
                {
                    double f2 = func0(x[0] + delta);
                    if (f2 < fmin)
                    {
                        x[0] += delta;
                        changed = true;
                    }
                    else
                    {
                        delta /= 2;
                    }
                }
                if (changed)
                {
                    temp[0] = x[0];
                    x[1] = Minimize(func1, x[1], out fmin);
                    delta *= 2;
                }
            }
        }

        public static double Minimize(Func<double, double> func, double x, out double fmin)
        {
            double fTol = 1e-15;
            double delta = 1;
            int maxIter = 100;
            int iter;
            fmin = func(x);
            for (iter = 0; iter < maxIter; iter++)
            {
                double oldmin = fmin;
                double newx = MinimizeBrent(func, x - 2 * delta, x, x + 2 * delta, out fmin);
                if (fmin > oldmin)
                    throw new Exception("objective increased");
                delta = newx - x;
                x = newx;
                if (MMath.AbsDiff(fmin, oldmin, 1e-14) < fTol)
                    break;
            }
            if (iter == maxIter)
                throw new Exception("exceeded maximum number of iterations");
            return x;
        }

        /* Minimize a multidimensional scalar function starting at x.
         * Modifies x to be the minimum.
         */
        internal static void MinimizePowell(Func<Vector, double> func, Vector x)
        {
            double fTol = 1e-15;
            Vector old_x = Vector.Copy(x);
            Vector ext_x = Vector.Copy(x);
            int d = x.Count;
            /* Initialize the directions to the unit vectors */
            Vector[] dirs = Util.ArrayInit(d, i => Vector.FromArray(Util.ArrayInit(d, j => (i == j) ? 1.0 : 0.0)));
            double fmin = func(x);
            int maxIter = 100;
            int iter;
            for (iter = 0; iter < maxIter; iter++)
            {
                double fx = fmin;
                int i_max = 0;
                double delta_max = 0;
                /* Minimize along each direction, remembering the direction of greatest
                 * function decrease.
                 */
                for (int i = 0; i < d; i++)
                {
                    double old_min = fmin;
                    Vector dir = dirs[i];
                    double a = MinimizeLine(func, x, dir, out fmin);
                    dir.Scale(a);
                    if (fmin > old_min)
                        throw new Exception("objective increased");
                    double delta = System.Math.Abs(old_min - fmin);
                    if (delta > delta_max)
                    {
                        delta_max = delta;
                        i_max = i;
                    }
                }
                if (MMath.AbsDiff(fx, fmin, 1e-14) < fTol)
                    break;
                /* Construct new direction from old_x to x. */
                Vector dir2 = x - old_x;
                old_x.SetTo(x);
                /* And extrapolate it. */
                ext_x.SetTo(x);
                x.SetToSum(x, dir2);
                /* Good extrapolation? */
                double fex = func(x);
                x.SetTo(ext_x);
                if (fex < fx)
                {
                    double t = fx - fmin - delta_max;
                    double delta = fx - fex;
                    t = 2 * (fx - 2 * fmin + fex) * t * t - delta_max * delta * delta;
                    if (t < 0)
                    {
                        double a = MinimizeLine(func, x, dir2, out fmin);
                        dir2.Scale(a);
                        /* Replace i_max with the new dir. */
                        dirs[i_max] = dir2;
                    }
                }
                Console.WriteLine("x = {0} f = {1}", x, fmin);
            }
            if (iter == maxIter)
                throw new Exception("exceeded maximum number of iterations");
        }

        /* Modifies x to be the minimum of f along the direction dir. */
        public static double MinimizeLine(Func<Vector, double> func, Vector x, Vector dir, out double fmin)
        {
            Vector temp = Vector.Zero(x.Count);
            Func<double, double> lineFunc = delegate (double u)
            {
                temp.SetToSum(1.0, x, u, dir);
                return func(temp);
            };
            double a = MinimizeBrent(lineFunc, -2, 0, 2, out fmin);
            if (lineFunc(a) > lineFunc(0))
                throw new Exception("objective increased");
            x.SetToSum(1.0, x, a, dir);
            return a;
        }

        /* Minimize the scalar function f in the interval [a,b] via Brent's method.
         * Requires a < b.
         * Modifies *fmin_return to be the ordinate of the minimum.
         * Returns the abscissa of the minimum.
         * Algorithm taken from Numerical Recipes and Matlab optimization toolbox.
         */
        public static double MinimizeBrent(Func<double, double> func, double min, double x, double max, out double fmin)
        {
            double tol = 1e-3;
            double d = 0, e = 0;
            double v, w;
            double fx, fv, fw;
            double u, fu;
            int iter;
            const double cgold = 0.38196601125011;
            const double zeps = 1e-10;

            //double x = min + cgold*(max-min); /* golden section to get third point */
            w = v = x;
            fx = func(x);
            fv = fx;
            fw = fx;

            int maxIter = 100;
            for (iter = 0; iter < maxIter; iter++)
            {
                bool golden_section_step = true;
                double xm = (min + max) / 2;
                double tol1 = zeps * System.Math.Abs(x) + tol / 3;
                double tol2 = 2 * tol1;
                if (System.Math.Abs(x - xm) <= (tol2 - (max - min) / 2))
                    break;

                /* Construct a trial parabolic fit */
                if (System.Math.Abs(e) > tol1)
                {
                    double r = (x - w) * (fx - fv);
                    double q = (x - v) * (fx - fw);
                    double p = (x - v) * q - (x - w) * r;
                    q = 2 * (q - r);
                    if (q > 0)
                        p = -p;
                    q = System.Math.Abs(q);
                    r = e;
                    e = d;
                    /* Is the parabola acceptable? */
                    if ((System.Math.Abs(p) < System.Math.Abs(0.5 * q * r)) && (p > q * (min - x)) && (p < q * (max - x)))
                    {
                        /* Yes, take the parabolic step */
                        d = p / q;
                        u = x + d;
                        if ((u - min < tol2) || (max - u < tol2))
                        {
                            d = tol1;
                            if (xm < x)
                                d = -d;
                        }
                        golden_section_step = false;
                    }
                }

                if (golden_section_step)
                {
                    /* Take the golden section step */
                    if (x >= xm)
                        e = min - x;
                    else
                        e = max - x;
                    d = cgold * e;
                }

                /* Evaluate f at x+d as long as fabs(d) > tol1 */
                if (System.Math.Abs(d) >= tol1)
                    u = d;
                else if (d >= 0)
                    u = tol1;
                else
                    u = -tol1;
                u += x;
                fu = func(u);

                if (fu <= fx)
                {
                    if (u >= x)
                        min = x;
                    else
                        max = x;
                    v = w;
                    w = x;
                    x = u;
                    fv = fw;
                    fw = fx;
                    fx = fu;
                }
                else
                {
                    if (u < x)
                        min = u;
                    else
                        max = u;
                    if (fu <= fw || w == x)
                    {
                        v = w;
                        w = u;
                        fv = fw;
                        fw = fu;
                    }
                    else if (fu <= fv || v == x || v == w)
                    {
                        v = u;
                        fv = fu;
                    }
                }
            }
            if (iter == maxIter)
            {
                throw new Exception("exceeded maximum number of iterations");
            }
            fmin = fx;
            return x;
        }


        // this is a very simple derivative-free optimizer
        internal static void Minimize(Func<Vector, double> func, Vector x, int maxIter = 1000, double xTol = 1e-10)
        {
            Vector delta = Vector.Constant(x.Count, 1);
            Vector temp = Vector.Zero(x.Count);
            for (int iter = 0; iter < maxIter; iter++)
            {
                bool changed = false;
                for (int i = 0; i < x.Count; i++)
                {
                    //if (i == 0) continue;
                    double f = func(x);
                    Console.WriteLine("x={0} f={1} delta={2}", x, f, delta);
                    while (delta[i] > xTol)
                    {
                        temp.SetTo(x);
                        temp[i] = x[i] - delta[i];
                        double f1 = func(temp);
                        if (f1 < f)
                        {
                            x[i] -= delta[i];
                            changed = true;
                            break;
                        }
                        temp[i] = x[i] + delta[i];
                        double f2 = func(temp);
                        if (f2 < f)
                        {
                            x[i] += delta[i];
                            changed = true;
                            break;
                        }
                        delta[i] /= 2;
                    }
                    delta[i] *= 2;
                }
                if (!changed)
                    break;
            }
        }

        internal void StudentIsPositiveTest3()
        {
            double shape = 1;
            Gamma precPrior = Gamma.FromShapeAndRate(shape, shape);

            Gaussian meanPrior = Gaussian.PointMass(0);
            Gaussian xB = Gaussian.Uniform();
            Gaussian xF = GaussianOp.SampleAverageConditional_slow(xB, meanPrior, precPrior);
            for (int iter = 0; iter < 100; iter++)
            {
                xB = IsPositiveOp.XAverageConditional(true, xF);
                xF = GetConstrainedMessage(xB, meanPrior, precPrior, xF);
            }
            Console.WriteLine("xF = {0} x = {1}", xF, xB * xF);
        }
        private static Gaussian GetConstrainedMessage(Gaussian sample, Gaussian mean, Gamma precision, Gaussian to_sample)
        {
            for (int iter = 0; iter < 100; iter++)
            {
                Gaussian old = to_sample;
                to_sample = GetConstrainedMessage1(sample, mean, precision, to_sample);
                if (old.MaxDiff(to_sample) < 1e-10)
                    break;
            }
            return to_sample;
        }
        private static Gaussian GetConstrainedMessage1(Gaussian sample, Gaussian mean, Gamma precision, Gaussian to_sample)
        {
            Gaussian sampleMarginal = sample * to_sample;
            double m1, v1;
            to_sample.GetMeanAndVariance(out m1, out v1);
            double m, v;
            sampleMarginal.GetMeanAndVariance(out m, out v);
            double moment2 = m * m + v;
            // vq < moment2 implies 1/vq > 1/moment2
            // implies 1/v2 > 1/moment2 - to_sample.Precision
            double v2max = 1 / (1 / moment2 - to_sample.Precision);
            double v2min = 1e-2;
            double[] v2s = linspace(v2min, v2max, 100);
            double p2min = 1 / moment2 - to_sample.Precision;
            if (p2min < 0.0)
                return to_sample;
            double p2max = sample.Precision * 10;
            double[] p2s = linspace(p2min, p2max, 100);
            Gaussian bestResult = to_sample;
            double bestScore = double.PositiveInfinity;
            for (int i = 0; i < p2s.Length; i++)
            {
                double p2 = p2s[i];
                double vq = 1 / (to_sample.Precision + p2);
                double m2 = (System.Math.Sqrt(moment2 - vq) / vq - to_sample.MeanTimesPrecision) / p2;
                // check
                double mq = vq * (to_sample.MeanTimesPrecision + m2 * p2);
                Assert.True(MMath.AbsDiff(mq * mq + vq, moment2) < 1e-10);
                Gaussian sample2 = Gaussian.FromMeanAndPrecision(m2, p2);
                Gaussian result = GaussianOp.SampleAverageConditional_slow(sample2, mean, precision);
                double score = System.Math.Abs(result.MeanTimesPrecision);
                if (score < bestScore)
                {
                    bestScore = score;
                    bestResult = result;
                }
            }
            return bestResult;
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
                precision = Variable.Copy(precisionT);
                precision.AddAttribute(new MarginalPrototype(new Gamma()));
            }
            else
            {
                precision = Variable.GammaFromShapeAndScale(1, 1).Named("precision");
                precision.AddAttribute(new PointEstimate());
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

        public static double BesselKAsympt(double a, double x)
        {
            double poch1 = a + 0.5;
            double poch2 = 0.5 - a;
            double z = -0.5 / x;
            double term = 1;
            double sum = 0;
            for (int k = 0; k < 1000; k++)
            {
                double oldterm = term;
                sum += term;
                term *= poch1 * poch2 / (k + 1) * z;
                poch1++;
                poch2++;
                // must not sum too many terms
                if (System.Math.Abs(term) > System.Math.Abs(oldterm))
                {
                    Console.WriteLine("BesselKAsympt: {0} terms", k);
                    break;
                }
            }
            return sum * 0.5 * MMath.Sqrt2PI / System.Math.Sqrt(x) * System.Math.Exp(-x);
        }
        public static double BesselKSeries(double a, double x)
        {
            if (a == System.Math.Floor(a))
                throw new Exception("a cannot be integer");
            return 0.5 * System.Math.PI / System.Math.Sin(System.Math.PI * a) * (MMath.BesselI(-a, x) - MMath.BesselI(a, x));
        }
        public static double[] xBesselKDerivativesAt0(int n, double a)
        {
            double[] derivs = new double[n + 1];
            derivs[0] = MMath.Gamma(System.Math.Abs(a)) * System.Math.Pow(2, a - 0.5) / System.Math.Sqrt(System.Math.PI);
            if (n > 0)
            {
                derivs[1] = derivs[0];
                if (n > 1)
                {
                    double[] derivs2 = xBesselKDerivativesAt0(n - 2, a - 1);
                    for (int i = 0; i < n - 2; i++)
                    {
                        derivs[i + 2] = derivs[i + 1] - (i + 1) * derivs2[i];
                    }
                }
            }
            return derivs;
        }
        internal void BesselKTest()
        {
            if (false)
            {
                // 12.695270349135780322468450599556863540081194237115
                Console.WriteLine(BesselKSeries(1.1, 0.1));
                Console.WriteLine(BesselKAsympt(1.1, 0.1));
                // 0.000018836375374259574280782608722012995073702923939575
                Console.WriteLine(BesselKSeries(1.1, 10));
                // requires approx x terms
                Console.WriteLine(BesselKAsympt(1.1, 10));
                Console.WriteLine(BesselKSeries(0.5, 10));
                return;
            }
            if (false)
            {
                double[] derivs = xBesselKDerivativesAt0(10, 0.5);
                for (int i = 0; i < derivs.Length; i++)
                {
                    Console.WriteLine(derivs[i]);
                }
                return;
            }
        }
        internal void GaussianFromMeanAndVarianceTest()
        {
            double mm = 4, vm = 3;
            double mx = 0, vx = 1;
            double b = 3;
            for (int i = 1; i < 10; i++)
            {
                double a = i * 0.5;
                GaussianFromMeanAndVarianceTest(mm, vm, mx, vx, a, b);
                if (true)
                {
                    // test extreme variances
                    GaussianFromMeanAndVarianceTest(mm, vm, mx, vx * 1e6, a, b);
                    GaussianFromMeanAndVarianceTest(mm, vm, mx, vx * 1e20, a, b);
                    GaussianFromMeanAndVarianceTest(mm, vm, mx, Double.PositiveInfinity, a, b);
                    GaussianFromMeanAndVarianceTest(mm, vm * 1e6, mx, vx, a, b);
                    GaussianFromMeanAndVarianceTest(mm, vm * 1e20, mx, vx, a, b);
                }
            }
        }
        internal void GaussianFromMeanAndVarianceTest2()
        {
            double mm = 4, vm = .3;
            double mx = 0, vx = .001;
            double b = 3;

            for (int i = 1; i < 10; i++)
            {
                //double a = i*0.5;
                double a = 1;
                GaussianFromMeanAndVarianceTest(mm, vm * System.Math.Pow(2, -i), mx, vx, a, b);
                if (true)
                {
                    // test extreme variances
                    GaussianFromMeanAndVarianceTest(mm, vm, mx, vx * 1e6, a, b);
                    GaussianFromMeanAndVarianceTest(mm, vm, mx, vx * 1e20, a, b);
                    //GaussianFromMeanAndVarianceTest(mm, vm, mx, Double.PositiveInfinity, a, b);
                    GaussianFromMeanAndVarianceTest(mm, vm * 1e6, mx, vx, a, b);
                    GaussianFromMeanAndVarianceTest(mm, vm * 1e20, mx, vx, a, b);
                }
            }
        }
        private void GaussianFromMeanAndVarianceTest(double mm, double vm, double mx, double vx, double a, double b)
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(mm, vm).Named("mean");
            Variable<double> variance = Variable.GammaFromShapeAndRate(a, b).Named("variance");
            Variable<double> x = Variable.GaussianFromMeanAndVariance(mean, variance).Named("x");
            Variable.ConstrainEqualRandom(x, new Gaussian(mx, vx));
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            double evExpected;
            Gaussian xExpected;
            Gamma vExpected;
            if (a == 1 || a == 2)
            {
                double c = System.Math.Sqrt(2 * b);
                double m = c * (mx - mm);
                double v = c * c * (vx + vm);
                double Z, mu, m2u;
                VarianceGammaTimesGaussianMoments(a, m, v, out Z, out mu, out m2u);
                evExpected = System.Math.Log(Z * c);
                double vu = m2u - mu * mu;
                double r = Double.IsPositiveInfinity(vx) ? 1.0 : vx / (vx + vm);
                double mp = r * (mu / c + mm) + (1 - r) * mx;
                double vp = r * r * vu / (c * c) + r * vm;
                xExpected = new Gaussian(mp, vp);
                double Zplus1, Zplus2;
                VarianceGammaTimesGaussianMoments(a + 1, m, v, out Zplus1, out mu, out m2u);
                VarianceGammaTimesGaussianMoments(a + 2, m, v, out Zplus2, out mu, out m2u);
                double vmp = a / b * Zplus1 / Z;
                double vm2p = a * (a + 1) / (b * b) * Zplus2 / Z;
                double vvp = vm2p - vmp * vmp;
                vExpected = Gamma.FromMeanAndVariance(vmp, vvp);
            }
            else
            {
                int n = 1000000;
                GaussianEstimator est = new GaussianEstimator();
                GammaEstimator vEst = new GammaEstimator();
                Gaussian xLike = new Gaussian(mx, vx);
                for (int i = 0; i < n; i++)
                {
                    double m = Gaussian.Sample(mm, 1 / vm);
                    double v = Rand.Gamma(a) / b;
                    double xSample = Gaussian.Sample(m, 1 / v);
                    double weight = System.Math.Exp(xLike.GetLogProb(xSample));
                    est.Add(xSample, weight);
                    vEst.Add(v, weight);
                }
                evExpected = System.Math.Log(est.mva.Count / n);
                xExpected = est.GetDistribution(new Gaussian());
                vExpected = vEst.GetDistribution(new Gamma());
            }
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Gaussian xActual = engine.Infer<Gaussian>(x);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Gamma vActual = engine.Infer<Gamma>(variance);
            Console.WriteLine("variance = {0} should be {1}", vActual, vExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-10) < 1e-4);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-4);
            Assert.True(vExpected.MaxDiff(vActual) < 1e-4);
        }

        private VariableArray<double> GaussianFromMeanAndVarianceSampleSizeModel(Variable<double> input, Range n, bool variance, double noisePrecision = 0)
        {
            var meanVar = Variable.GaussianFromMeanAndVariance(0, 10);
            var b = Variable.Array<double>(n);
            var cleanX = Variable.Array<double>(n);
            if (variance)
                cleanX[n] = Variable.GaussianFromMeanAndVariance(meanVar, input).ForEach(n);
            else
                cleanX[n] = Variable.GaussianFromMeanAndPrecision(meanVar, input).ForEach(n);
            var x = Variable.Array<double>(n);
            x[n] = Variable.GaussianFromMeanAndPrecision(cleanX[n], noisePrecision);

            return x;
        }

        // Commented out since this test is very slow and does not have any assertions
        //[Fact]
        internal void GaussianFromMeanAndVarianceSampleSize()
        {

            double mean = 2.0, variance = 2.0, noisePrecision = 10.0;

            var Nvar = Variable.Observed<int>(1);
            var n = new Range(Nvar);

            var variance_EP = Variable.GammaFromShapeAndRate(1, 1);
            var x_ep = GaussianFromMeanAndVarianceSampleSizeModel(variance_EP, n, true, noisePrecision);
            var ie_EP = new InferenceEngine();
            ie_EP.ShowWarnings = false;
            ie_EP.ShowProgress = false;

            var ev = Variable.Bernoulli(.5);
            var modelBlock = Variable.If(ev);
            var variance_MC = Variable.GammaFromShapeAndRate(1, 1);
            var x_mc = GaussianFromMeanAndVarianceSampleSizeModel(variance_MC, n, true, noisePrecision);
            modelBlock.CloseBlock();
            var ie_MC = new InferenceEngine();
            ie_MC.ShowWarnings = false;
            ie_MC.ShowProgress = false;

            var precision_VMP = Variable.GammaFromShapeAndRate(1, 1);
            var x_vmp = GaussianFromMeanAndVarianceSampleSizeModel(precision_VMP, n, false, noisePrecision);
            var ie_VMP = new InferenceEngine(new VariationalMessagePassing());
            ie_VMP.ShowWarnings = false;
            ie_VMP.ShowProgress = false;

            Console.WriteLine("N EP MC VMP EMP");
            for (int j = 0; j < 10; j++)
            {
                Rand.Restart(2);
                //int N = (int)Math.Pow(10, j + 1);
                int N = 10 * (j + 1);

                var data = Enumerable.Range(0, N).Select(i => Gaussian.Sample(mean, 1.0 / variance) + Gaussian.Sample(0, noisePrecision)).ToArray();

                Nvar.ObservedValue = N;
                x_ep.ObservedValue = data;
                x_mc.ObservedValue = data;
                x_vmp.ObservedValue = data;

                double epMean = double.NaN;
                try
                {
                    epMean = ie_EP.Infer<Gamma>(variance_EP).GetMean();
                }
                catch
                {
                }
                ;

                double vmpMean = ie_VMP.Infer<Gamma>(precision_VMP).GetMeanInverse();
                double varianceSample = 2;
                var ge = new GammaEstimator();

                Converter<double, double> f = vari =>
                {
                    variance_MC.ObservedValue = vari;
                    return ie_MC.Infer<Bernoulli>(ev).LogOdds;
                };
                int burnin = 1000, thin = 5, numSamples = 1000;
                var samples = new double[numSamples];
                for (int i = 0; i < burnin; i++)
                    varianceSample = NonconjugateVMP2Tests.SliceSampleUnivariate(varianceSample, f, lower_bound: 0);
                for (int i = 0; i < numSamples; i++)
                {
                    for (int k = 0; k < thin; k++)
                        varianceSample = NonconjugateVMP2Tests.SliceSampleUnivariate(varianceSample, f, lower_bound: 0);
                    samples[i] = varianceSample;
                    ge.Add(varianceSample);
                }
                double mcMean = samples.Sum() / numSamples;
                var empiricalMean = data.Sum() / data.Length;
                var empVar = data.Sum(o => o * o) / data.Length - empiricalMean * empiricalMean;
                Console.WriteLine(N + " " + epMean + " " + mcMean + " " + vmpMean + " " + empVar);
            }
        }

        private VariableArray<double> GaussianPlusNoiseModel(Variable<double> varianceOrPrecision, Range n, bool isVariance, VariableArray<double> noisePrecision)
        {
            var mean = Variable.GaussianFromMeanAndVariance(0, 100);
            var b = Variable.Array<double>(n);
            var cleanX = Variable.Array<double>(n);
            if (isVariance)
                cleanX[n] = Variable.GaussianFromMeanAndVariance(mean, varianceOrPrecision).ForEach(n);
            else
                cleanX[n] = Variable.GaussianFromMeanAndPrecision(mean, varianceOrPrecision).ForEach(n);
            var x = Variable.Array<double>(n);
            x[n] = Variable.GaussianFromMeanAndPrecision(cleanX[n], noisePrecision[n]);
            return x;
        }


        // Commented out since this test is very slow and does not have any assertions
        //[Fact]
        internal void GaussianFromMeanAndVarianceVaryNoise()
        {

            double mean = 2.0, variance = 2.0;

            var Nvar = Variable.Observed<int>(1);
            var n = new Range(Nvar);

            var noisePrecision = Variable.Array<double>(n);

            var variance_EP = Variable.GammaFromShapeAndRate(1, 1);
            var x_ep = GaussianPlusNoiseModel(variance_EP, n, true, noisePrecision);
            var ie_EP = new InferenceEngine();
            ie_EP.ShowWarnings = false;
            ie_EP.ShowProgress = false;

            var ev = Variable.Bernoulli(.5);
            var modelBlock = Variable.If(ev);
            var variance_MC = Variable.GammaFromShapeAndRate(1, 1);
            var x_mc = GaussianPlusNoiseModel(variance_MC, n, true, noisePrecision);
            modelBlock.CloseBlock();
            var ie_MC = new InferenceEngine();
            ie_MC.ShowWarnings = false;
            ie_MC.ShowProgress = false;

            var precision_VMP = Variable.GammaFromShapeAndRate(1, 1);
            var x_vmp = GaussianPlusNoiseModel(precision_VMP, n, false, noisePrecision);
            var ie_VMP = new InferenceEngine(new VariationalMessagePassing());
            ie_VMP.ShowWarnings = false;
            ie_VMP.ShowProgress = false;

            Console.WriteLine("N EP MC VMP EMP");
            for (int j = 0; j < 10; j++)
            {
                Rand.Restart(2);
                //int N = (int)Math.Pow(10, j + 1);
                int N = 10 * (j + 1);

                var noiseP = Enumerable.Range(0, N).Select(i =>
                {
                    if (i % 3 == 0)
                        return .1;
                    if (i % 3 == 1)
                        return 1;
                    else
                        return 10;
                }).ToArray();

                var data = noiseP.Select(i => Gaussian.Sample(mean, 1.0 / variance) + Gaussian.Sample(0, i)).ToArray();

                Nvar.ObservedValue = N;
                noisePrecision.ObservedValue = noiseP;
                x_ep.ObservedValue = data;
                x_mc.ObservedValue = data;
                x_vmp.ObservedValue = data;

                double epMean = double.NaN;
                try
                {
                    epMean = ie_EP.Infer<Gamma>(variance_EP).GetMean();
                }
                catch
                {
                }
                ;

                double vmpMean = ie_VMP.Infer<Gamma>(precision_VMP).GetMeanInverse();
                double varianceSample = 2;
                var ge = new GammaEstimator();

                Converter<double, double> f = vari =>
                {
                    variance_MC.ObservedValue = vari;
                    return ie_MC.Infer<Bernoulli>(ev).LogOdds;
                };
                int burnin = 1000, thin = 5, numSamples = 1000;
                var samples = new double[numSamples];
                for (int i = 0; i < burnin; i++)
                    varianceSample = NonconjugateVMP2Tests.SliceSampleUnivariate(varianceSample, f, lower_bound: 0);
                for (int i = 0; i < numSamples; i++)
                {
                    for (int k = 0; k < thin; k++)
                        varianceSample = NonconjugateVMP2Tests.SliceSampleUnivariate(varianceSample, f, lower_bound: 0);
                    samples[i] = varianceSample;
                    ge.Add(varianceSample);
                }
                double mcMean = samples.Sum() / numSamples;
                var empiricalMean = data.Sum() / data.Length;
                var empVar = data.Sum(o => o * o) / data.Length - empiricalMean * empiricalMean;
                Console.WriteLine(N + " " + epMean + " " + mcMean + " " + vmpMean + " " + empVar);
            }
        }

        // Commented out since this test is very slow and does not have any assertions
        //[Fact]
        internal void GaussianFromMeanAndVarianceVaryTruePrec()
        {

            double mean = 2.0;

            var Nvar = Variable.Observed<int>(50);
            var n = new Range(Nvar);

            var noisePrecision = 10;

            var variancePriors = new double[] { 1, 5, 10 }.Select(i => Gamma.FromShapeAndRate(i, i)).ToArray();

            var varPriorVar = Variable.Observed(new Gamma());

            var variance_EP = Variable<double>.Random(varPriorVar);
            var x_ep = GaussianFromMeanAndVarianceSampleSizeModel(variance_EP, n, true, noisePrecision);
            var ie_EP = new InferenceEngine();
            ie_EP.ShowWarnings = false;
            ie_EP.ShowProgress = false;

            var ev = Variable.Bernoulli(.5);
            var modelBlock = Variable.If(ev);
            var variance_MC = Variable<double>.Random(varPriorVar);
            var x_mc = GaussianFromMeanAndVarianceSampleSizeModel(variance_MC, n, true, noisePrecision);
            modelBlock.CloseBlock();
            var ie_MC = new InferenceEngine();
            ie_MC.ShowWarnings = false;
            ie_MC.ShowProgress = false;

            var precision_VMP = Variable.GammaFromShapeAndRate(1, 1);
            var x_vmp = GaussianFromMeanAndVarianceSampleSizeModel(precision_VMP, n, false, noisePrecision);
            var ie_VMP = new InferenceEngine(new VariationalMessagePassing());
            ie_VMP.ShowWarnings = false;
            ie_VMP.ShowProgress = false;

            var ie_EP_prec = new InferenceEngine();
            ie_EP_prec.ShowWarnings = false;
            ie_EP_prec.ShowProgress = false;

            Console.WriteLine("var " + variancePriors.Select(i => "EP" + i.Shape).Aggregate((i, j) => i + " " + j) + " "
                      + variancePriors.Select(i => "MC" + i.Shape).Aggregate((i, j) => i + " " + j) + " EPp VMP EMP");
            for (int j = 0; j < 10; j++)
            {
                Rand.Restart(2);
                //int N = (int)Math.Pow(10, j + 1);
                double variance = System.Math.Exp(-5 + j);

                var data = Enumerable.Range(0, Nvar.ObservedValue).Select(i => Gaussian.Sample(mean, 1.0 / variance) + Gaussian.Sample(0, noisePrecision)).ToArray();

                x_ep.ObservedValue = data;
                x_mc.ObservedValue = data;
                x_vmp.ObservedValue = data;

                var epMeans = variancePriors.Select(i =>
                {
                    double res = double.NaN;
                    varPriorVar.ObservedValue = i;
                    try
                    {
                        res = ie_EP.Infer<Gamma>(variance_EP).GetMean();
                    }
                    catch
                    {
                    }
                    ;
                    return res;
                }).ToArray();

                var mcMeans = variancePriors.Select(p =>
                {
                    double varianceSample = 2;
                    var ge = new GammaEstimator();
                    varPriorVar.ObservedValue = p;
                    Converter<double, double> f = vari =>
                    {
                        variance_MC.ObservedValue = vari;
                        return ie_MC.Infer<Bernoulli>(ev).LogOdds;
                    };
                    int burnin = 1000, thin = 5, numSamples = 1000;
                    var samples = new double[numSamples];
                    for (int i = 0; i < burnin; i++)
                        varianceSample = NonconjugateVMP2Tests.SliceSampleUnivariate(varianceSample, f, lower_bound: 0);
                    for (int i = 0; i < numSamples; i++)
                    {
                        for (int k = 0; k < thin; k++)
                            varianceSample = NonconjugateVMP2Tests.SliceSampleUnivariate(varianceSample, f, lower_bound: 0);
                        samples[i] = varianceSample;
                        ge.Add(varianceSample);
                    }
                    return samples.Sum() / numSamples;
                }).ToArray();


                double epMean2 = double.NaN;
                try
                {
                    epMean2 = ie_EP_prec.Infer<Gamma>(precision_VMP).GetMeanInverse();
                }
                catch
                {
                }
                ;

                double vmpMean = ie_VMP.Infer<Gamma>(precision_VMP).GetMeanInverse();

                var empiricalMean = data.Sum() / data.Length;
                var empVar = data.Sum(o => o * o) / data.Length - empiricalMean * empiricalMean;
                Console.Write("{0:G3} ", variance);
                foreach (var i in epMeans)
                    Console.Write("{0:G3} ", i);
                foreach (var i in mcMeans)
                    Console.Write("{0:G3} ", i);
                Console.WriteLine("{0:G3} {1:G3} {2:G3}", epMean2, vmpMean, empVar);
            }
        }



        private static void VarianceGammaTimesGaussianMoments(double a, double m, double v, out double sum, out double moment1, out double moment2)
        {
            int n = 1000000;
            double lowerBound = -20;
            double upperBound = 20;
            double range = upperBound - lowerBound;
            sum = 0;
            double sumx = 0, sumx2 = 0;
            for (int i = 0; i < n; i++)
            {
                double x = range * i / (n - 1) + lowerBound;
                double absx = System.Math.Abs(x);
                double diff = x - m;
                double logp = -0.5 * diff * diff / v - absx;
                double p = System.Math.Exp(logp);
                if (a == 1)
                { // do nothing
                }
                else if (a == 2)
                {
                    p *= 1 + absx;
                }
                else if (a == 3)
                {
                    p *= 3 + 3 * absx + absx * absx;
                }
                else if (a == 4)
                {
                    p *= 15 + 15 * absx + 6 * absx * absx + absx * absx * absx;
                }
                else
                    throw new ArgumentException("a is not in {1,2,3,4}");
                sum += p;
                sumx += x * p;
                sumx2 += x * x * p;
            }
            moment1 = sumx / sum;
            moment2 = sumx2 / sum;
            sum /= MMath.Gamma(a) * System.Math.Pow(2, a);
            sum /= MMath.Sqrt2PI * System.Math.Sqrt(v);
            double inc = range / (n - 1);
            sum *= inc;
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
                /////////////////////////////////////////
                // Reading Data and Checking Missing Data
                /////////////////////////////////////////

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


                /////////////////////////////////////////
                // The MultiState Probit Regression Model
                /////////////////////////////////////////                              

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

                /////////////////////////////////////////
                // Inference and Display of Results
                /////////////////////////////////////////        
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

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

}

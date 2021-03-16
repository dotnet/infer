// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;
using System.Diagnostics;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Serialization;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
    using Assert = Xunit.Assert;
    using GaussianArray = DistributionStructArray<Gaussian, double>;
    using GaussianArray2D = DistributionStructArray2D<Gaussian, double>;
    using GammaArray = DistributionStructArray<Gamma, double>;

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif


    public class VmpTests
    {
        internal void VmpClutterTest2()
        {
            var evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            var mean = Variable.GaussianFromMeanAndVariance(0.0, 1e2).Named("mean");
            var meanInit = Variable.Observed(new Gaussian(10, 10)).Named("meanInit");
            // if the initial variance is too large, VMP converges to the prior
            mean.InitialiseTo(meanInit);
            var b1 = Variable.Bernoulli(0.5).Named("b1");
            var x1 = Variable.New<double>().Named("x1");
            using (Variable.If(b1))
            {
                x1.SetTo(Variable.GaussianFromMeanAndPrecision(mean, 1));
            }
            using (Variable.IfNot(b1))
            {
                x1.SetTo(Variable.GaussianFromMeanAndPrecision(0, 0.1));
            }
            x1.ObservedValue = 10;
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;
            engine.Algorithm = new VariationalMessagePassing();
            engine.OptimiseForVariables = new List<IVariable>() { mean, evidence };

            if (false)
            {
                var gs = new GibbsSampling();
                gs.BurnIn = 0;
                engine.Algorithm = gs;
                mean.AddAttribute(QueryTypes.Samples);
                mean.AddAttribute(QueryTypes.Marginal);
                IList<double> meanSamples = engine.Infer<IList<double>>(mean, QueryTypes.Samples);
                Console.WriteLine(StringUtil.VerboseToString(meanSamples));
            }

            Gaussian meanActual = engine.Infer<Gaussian>(mean);
            Console.WriteLine("mean = {0}", meanActual);
            Console.WriteLine("evidence = {0}", engine.Infer<Bernoulli>(evidence).LogOdds);

            if (false)
            {
                // good solution requires initializing with mean inside [5,15] and variance <= 20
                double[] ms = EpTests.linspace(0, 20, 50);
                double[] logvs = EpTests.linspace(0, System.Math.Log(100), 50);
                Matrix ests = new Matrix(ms.Length, logvs.Length);
                for (int i = 0; i < ms.Length; i++)
                {
                    for (int j = 0; j < logvs.Length; j++)
                    {
                        meanInit.ObservedValue = new Gaussian(ms[i], System.Math.Exp(logvs[j]));
                        meanActual = engine.Infer<Gaussian>(mean);
                        ests[i, j] = meanActual.GetMean();
                    }
                }
                using (var writer = new MatlabWriter("VmpClutterTest.mat"))
                {
                    writer.Write("logvs", logvs);
                    writer.Write("ms", ms);
                    writer.Write("ests", ests);
                }
            }
        }

        // demonstrate multiple fixed points for VMP in Clutter problem
        internal void VmpClutterTest()
        {
            var evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            var mean = Variable.GaussianFromMeanAndVariance(0.0, 1e2).Named("mean");
            var meanInit = Variable.Observed(new Gaussian(10, 40)).Named("meanInit");
            // if the initial variance is too large, VMP converges to the prior
            mean.InitialiseTo(meanInit);
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
            x1.ObservedValue = 10;

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
            x2.ObservedValue = 10;
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;
            engine.Algorithm = new VariationalMessagePassing();
            engine.OptimiseForVariables = new List<IVariable>() { mean, evidence };

            Gaussian meanActual = engine.Infer<Gaussian>(mean);
            Console.WriteLine("mean = {0}", meanActual);
            Console.WriteLine("evidence = {0}", engine.Infer<Bernoulli>(evidence).LogOdds);

            if (false)
            {
                // good solution requires initializing with mean inside [5,15] and variance <= 20
                double[] ms = EpTests.linspace(0, 20, 50);
                double[] logvs = EpTests.linspace(0, System.Math.Log(100), 50);
                Matrix ests = new Matrix(ms.Length, logvs.Length);
                for (int i = 0; i < ms.Length; i++)
                {
                    for (int j = 0; j < logvs.Length; j++)
                    {
                        meanInit.ObservedValue = new Gaussian(ms[i], System.Math.Exp(logvs[j]));
                        meanActual = engine.Infer<Gaussian>(mean);
                        ests[i, j] = meanActual.GetMean();
                    }
                }
                using (var writer = new MatlabWriter("VmpClutterTest.mat"))
                {
                    writer.Write("logvs", logvs);
                    writer.Write("ms", ms);
                    writer.Write("ests", ests);
                }
            }
        }

        [Fact]
        public void WishartTimesGammaTest()
        {
            WishartTimesGamma(false);
            WishartTimesGamma(true);
        }

        private void WishartTimesGamma(bool flip)
        {
            PositiveDefiniteMatrix m2 = new PositiveDefiniteMatrix(new double[,] { { 2, 1 }, { 1, 2 } });
            Variable<PositiveDefiniteMatrix> x = Variable.WishartFromShapeAndScale(3, m2).Named("x");
            Variable<double> s = Variable.GammaFromShapeAndRate(6, 3).Named("s");
            Variable<PositiveDefiniteMatrix> y;
            if (flip) y = Variable<PositiveDefiniteMatrix>.Factor(Factor.Product, s, x);
            else y = Variable<PositiveDefiniteMatrix>.Factor(Factor.Product, x, s);
            y.Name = "y";
            Variable.ConstrainEqualRandom(y, new Wishart(3, m2));

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.NumberOfIterations = 1000;
            for (int xobs = 0; xobs < 2; xobs++)
            {
                if (xobs > 0) x.ObservedValue = m2;
                else x.ClearObservedValue();
                for (int sobs = 0; sobs < 2; sobs++)
                {
                    if (sobs > 0) s.ObservedValue = 2;
                    else s.ClearObservedValue();
                    Wishart xActual = engine.Infer<Wishart>(x);
                    Gamma sActual = engine.Infer<Gamma>(s);
                    Wishart yActual = engine.Infer<Wishart>(y);
                    Wishart xExpected = new Wishart(3, m2);
                    Gamma sExpected = Gamma.FromShapeAndRate(6, 3);
                    Wishart yExpected = new Wishart(3, m2);
                    Console.WriteLine(StringUtil.JoinColumns("x = ", xActual, " should be ", xExpected));
                    Console.WriteLine("s = {0} should be {1}", sActual, sExpected);
                    Console.WriteLine(StringUtil.JoinColumns("y = ", yActual, " should be ", yExpected));
                }
            }
            if (true)
            {
                s.ClearObservedValue();
                Gamma sActual, sExpected = new Gamma();
                for (int xobs = 0; xobs < 2; xobs++)
                {
                    for (int yobs = 0; yobs < 2; yobs++)
                    {
                        x.ObservedValue = new PositiveDefiniteMatrix(2, 2);
                        if (xobs > 0) x.ObservedValue[1] = 1;
                        y.ObservedValue = new PositiveDefiniteMatrix(2, 2);
                        if (yobs > 0) y.ObservedValue[1] = 1;
                        sActual = engine.Infer<Gamma>(s);
                        if (xobs == 0)
                            sExpected = Gamma.FromShapeAndRate(6, 3);
                        else if (xobs == 1 && yobs == 0)
                            sExpected = Gamma.PointMass(0);
                        else if (xobs == 1 && yobs == 1)
                            sExpected = Gamma.PointMass(1);
                        Console.WriteLine("s = {0} should be {1}", sActual, sExpected);
                    }
                }
            }
        }

        // Tests for Matt Wand
        internal void CauchyTest()
        {
            double sigsq_beta = 100;
            double A = 0.01, B = 0.01;
            int n = 100;
            // Specify priors for beta0 and beta1:
            Variable<double> beta0 = Variable.GaussianFromMeanAndVariance(0.0, sigsq_beta).Named("beta0");
            Variable<double> Precision = Variable.GammaFromShapeAndRate(0.5, 0.5);
            Variable<double> eta = Variable.GaussianFromMeanAndPrecision(0, Precision);
            Variable<double> xi = Variable.GaussianFromMeanAndVariance(0.0, 625);
            Variable<double> beta1 = xi * eta;
            xi.InitialiseTo(new Gaussian(10, 10)); // anything that doesn't have zero mean
            // Specify prior for the error precision:
            Variable<double> tau =
                Variable.GammaFromShapeAndScale(A, 1 / B).Named("tau");
            // Specify the likelihood:
            Range item = new Range(n).Named("item");
            VariableArray<double> y = Variable.Array<double>(item).Named("y");
            VariableArray<double> x = Variable.Array<double>(item).Named("x");
            VariableArray<double> meany =
                Variable.Array<double>(item).Named("meany");
            meany[item] = beta0 + beta1 * x[item];
            y[item] = Variable.GaussianFromMeanAndPrecision(meany[item], tau);

            x.ObservedValue = Util.ArrayInit(n, i => Rand.Normal());
            y.ObservedValue = Util.ArrayInit(n, i => x.ObservedValue[i] * 3 + 1);
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(engine.Infer(beta1));
        }

        internal void CauchyTest2()
        {
            double sigsq_beta = 100;
            int n = 100;
            double tau = 1;
            // Specify priors for beta0 and beta1:
            Variable<double> beta0 =
                Variable.GaussianFromMeanAndVariance(0.0, sigsq_beta).Named("beta0");
            Variable<double> beta1 =
                Variable.GaussianFromMeanAndVariance(0.0, sigsq_beta).Named("beta1");
            // Specify the likelihood:
            Variable<double> Precision = Variable.GammaFromShapeAndRate(0.5, 0.5);
            Range item = new Range(n).Named("item");
            VariableArray<double> y = Variable.Array<double>(item).Named("y");
            VariableArray<double> x = Variable.Array<double>(item).Named("x");
            VariableArray<double> meany =
                Variable.Array<double>(item).Named("meany");
            meany[item] = beta0 + beta1 * x[item];
            if (true)
            {
                // invSigmaSquared has a Cauchy^2 distribution
                Variable<double> invSigmaSquared = Variable.GammaFromShapeAndRate(0.5, Variable.GammaFromShapeAndRate(0.5, 0.0016));
                y[item] = Variable.GaussianFromMeanAndPrecision(meany[item], invSigmaSquared);
            }
            if (false)
            {
                Variable<double> invSigma = Variable.GaussianFromMeanAndPrecision(0, Precision);
                invSigma.InitialiseTo(new Gaussian(10, 1)); // anything that doesn't have zero mean
                //eta.ObservedValue = 1;
                VariableArray<double> temp = Variable.Array<double>(item).Named("temp");
                temp[item] = Variable.GaussianFromMeanAndPrecision((y[item] - meany[item]) * invSigma, tau);
                temp.ObservedValue = Util.ArrayInit(n, i => 0.0);
            }

            x.ObservedValue = Util.ArrayInit(n, i => Rand.Normal());
            y.ObservedValue = Util.ArrayInit(n, i => x.ObservedValue[i] * 3 + 1);
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.NumberOfIterations = 100;
            Console.WriteLine("beta0 = {0}", engine.Infer(beta0));
            Console.WriteLine("beta1 = {0}", engine.Infer(beta1));
        }

        /// <summary>
        /// this test fails if SumOp.ArrayAL has a trigger on array. 
        /// </summary>
        [Fact]
        public void SumTriggerTest()
        {
            int P = 2;
            Range dim = new Range(P).Named("dim");

            // weight vector
            VariableArray<double> w = Variable.Array<double>(dim).Named("w");
            VariableArray<bool> probOn = Variable.Array<bool>(dim).Named("p");

            using (Variable.ForEach(dim))
            {
                probOn[dim] = Variable.Bernoulli(0.5);
                using (Variable.If(probOn[dim]))
                {
                    w[dim] = Variable.GaussianFromMeanAndPrecision(0, 1);
                }
                using (Variable.IfNot(probOn[dim]))
                {
                    w[dim] = Variable.GaussianFromMeanAndPrecision(0, 10000);
                }
            }
            var x = Variable.Array<double>(dim).Named("x");
            var wx = Variable.Array<double>(dim).Named("wx");
            wx[dim] = w[dim] * x[dim];
            var sum = Variable.Sum(wx);
            var y = Variable.GaussianFromMeanAndVariance(sum, 1);

            InferenceEngine ieVMP = new InferenceEngine(new VariationalMessagePassing());
            //ieVMP.Compiler.UnrollLoops = true;
            y.ObservedValue = 1;
            x.ObservedValue = Util.ArrayInit(P, i => 1.0);

            Gaussian wExpected = new Gaussian(0.1854, 0.2687);

            var wActual = ieVMP.Infer<DistributionArray<Gaussian>>(w);
            Console.WriteLine(wActual);
            for (int i = 0; i < wActual.Count; i++)
            {
                Assert.True(wExpected.MaxDiff(wActual[i]) < 1e-3);
            }
        }

        /// <summary>
        /// This test fails if GateExitOp.CasesAL has a trigger on exit
        /// </summary>
        [Fact]
        public void GateExitTriggerTest()
        {
            var w1 = Variable.New<double>().Named("w1");
            var w2 = Variable.New<double>().Named("w2");
            var c1 = Variable.Bernoulli(0.5).Named("c1");
            var c2 = Variable.Bernoulli(0.5).Named("c2");
            using (Variable.If(c1))
            {
                w1.SetTo(Variable.GaussianFromMeanAndPrecision(0, 1));
            }
            using (Variable.IfNot(c1))
            {
                w1.SetTo(Variable.GaussianFromMeanAndPrecision(0, 10000));
            }
            using (Variable.If(c1))
            {
                w2.SetTo(Variable.GaussianFromMeanAndPrecision(0, 1));
            }
            using (Variable.IfNot(c1))
            {
                w2.SetTo(Variable.GaussianFromMeanAndPrecision(0, 10000));
            }
            var sum = w1 + w2;
            var y = Variable.GaussianFromMeanAndVariance(sum, 1);

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.Compiler.OptimiseInferenceCode = false;
            y.ObservedValue = 1;

            // if c1 is used for both, should be
            Gaussian w1Expected = new Gaussian(0.1713, 0.2484);
            Gaussian w2Expected = new Gaussian(0.1713, 0.2484);
            Gaussian w1Actual = engine.Infer<Gaussian>(w1);
            Gaussian w2Actual = engine.Infer<Gaussian>(w2);
            Console.WriteLine("w1 = {0}", w1Actual);
            Console.WriteLine("w2 = {0}", w2Actual);
            Assert.True(w1Expected.MaxDiff(w1Actual) < 3e-4);
            Assert.True(w2Expected.MaxDiff(w2Actual) < 3e-4);
        }

        // There are two bugs here: EP does not converge for small N, and VMP never compiles
        [Fact]
        public void LinearRegressionSpikeAndSlabTest()
        {
            int P = 8, N = 100;
            //Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            //IfBlock block = Variable.If(evidence);
            Range dim = new Range(P).Named("dim");

            // weight vector
            VariableArray<double> w = Variable.Array<double>(dim).Named("w");
            VariableArray<bool> probOn = Variable.Array<bool>(dim).Named("p");

            using (Variable.ForEach(dim))
            {
                probOn[dim] = Variable.Bernoulli(0.5);
                using (Variable.If(probOn[dim]))
                {
                    w[dim] = Variable.GaussianFromMeanAndPrecision(0, 1);
                }
                using (Variable.IfNot(probOn[dim]))
                {
                    w[dim] = Variable.GaussianFromMeanAndPrecision(0, 10000 /*double.PositiveInfinity*/);
                }
            }
            Range item = new Range(N).Named("item");

            // Covariates
            var x = Variable.Array(Variable.Array<double>(dim), item).Named("x");
            x[item][dim] = Variable.GaussianFromMeanAndPrecision(2.0, 0.5).ForEach(item, dim);

            var wx = Variable.Array(Variable.Array<double>(dim), item).Named("wx");
            wx[item][dim] = w[dim] * x[item][dim];
            var sum = Variable.Array<double>(item).Named("sum");
            sum[item] = Variable.Sum(wx[item]);

            // Observed labels
            var y = Variable.Array<double>(item).Named("y");
            y[item] = Variable.GaussianFromMeanAndPrecision(sum[item], 10);
            //block.CloseBlock();

            InferenceEngine ieVMP = new InferenceEngine(new VariationalMessagePassing());
            InferenceEngine ieEP = new InferenceEngine(new ExpectationPropagation());
            int iterations = 100;
            ieEP.ShowProgress = false;
            ieEP.ShowTimings = true;
            ieEP.NumberOfIterations = iterations;
            ieVMP.ShowProgress = false;
            ieVMP.ShowTimings = true;
            ieVMP.NumberOfIterations = iterations;
            // Fails with unrolling because FastSumOp.ArrayAverageLogarithm assumes the array is updated in parallel
            ieVMP.Compiler.UnrollLoops = false;
            ieVMP.Compiler.UnrollLoops = true;
            int repeats = 10;
            double[] resultsVMP = new double[10];
            double[] resultsEP = new double[10];
            double[] evidenceVMP = new double[10];
            double[] evidenceEP = new double[10];
            for (int seed = 0; seed < repeats; seed++)
            {
                // Set random set
                Rand.Restart(seed);
                double[][] xTrue = new double[N][]; // data drawn from the model
                double[] wTrue = new double[P]; // "true" weights
                var yObs = new double[N]; // observed data
                double[] g = new double[N];
                for (int i = 0; i < P; i++)
                    wTrue[i] = Rand.Double() > 0.5 ? Rand.Normal(0, 1) : 0.0;

                // sample data from the model
                for (int j = 0; j < N; j++)
                {
                    xTrue[j] = new double[P];
                    g[j] = 0;
                    for (int i = 0; i < P; i++)
                    {
                        double mean = Rand.Normal(0, 1);
                        double std = Rand.Gamma(1);
                        xTrue[j][i] = Rand.Normal(mean, std);
                        g[j] += xTrue[j][i] * wTrue[i];
                    }
                    yObs[j] = g[j] + Rand.Normal(0, .1);
                }

                // set observed values
                y.ObservedValue = yObs;
                x.ObservedValue = xTrue;

                double G_mean, G_var;
                DistributionArray<Gaussian> G;
                if (true)
                {
                    // run inference
                    Console.WriteLine("EP");
                    G = ieEP.Infer<DistributionArray<Gaussian>>(w);
                    var p = ieEP.Infer<Bernoulli[]>(probOn);
                    //evidenceEP[seed] = ieEP.Infer<Bernoulli>(evidence).LogOdds;
                    resultsEP[seed] = 0.0;
                    for (int i = 0; i < P; i++)
                    {
                        G[i].GetMeanAndVariance(out G_mean, out G_var);
                        resultsEP[seed] += G[i].GetLogProb(wTrue[i]);
                        var pMean = p[i].GetMean().ToString("G4");
                        Console.WriteLine("True w: " + wTrue[i] + " inferred: ProbOn:" + pMean + " Marg: N(" + G_mean + "," + System.Math.Sqrt(G_var) + "^2)");
                    }
                }
                if (true)
                {
                    Console.WriteLine("VMP");
                    G = ieVMP.Infer<DistributionArray<Gaussian>>(w);
                    //evidenceVMP[seed] = ieVMP.Infer<Bernoulli>(evidence).LogOdds;
                    resultsVMP[seed] = 0.0;
                    for (int i = 0; i < P; i++)
                    {
                        G[i].GetMeanAndVariance(out G_mean, out G_var);
                        resultsVMP[seed] += G[i].GetLogProb(wTrue[i]);
                        Console.WriteLine("True w: " + wTrue[i] + " inferred: " + G_mean + " +/- " + System.Math.Sqrt(G_var));
                    }
                }
            }
        }

        /// <summary>
        /// Test the convergence rate of VMP
        /// </summary>
        [Fact]
        public void AdditiveSparseBlockmodel()
        {
            var YObs = new bool[5][];
            YObs[0] = new bool[] { false, true, true, false, false };
            YObs[1] = new bool[] { true, false, true, false, false };
            YObs[2] = new bool[] { true, true, false, false, false };
            YObs[3] = new bool[] { false, false, false, false, true };
            YObs[4] = new bool[] { false, false, false, true, false };
            int K = 2; // Number of blocks
            int C = 2;
            int N = YObs.Length; // Number of nodes

            // Ranges
            Range n = new Range(N).Named("n"); // Range for initiator
            Range n2 = n.Clone().Named("n2"); // Range for receiver
            Range k = new Range(K).Named("k"); // Range for initiator block membership
            Range k2 = k.Clone().Named("k2"); // Range for receiver block membership
            var c = new Range(C).Named("c");

            // The model
            var Y = Variable.Array(Variable.Array<bool>(n2), n).Named("data"); // Interaction matrix
            var pi = Variable.Array<Vector>(c).Named("pi");
            pi[c] = Variable.DirichletSymmetric(k, 1.0).ForEach(c).Named("pi"); // Block-membership probability vector
            var z = Variable.Array(Variable.Array<int>(c), n).Named("z");
            z[n][c] = Variable.Discrete(pi[c]).ForEach(n); // Draw initiator membership indicator
            var z2 = Variable.Array(Variable.Array<int>(c), n2).Named("z2");
            z2[n2][c] = Variable.Copy(z[n2][c]);
            z2.SetValueRange(k2);
            var B = Variable.Array(Variable.Array<double>(k, k2), c).Named("B"); // Link probability matrix
            B[c][k, k2] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(c, k, k2);

            var ibpAlpha = 1.0;
            var betas = Variable.Array<double>(c).Named("betas");
            betas[c] = Variable.Beta(ibpAlpha / (double)K, 1.0).ForEach(c);
            var ibpZ = Variable.Array<bool>(n, c).Named("ibpz");
            ibpZ[n, c] = Variable.Bernoulli(betas[c]).ForEach(n);

            var temp = Variable.Array(Variable.Array(Variable.Array<double>(c), n2), n).Named("temp");
            using (Variable.ForEach(n))
            {
                using (Variable.ForEach(c))
                {
                    using (Variable.If(ibpZ[n, c]))
                    {
                        using (Variable.Switch(z[n][c]))
                        using (Variable.ForEach(n2))
                        {
                            using (Variable.If(ibpZ[n2, c]))
                            using (Variable.Switch(z2[n2][c]))
                                temp[n][n2][c] = Variable.GaussianFromMeanAndPrecision(B[c][z[n][c], z2[n2][c]], 10);
                            using (Variable.IfNot(ibpZ[n2, c]))
                                temp[n][n2][c] = Variable.GaussianFromMeanAndPrecision(0, 10);
                        }
                    }
                    using (Variable.IfNot(ibpZ[n, c]))
                    {
                        using (Variable.ForEach(n2))
                            temp[n][n2][c] = Variable.GaussianFromMeanAndPrecision(0, 10);
                    }
                }
                using (Variable.ForEach(n2))
                {
                    var sum = Variable.Sum(temp[n][n2]).Named("sum");
                    var g = Variable.Logistic(sum).Named("g");
                    Y[n][n2] = Variable.Bernoulli(g); // Sample interaction value
                }
            }

            // Initialise to break symmetry
            var zInit = new Discrete[N][];
            for (int i = 0; i < N; i++)
            {
                zInit[i] = new Discrete[C];
                for (int j = 0; j < C; j++)
                {
                    zInit[i][j] = Discrete.PointMass(Rand.Int(K), K);
                }
            }

            // Hook up the data
            Y.ObservedValue = YObs;

            // Infer
            var engine = new InferenceEngine(new VariationalMessagePassing());
            engine.NumberOfIterations = 50;
            //engine.ShowFactorGraph = true;
            //z.InitialiseTo(Distribution<int>.Array(zInit));
            var BExpected = new Gaussian(-0.0002843, 0.98371817);
            var posteriorPi = engine.Infer<Dirichlet[]>(pi);
            var posteriorB = engine.Infer<IList<GaussianArray2D>>(B);
            Assert.True(BExpected.MaxDiff(posteriorB[0][0, 0]) < 1e-7);
        }


        // Demonstrates scaling with length of cycle when you don't repeat nodes
        internal void CycleTest()
        {
            CycleTest2(7);
            return;
            for (int i = 4; i < 100; i += 2)
            {
                CycleTest2(i + 1);
            }
            CycleTest2(200);
        }

        internal void CycleTest(int n)
        {
            int halfN = (int)System.Math.Floor(0.5 * n);
            Bernoulli[] priors = new Bernoulli[n];
            Variable<bool>[] x = new Variable<bool>[n];
            for (int i = 0; i < n; i++)
            {
                double xPrior = 0.5;
                if (i == 0) xPrior = 0.1;
                if (i == halfN) xPrior = 0.9;
                priors[i] = new Bernoulli(xPrior);
                x[i] = Variable.Bernoulli(xPrior).Named("x" + i);
            }
            Bernoulli xLike = new Bernoulli(0.7);
            for (int i = 0; i < n; i++)
            {
                int prev = (i == 0) ? (n - 1) : (i - 1);
                Variable.ConstrainEqualRandom(x[i] == x[prev], xLike);
            }
            InferenceEngine engine = new InferenceEngine();
            //engine.ShowSchedule = true;
            engine.ShowProgress = false;
            engine.Algorithm = new VariationalMessagePassing();
            Bernoulli[] xPost = new Bernoulli[n];
            if (false)
            {
                for (int iter = 0; iter < 1000; iter++)
                {
                    engine.NumberOfIterations = iter + 1;
                    double maxDiff = 0;
                    for (int i = 0; i < n; i++)
                    {
                        Bernoulli newPost = engine.Infer<Bernoulli>(x[i]);
                        maxDiff = System.Math.Max(maxDiff, newPost.MaxDiff(xPost[i]));
                        xPost[i] = newPost;
                    }
                    //Console.WriteLine(maxDiff);
                    if (maxDiff < 1e-8)
                    {
                        Console.WriteLine("converged in {0} iterations", iter + 1);
                        break;
                    }
                }
                Console.WriteLine("x[0] = {0}", xPost[0]);
                Console.WriteLine("x[{0}] = {1}", halfN, xPost[halfN]);
                Console.WriteLine("x[{0}] = {1}", n - 1, xPost[n - 1]);
            }

            for (int i = 0; i < n; i++)
            {
                //xPost[i] = priors[i];
                xPost[i] = Bernoulli.Uniform();
            }
            for (int iter = 0; iter < 1000; iter++)
            {
                double maxDiff = 0;
                if (iter % 2 == 0)
                {
                    for (int i = 0; i < n; i++)
                    {
                        int prev = (i == 0) ? (n - 1) : (i - 1);
                        int next = (i == n - 1) ? 0 : (i + 1);
                        Bernoulli fromPrev = BooleanAreEqualOp.AAverageLogarithm(xLike, xPost[prev]);
                        Bernoulli fromNext = BooleanAreEqualOp.AAverageLogarithm(xLike, xPost[next]);
                        Bernoulli newPost = priors[i] * fromPrev * fromNext;
                        maxDiff = System.Math.Max(maxDiff, newPost.MaxDiff(xPost[i]));
                        xPost[i] = newPost;
                    }
                }
                else
                {
                    for (int i = n - 1; i >= 0; i--)
                    {
                        int prev = (i == 0) ? (n - 1) : (i - 1);
                        int next = (i == n - 1) ? 0 : (i + 1);
                        Bernoulli fromPrev = BooleanAreEqualOp.AAverageLogarithm(xLike, xPost[prev]);
                        Bernoulli fromNext = BooleanAreEqualOp.AAverageLogarithm(xLike, xPost[next]);
                        Bernoulli newPost = priors[i] * fromPrev * fromNext;
                        maxDiff = System.Math.Max(maxDiff, newPost.MaxDiff(xPost[i]));
                        xPost[i] = newPost;
                    }
                }
                //Console.WriteLine(maxDiff);
                if (maxDiff < 1e-8)
                {
                    Console.WriteLine("converged in {0} iterations", iter + 1);
                    break;
                }
                if (false)
                {
                    Console.WriteLine("x[0] = {0}", xPost[0]);
                    Console.WriteLine("x[{0}] = {1}", halfN, xPost[halfN]);
                    Console.WriteLine("x[{0}] = {1}", n - 1, xPost[n - 1]);
                }
            }
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 429
#endif

        private void CycleTest2(int n)
        {
            int halfN = (int)System.Math.Floor(0.5 * n);
            Gaussian[] priors = new Gaussian[n];
            Variable<double>[] x = new Variable<double>[n];
            for (int i = 0; i < n; i++)
            {
                double xPrior = 0;
                if (i == 0) xPrior = 1;
                if (i == halfN) xPrior = -1;
                priors[i] = new Gaussian(xPrior, 1);
                x[i] = Variable.GaussianFromMeanAndVariance(xPrior, 1).Named("x" + i);
            }
            Gaussian xLike = new Gaussian(0, 1);
            for (int i = 0; i < n; i++)
            {
                int prev = (i == 0) ? (n - 1) : (i - 1);
                Variable.ConstrainEqualRandom(x[i] - x[prev], xLike);
            }
            InferenceEngine engine = new InferenceEngine();
            //engine.ShowSchedule = true;
            engine.ShowProgress = false;
            engine.Algorithm = new VariationalMessagePassing();
            Gaussian[] xPost = new Gaussian[n];
            if (true)
            {
                for (int iter = 0; iter < 1000; iter++)
                {
                    engine.NumberOfIterations = iter + 1;
                    double maxDiff = 0;
                    for (int i = 0; i < n; i++)
                    {
                        Gaussian newPost = engine.Infer<Gaussian>(x[i]);
                        maxDiff = System.Math.Max(maxDiff, newPost.MaxDiff(xPost[i]));
                        xPost[i] = newPost;
                    }
                    //Console.WriteLine(maxDiff);
                    if (maxDiff < 1e-8)
                    {
                        Console.WriteLine("converged in {0} iterations", iter + 1);
                        break;
                    }
                }
                Console.WriteLine("x[0] = {0}", xPost[0]);
                Console.WriteLine("x[{0}] = {1}", halfN, xPost[halfN]);
                Console.WriteLine("x[{0}] = {1}", n - 1, xPost[n - 1]);
            }

            for (int i = 0; i < n; i++)
            {
                //xPost[i] = priors[i];
                xPost[i] = Gaussian.Uniform();
            }
            for (int iter = 0; iter < 1000; iter++)
            {
                double maxDiff = 0;
                if (true || iter % 2 == 0)
                {
                    for (int i = 0; i < n; i++)
                    {
                        int prev = (i == 0) ? (n - 1) : (i - 1);
                        int next = (i == n - 1) ? 0 : (i + 1);
                        Gaussian fromPrev = DoubleMinusVmpOp.AAverageLogarithm(xLike, xPost[prev]);
                        Gaussian fromNext = DoubleMinusVmpOp.BAverageLogarithm(xLike, xPost[next]);
                        Gaussian newPost = priors[i] * fromPrev * fromNext;
                        maxDiff = System.Math.Max(maxDiff, newPost.MaxDiff(xPost[i]));
                        xPost[i] = newPost;
                    }
                }
                else
                {
                    for (int i = n - 1; i >= 0; i--)
                    {
                        int prev = (i == 0) ? (n - 1) : (i - 1);
                        int next = (i == n - 1) ? 0 : (i + 1);
                        Gaussian fromPrev = DoubleMinusVmpOp.AAverageLogarithm(xLike, xPost[prev]);
                        Gaussian fromNext = DoubleMinusVmpOp.BAverageLogarithm(xLike, xPost[next]);
                        Gaussian newPost = priors[i] * fromPrev * fromNext;
                        maxDiff = System.Math.Max(maxDiff, newPost.MaxDiff(xPost[i]));
                        xPost[i] = newPost;
                    }
                }
                //Console.WriteLine(maxDiff);
                if (maxDiff < 1e-8)
                {
                    Console.WriteLine("converged in {0} iterations", iter + 1);
                    break;
                }
                if (true)
                {
                    Console.WriteLine("x[0] = {0}", xPost[0]);
                    Console.WriteLine("x[{0}] = {1}", halfN, xPost[halfN]);
                    Console.WriteLine("x[{0}] = {1}", n - 1, xPost[n - 1]);
                }
            }
            if (false)
            {
                Console.WriteLine("x[0] = {0}", xPost[0]);
                Console.WriteLine("x[{0}] = {1}", halfN, xPost[halfN]);
                Console.WriteLine("x[{0}] = {1}", n - 1, xPost[n - 1]);
            }
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 429
#endif

        [Fact]
        public void RotateTest()
        {
            Variable<double> x = Variable.GaussianFromMeanAndVariance(2, 0).Named("x");
            Variable<double> y = Variable.GaussianFromMeanAndVariance(1, 0).Named("y");
            Variable<double> angle = Variable<double>.Random(new WrappedGaussian(0, 100)).Named("angle");
            Variable<Vector> rot = Variable.Rotate(x, y, angle).Named("rot");
            Variable<double> rotX = Variable.GetItem(rot, 0).Named("rotX");
            Variable<double> rotY = Variable.GetItem(rot, 1).Named("rotY");
            Variable<double> obsX = Variable.GaussianFromMeanAndVariance(rotX, 1).Named("obsX");
            Variable<double> obsY = Variable.GaussianFromMeanAndVariance(rotY, 1).Named("obsY");
            obsX.ObservedValue = -2;
            obsY.ObservedValue = 3;

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            WrappedGaussian angleActual = engine.Infer<WrappedGaussian>(angle);
            WrappedGaussian angleExpected = new WrappedGaussian(1.693, 0.1239);
            Console.WriteLine("angle = {0} should be {1}", angleActual, angleExpected);
            Assert.True(angleExpected.MaxDiff(angleActual) < 1e-2);
        }

        internal void RotateTest2()
        {
            Range item = new Range(2);
            VariableArray<double> x = Variable.Array<double>(item).Named("x");
            VariableArray<double> y = Variable.Array<double>(item).Named("y");
            Variable<double> angle = Variable<double>.Random(new WrappedGaussian(0, 100)).Named("angle");
            VariableArray<double> multipliers = Variable.Observed(new double[] { 1, 2 }, item).Named("multipliers");
            Variable<Vector> rot = Variable.Rotate(x[item], y[item], angle * multipliers[item]).Named("rot");
            Variable<double> rotX = Variable.GetItem(rot, 0).Named("rotX");
            Variable<double> rotY = Variable.GetItem(rot, 1).Named("rotY");
            VariableArray<double> obsX = Variable.Array<double>(item).Named("obsX");
            VariableArray<double> obsY = Variable.Array<double>(item).Named("obsY");
            obsX[item] = Variable.GaussianFromMeanAndVariance(rotX, 1);
            obsY[item] = Variable.GaussianFromMeanAndVariance(rotY, 1);
            obsX.ObservedValue = new double[] { -1, -2 };
            obsY.ObservedValue = new double[] { 2, -1 };
            x.ObservedValue = new double[] { 2, 2 };
            y.ObservedValue = new double[] { 1, 1 };

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            WrappedGaussian angleActual = engine.Infer<WrappedGaussian>(angle);
            WrappedGaussian angleExpected = new WrappedGaussian(System.Math.PI / 2, 0.1);
            Console.WriteLine("angle = {0} should be {1}", angleActual, angleExpected);
            //Assert.True(angleExpected.MaxDiff(angleActual) < 1e-2);
        }

        internal void RotateTest3()
        {
            Variable<double> x = Variable.GaussianFromMeanAndVariance(2, 0).Named("x");
            Variable<double> y = Variable.GaussianFromMeanAndVariance(0, 0).Named("y");
            Variable<double> angle = Variable<double>.Random(new WrappedGaussian(System.Math.PI / 4, 0.1)).Named("angle");
            Variable<Vector> rot = Variable.Rotate(x, y, angle).Named("rot");
            Variable<double> rotX = Variable.GetItem(rot, 0).Named("rotX");
            Variable<double> rotY = Variable.GetItem(rot, 1).Named("rotY");
            Variable<double> obsX = Variable.GaussianFromMeanAndVariance(rotX, 1).Named("obsX");
            Variable<double> obsY = Variable.GaussianFromMeanAndVariance(rotY, 1).Named("obsY");
            //angle.ObservedValue = Math.PI/2;

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(engine.Infer<VectorGaussian>(rot));
            Console.WriteLine(engine.Infer<Gaussian>(rotX));
        }

        [Fact]
        public void VmpTruncatedGaussianTest()
        {
            var ev = Variable.Bernoulli(0.5).Named("ev");
            var model = Variable.If(ev);
            Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 1).Named("x");
            Variable.ConstrainPositive(x);
            model.CloseBlock();

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Gaussian xActual = engine.Infer<Gaussian>(x);
            Console.WriteLine(xActual);
            double logEvidence = engine.Infer<Bernoulli>(ev).LogOdds;
            Console.WriteLine(logEvidence);


            var ev2 = Variable.Bernoulli(0.5).Named("ev");
            var model2 = Variable.If(ev2);
            Variable<double> x2 = Variable.GaussianFromMeanAndVariance(0, 1).Named("x2");
            var x3 = Variable.Copy(x2);
            x3.AddAttribute(new MarginalPrototype(new TruncatedGaussian()));
            Variable.ConstrainPositive(x3);
            model2.CloseBlock();

            Gaussian x2Actual = engine.Infer<Gaussian>(x2);
            Console.WriteLine(x2Actual);
            double logEvidence2 = engine.Infer<Bernoulli>(ev2).LogOdds;
            Console.WriteLine(logEvidence2);

            Assert.True(xActual.MaxDiff(x2Actual) < 1e-8);
            Assert.True(MMath.AbsDiff(logEvidence, logEvidence2) < 1e-8);
            Assert.True(MMath.AbsDiff(logEvidence, System.Math.Log(.5)) < 1e-8);
        }

        [Fact]
        public void VmpTruncatedGaussianMultipleUse()
        {
            Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 1).Named("x2");
            var y = Variable.Copy(x);
            y.AddAttribute(new MarginalPrototype(new TruncatedGaussian()));
            Variable.ConstrainEqualRandom(y, new TruncatedGaussian(Gaussian.FromMeanAndPrecision(-1, 1)));
            Variable.ConstrainPositive(y);

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Gaussian xActual = engine.Infer<Gaussian>(x);
            Console.WriteLine(xActual);

            Variable<double> x2 = Variable.GaussianFromMeanAndVariance(0, 1).Named("x2");
            var x4 = Variable.GaussianFromMeanAndPrecision(x2, 1);
            x4.ObservedValue = -1;
            var y2 = Variable.Copy(x2);
            y2.AddAttribute(new MarginalPrototype(new TruncatedGaussian()));
            Variable.ConstrainPositive(y2);

            Gaussian x2Actual = engine.Infer<Gaussian>(x2);
            Console.WriteLine(x2Actual);
            Assert.True(xActual.MaxDiff(x2Actual) < 1e-8);

            Variable<double> x3 = Variable.GaussianFromMeanAndVariance(0, 1).Named("x2");
            Variable.ConstrainEqualRandom(x3, Gaussian.FromMeanAndPrecision(-1, 1));
            Variable.ConstrainPositive(x3);
            Gaussian x3Actual = engine.Infer<Gaussian>(x3);
            Console.WriteLine(x3Actual);
            Assert.True(x3Actual.MaxDiff(x2Actual) < 1e-8);
        }

        //[Fact]
        internal void VmpTruncatedGaussianMixTest()
        {
            var ev = Variable.Bernoulli(0.5).Named("ev");
            var model = Variable.If(ev);
            Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 1).Named("x");
            var p = Variable.IsPositive(x).Named("p");
            Variable.ConstrainEqualRandom(p, new Bernoulli(0.25));
            model.CloseBlock();

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Gaussian xActual = engine.Infer<Gaussian>(x);
            Console.WriteLine(xActual);
            double logEvidence = engine.Infer<Bernoulli>(ev).LogOdds;
            Console.WriteLine(logEvidence);

            InferenceEngine ep = new InferenceEngine();
            Gaussian x2Actual = ep.Infer<Gaussian>(x);
            Console.WriteLine(x2Actual);
            double logEvidence2 = ep.Infer<Bernoulli>(ev).LogOdds;
            Console.WriteLine(logEvidence2);

            Assert.True(xActual.MaxDiff(x2Actual) < 1e-8);
            Assert.True(MMath.AbsDiff(logEvidence2, System.Math.Log(.5)) < 1e-8);
            //Assert.True(MMath.AbsDiff(logEvidence, logEvidence2) < 1e-8); maybe these shouldn't be the same?
        }

        // Tests a model where a variable factor has different types of incoming message.
        [Fact]
        public void VmpTruncatedGaussianTest2()
        {
            var ev = Variable.Bernoulli(0.5).Named("ev");
            var model = Variable.If(ev);
            Variable<double> x = Variable.GaussianFromMeanAndPrecision(Variable.GaussianFromMeanAndPrecision(0, 1),
                                                                       Variable.GammaFromShapeAndRate(1, 1)).Named("x");
            Variable.ConstrainPositive(x);
            model.CloseBlock();

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Gaussian xActual = engine.Infer<Gaussian>(x);
            Console.WriteLine(xActual);
            double logEvidence = engine.Infer<Bernoulli>(ev).LogOdds;
            Console.WriteLine(logEvidence);


            var ev2 = Variable.Bernoulli(0.5).Named("ev");
            var model2 = Variable.If(ev2);
            Variable<double> x2 = Variable.GaussianFromMeanAndPrecision(Variable.GaussianFromMeanAndPrecision(0, 1),
                                                                        Variable.GammaFromShapeAndRate(1, 1)).Named("x2");
            var x3 = Variable.Copy(x2);
            x3.AddAttribute(new MarginalPrototype(new TruncatedGaussian()));
            Variable.ConstrainPositive(x3);
            model2.CloseBlock();

            Gaussian x2Actual = engine.Infer<Gaussian>(x2);
            Console.WriteLine(x2Actual);
            double logEvidence2 = engine.Infer<Bernoulli>(ev2).LogOdds;
            Console.WriteLine(logEvidence2);

            Assert.True(xActual.MaxDiff(x2Actual) < 1e-8);
            Assert.True(MMath.AbsDiff(logEvidence, logEvidence2) < 1e-8);
        }

        [Fact]
        public void VmpTruncatedGaussianTest3()
        {
            var x = Variable.GaussianFromMeanAndPrecision(
                Variable.GaussianFromMeanAndVariance(1, 1) * Variable.GaussianFromMeanAndVariance(1, 1),
                10);
            Variable.ConstrainPositive(x);

            var x2 = Variable.GaussianFromMeanAndPrecision(
                Variable.GaussianFromMeanAndVariance(1, 1) * Variable.GaussianFromMeanAndVariance(1, 1),
                10);
            var x3 = Variable.Copy(x2);
            x3.AddAttribute(new MarginalPrototype(new TruncatedGaussian()));
            Variable.ConstrainPositive(x3);

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Gaussian xActual = engine.Infer<Gaussian>(x);
            Console.WriteLine(xActual);

            Gaussian x2Actual = engine.Infer<Gaussian>(x2);
            Console.WriteLine(x2Actual);

            Assert.True(xActual.MaxDiff(x2Actual) < 1e-8);
        }

        // Test that the compiler rejects this model, or gives the right answer.
        // Currently we must reject otherwise inference crashes with improper messages.
        [Fact]
        public void VmpIsPositiveMustBeStochasticError()
        {
            Assert.Throws<CompilationFailedException>(() =>
            {
                var x = Variable.GaussianFromMeanAndPrecision(0, 1);
                var y = Variable.GaussianFromMeanAndPrecision(0, 1);
                var z = (x > y);
                z.ObservedValue = true;

                InferenceEngine engine = new InferenceEngine();
                engine.Algorithm = new VariationalMessagePassing();
                Console.WriteLine(engine.Infer(x));
            });
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void VmpIsBetween() // to be fixed! Need a stochastic copy operator I think
        {
            var ev = Variable.Bernoulli(0.5).Named("ev");
            var model = Variable.If(ev);
            Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 1).Named("x");
            Variable.ConstrainBetween(x, 0, 1);
            model.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            //var engine = new InferenceEngine();
            Gaussian xActual = engine.Infer<Gaussian>(x);
            Console.WriteLine(xActual);
            double logEvidence = engine.Infer<Bernoulli>(ev).LogOdds;
            Console.WriteLine(logEvidence);

            var ev2 = Variable.Bernoulli(0.5).Named("ev");
            //var model2 = Variable.If(ev2);
            Variable<double> x2 = Variable.GaussianFromMeanAndVariance(0, 1).Named("x2");
            var x3 = Variable.Copy(x2);
            x3.AddAttribute(new MarginalPrototype(new TruncatedGaussian()));
            Variable.ConstrainBetween(x3, 0, 1);
            //model2.CloseBlock();

            InferenceEngine engine2 = new InferenceEngine(new VariationalMessagePassing());
            Gaussian x2Actual = engine2.Infer<Gaussian>(x2);
            Console.WriteLine(x2Actual);
            double logEvidence2 = engine.Infer<Bernoulli>(ev2).LogOdds;
            Console.WriteLine(logEvidence2);

            Assert.True(xActual.MaxDiff(x2Actual) < 1e-8);
            Assert.True(MMath.AbsDiff(logEvidence, logEvidence2) < 1e-8);
            Assert.True(MMath.AbsDiff(logEvidence, System.Math.Log(.5)) < 1e-8);
        }

        // TODO: make this pass
        [Fact]
        [Trait("Category", "OpenBug")]
        public void TruncatedGaussianProductTest()
        {
            var x = Variable.TruncatedGaussian(0, 1, 0, double.PositiveInfinity);
            var y = Variable.GaussianFromMeanAndPrecision(0, 1);
            var z = x * y;
            z.AddAttribute(new MarginalPrototype(new Gaussian()));
            Variable.ConstrainEqualRandom(z, new Gaussian(2, .1));
            var ie = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine("x: " + ie.Infer(x));
            Console.WriteLine("y: " + ie.Infer(y));
        }

        // TODO: check if this is still needed
        internal void ConstrainPositiveAndBetween()
        {
            Variable<double> x2 = Variable.GaussianFromMeanAndVariance(0, 1).Named("x2");
            var x3 = Variable.Copy(x2);
            x3.AddAttribute(new MarginalPrototype(new TruncatedGaussian()));
            Variable.ConstrainPositive(x3);
            Variable.ConstrainBetween(x3, -1, 1);
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(engine.Infer(x2));
        }

        internal void BernoulliFromLogOddsVsLogisticTest()
        {
            var x = Variable.GaussianFromMeanAndPrecision(0, 1);
            var s = Variable.BernoulliFromLogOdds(x);
            s.ObservedValue = true;
            var ie = new InferenceEngine(new ExpectationPropagation());
            Console.WriteLine(ie.Infer(x));

            var x2 = Variable.GaussianFromMeanAndPrecision(0, 1);
            var s2 = Variable.Bernoulli(Variable.Logistic(x2));
            s2.ObservedValue = true;
            Console.WriteLine(ie.Infer(x2));
        }

        //[Fact]
        internal void HybridProductTest2()
        {
            // Gaussian aPrior = new Gaussian(0.1, 1.4);
            // Gaussian bPrior = new Gaussian(0.2, 1.5);
            // double t = 0.0;
            // exact result: a = Gaussian(0.220965800748222, 1.373271271032174), b = Gaussian(0.264384550583005, 1.482977905817094)
            // EP result: a = Gaussian(0.2139, 1.375), b = Gaussian(0.2616, 1.483)
            // hybrid 1: a = Gaussian(0.5963, 0.4137), b = Gaussian(0.6501, 0.4678)
            // hybrid 2: a = Gaussian(0.8189, 0.4047), b = Gaussian(0.8857, 0.457)

            // Gaussian aPrior = new Gaussian(1.2, 0.4);
            // Gaussian bPrior = new Gaussian(2.3, 0.5);
            // double t = 2.6;
            // exact:    a = Gaussian(1.630263073476629, 0.197111741665093), b = Gaussian(2.596421095579687, 0.351082523413547)
            // EP:       a = Gaussian(1.622, 0.2023), b = Gaussian(2.592, 0.3511)
            // hybrid 1: a = Gaussian(1.534, 0.2173), b = Gaussian(2.521, 0.3564)
            // hybrid 2: a = Gaussian(1.595, 0.1701), b = Gaussian(2.572, 0.3007)

            Gaussian aPrior = new Gaussian(0.0001, 1.4);
            Gaussian bPrior = new Gaussian(0.0002, 1.5);
            double t = 0;

            if (true)
            {
                // exact
                double aMin = -10, aMax = 10;
                double bMin = -10, bMax = 10;
                int nSamples = (int)1e4;
                double aMean, aVar;
                aPrior.GetMeanAndVariance(out aMean, out aVar);
                double bMean, bVar;
                bPrior.GetMeanAndVariance(out bMean, out bVar);
                GaussianEstimator aEst = new GaussianEstimator();
                GaussianEstimator bEst = new GaussianEstimator();
                for (int i = 0; i < nSamples; i++)
                {
                    double ax = i * (aMax - aMin) / (nSamples - 1) + aMin;
                    double aweight = System.Math.Exp(aPrior.GetLogProb(ax) + MMath.NormalCdfLn((ax * bMean - t) / System.Math.Sqrt(ax * ax * bVar)));
                    aEst.Add(ax, aweight);
                    double bx = i * (bMax - bMin) / (nSamples - 1) + bMin;
                    double bweight = System.Math.Exp(bPrior.GetLogProb(bx) + MMath.NormalCdfLn((bx * aMean - t) / System.Math.Sqrt(bx * bx * aVar)));
                    bEst.Add(bx, bweight);
                }
                Console.WriteLine("a = {0}, b = {1}", aEst.GetDistribution(new Gaussian()), bEst.GetDistribution(new Gaussian()));
            }

            if (true)
            {
                // EP
                Variable<double> a = Variable.Random(aPrior);
                Variable<double> b = Variable.Random(bPrior);
                Variable<double> c = a * b - t;
                Variable.ConstrainPositive(c);
                InferenceEngine engine = new InferenceEngine();
                Gaussian aMarg = engine.Infer<Gaussian>(a);
                Gaussian bMarg = engine.Infer<Gaussian>(b);
                Console.WriteLine("a = {0}, b = {1}", aMarg, bMarg);
                Gaussian cMarg = GaussianProductVmpOp.ProductAverageLogarithm(aMarg, bMarg);
                Console.WriteLine("  product = {0}, p(a*b>0) = {1}", cMarg, MMath.NormalCdf(cMarg.GetMean() / System.Math.Sqrt(cMarg.GetVariance())));
            }

            // hybrid method 1
            if (true)
            {
                Gaussian aLike = Gaussian.Uniform();
                Gaussian bLike = Gaussian.Uniform();
                Gaussian aMarg = aPrior * aLike;
                Gaussian bMarg = bPrior * bLike;
                Gaussian cMarg;
                for (int iter = 0; iter < 20; iter++)
                {
                    cMarg = GaussianProductVmpOp.ProductAverageLogarithm(aMarg, bMarg);
                    Gaussian dMarg = DoublePlusOp.SumAverageConditional(cMarg, -t);
                    Gaussian dLike = IsPositiveOp.XAverageConditional(true, dMarg);
                    Gaussian cLike = DoublePlusOp.AAverageConditional(dLike, -t);
                    aLike = GaussianProductVmpOp.AAverageLogarithm(cLike, bMarg);
                    aMarg = aPrior * aLike;
                    bLike = GaussianProductVmpOp.BAverageLogarithm(cLike, aMarg);
                    bMarg = bPrior * bLike;
                }
                Console.WriteLine("a = {0}, b = {1}", aMarg, bMarg);
                cMarg = GaussianProductVmpOp.ProductAverageLogarithm(aMarg, bMarg);
                Console.WriteLine("  product = {0}, p(a*b>0) = {1}", cMarg, MMath.NormalCdf(cMarg.GetMean() / System.Math.Sqrt(cMarg.GetVariance())));
            }

            // hybrid method 2
            if (true)
            {
                Gaussian aLike = Gaussian.Uniform();
                Gaussian bLike = Gaussian.Uniform();
                Gaussian cLike = Gaussian.Uniform();
                Gaussian aMarg = aPrior;
                Gaussian bMarg = bPrior;
                for (int iter = 0; iter < 20; iter++)
                {
                    //cLike = Gaussian.Uniform();
                    Gaussian cPrior = GaussianProductOp.ProductAverageConditional(cLike, aPrior, bPrior);
                    Gaussian dPrior = DoublePlusOp.SumAverageConditional(cPrior, -t);
                    Gaussian dLike = IsPositiveOp.XAverageConditional(true, dPrior);
                    cLike = DoublePlusOp.AAverageConditional(dLike, -t);
                    bMarg = bPrior * bLike;
                    aLike = GaussianProductVmpOp.AAverageLogarithm(cLike, bMarg);
                    aMarg = aPrior * aLike;
                    bLike = GaussianProductVmpOp.BAverageLogarithm(cLike, aMarg);
                }
                Console.WriteLine("a = {0}, b = {1}", aMarg, bMarg);
                Gaussian cMarg = GaussianProductVmpOp.ProductAverageLogarithm(aMarg, bMarg);
                Console.WriteLine("  product = {0}, p(a*b>0) = {1}", cMarg, MMath.NormalCdf(cMarg.GetMean() / System.Math.Sqrt(cMarg.GetVariance())));
            }
        }

        [Fact]
        public void BayesianLinearMixedModelTest()
        {
            int d = 2;
            int n = 2;
            Vector[] xObs = new Vector[n];
            for (int i = 0; i < n; i++)
            {
                xObs[i] = Vector.Zero(d);
                for (int j = 0; j < d; j++)
                {
                    xObs[i][j] = (i + 1) * (j + 1);
                }
            }
            double[] yObs = new double[n];
            for (int i = 0; i < n; i++)
            {
                yObs[i] = (i + 1) * (i + 1);
            }
            BlmmElement(xObs, yObs);
            BlmmSubvector(xObs, yObs);
            BlmmConcat(xObs, yObs);
        }

        private void BlmmElement(Vector[] xObs, double[] yObs)
        {
            int d = xObs[0].Count;
            int n = yObs.Length;
            Vector wMean = Vector.Zero(d);
            PositiveDefiniteMatrix wPrec = new PositiveDefiniteMatrix(d, d);
            for (int i = 0; i < d; i++)
            {
                wPrec[i, i] = 1e-10;
            }
            Variable<Vector> w = Variable.VectorGaussianFromMeanAndPrecision(wMean, wPrec).Named("w");
            Variable<double> w1 = Variable.GetItem(w, 0).Named("w1");
            Variable<double> w2 = Variable.GetItem(w, 1).Named("w2");
            Variable<double> prec1 = Variable.GammaFromShapeAndRate(2, 2).Named("prec1");
            Variable<double> prec2 = Variable.GammaFromShapeAndRate(2, 2).Named("prec2");
            Variable<double> mean1 = Variable.GaussianFromMeanAndPrecision(w1, prec1).Named("mean1");
            mean1.ObservedValue = 0.0;
            Variable<double> mean2 = Variable.GaussianFromMeanAndPrecision(w2, prec2).Named("mean2");
            mean2.ObservedValue = 0.0;

            Range item = new Range(n).Named("item");
            VariableArray<double> y = Variable.Array<double>(item).Named("y");
            VariableArray<Vector> x = Variable.Array<Vector>(item).Named("x");
            y[item] = Variable.GaussianFromMeanAndPrecision(Variable.InnerProduct(w, x[item]), 1.0);

            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = new VariationalMessagePassing();
            x.ObservedValue = xObs;
            y.ObservedValue = yObs;
            VectorGaussian wActual = engine.Infer<VectorGaussian>(w);
            Console.WriteLine(wActual);
            Console.WriteLine("precision = ");
            Console.WriteLine(wActual.Precision);
        }

        private void BlmmSubvector(Vector[] xObs, double[] yObs)
        {
            int d = xObs[0].Count;
            int n = yObs.Length;
            Vector wMean = Vector.Zero(d);
            PositiveDefiniteMatrix wPrec = new PositiveDefiniteMatrix(d, d);
            for (int i = 0; i < d; i++)
            {
                wPrec[i, i] = 1e-10;
            }
            Variable<Vector> w = Variable.VectorGaussianFromMeanAndPrecision(wMean, wPrec).Named("w");
            Variable<Vector> w1 = Variable.Subvector(w, 0, 1).Named("w1");
            Variable<Vector> w2 = Variable.Subvector(w, 1, 1).Named("w2");
            PositiveDefiniteMatrix scale = new PositiveDefiniteMatrix(1, 1);
            scale[0, 0] = 0.5;
            Variable<PositiveDefiniteMatrix> prec1 = Variable.WishartFromShapeAndScale(2, scale).Named("prec1");
            Variable<PositiveDefiniteMatrix> prec2 = Variable.WishartFromShapeAndScale(2, scale).Named("prec2");
            Variable<Vector> mean1 = Variable.VectorGaussianFromMeanAndPrecision(w1, prec1).Named("mean1");
            Variable<Vector> mean2 = Variable.VectorGaussianFromMeanAndPrecision(w2, prec2).Named("mean2");
            mean1.ObservedValue = Vector.Zero(1);
            mean2.ObservedValue = Vector.Zero(1);

            Range item = new Range(n).Named("item");
            VariableArray<double> y = Variable.Array<double>(item).Named("y");
            VariableArray<Vector> x = Variable.Array<Vector>(item).Named("x");
            y[item] = Variable.GaussianFromMeanAndPrecision(Variable.InnerProduct(w, x[item]), 1.0);

            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = new VariationalMessagePassing();
            x.ObservedValue = xObs;
            y.ObservedValue = yObs;
            VectorGaussian wActual = engine.Infer<VectorGaussian>(w);
            Console.WriteLine(wActual);
            Console.WriteLine("precision = ");
            Console.WriteLine(wActual.Precision);
        }

        private void BlmmConcat(Vector[] xObs, double[] yObs)
        {
            int d = xObs[0].Count;
            int n = yObs.Length;
            Range dim = new Range(d).Named("dim");
            VariableArray<double> prec = Variable.Array<double>(dim).Named("prec");
            prec[dim] = Variable.GammaFromShapeAndRate(2, 2).ForEach(dim);
            VariableArray<double> wElts = Variable.Array<double>(dim).Named("wElts");
            wElts[dim] = Variable.GaussianFromMeanAndPrecision(0.0, prec[dim]);
            Variable<Vector> w = Variable.Vector(wElts).Named("w");

            Range item = new Range(n).Named("item");
            VariableArray<double> y = Variable.Array<double>(item).Named("y");
            VariableArray<Vector> x = Variable.Array<Vector>(item).Named("x");
            y[item] = Variable.GaussianFromMeanAndPrecision(Variable.InnerProduct(w, x[item]), 1.0);

            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = new VariationalMessagePassing();
            x.ObservedValue = xObs;
            y.ObservedValue = yObs;
            VectorGaussian wActual = engine.Infer<VectorGaussian>(w);
            Console.WriteLine(wActual);
            Console.WriteLine("precision = ");
            Console.WriteLine(wActual.Precision);
        }

        [Fact]
        public void GaussianTimesBetaTest()
        {
            Variable<double> x = Variable.GaussianFromMeanAndVariance(1.2, 3.4).Named("x");
            Variable<double> s = Variable.Beta(5.6, 4.8).Named("s");
            Variable<double> y = x * s;
            y.Name = "y";
            Variable.ConstrainEqualRandom(y, new Gaussian(2.7, 1.9));

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Gaussian xActual = engine.Infer<Gaussian>(x);
            Beta sActual = engine.Infer<Beta>(s);
            Gaussian xExpected = new Gaussian(2.445726690056733, 2.122166142748496);
            Beta sExpected = new Beta(6.383336506695946, 4.961772201516414);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-4);
            Console.WriteLine("s = {0} should be {1}", sActual, sExpected);
            Assert.True(sExpected.MaxDiff(sActual) < 1e-4);
        }

        [Fact]
        public void GaussianTimesGammaTest()
        {
            Variable<double> x = Variable.GaussianFromMeanAndVariance(1.2, 3.4).Named("x");
            Variable<double> s = Variable.GammaFromShapeAndRate(5.6, 4.8).Named("s");
            Variable<double> y = x * s;
            y.Name = "y";
            Variable.ConstrainEqualRandom(y, new Gaussian(2.7, 1.9));

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.NumberOfIterations = 1000;
            Gaussian xActual = engine.Infer<Gaussian>(x);
            Gamma sActual = engine.Infer<Gamma>(s);
            Gaussian xExpected = new Gaussian(1.902653436767819, 0.992239706783399);
            Gamma sExpected = Gamma.FromShapeAndRate(8.431332876585183, 7.657476147105538);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Console.WriteLine("s = {0} should be {1}", sActual, sExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-4);
            Assert.True(sExpected.MaxDiff(sActual) < 1e-4);
        }

        [Fact]
        public void PoissonExpTest()
        {
            Variable<double> x = Variable.GaussianFromMeanAndVariance(1.2, 3.4).Named("x");
            Variable<double> ex = Variable.Exp(x).Named("ex");
            int[] data = new int[] { 5, 6, 7 };
            Range item = new Range(data.Length).Named("item");
            VariableArray<int> y = Variable.Array<int>(item).Named("y");
            y[item] = Variable.Poisson(ex).ForEach(item);
            y.ObservedValue = data;

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.NumberOfIterations = 200;
            Gaussian xExpected = new Gaussian(1.755071011884509, 0.055154577283323);
            Gaussian xActual = engine.Infer<Gaussian>(x);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-8);
        }

        [Fact]
        public void UnobservedPoissonTest()
        {
            double a = 1, b = 1;
            var mean = Variable.GammaFromShapeAndRate(a, b);
            var x1 = Variable.Poisson(mean);
            var x2 = Variable.Poisson(mean);
            x1.ObservedValue = 5;
            mean.Name = "mean";
            x1.Name = "x1";
            x2.Name = "x2";
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Gamma meanExpected = Gamma.FromShapeAndRate(a + x1.ObservedValue, b + 1);
            Gamma meanActual = engine.Infer<Gamma>(mean);
            Console.WriteLine("mean = {0} should be {1}", meanActual, meanExpected);
            //Assert.True(meanExpected.MaxDiff(meanActual) < 1e-10);
            Poisson x2Actual = engine.Infer<Poisson>(x2);
            Poisson x2Expected = new Poisson(meanExpected.GetMean());
            Console.WriteLine("x2 = {0} should be {1}", x2Actual, x2Expected);
            //Assert.True(x2Expected.MaxDiff(x2Actual) < 1e-10);
        }

        /// <summary>
        /// Test PlusGammaVmpOp
        /// </summary>
        [Fact]
        public void GammaSumTest()
        {
            var prior1 = Gamma.FromShapeAndRate(2, 3);
            var prior2 = Gamma.FromShapeAndRate(4, 5);
            var a = Variable.Random(prior1).Named("a");
            var b = Variable.Random(prior2).Named("b");
            var mean = a + b;
            mean.Name = "mean";
            var x = Variable.Poisson(mean).Named("x");
            x.ObservedValue = 1;

            Gamma aExpected = new Gamma(2.2, 0.2755); // estimate from Monte Carlo
            if (false)
            {
                // importance sampling
                int nSamples = 1000000;
                Gamma like = PoissonOp.MeanAverageConditional(x.ObservedValue);
                GammaEstimator est = new GammaEstimator();
                for (int iter = 0; iter < nSamples; iter++)
                {
                    double aSample = prior1.Sample();
                    double bSample = prior2.Sample();
                    double mSample = aSample + bSample;
                    double logWeight = like.GetLogProb(mSample);
                    est.Add(aSample, System.Math.Exp(logWeight));
                }
                aExpected = est.GetDistribution(new Gamma());
            }

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Gamma aActual = engine.Infer<Gamma>(a);
            Console.WriteLine("a = {0} should be {1}", aActual, aExpected);
            Assert.True(aExpected.MaxDiff(aActual) < 2e-1);
        }

        /// <summary>
        /// Test VMP with Sum_Expanded
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")]
        public void GammaSumArrayTest()
        {
            var priors = new Gamma[]
            {
                Gamma.FromShapeAndRate(2, 3),
                Gamma.FromShapeAndRate(4, 5),
                Gamma.FromShapeAndRate(6, 7),
            };
            VariableArray<Gamma> prior = Variable.Observed(priors);
            Range item = prior.Range;
            item.Name = nameof(item);
            VariableArray<double> array = Variable.Array<double>(item);
            array[item] = Variable<double>.Random(prior[item]);
            var sum = Variable.Sum_Expanded(array);
            sum.Name = "sum";
            var x = Variable.Poisson(sum).Named("x");
            x.ObservedValue = 1;

            GammaArray arrayExpected = new GammaArray(new Gamma[] {
                new Gamma(2.083, 0.2712),
                new Gamma(4.129, 0.1754),
                new Gamma(6.162, 0.1296),
            }); // estimate from Monte Carlo
            if (false)
            {
                // importance sampling
                int nSamples = 1000000;
                Gamma like = PoissonOp.MeanAverageConditional(x.ObservedValue);
                GammaEstimator[] est = Util.ArrayInit(priors.Length, i => new GammaEstimator());
                for (int iter = 0; iter < nSamples; iter++)
                {
                    double[] arraySample = Util.ArrayInit(priors.Length, i => priors[i].Sample());
                    double sumSample = arraySample.Sum();
                    double logWeight = like.GetLogProb(sumSample);
                    for (int i = 0; i < priors.Length; i++)
                    {
                        est[i].Add(arraySample[i], System.Math.Exp(logWeight));
                    }
                }
                for (int i = 0; i < priors.Length; i++)
                {
                    arrayExpected[i] = est[i].GetDistribution(new Gamma());
                }
            }

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Gamma arrayActual = engine.Infer<Gamma>(array);
            Console.WriteLine("a = {0} should be {1}", arrayActual, arrayExpected);
            Assert.True(arrayExpected.MaxDiff(arrayActual) < 2e-1);
        }

        [Fact]
        public void GammaRatioTest()
        {
            var evidence = Variable.Bernoulli(0.5).Named("evidence");
            var evBlock = Variable.If(evidence);
            var prior1 = Gamma.FromShapeAndRate(2, 3);
            var prior2 = Gamma.FromShapeAndRate(4, 5);
            var a = Variable.Random(prior1).Named("a");
            var b = Variable.Random(prior2).Named("b");
            var ratio = a / b;
            ratio.Name = "ratio";
            var x = Variable.GaussianFromMeanAndPrecision(0, ratio).Named("x");
            x.ObservedValue = 1;
            evBlock.CloseBlock();

            // estimates from Monte Carlo (these are not expected to exactly match VMP)
            Gamma aExpected = new Gamma(2.409, 0.2765);
            Gamma bExpected = new Gamma(4.376, 0.1829);
            Gamma rExpected = new Gamma(1.578, 0.6356);
            double evExpected = -1.5926000332361614;
            if (false)
            {
                // importance sampling
                int nSamples = 10000000;
                Gamma like = GaussianOp.PrecisionAverageConditional(x.ObservedValue, 0);
                GammaEstimator estA = new GammaEstimator();
                GammaEstimator estB = new GammaEstimator();
                GammaEstimator estR = new GammaEstimator();
                double averageWeight = 0;
                for (int iter = 0; iter < nSamples; iter++)
                {
                    double aSample = prior1.Sample();
                    double bSample = prior2.Sample();
                    double rSample = aSample / bSample;
                    double logWeight = like.GetLogProb(rSample);
                    double weight = System.Math.Exp(logWeight);
                    estA.Add(aSample, weight);
                    estB.Add(bSample, weight);
                    estR.Add(rSample, weight);
                    averageWeight += weight;
                }
                averageWeight /= nSamples;
                evExpected = System.Math.Log(averageWeight);
                aExpected = estA.GetDistribution(new Gamma());
                bExpected = estB.GetDistribution(new Gamma());
                rExpected = estR.GetDistribution(new Gamma());
            }

            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = new VariationalMessagePassing();
            Gamma aActual = engine.Infer<Gamma>(a);
            Gamma bActual = engine.Infer<Gamma>(b);
            Gamma rActual = engine.Infer<Gamma>(ratio);
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("a = {0} should be {1}", aActual, aExpected);
            Console.WriteLine("b = {0} should be {1}", bActual, bExpected);
            Console.WriteLine("ratio = {0} should be {1}", rActual, rExpected);
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(aExpected.MaxDiff(aActual) < 1e-1);
            Assert.True(bExpected.MaxDiff(bActual) < 1e-1);
            Assert.True(rExpected.MaxDiff(rActual) < 1e-1);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 2e-2);
        }

        [Fact]
        public void DirichletMultinomialTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            double[] alpha = new double[] { 1.2, 3.4 };
            Range dim = new Range(alpha.Length).Named("dim");
            Variable<Vector> p = Variable.Dirichlet(dim, alpha).Named("p");
            int n = 10;
            VariableArray<int> x = Variable.Multinomial(n, p).Named("x");
            block.CloseBlock();
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            int[] k = new int[] { 3, 7 };
            x.ObservedValue = k;
            Dirichlet pActual = engine.Infer<Dirichlet>(p);
            Dirichlet pExpected = (new Dirichlet(alpha)) * (new Dirichlet(k[0] + 1, k[1] + 1));
            Console.WriteLine("p = {0} should be {1}", pActual, pExpected);
            Assert.True(pExpected.MaxDiff(pActual) < 1e-10);
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            double evExpected = MMath.ChooseLn(n, k[0]);
            double sumAlpha = 0;
            for (int d = 0; d < alpha.Length; d++)
            {
                evExpected += MMath.GammaLn(alpha[d] + k[d]) - MMath.GammaLn(alpha[d]);
                sumAlpha += alpha[d];
            }
            evExpected += MMath.GammaLn(sumAlpha) - MMath.GammaLn(sumAlpha + n);
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-10);
        }

        [Fact]
        public void SoftmaxMultinomialTest()
        {
            SoftmaxMultinomial(typeof(LogisticOp_SJ99), typeof(SoftmaxOp_KM11));
            SoftmaxMultinomial(null, null);
        }

        private void SoftmaxMultinomial(object logistic_op, object softmax_op)
        {
            int n = 10;
            int k = 3;
            Gaussian wPrior = new Gaussian(1.2, 3.4);

            Gaussian wExpected1;
            double evExpected;
            LogisticBinomialUnrolled(logistic_op, k, n, wPrior, out wExpected1, out evExpected);
            IDistribution<double[]> wExpected = Distribution<double>.Array(new Gaussian[] { wExpected1, Gaussian.PointMass(0) });
            evExpected += MMath.ChooseLn(n, k);


            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Range dim = new Range(2).Named("dim");
            Gaussian[] priors = new Gaussian[]
                {
                    wPrior,
                    Gaussian.PointMass(0)
                };
            VariableArray<Gaussian> priorsVar = Variable.Constant(priors, dim).Named("priors");
            VariableArray<double> w = Variable.Array<double>(dim).Named("w");
            w[dim] = Variable.Random<double, Gaussian>(priorsVar[dim]);
            Variable<Vector> p = Variable.Softmax(w).Named("p");
            VariableArray<int> y = Variable.Multinomial(n, p).Named("y");
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.NumberOfIterations = 100;
            if (softmax_op != null) engine.Compiler.GivePriorityTo(softmax_op);
            y.ObservedValue = new int[] { k, n - k };
            object wActual = engine.Infer(w);
            Console.WriteLine("w = {0} should be {1}", wActual, wExpected);
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(wExpected.MaxDiff(wActual) < 1e-6);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-6);
        }

        [Fact]
        public void TestLinearTimeSoftmax()
        {
            int K = 4, N = 1000 + 100 + 10 + 1;
            var data = new int[N];
            for (int i = 0; i < 1000; i++)
                data[i] = 0;
            for (int i = 1000; i < 1000 + 100; i++)
                data[i] = 1;
            for (int i = 1000 + 100; i < 1000 + 100 + 10; i++)
                data[i] = 2;
            data[1000 + 100 + 10] = 3;
            // Calculated using 1e6 samples from an MH sampler
            var trueMeans = Vector.FromArray(new double[] { 3.423958, 1.114536, -1.197399, -3.152047 });
            // model
            var k = new Range(K);
            var n = new Range(N);
            var x = Variable.Array<double>(k);
            x[k] = Variable.GaussianFromMeanAndVariance(0, 4).ForEach(k);
            var p = Variable.Softmax(x);
            var d = Variable.Array<int>(n);
            d[n] = Variable.Discrete(p).ForEach(n);
            d.ObservedValue = data;

            var Operators = new Type[]
                {
                    typeof (SoftmaxOp_BL06_LBFGS), typeof (SoftmaxOp_KM11_LBFGS), typeof (SoftmaxOp_KM11), typeof (SoftmaxOp_Bohning),
                    typeof (SoftmaxOp_Taylor)
                };
            var numIters = new int[] { 10000, 50, 50, 10000, 10000, 10000 };

            for (int i = 0; i < Operators.Length; i++)
            {
                Console.WriteLine(Operators[i].Name);
                var ie = new InferenceEngine(new VariationalMessagePassing());
                ie.NumberOfIterations = numIters[i];
                ie.ShowProgress = false;
                ie.ShowTimings = true;
                // Fails with unrolling because Blei06SoftmaxOp.XAverageLogarithm assumes the array is updated in parallel
                ie.Compiler.UnrollLoops = false;
                //if (i == 3) ie.BrowserMode = BrowserMode.Always;
                ie.Compiler.GivePriorityTo(Operators[i]);
                var post = ie.Infer<DistributionArray<Gaussian>>(x);
                Console.WriteLine(post);
                Console.WriteLine("Error: " + NonconjugateVMP2Tests.RMSE(post, trueMeans));
                //for (int i = 0; i < K; i++)
                //    Assert.True(MMath.AbsDiff(post[i].GetMean(), trueMeans[i], 1e-3) < 1e-1);
            }
        }

        private static double LogProb(DistributionArray<Gaussian> inferred, Vector truth)
        {
            int K = truth.Count;
            double sum = 0;
            for (int k = 0; k < K; k++)
            {
                if (!inferred[k].IsPointMass)
                    sum += inferred[k].GetLogProb(truth[k]);
            }
            return sum / (double)K;
        }

        internal void TestLinearTimeSoftmax2()
        {
            Rand.Restart(4);
            int K = 2, N = 1000;
            // generate data
            var trueX = Vector.Zero(K);
            trueX.SetToFunction(trueX, y => Rand.Normal(0, System.Math.Sqrt(10)));
            bool fixFirstElementToZero = false;
            if (fixFirstElementToZero)
                trueX[0] = 0.0;
            //var trueP = Dirichlet.Sample(Vector.Constant(K, .5), Vector.Zero(K));
            Console.WriteLine(trueX);
            var trueP = MMath.Softmax(trueX.ToArray());
            var data = new int[N];
            for (int i = 0; i < N; i++)
                data[i] = Rand.Sample(trueP);
            // model
            var ev = Variable.Bernoulli(.5);
            var block = Variable.If(ev);
            var k = new Range(K);
            var n = new Range(N);
            var x = Variable.Array<double>(k);
            var arrayPrior = Variable.Array<Gaussian>(k).Named("arrayPrior");
            var priorObs = new Gaussian[K];
            for (int i = 0; i < K; i++)
            {
                priorObs[i] = new Gaussian(0, 10);
            }
            if (fixFirstElementToZero)
                priorObs[0] = Gaussian.PointMass(0);
            arrayPrior.ObservedValue = (Gaussian[])priorObs.Clone();
            x[k] = Variable<double>.Random(arrayPrior[k]);
            var p = Variable.Softmax(x);
            var d = Variable.Array<int>(n);
            d[n] = Variable.Discrete(p).ForEach(n);
            d.ObservedValue = data;
            block.CloseBlock();
            var Operators = new Type[] { typeof(SoftmaxOp_BL06_LBFGS), typeof(SoftmaxOp_KM11_LBFGS), typeof(SoftmaxOp_KM11) };

            for (int i = 0; i < Operators.Length; i++)
            {
                // var bound = SoftmaxOp.SoftmaxBounds.SaulJordan99_LBFGS;
                Console.WriteLine(Operators[i].Name);
                var ie = new InferenceEngine(new VariationalMessagePassing());
                if (i == 3) ie.NumberOfIterations = 10000;
                ie.ShowProgress = false;
                ie.ShowTimings = true;
                ie.Compiler.GivePriorityTo(Operators[i]);
                var xPost = ie.Infer<DistributionArray<Gaussian>>(x);
                Console.WriteLine("RMSE: " + NonconjugateVMP2Tests.RMSE(xPost, trueX));
                Console.WriteLine("P(truth|inferred): " + LogProb(xPost, trueX));
                Console.WriteLine("Evidence: " + ie.Infer<Bernoulli>(ev).LogOdds);
            }
        }


        internal void TestLinearTimeSoftmax3()
        {
            int Ntrain = 10000, Ntest = 10000, repeats = 1;
            //var Ks = new int[] { 2, 3, 5, 10, 100 };
            var Ks = new int[] { 100 };
            bool fixFirstElementToZero = false;
            bool withPredictions = false;
            // model
            var ev = Variable.Bernoulli(.5).Named("ev");
            var block = Variable.If(ev);
            var Kint = Variable.New<int>().Named("Kint");
            Kint.ObservedValue = 1;
            var k = new Range(Kint).Named("k");
            var n = new Range(Ntrain).Named("n");
            var x = Variable.Array<double>(k).Named("x");
            var arrayPrior = Variable.Array<Gaussian>(k).Named("arrayPrior");
            arrayPrior.ObservedValue = new Gaussian[] { new Gaussian() };
            x[k] = Variable<double>.Random(arrayPrior[k]);
            var p = Variable.Softmax(x).Named("p");
            var trainVar = Variable.Array<int>(n).Named("trainData");
            trainVar[n] = Variable.Discrete(p).ForEach(n);
            trainVar.ObservedValue = Enumerable.Range(0, Ntrain).Select(z => 0).ToArray();
            Variable<Vector> p2;
            Variable<int> predictive = null;
            if (withPredictions)
            {
                p2 = Variable.Softmax(x).Named("p2");
                predictive = Variable.Discrete(p2).Named("predictive");
            }
            block.CloseBlock();

            //ie.ShowTimings = true;

            var Operators = new Type[] {/* typeof(ProductOfLogisticsSoftmaxOp), typeof(Blei06SoftmaxOp),*/ typeof(SoftmaxOp_KM11_LBFGS), typeof(SoftmaxOp_KM11) };
            Console.WriteLine("Method K RMSE probTruth Evidence probTest");

            for (int method = 0; method < Operators.Length; method++)
            {
                var ie = new InferenceEngine(new VariationalMessagePassing());
                //if (method == 3) ie.NumberOfIterations = 10000;
                ie.ShowProgress = false;
                //Console.WriteLine(bound.ToString());
                //SoftmaxOp.SoftmaxBound = bound;
                ie.Compiler.GivePriorityTo(Operators[method]);
                var ga = withPredictions ? ie.GetCompiledInferenceAlgorithm(x, predictive, ev) : ie.GetCompiledInferenceAlgorithm(x, ev);
                foreach (int K in Ks)
                {
                    //Kint.ObservedValue = K;
                    ga.SetObservedValue(Kint.Name, K);
                    for (int r = 0; r < repeats; r++)
                    {
                        // generate data
                        Rand.Restart(r);
                        var trueX = Vector.Zero(K);
                        trueX.SetToFunction(trueX, y => Rand.Normal(0, System.Math.Sqrt(10)));
                        if (fixFirstElementToZero)
                            trueX[0] = 0.0;
                        var trueP = MMath.Softmax(trueX.ToArray());
                        var train = new int[Ntrain];
                        for (int i = 0; i < Ntrain; i++)
                            train[i] = Rand.Sample(trueP);
                        int[] test = null;
                        if (withPredictions)
                        {
                            test = new int[Ntest];
                            for (int i = 0; i < Ntest; i++)
                                test[i] = Rand.Sample(trueP);
                        }
                        //trainVar.ObservedValue = train;
                        ga.SetObservedValue(trainVar.NameInGeneratedCode, train);
                        var priorObs = new Gaussian[K];
                        for (int i = 0; i < K; i++)
                        {
                            priorObs[i] = new Gaussian(0, 10);
                        }
                        if (fixFirstElementToZero)
                            priorObs[0] = Gaussian.PointMass(0);
                        //arrayPrior.ObservedValue = (Gaussian[])priorObs.Clone();
                        ga.SetObservedValue("arrayPrior", (Gaussian[])priorObs.Clone());
                        //try
                        //{
                        //var xPost = ie.Infer<DistributionArray<Gaussian>>(x);
                        DistributionArray<Gaussian> xPost = null;
                        Discrete predictiveProb = null;
                        double logEvidence = 0, oldLogEvidence = double.NegativeInfinity;
                        int iter;
                        for (iter = 0; iter < 10000; iter++)
                        {
                            ga.Update(1);
                            xPost = ga.Marginal<DistributionArray<Gaussian>>("x");
                            //var predictiveProb = ie.Infer<Discrete>(predictive);
                            if (withPredictions)
                                predictiveProb = ga.Marginal<Discrete>("predictive");
                            logEvidence = ga.Marginal<Bernoulli>(ev.Name).LogOdds;
                            Console.WriteLine("{0} {1}", Operators[method].Name, logEvidence);
                            if (System.Math.Abs(logEvidence - oldLogEvidence) < 1e-6)
                                break;
                            oldLogEvidence = logEvidence;
                        }
                        //Console.WriteLine("RMSE: " + NonconjugateVMP2Tests.RMSE(xPost, trueX));
                        //Console.WriteLine("P(truth|inferred): " + LogProb(xPost, trueX));
                        //Console.WriteLine("P(test|inferred): " + test.Select(z => predictiveProb.GetLogProb(z)).Sum() / (double)Ntest);
                        Console.Write("{0} {1} {2} {3} {4} {5}",
                                      Operators[method].Name,
                                      K,
                                      NonconjugateVMP2Tests.RMSE(xPost, trueX),
                                      LogProb(xPost, trueX),
                                      logEvidence,
                                      iter);
                        Console.WriteLine(withPredictions ? " " + test.Select(z => predictiveProb.GetLogProb(z)).Sum() / (double)Ntest : "");
                        //}
                        //catch
                        //{
                        //    Console.WriteLine("{0} {1} {2} {3} {4} {5}", Names[method], K, "NA", "NA", "NA", "NA"); 
                        //}
                    }
                }
            }
        }

        internal void SoftGWAS()
        {
            int N = 900;
            int numClasses = 5;
            int numGenotypes = 3;
            var n = new Range(N);
            var k = new Range(numClasses);
            var g = new Range(numGenotypes);
            var classLabels = Variable.Array<int>(n);
            var genotypes = Variable.Array<int>(n);
            // model
            var isAssociated = Variable.Bernoulli(.5);
            VariableArray<Vector> cpt; // conditional probability table
            using (Variable.If(isAssociated))
            {
                cpt = Variable.Array<Vector>(g);
                cpt[g] = Variable.DirichletUniform(numClasses).ForEach(g);
                classLabels[n] = Variable.Discrete(cpt[genotypes[n]]);
            }
            using (Variable.IfNot(isAssociated))
            {
                var prob = Variable.DirichletUniform(numClasses);
                classLabels[n] = Variable.Discrete(prob).ForEach(n);
            }
            // synthetic data
            var genotypeData = new int[N];
            var classData = new Discrete[N];
            var classDataVar = Variable.Array<Discrete>(n);
            for (int i = 0; i < N; i++)
            {
                genotypeData[i] = Rand.Sample(Vector.Constant(numGenotypes, 1.0 / numGenotypes));
                var temp = Vector.Zero(numClasses);
                Dirichlet.Sample(Vector.Constant(numClasses, 1.0), temp);
                classData[i] = new Discrete(temp);
            }
            classDataVar.ObservedValue = classData;
            genotypes.ObservedValue = genotypeData;
            Variable.ConstrainEqualRandom(classLabels[n], classDataVar[n]);
            // inference
            var ie = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine("Probability SNP is associated: " + ie.Infer<Bernoulli>(isAssociated).GetMean());
            Console.WriteLine(ie.Infer(cpt));
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void VectorSoftmaxTest()
        {
            int Ntrain = 10000, Ntest = 10000, repeats = 10;
            var Ks = new int[] { 2, 3, 5, 10, 100 };
            //Ks = new int[] { 2 };
            repeats = 1;
            bool fixFirstElementToZero = true;
            bool withPredictions = false;
            //Console.WriteLine("K RMSE probTruth Evidence probTest");
            foreach (int K in Ks)
            {
                var ev = Variable.Bernoulli(.5).Named("ev");
                var block = Variable.If(ev);
                var Kint = Variable.Observed<int>(K).Named("Kint");
                //Kint.ObservedValue = 1;
                var k = new Range(Kint).Named("k");
                var n = new Range(Ntrain).Named("n");
                var xPrior = Variable.Observed<VectorGaussian>(new VectorGaussian(K)).Named("xPrior");
                var x = Variable<Vector>.Random(xPrior).Named("x");
                var p = Variable.Softmax(x).Named("p");
                var trainVar = Variable.Array<int>(n).Named("trainData");
                trainVar[n] = Variable.Discrete(p).ForEach(n);
                Variable<Vector> p2;
                Variable<int> predictive = null;
                if (withPredictions)
                {
                    p2 = Variable.Softmax(x).Named("p2");
                    predictive = Variable.Discrete(p2).Named("predictive");
                }
                block.CloseBlock();

                var ie = new InferenceEngine(new VariationalMessagePassing());
                //if (method == 3) ie.NumberOfIterations = 10000;
                ie.ShowProgress = false;
                //Console.WriteLine(bound.ToString());
                //SoftmaxOp.SoftmaxBound = bound;
                //p.AddAttribute(new MarginalPrototype(Dirichlet.Uniform(K)));

                var xPriorCov = PositiveDefiniteMatrix.IdentityScaledBy(K, 10);
                if (fixFirstElementToZero) xPriorCov[0, 0] = 1e-15;
                xPrior.ObservedValue = VectorGaussian.FromMeanAndVariance(Vector.Zero(K), xPriorCov);

                for (int r = 0; r < repeats; r++)
                {
                    // generate data
                    Rand.Restart(r);
                    var trueX = Vector.Zero(K);
                    trueX.SetToFunction(trueX, y => Rand.Normal(0, System.Math.Sqrt(10)));
                    if (fixFirstElementToZero)
                        trueX[0] = 0.0;
                    var trueP = MMath.Softmax(trueX.ToArray());
                    var train = new int[Ntrain];
                    for (int i = 0; i < Ntrain; i++)
                        train[i] = Rand.Sample(trueP);
                    int[] test = null;
                    if (withPredictions)
                    {
                        test = new int[Ntest];
                        for (int i = 0; i < Ntest; i++)
                            test[i] = Rand.Sample(trueP);
                    }
                    trainVar.ObservedValue = train;
                    VectorGaussian xPost = null;
                    Discrete predictiveProb = null;
                    double logEvidence = 0, oldLogEvidence = double.NegativeInfinity;
                    int iter;
                    for (iter = 0; iter < 10; iter++)
                    {
                        ie.NumberOfIterations = iter + 1;
                        xPost = ie.Infer<VectorGaussian>(x);
                        //var predictiveProb = ie.Infer<Discrete>(predictive);
                        if (withPredictions)
                            predictiveProb = ie.Infer<Discrete>(predictive);
                        logEvidence = ie.Infer<Bernoulli>(ev).LogOdds;
                        Console.WriteLine(logEvidence);
                        if (System.Math.Abs(logEvidence - oldLogEvidence) < 1e-8)
                            break;
                        oldLogEvidence = logEvidence;
                    }
                    //Console.WriteLine("RMSE: " + NonconjugateVMP2Tests.RMSE(xPost, trueX));
                    //Console.WriteLine("P(truth|inferred): " + LogProb(xPost, trueX));
                    //Console.WriteLine("P(test|inferred): " + test.Select(z => predictiveProb.GetLogProb(z)).Sum() / (double)Ntest);
                    DistributionArray<Gaussian> xPostIndep = (DistributionArray<Gaussian>)Distribution<double>.Array(VectorSoftmaxOp_KM11.VectorGaussianToGaussianList(xPost));
                    double error = NonconjugateVMP2Tests.RMSE(xPostIndep, trueX);
                    double logProb = LogProb(xPostIndep, trueX);
                    Console.Write($"K={K} error={error} logProb={logProb} logEvidence={logEvidence}");
                    Console.WriteLine(withPredictions ? " " + test.Select(z => predictiveProb.GetLogProb(z)).Average() : "");
                    Assert.True(error < 0.1);
                }
            }
        }

        [Fact]
        public void VectorSoftmax_PointSoftmax_Throws()
        {
            var data = new Vector[]
            {
                Vector.FromArray(0.1, 0.3, 0.5, 0.1),
                Vector.FromArray(0.05, 0.5, 0.2, 0.25),
                Vector.FromArray(0.2, 0.4, 0.3, 0.1),
                Vector.FromArray(0.15, 0.2, 0.35, 0.3),
                Vector.FromArray(0.15, 0.3, 0.4, 0.15)
            };
            var mean = Variable.VectorGaussianFromMeanAndPrecision(Vector.Constant(4, 0.0), PositiveDefiniteMatrix.IdentityScaledBy(4, 100)).Named("mean");
            var prec = Variable.WishartFromShapeAndRate(7.0, PositiveDefiniteMatrix.Identity(4)).Named("prec");
            var x = Variable.VectorGaussianFromMeanAndPrecision(mean, prec).Named("x");
            var numDocs = Variable.New<int>().Named("numDocs");
            var doc = new Range(numDocs);
            var probs = Variable.Array<Vector>(doc).Named("probs");
            probs[doc] = Variable.Softmax(x).ForEach(doc);

            // Observations
            numDocs.ObservedValue = data.Length;
            probs.ObservedValue = data;

            var alg = new VariationalMessagePassing();
            var engine = new InferenceEngine(alg);
            var exception = Record.Exception(() => engine.Infer(mean));
            Assert.NotNull(exception);
        }

        [Fact]
        public void VectorSoftmax_PointSoftmax_WithConstraint()
        {
            var xExpected = Vector.FromArray(1, 3, 5, 1);
            var data = MMath.Softmax(xExpected);
            var mean = Vector.Zero(4);
            var prec = PositiveDefiniteMatrix.IdentityScaledBy(4, 2);
            mean[0] = xExpected[0];
            prec[0, 0] = double.PositiveInfinity;
            var x = Variable.VectorGaussianFromMeanAndPrecision(mean, prec).Named("x");
            var probs = Variable.Softmax(x);

            // Observations
            probs.ObservedValue = data;

            var alg = new VariationalMessagePassing();
            var engine = new InferenceEngine(alg);
            var xActual = engine.Infer<VectorGaussian>(x);
            Assert.True(xActual.IsPointMass);
            Assert.Equal(xExpected, xActual.Point);
        }

        [Fact]
        public void DiscreteFromLogProbsTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Gaussian[] priors = new Gaussian[]
                {
                    Gaussian.FromMeanAndPrecision(1.2, 0.4),
                    Gaussian.PointMass(0) //new Gaussian(5.6,7.8)
                };
            Range dim = new Range(priors.Length).Named("dim");
            var priorsVar = Variable.Constant(priors, dim).Named("priors");
            VariableArray<double> logProbs = Variable.Array<double>(dim).Named("logProbs");
            logProbs[dim] = Variable.Random<double, Gaussian>(priorsVar[dim]);
            Variable<int> y = Variable.DiscreteFromLogProbs(logProbs).Named("y");
            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            y.ObservedValue = 0;
            double[][] vibesResult = new double[][]
                {
                    new double[] {1.674934882797111, 4.514524088954564},
                    new double[] {0, 0}
                };
            VmpTests.TestGaussianArrayMoments(ie, logProbs, vibesResult);
            VmpTests.TestEvidence(ie, evidence, -0.41484588, 1e-1);

            y.ObservedValue = 1;
            vibesResult = new double[][]
                {
                    new double[] {-0.032201729668443, 1.611123434815797},
                    new double[] {0, 0}
                };
            VmpTests.TestGaussianArrayMoments(ie, logProbs, vibesResult);
            VmpTests.TestEvidence(ie, evidence, -1.2118952, 1e-2);
        }

        [Fact]
        public void LogisticBinomialTest()
        {
            LogisticBinomial(typeof(LogisticOp_JJ96), 3, 10, new Gaussian(1.2, 3.4));
            LogisticBinomial(typeof(LogisticOp_SJ99), 3, 10, new Gaussian(1.2, 3.4));
            LogisticBinomial(typeof(LogisticOp_JJ96), 1, 1000, new Gaussian(0, 1e6));
            LogisticBinomial(typeof(LogisticOp_SJ99), 1, 200000, new Gaussian(0, 1e6));
        }

        private void LogisticBinomial(object op, int k, int n, Gaussian wPrior)
        {
            Gaussian wExpected;
            double evExpected;
            LogisticBinomialUnrolled(op, k, n, wPrior, out wExpected, out evExpected);
            evExpected += MMath.ChooseLn(n, k);

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> w = Variable.Random(wPrior).Named("w");
            Variable<double> p = Variable.Logistic(w).Named("p");
            Variable<int> y = Variable.Binomial(n, p).Named("y");
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.NumberOfIterations = 500;
            engine.Compiler.GivePriorityTo(op);
            y.ObservedValue = k;
            Gaussian wActual = engine.Infer<Gaussian>(w);
            Console.WriteLine("w = {0} should be {1}", wActual, wExpected);
            Console.WriteLine("w should be approx {0}", MMath.Logit((double)k / n));
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(wExpected.MaxDiff(wActual) < 1e-6);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-6);
        }

        private void LogisticBinomialUnrolled(object op, int k, int n, Gaussian wPrior, out Gaussian wPost, out double evPost)
        {
            bool[] data = new bool[n];
            for (int i = 0; i < k; i++)
            {
                data[i] = true;
            }
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> w = Variable.Random(wPrior).Named("w");
            Range item = new Range(data.Length).Named("item");
            VariableArray<bool> y = Variable.Array<bool>(item).Named("y");
            y[item] = Variable.BernoulliFromLogOdds(w).ForEach(item);
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            if (op != null) engine.Compiler.GivePriorityTo(op);
            engine.NumberOfIterations = 500;
            y.ObservedValue = data;
            wPost = engine.Infer<Gaussian>(w);
            evPost = engine.Infer<Bernoulli>(evidence).LogOdds;
        }

        [Fact]
        public void LogisticSpecialFunctions()
        {
            var m_range = new double[] { -5, 1, 10 };
            var v_range = new double[] { 0.1, 1, 100 };
            // Results from Matlab
            double[,] sigma = new double[,]
                {
                    {0.007028432540686337, 0.01079678947391399, 0.311345569267483},
                    {0.7266643409440724, 0.6967346701431447, 0.539196266655789},
                    {0.9999522748829294, 0.9999251633916685, 0.8374533818955011}
                };
            double[,] sigmaPrime = new double[,]
                {
                    {0.006973926871489981, 0.01050465040932385, 0.03478229369885535},
                    {0.1947924404182624, 0.1779433801647277, 0.0390698635999826},
                    {4.772259937957203e-005, 7.482139514819998e-005, 0.02418861109897131}
                };
            double[,] sigmaPrimePrime = new double[,]
                {
                    {0.006865848081257345, 0.009955549791055831, 0.001684628721495947},
                    {-0.08502292473823414, -0.05262143047337364, -0.0003784746107552803},
                    {-4.771756526256026e-005, -7.479098557853992e-005, -0.00234275823540984}
                };

            double tolerance = 1e-8;
            Console.WriteLine("Function       ({0,4},{1,4}) | {2,-10} | {3,-10} | {4,-10}", "m", "v", "result", "truth", "err");

            for (int im = 0; im < m_range.Length; im++)
            {
                for (int iv = 0; iv < v_range.Length; iv++)
                {
                    double m = m_range[im];
                    double v = v_range[iv];
                    double result, err, relativeErr;

                    result = MMath.LogisticGaussian(m, v);
                    err = System.Math.Abs(result - sigma[im, iv]);
                    relativeErr = err / sigma[im, iv];
                    Console.WriteLine("sigma          ({0,4},{1,4}) | {2,-10:e1} | {3,-10:e1} | {4,-10:e1}", m, v, result, sigma[im, iv], err);
                    Assert.True(relativeErr < tolerance);

                    if (true)
                    {
                        Gaussian x = new Gaussian(m, v);
                        Gaussian falseMsg = LogisticOp.FalseMsg(new Beta(2, 1), x, new Gaussian());
                        result = System.Math.Exp(LogisticOp.LogAverageFactor(new Beta(2, 1), x, falseMsg) - MMath.GammaLn(3));
                        err = System.Math.Abs(result - sigma[im, iv]);
                        relativeErr = err / sigma[im, iv];
                        Console.WriteLine("sigma          ({0,4},{1,4}) | {2,-10:e1} | {3,-10:e1} | {4,-10:e1}", m, v, result, sigma[im, iv], err);
                        Assert.True(relativeErr < tolerance);
                    }
                    if (true)
                    {
                        Gaussian x = new Gaussian(-m, v);
                        Gaussian falseMsg = LogisticOp.FalseMsg(new Beta(1, 2), x, new Gaussian());
                        result = System.Math.Exp(LogisticOp.LogAverageFactor(new Beta(1, 2), x, falseMsg) - MMath.GammaLn(3));
                        err = System.Math.Abs(result - sigma[im, iv]);
                        relativeErr = err / sigma[im, iv];
                        Console.WriteLine("sigma          ({0,4},{1,4}) | {2,-10:e1} | {3,-10:e1} | {4,-10:e1}", m, v, result, sigma[im, iv], err);
                        Assert.True(relativeErr < tolerance);
                    }

                    result = MMath.LogisticGaussianDerivative(m, v);
                    err = System.Math.Abs(result - sigmaPrime[im, iv]);
                    relativeErr = err / sigmaPrime[im, iv];
                    Console.WriteLine("sigmaPrime     ({0,4},{1,4}) | {2,-10:e1} | {3,-10:e1} | {4,-10:e1}", m, v, result, sigmaPrime[im, iv], err);
                    Assert.True(relativeErr < tolerance);

                    result = MMath.LogisticGaussianDerivative2(m, v);
                    err = System.Math.Abs(result - sigmaPrimePrime[im, iv]);
                    relativeErr = err / sigmaPrimePrime[im, iv];
                    Console.WriteLine("sigmaPrimePrime({0,4},{1,4}) | {2,-10:e1} | {3,-10:e1} | {4,-10:e1}", m, v, result, sigmaPrimePrime[im, iv], err);
                    Assert.True(relativeErr < tolerance);
                }
            }
        }

        [Fact]
        public void BernoulliFromLogOddsExact()
        {
            Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 1).Named("x");
            Variable<bool> s = Variable.BernoulliFromLogOdds(x).Named("s");
            s.ObservedValue = true;
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.NumberOfIterations = 100;
            ie.Compiler.GivePriorityTo(typeof(LogisticOp));
            Gaussian xActual = ie.Infer<Gaussian>(x);
            Console.WriteLine("x = {0}", xActual);
            double m, v;
            xActual.GetMeanAndVariance(out m, out v);
            double matlabM = 0.413126805979683;
            double matlabV = 0.828868291887001;
            Gaussian xExpected = new Gaussian(matlabM, matlabV);
            double relErr = System.Math.Abs((m - matlabM) / matlabM);
            Console.WriteLine("Posterior mean is {0} should be {1} (relErr = {2})", m, matlabM, relErr);
            Assert.True(relErr < 1e-9);
            relErr = System.Math.Abs((v - matlabV) / matlabV);
            Console.WriteLine("Posterior variance is {0} should be {1} (relErr = {2})", v, matlabV, relErr);
            Assert.True(relErr < 1e-9);
        }

        [Fact]
        public void BetaBinomialTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            double a = 1.2;
            double b = 3.4;
            Variable<double> p = Variable.Beta(a, b).Named("p");
            int n = 10;
            Variable<int> x = Variable.Binomial(n, p).Named("x");
            block.CloseBlock();
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            int k = 3;
            x.ObservedValue = k;
            Beta pActual = engine.Infer<Beta>(p);
            Beta pExpected = (new Beta(a, b)) * (new Beta(k + 1, n - k + 1));
            Console.WriteLine("p = {0} should be {1}", pActual, pExpected);
            Assert.True(pExpected.MaxDiff(pActual) < 1e-10);
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            double evExpected = MMath.ChooseLn(n, k) +
                                MMath.GammaLn(a + k) - MMath.GammaLn(a) +
                                MMath.GammaLn(b + n - k) - MMath.GammaLn(b) +
                                MMath.GammaLn(a + b) - MMath.GammaLn(a + b + n);
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-10);
        }

        private static void BernoulliFromLogOddsAuxVar(bool sample, Variable<double> w)
        {
            if (sample)
            {
                if (!sample) w = -w;
                Variable<double> u = Variable.Exp(w).Attrib(new DoNotInfer());
                // f(x,u) = u*exp(-u*x)
                Variable<double> x = Variable.GammaFromShapeAndRate(1.0, u).Attrib(new DoNotInfer());
                // CER(Ga(1,1)) is equiv to adding the factor f(x) = exp(-x)
                // marginalizing x gives:  int_0^inf u*exp(-u*x - x) dx = u/(u+1)
                // since u = exp(w) this gives exp(w)/(exp(w)+1) = 1/(1 + exp(-w))
                Variable.ConstrainEqualRandom(x, new Gamma(1, 1));
            }
            else
            {
                // this matches Blei06 bound when sample=false
                Variable<double> u = Variable.Exp(w).Attrib(new DoNotInfer());
                Variable<double> x = Variable.GammaFromShapeAndRate(1, 1).Attrib(new DoNotInfer());
                Variable.ConstrainEqualRandom(x * u, new Gamma(1, 1));
                // factors are: exp(-u*x) exp(-x)
                // marginalizing x gives: 1/(u+1)
            }
        }

        //[Fact]
        internal void BernoulliFromLogOddsAuxVarTest()
        {
            // Test of an auxiliary variable representation of the BernoulliFromLogOdds factor.
            // exact w = 1.735
            // VMP JJ = 1.67
            // AuxVar = N(1.52, 0.7865)
            //Variable<double> w = Variable.GaussianFromMeanAndPrecision(1.2, 0.4).Named("w");
            Variable<double> w = Variable.GaussianFromMeanAndPrecision(1.2, 16).Named("w");
            BernoulliFromLogOddsAuxVar(true, w);
            InferenceEngine ie = new InferenceEngine();
            ie.Algorithm = new VariationalMessagePassing();
            ie.NumberOfIterations = 1000;
            ie.ShowProgress = false;
            Console.WriteLine(ie.Infer(w));
        }

        [Fact]
        public void BernoulliLogisticCRTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> w = Variable.GaussianFromMeanAndPrecision(1.2, 0.4).Named("w");
            Variable<bool> y = Variable.Bernoulli(Variable.Logistic(w).Named("p")).Named("y");
            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.Compiler.GivePriorityTo(typeof(LogisticOp_JJ96));

            y.ObservedValue = true;
            VmpTests.TestGaussianMoments(ie, w, 1.674934882797111, 4.514524088954564);
            VmpTests.TestEvidence(ie, evidence, -0.41484588);

            y.ObservedValue = false;
            VmpTests.TestGaussianMoments(ie, w, -0.032201729668443, 1.611123434815797);
            VmpTests.TestEvidence(ie, evidence, -1.2118952);
        }

        [Fact]
        public void BernoulliLogisticRRTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> w = Variable.GaussianFromMeanAndPrecision(1.2, 0.4).Named("w");
            Variable<bool> y = Variable.Bernoulli(Variable.Logistic(w).Named("p")).Named("y");
            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.Compiler.GivePriorityTo(typeof(LogisticOp_JJ96));

            Bernoulli yActual = ie.Infer<Bernoulli>(y);
            Bernoulli yExpected = new Bernoulli(0.781056415838566);
            Console.WriteLine("y = {0} should be {1}", yActual, yExpected);
            Assert.True(yExpected.MaxDiff(yActual) < 1e-4);
            VmpTests.TestGaussianMoments(ie, w, 1.271833339893970, 3.288702025399131);
            VmpTests.TestEvidence(ie, evidence, -0.21144332);
        }

        [Fact]
        public void BernoulliFromLogOddsCRTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> w = Variable.GaussianFromMeanAndPrecision(1.2, 0.4).Named("w");
            Variable<bool> y = Variable.BernoulliFromLogOdds(w).Named("y");
            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.Compiler.GivePriorityTo(typeof(LogisticOp_JJ96));

            y.ObservedValue = true;
            VmpTests.TestGaussianMoments(ie, w, 1.674934882797111, 4.514524088954564);
            VmpTests.TestEvidence(ie, evidence, -0.41484588);

            y.ObservedValue = false;
            VmpTests.TestGaussianMoments(ie, w, -0.032201729668443, 1.611123434815797);
            VmpTests.TestEvidence(ie, evidence, -1.2118952);
        }

        [Fact]
        public void BernoulliFromLogOddsRRTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> w = Variable.GaussianFromMeanAndPrecision(1.2, 0.4).Named("w");
            Variable<bool> y = Variable.BernoulliFromLogOdds(w).Named("y");
            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.Compiler.GivePriorityTo(typeof(LogisticOp_JJ96));

            if (true)
            {
                double wPrec0 = 0.4;
                double wMeanPrec0 = 1.2 * wPrec0;
                double wPrec = wPrec0;
                double wMeanPrec = wMeanPrec0;
                double yProbTrue = 0;
                double m = 0, v = 0;
                for (int iter = 0; iter < 20; iter++)
                {
                    v = 1.0 / wPrec;
                    m = wMeanPrec * v;
                    yProbTrue = MMath.Logistic(m);
                    double wMeanPrecY = yProbTrue - 0.5;
                    double t = System.Math.Sqrt(m * m + v);
                    double lambda = (t == 0) ? 0.25 : System.Math.Tanh(t / 2) / (2 * t);
                    wMeanPrec = wMeanPrec0 + wMeanPrecY;
                    wPrec = wPrec0 + lambda;
                }
                Console.WriteLine("expect y = {0} m = {1} v = {2}", yProbTrue, m, v);
            }

            Bernoulli yActual = ie.Infer<Bernoulli>(y);
            Bernoulli yExpected = new Bernoulli(0.781056415838566);
            Console.WriteLine("y = {0} should be {1}", yActual, yExpected);
            Assert.True(yExpected.MaxDiff(yActual) < 1e-4);
            VmpTests.TestGaussianMoments(ie, w, 1.271833339893970, 3.288702025399131);
            VmpTests.TestEvidence(ie, evidence, -0.21144332);
        }

        internal void SoftmaxLbfgsTest()
        {
            GaussianArray result = new GaussianArray(2);
            Dirichlet softmax = new Dirichlet(2, 1);
            GaussianArray x = new GaussianArray(2);
            x[0] = new Gaussian(0, 1);
            x[1] = new Gaussian(0, 1);
            Vector a = Vector.FromArray(0, 0);

            var s = new LBFGS(5);
            s.MaximumStep = 1e3;
            s.MaximumIterations = 100;
            s.Epsilon = 1e-10;
            s.convergenceCriteria = BFGS.ConvergenceCriteria.Objective;
            double[] mu = new double[] { 0, 0 };
            double[] s2 = new double[] { 1, 1 };
            Vector z = Vector.Zero(4);
            Vector counts = Vector.FromArray(1, 0);
            Console.WriteLine("initial z = {0}, a = {1}", z, a);
            z = s.Run(z, 1.0, delegate (Vector y, ref Vector grad) { return SoftmaxOp_KM11_LBFGS.GradientAndValueAtPoint(mu, s2, a, y, counts, grad); });
            Console.WriteLine("final z = {0}", z);
            a.SetAllElementsTo(0.2);
            Console.WriteLine("initial z = {0}, a = {1}", z, a);
            z = s.Run(z, 1.0, delegate (Vector y, ref Vector grad) { return SoftmaxOp_KM11_LBFGS.GradientAndValueAtPoint(mu, s2, a, y, counts, grad); });
            Console.WriteLine("final z = {0}", z);
            Vector z2 = Vector.Copy(z);
            z.SetAllElementsTo(0.0);
            Console.WriteLine("initial z = {0}, a = {1}", z, a);
            z = s.Run(z, 1.0, delegate (Vector y, ref Vector grad) { return SoftmaxOp_KM11_LBFGS.GradientAndValueAtPoint(mu, s2, a, y, counts, grad); });
            Console.WriteLine("final z = {0}", z);
            Assert.True(z.MaxDiff(z2) < 1e-6);
        }

        internal void BernoulliFromLogOddsComparison(bool obs)
        {
            Variable<Gaussian> xPrior = Variable.New<Gaussian>().Named("xPrior");
            Variable<double> x = Variable<double>.Random(xPrior).Named("x");
            Variable<bool> y = Variable.BernoulliFromLogOdds(x).Named("y");
            y.ObservedValue = obs;
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.NumberOfIterations = 1000;
            engine.ShowProgress = false;
            Gaussian[] xExpected = new Gaussian[1];
            Gaussian[] xPriors = new Gaussian[1];
            //xPrior.ObservedValue = new Gaussian(1.2, Math.Pow(0.5, trial));
            xPriors[0] = new Gaussian(0, 10);
            for (int model = 0; model < 3; model++)
            {
                object logistic_op;
                if (model == 0)
                {
                    Console.WriteLine("Exact");
                    logistic_op = typeof(LogisticOp);
                }
                else if (model == 1)
                {
                    Console.WriteLine("JJ96");
                    logistic_op = typeof(LogisticOp_JJ96);
                }
                else
                {
                    Console.WriteLine("SJ99");
                    logistic_op = typeof(LogisticOp_SJ99);
                }
                engine.Compiler.GivePriorityTo(logistic_op);
                try
                {
                    for (int trial = 0; trial < xExpected.Length; trial++)
                    {
                        xPrior.ObservedValue = xPriors[trial];
                        Gaussian xActual = engine.Infer<Gaussian>(x);
                        Console.Write("x = {1}", xPrior.ObservedValue, xActual);
                        if (model == 0) xExpected[trial] = xActual;
                        else
                        {
                            Console.Write(", error = {0}", xExpected[trial].MaxDiff(xActual));
                        }
                        Console.WriteLine();
                    }
                }
                finally
                {
                    engine.Compiler.RemovePriority(logistic_op);
                }
            }

            Variable<Gaussian> xPrior2 = Variable.New<Gaussian>().Named("xPrior2");
            Variable<double> x2 = Variable<double>.Random(xPrior2).Named("x2");
            BernoulliFromLogOddsAuxVar(y.ObservedValue, x2);
            Console.WriteLine("AuxVar");
            //engine.Compiler.GivePriorityTo(typeof(ExpOp_BFGS));
            for (int trial = 0; trial < xExpected.Length; trial++)
            {
                xPrior2.ObservedValue = xPriors[trial];
                Gaussian xActual = engine.Infer<Gaussian>(x2);
                Console.Write("x = {1}, error = {2}", xPrior2.ObservedValue, xActual, xExpected[trial].MaxDiff(xActual));
                Console.WriteLine();
            }

            Console.WriteLine("Softmax --------------");
            Range item = new Range(2);
            VariableArray<Gaussian> arrayPrior = Variable.Array<Gaussian>(item).Named("arrayPrior");
            VariableArray<double> array = Variable.Array<double>(item).Named("array");
            array[item] = Variable<double>.Random(arrayPrior[item]);
            Variable<Vector> softmax = Variable.Softmax(array);
            Variable<int> yInt = Variable.Discrete(softmax).Named("y");
            yInt.ObservedValue = y.ObservedValue ? 1 : 0;
            Gaussian[] prior = new Gaussian[2];
            prior[0] = Gaussian.PointMass(0);
            var Operators = new Type[]
                {
                    typeof (SoftmaxOp_BL06_LBFGS), typeof (SoftmaxOp_KM11_LBFGS), typeof (SoftmaxOp_KM11), typeof (SoftmaxOp_Bohning),
                    typeof (SoftmaxOp_Taylor)
                };

            for (int i = 0; i < Operators.Length; i++)
            {
                Console.WriteLine(Operators[i].Name);
                var ie = new InferenceEngine(new VariationalMessagePassing());
                //if (i == 2) ie.BrowserMode = BrowserMode.Always;
                //ie.NumberOfIterations = 1000;
                ie.ShowProgress = false;
                ie.Compiler.GivePriorityTo(Operators[i]);
                for (int trial = 0; trial < xExpected.Length; trial++)
                {
                    prior[1] = xPriors[trial];
                    arrayPrior.ObservedValue = (Gaussian[])prior.Clone();
                    Gaussian[] arrayActual = ie.Infer<Gaussian[]>(array);
                    Console.WriteLine("x = {1}, error = {2}", arrayPrior.ObservedValue[1], arrayActual[1], xExpected[trial].MaxDiff(arrayActual[1]));
                }
            }
        }

        /// <summary>
        /// Test the convergence rate of VMP when using arrays vs. unrolling.
        /// </summary>
        /// Verdict: VMP is very slow to converge on this model (1e4 iterations).  Unrolling doesn't help.
        internal void SparseProductTest()
        {
            int n1 = 2, n2 = 2;
            double[] aTrue = new double[n1];
            double[] bTrue = new double[n2];
            for (int i = 0; i < n1; i++)
            {
                aTrue[i] = i + 1;
            }
            for (int i = 0; i < n2; i++)
            {
                bTrue[i] = n2 - i;
            }
            double[,] present = new double[n1, n2];
            double[,] data = new double[n1, n2];
            for (int i = 0; i < n1; i++)
            {
                for (int j = 0; j < n2; j++)
                {
                    if (i == j || i == j + 1)
                    {
                        // chain structure
                        //if (Math.Abs(i - j) <= 1) {
                        present[i, j] = 1.0;
                        data[i, j] = aTrue[i] * bTrue[j];
                    }
                }
            }
            //WriteFactors(SparseProductArrayModel(data, present));
            WriteFactors(SparseProductUnrolledModel(data, present));
        }

        private void WriteFactors(DistributionArray<Gaussian>[] result)
        {
            double scale = result[0][0].GetMean();
            //ScaleGaussians(result[0], 1.0/scale);
            //ScaleGaussians(result[1], scale);
            Console.WriteLine(result[0]);
            Console.WriteLine(result[1]);
        }

        internal void ScaleGaussians(DistributionArray<Gaussian> dists, double scale)
        {
            for (int i = 0; i < dists.Count; i++)
            {
                Gaussian dist = dists[i];
                double mean, precision;
                dist.GetMeanAndPrecision(out mean, out precision);
                dist.SetMeanAndPrecision(mean * scale, precision / (scale * scale));
                dists[i] = dist;
            }
        }

        public DistributionArray<Gaussian>[] SparseProductArrayModel(double[,] data, double[,] present)
        {
            int n1 = data.GetLength(0);
            int n2 = data.GetLength(1);
            Range r1 = new Range(n1);
            Range r2 = new Range(n2);
            VariableArray<double> a = Variable.Array<double>(r1).Named("a");
            double[] aPrecisions = new double[n1];
            for (int i = 0; i < n1; i++)
            {
                aPrecisions[i] = 1e-4;
            }
            //aPrecisions[0] = Double.PositiveInfinity;
            VariableArray<double> aPrecision = Variable.Constant<double>(aPrecisions, r1);
            a[r1] = Variable.GaussianFromMeanAndPrecision(1, aPrecision[r1]);
            VariableArray<double> b = Variable.Array<double>(r2).Named("b");
            b[r2] = Variable.GaussianFromMeanAndPrecision(1, 1e-4).ForEach(r2);
            VariableArray2D<double> precision = Variable.Constant<double>(present, r1, r2).Named("present");
            VariableArray2D<double> x = Variable.Constant<double>(data, r1, r2).Named("x");
            x[r1, r2] = Variable.GaussianFromMeanAndPrecision((a[r1] * b[r2]).Named("ab"), precision[r1, r2]);

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.ShowProgress = false;
            DistributionArray<Gaussian> aActual = null, bActual = null;
            if (false)
            {
                engine.NumberOfIterations = 10;
                aActual = engine.Infer<DistributionArray<Gaussian>>(a);
                bActual = engine.Infer<DistributionArray<Gaussian>>(b);
            }
            else
            {
                var ca = engine.GetCompiledInferenceAlgorithm(a, b);
                ca.Execute((int)1e6);
                List<DistributionArray<Gaussian>> aHistory = new List<DistributionArray<Gaussian>>();
                List<DistributionArray<Gaussian>> bHistory = new List<DistributionArray<Gaussian>>();
                Stopwatch watch = new Stopwatch();
                for (int iter = 0; iter < 1e6; iter++)
                {
                    watch.Start();
                    //ca.Update();
                    watch.Stop();
                    aActual = ca.Marginal<DistributionArray<Gaussian>>(a.Name);
                    bActual = ca.Marginal<DistributionArray<Gaussian>>(b.Name);
                    aHistory.Add((DistributionArray<Gaussian>)aActual.Clone());
                    bHistory.Add((DistributionArray<Gaussian>)bActual.Clone());
                }
                Console.WriteLine("ms per iteration = {0}", (double)watch.ElapsedMilliseconds / 1e6);
                WriteDistanceGraph("arrayDistance.mat", aHistory, bHistory);
                //Console.WriteLine("iters = " + iter);
            }
            return new DistributionArray<Gaussian>[] { aActual, bActual };
        }

        public DistributionArray<Gaussian>[] SparseProductUnrolledModel(double[,] data, double[,] present)
        {
            int n1 = data.GetLength(0);
            int n2 = data.GetLength(1);
            Variable<double>[] a = new Variable<double>[n1];
            for (int i = 0; i < n1; i++)
            {
                a[i] = Variable.GaussianFromMeanAndPrecision(1, (i == 0) ? Double.PositiveInfinity : 1e-8).Named("a" + i);
                //a[i] = Variable.GaussianFromMeanAndPrecision(1, 1e-4).Named("a" + i);
            }
            Variable<double>[] b = new Variable<double>[n2];
            for (int i = 0; i < n2; i++)
            {
                b[i] = Variable.GaussianFromMeanAndPrecision(1, 1e-4).Named("b" + i);
            }
            for (int i = 0; i < n1; i++)
            {
                for (int j = 0; j < n2; j++)
                {
                    if (present[i, j] != 0)
                    {
                        Variable.ConstrainEqual(data[i, j], Variable.GaussianFromMeanAndPrecision((a[i] * b[j]).Named("a" + i + "b" + j), 1).Named("data" + i + j));
                    }
                }
            }

            InferenceEngine engine = new InferenceEngine(); //new VariationalMessagePassing());
            engine.ShowSchedule = true;
            //engine.NumberOfIterations = 100;
            DistributionArray<Gaussian> aActual = new DistributionStructArray<Gaussian, double>(n1);
            DistributionArray<Gaussian> bActual = new DistributionStructArray<Gaussian, double>(n2);
            if (false)
            {
                for (int i = 0; i < n1; i++)
                {
                    aActual[i] = engine.Infer<Gaussian>(a[i]);
                }
                for (int i = 0; i < n2; i++)
                {
                    bActual[i] = engine.Infer<Gaussian>(b[i]);
                }
            }
            else
            {
                IVariable[] vars = new IVariable[n1 + n2];
                a.CopyTo(vars, 0);
                b.CopyTo(vars, n1);
                var ca = engine.GetCompiledInferenceAlgorithm(vars);
                List<DistributionArray<Gaussian>> aHistory = new List<DistributionArray<Gaussian>>();
                List<DistributionArray<Gaussian>> bHistory = new List<DistributionArray<Gaussian>>();
                Stopwatch watch = new Stopwatch();
                ca.Execute(100);
                for (int iter = 0; iter < 1e2; iter++)
                {
                    watch.Start();
                    //ca.Update();
                    watch.Stop();
                    for (int i = 0; i < n1; i++)
                    {
                        aActual[i] = ca.Marginal<Gaussian>(a[i].Name);
                    }
                    aHistory.Add((DistributionArray<Gaussian>)aActual.Clone());
                    for (int i = 0; i < n2; i++)
                    {
                        bActual[i] = ca.Marginal<Gaussian>(b[i].Name);
                    }
                    bHistory.Add((DistributionArray<Gaussian>)bActual.Clone());
                }
                Console.WriteLine("ms per iteration = {0}", (double)watch.ElapsedMilliseconds / 1e6);
                WriteDistanceGraph("unrolledDistance.mat", aHistory, bHistory);
                // compute an upper bound on the distance
                //Console.WriteLine("iters = " + iter);
            }
            return new DistributionArray<Gaussian>[] { aActual, bActual };
        }

        private static void WriteDistanceGraph<T>(string fileName, IList<T> history, IList<T> history2)
            where T : Diffable
        {
            T final = history[history.Count - 1];
            T final2 = history2[history2.Count - 1];
            double[] distance = new double[history.Count];
            distance[history.Count - 1] = 0;
            for (int i = history.Count - 2; i >= 0; i--)
            {
                double d = distance[i + 1];
                d = System.Math.Max(d, final.MaxDiff(history[i]));
                d = System.Math.Max(d, final2.MaxDiff(history2[i]));
                distance[i] = d;
            }
            using (MatlabWriter writer = new MatlabWriter(fileName))
            {
                writer.Write("distance", distance);
            }
        }

        internal void GaussianNotObservedTest()
        {
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(2, 3).Named("mean");
            Variable<double> precision = Variable.GammaFromShapeAndScale(4, 5).Named("precision");
            Variable<double> x = Variable.GaussianFromMeanAndPrecision(mean, precision).Named("x");

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine("mean = {0}", engine.Infer<Gaussian>(mean));
            Console.WriteLine("precision = {0}", engine.Infer<Gamma>(precision));
            Console.WriteLine("x = {0}", engine.Infer<Gaussian>(x));
        }

        internal void GaussianNotObserved2Test()
        {
            // this is a re-expression of the model in GaussianNotObservedTest, to give different results.
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(2, 3).Named("mean");
            Variable<double> precision = Variable.GammaFromShapeAndScale(4, 5).Named("precision");
            Variable<double> noise = Variable.GaussianFromMeanAndPrecision(0, precision).Named("noise");
            Variable<double> x = (mean + noise).Named("x");

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine("mean = {0}", engine.Infer<Gaussian>(mean));
            Console.WriteLine("precision = {0}", engine.Infer<Gamma>(precision));
            Console.WriteLine("x = {0}", engine.Infer<Gaussian>(x));
        }

        internal void GaussianNotObserved3Test()
        {
            // this is a re-expression of the model in GaussianNotObservedTest, to give different results.
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(2, 3).Named("mean");
            Variable<double> sqrtPrecision = Variable.GaussianFromMeanAndVariance(4, 5).Named("sqrtPrecision");
            //Variable<double> sqrtPrecision = Variable.GammaFromShapeAndScale(4,5).Named("sqrtPrecision");
            Variable<double> stdNoise = Variable.GaussianFromMeanAndPrecision(0, 1).Named("stdNoise");
            Variable<double> noise = (stdNoise * sqrtPrecision).Named("noise");
            Variable<double> x = (mean + noise).Named("x");

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine("mean = {0}", engine.Infer<Gaussian>(mean));
            Console.WriteLine("sqrtPrecision = {0}", engine.Infer(sqrtPrecision));
            Console.WriteLine("x = {0}", engine.Infer<Gaussian>(x));
        }


        [Fact]
        public void ConstraintsTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable.ConstrainEqual(.75, Variable.GaussianFromMeanAndVariance(0, 1));
            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            Bernoulli B1 = ie.Infer<Bernoulli>(evidence);

            Variable<bool> evidence2 = Variable.Bernoulli(0.5).Named("evidence2");
            IfBlock block2 = Variable.If(evidence2);
            Variable.ConstrainEqualRandom(Variable.Constant(.75), new Gaussian(0, 1));
            block2.CloseBlock();
            InferenceEngine ie2 = new InferenceEngine(new VariationalMessagePassing());
            Bernoulli B2 = ie.Infer<Bernoulli>(evidence2);


            Variable<bool> evidence3 = Variable.Bernoulli(0.5).Named("evidence3");
            IfBlock block3 = Variable.If(evidence3);
            Variable.ConstrainEqual(Variable.Constant(.75), Variable.GaussianFromMeanAndVariance(0, 1));
            block3.CloseBlock();
            InferenceEngine ie3 = new InferenceEngine(new VariationalMessagePassing());
            Bernoulli B3 = ie.Infer<Bernoulli>(evidence3);


            Assert.True(B1.MaxDiff(B2) < 1e-10);
            Assert.True(B1.MaxDiff(B3) < 1e-10);
            Assert.True(B2.MaxDiff(B3) < 1e-10);
        }


        [Fact]
        public void InferNoUsesTest()
        {
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100);
            Variable<double> x = Variable.GaussianFromMeanAndVariance(mean, 1);

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Gaussian expected = Gaussian.FromMeanAndVariance(0, 1);
            Gaussian actual = engine.Infer<Gaussian>(x);
            Assert.True(expected.MaxDiff(actual) < 1e-10);
            expected = Gaussian.FromMeanAndVariance(0, 0.9901);
            actual = engine.Infer<Gaussian>(mean);
            Assert.True(expected.MaxDiff(actual) < 1e-4);
        }


        [Fact]
        public void DisconnectedModelTest()
        {
            Variable<double> a = Variable.GaussianFromMeanAndVariance(0, 1);
            Variable<double> b = Variable.GaussianFromMeanAndVariance(0, 1);

            // Create an inference engine for VMP
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            // Retrieve the posterior distributions
            Gaussian aActual = engine.Infer<Gaussian>(a);
            Gaussian bActual = engine.Infer<Gaussian>(b);
            Gaussian aExpected = Gaussian.FromMeanAndVariance(0, 1);
            Gaussian bExpected = Gaussian.FromMeanAndVariance(0, 1);
            Assert.True(aActual.Equals(aExpected));
            Assert.True(bActual.Equals(bExpected));

            // check that model is not recompiled unnecessarily
            for (int iter = 0; iter < 2; iter++)
            {
                engine.Infer<Gaussian>(a);
                engine.Infer<Gaussian>(b);
            }
        }

        [Fact]
        public void DiscreteTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<Vector> pi = Variable.Dirichlet(new double[] { 1, 1, 1, 1 }).Named("pi");
            int xindex = 2;
            Variable<int> x = Variable.Observed<int>(xindex).Named("x");
            Variable.ConstrainEqual<int>(x, Variable.Discrete(pi));
            block.CloseBlock();

            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            Dirichlet pipost = ie.Infer<Dirichlet>(pi);
            double[] VibesSuffStats = new double[] { -2.08333333351352, -2.08333333351352, -1.08333333348476, -2.08333333351352 };

            VmpTests.TestDirichletMoments(ie, pi, VibesSuffStats);
            VmpTests.TestEvidence(ie, evidence, -1.3862944);
        }


        [Fact]
        public void Gaussian_nonNoisyTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> prec = Variable.GammaFromShapeAndScale(1, 1).Named("prec");
            Variable<double> m = Variable.GaussianFromMeanAndPrecision(0, 10).Named("m");

            Variable<double> x = Variable.Observed<double>(.5).Named("x");
            Variable.ConstrainEqual<double>(x, Variable.GaussianFromMeanAndPrecision(m, prec));
            block.CloseBlock();

            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            TestGammaMoments(ie, prec, 1.31365592127115, -0.09616110436570);
            VmpTests.TestGaussianMoments(ie, m, 0.05805620793193, 0.09175928169305);
            VmpTests.TestEvidence(ie, evidence, -1.2592065);
        }


        [Fact]
        public void GaussianTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> prec = Variable.GammaFromShapeAndScale(1, 1).Named("prec");
            Variable<double> m = Variable.GaussianFromMeanAndPrecision(0, 10).Named("m");
            Variable<double> x = Variable.GaussianFromMeanAndPrecision(m, prec).Named("x");

            Variable<double> xNoisy = Variable.Observed<double>(.5).Named("xNoisy");
            Variable.ConstrainEqual<double>(xNoisy, Variable.GaussianFromMeanAndPrecision(x, 1.0));
            block.CloseBlock();

            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.NumberOfIterations = 60;
            TestGammaMoments(ie, prec, 1.15259073127590, -0.22696291577087);
            VmpTests.TestGaussianMoments(ie, m, 0.02541152365878, 0.09031101023371);
            VmpTests.TestGaussianMoments(ie, x, 0.24588461067251, 0.52501573430760);
            VmpTests.TestEvidence(ie, evidence, -1.5888991);
        }


        [Fact]
        public void SumofSumPlusProd()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> a = Variable.GaussianFromMeanAndPrecision(-1, 1).Named("a");
            Variable<double> b = Variable.GaussianFromMeanAndPrecision(2, 10).Named("b");
            Variable<double> c = Variable.GaussianFromMeanAndPrecision(10, 10).Named("c");
            Variable<double> prec = Variable.GammaFromShapeAndScale(1, 1).Named("prec");

            Variable<double> xNoisy = Variable.Observed<double>(45).Named("xNoisy");
            Variable.ConstrainEqual<double>(xNoisy, Variable.GaussianFromMeanAndPrecision(((a + b) + (b * c)) * ((c + b) + (a * c)), prec));

            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            block.CloseBlock();

            // VIBES is not accurate enough - this test fails (slightly out of tolerance) because we 
            // believe Infer.NET is more accurate.
            /*VmpTests.TestGaussianMoments(ie, a, -0.96321139612027, 0.92802007543107);
            VmpTests.TestGaussianMoments(ie, b, 1.89666231973957, 3.60679969188059);
            VmpTests.TestGaussianMoments(ie, c, 9.96760307110386, 99.36404187723319);
            VmpTests.TestGammaMoments(ie, prec, 0.10454746778161, -2.62708920774696);
            VmpTests.TestEvidence(ie, evidence, -10.141172);*/
        }

        /// <summary>
        /// This test fails because Infer.NET does not recognize multi-linear functions.
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")]
        public void MultilinearFunctionTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Gaussian aPrior = Gaussian.FromMeanAndPrecision(-1, 0.1);
            Gaussian bPrior = Gaussian.FromMeanAndPrecision(2, 0.1);
            Gaussian cPrior = Gaussian.FromMeanAndPrecision(10, 1);
            Variable<double> a = Variable.Random(aPrior).Named("a");
            Variable<double> b = Variable.Random(bPrior).Named("b");
            Variable<double> c = Variable.Random(cPrior).Named("c");

            Variable<double> xNoisy = Variable.Observed<double>(45).Named("xNoisy");
            Variable.ConstrainEqual<double>(xNoisy, Variable.GaussianFromMeanAndPrecision(a * b + b * c + a * c, 1));

            block.CloseBlock();

            a.InitialiseTo(aPrior);
            b.InitialiseTo(bPrior);
            //c.InitialiseTo(cPrior);

            Gaussian aExpected = new Gaussian(1.012, 0.00577);
            Gaussian bExpected = new Gaussian(3.167, 0.008247);
            Gaussian cExpected = new Gaussian(9.992, 0.05412);
            if (true)
            {
                // compute the correct VB solution
                double mx = 45;
                Gaussian aPost = aPrior;
                Gaussian bPost = bPrior;
                Gaussian cPost = cPrior;
                for (int iter = 0; iter < 1000; iter++)
                {
                    // expanding the square in the log-factor gives:
                    // a^2*b^2 + b^2*c^2 + a^2*c^2 + 2ab^2*c + 2a*b*c^2 + 2a^2*b*c + x^2 - 2xab - 2xbc - 2xac
                    double ma, va;
                    aPost.GetMeanAndVariance(out ma, out va);
                    double ma2 = ma * ma + va;
                    double mb, vb;
                    bPost.GetMeanAndVariance(out mb, out vb);
                    double mb2 = mb * mb + vb;
                    double mc, vc;
                    cPost.GetMeanAndVariance(out mc, out vc);
                    double mc2 = mc * mc + vc;

                    Gaussian aB = Gaussian.FromNatural(mx * (mb + mc) - mb2 * mc - mb * mc2, mb2 + mc2 + 2 * mb * mc);
                    aPost = aB * aPrior;
                    aPost.GetMeanAndVariance(out ma, out va);
                    ma2 = ma * ma + va;

                    Gaussian bB = Gaussian.FromNatural(mx * (ma + mc) - ma2 * mc - ma * mc2, ma2 + mc2 + 2 * ma * mc);
                    bPost = bB * bPrior;
                    bPost.GetMeanAndVariance(out mb, out vb);
                    mb2 = mb * mb + vb;

                    Gaussian cB = Gaussian.FromNatural(mx * (ma + mb) - mb2 * ma - mb * ma2, mb2 + ma2 + 2 * mb * ma);
                    cPost = cB * cPrior;
                    //Console.WriteLine("{0} a={1} b={2} c={3}", iter, aPost, bPost, cPost);
                }
                aExpected = aPost;
                bExpected = bPost;
                cExpected = cPost;
            }


            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.OptimiseForVariables = new[] { a, b, c };
            ie.ShowProgress = false;
            ie.NumberOfIterations = 1000;
            VmpTests.TestGaussianMoments(ie, a, aExpected);
            VmpTests.TestGaussianMoments(ie, b, bExpected);
            VmpTests.TestGaussianMoments(ie, c, cExpected);
            //VmpTests.TestEvidence(ie, evidence, -10.141172);  // TODO
        }

        [Fact]
        public void SumTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> z = Variable.GaussianFromMeanAndPrecision(2, 1).Named("z");
            Variable<double> y = Variable.GaussianFromMeanAndPrecision(10, 1).Named("y");
            Variable<double> x = Variable<double>.Factor(Factor.Plus, z, y).Named("x");
            Variable<double> xNoisy = Variable.Observed<double>(11).Named("xNoisy");
            Variable.ConstrainEqual<double>(xNoisy, Variable.GaussianFromMeanAndPrecision(x, 10));
            block.CloseBlock();

            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.NumberOfIterations = 60;

            VmpTests.TestGaussianMoments(ie, z, 1.52382304335853, 2.41294575837955);
            VmpTests.TestGaussianMoments(ie, y, 9.52379465230561, 90.79357367019412);
            VmpTests.TestGaussianMoments(ie, x, 11.047617695664, 122.231674931370);
            VmpTests.TestEvidence(ie, evidence, -2.4036365);
        }

        /// <summary>
        /// Test hybrid inference
        /// </summary>
        [Fact]
        public void HybridProductTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> a = Variable.GaussianFromMeanAndPrecision(1, 100).Named("a");
            Variable<double> b = Variable.GaussianFromMeanAndPrecision(5, 10).Named("b");
            Variable<double> ab = (a * b).Named("ab");
            Variable<double> abNoisy = Variable.GaussianFromMeanAndPrecision(ab, 10).Named("abNoisy");
            abNoisy.ObservedValue = 3;
            block.CloseBlock();

            InferenceEngine ie = new InferenceEngine();
            var alg = new VariationalMessagePassing();
            bool useHybrid = true;
            if (useHybrid)
            {
                var attr = new Algorithm(alg);
                var attr2 = new FactorAlgorithm(alg);
                a.AddAttribute(attr);
                b.AddAttribute(attr);
                ab.AddAttribute(attr); // needed for ab's moments to be correct
                a.AddAttribute(attr2);
                b.AddAttribute(attr2);
                ab.AddAttribute(attr2);
                abNoisy.AddAttribute(attr2);
            }
            else
            {
                ie.Algorithm = alg;
            }
            ie.NumberOfIterations = 60;

            VmpTests.TestGaussianMoments(ie, a, 0.76234298935992, 0.58437022931749);
            VmpTests.TestGaussianMoments(ie, b, 4.59932121240072, 21.21687215512205);
            VmpTests.TestGaussianMoments(ie, ab, 3.50626028208806, 12.39850844668848);
            VmpTests.TestEvidence(ie, evidence, -5.474173);
        }

        /// <summary>
        /// Test the convergence rate of VMP
        /// This test only passes when scheduler handles triggers correctly
        /// </summary>
        [Fact]
        public void SumOfProductTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> a = Variable.GaussianFromMeanAndPrecision(-1, 10).Named("a");
            Variable<double> b = Variable.GaussianFromMeanAndPrecision(2, 10).Named("b");
            Variable<double> c = Variable.GaussianFromMeanAndPrecision(10, 10).Named("c");
            Variable<double> ab = Variable<double>.Factor(Factor.Product, a, b).Named("ab");
            Variable<double> bc = Variable<double>.Factor(Factor.Product, b, c).Named("bc");
            Variable<double> x = Variable<double>.Factor(Factor.Plus, ab, bc).Named("x");
            Variable<double> xNoisy = Variable.Observed<double>(18).Named("xNoisy");
            Variable.ConstrainEqual<double>(xNoisy, Variable.GaussianFromMeanAndPrecision(x, 10));
            block.CloseBlock();

            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.NumberOfIterations = 60;
            VmpTests.TestGaussianMoments(ie, a, -0.98270731643429, 0.98580908124649);
            VmpTests.TestGaussianMoments(ie, b, 1.99381042664816, 3.97625909745750);
            VmpTests.TestGaussianMoments(ie, c, 10.006537734961, 100.150892857865);
            VmpTests.TestGaussianMoments(ie, ab, -1.95933209385013, 3.91983232766259);
            VmpTests.TestGaussianMoments(ie, bc, 19.951139270614, 398.225898844578);
            VmpTests.TestGaussianMoments(ie, x, 17.991807176764, 323.963916208667);
            VmpTests.TestEvidence(ie, evidence, -3.6875198);
        }


        [Fact]
        public void MultipleProductTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> a = Variable.GaussianFromMeanAndPrecision(1, 100).Named("a");
            Variable<double> b = Variable.GaussianFromMeanAndPrecision(10, 10).Named("b");
            Variable<double> ab = (a * b).Named("ab");
            Variable<double> c = Variable.GaussianFromMeanAndPrecision(5, 100).Named("c");
            Variable<double> abc = (ab * c).Named("abc");
            Variable<double> d = Variable.GaussianFromMeanAndPrecision(15, 100).Named("d");
            Variable<double> abd = (ab * d).Named("abd");

            Variable.ConstrainEqual<double>(4, Variable.GaussianFromMeanAndPrecision(abc, 1).Named("abcNoisy"));
            Variable.ConstrainEqual<double>(10, Variable.GaussianFromMeanAndPrecision(abd, 10).Named("abdNoisy"));
            block.CloseBlock();

            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            // the number of iterations required depends a lot on the schedule chosen.
            //ie.NumberOfIterations = 66;  // old scheduler
            ie.NumberOfIterations = 100; // new scheduler

            // hand-coded VMP
            Gaussian pa = Gaussian.FromMeanAndPrecision(1, 100);
            Gaussian pb = Gaussian.FromMeanAndPrecision(10, 10);
            Gaussian pc = Gaussian.FromMeanAndPrecision(5, 100);
            Gaussian pd = Gaussian.FromMeanAndPrecision(15, 100);
            Gaussian qa = pa;
            Gaussian qb = pb;
            Gaussian qc = pc;
            Gaussian qd = pd;
            for (int iter = 0; iter < 200; iter++)
            {
                Gaussian old_qa = qa;
                // q(a) = p(a) exp(sum_{b,d} q(b) q(d) log N(abd; 10, 1/10) + sum_{b,c} q(b) q(c) log N(abc; 4, 1))
                double mb, vb;
                qb.GetMeanAndVariance(out mb, out vb);
                double mc, vc;
                qc.GetMeanAndVariance(out mc, out vc);
                double md, vd;
                qd.GetMeanAndVariance(out md, out vd);
                qa.Precision = pa.Precision + 10 * (mb * mb + vb) * (md * md + vd) + 1 * (mb * mb + vb) * (mc * mc + vc);
                qa.MeanTimesPrecision = pa.MeanTimesPrecision + 10 * mb * md * 10 + 1 * mb * mc * 4;
                // q(b) = p(b) exp(sum_{a,d} q(a) q(d) log N(abd; 10, 1/10) + sum_{a,c} q(a) q(c) log N(abc; 4, 1))
                double ma, va;
                qa.GetMeanAndVariance(out ma, out va);
                qb.Precision = pb.Precision + 10 * (ma * ma + va) * (md * md + vd) + 1 * (ma * ma + va) * (mc * mc + vc);
                qb.MeanTimesPrecision = pb.MeanTimesPrecision + 10 * ma * md * 10 + 1 * ma * mc * 4;
                // q(c) = p(c) exp(sum_{a,b} q(a) q(b) log N(abc; 4, 1))
                qb.GetMeanAndVariance(out mb, out vb);
                qc.Precision = pc.Precision + 1 * (ma * ma + va) * (mb * mb + vb);
                qc.MeanTimesPrecision = pc.MeanTimesPrecision + 1 * ma * mb * 4;
                // q(d) = p(d) exp(sum_{a,b} q(a) q(b) log N(abc; 10, 1/10))
                qd.Precision = pd.Precision + 10 * (ma * ma + va) * (mb * mb + vb);
                qd.MeanTimesPrecision = pd.MeanTimesPrecision + 10 * ma * mb * 10;
                if (qa.MaxDiff(old_qa) < 1e-10)
                {
                    Console.WriteLine("Converged after {0} iterations", iter);
                    break;
                }
            }
            Gaussian aActual = ie.Infer<Gaussian>(a);
            Gaussian bActual = ie.Infer<Gaussian>(b);
            Gaussian cActual = ie.Infer<Gaussian>(c);
            Gaussian dActual = ie.Infer<Gaussian>(d);
            Console.WriteLine("a = {0} (should be {1})", aActual, qa);
            Console.WriteLine("b = {0} (should be {1})", bActual, qb);
            Console.WriteLine("c = {0} (should be {1})", cActual, qc);
            Console.WriteLine("d = {0} (should be {1})", dActual, qd);
            Assert.True(qa.MaxDiff(aActual) < 1e-8);
            Assert.True(qb.MaxDiff(bActual) < 1e-8);
            Assert.True(qc.MaxDiff(cActual) < 1e-8);
            Assert.True(qd.MaxDiff(dActual) < 1e-8);

            /*  Vibes doesn't work correctly on this model
            VmpTests.TestGaussianMoments(ie, a, 0.11758075658733, 0.01422137725767);
            VmpTests.TestGaussianMoments(ie, b, 9.88567402583078, 97.82314646501126);
            VmpTests.TestGaussianMoments(ie, ab, 1.16236503133285, 1.39117987041135);
            VmpTests.TestGaussianMoments(ie, c, 4.97725207232352, 24.78290098228794);
            VmpTests.TestGaussianMoments(ie, d, 14.188490757937, 201.322048709949);
            VmpTests.TestGaussianMoments(ie, abc, 5.78538376099783, 34.47747297695656);
            VmpTests.TestGaussianMoments(ie, abd, 16.492205504415, 280.075181635254);
            VmpTests.TestEvidence(ie,evidence, -327.02942);
             */
        }

        [Fact]
        public void MultipleProduct2Test()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> a = Variable.GaussianFromMeanAndPrecision(1, 100).Named("a");
            Variable<double> b = Variable.GaussianFromMeanAndPrecision(10, 10).Named("b");
            Variable<double> ab = (a * b).Named("ab");
            Variable<double> c = Variable.GaussianFromMeanAndPrecision(7, 10).Named("c");
            Variable<double> abc = (ab * c).Named("abc");
            Variable<double> abcNoisy = Variable.Observed<double>(35).Named("abcNoisy");
            Variable.ConstrainEqual<double>(abcNoisy, Variable.GaussianFromMeanAndPrecision(abc, 1));
            block.CloseBlock();

            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.NumberOfIterations = 60;

            VmpTests.TestGaussianMoments(ie, a, 0.55261061332515, 0.30561281284074);
            VmpTests.TestGaussianMoments(ie, b, 9.74207525518016, 94.95073960987557);
            VmpTests.TestGaussianMoments(ie, ab, 5.38357418182485, 29.01816261348262);
            VmpTests.TestGaussianMoments(ie, c, 6.62319931133732, 43.89239820210073);
            VmpTests.TestGaussianMoments(ie, abc, 35.65648481360, 1273.67674852429);
            VmpTests.TestEvidence(ie, evidence, -15.167053);
        }

        [Fact]
        public void ProductTest2()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            //IfBlock block = Variable.If(evidence);
            Variable<double> a = Variable.GaussianFromMeanAndPrecision(1, 100).Named("a");
            Variable<double> b = Variable.GaussianFromMeanAndPrecision(10, 10).Named("b");
            Variable<double> ab = (a * b).Named("ab");
            Variable<double> c = Variable.GammaFromShapeAndScale(1, 1).Named("c");
            Variable<double> d = Variable.GammaFromShapeAndScale(4, .01).Named("d");

            Variable.ConstrainEqual<double>(4, Variable.GaussianFromMeanAndPrecision(ab, c).Named("abcNoisy"));
            Variable.ConstrainEqual<double>(8, Variable.GaussianFromMeanAndPrecision(ab, d).Named("abdNoisy"));
            //block.CloseBlock();

            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.NumberOfIterations = 60;

            /*  Vibes doesn't work correctly on this model
            VmpTests.TestGaussianMoments(ie, a, 0.95020225184138, 0.91205222217777);
            VmpTests.TestGaussianMoments(ie, b, 9.95169948645992, 99.13549456794992);
            VmpTests.TestGaussianMoments(ie, ab, 9.45612726168295, 90.41674811739067);
            TestGammaMoments(ie, c, 0.09155348868565, -2.75980703587893);
            TestGammaMoments(ie, d, 0.04430906396862, -3.23177248893962);
            VmpTests.TestEvidence(ie,evidence, -8.000849);
            */
        }

        // test inference in product factor with and without symmetry breaking
        internal void ProductTest3()
        {
            var evidence = Variable.Bernoulli(0.5).Named("evidence");
            var block = Variable.If(evidence);
            var aPrec = Variable.Observed(1.0);
            Variable<double> a = Variable.GaussianFromMeanAndPrecision(0, aPrec).Named("a");
            Variable<double> b = Variable.GaussianFromMeanAndPrecision(0, 1).Named("b");
            Variable<double> ab = (a * b).Named("ab");
            var noiseVar = Variable.Observed(1.0);
            var abn = Variable.GaussianFromMeanAndVariance(ab, noiseVar).Named("abNoisy");
            double x = 2;
            abn.ObservedValue = x;
            Variable.ConstrainPositive(a);
            //a.InitialiseTo(Gaussian.PointMass(1));
            block.CloseBlock();

            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.ShowProgress = false;

            InferenceEngine ie2 = new InferenceEngine();
            ie2.ShowProgress = false;
            ie2.Compiler.GivePriorityTo(typeof(GaussianProductOp_Slow));
            //GaussianProductOp.ForceProper = false;

            if (false)
            {
                for (int iter = 0; iter < 20; iter++)
                {
                    ie.NumberOfIterations = iter;

                    Console.WriteLine("iter {3}: a = {0} b = {1} ab = {2}", ie.Infer(a), ie.Infer(b), ie.Infer(ab), iter);

                }
                Console.WriteLine("ab should be {0}", (1 - 1 / (x * x)) * x - 1);
            }
            for (int i = 0; i < 100; i++)
            {
                double v = 4.0 * (1 + i) / 100;
                //noiseVar.ObservedValue = v;
                aPrec.ObservedValue = v;
                var abPost = ie.Infer<Gaussian>(ab);
                var abPost2 = ie2.Infer<Gaussian>(ab);
                var ev = ie.Infer<Bernoulli>(evidence).LogOdds;
                var ev2 = ie2.Infer<Bernoulli>(evidence).LogOdds;
                // this is more accurate than above since it doesn't have ForceProper
                ev2 = GaussianProductOp.LogAverageFactor(new Gaussian(x, noiseVar.ObservedValue),
                    Gaussian.FromMeanAndPrecision(0, aPrec.ObservedValue), new Gaussian(0, 1), new Gaussian());
                Console.WriteLine("{0}: {1} {2} {3}", System.Math.Sqrt(v), abPost.GetMean(), abPost2.GetMean(), ev);
            }
        }

        [Fact]
        public void ProductObservedTwiceTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            //IfBlock block = Variable.If(evidence);
            Variable<double> a = Variable.GaussianFromMeanAndPrecision(1, 100).Named("a");
            Variable<double> b = Variable.GaussianFromMeanAndPrecision(10, 10).Named("b");
            Variable<double> ab = (a * b).Named("ab");

            Variable.ConstrainEqual<double>(8, Variable.GaussianFromMeanAndPrecision(ab, 2).Named("abNoisy"));
            Variable.ConstrainEqual<double>(11, Variable.GaussianFromMeanAndPrecision(ab, 3).Named("ab2Noisy"));
            //block.CloseBlock();

            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());

            // hand-coded VMP
            Gaussian pa = Gaussian.FromMeanAndPrecision(1, 100);
            Gaussian pb = Gaussian.FromMeanAndPrecision(10, 10);
            Gaussian qa = pa;
            Gaussian qb = pb;
            for (int iter = 0; iter < 100; iter++)
            {
                // q(a) = p(a) exp(sum_{b} q(b) (log N(ab; 8, 1/2) + log N(ab; 11, 1/3)))
                double mb, vb;
                qb.GetMeanAndVariance(out mb, out vb);
                qa.Precision = pa.Precision + 2 * (mb * mb + vb) + 3 * (mb * mb + vb);
                qa.MeanTimesPrecision = pa.MeanTimesPrecision + 2 * mb * 8 + 3 * mb * 11;
                // q(b) = p(b) exp(sum_{a} q(a) (log N(ab; 8, 1/2) + log N(ab; 11, 1/3)))
                double ma, va;
                qa.GetMeanAndVariance(out ma, out va);
                qb.Precision = pb.Precision + 2 * (ma * ma + va) + 3 * (ma * ma + va);
                qb.MeanTimesPrecision = pb.MeanTimesPrecision + 2 * ma * 8 + 3 * ma * 11;
            }
            Gaussian aActual = ie.Infer<Gaussian>(a);
            Gaussian bActual = ie.Infer<Gaussian>(b);
            Console.WriteLine("a = {0} (should be {1})", aActual, qa);
            Console.WriteLine("b = {0} (should be {1})", bActual, qb);
            Assert.True(qa.MaxDiff(aActual) < 1e-8);
            Assert.True(qb.MaxDiff(bActual) < 1e-8);

            /*  Vibes doesn't work correctly on this model
            VmpTests.TestGaussianMoments(ie, a, 0.710459128624146, 0.509858207250019);
            VmpTests.TestGaussianMoments(ie, b, 9.785273453279334, 95.846725317716462);
            VmpTests.TestGaussianMoments(ie, ab, 6.952036850965825, 48.868239541275962);
            VmpTests.TestEvidence(ie,evidence, -12.620225);
            */
        }

        [Fact]
        public void ProductObservedTwiceCopiedTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            //IfBlock block = Variable.If(evidence);
            Variable<double> a = Variable.GaussianFromMeanAndPrecision(1, 100).Named("a");
            Variable<double> b = Variable.GaussianFromMeanAndPrecision(10, 10).Named("b");
            Variable<double> ab = (a * b).Named("ab");
            Variable<double> ab2 = (a * b).Named("ab2");

            Variable.ConstrainEqual<double>(8, Variable.GaussianFromMeanAndPrecision(ab, 2).Named("abNoisy"));
            Variable.ConstrainEqual<double>(11, Variable.GaussianFromMeanAndPrecision(ab2, 3).Named("ab2Noisy"));
            //block.CloseBlock();

            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.NumberOfIterations = 60;

            // hand-coded VMP
            Gaussian pa = Gaussian.FromMeanAndPrecision(1, 100);
            Gaussian pb = Gaussian.FromMeanAndPrecision(10, 10);
            Gaussian qa = pa;
            Gaussian qb = pb;
            for (int iter = 0; iter < 60; iter++)
            {
                // q(a) = p(a) exp(sum_{b} q(b) (log N(ab; 8, 1/2) + log N(ab; 11, 1/3)))
                double mb, vb;
                qb.GetMeanAndVariance(out mb, out vb);
                qa.Precision = pa.Precision + 2 * (mb * mb + vb) + 3 * (mb * mb + vb);
                qa.MeanTimesPrecision = pa.MeanTimesPrecision + 2 * mb * 8 + 3 * mb * 11;
                // q(b) = p(b) exp(sum_{a} q(a) (log N(ab; 8, 1/2) + log N(ab; 11, 1/3)))
                double ma, va;
                qa.GetMeanAndVariance(out ma, out va);
                qb.Precision = pb.Precision + 2 * (ma * ma + va) + 3 * (ma * ma + va);
                qb.MeanTimesPrecision = pb.MeanTimesPrecision + 2 * ma * 8 + 3 * ma * 11;
            }
            Gaussian aActual = ie.Infer<Gaussian>(a);
            Gaussian bActual = ie.Infer<Gaussian>(b);
            Console.WriteLine("a = {0} (should be {1})", aActual, qa);
            Console.WriteLine("b = {0} (should be {1})", bActual, qb);
            Assert.True(qa.MaxDiff(aActual) < 1e-8);
            Assert.True(qb.MaxDiff(bActual) < 1e-8);

            /*  Vibes doesn't work correctly on this model
            VmpTests.TestGaussianMoments(ie, a, 0.710459128624146, 0.509858207250019);
            VmpTests.TestGaussianMoments(ie, b, 9.785273453279334, 95.846725317716462);
            VmpTests.TestGaussianMoments(ie, ab, 6.952036850965825, 48.868239541275962);
            VmpTests.TestEvidence(ie,evidence, -12.620225);
            */
        }

        [Fact]
        public void SimpleGaussianTest2()
        {
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100).Named("mean");
            Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1).Named("precision");
            Range i = new Range(2).Named("i");
            VariableArray<double> data = Variable.Array<double>(i).Named("data");
            data[i] = Variable.GaussianFromMeanAndPrecision(mean, precision).ForEach(i);
            data.ObservedValue = new double[] { 5.0, 7.0 };

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Gaussian meanExpected = Gaussian.FromMeanAndVariance(5.9603207170807826, 0.66132138200164436);
            Gamma precisionExpected = Gamma.FromShapeAndRate(2, 2.6628958274937107);
            Gaussian meanActual = engine.Infer<Gaussian>(mean);
            Gamma precisionActual = engine.Infer<Gamma>(precision);
            Console.WriteLine("mean = {0} should be {1}", meanActual, meanExpected);
            Console.WriteLine("precision = {0} should be {1}", precisionActual, precisionExpected);
            Assert.True(meanExpected.MaxDiff(meanActual) < 1e-10);
            Assert.True(precisionExpected.MaxDiff(precisionActual) < 1e-10);

            if (false)
            {
                engine.Compiler.GivePriorityTo(typeof(UsesEqualDefVmpOp));
                Console.WriteLine("mean = {0}", engine.Infer(mean));
                engine.Compiler.RemovePriority(typeof(UsesEqualDefVmpOp));
                Console.WriteLine("mean = {0}", engine.Infer(mean));
            }
        }

        // The following methods checks moments returned by Vibes with those obtained using Infer.NET

        internal static void TestVectorGaussianDiagonalMoments(InferenceEngine ie, Variable<Vector> Gvar, double[][] vibesResult)
        {
            VectorGaussian G = ie.Infer<VectorGaussian>(Gvar);
            Vector mean = Vector.Zero(G.Dimension);
            PositiveDefiniteMatrix variance = new PositiveDefiniteMatrix(mean.Count, mean.Count);
            G.GetMeanAndVariance(mean, variance);
            for (int i = 0; i < G.Dimension; i++)
            {
                double mean2 = mean[i] * mean[i] + variance[i, i];
                double error_mean = MMath.AbsDiff(mean[i], vibesResult[i][0], 1e-6);
                double error_mean2 = MMath.AbsDiff(mean2, vibesResult[i][1], 1e-6);
                Console.WriteLine("{0}'s moments: {1} (error {3}), {2} (error {4})", Gvar.Name + i, mean[i].ToString("g4"), variance[i, i].ToString("g4"),
                                  error_mean.ToString("g4"), error_mean2.ToString("g4"));
                Assert.True(error_mean < 1e-1);
                Assert.True(error_mean2 < 1e-1);
            }
        }

        internal static void TestGaussianArrayMoments(InferenceEngine ie, VariableArray<double> Gvar, double[][] vibesResult)
        {
            DistributionArray<Gaussian> G = ie.Infer<DistributionArray<Gaussian>>(Gvar);
            double G_mean, G_var, G_mean2;
            for (int i = 0; i < G.Count; i++)
            {
                G[i].GetMeanAndVariance(out G_mean, out G_var);
                G_mean2 = G_mean * G_mean + G_var;
                double error_mean = MMath.AbsDiff(G_mean, vibesResult[i][0], 1e-6);
                double error_mean2 = MMath.AbsDiff(G_mean2, vibesResult[i][1], 1e-6);
                Console.WriteLine("{0}'s moments: {1} (error {3}), {2} (error {4})", Gvar.Name + i, G_mean.ToString("g4"), G_var.ToString("g4"), error_mean.ToString("g4"),
                                  error_mean2.ToString("g4"));
                Assert.True(error_mean < 1e-1);
                Assert.True(error_mean2 < 1e-1);
            }
        }


        internal static void TestGaussianMoments(InferenceEngine ie, Variable<double> Gvar, double g_Ex, double g_Exx, double tolerance = 1e-4)
        {
            Gaussian G = ie.Infer<Gaussian>(Gvar);
            double G_mean, G_var, G_mean2;
            G.GetMeanAndVariance(out G_mean, out G_var);
            G_mean2 = G_mean * G_mean + G_var;
            double error_mean = MMath.AbsDiff(G_mean, g_Ex, 1e-6);
            double error_mean2 = MMath.AbsDiff(G_mean2, g_Exx, 1e-6);
            Console.WriteLine("{0}'s moments: {1} (error {3}), {2} (error {4})", Gvar.Name, G_mean.ToString("g4"), G_var.ToString("g4"), error_mean.ToString("g4"),
                              error_mean2.ToString("g4"));

            Assert.True(error_mean < tolerance);
            Assert.True(error_mean2 < tolerance);
        }

        internal static void TestGaussianMoments(InferenceEngine ie, Variable<double> Gvar, Gaussian expected)
        {
            double m, v;
            expected.GetMeanAndVariance(out m, out v);
            TestGaussianMoments(ie, Gvar, m, m * m + v);
        }

        internal static void TestGammaMoments(InferenceEngine ie, Variable<double> Gvar, double g_Ex, double g_Elogx)
        {
            Gamma G = ie.Infer<Gamma>(Gvar);

            double G_Ex, G_Elogx;
            G_Ex = G.GetMean();
            G_Elogx = G.GetMeanLog();
            double error_mean = MMath.AbsDiff(G_Ex, g_Ex, 1e-6);
            double error_meanLog = MMath.AbsDiff(G_Elogx, g_Elogx, 1e-6);
            Console.WriteLine("{0}'s moments: {1} (error {3}), {2} (error {4})", Gvar.Name, G_Ex.ToString("g4"), G_Elogx.ToString("g4"), error_mean.ToString("g4"),
                              error_meanLog.ToString("g4"));

            Assert.True(error_mean < 1e-2);
            Assert.True(error_meanLog < 1e-2);
        }

        internal static void TestDirichletMoments(InferenceEngine ie, Variable<Vector> Gvar, double[] g_Elogx)
        {
            Dirichlet G = ie.Infer<Dirichlet>(Gvar);
            Vector u = G.Point;
            //    Console.WriteLine("{0}'s posterior marginal: {1}", Gvar.Name, u);
            Vector eSuffStats = G.GetMeanLog();
            double maxError = 0;
            for (int i = 0; i < u.Count; i++)
            {
                double error = MMath.AbsDiff(eSuffStats[i], g_Elogx[i], 1e-6);
                Console.WriteLine("error = {0} esuff = {1} esuff vibes {2}", error, eSuffStats[i], g_Elogx[i]);
                Assert.True(error < 1e-2);
                maxError = System.Math.Max(maxError, error);
            }
            Console.WriteLine("{0}'s max error:{1} ", Gvar.Name, maxError.ToString("g4"));
        }

        internal static void TestEvidence(InferenceEngine ie, Variable<bool> evidence, double bound, double tolerance = 1e-4)
        {
            double e = ie.Infer<Bernoulli>(evidence).LogOdds;
            double error = MMath.AbsDiff(e, bound, 1e-6);
            Console.WriteLine("evidence = {0} (should be {1}) error = {2}", e.ToString("g4"), bound.ToString("g4"), error.ToString("g4"));
            Assert.True(error < tolerance);
        }

        internal static void TestDiscrete(InferenceEngine ie, VariableArray<int> Cvar, double[][] cVibesResult)
        {
            DistributionArray<Discrete> C = ie.Infer<DistributionArray<Discrete>>(Cvar);
            double error;
            for (int t = 0; t < C.Count; t++)
            {
                error = C[t].MaxDiff(new Discrete(cVibesResult[t]));
                Assert.True(error < 1e-5);
                Console.WriteLine("{0}'s[{1}] error = {2}", Cvar.Name, t, error);
            }
        }


        internal static void TestDirichletMoments(InferenceEngine ie, VariableArray<Vector> Gvar, double[][] g_Elogx)
        {
            Console.WriteLine("AK: Need to check this method:TestDirichletMoments. Used first with MixtureOfDiscretePartialObs() that doesn't work");
            DistributionArray<Dirichlet> G = ie.Infer<DistributionArray<Dirichlet>>(Gvar);
            for (int t = 0; t < G.Count; t++)
            {
                Vector u = G[t].Point;
                //    Console.WriteLine("{0}'s posterior marginal: {1}", Gvar.Name, u);
                Vector eSuffStats = G[t].GetMeanLog();
                double maxError = 0;
                for (int i = 0; i < u.Count; i++)
                {
                    double error = MMath.AbsDiff(eSuffStats[i], g_Elogx[t][i], 1e-6);
                    Console.WriteLine("error = {0} esuff = {1} esuff vibes {2}", error, eSuffStats[i], g_Elogx[t][i]);
                    Assert.True(error < 1e-2);
                    maxError = System.Math.Max(maxError, error);
                }
                Console.WriteLine("{0}'s max error:{1} ", Gvar.Name, maxError.ToString("g4"));
            }
        }


        internal static void TestDiscrete(InferenceEngine ie, Variable<int> Cvar, double[] cVibesResult)
        {
            Discrete C = ie.Infer<Discrete>(Cvar);
            // Console.WriteLine(C.GetProbs().ToString());
            double error = C.MaxDiff(new Discrete(cVibesResult));
            Console.WriteLine("{0}'s error = {1}", Cvar.Name, error);
            Assert.True(error < 1e-5);
        }
    }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif
}
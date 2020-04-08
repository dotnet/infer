// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Text;
using System.Collections.Generic;
using Xunit;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Tests
{
    using Microsoft.ML.Probabilistic.Algorithms;
    using System.Diagnostics;
    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
    using Range = Microsoft.ML.Probabilistic.Models.Range;

    /// <summary>
    /// Summary description for BugsTests
    /// </summary>
    public class BugsTests
    {
        private double[,] RatsHeightData = new double[,]
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

        private double[] RatsXData = {8.0, 15.0, 22.0, 29.0, 36.0};

        [Fact]
        public void BugsPumpModified()
        {
            //model
            //{
            //    for (i in 1 : N) {
            //        theta[i] ~ dgamma(alpha, beta)
            //        lambda[i] <- theta[i] * t[i]
            //        x[i] ~ dpois(lambda[i])
            //    }
            //    beta ~ dgamma(0.1, 1.0)
            //}

            // Data:
            //   list(t = c(94.3, 15.7, 62.9, 126, 5.24, 31.4, 1.05, 1.05, 2.1, 10.5),
            //        x = c(5, 1, 5, 14, 3, 19, 1, 1, 4, 22), N = 10, alpha=0.7)

            Rand.Restart(12347);
            int N = 10;
            double[] tData = new double[] {94.3, 15.7, 62.9, 126, 5.24, 31.4, 1.05, 1.05, 2.1, 10.5};
            int[] xData = new int[] {5, 1, 5, 14, 3, 19, 1, 1, 4, 22};
            // Exponential distribution is a special case of Gamma distribution (Shape =1)
            //var alpha = Variable.GammaFromShapeAndRate(1.0, 1.0);
            var beta = Variable.GammaFromShapeAndRate(0.1, 1.0);
            var alpha = 0.7;
            Range i = new Range(N);
            var theta = Variable.Array<double>(i);
            var lambda = Variable.Array<double>(i);
            var t = Variable.Array<double>(i);
            var x = Variable.Array<int>(i);
            theta[i] = Variable.GammaFromShapeAndRate(alpha, beta).ForEach(i);
            lambda[i] = theta[i]*t[i];
            x[i] = Variable.Poisson(lambda[i]);
            x.ObservedValue = xData;

            t.ObservedValue = tData;
            var engine = new InferenceEngine(new GibbsSampling());
            engine.NumberOfIterations = 100000;
            engine.ShowProgress = false;
            engine.ShowTimings = true;

            // Expected values (from WinBugs)
            double[] expectedThetaPostMean = {0.06003, 0.1017, 0.08966, 0.1158, 0.5998, 0.6082, 0.8838, 0.8935, 1.565, 1.987};
            double[] expectedThetaPostSD = {0.02509, 0.07878, 0.03766, 0.03019, 0.3166, 0.1401, 0.7187, 0.7132, 0.7583, 0.4219};
            IDistribution<double[]> thetaExpected =
                Distribution<double>.Array(Util.ArrayInit(N, k => Gamma.FromMeanAndVariance(expectedThetaPostMean[k], expectedThetaPostSD[k]*expectedThetaPostSD[k])));

            var thetaPost = engine.Infer(theta);
            Console.WriteLine(StringUtil.JoinColumns("theta = ", thetaPost, " should be ", thetaExpected));
            Assert.True(thetaExpected.MaxDiff(thetaPost) < 1.3);
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void BugsSeeds()
        {
            Assert.True(false, "This Bugs example test will always fail with Gibbs - we need hybrid Metropolis/Gibbs");

            //model
            //{
            //    for( i in 1 : N ) {
            //        r[i] ~ dbin(p[i],n[i])
            //        b[i] ~ dnorm(0.0,tau)
            //        logit(p[i]) <- alpha0 + alpha1 * x1[i] + alpha2 * x2[i] + 
            //            alpha12 * x1[i] * x2[i] + b[i]
            //    }
            //    alpha0 ~ dnorm(0.0,1.0E-6)
            //    alpha1 ~ dnorm(0.0,1.0E-6)
            //    alpha2 ~ dnorm(0.0,1.0E-6)
            //    alpha12 ~ dnorm(0.0,1.0E-6)
            //    tau ~ dgamma(0.001,0.001)
            //    sigma <- 1 / sqrt(tau)
            //}
            //list(r = c(10, 23, 23, 26, 17, 5, 53, 55, 32, 46, 10,   8, 10,   8, 23, 0,  3, 22, 15, 32, 3),
            //     n = c(39, 62, 81, 51, 39, 6, 74, 72, 51, 79, 13, 16, 30, 28, 45, 4, 12, 41, 30, 51, 7),
            //     x1 = c(0,   0,  0,   0,   0, 0,   0,   0,  0,   0,   0,  1,   1,   1,   1, 1,   1,  1,   1,   1, 1),
            //     x2 = c(0,   0,  0,   0,   0, 1,   1,   1,  1,   1,   1,  0,   0,   0,   0, 0,   1,  1,   1,   1, 1),
            //     N = 21)
            //Inits list(alpha0 = 0, alpha1 = 0, alpha2 = 0, alpha12 = 0, tau = 1)

            Rand.Restart(12347);
            int[] rData = {10, 23, 23, 26, 17, 5, 53, 55, 32, 46, 10, 8, 10, 8, 23, 0, 3, 22, 15, 32, 3};
            int[] nData = {39, 62, 81, 51, 39, 6, 74, 72, 51, 79, 13, 16, 30, 28, 45, 4, 12, 41, 30, 51, 7};
            double[] x1Data = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
            double[] x2Data = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
            int N = rData.Length;

            var alpha0 = Variable.GaussianFromMeanAndPrecision(0.0, 1.0E-6);
            var alpha1 = Variable.GaussianFromMeanAndPrecision(0.0, 1.0E-6);
            var alpha2 = Variable.GaussianFromMeanAndPrecision(0.0, 1.0E-6);
            var alpha12 = Variable.GaussianFromMeanAndPrecision(0.0, 1.0E-6);
            var tau = Variable.GammaFromShapeAndRate(0.001, 0.001);
            Range i = new Range(N);
            var s = Variable.Array<double>(i);
            var p = Variable.Array<double>(i);
            var b = Variable.Array<double>(i);
            var x1 = Variable.Array<double>(i);
            var x2 = Variable.Array<double>(i);
            var n = Variable.Array<int>(i);
            var r = Variable.Array<int>(i);
            b[i] = Variable.GaussianFromMeanAndPrecision(0.0, tau).ForEach(i);
            s[i] = alpha0 + alpha1*x1[i] + alpha2*x2[i] + alpha12*x1[i]*x2[i] + b[i];
            p[i] = Variable.Logistic(s[i]);
            r[i] = Variable.Binomial(n[i], p[i]);

            r.ObservedValue = rData;
            n.ObservedValue = nData;
            x1.ObservedValue = x1Data;
            x2.ObservedValue = x2Data;

            var engine = new InferenceEngine(new GibbsSampling());
            engine.ShowProgress = false;
            engine.ShowTimings = true;
            var alpha0Post = engine.Infer(alpha0);
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void BugsAir()
        {
            Assert.True(false, "This Bugs example test will always fail with Gibbs - we need hybrid Metropolis/Gibbs");

            //model
            //{
            //  for(j in 1 : J) {
            //     y[j] ~ dbin(p[j], n[j])
            //     logit(p[j]) <- theta[1] + theta[2] * X[j]
            //     X[j] ~ dnorm(mu[j], tau)
            //     mu[j] <- alpha + beta * Z[j]
            //  }
            //  theta[1] ~ dnorm(0.0, 0.001)
            //  theta[2] ~ dnorm(0.0, 0.001)
            //}

            Rand.Restart(12347);
            double alpha = 4.48;
            double beta = 0.76;
            double tau = 0.01234;
            Range j = new Range(3).Named("j");
            var theta1 = Variable.GaussianFromMeanAndPrecision(0.0, 0.001).Named("theta1");
            var theta2 = Variable.GaussianFromMeanAndPrecision(0.0, 0.001).Named("theta2");
            var X = Variable.Array<double>(j).Named("X");
            var Z = Variable.Array<double>(j).Named("Z");
            var mu = Variable.Array<double>(j).Named("mu");
            var p = Variable.Array<double>(j).Named("p");
            var y = Variable.Array<int>(j).Named("y");
            var n = Variable.Array<int>(j).Named("n");

            mu[j] = alpha + beta*Z[j];
            X[j] = Variable.GaussianFromMeanAndPrecision(mu[j], tau);
            p[j] = Variable.Logistic(theta1 + theta2*X[j]);
            y[j] = Variable.Binomial(n[j], p[j]);

            Z.ObservedValue = new double[] {10.0, 30.0, 50.0};
            n.ObservedValue = new int[] {48, 34, 21};
            y.ObservedValue = new int[] {21, 20, 15};

            var engine = new InferenceEngine(new GibbsSampling());
            engine.ShowProgress = false;
            engine.ShowTimings = true;

            var theta1Post = engine.Infer<Gaussian>(theta1);
            var theta2Post = engine.Infer<Gaussian>(theta2);
            var XPost = engine.Infer<Gaussian[]>(X);
        }

        [Fact]
        public void BugsSurgicalFixedEffects()
        {
            //model
            //{
            //   for( i in 1 : N ) {
            //      p[i] ~ dbeta(1.0, 1.0)
            //    r[i] ~ dbin(p[i], n[i])
            //   }
            //}
            // list(n = c(47, 148, 119, 810, 211, 196, 148, 215, 207, 97, 256, 360),
            //      r  = c(0, 18, 8, 46, 8, 13, 9, 31, 14, 8, 29, 24),
            //      N = 12)

            Rand.Restart(12347);
            int[] nData = new int[] {47, 148, 119, 810, 211, 196, 148, 215, 207, 97, 256, 360};
            int[] rData = new int[] {0, 18, 8, 46, 8, 13, 9, 31, 14, 8, 29, 24};

            int N = nData.Length;
            Range i = new Range(N);
            var p = Variable.Array<double>(i);
            var n = Variable.Array<int>(i);
            var r = Variable.Array<int>(i);
            p[i] = Variable.Beta(1.0, 1.0).ForEach(i);
            r[i] = Variable.Binomial(n[i], p[i]);
            n.ObservedValue = nData;
            r.ObservedValue = rData;
            var engine = new InferenceEngine(new GibbsSampling());
            engine.ShowProgress = false;
            engine.ShowTimings = true;

            // Expected values (from WinBugs)
            double[] expected_pPostMean = {0.02001, 0.1266, 0.0744, 0.05788, 0.04241, 0.07084, 0.06681, 0.1474, 0.07218, 0.09089, 0.1164, 0.06904};
            double[] expected_pPostSD = {0.01945, 0.02731, 0.02377, 0.008246, 0.01397, 0.01813, 0.02021, 0.024, 0.01789, 0.02894, 0.02013, 0.01346};
            var pPost = engine.Infer<Beta[]>(p);
            for (int k = 0; k < pPost.Length; k++)
            {
                Assert.Equal(expected_pPostMean[k], pPost[k].GetMean(), 0.02);
                Assert.Equal(expected_pPostSD[k], System.Math.Sqrt(pPost[k].GetVariance()), 0.02);
            }
        }

        [Fact]
        public void BugsDyes()
        {
            // Dyes: variance components model
            // http://users.aims.ac.za/~mackay/BUGS/Examples/Dyes.html
            //model
            //{
            //    for( i in 1 : batches ) {
            //        m[i] ~ dnorm(theta, tau.btw)
            //        for( j in 1 : samples ) {
            //            y[i , j] ~ dnorm(m[i], tau.with)
            //        }
            //    }
            //    sigma2.with <- 1 / tau.with
            //    sigma2.btw <- 1 / tau.btw
            //    tau.with ~ dgamma(0.001, 0.001)
            //    tau.btw ~ dgamma(0.001, 0.001)
            //    theta ~ dnorm(0.0, 1.0E-10)
            //}
            //list(batches = 6, samples = 5,
            //     y = structure(
            //      .Data = c(1545, 1440, 1440, 1520, 1580,
            //                1540, 1555, 1490, 1560, 1495,
            //                1595, 1550, 1605, 1510, 1560,
            //                1445, 1440, 1595, 1465, 1545,
            //                1595, 1630, 1515, 1635, 1625,
            //                1520, 1455, 1450, 1480, 1445), .Dim = c(6, 5)))

            Rand.Restart(12347);
            double[,] yData = new double[,]
                {
                    {1545, 1440, 1440, 1520, 1580},
                    {1540, 1555, 1490, 1560, 1495},
                    {1595, 1550, 1605, 1510, 1560},
                    {1445, 1440, 1595, 1465, 1545},
                    {1595, 1630, 1515, 1635, 1625},
                    {1520, 1455, 1450, 1480, 1445}
                };

            var theta = Variable.GaussianFromMeanAndPrecision(0.0, 1.0E-10).Named("theta");
            var tauWithin = Variable.GammaFromShapeAndRate(0.001, 0.001).Named("tauWithin");
            var tauBetween = Variable.GammaFromShapeAndRate(0.001, 0.001).Named("tauBetween");
            Range i = new Range(yData.GetLength(0)).Named("i");
            Range j = new Range(yData.GetLength(1)).Named("j");
            var m = Variable.Array<double>(i).Named("m");
            var y = Variable.Array<double>(i, j).Named("y");
            using (Variable.ForEach(i))
            {
                m[i] = Variable.GaussianFromMeanAndPrecision(theta, tauBetween);
                using (Variable.ForEach(j))
                    y[i, j] = Variable.GaussianFromMeanAndPrecision(m[i], tauWithin);
            }
            y.ObservedValue = yData;
            var engine = new InferenceEngine(new GibbsSampling());
            engine.NumberOfIterations = 50000;
            engine.ShowProgress = false;

            tauWithin.InitialiseTo(Gamma.FromShapeAndRate(1.0, 1.0));
            //tauBetween.InitialiseTo(Gamma.FromShapeAndRate(1.0, 1.0));
            theta.InitialiseTo(Gaussian.FromMeanAndPrecision(1500, 10.0));

            var thetaPost = engine.Infer<Gaussian>(theta);
            // We want the mean and variance of (varianceWithin, varianceBetween).  These can only be obtained from the raw samples.
            IList<double> tauWithinSamples = engine.Infer<IList<double>>(tauWithin, QueryTypes.Samples);
            IList<double> tauBetweenSamples = engine.Infer<IList<double>>(tauBetween, QueryTypes.Samples);
            GaussianEstimator varianceWithinEst = new GaussianEstimator();
            foreach (double t in tauWithinSamples)
            {
                varianceWithinEst.Add(1/t);
            }
            Gaussian varianceWithinPost = varianceWithinEst.GetDistribution(new Gaussian());
            GaussianEstimator varianceBetweenEst = new GaussianEstimator();
            foreach (double t in tauBetweenSamples)
            {
                varianceBetweenEst.Add(1/t);
            }
            Gaussian varianceBetweenPost = varianceBetweenEst.GetDistribution(new Gaussian());

            Gaussian thetaExpected = Gaussian.FromMeanAndVariance(1528, 478.5);
            Gaussian varianceWithinExpected = Gaussian.FromMeanAndVariance(3010, 1.203e+06);
            Gaussian varianceBetweenExpected = Gaussian.FromMeanAndVariance(2272, 2.069e+07);

            Console.WriteLine("theta = {0} should be {1}", thetaPost, thetaExpected);
            Console.WriteLine("varianceWithin = {0} should be {1}", varianceWithinPost, varianceWithinExpected);
            Console.WriteLine("varianceBetween = {0} should be {1}", varianceBetweenPost, varianceBetweenExpected);
            Assert.True(thetaExpected.MaxDiff(thetaPost) < 1e-1);
            Assert.True(varianceWithinExpected.MaxDiff(varianceWithinPost) < 2e-4);
            Assert.True(varianceBetweenExpected.MaxDiff(varianceBetweenPost) < 2e-4);
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void BugsJaws()
        {
            Assert.True(false, "This Bugs example is incomplete -- TODO");

            Vector[] jawData =
                {
                    Vector.FromArray(47.8, 48.8, 49.0, 49.7),
                    Vector.FromArray(46.4, 47.3, 47.7, 48.4),
                    Vector.FromArray(46.3, 46.8, 47.8, 48.5),
                    Vector.FromArray(45.1, 45.3, 46.1, 47.2),
                    Vector.FromArray(47.6, 48.5, 48.9, 49.3),
                    Vector.FromArray(52.5, 53.2, 53.3, 53.7),
                    Vector.FromArray(51.2, 53.0, 54.3, 54.5),
                    Vector.FromArray(49.8, 50.0, 50.3, 52.7),
                    Vector.FromArray(48.1, 50.8, 52.3, 54.4),
                    Vector.FromArray(45.0, 47.0, 47.3, 48.3),
                    Vector.FromArray(51.2, 51.4, 51.6, 51.9),
                    Vector.FromArray(48.5, 49.2, 53.0, 55.5),
                    Vector.FromArray(52.1, 52.8, 53.7, 55.0),
                    Vector.FromArray(48.2, 48.9, 49.3, 49.8),
                    Vector.FromArray(49.6, 50.4, 51.2, 51.8),
                    Vector.FromArray(50.7, 51.7, 52.7, 53.3),
                    Vector.FromArray(47.2, 47.7, 48.4, 49.5),
                    Vector.FromArray(53.3, 54.6, 55.1, 55.3),
                    Vector.FromArray(46.2, 47.5, 48.1, 48.4),
                    Vector.FromArray(46.3, 47.6, 51.3, 51.8)
                };
            Vector ageData = Vector.FromArray(8.0, 8.5, 9.0, 9.5);

            // Rate and therefore scale are both the identity matrix
            PositiveDefiniteMatrix Scale = PositiveDefiniteMatrix.Identity(4);
            // Shape parameter k in Bugs is set to 4 - this corresponds to
            // a value of k/2 = 4 for the Infer.NET shape parameter
            double shape = 2.0;
            Range d = new Range(jawData.Length);
            VariableArray<Vector> data = Variable.Observed<Vector>(jawData, d).Named("data");
            Variable<Vector> age = Variable.Observed<Vector>(ageData);
            Variable<double> beta0 = Variable.GaussianFromMeanAndPrecision(0.0, 0.001).Named("beta0");
            Variable<double> beta1 = Variable.GaussianFromMeanAndPrecision(0.0, 0.001).Named("beta1");
            Variable<PositiveDefiniteMatrix> omega = Variable.WishartFromShapeAndScale(shape, Scale);
            //Variable<Vector> mean = age; // beta0 + beta1 * age;
            //data[d] = Variable.VectorGaussianFromMeanAndPrecision(mean, omega).ForEach(d);
        }

        // TODO: check that the first sample matches the initialization.
        [Fact]
        public void BugsRatsWithInitialisation()
        {
            for (int i1 = 0; i1 < 2; i1++)
            {
                bool initialiseAlpha = (i1 == 1);
                for (int i2 = 0; i2 < 2; i2++)
                {
                    bool initialiseAlphaC = (i2 == 1);
                    if (i1 == 0 && i2 == 0) continue;
                    BugsRats(initialiseAlpha, initialiseAlphaC);
                }
            }
        }

        [Trait("Category", "OpenBug")]
        [Fact]
        public void BugsRatsWithoutInitialisation()
        {
            BugsRats(false, false);
        }

        private void BugsRats(bool initialiseAlpha, bool initialiseAlphaC)
        {
            Rand.Restart(0);
            double precOfGaussianPrior = 1.0E-6;
            double shapeRateOfGammaPrior = 0.02; // smallest choice that will avoid zeros

            double meanOfBetaPrior = 0.0;
            double meanOfAlphaPrior = 0.0;

            // The model
            int N = RatsHeightData.GetLength(0);
            int T = RatsHeightData.GetLength(1);
            double xbar = 22.0;
            double[] xDataZeroMean = new double[RatsXData.Length];
            for (int i = 0; i < RatsXData.Length; i++) xDataZeroMean[i] = RatsXData[i] - xbar;
            Range r = new Range(N).Named("N");
            Range w = new Range(T).Named("T");
            VariableArray2D<double> y = Variable.Observed<double>(RatsHeightData, r, w).Named("y");
            VariableArray<double> x = Variable.Observed<double>(xDataZeroMean, w).Named("x");
            Variable<double> tauC = Variable.GammaFromShapeAndRate(shapeRateOfGammaPrior, shapeRateOfGammaPrior).Named("tauC");
            Variable<double> alphaC = Variable.GaussianFromMeanAndPrecision(meanOfAlphaPrior, precOfGaussianPrior).Named("alphaC");
            Variable<double> alphaTau = Variable.GammaFromShapeAndRate(shapeRateOfGammaPrior, shapeRateOfGammaPrior).Named("alphaTau");
            Variable<double> betaC = Variable.GaussianFromMeanAndPrecision(meanOfBetaPrior, precOfGaussianPrior).Named("betaC");
            Variable<double> betaTau = Variable.GammaFromShapeAndRate(shapeRateOfGammaPrior, shapeRateOfGammaPrior).Named("betaTau");
            VariableArray<double> alpha = Variable.Array<double>(r).Named("alpha");
            alpha[r] = Variable.GaussianFromMeanAndPrecision(alphaC, alphaTau).ForEach(r);
            VariableArray<double> beta = Variable.Array<double>(r).Named("beta");
            beta[r] = Variable.GaussianFromMeanAndPrecision(betaC, betaTau).ForEach(r);
            VariableArray2D<double> mu = Variable.Array<double>(r, w).Named("mu");
            VariableArray2D<double> betaX = Variable.Array<double>(r, w).Named("betax");
            betaX[r, w] = beta[r]*x[w];
            mu[r, w] = alpha[r] + betaX[r, w];
            y[r, w] = Variable.GaussianFromMeanAndPrecision(mu[r, w], tauC);
            Variable<double> alpha0 = (alphaC - xbar*betaC).Named("alpha0");

            InferenceEngine ie;
            GibbsSampling gs = new GibbsSampling();
            // Initialise both alpha and beta together.
            // Initialising only alpha (or only beta) is not reliable because you could by chance get a large betaTau and small tauC to start, 
            // at which point beta and alphaC become garbage, leading to alpha becoming garbage on the next iteration.
            bool initialiseBeta = initialiseAlpha;
            bool initialiseBetaC = initialiseAlphaC;
            if (initialiseAlpha)
            {
                Gaussian[] alphaInit = new Gaussian[N];
                for (int i = 0; i < N; i++) alphaInit[i] = Gaussian.FromMeanAndPrecision(250.0, 1.0);
                alpha.InitialiseTo(Distribution<double>.Array(alphaInit));
            }
            if (initialiseBeta)
            {
                Gaussian[] betaInit = new Gaussian[N];
                for (int i = 0; i < N; i++) betaInit[i] = Gaussian.FromMeanAndPrecision(6.0, 1.0);
                beta.InitialiseTo(Distribution<double>.Array(betaInit));
            }
            if (initialiseAlphaC)
            {
                alphaC.InitialiseTo(Gaussian.FromMeanAndVariance(250.0, 1.0));
            }
            if (initialiseBetaC)
            {
                betaC.InitialiseTo(Gaussian.FromMeanAndVariance(6.0, 1.0));
            }
            if (false)
            {
                //tauC.InitialiseTo(Gamma.FromMeanAndVariance(1.0, 0.1));
                //alphaTau.InitialiseTo(Gamma.FromMeanAndVariance(1.0, 0.1));
                //betaTau.InitialiseTo(Gamma.FromMeanAndVariance(1.0, 0.1));
            }
            if (!initialiseAlpha && !initialiseBeta && !initialiseAlphaC && !initialiseBetaC)
            {
                gs.BurnIn = 1000;
            }
            ie = new InferenceEngine(gs);
            ie.ShowProgress = false;
            ie.ModelName = "BugsRats";
            ie.NumberOfIterations = 4000;
            ie.OptimiseForVariables = new List<IVariable>() {alphaC, betaC, alpha0, tauC};
            betaC.AddAttribute(QueryTypes.Marginal);
            betaC.AddAttribute(QueryTypes.Samples);
            alpha0.AddAttribute(QueryTypes.Marginal);
            alpha0.AddAttribute(QueryTypes.Samples);
            tauC.AddAttribute(QueryTypes.Marginal);
            tauC.AddAttribute(QueryTypes.Samples);

            // Inference
            object alphaCActual = ie.Infer(alphaC);
            Gaussian betaCMarg = ie.Infer<Gaussian>(betaC);
            Gaussian alpha0Marg = ie.Infer<Gaussian>(alpha0);
            Gamma tauCMarg = ie.Infer<Gamma>(tauC);

            // Check results against BUGS
            Gaussian betaCExpected = new Gaussian(6.185, System.Math.Pow(0.1068, 2));
            Gaussian alpha0Expected = new Gaussian(106.6, System.Math.Pow(3.625, 2));
            double sigmaMeanExpected = 6.082;
            double sigmaMean = System.Math.Sqrt(1.0/ tauCMarg.GetMean());

            if (!initialiseAlpha && !initialiseAlphaC)
            {
                Debug.WriteLine("betaC = {0} should be {1}", betaCMarg, betaCExpected);
                Debug.WriteLine("alpha0 = {0} should be {1}", alpha0Marg, alpha0Expected);
            }
            Assert.True(GaussianDiff(betaCExpected, betaCMarg) < 0.1);
            Assert.True(GaussianDiff(alpha0Expected, alpha0Marg) < 0.1);
            Assert.True(MMath.AbsDiff(sigmaMeanExpected, sigmaMean, 0.1) < 0.1);

            IList<double> betaCSamples = ie.Infer<IList<double>>(betaC, QueryTypes.Samples);
            IList<double> alpha0Samples = ie.Infer<IList<double>>(alpha0, QueryTypes.Samples);
            IList<double> tauCSamples = ie.Infer<IList<double>>(tauC, QueryTypes.Samples);

            GaussianEstimator est = new GaussianEstimator();
            foreach (double sample in betaCSamples)
            {
                est.Add(sample);
            }
            Gaussian betaCMarg2 = est.GetDistribution(new Gaussian());
            Assert.True(GaussianDiff(betaCMarg, betaCMarg2) < 0.1);
        }

        public static double GaussianDiff(Gaussian a, Gaussian b)
        {
            return System.Math.Max(MMath.AbsDiff(a.GetMean(), b.GetMean(), 1e-4),
                            MMath.AbsDiff(System.Math.Sqrt(a.GetVariance()), System.Math.Sqrt(b.GetVariance()), 1e-4));
        }
    }
}
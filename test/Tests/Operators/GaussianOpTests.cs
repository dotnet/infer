// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Xunit;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Tests
{
    using Assert = Xunit.Assert;
    using System.Diagnostics;


    public class GaussianOpTests
    {
        // There are 27 cases to test, for each method of the operator.
        [Fact]
        public void GaussianOpLogAverageFactor()
        {
            Gaussian uniform = new Gaussian();
            Gaussian X0 = Gaussian.FromMeanAndVariance(3, 0.5);
            Gaussian Mean0 = Gaussian.FromMeanAndVariance(7, 1.0 / 3);
            Gamma Precision0 = Gamma.FromShapeAndScale(3, 3);

            // Fixed precision
            Gamma Precision = Gamma.PointMass(3);
            Gaussian X = X0;
            Gaussian Mean = uniform;
            Assert.True(MMath.AbsDiff(GaussianOp.LogAverageFactor_slow(X, Mean, Precision), 0, 1e-4) < 1e-4);
            Assert.True(MMath.AbsDiff(GaussianOp_Slow.LogAverageFactor(X, Mean, Precision), 0, 1e-4) < 1e-4);
            Assert.True(MMath.AbsDiff(GaussianOp.LogAverageFactor(X, Mean, Precision.Point), 0, 1e-4) < 1e-4);
            Mean = Mean0;
            // in matlab: normpdfln(3,7,[],0.5+1/3+1/3)
            Assert.True(MMath.AbsDiff(GaussianOp.LogAverageFactor_slow(X, Mean, Precision), -7.8532, 1e-4) < 1e-4);
            Assert.True(MMath.AbsDiff(GaussianOp_Slow.LogAverageFactor(X, Mean, Precision), -7.8532, 1e-4) < 1e-4);
            Assert.True(MMath.AbsDiff(GaussianOp.LogAverageFactor(X, Mean, Precision.Point), -7.8532, 1e-4) < 1e-4);
            Mean = Gaussian.PointMass(Mean0.GetMean());
            // in matlab: normpdfln(3,7,[],0.5+1/3)
            Assert.True(MMath.AbsDiff(GaussianOp.LogAverageFactor_slow(X, Mean, Precision), -10.42777775, 1e-4) < 1e-4);
            Assert.True(MMath.AbsDiff(GaussianOp_Slow.LogAverageFactor(X, Mean, Precision), -10.42777775, 1e-4) < 1e-4);
            Assert.True(MMath.AbsDiff(GaussianOp.LogAverageFactor(X, Mean, Precision.Point), -10.42777775, 1e-4) < 1e-4);
            X = uniform;
            Assert.True(MMath.AbsDiff(GaussianOp.LogAverageFactor_slow(X, Mean, Precision), 0, 1e-4) < 1e-4);
            Assert.True(MMath.AbsDiff(GaussianOp_Slow.LogAverageFactor(X, Mean, Precision), 0, 1e-4) < 1e-4);
            Assert.True(MMath.AbsDiff(GaussianOp.LogAverageFactor(X, Mean, Precision.Point), 0, 1e-4) < 1e-4);

            // Unknown precision
            Precision = Precision0;
            X = X0;
            Mean = Mean0;
            // converge the precision message.  (only matters if KeepLastMessage is set).
            //for (int i = 0; i < 10; i++) PrecisionAverageConditional(precisionMessage);
            // in matlab: log(t_normal_exact(mx-my,vx+vy,a+1,b))
            //            log(t_normal_exact(3-7,0.5+1/3,3,1/3))
            Assert.True(MMath.AbsDiff(GaussianOp.LogAverageFactor_slow(X, Mean, Precision), -8.4363, 1e-4) < 1e-4);
            Assert.True(MMath.AbsDiff(GaussianOp_Slow.LogAverageFactor(X, Mean, Precision), -8.4363, 1e-4) < 1e-4);
            Mean = Gaussian.PointMass(Mean0.GetMean());
            // converge the precision message.  (only matters if KeepLastMessage is set).
            //for (int i = 0; i < 10; i++) PrecisionAverageConditional(precisionMessage);
            // in matlab: log(t_normal_exact(3-7,0.5,3,1/3))
            Assert.True(MMath.AbsDiff(GaussianOp.LogAverageFactor_slow(X, Mean, Precision), -9.9890, 1e-4) < 1e-4);
            X = Gaussian.PointMass(X0.GetMean());
            Mean = Mean0;
            // converge the precision message.  (only matters if KeepLastMessage is set).
            //for (int i = 0; i < 10; i++) PrecisionAverageConditional(precisionMessage);
            // in matlab: log(t_normal_exact(3-7,1/3,3,1/3))
            Assert.True(MMath.AbsDiff(GaussianOp.LogAverageFactor_slow(X, Mean, Precision), -10.478382, 1e-4) < 1e-4);
            X = Gaussian.PointMass(X0.GetMean());
            Mean = Gaussian.PointMass(Mean0.GetMean());
            // in matlab: log(t_normal_exact(3-7,1e-4,3,1/3)) or tpdfln(3-7,0,2*1/3,2*3+1)
            Assert.True(MMath.AbsDiff(GaussianOp.LogAverageFactor_slow(X, Mean, Precision), -11.1278713, 1e-4) < 1e-4);
            X = uniform;
            Assert.True(MMath.AbsDiff(GaussianOp.LogAverageFactor_slow(X, Mean, Precision), 0, 1e-4) < 1e-4);

            // uniform precision
            // the answer should always be Double.PositiveInfinity
            Precision = Gamma.Uniform();
            X = X0;
            Mean = Mean0;
            Assert.True(MMath.AbsDiff(GaussianOp.LogAverageFactor_slow(X, Mean, Precision), Double.PositiveInfinity, 1e-4) < 1e-4);

            Assert.True(MMath.AbsDiff(GaussianOp_Slow.LogAverageFactor(new Gaussian(-0.641, 9.617e-22), Gaussian.PointMass(-1), new Gamma(1, 1)), -1.133394734344457, 1e-8) <
                          1e-4);
            GaussianOp_Slow.LogAverageFactor(new Gaussian(8.156, 9.653), Gaussian.PointMass(-1), new Gamma(1, 1));
        }

        // Test that the operator behaves correctly for arguments with small variance
        [Fact]
        [Trait("Category", "ModifiesGlobals")]
        public void GaussianOpX2Test()
        {
            using (TestUtils.TemporarilyAllowGaussianImproperMessages)
            {
                Gamma precision = Gamma.FromShapeAndScale(3, 3);
                Gaussian mean;
                mean = Gaussian.PointMass(7);
                GaussianOpX2(mean, precision);
                mean = Gaussian.FromMeanAndVariance(7, 1.0 / 3);
                GaussianOpX2(mean, precision);
            }
        }

        private void GaussianOpX2(Gaussian mean, Gamma precision)
        {
            Gaussian sample, result, result2;
            sample = Gaussian.PointMass(2);
            result = GaussianOp.SampleAverageConditional_slow(sample, mean, precision);
            result2 = GaussianOp_Slow.SampleAverageConditional(sample, mean, precision);
            Console.WriteLine("{0}: {1} {2}", sample, result, result2);
            Assert.True(result.MaxDiff(result2) < 1e-8);
            double prevDiff = double.PositiveInfinity;
            for (int i = 8; i < 30; i++)
            {
                double v = System.Math.Pow(0.1, i);
                sample = Gaussian.FromMeanAndVariance(2, v);
                result2 = GaussianOp_Slow.SampleAverageConditional(sample, mean, precision);
                double diff = result.MaxDiff(result2);
                Console.WriteLine("{0}: {1} diff={2}", sample, result2, diff.ToString("g4"));
                Assert.True(diff <= prevDiff || diff < 1e-6);
                result2 = GaussianOp.SampleAverageConditional_slow(sample, mean, precision);
                diff = result.MaxDiff(result2);
                Console.WriteLine("{0}: {1} diff={2}", sample, result2, diff.ToString("g4"));
                Assert.True(diff <= prevDiff || diff < 1e-6);
                prevDiff = diff;
            }
        }

        internal void GaussianOpIntegrationBoundsTest()
        {
            double logrmin, logrmax, logr0;
            double m, v, a, b;
            for (int option = 0; option < 6; option++)
            {
                if (option == 0)
                {
                    m = -58.3;
                    v = 3091;
                    a = 0.001;
                    b = 0.001;
                }
                else if (option == 1)
                {
                    m = -8.3208;
                    v = 11.43;
                    a = 0.63592;
                    b = 0.72846;
                }
                else if (option == 2)
                {
                    m = -4.42275;
                    v = 13.8193;
                    a = 0.0398852;
                    b = 0.260568;
                }
                else if (option == 3)
                {
                    m = -714.691;
                    v = 231.803;
                    a = 13.5296;
                    b = 4.5111;
                }
                else if (option == 4)
                {
                    m = 230.044;
                    v = 3.62396e+007;
                    a = 0.00254166;
                    b = 2.18533;
                }
                else
                {
                    m = -5.23379;
                    v = 9.18905;
                    a = 0.202543;
                    b = 0.147646;
                }
                GaussianOp_Slow.GetIntegrationBoundsForPrecision(m, v, a, b, out logrmin, out logrmax, out logr0);
                Console.WriteLine("rmin = {0} rmax = {1} rmode = {2}", System.Math.Exp(logrmin), System.Math.Exp(logrmax), System.Math.Exp(logr0));
            }
        }

        [Fact]
        public void GetRootsTest()
        {
            double[] coeffs = { -95.172, 66.429, -15.094, 1.1359 };
            //coeffs = new double[] { -1000, 0, -3000, 0, -3000, 0, -1000 };
            double[] rootsReal, rootsImag;
            GaussianOp_Slow.GetRoots(coeffs, out rootsReal, out rootsImag);
            Console.WriteLine("rootsReal = {0}", StringUtil.CollectionToString(rootsReal, " "));
            Console.WriteLine("rootsImag = {0}", StringUtil.CollectionToString(rootsImag, " "));
            double[] rootsRealExpected = new double[] { 0.313719909680864, 0.19213449730936, 0.19213449730936 };
            double[] rootsImagExpected = new double[] { 0, -0.0335941516159427, 0.0335941516159427 };
            for (int i = 0; i < rootsRealExpected.Length; i++)
            {
                Assert.Equal(rootsRealExpected[i], rootsReal[i], 10);
                Assert.Equal(rootsImagExpected[i], rootsImag[i], 10);
            }
        }

        internal void GaussianOpQ_Timing()
        {
            Gaussian X, Mean;
            Gamma Precision;
            Gamma q;
            int n = 1;
            X = Gaussian.FromNatural(3.9112579392580757, 11.631097473681082);
            Mean = Gaussian.FromNatural(10.449696977834144, 5.5617978202886995);
            Precision = Gamma.FromShapeAndRate(1.0112702817305146, 0.026480506235719053);
            q = Gamma.FromMeanAndVariance(1, 1);
            Stopwatch watch = new Stopwatch();
            watch.Start();
            for (int i = 0; i < n; i++)
            {
                GaussianOp_Laplace.Q(X, Mean, Precision, q);
            }
            watch.Stop();
            Console.WriteLine("Q = {0}", watch.ElapsedTicks);
            watch.Restart();
            for (int i = 0; i < n; i++)
            {
                GaussianOp_Laplace.Q_Slow(X, Mean, Precision);
            }
            watch.Stop();
            Console.WriteLine("Q2 = {0}", watch.ElapsedTicks);
        }

        [Fact]
        public void GaussianOp_Laplace_Q_Test()
        {
            Gaussian sample = Gaussian.FromNatural(5.3861033232682936E-79, 2.901010900892175E-157);
            Gaussian mean = Gaussian.FromNatural(-2.7232954231713977, 0.074308384738968308);
            Gamma precision = Gamma.FromShapeAndRate(656.04827139518625, 1.4379651227587877E+159);
            Gamma q = Gamma.FromShapeAndRate(1.1680712992725464, 2.0252330545334344E+155);
            // Fails in 32-bit
            GaussianOp.MeanAverageConditional(sample, mean, precision, q);
        }

        [Fact]
        public void GaussianOpX()
        {
            Gaussian uniform = Gaussian.Uniform();
            Gaussian X0 = Gaussian.FromMeanAndVariance(3, 0.5);
            Gaussian Mean0 = Gaussian.FromMeanAndVariance(7, 1.0 / 3);
            double Precision0 = 3;
            Gaussian X, Mean;
            Gamma Precision, to_precision;
            Gaussian xActual, xExpected;

            bool testImproper = false;
            if (testImproper)
            {
                // Test the case where precisionIsBetween = false
                X = Gaussian.FromNatural(1, 2);
                Mean = Gaussian.FromNatural(3, -1);
                Precision = Gamma.FromShapeAndRate(4, 5);
                to_precision = Gamma.FromShapeAndRate(6, 7);
                xActual = GaussianOp.SampleAverageConditional(X, Mean, Precision, to_precision);
            }

            X = Gaussian.FromNatural(-2.7793306963303595, 0.050822473645365768);
            Mean = Gaussian.FromNatural(-5.9447032851878134E-09, 3.2975231004586637E-204);
            Precision = Gamma.FromShapeAndRate(318.50907574398883, 9.6226982361933746E+205);
            to_precision = Gamma.PointMass(0);
            xActual = GaussianOp.SampleAverageConditional(X, Mean, Precision, to_precision);
			
			X = Gaussian.FromNatural(0.1559599323109816, 8.5162535450918462);
            Mean = Gaussian.PointMass(0.57957597647840942);
            Precision = Gamma.FromShapeAndRate(7.8308812008325587E+30, 8.2854255911709925E+30);
            to_precision = Gamma.FromShapeAndRate(1.4709139487775529, 0.14968339171493822);
            xActual = GaussianOp.SampleAverageConditional(X, Mean, Precision, to_precision);

            X = Gaussian.FromNatural(0.15595993233964134, 8.5162535466550349);
            Mean = Gaussian.PointMass(0.57957597647840942);
            Precision = Gamma.FromShapeAndRate(3.9206259406339067E+20, 4.1481991194547565E+20);
            to_precision = Gamma.FromShapeAndRate(1.4709139487806249, 0.14968339171413536);
            xActual = GaussianOp.SampleAverageConditional(X, Mean, Precision, to_precision);

            X = Gaussian.FromNatural(0.15595993261634511, 8.5162535617468418);
            Mean = Gaussian.PointMass(0.57957597647840942);
            Precision = Gamma.FromShapeAndRate(1.825759224425317E+19, 1.9317356258150703E+19);
            to_precision = Gamma.FromShapeAndRate(1.4709139487887679, 0.14968339176002607);
            xActual = GaussianOp.SampleAverageConditional(X, Mean, Precision, to_precision);

            X = Gaussian.FromNatural(0.16501264432785923, 9.01);
            Mean = Gaussian.PointMass(0.57957597647840942);
            Precision = Gamma.FromShapeAndRate(1.6965139612477539E+21, 1.6965139612889427E+21);
            to_precision = Gamma.FromShapeAndRate(1.4695136363119978, 0.14707291154227081);
            xActual = GaussianOp.SampleAverageConditional(X, Mean, Precision, to_precision);

            // initialized in a bad place, gets stuck in a flat region
            X = Gaussian.FromNatural(3.9112579392580757, 11.631097473681082);
            Mean = Gaussian.FromNatural(10.449696977834144, 5.5617978202886995);
            Precision = Gamma.FromShapeAndRate(1.0112702817305146, 0.026480506235719053);
            to_precision = Gamma.FromShapeAndRate(1, 0.029622790537514355);
            xActual = GaussianOp.SampleAverageConditional(X, Mean, Precision, to_precision);

            X = Gaussian.FromNatural(57788.170908674481, 50207.150004827061);
            Mean = Gaussian.PointMass(0);
            Precision = Gamma.FromShapeAndRate(19764.051194189466, 0.97190264412377791);
            xActual = GaussianOp.SampleAverageConditional_slow(X, Mean, Precision);

            // integration bounds should be [-36,4]
            X = Gaussian.FromNatural(1.696828485456396, 0.71980672726406147);
            Mean = Gaussian.PointMass(-1);
            Precision = new Gamma(1, 1);
            xActual = GaussianOp.SampleAverageConditional_slow(X, Mean, Precision);
            xExpected = GaussianOp_Slow.SampleAverageConditional(X, Mean, Precision);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-4);

            X = new Gaussian(-1.565, 0.8466);
            Mean = new Gaussian(0.0682, 0.3629);
            Precision = new Gamma(103.2, 0.009786);
            xActual = GaussianOp.SampleAverageConditional_slow(X, Mean, Precision);
            xExpected = GaussianOp_Slow.SampleAverageConditional(X, Mean, Precision);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-4);

            // Fixed precision
            X = X0;
            Mean = uniform;
            Assert.True(GaussianOp.SampleAverageConditional(Mean, Precision0).MaxDiff(uniform) < 1e-10);
            Mean = Mean0;
            Assert.True(GaussianOp.SampleAverageConditional(Mean, Precision0).MaxDiff(new Gaussian(Mean.GetMean(), Mean.GetVariance() + 1 / Precision0)) < 1e-10);
            Mean = Gaussian.PointMass(Mean0.GetMean());
            Assert.True(GaussianOp.SampleAverageConditional(Mean, Precision0).MaxDiff(new Gaussian(Mean.GetMean(), 1 / Precision0)) < 1e-10);

            // Uniform precision
            // the answer should always be uniform
            Precision = Gamma.Uniform();
            Assert.True(GaussianOp.SampleAverageConditional_slow(X, Mean, Precision).MaxDiff(uniform) < 1e-10);

            // Unknown precision
            Precision = Gamma.FromShapeAndScale(3, 3);
            // Known X
            X = Gaussian.PointMass(X0.GetMean());
            Mean = uniform;
            Assert.True(GaussianOp.SampleAverageConditional_slow(X, Mean, Precision).MaxDiff(uniform) < 1e-10);
            // Unknown X
            X = X0;
            Mean = uniform;
            //Console.WriteLine(XAverageConditional2(result));
            Assert.True(GaussianOp.SampleAverageConditional_slow(X, Mean, Precision).MaxDiff(uniform) < 1e-10);
            X = X0;
            Mean = Mean0;
            // converge the precision message.  (only matters if KeepLastMessage is set).
            //for (int i = 0; i < 10; i++) GaussianOp.PrecisionAverageConditional(X, Mean, Precision, precisionMessage);
            // in matlab: test_t_msg
            if (GaussianOp.ForceProper)
                Assert.True(GaussianOp.SampleAverageConditional_slow(X, Mean, Precision).MaxDiff(Gaussian.FromNatural(3.1495, 0)) < 1e-4);
            else
                Assert.True(GaussianOp.SampleAverageConditional_slow(X, Mean, Precision).MaxDiff(new Gaussian(-9.9121, -4.5998)) < 1e-4);
            X = X0;
            Mean = Gaussian.PointMass(Mean0.GetMean());
            // converge the precision message.  (only matters if KeepLastMessage is set).
            //for (int i = 0; i < 10; i++) GaussianOp.PrecisionAverageConditional(X, Mean, Precision, precisionMessage);
            // in matlab: test_t_msg
            if (GaussianOp.ForceProper)
                Assert.True(GaussianOp.SampleAverageConditional_slow(X, Mean, Precision).MaxDiff(Gaussian.FromNatural(2.443, 0)) < 1e-4);
            else
                Assert.True(GaussianOp.SampleAverageConditional_slow(X, Mean, Precision).MaxDiff(new Gaussian(0.81394, -1.3948)) < 1e-4);
            // Uniform X
            X = uniform;
            Mean = Mean0;
            // converge the precision message.  (only matters if KeepLastMessage is set).
            //for (int i = 0; i < 10; i++) GaussianOp.PrecisionAverageConditional(X, Mean, Precision, precisionMessage);
            Assert.True(GaussianOp.SampleAverageConditional_slow(X, Mean, Precision).MaxDiff(new Gaussian(7, 0.5)) < 1e-10);
            X = uniform;
            Mean = Gaussian.PointMass(Mean0.GetMean());
            // converge the precision message.  (only matters if KeepLastMessage is set).
            //for (int i = 0; i < 10; i++) GaussianOp.PrecisionAverageConditional(X, Mean, Precision, precisionMessage);
            Assert.True(GaussianOp.SampleAverageConditional_slow(X, Mean, Precision).MaxDiff(new Gaussian(7, 1.0 / 6)) < 1e-10);
        }

        [Fact]
        public void GaussianOpMean()
        {
            Gaussian result = new Gaussian();
            Gaussian X0 = Gaussian.FromMeanAndVariance(7, 1.0 / 3);
            Gaussian Mean0 = Gaussian.FromMeanAndVariance(3, 0.5);
            Gamma Precision0 = Gamma.FromShapeAndScale(3, 3);

            // Unknown precision
            Gamma Precision = Gamma.FromShapeAndScale(3, 3);
            Gaussian X = X0;
            Gaussian Mean = Mean0;
            // in matlab: test_t_msg
            result = GaussianOp.MeanAverageConditional_slow(X, Mean, Precision);
            Console.WriteLine(result);
            Assert.True(GaussianOp.MeanAverageConditional_slow(X, Mean, Precision).MaxDiff(new Gaussian(-9.9121, -4.5998)) < 1e-0);

            X = Gaussian.FromMeanAndVariance(1, 2);
            Mean = Gaussian.PointMass(0);
            Precision = Gamma.FromShapeAndRate(3, 1);
            Gaussian xPostExpected = Gaussian.FromMeanAndVariance(0.178378819440295, 0.365796599498963);
            Console.WriteLine(GaussianOp.SampleAverageConditional_slow(X, Mean, Precision) * X);
            Assert.True(GaussianOp.SampleAverageConditional_slow(X, Mean, Precision).MaxDiff(xPostExpected / X) < 5e-7);
            Console.WriteLine(GaussianOp_Slow.SampleAverageConditional(X, Mean, Precision) * X);
            Assert.True(GaussianOp_Slow.SampleAverageConditional(X, Mean, Precision).MaxDiff(xPostExpected / X) < 5e-7);
        }

        /// <summary>
        /// Test that the operator behaves correctly when sample has large variance.
        /// Here we see that the message.Rate is non-monotonic in the sample variance, which doesn't seem right.
        /// </summary>
        [Fact]
        [Trait("Category", "ModifiesGlobals")]
        public void GaussianOpPrecision_IsMonotonicInSampleVariance()
        {
            using (TestUtils.TemporarilyAllowGaussianImproperMessages)
            {
                Gaussian mean = Gaussian.PointMass(0);
                for (int logRate = 0; logRate < 310; logRate++)
                {
                    Gamma precision = Gamma.FromShapeAndRate(300, System.Math.Pow(10, logRate));
                    double previousRate = double.PositiveInfinity;
                    for (int i = 0; i < 310; i++)
                    {
                        Gaussian sample = Gaussian.FromMeanAndPrecision(0, System.Math.Pow(10, -i));
                        Gamma precMsg = GaussianOp.PrecisionAverageConditional(sample, mean, precision);
                        //precMsg = GaussianOp_Laplace.PrecisionAverageConditional_slow(sample, mean, precision);
                        //Gamma precMsg2 = GaussianOp_Slow.PrecisionAverageConditional(sample, mean, precision);
                        //Console.WriteLine("{0}: {1} should be {2}", sample, precMsg, precMsg2);
                        Gamma post = precMsg * precision;
                        //Trace.WriteLine($"{sample}: {precMsg.Rate} post = {post.Rate}");
                        if (i >= logRate)
                            Assert.True(precMsg.Rate <= previousRate);
                        previousRate = precMsg.Rate;
                    }
                }
            }
        }

        [Fact]
        public void GaussianOpPrecision()
        {
            Gamma precMsg, precMsg2;
            Gaussian X, Mean;
            Gamma Precision;

            X = Gaussian.FromNatural(-1.5098177152950143E-09, 1.061649960537027E-168);
            Mean = Gaussian.FromNatural(-3.6177299471249587, 0.11664740799025652);
            Precision = Gamma.FromShapeAndRate(306.39423695125572, 1.8326832031565403E+170);
            precMsg = Gamma.PointMass(0);
            precMsg2 = GaussianOp_Slow.PrecisionAverageConditional(X, Mean, Precision);
            Assert.True(precMsg.MaxDiff(precMsg2) < 1e-4);
            precMsg2 = GaussianOp.PrecisionAverageConditional(X, Mean, Precision);
            Assert.True(precMsg.MaxDiff(precMsg2) < 1e-4);

            X = Gaussian.FromNatural(-0.55657497231637854, 6.6259783218464713E-141);
            Mean = Gaussian.FromNatural(-2.9330116542965374, 0.07513822741674292);
            Precision = Gamma.FromShapeAndRate(308.8184220331475, 4.6489382805051884E+142);
            precMsg = Gamma.FromShapeAndRate(1.5000000000000628, 3.5279086383286634E+279);
            precMsg2 = GaussianOp_Slow.PrecisionAverageConditional(X, Mean, Precision);
            Assert.True(precMsg.MaxDiff(precMsg2) < 1e-4);
            precMsg2 = GaussianOp.PrecisionAverageConditional(X, Mean, Precision);
            Assert.True(precMsg.MaxDiff(precMsg2) < 1e-4);

            X = Gaussian.FromNatural(0, 1.0705890985886898E-153);
            Mean = Gaussian.PointMass(0);
            Precision = Gamma.FromShapeAndRate(1.6461630749684018, 1.0021354807958952E+153);
            precMsg = Gamma.FromShapeAndRate(1.3230815374839406, 5.7102212927459039E+151);
            precMsg2 = GaussianOp_Slow.PrecisionAverageConditional(X, Mean, Precision);
            Assert.True(precMsg.MaxDiff(precMsg2) < 1e-4);
            precMsg2 = GaussianOp.PrecisionAverageConditional(X, Mean, Precision);
            Assert.True(precMsg.MaxDiff(precMsg2) < 1e-4);

            GaussianOp.PrecisionAverageConditional(Gaussian.FromNatural(0, 0.00849303091340374), Gaussian.FromNatural(0.303940178036662, 0.0357912415805232),
                                                   Gamma.FromNatural(0.870172077263786 - 1, 0.241027170904459));
            GaussianOp.PrecisionAverageConditional(Gaussian.FromNatural(0, 0.932143343115292), Gaussian.FromNatural(0.803368837946732, 0.096549750816333),
                                                   Gamma.FromNatural(0.63591693650741 - 1, 0.728459389753854));
            GaussianOp.PrecisionAverageConditional(Gaussian.FromNatural(0, 0.799724777601531), Gaussian.FromNatural(0.351882387116497, 0.0795619408970522),
                                                   Gamma.FromNatural(0.0398852019756498 - 1, 0.260567798400562));
            GaussianOp.PrecisionAverageConditional(Gaussian.FromNatural(0, 0.826197353576402), Gaussian.FromNatural(0.655970732055591, 0.125333868956814),
                                                   Gamma.FromNatural(0.202543332801453 - 1, 0.147645744563847));

            precMsg = GaussianOp.PrecisionAverageConditional(new Gaussian(-6.235e+207, 1.947e+209), Gaussian.PointMass(11), Gamma.PointMass(7));
            Assert.True(!double.IsNaN(precMsg.Rate));

            Gaussian X0 = Gaussian.FromMeanAndVariance(3, 0.5);
            Gaussian Mean0 = Gaussian.FromMeanAndVariance(7, 1.0 / 3);
            Gamma Precision0 = Gamma.FromShapeAndScale(3, 3);

            precMsg = GaussianOp_Slow.PrecisionAverageConditional(Gaussian.FromNatural(0.010158033515400506, 0.0041117304509528533),
                                                                  Gaussian.FromNatural(33.157651455559929, 13.955304749880149),
                                                                  Gamma.FromShapeAndRate(7.1611372018172794, 1.8190207317123008));

            precMsg = GaussianOp_Slow.PrecisionAverageConditional(Gaussian.FromNatural(-0.020177353724675218, 0.0080005002339157711),
                                                                  Gaussian.FromNatural(-12.303440746896294, 4.6439574387849714),
                                                                  Gamma.FromShapeAndRate(5.6778922774773992, 1.0667129560350435));

            precMsg = GaussianOp_Slow.PrecisionAverageConditional(Gaussian.PointMass(248), Gaussian.FromNatural(0.099086933095776319, 0.00032349393599347853),
                                                                  Gamma.FromShapeAndRate(0.001, 0.001));
            precMsg2 = GaussianOp.PrecisionAverageConditional_slow(Gaussian.PointMass(248), Gaussian.FromNatural(0.099086933095776319, 0.00032349393599347853),
                                                                   Gamma.FromShapeAndRate(0.001, 0.001));
            Assert.True(precMsg.MaxDiff(precMsg2) < 0.3);

            precMsg =
                GaussianOp_Slow.PrecisionAverageConditional(Gaussian.FromNatural(-0.21769764449791806, 0.0000024898838689952023),
                                                            Gaussian.FromNatural(0, 0.5), Gamma.FromShapeAndRate(5, 5));
            precMsg2 =
                GaussianOp.PrecisionAverageConditional_slow(Gaussian.FromNatural(-0.21769764449791806, 0.0000024898838689952023),
                                                            Gaussian.FromNatural(0, 0.5), Gamma.FromShapeAndRate(5, 5));
            //Console.WriteLine("{0} should be {1}", precMsg2, precMsg);
            Assert.True(precMsg.MaxDiff(precMsg2) < 1e-4);
            Gamma precMsg3 =
                GaussianOp_Laplace.PrecisionAverageConditional_slow(Gaussian.FromNatural(-0.21769764449791806, 0.0000024898838689952023),
                                                                    Gaussian.FromNatural(0, 0.5), Gamma.FromShapeAndRate(5, 5));
            precMsg2 =
                GaussianOp.PrecisionAverageConditional(Gaussian.FromNatural(-0.21769764449791806, 0.0000024898838689952023),
                                                       Gaussian.FromNatural(0, 0.5), Gamma.FromShapeAndRate(5, 5));
            //Console.WriteLine("{0} should be {1}", precMsg2, precMsg);
            Assert.True(precMsg.MaxDiff(precMsg2) < 1e-4);

            Assert.True(GaussianOp.PrecisionAverageConditional_slow(Gaussian.FromNatural(-2.3874057896477092, 0.0070584383295080044),
                                                                      Gaussian.FromNatural(1.3999879871144227, 0.547354438587195), Gamma.FromShapeAndRate(3, 1))
                                    .MaxDiff(Gamma.FromShapeAndRate(1.421, 55546)) < 10);

            // Unknown precision
            if (GaussianOp.ForceProper)
                Assert.True(GaussianOp.PrecisionAverageConditional_slow(X0, Mean0, Precision0).MaxDiff(Gamma.FromShapeAndRate(1, 0.3632)) < 1e-4);
            else
                Assert.True(GaussianOp.PrecisionAverageConditional_slow(X0, Mean0, Precision0).MaxDiff(Gamma.FromShapeAndRate(-0.96304, -0.092572)) < 1e-4);
            if (GaussianOp.ForceProper)
                Assert.True(GaussianOp.PrecisionAverageConditional_slow(Gaussian.PointMass(3.0), Mean0, Precision0).MaxDiff(Gamma.FromShapeAndRate(1, 4.13824)) < 1e-4);
            else
                Assert.True(GaussianOp.PrecisionAverageConditional_slow(Gaussian.PointMass(3.0), Mean0, Precision0).MaxDiff(Gamma.FromShapeAndRate(-0.24693, 2.2797)) < 1e-4);
            Assert.True(GaussianOp.PrecisionAverageConditional(3.0, 7.0).MaxDiff(Gamma.FromShapeAndRate(1.5, 8.0)) < 1e-4);
            Assert.True(GaussianOp.PrecisionAverageConditional_slow(new Gaussian(), Gaussian.PointMass(7.0), Precision0).MaxDiff(Gamma.FromShapeAndRate(1.0, 0.0)) < 1e-4);
        }

        /// <summary>
        /// Test that the operator behaves correctly for arguments with small variance
        /// </summary>
        [Fact]
        public void GaussianOpPrecision2()
        {
            Gamma Precision = Gamma.PointMass(3);
            Gaussian X = Gaussian.FromMeanAndVariance(3, 0.5);
            Gaussian Mean = Gaussian.FromMeanAndVariance(4, 1.0 / 3);
            Gamma precMsg = GaussianOp.PrecisionAverageConditional_slow(X, Mean, Precision);
            Console.WriteLine("{0}: {1}", Precision, precMsg);
            for (int i = 8; i < 30; i++)
            {
                Precision = Gamma.FromMeanAndVariance(3, System.Math.Pow(10, -i));
                Gamma precMsg2 = GaussianOp.PrecisionAverageConditional_slow(X, Mean, Precision);
                double diff = precMsg.MaxDiff(precMsg2);
                Console.WriteLine("{0}: {1} diff={2}", Precision, precMsg2, diff);
                Assert.True(diff < 1e-8);
                precMsg2 = GaussianOp_Slow.PrecisionAverageConditional(X, Mean, Precision);
                diff = precMsg.MaxDiff(precMsg2);
                Console.WriteLine("{0}: {1} diff={2}", Precision, precMsg2, diff);
                Assert.True(diff < 1e-8);
            }

        }

        internal void GaussianOpRandomTest()
        {
            int count = 0;
            while (true)
            {
                var mean = Gaussian.FromNatural(Rand.Double(), Rand.Double());
                var sample = Gaussian.FromNatural(0, Rand.Double());
                var precision = Gamma.FromNatural(Rand.Double() - 1, Rand.Double());
                try
                {
                    GaussianOp.PrecisionAverageConditional(sample, mean, precision);
                }
                catch (Exception e)
                {
                    if (e.Message == "not converging")
                    {
                        Console.WriteLine("sample {0} {1}", sample.MeanTimesPrecision, sample.Precision);
                        Console.WriteLine("mean {0} {1}", mean.MeanTimesPrecision, mean.Precision);
                        Console.WriteLine("prec {0} {1}", precision.Shape, precision.Rate);
                        Console.WriteLine();
                    }
                    else
                    {
                        Console.WriteLine(e);
                    }
                }
                count++;
                if (count % 100 == 0)
                    Console.WriteLine("{0}", count);
            }
        }
    }
}
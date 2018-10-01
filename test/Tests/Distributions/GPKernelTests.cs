// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Unit tests for Kernel functions

using System;
using System.Collections.Generic;
using System.IO;
using Xunit;
using Microsoft.ML.Probabilistic.Distributions.Kernels;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Tests
{
    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;

    /// <summary>
    /// Summary description for KernelTest
    /// </summary>
    public class GPKernelTests
    {
        private static double DITHER = 0.00000001;
        private static double DITHER_MULT = 1.0/(2.0*DITHER);
        private static double TOLERANCE = 0.0001;

        public GPKernelTests()
        {
            //
            // TODO: Add constructor logic here
            //
        }

        /// <summary>
        /// Local function that can be used to compare analytic with numeric derivatives
        /// </summary>
        /// <param name="kf">Kernel function</param>
        private void TestDerivatives(IKernelFunctionWithParams kf, Vector vec1, Vector vec2)
        {
            int thcnt = kf.ThetaCount;
            Vector analyticLogThetaDeriv = Vector.Zero(thcnt);
            Vector numericLogThetaDeriv = Vector.Zero(thcnt);
            Vector analytic_x1Deriv = Vector.Zero(vec1.Count);
            Vector numeric_x1Deriv = Vector.Zero(vec1.Count);
            Vector xNull = null;

            kf.EvaluateX1X2(vec1, vec2, ref analytic_x1Deriv, ref analyticLogThetaDeriv);

            // Calculate a numeric derivative for each parameter and compare with analytic
            for (int i = 0; i < thcnt; i++)
            {
                double d = kf[i];
                kf[i] = d + DITHER;
                double dplus = kf.EvaluateX1X2(vec1, vec2, ref xNull, ref xNull);
                kf[i] = d - DITHER;
                double dminus = kf.EvaluateX1X2(vec1, vec2, ref xNull, ref xNull);
                numericLogThetaDeriv[i] = DITHER_MULT*(dplus - dminus);
                Assert.True(System.Math.Abs(numericLogThetaDeriv[i] - analyticLogThetaDeriv[i]) < TOLERANCE);
                // Restore value
                kf[i] = d;
            }

            // Calculate a numeric derivative for each of the two vectors, and compare with analytic
            for (int i = 0; i < vec1.Count; i++)
            {
                double d = vec1[i];
                vec1[i] = d + DITHER;
                double dplus = kf.EvaluateX1X2(vec1, vec2, ref xNull, ref xNull);
                vec1[i] = d - DITHER;
                double dminus = kf.EvaluateX1X2(vec1, vec2, ref xNull, ref xNull);
                numeric_x1Deriv[i] = DITHER_MULT*(dplus - dminus);
                Assert.True(System.Math.Abs(numeric_x1Deriv[i] - analytic_x1Deriv[i]) < TOLERANCE);
                // Restore value
                vec1[i] = d;
            }
        }

        #region Additional test attributes

        //
        // You can use the following additional attributes as you write your tests:
        //
        // Use ClassInitialize to run code before running the first test in the class
        // [ClassInitialize()]
        // public static void MyClassInitialize(TestContext testContext) { }
        //
        // Use ClassCleanup to run code after all tests in a class have run
        // [ClassCleanup()]
        // public static void MyClassCleanup() { }
        //
        // Use TestInitialize to run code before running each test 
        // [TestInitialize()]
        // public void MyTestInitialize() { }
        //
        // Use TestCleanup to run code after each test has run
        // [TestCleanup()]
        // public void MyTestCleanup() { }
        //

        #endregion

        [Fact]
        public void TestKernelSummation()
        {
            double log_length = System.Math.Log(1.234);
            double log_sig_sd = System.Math.Log(2.345);
            double log_nse_sd = System.Math.Log(3.456);
            SummationKernel kf = new SummationKernel(new SquaredExponential(log_length, log_sig_sd));
            kf += new WhiteNoise(log_nse_sd);

            Assert.Equal(log_length, kf[0]);
            Assert.Equal(log_sig_sd, kf[1]);
            Assert.Equal(log_nse_sd, kf[2]);

            // Try resetting directly on the summation, using the name indexer
            log_length = 4.321;
            log_sig_sd = 5.432;
            log_nse_sd = 6.543;

            kf["Length"] = log_length;
            kf["SignalSD"] = log_sig_sd;
            kf["NoiseSD"] = log_nse_sd;

            Assert.Equal(log_length, kf[0]);
            Assert.Equal(log_sig_sd, kf[1]);
            Assert.Equal(log_nse_sd, kf[2]);
        }

        [Fact]
        public void TestSEKernelDerivs()
        {
            double log_length = System.Math.Log(1.234);
            double log_sig_sd = System.Math.Log(2.345);
            SquaredExponential sekf = new SquaredExponential(log_length, log_sig_sd);
            double[] x1 = {0.1, 0.2, 0.3};
            double[] x2 = {0.9, 0.7, 0.5};
            Vector x1Vec = Vector.FromArray(x1);
            Vector x2Vec = Vector.FromArray(x2);
            TestDerivatives(sekf, x1Vec, x1Vec);
            TestDerivatives(sekf, x1Vec, x2Vec);
        }

        [Fact]
        public void TestWNKernelDerivs()
        {
            double log_nse_sd = System.Math.Log(3.456);
            WhiteNoise wnkf = new WhiteNoise(log_nse_sd);
            double[] x1 = {0.1, 0.2, 0.3};
            double[] x2 = {0.9, 0.7, 0.5};
            Vector x1Vec = Vector.FromArray(x1);
            Vector x2Vec = Vector.FromArray(x2);
            TestDerivatives(wnkf, x1Vec, x1Vec);
            TestDerivatives(wnkf, x1Vec, x2Vec);
        }

        [Fact]
        public void TestARDKernelDerivs()
        {
            double[] logLengths = {-0.123, 0.456, 1.789};
            double log_sig_sd = System.Math.Log(3.456);
            ARD ardk = new ARD(logLengths, log_sig_sd);

            double[] x1 = {0.1, 0.2, 0.3};
            double[] x2 = {0.9, 0.7, 0.5};
            Vector v1 = Vector.FromArray(x1);
            Vector v2 = Vector.FromArray(x2);
            List<Vector> xlist = new List<Vector>(2);
            xlist.Add(v1);
            xlist.Add(v2);
            Vector y = Vector.Zero(2);
            y[0] = 0.3;
            y[1] = 0.5;

            ardk.InitialiseFromData(xlist, y);
            Vector x1Vec = Vector.FromArray(x1);
            Vector x2Vec = Vector.FromArray(x2);
            TestDerivatives(ardk, x1Vec, x1Vec);
            TestDerivatives(ardk, x1Vec, x2Vec);
        }

        [Fact]
        public void TestLinearKernelDerivs()
        {
            double[] logVariances = {-0.123, 0.456, 1.789};
            LinearKernel lk = new LinearKernel(logVariances);

            double[] x1 = {0.1, 0.2, 0.3};
            double[] x2 = {0.9, 0.7, 0.5};
            Vector v1 = Vector.FromArray(x1);
            Vector v2 = Vector.FromArray(x2);
            List<Vector> xlist = new List<Vector>(2);
            xlist.Add(v1);
            xlist.Add(v2);
            lk.InitialiseFromData(xlist);

            Vector x1Vec = Vector.FromArray(x1);
            Vector x2Vec = Vector.FromArray(x2);
            TestDerivatives(lk, x1Vec, x1Vec);
            TestDerivatives(lk, x1Vec, x2Vec);
        }

        [Fact]
        public void TestNNKernelDerivs()
        {
            NNKernel nnk = new NNKernel();

            double[] x1 = {0.1, 0.2, 0.3};
            double[] x2 = {0.9, 0.7, 0.5};
            Vector v1 = Vector.FromArray(x1);
            Vector v2 = Vector.FromArray(x2);
            List<Vector> xlist = new List<Vector>(2);
            xlist.Add(v1);
            xlist.Add(v2);
            nnk.InitialiseFromData(xlist);

            Vector x1Vec = Vector.FromArray(x1);
            Vector x2Vec = Vector.FromArray(x2);
            TestDerivatives(nnk, x1Vec, x1Vec);
            TestDerivatives(nnk, x1Vec, x2Vec);
        }

        [Fact]
        public void TestSumKDerivs()
        {
            double[] log_length = { System.Math.Log(0.543), System.Math.Log(0.432), System.Math.Log(0.321)};
            double log_sig_sd = System.Math.Log(2.345);
            double[] log_var = { System.Math.Log(0.987), System.Math.Log(0.876), System.Math.Log(0.765)};
            double log_nse_sd = System.Math.Log(3.456);
            SummationKernel kf = new SummationKernel(new ARD(log_length, log_sig_sd));
            kf += new LinearKernel(log_var);
            kf += new WhiteNoise(log_nse_sd);
            double[] x1 = {0.1, 0.2, 0.3};
            double[] x2 = {0.9, 0.7, 0.5};
            Vector x1Vec = Vector.FromArray(x1);
            Vector x2Vec = Vector.FromArray(x2);
            TestDerivatives(kf, x1Vec, x1Vec);
            TestDerivatives(kf, x1Vec, x2Vec);
        }

        [Fact]
        public void TestKernelFactory()
        {
            KernelFactory kfact = KernelFactory.Instance;
            IKernelFunction kf1 = kfact.CreateKernelFunction("WhiteNoise");
            IKernelFunction kf2 = kfact.CreateKernelFunction("SquaredExponential");
            IKernelFunction kf3 = kfact.CreateKernelFunction("ARD");
            IKernelFunction kf4 = kfact.CreateKernelFunction("LinearKernel");
            IKernelFunction kf5 = kfact.CreateKernelFunction("SummationKernel");
            IKernelFunction kf6 = kfact.CreateKernelFunction("NNKernel");

            Assert.NotNull(kf1);
            Assert.NotNull(kf2);
            Assert.NotNull(kf3);
            Assert.NotNull(kf4);
            Assert.NotNull(kf5);
            Assert.NotNull(kf6);

            Assert.IsType<WhiteNoise>(kf1);
            Assert.IsType<SquaredExponential>(kf2);
            Assert.IsType<ARD>(kf3);
            Assert.IsType<LinearKernel>(kf4);
            Assert.IsType<SummationKernel>(kf5);
            Assert.IsType<NNKernel>(kf6);
        }

        private static void TestReadWrite(IKernelFunctionWithParams kf)
        {
            string typeName = kf.GetType().Name;
            string fn = typeName + ".txt";

            // Write out the kernel
            StreamWriter sw = new StreamWriter(fn);
            kf.Write(sw);
            sw.Close();

            // Now read it back again
            StreamReader sr = new StreamReader(fn);
            KernelFactory kfact = KernelFactory.Instance;
            IKernelFunctionWithParams kf1 = kfact.CreateKernelFunction(typeName);
            kf1.Read(sr);
            sr.Close();

            // Now test that they're the same
            Assert.Equal(kf.ThetaCount, kf1.ThetaCount);
            for (int i = 0; i < kf.ThetaCount; i++)
                Assert.Equal(kf[i], kf1[i], 1e-6);
        }

        [Fact]
        public void TestWNReadWrite()
        {
            double log_nse_sd = System.Math.Log(3.456);
            WhiteNoise wnkf = new WhiteNoise(log_nse_sd);
            TestReadWrite(wnkf);
        }

        [Fact]
        public void TestSEReadWrite()
        {
            double log_length = System.Math.Log(1.234);
            double log_sig_sd = System.Math.Log(2.345);
            SquaredExponential sekf = new SquaredExponential(log_length, log_sig_sd);
            TestReadWrite(sekf);
        }

        [Fact]
        public void TestARDReadWrite()
        {
            double[] logLengths = {-0.123, 0.456, 1.789};
            double log_sig_sd = System.Math.Log(3.456);
            ARD ardk = new ARD(logLengths, log_sig_sd);
            TestReadWrite(ardk);
        }

        [Fact]
        public void TestNNReadWrite()
        {
            double[] logWeightVariances = {-0.123, 0.456, 1.789};
            double logBiasWtVars = System.Math.Log(3.456);
            NNKernel nnk = new NNKernel(logWeightVariances, logBiasWtVars);
            TestReadWrite(nnk);
        }

        [Fact]
        public void TestSummmationReadWrite()
        {
            double log_length = System.Math.Log(1.234);
            double log_sig_sd = System.Math.Log(2.345);
            double log_nse_sd = System.Math.Log(3.456);
            double[] log_var = { System.Math.Log(0.987), System.Math.Log(0.876), System.Math.Log(0.765)};
            SummationKernel kf = new SummationKernel(new SquaredExponential(log_length, log_sig_sd));
            kf += new LinearKernel(log_var);
            kf += new WhiteNoise(log_nse_sd);
            TestReadWrite(kf);
        }
    }
}
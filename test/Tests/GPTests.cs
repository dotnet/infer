// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Xunit;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Distributions.Kernels;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;
using Assert = Xunit.Assert;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
    /// <summary>
    /// Summary description for GPTests
    /// </summary>
    public class GPTests
    {
        [Fact]
        public void SparseGPTest()
        {
            // The basis
            Vector[] basis = new Vector[]
                {
                    Vector.FromArray(new double[2] {0.2, 0.2}),
                    Vector.FromArray(new double[2] {0.2, 0.8}),
                    Vector.FromArray(new double[2] {0.8, 0.2}),
                    Vector.FromArray(new double[2] {0.8, 0.8})
                };

            // The kernel
            IKernelFunction kf = new NNKernel(new double[] {0.0, 0.0}, 0.0);

            // The fixed parameters
            SparseGPFixed sgpf = new SparseGPFixed(kf, basis);

            // Alpha and beta
            Vector alpha = Vector.Zero(basis.Length);
            PositiveDefiniteMatrix beta = new PositiveDefiniteMatrix(basis.Length, basis.Length);
            for (int i = 0; i < alpha.Count; i++)
            {
                alpha[i] = i;
                for (int j = 0; j < alpha.Count; j++)
                {
                    beta[i, j] = alpha.Count - System.Math.Abs(i - j);
                }
            }

            SparseGP a = new SparseGP(sgpf);
#if false
            Rank1Pot r1p = new Rank1Pot();
            r1p.Xi = basis[0];   // must be a basis point for this test
            r1p.Yi = 2.2;
            r1p.LambdaInv = 0.7;
            SparseGP b = new SparseGP(sgpf, r1p);
            DistributionTests.SettableToRatioTest(a, b);
#endif
            DistributionTests.ProductWithUniformTest(a);
            DistributionTests.RatioWithUniformTest(a);
            DistributionTests.SettableToTest(a);
        }

        [Fact]
        public void GPClassificationTest()
        {
            bool[] yData = new bool[] {false, false, false, true, true, true, true, false, false, false};
            double[] xData = new double[]
                {
                    -2, -1.555555555555556, -1.111111111111111, -0.6666666666666667, -0.2222222222222223, 0.2222222222222223, 0.6666666666666665, 1.111111111111111,
                    1.555555555555555, 2
                };
            Vector[] xVec = Array.ConvertAll(xData, v => Vector.Constant(1, v));
            Vector[] basis = new Vector[] {xVec[1], xVec[4], xVec[8]};
            //basis = xVec;
            IKernelFunction kf = new SquaredExponential(0.0);
            SparseGPFixed sgpf = new SparseGPFixed(kf, basis);
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<IFunction> f = Variable.Random<IFunction>(new SparseGP(sgpf)).Named("f");
            Range item = new Range(xVec.Length).Named("item");
            VariableArray<Vector> x = Variable.Array<Vector>(item).Named("x");
            x.ObservedValue = xVec;
            VariableArray<bool> y = Variable.Array<bool>(item).Named("y");
            y.ObservedValue = yData;
            VariableArray<double> h = Variable.Array<double>(item).Named("h");
            h[item] = Variable.FunctionEvaluate(f, x[item]);
            y[item] = (h[item] > 0);
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            SparseGP sgp = engine.Infer<SparseGP>(f);
            Vector alphaExpected = Vector.FromArray(new double[] {-1.410457563120709, 1.521306076273262, -1.008600221619413});
            Console.WriteLine("alpha = {0} should be {1}", sgp.Alpha, alphaExpected);
            double[] xTest = new double[]
                {
                    -2, -1, 0.0
                };
            Vector[] xTestVec = Array.ConvertAll(xTest, v => Vector.Constant(1, v));
            // computed by matlab/MNT/GP/test_gpc.m
            double[] yMeanTest = new double[]
                {
                    -0.966351175090184, -0.123034591744284, 0.762757400008960
                };
            double[] yVarTest = new double[]
                {
                    0.323871157983366, 0.164009511251333, 0.162068482365962
                };
            for (int i = 0; i < xTestVec.Length; i++)
            {
                Gaussian pred = sgp.Marginal(xTestVec[i]);
                Gaussian predExpected = new Gaussian(yMeanTest[i], yVarTest[i]);
                Console.WriteLine("f({0}) = {1} should be {2}", xTest[i], pred, predExpected);
                Assert.True(predExpected.MaxDiff(pred) < 1e-4);
            }
            double evExpected = -4.907121241357144;
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-4);
        }

        [Fact]
        public void GPClassificationTest1()
        {
            bool[] yData = new bool[] {true};
            double[] xData = new double[] {-0.2222222222222223};
            Vector[] xVec = Array.ConvertAll(xData, v => Vector.Constant(1, v));
            Vector[] basis = new Vector[] {Vector.Zero(1)};
            IKernelFunction kf = new SquaredExponential(0.0);
            SparseGPFixed sgpf = new SparseGPFixed(kf, basis);
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<IFunction> f = Variable.Random<IFunction>(new SparseGP(sgpf)).Named("f");
            Range item = new Range(xVec.Length).Named("item");
            VariableArray<Vector> x = Variable.Array<Vector>(item).Named("x");
            x.ObservedValue = xVec;
            VariableArray<bool> y = Variable.Array<bool>(item).Named("y");
            y.ObservedValue = yData;
            VariableArray<double> h = Variable.Array<double>(item).Named("h");
            h[item] = Variable.FunctionEvaluate(f, x[item]);
            y[item] = (h[item] > 0);
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            SparseGP sgp = engine.Infer<SparseGP>(f);
            Vector alphaExpected = Vector.FromArray(new double[] {0.778424938343491});
            Console.WriteLine("alpha = {0} should be {1}", sgp.Alpha, alphaExpected);
            double[] xTest = new double[]
                {
                    -2, -1, 0.0
                };
            Vector[] xTestVec = Array.ConvertAll(xTest, v => Vector.Constant(1, v));
            double[] yMeanTest = new double[]
                {
                    0.105348359509159, 0.472138591390244, 0.778424938343491
                };
            double[] yVarTest = new double[]
                {
                    0.988901723148729, 0.777085150520037, 0.394054615364932
                };
            for (int i = 0; i < xTestVec.Length; i++)
            {
                Gaussian pred = sgp.Marginal(xTestVec[i]);
                Gaussian predExpected = new Gaussian(yMeanTest[i], yVarTest[i]);
                Console.WriteLine("f({0}) = {1} should be {2}", xTest[i], pred, predExpected);
                Assert.True(predExpected.MaxDiff(pred) < 1e-4);
            }
            double evExpected = -0.693147180559945;
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-4);
        }

        [Fact]
        public void GPClassificationTest2()
        {
            bool[] yData = new bool[] {false, true, false};
            double[] xData = new double[]
                {
                    -1.555555555555556, -0.2222222222222223, 1.555555555555555
                };
            Vector[] xVec = Array.ConvertAll(xData, v => Vector.Constant(1, v));
            Vector[] basis = new Vector[] {Vector.Zero(1)};
            IKernelFunction kf = new SquaredExponential(0.0);
            SparseGPFixed sgpf = new SparseGPFixed(kf, basis);
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<IFunction> f = Variable.Random<IFunction>(new SparseGP(sgpf)).Named("f");
            Range item = new Range(xVec.Length).Named("item");
            VariableArray<Vector> x = Variable.Array<Vector>(item).Named("x");
            x.ObservedValue = xVec;
            VariableArray<bool> y = Variable.Array<bool>(item).Named("y");
            y.ObservedValue = yData;
            VariableArray<double> h = Variable.Array<double>(item).Named("h");
            h[item] = Variable.FunctionEvaluate(f, x[item]);
            y[item] = (h[item] > 0);
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            SparseGP sgp = engine.Infer<SparseGP>(f);
            Vector alphaExpected = Vector.FromArray(new double[] {0.573337393823702});
            Console.WriteLine("alpha = {0} should be {1}", sgp.Alpha, alphaExpected);
            double[] xTest = new double[]
                {
                    -2, -1, 0.0
                };
            Vector[] xTestVec = Array.ConvertAll(xTest, v => Vector.Constant(1, v));
            double[] yMeanTest = new double[]
                {
                    0.077592778583272, 0.347746707713812, 0.573337393823702
                };
            double[] yVarTest = new double[]
                {
                    0.986784459962251, 0.734558782611933, 0.278455962249970
                };
            for (int i = 0; i < xTestVec.Length; i++)
            {
                Gaussian pred = sgp.Marginal(xTestVec[i]);
                Gaussian predExpected = new Gaussian(yMeanTest[i], yVarTest[i]);
                Console.WriteLine("f({0}) = {1} should be {2}", xTest[i], pred, predExpected);
                Assert.True(predExpected.MaxDiff(pred) < 1e-4);
            }
            double evExpected = -2.463679892165236;
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-4);
        }

        [Fact]
        public void GPClassificationTest5()
        {
            bool[] yData = new bool[] {false, false, false, true, false};
            double[] xData = new double[]
                {
                    -2, -1.555555555555556, -1.111111111111111, -0.2222222222222223, 1.555555555555555
                };
            Vector[] xVec = Array.ConvertAll(xData, v => Vector.Constant(1, v));
            Vector[] basis = new Vector[] {Vector.Zero(1)};
            IKernelFunction kf = new SquaredExponential(0.0);
            SparseGPFixed sgpf = new SparseGPFixed(kf, basis);
            Variable<IFunction> f = Variable.Random<IFunction>(new SparseGP(sgpf)).Named("f");
            Range item = new Range(xVec.Length).Named("item");
            VariableArray<Vector> x = Variable.Array<Vector>(item).Named("x");
            x.ObservedValue = xVec;
            VariableArray<bool> y = Variable.Array<bool>(item).Named("y");
            y.ObservedValue = yData;
            VariableArray<double> h = Variable.Array<double>(item).Named("h");
            h[item] = Variable.FunctionEvaluate(f, x[item]);
            y[item] = (h[item] > 0);

            InferenceEngine engine = new InferenceEngine();
            engine.NumberOfIterations = 10;
            SparseGP sgp = engine.Infer<SparseGP>(f);
            Vector alphaExpected = Vector.FromArray(new double[] {0.387823538790656});
            Console.WriteLine("alpha = {0} should be {1}", sgp.Alpha, alphaExpected);
        }

        [Fact]
        public void GPClassificationTest4()
        {
            bool[] yData = new bool[] {false, false, true, false};
            double[] xData = new double[]
                {
                    -1.555555555555556, -1.111111111111111, -0.2222222222222223, 1.555555555555555
                };
            Vector[] xVec = Array.ConvertAll(xData, v => Vector.Constant(1, v));
            Vector[] basis = new Vector[] {Vector.Zero(1)};
            IKernelFunction kf = new SquaredExponential(0.0);
            SparseGPFixed sgpf = new SparseGPFixed(kf, basis);
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<IFunction> f = Variable.Random<IFunction>(new SparseGP(sgpf)).Named("f");
            Range item = new Range(xVec.Length).Named("item");
            VariableArray<Vector> x = Variable.Array<Vector>(item).Named("x");
            x.ObservedValue = xVec;
            VariableArray<bool> y = Variable.Array<bool>(item).Named("y");
            y.ObservedValue = yData;
            VariableArray<double> h = Variable.Array<double>(item).Named("h");
            h[item] = Variable.FunctionEvaluate(f, x[item]);
            y[item] = (h[item] > 0);
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            SparseGP sgp = engine.Infer<SparseGP>(f);
            Vector alphaExpected = Vector.FromArray(new double[] {0.409693797629808});
            Console.WriteLine("alpha = {0} should be {1}", sgp.Alpha, alphaExpected);
        }

        [Fact]
        public void GPRegressionTest()
        {
            double[] yData = new double[]
                {
                    -0.06416828853982412, -0.6799959810206935, -0.4541652863622044, 0.155770359928991, 1.036659040456137, 0.7353821980830825, 0.8996680933259047,
                    -0.05368704705684217, -0.7905775695015919, -0.1436284683992815
                };
            double[] xData = new double[]
                {
                    -2, -1.555555555555556, -1.111111111111111, -0.6666666666666667, -0.2222222222222223, 0.2222222222222223, 0.6666666666666665, 1.111111111111111,
                    1.555555555555555, 2
                };
            Vector[] xVec = Array.ConvertAll(xData, v => Vector.Constant(1, v));
            Vector[] basis = new Vector[] {xVec[1], xVec[4], xVec[8]};
            IKernelFunction kf = new SquaredExponential(System.Math.Log(2.0));
            SparseGPFixed sgpf = new SparseGPFixed(kf, basis);
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<IFunction> f = Variable.Random<IFunction>(new SparseGP(sgpf)).Named("f");
            Range item = new Range(xVec.Length).Named("item");
            VariableArray<Vector> x = Variable.Array<Vector>(item).Named("x");
            x.ObservedValue = xVec;
            VariableArray<double> y = Variable.Array<double>(item).Named("y");
            y.ObservedValue = yData;
            VariableArray<double> h = Variable.Array<double>(item).Named("h");
            h[item] = Variable.FunctionEvaluate(f, x[item]);
            y[item] = Variable.GaussianFromMeanAndVariance(h[item], 0.1);
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            SparseGP sgp = engine.Infer<SparseGP>(f);
            Vector alphaExpected = Vector.FromArray(new double[] {-3.250044160725389, 4.579296091435270, -2.227005562666341});
            PositiveDefiniteMatrix betaExpected = new PositiveDefiniteMatrix(new double[,]
                {
                    {3.187555652658986, -3.301824438047169, 1.227566907279797},
                    {-3.30182443804717, 5.115027119603418, -2.373085083966294},
                    {1.227566907279797, -2.373085083966294, 2.156308696222915}
                });
            Console.WriteLine("alpha = {0} should be {1}", sgp.Alpha, alphaExpected);
            Console.WriteLine(StringUtil.JoinColumns("beta = ", sgp.Beta, " should be ", betaExpected));
            double[] xTest = new double[]
                {
                    -2, -1, 0.0
                };
            Vector[] xTestVec = Array.ConvertAll(xTest, v => Vector.Constant(1, v));
            // computed by matlab/MNT/GP/test_gpr.m
            double[] yMeanTest = new double[]
                {
                    -0.544583265595561, 0.134323399801302, 0.503623822120711
                };
            double[] yVarTest = new double[]
                {
                    0.058569682375201, 0.022695532903985, 0.024439582002951
                };
            for (int i = 0; i < xTestVec.Length; i++)
            {
                Gaussian pred = sgp.Marginal(xTestVec[i]);
                Gaussian predExpected = new Gaussian(yMeanTest[i], yVarTest[i]);
                Console.WriteLine("f({0}) = {1} should be {2}", xTest[i], pred, predExpected);
                Assert.True(predExpected.MaxDiff(pred) < 1e-4);
            }
            double evExpected = -13.201173794945003;
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-4);
        }

        [Fact]
        public void GPRegressionTest1()
        {
            double[] yData = new double[] {1.036659040456137};
            double[] xData = new double[] {-0.2222222222222223};
            Vector[] xVec = Array.ConvertAll(xData, v => Vector.Constant(1, v));
            Vector[] basis = new Vector[] {xVec[0]};
            IKernelFunction kf = new SquaredExponential(System.Math.Log(2.0));
            SparseGPFixed sgpf = new SparseGPFixed(kf, basis);
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<IFunction> f = Variable.Random<IFunction>(new SparseGP(sgpf)).Named("f");
            Range item = new Range(xVec.Length).Named("item");
            VariableArray<Vector> x = Variable.Array<Vector>(item).Named("x");
            x.ObservedValue = xVec;
            VariableArray<double> y = Variable.Array<double>(item).Named("y");
            y.ObservedValue = yData;
            VariableArray<double> h = Variable.Array<double>(item).Named("h");
            h[item] = Variable.FunctionEvaluate(f, x[item]);
            y[item] = Variable.GaussianFromMeanAndVariance(h[item], 0.1);
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            SparseGP sgp = engine.Infer<SparseGP>(f);
            Vector alphaExpected = Vector.FromArray(new double[] {0.942417309505579});
            Console.WriteLine("alpha = {0} should be {1}", sgp.Alpha, alphaExpected);
            double[] xTest = new double[]
                {
                    -2, -1, 0.0
                };
            Vector[] xTestVec = Array.ConvertAll(xTest, v => Vector.Constant(1, v));
            // computed by matlab/MNT/GP/test_gpr.m
            double[] yTest = new double[]
                {
                    0.634848540665472, 0.873781982196160, 0.936617836728720
                };
            for (int i = 0; i < xTestVec.Length; i++)
            {
                double pred = sgp.Mean(xTestVec[i]);
                Console.WriteLine("Ef({0}) = {1} should be {2}", xTest[i], pred, yTest[i]);
                Assert.True(MMath.AbsDiff(pred, yTest[i], 1e-6) < 1e-4);
            }
            double evExpected = -1.455076334997490;
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-4);
        }

        [Fact]
        public void GPClassificationExample()
        {
            // The data
            bool[] ydata = {true, true, false, true, false, false};
            Vector[] xdata = new Vector[]
                {
                    Vector.FromArray(new double[2] {0, 0}),
                    Vector.FromArray(new double[2] {0, 1}),
                    Vector.FromArray(new double[2] {1, 0}),
                    Vector.FromArray(new double[2] {0, 0.5}),
                    Vector.FromArray(new double[2] {1.5, 0}),
                    Vector.FromArray(new double[2] {0.5, 1.0})
                };

            // Open an evidence block to allow model scoring
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);

            // Set up the GP prior, which will be filled in later
            Variable<SparseGP> prior = Variable.New<SparseGP>().Named("prior");

            // The sparse GP variable - a distribution over functions
            Variable<IFunction> f = Variable<IFunction>.Random(prior).Named("f");

            // The locations to evaluate the function
            VariableArray<Vector> x = Variable.Constant(xdata).Named("x");
            Range j = x.Range.Named("j");

            // The observation model
            VariableArray<bool> y = Variable.Array<bool>(j).Named("y");
            y[j] = (Variable.GaussianFromMeanAndVariance(Variable.FunctionEvaluate(f, x[j]), 0.1) > 0);

            // Attach the observations
            y.ObservedValue = ydata;

            // Close the evidence block
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();

            // The basis
            Vector[] basis = new Vector[]
                {
                    Vector.FromArray(new double[2] {0.2, 0.2}),
                    Vector.FromArray(new double[2] {0.2, 0.8}),
                    Vector.FromArray(new double[2] {0.8, 0.2}),
                    Vector.FromArray(new double[2] {0.8, 0.8})
                };

            for (int trial = 0; trial < 3; trial++)
            {
                // The kernel
                IKernelFunction kf;
                if (trial == 0)
                {
                    kf = new SquaredExponential(-0.0);
                    //kf = new LinearKernel(new double[] { 0.0, 0.0 });
                }
                else if (trial == 1)
                {
                    kf = new SquaredExponential(-0.5);
                }
                else
                {
                    kf = new NNKernel(new double[] {0.0, 0.0}, -1.0);
                }

                // Fill in the sparse GP prior
                prior.ObservedValue = new SparseGP(new SparseGPFixed(kf, basis));

                // Model score
                double NNscore = engine.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("{0} evidence = {1}", kf, NNscore.ToString("g4"));
            }

            // Infer the posterior Sparse GP
            SparseGP sgp = engine.Infer<SparseGP>(f);

            // Check that training set is classified correctly
            Console.WriteLine();
            Console.WriteLine("Predictions on training set:");
            for (int i = 0; i < ydata.Length; i++)
            {
                Gaussian post = sgp.Marginal(xdata[i]);
                double postMean = post.GetMean();
                string comment = (ydata[i] == (postMean > 0.0)) ? "correct" : "incorrect";
                Console.WriteLine("f({0}) = {1} ({2})", xdata[i], post, comment);
                Assert.True(ydata[i] == (postMean > 0.0));
            }
        }

        public static PositiveDefiniteMatrix GramMatrix(IKernelFunction kf, Vector[] xData)
        {
            int nData = xData.Length;

            // Allocate and fill the Kernel matrix.
            PositiveDefiniteMatrix K = new PositiveDefiniteMatrix(nData, nData);

            for (int i = 0; i < nData; i++)
            {
                for (int j = 0; j < nData; j++)
                {
                    // Evaluate the kernel. All hyperparameters, including noise
                    // variance are handled in the kernel.
                    K[i, j] = kf.EvaluateX1X2(xData[i], xData[j]);
                }
            }
            return K;
        }

        [Fact]
        public void BasicGPC()
        {
            Vector[] inputs = new Vector[]
                {
                    Vector.FromArray(new double[2] {0, 0}),
                    Vector.FromArray(new double[2] {0, 1}),
                    Vector.FromArray(new double[2] {1, 0}),
                    Vector.FromArray(new double[2] {0, 0.5}),
                    Vector.FromArray(new double[2] {1.5, 0}),
                    Vector.FromArray(new double[2] {0.5, 1.0})
                };
            bool[] outputs = {true, true, false, true, false, false};

            var kf = new SummationKernel(new SquaredExponential(0));
            kf += new WhiteNoise(System.Math.Log(0.1));
            var K = GramMatrix(kf, inputs);

            var n = new Range(inputs.Length);
            var x = Variable.VectorGaussianFromMeanAndVariance(Vector.Zero(inputs.Length), K);
            var g = Variable.ArrayFromVector(x, n);
            var p = Variable.Array<bool>(n);
            p[n] = Variable.IsPositive(g[n]);
            p.ObservedValue = outputs;
            var ie = new InferenceEngine();
            Console.WriteLine(ie.Infer(x));
        }
    }
}
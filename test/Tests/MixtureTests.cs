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
using Assert = Xunit.Assert;
using System.Linq;
using Microsoft.ML.Probabilistic.Algorithms;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
    /// <summary>
    /// EP mixture tests
    /// </summary>
    public class MixtureTests
    {
        internal void MixtureLinearRegressionTest()
        {
            int nFeature = 2; // need to change w and probs in MakeData method, if this is changed
            int nComponent = 2; // need to change  w and probs in MakeData method, if this is changed
            double[] wtVector = new double[nComponent];
            Gaussian[][] wPriorObs = new Gaussian[nComponent][];
            for (int c = 0; c < nComponent; c++)
            {
                wtVector[c] = 2;
            }

            Range feature = new Range(nFeature);
            Range component = new Range(nComponent);
            Variable<int> nItem = Variable.New<int>().Named("nItems");
            Range item = new Range(nItem).Named("item");
            var w = Variable.Array(Variable.Array<double>(feature), component).Named("w");
            var wPrior = Variable.Array(Variable.Array<Gaussian>(feature), component).Named("wPrior");
            using (Variable.ForEach(component))
            {
                w[component][feature] = Variable.Random<double, Gaussian>(wPrior[component][feature]);
            }

            var x = Variable.Array(Variable.Array<double>(feature), item).Named("x");
            VariableArray<double> y = Variable.Array<double>(item).Named("y");
            VariableArray<int> yClass = Variable.Array<int>(item).Named("yClass");
            Variable<Vector> weights = Variable.Dirichlet(component, wtVector).Named("weights");


            using (Variable.ForEach(item))
            {
                yClass[item] = Variable.Discrete(weights);
                VariableArray<double> product = Variable.Array<double>(feature).Named("product");
                using (Variable.Switch(yClass[item]))
                {
                    product[feature] = w[yClass[item]][feature]*x[item][feature];
                }
                y[item] = Variable.GaussianFromMeanAndVariance(Variable.Sum(product), 1);
            }

            double[][] xData;
            double[] yData;
            MakeData(out xData, out yData, out wPriorObs);

            wPrior.ObservedValue = wPriorObs;
            x.ObservedValue = xData;
            y.ObservedValue = yData;
            nItem.ObservedValue = yData.Length;
            InferenceEngine ie = new InferenceEngine();
            ie.Infer(w);
            //    DistributionArray<GaussianArray> wInfer = ie.Infer<DistributionArray<GaussianArray>>(w);
        }

        private static void MakeData(out double[][] x, out double[] y, out Gaussian[][] wPriorObs)
        {
            double[][] w = {new double[] {3, 1}, new double[] {-1, 0}};
            double[] probs = new double[] {.5, .5};

            int nComponent = w.GetLength(0);
            int nFeature = w[0].GetLength(0);
            int T = 5;
            x = new double[T][];
            y = new double[T];
            for (int i = 0; i < T; i++)
            {
                int t = (new Discrete(probs)).Sample();
                x[i] = new double[nComponent];
                double tot = 0;
                for (int j = 0; j < nFeature; j++)
                {
                    x[i][j] = (new Gaussian(1, .01)).Sample();
                    tot = tot + x[i][j]*w[t][j];
                }
                y[i] = tot;
            }
            wPriorObs = new Gaussian[2][];
            for (int c = 0; c < 2; c++)
            {
                wPriorObs[c] = new Gaussian[2];
                for (int j = 0; j < 2; j++)
                {
                    wPriorObs[c][j] = new Gaussian(0, 1);
                }
            }
        }

        [Fact]
        public void BernoulliMixtureGaussianTest()
        {
            int N = 10, D = 2, K = 2;
            Range n = new Range(N).Named("n");
            Range k = new Range(K).Named("k");
            Range d = new Range(D).Named("d");
            VariableArray2D<double> p = Variable.Array<double>(k, d).Named("p");
            p[k, d] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(k, d);
            VariableArray2D<bool> x = Variable.Array<bool>(n, d).Named("x");
            VariableArray<int> c = Variable.Array<int>(n).Named("c");
            using (Variable.ForEach(n))
            {
                c[n] = Variable.Discrete(k, 0.5, 0.5);
                using (Variable.Switch(c[n]))
                {
                    x[n, d] = (Variable.GaussianFromMeanAndVariance(p[c[n], d], 1.0) > 0);
                }
            }
            bool geForceProper = GateEnterOp<double>.ForceProper;
            try
            {
                GateEnterOp<double>.ForceProper = true;
                InferenceEngine engine = new InferenceEngine(); //new VariationalMessagePassing());
                engine.Compiler.GivePriorityTo(typeof (IsPositiveOp_Proper)); // needed to avoid improper messages in EP
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

                engine.NumberOfIterations = 1;
                var pExpected = engine.Infer(p);
                engine.NumberOfIterations = engine.Algorithm.DefaultNumberOfIterations;
                DistributionArray<Discrete> cPost = engine.Infer<DistributionArray<Discrete>>(c);
                Console.WriteLine(cPost);
                DistributionArray2D<Gaussian> pPost = engine.Infer<DistributionArray2D<Gaussian>>(p);
                Console.WriteLine(pPost);

                // test resetting inference
                engine.NumberOfIterations = 1;
                var pActual = engine.Infer<Diffable>(p);
                Assert.True(pActual.MaxDiff(pExpected) < 1e-10);
            }
            finally
            {
                GateEnterOp<double>.ForceProper = geForceProper;
            }
        }

        [Fact]
        public void BernoulliMixtureTest()
        {
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
            InferenceEngine engine = new InferenceEngine();
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

            engine.NumberOfIterations = 1;
            var pExpected = engine.Infer(p);
            engine.NumberOfIterations = engine.Algorithm.DefaultNumberOfIterations;
            DistributionArray<Discrete> cPost = engine.Infer<DistributionArray<Discrete>>(c);
            Console.WriteLine(cPost);
            DistributionArray2D<Beta> pPost = engine.Infer<DistributionArray2D<Beta>>(p);
            Console.WriteLine(pPost);

            // test resetting inference
            engine.NumberOfIterations = 1;
            var pActual = engine.Infer<Diffable>(p);
            Assert.True(pActual.MaxDiff(pExpected) < 1e-10);
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void ClutterTest()
        {
            //Evidence:
            //  adf     = -5.14434 (actually -5.14434)
            //  ep      = -5.12942 (actually -5.12942)
            //  vb      = -6.0229 (actually -6.0229)
            //  laplace = -5.79976 (actually -5.79976)
            //  exact   = -5.1055 (actually -5.1055)

            //Posterior mean:
            //  adf     = 0.724355
            //  ep      = 0.789959
            //  vb      = 1.31257
            //  laplace = 1.26649
            //  exact   = 0.637152

            //Posterior variance:
            //  adf     = 53.8953
            //  ep      = 40.1263
            //  vb      = 0.892128
            //  laplace = 1.2662
            //  exact   = 51.5467

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(Clutter, new Gaussian(0, 100), new Gaussian(0, 10), 0.1, 2.3);
            ca.Execute(20);

            Gaussian meanExpected = new Gaussian(0.789959, 40.1263);
            Gaussian meanActual = ca.Marginal<Gaussian>("mean");
            Console.WriteLine("mean={0} (expected {1})", meanActual, meanExpected);
            Console.WriteLine("b1=" + ca.Marginal("b1"));
            Console.WriteLine("b2=" + ca.Marginal("b2"));

            double evidenceExpected = -5.12942;
            double evidenceActual = ca.Marginal<Bernoulli>("evidence").LogOdds;
            Console.WriteLine("evidence={0} (expected {1})", evidenceActual, evidenceExpected);
            Assert.True(MMath.AbsDiff(evidenceActual, evidenceExpected, 1e-4) < 1e-4);

            Assert.True(meanActual.MaxDiff(meanExpected) < 1e-4);
        }

        private static void Clutter(Gaussian meanPrior, Gaussian noiseDist, double data1, double data2)
        {
            double prec = 1;
            double mixWeight = 0.5;
            bool evidence = Factor.Random(new Bernoulli(0.5));
            if (evidence)
            {
                double mean = Factor.Random(meanPrior);
                bool b1 = Factor.Bernoulli(mixWeight);
                double x1;
                if (b1)
                {
                    x1 = Factor.Gaussian(mean, prec);
                }
                else
                {
                    x1 = Factor.Random(noiseDist);
                }
                Constrain.Equal(x1, data1);
                bool b2 = Factor.Bernoulli(mixWeight);
                double x2;
                if (b2)
                {
                    x2 = Factor.Gaussian(mean, prec);
                }
                else
                {
                    x2 = Factor.Random(noiseDist);
                }
                Constrain.Equal(x2, data2);
                InferNet.Infer(mean, nameof(mean));
                InferNet.Infer(b1, nameof(b1));
                InferNet.Infer(b2, nameof(b2));
            }
            InferNet.Infer(evidence, nameof(evidence));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void ClutterWithLoopsTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            double[] data = {0.1, 2.3};
            var ca = engine.Compiler.Compile(ClutterWithLoops, new Gaussian(0, 100), new Gaussian(0, 10), data);
            ca.Execute(20);

            Gaussian meanExpected = new Gaussian(0.789959, 40.1263);
            Gaussian meanActual = ca.Marginal<Gaussian>("mean");
            Console.WriteLine("mean={0} (expected {1})", meanActual, meanExpected);
            Console.WriteLine("b =");
            Console.WriteLine(ca.Marginal("b"));

            double evidenceExpected = -5.12942;
            double evidenceActual = ca.Marginal<Bernoulli>("evidence").LogOdds;
            Console.WriteLine("evidence={0} (expected {1})", evidenceActual, evidenceExpected);
            Assert.True(MMath.AbsDiff(evidenceActual, evidenceExpected, 1e-4) < 1e-4);

            Assert.True(meanActual.MaxDiff(meanExpected) < 1e-4);
        }

        private void ClutterWithLoops(Gaussian meanPrior, Gaussian noiseDist, double[] data)
        {
            bool evidence = Factor.Random(new Bernoulli(0.5));
            if (evidence)
            {
                double prec = 1;
                double mixWeight = 0.5;
                double mean = Factor.Random(meanPrior);
                bool[] b = new bool[data.Length];
                for (int i = 0; i < data.Length; i++)
                {
                    b[i] = Factor.Bernoulli(mixWeight);
                    double x;
                    if (b[i])
                    {
                        x = Factor.Gaussian(mean, prec);
                    }
                    else
                    {
                        x = Factor.Random(noiseDist);
                    }
                    Constrain.Equal(x, data[i]);
                }
                InferNet.Infer(mean, nameof(mean));
                InferNet.Infer(b, nameof(b));
            }
            InferNet.Infer(evidence, nameof(evidence));
        }

        internal Vector[] GenerateData(int nData, double truePi)
        {
            Vector trueM1 = Vector.FromArray(new double[] {2.0, 3.0});
            Vector trueM2 = Vector.FromArray(new double[] {7.0, 5.0});
            PositiveDefiniteMatrix trueP1 = new PositiveDefiniteMatrix(
                new double[,] {{3.0, 0.2}, {0.2, 2.0}});
            PositiveDefiniteMatrix trueP2 = new PositiveDefiniteMatrix(
                new double[,] {{2.0, 0.4}, {0.4, 4.0}});
            VectorGaussian trueVG1 = VectorGaussian.FromMeanAndPrecision(trueM1, trueP1);
            VectorGaussian trueVG2 = VectorGaussian.FromMeanAndPrecision(trueM2, trueP2);
            Bernoulli trueB = new Bernoulli(truePi);
            // Restart the infer.NET random number generator
            Rand.Restart(12347);
            Vector[] data = new Vector[nData];
            for (int j = 0; j < nData; j++)
            {
                bool bSamp = trueB.Sample();
                data[j] = bSamp ? trueVG1.Sample() : trueVG2.Sample();
            }
            return data;
        }

        /// <summary>
        /// Test initialization outside the iteration loop.
        /// </summary>
        [Fact]
        public void PoissonMixtureTest()
        {
            Rand.Restart(1);

            int N = 40, D = 2, K = 2;
            Range n = new Range(N).Named("n");
            Range k = new Range(K).Named("k");
            Range d = new Range(D).Named("d");
            VariableArray2D<double> p = Variable.Array<double>(k, d).Named("p");
            p[k, d] = Variable.GammaFromMeanAndVariance(10, 100).ForEach(k, d);
            VariableArray2D<int> x = Variable.Array<int>(n, d).Named("x");
            VariableArray<int> c = Variable.Array<int>(n).Named("c");
            using (Variable.ForEach(n))
            {
                c[n] = Variable.Discrete(k, 0.5, 0.5);
                using (Variable.Switch(c[n]))
                {
                    x[n, d] = Variable.Poisson(p[c[n], d]);
                }
            }
            //n.AddAttribute(new Sequential());
            //c.AddAttribute(new DivideMessages(false));
            InferenceEngine engine = new InferenceEngine();
            //engine.Algorithm = new VariationalMessagePassing();
            int[,] data = new int[N,D];
            int N1 = N/2;
            double[,] mean = new double[K,D];
            for (int i = 0; i < K; i++)
            {
                for (int j = 0; j < D; j++)
                {
                    //mean[i, j] = i+j;
                    mean[i, j] = (i + j + 1)*10;
                }
            }
            Discrete[] cInit = new Discrete[N];
            for (int i = 0; i < N; i++)
            {
                int cluster = i%2;
                for (int j = 0; j < D; j++)
                {
                    data[i, j] = Rand.Poisson(mean[cluster, j]);
                }
                double r = cluster;
                cInit[i] = new Discrete(1 - r, r);
            }
            x.ObservedValue = data;
            c.InitialiseTo(Distribution<int>.Array(cInit));

            engine.NumberOfIterations = 1;
            var pPost1 = engine.Infer(p);

            engine.NumberOfIterations = 200;
            Gamma[,] pPost = engine.Infer<Gamma[,]>(p);
            for (int i = 0; i < pPost.GetLength(0); i++)
            {
                for (int j = 0; j < pPost.GetLength(1); j++)
                {
                    double mActual = pPost[i, j].GetMean();
                    double mExpected = mean[i, j];
                    Console.WriteLine(String.Format("pPost[{0}][{1}] = {2} should be {3}", i, j, mActual, mExpected));
                    Assert.True(MMath.AbsDiff(mExpected, mActual, 1e-6) < 0.3);
                }
            }

            // test resetting inference
            engine.NumberOfIterations = 1;
            var pPost2 = engine.Infer<Diffable>(p);
            Assert.True(pPost2.MaxDiff(pPost1) < 1e-10);
        }

        internal void PoissonMixtureTest2()
        {
            bool constraintInside = true;

            int components = 3;
            Range k = new Range(components).Named("k");

            VariableArray<Gamma> meanPriors = Variable.Array<Gamma>(k).Named("meanPriors");

            VariableArray<double> means = Variable.Array<double>(k).Named("means");
            means[k] = Variable<double>.Random(meanPriors[k]);

            double[] priorWeights = Enumerable.Repeat<double>(1, components).ToArray();
            Variable<Vector> weights = Variable.Dirichlet(k, priorWeights).Named("weights");

            Variable<int> num = Variable.New<int>().Named("num");
            Range n = new Range(num).Named("n");

            VariableArray<int> z = Variable.Array<int>(n).Named("z");
            VariableArray<int> numSessions = Variable.Array<int>(n).Named("numSessions");

            using (Variable.ForEach(n))
            {
                z[n] = Variable.Discrete(weights);
                using (Variable.Switch(z[n]))
                {
                    numSessions[n] = Variable.Poisson(means[z[n]]);
                    if(constraintInside)
                        Variable.ConstrainTrue(numSessions[n] >= 5);
                }
            }

            // Set observed priors
            meanPriors.ObservedValue = Util.ArrayInit(components, t => new Gamma(100, 0.01));

            meanPriors.ObservedValue = new[] {
                new Gamma(5.363e+04, 0.001202), 
                new Gamma(5.583e+04, 0.0002394), 
                new Gamma(8.217e+04, 2.045e-05)
            };

            // Do some inference
            num.ObservedValue = 1;
            if(!constraintInside)
                Variable.ConstrainTrue(numSessions[n] >= 5);

            var engine = new InferenceEngine();

            // Print out weight vector and count
            Console.WriteLine(engine.Infer(z));
            Console.WriteLine(engine.Infer(numSessions));
        }

        /// <summary>
        /// Runs the MixtureOfGaussians tutorial, with assertions.
        /// </summary>
        [Fact]
        public void MixtureOfMultivariateGaussians()
        {
            // Define a range for the number of mixture components
            Range k = new Range(2).Named("k");

            // Mixture component means
            VariableArray<Vector> means = Variable.Array<Vector>(k).Named("means");
            means[k] = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(2), PositiveDefiniteMatrix.IdentityScaledBy(2, 0.01)).ForEach(k);

            // Mixture component precisions
            VariableArray<PositiveDefiniteMatrix> precs = Variable.Array<PositiveDefiniteMatrix>(k).Named("precs");
            precs[k] = Variable.WishartFromShapeAndScale(100.0, PositiveDefiniteMatrix.IdentityScaledBy(2, 0.01)).ForEach(k);

            // Mixture weights 
            Variable<Vector> weights = Variable.Dirichlet(k, new double[] {1, 1}).Named("weights");

            // Create a variable array which will hold the data
            Range n = new Range(300).Named("n");
            VariableArray<Vector> data = Variable.Array<Vector>(n).Named("x");
            // Create latent indicator variable for each data point
            VariableArray<int> z = Variable.Array<int>(n).Named("z");

            // The mixture of Gaussians model
            using (Variable.ForEach(n))
            {
                z[n] = Variable.Discrete(weights);
                using (Variable.Switch(z[n]))
                {
                    data[n] = Variable.VectorGaussianFromMeanAndPrecision(means[z[n]], precs[z[n]]);
                }
            }

            // Attach some generated data
            double truePi = 0.6;
            data.ObservedValue = GenerateData(n.SizeAsInt, truePi);

            // Initialise messages randomly to break symmetry
            VariableArray<Discrete> zInit = Variable.Array<Discrete>(n).Named("zInit");
            bool useObservedValue = true;
            if (useObservedValue)
            {
                zInit.ObservedValue = Util.ArrayInit(n.SizeAsInt, i => Discrete.PointMass(Rand.Int(k.SizeAsInt), k.SizeAsInt));
            }
            else
            {
                // This approach doesn't work, because Infer.NET notices that Rand.Int is stochastic and thinks that it should perform message-passing here.
                using (Variable.ForEach(n))
                {
                    var randk = Variable<int>.Factor(new Func<int, int>(Rand.Int), (Variable<int>)k.Size);
                    randk.SetValueRange(k);
                    zInit[n] = Variable<Discrete>.Factor(Discrete.PointMass, randk, (Variable<int>)k.Size);
                }
            }
            z[n].InitialiseTo(zInit[n]);

            // The inference
            InferenceEngine ie = new InferenceEngine();
            ie.Algorithm = new VariationalMessagePassing();
            //ie.Compiler.GenerateInMemory = false;
            //ie.NumberOfIterations = 200;
            Dirichlet wDist = (Dirichlet) ie.Infer(weights);
            Vector wEstMean = wDist.GetMean();

            object meansActual = ie.Infer(means);
            Console.WriteLine("means = ");
            Console.WriteLine(meansActual);
            var precsActual = ie.Infer<IList<Wishart>>(precs);
            Console.WriteLine("precs = ");
            Console.WriteLine(precsActual);
            Console.WriteLine("w = {0} should be {1}", wEstMean, Vector.FromArray(truePi, 1 - truePi));
            //Console.WriteLine(StringUtil.JoinColumns("z = ", ie.Infer(z)));
            Assert.True(
                MMath.AbsDiff(wEstMean[0], truePi) < 0.05 ||
                MMath.AbsDiff(wEstMean[1], truePi) < 0.05);
        }
    }
}
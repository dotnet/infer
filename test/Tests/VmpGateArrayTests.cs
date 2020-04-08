// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Xunit;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Algorithms;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
    
    public class VmpGateArrayTests
    {
        [Fact]
        public void MissingDataGaussianTest()
        {
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100).Named("mean");
            Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1).Named("precision");
            Variable<int> n = Variable.New<int>().Named("n");
            Range i = new Range(n).Named("i");
            VariableArray<double> x = Variable.Array<double>(i).Named("x");
            using (Variable.ForEach(i))
            {
                using (Variable.If(x[i] > 0))
                {
                    x[i] = Variable.GaussianFromMeanAndPrecision(mean, precision);
                }
            }
            x.ObservedValue = new double[] {-1, 5.0, -1, 7.0, -1};
            n.ObservedValue = x.ObservedValue.Length;

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            //Console.WriteLine(engine.Infer(isMissing));
            Gaussian meanExpected = Gaussian.FromMeanAndVariance(5.9603207170807826, 0.66132138200164436);
            Gamma precisionExpected = Gamma.FromShapeAndRate(2, 2.6628958274937107);
            Gaussian meanActual = engine.Infer<Gaussian>(mean);
            Gamma precisionActual = engine.Infer<Gamma>(precision);
            Console.WriteLine("mean = {0} should be {1}", meanActual, meanExpected);
            Console.WriteLine("precision = {0} should be {1}", precisionActual, precisionExpected);
            Assert.True(meanExpected.MaxDiff(meanActual) < 1e-10);
            Assert.True(precisionExpected.MaxDiff(precisionActual) < 1e-10);
        }

        [Fact]
        public void MixtureWithConstantSelector()
        {
            Range Trange = new Range(2).Named("Trange");
            VariableArray<double> x = Variable.Constant(new double[] {1, 15}, Trange).Named("x");

            Range Krange = new Range(2).Named("Krange");
            VariableArray<double> means = Variable.Array<double>(Krange);
            means[Krange] = Variable.GaussianFromMeanAndPrecision(2, 1).ForEach(Krange);

            VariableArray<int> c = Variable.Constant(new int[] {0, 1}, Trange).Named("c");
            c[Trange] = Variable.DiscreteUniform(Krange).ForEach(Trange);

            using (Variable.ForEach(Trange))
            {
                using (Variable.Switch(c[Trange]))
                {
                    x[Trange] = Variable.GaussianFromMeanAndPrecision(means[c[Trange]], 1);
                }
            }
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(ie.Infer(means));
        }


        [Fact]
        public void MixtureOfDiscretePartialObs()
        {
            int K = 2;
            Range Krange = new Range(K).Named("K");

            Variable<Vector> D = Variable.Dirichlet(Krange, new double[] {1, 1}).Named("D");

            VariableArray<Vector> pi = Variable.Array<Vector>(Krange).Named("pi");
            pi[Krange] = Variable.Dirichlet(new double[] {1, 1}).ForEach(Krange);
            VariableArray<Vector> pi2 = Variable.Array<Vector>(Krange);
            pi2[Krange] = Variable.Dirichlet(new double[] {2, 2}).ForEach(Krange).Named("pi2");

            int[] xData = new int[] {1, 0};
            int[] yData = new int[] {1, 0};
            Range T = new Range(2).Named("T");
            VariableArray<int> x = Variable.Constant(xData, T).Named("x");
            VariableArray<int> y = Variable.Constant(yData, T).Named("y");
            VariableArray<int> c = Variable.Array<int>(T).Named("c");
            using (Variable.ForEach(T))
            {
                c[T] = Variable.Discrete(D);
                using (Variable.Switch(c[T]))
                {
                    x[T] = Variable.Discrete(pi[c[T]]);
                    y[T] = Variable.Discrete(pi2[c[T]]);
                }
            }

            int[] xObsData = new int[] {1};
            int[] yObsData = new int[] {1};
            int[] cObsData = new int[] {1};
            Range TObs = new Range(1).Named("TObs");
            VariableArray<int> xObs = Variable.Constant(xObsData, TObs).Named("xObs");
            VariableArray<int> yObs = Variable.Constant(yObsData, TObs).Named("yObs");
            VariableArray<int> cObs = Variable.Constant(cObsData, TObs).Named("cObs");
            using (Variable.ForEach(TObs))
            {
                cObs[TObs] = Variable.Discrete(D);
                using (Variable.Switch(cObs[TObs]))
                {
                    xObs[TObs] = Variable.Discrete(pi[cObs[TObs]]);
                    yObs[TObs] = Variable.Discrete(pi2[cObs[TObs]]);
                }
            }

            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            VmpTests.TestDirichletMoments(ie, D, new double[] {-1.33369062459866, -0.45559642996801});
            double[][] cVibesResult = new double[2][];
            cVibesResult[0] = new double[] {0.12507345314883, 0.87492654685117};
            cVibesResult[1] = new double[] {0.53017394027554, 0.46982605972446};
            VmpTests.TestDiscrete(ie, c, cVibesResult);

            double[][] piVibesResult = new double[2][];
            piVibesResult[0] = new double[] {-0.71224535522876, -1.16495845985210};
            piVibesResult[1] = new double[] {-1.34159372292073, -0.47735375561174};
            VmpTests.TestDirichletMoments(ie, pi, piVibesResult);

            double[][] pi2VibesResult = new double[2][];
            pi2VibesResult[0] = new double[] {-0.70890921479891, -0.92632718538028};
            pi2VibesResult[1] = new double[] {-1.07851393492695, -0.54677568970466};
            VmpTests.TestDirichletMoments(ie, pi2, pi2VibesResult);
            // bound: -5.7515783
        }


        // AK: ignore this for now.
        /*  public void tmp()
            {
                    Range xRange = new Range("xRange" ,2);
                    VariableArray<double> x = Variable.Array<double>(xRange).Named("x");
                    x[xRange] = Variable.GaussianFromMeanAndPrecision(1, .01).ForEach(xRange);

                    Variable<double> y = Variable.GaussianFromMeanAndPrecision(4, 5);
                    Variable<double> prec = Variable.GammaFromShapeAndScale(1, 1);
                    double[] data1 = { 20, 10 };
                    ConstantArray<double> xPyNoisy = Variable.Constant(data1,xRange).Named("xPyNoisy");
                    Variable.ConstrainEqual(xPyNoisy[xRange], Variable.GaussianFromMeanAndPrecision(x[xRange] + y, prec));


                    double[] data = { 33,3};
                    int T = data.Length;
                    Range zRange = new Range("zRange", T);
                    ConstantArray<double> z = Variable.Constant(data, zRange).Named("z");

         
                    Variable<Vector> D = Variable.Dirichlet(new double[] { 1, 1 });
                    VariableArray<int> c = Variable.Array<int>(zRange);
                    c[zRange] = Variable.Discrete(D).ForEach(zRange);
                    using (Variable.Case(c[zRange], 0))
                    {
                            z[zRange] = Variable.GaussianFromMeanAndPrecision(xPyNoisy[xRange], prec);
                    }
                    using (Variable.Case(c[zRange], 1))
                    {
                            z[zRange] = Variable.GaussianFromMeanAndPrecision(xPyNoisy[xRange], prec);
                    }
                    InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());

                    Console.WriteLine(ie.Infer(c));
                    Console.WriteLine(ie.Infer(x));
                    Console.WriteLine(ie.Infer(y));
                    Console.WriteLine(ie.Infer(D));

            }*/


        [Fact]
        public void Mixture1()
        {
            double[] data = {7};
            int T = data.Length;
            VariableArray<double> x = Variable.Constant(data).Named("data");
            Range i = x.Range;
            Variable<Vector> D = Variable.Dirichlet(new double[] {1, 1});

            VariableArray<int> c = Variable.Array<int>(i);
            using (Variable.ForEach(i))
            {
                c[i] = Variable.Discrete(D);
                using (Variable.Case(c[i], 0))
                {
                    x[i] = Variable.GaussianFromMeanAndPrecision(5.5, 1);
                }
                using (Variable.Case(c[i], 1))
                {
                    x[i] = Variable.GaussianFromMeanAndPrecision(8, 1);
                }
            }

            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());

            double[] DVibesResult = new double[] {-1.15070203880043, -0.67501576717763};
            VmpTests.TestDirichletMoments(ie, D, DVibesResult);
            double[][] cVibesResult = new double[T][];
            cVibesResult[0] = new double[] {0.24961241199438, 0.75038758800562};
            VmpTests.TestDiscrete(ie, c, cVibesResult);
        }

        [Fact]
        public void Mixture2()
        {
            double[] data = {.5, 12, 11};
            int T = data.Length;
            VariableArray<double> x = Variable.Constant(data).Named("data");
            Range i = x.Range;
            Variable<Vector> D = Variable.Dirichlet(new double[] {1, 1});

            VariableArray<int> c = Variable.Array<int>(i);
            using (Variable.ForEach(i))
            {
                c[i] = Variable.Discrete(D);
                using (Variable.Case(c[i], 0))
                {
                    x[i] = Variable.GaussianFromMeanAndPrecision(5, 1);
                }
                using (Variable.Case(c[i], 1))
                {
                    x[i] = Variable.GaussianFromMeanAndPrecision(10, 1);
                }
            }

            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());

            double[] DVibesResult = new double[] {-1.08333333348476, -0.58333333335936};
            VmpTests.TestDirichletMoments(ie, D, DVibesResult);
            double[][] cVibesResult = new double[T][];
            cVibesResult[0] = new double[] {1.00000000000000, 0.00000000000000};
            cVibesResult[1] = new double[] {0.00000000000000, 1.00000000000000};
            cVibesResult[2] = new double[] {0.00000000000000, 1.00000000000000};
            VmpTests.TestDiscrete(ie, c, cVibesResult);
        }


        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MixtureOfTwoGaussiansWithLoopsTest()
        {
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            double[] data = {0.1, 9.3};
            var ca = engine.Compiler.Compile(MixtureOfTwoGaussiansWithLoopsModel, data);
            ca.Execute(20);
            Console.WriteLine("mean1=" + ca.Marginal("mean1"));
            Console.WriteLine("mean2=" + ca.Marginal("mean2"));
        }

        private void MixtureOfTwoGaussiansWithLoopsModel(double[] x)
        {
            double mean1 = Factor.Gaussian(0.1, 0.0001);
            double mean2 = Factor.Gaussian(10.2, 0.0001);
            double prec = 1;
            double mixWeight = 0.5;
            bool[] b = new bool[x.Length];
            for (int i = 0; i < x.Length; i++)
            {
                b[i] = Factor.Bernoulli(mixWeight);
                if (b[i])
                {
                    x[i] = Factor.Gaussian(mean1, prec);
                }
                else
                {
                    x[i] = Factor.Gaussian(mean2, prec);
                }
            }
            InferNet.Infer(mean1, nameof(mean1));
            InferNet.Infer(mean2, nameof(mean2));
            InferNet.Infer(b, nameof(b));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MixtureOfThreeGaussiansTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = new VariationalMessagePassing();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            double[] data = {0, 1, 2, 5, 6, 7, 10, 11, 12};
            var ca = engine.Compiler.Compile(MixtureOfThreeGaussiansModel, data);
            ca.Execute(20);
            Console.WriteLine("mean1=" + ca.Marginal("mean1"));
            Console.WriteLine("mean2=" + ca.Marginal("mean2"));
            Console.WriteLine("mean3=" + ca.Marginal("mean3"));
        }


        private void MixtureOfThreeGaussiansModel(double[] x)
        {
            double mean1 = Factor.Gaussian(1, 0.1);
            double mean2 = Factor.Gaussian(2, 0.1);
            double mean3 = Factor.Gaussian(3, 0.1);
            double prec = 1;
            for (int i = 0; i < x.Length; i++)
            {
                int j = Factor.Random(Discrete.Uniform(3));
                if (j == 0)
                {
                    x[i] = Factor.Gaussian(mean1, prec);
                }
                if (j == 1)
                {
                    x[i] = Factor.Gaussian(mean2, prec);
                }
                if (j == 2)
                {
                    x[i] = Factor.Gaussian(mean3, prec);
                }
                //InferNet.Infer(j, nameof(j));
            }
            InferNet.Infer(mean1, nameof(mean1));
            InferNet.Infer(mean2, nameof(mean2));
            InferNet.Infer(mean3, nameof(mean3));
        }


        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MixtureOfThreeGaussiansWithArraysTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = new VariationalMessagePassing();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            double[] data = {0, 1, 2, 5, 6, 7, 10, 11, 12};
            var ca = engine.Compiler.Compile(MixtureOfThreeGaussiansWithArraysModel, data);
            ca.Execute(20);
            Console.WriteLine("mean=" + ca.Marginal("mean"));
        }


        private void MixtureOfThreeGaussiansWithArraysModel(double[] x)
        {
            double[] mean = new double[3];
            double prec = 1;
            mean[0] = Factor.Gaussian(1, 0.1);
            mean[1] = Factor.Gaussian(2, 0.1);
            mean[2] = Factor.Gaussian(3, 0.1);
            for (int i = 0; i < x.Length; i++)
            {
                int j = Factor.Random(Discrete.Uniform(3));
                if (j == 0)
                {
                    x[i] = Factor.Gaussian(mean[j], prec);
                }
                if (j == 1)
                {
                    x[i] = Factor.Gaussian(mean[j], prec);
                }
                if (j == 2)
                {
                    x[i] = Factor.Gaussian(mean[j], prec);
                }
                //InferNet.Infer(j, nameof(j));
            }
            InferNet.Infer(mean, nameof(mean));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MixtureOfManyGaussiansTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = new VariationalMessagePassing();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            double[] data = {0, 1, 2, 5, 6, 7, 10, 11, 12};
            var ca = engine.Compiler.Compile(MixtureOfManyGaussiansModel, data, 3);
            ca.Execute(20);
            Console.WriteLine("mean=" + ca.Marginal("mean"));
        }


        private void MixtureOfManyGaussiansModel(double[] x, int N)
        {
            double[] meanPriorMeans = new double[] {1, 2, 3};
            double[] mean = new double[N];
            for (int n = 0; n < N; n++)
            {
                mean[n] = Factor.Gaussian(meanPriorMeans[n], 0.1);
            }
            double prec = 1; // Factor.Random(new Gamma(1, 1)); ;
            for (int i = 0; i < x.Length; i++)
            {
                int j = Factor.Random(Discrete.Uniform(N));
                for (int k = 0; k < N; k++)
                {
                    if (j == k)
                    {
                        x[i] = Factor.Gaussian(mean[j], prec);
                    }
                }
            }
            InferNet.Infer(mean, nameof(mean));
        }
    }
}
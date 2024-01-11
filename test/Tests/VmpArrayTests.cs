// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using Xunit;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Models;
    using Microsoft.ML.Probabilistic.Utilities;
    using Assert = Xunit.Assert;
    using Microsoft.ML.Probabilistic.Collections;
    using GaussianArray = Microsoft.ML.Probabilistic.Distributions.DistributionStructArray<Microsoft.ML.Probabilistic.Distributions.Gaussian, double>;
    using GaussianArray2D = Microsoft.ML.Probabilistic.Distributions.DistributionStructArray2D<Microsoft.ML.Probabilistic.Distributions.Gaussian, double>;
    using System.Diagnostics;
    using System.IO;
    using Microsoft.ML.Probabilistic.Algorithms;
    using Microsoft.ML.Probabilistic.Compiler;
    using Microsoft.ML.Probabilistic.Serialization;
    using Range = Microsoft.ML.Probabilistic.Models.Range;

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif


    public class VmpArrayTests
    {
        /// <summary>
        /// Test that the compiler rejects this model.  Otherwise you get a bad non-convergent schedule.
        /// </summary>
        [Fact]
        public void MatrixMultiplyMustBeStochasticError()
        {
            Assert.Throws<CompilationFailedException>(() =>
            {

                Range observation = new Range(2).Named("observation");
                Range trait = new Range(2).Named("trait");
                Range user = new Range(2).Named("user");
                Range item = new Range(2).Named("item");
                var userTraits = Variable.Array(Variable.Array<double>(trait), user).Named("userTraits");
                userTraits[user][trait] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(user, trait);
                var itemTraits = Variable.Array(Variable.Array<double>(trait), item).Named("itemTraits");
                var userData = Variable.Observed(new int[] { 0, 1 }, observation).Named("userData");
                userData.SetValueRange(user);
                var itemData = Variable.Observed(new int[] { 0, 1 }, observation).Named("itemData");
                itemData.SetValueRange(item);
                Range trivial1 = new Range(1);
                Range trivial2 = new Range(1);
                var affinities = Variable.Array<double>(observation).Named("trueAffinity");
                using (Variable.ForEach(observation))
                {
                    var r1 = Variable.Array<double>(trivial1, trait).Named("userRowMatrix");
                    r1[0, trait] = userTraits[userData[observation]][trait];

                    var r2 = Variable.Array<double>(trait, trivial2).Named("itemRowMatrix");
                    r2[trait, 0] = itemTraits[itemData[observation]][trait];

                    Variable<double> innerProd = Variable.MatrixMultiply(r1, r2)[0, 0];
                    var noisy = Variable.GaussianFromMeanAndVariance(innerProd, 1);
                    Variable.ConstrainEqual(noisy, 1.0);
                }

                InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
                itemTraits.ObservedValue = new double[][] { new double[] { 1.0, 2.0 }, new double[] { 3.0, 4.0 } };
                Console.WriteLine(engine.Infer(userTraits));

            });
        }

        // Test MatrixMultiply with uniform and point mass output
        [Fact]
        public void MatrixMultiplyTest2()
        {
            var Aprior = new DistributionStructArray2D<Gaussian, double>(
                 1, 1, new Gaussian[] { Gaussian.FromMeanAndPrecision(0, 1) });
            var Bprior = new DistributionStructArray2D<Gaussian, double>(
                  1, 2, new Gaussian[] { Gaussian.FromMeanAndPrecision(3, 1), Gaussian.FromMeanAndPrecision(2, 1) });
            Range n = new Range(1);
            Range k = new Range(1);
            Range m = new Range(2);
            var A = Variable.Array<double>(n, k).Named("A");
            A.SetTo(Variable.Random(Aprior));
            var B = Variable.Array<double>(k, m).Named("B");
            B.SetTo(Variable.Random(Bprior));
            var product = Variable.MatrixMultiply(A, B).Named("product");
            var productLike = new DistributionStructArray2D<Gaussian, double>(
                  1, 2, new Gaussian[] { Gaussian.Uniform(), Gaussian.PointMass(2) });
            Variable.ConstrainEqualRandom(product, productLike);

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            var Apost = engine.Infer<IArray2D<Gaussian>>(A);
            Console.WriteLine(Apost);
            Assert.False(double.IsNaN(Apost[0, 0].GetMean()));

            B.ObservedValue = new double[,] { { 3, 2 } };
            Apost = engine.Infer<IArray2D<Gaussian>>(A);
            Console.WriteLine(Apost);
            Assert.False(double.IsNaN(Apost[0, 0].GetMean()));
        }

        /// <summary>
        /// Tests that MatrixMultiply works with loop replication.  
        /// Currently fails for similar reason as ChangePointTest2
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")]
        public void MatrixMultiplyTest()
        {
            var result1 = MatrixMultiply(false);
            var result2 = MatrixMultiply(true);
            Assert.True(result1.MaxDiff(result2) < 1e-4);
        }
        private Diffable MatrixMultiply(bool unroll)
        {
            var rN = new Range(2).Named("N");
            var rD = new Range(2).Named("D");
            var rK = new Range(2).Named("K");
            var rM = new Range(2).Named("M");

            var vW = Variable.Array(Variable.Array<double>(rK, rD), rM).Named("W");
            vW[rM][rK, rD] = Variable.GaussianFromMeanAndPrecision(2, 1).ForEach(rM, rK, rD);

            // Data mean
            var product = Variable.Array(Variable.Array<double>(rN, rD), rM).Named("T");

            // Latent variables are drawn from a standard Gaussian
            var vZ = Variable.Array<double>(rN, rK).Named("Z");
            vZ[rN, rK] = Variable.GaussianFromMeanAndPrecision(3, 1.0).ForEach(rN, rK);

            if (unroll)
            {
                for (int i = 0; i < rM.SizeAsInt; i++)
                    product[i] = Variable.MatrixMultiply(vZ, vW[i]);
            }
            else
            {
                product[rM] = Variable.MatrixMultiply(vZ, vW[rM]);
            }
            var productLike =
                new DistributionRefArray<GaussianArray2D, double[,]>(rM.SizeAsInt, i =>
                 new DistributionStructArray2D<Gaussian, double>(
                   rN.SizeAsInt, rD.SizeAsInt, new Gaussian[] { Gaussian.Uniform(), Gaussian.PointMass(2), new Gaussian(4, 5), new Gaussian(6, 7) }));
            Variable.ConstrainEqualRandom(product, productLike);

            var engine = new InferenceEngine(new VariationalMessagePassing());
            return engine.Infer<Diffable>(product);
        }

        [Fact]
        public void SumWhereTest()
        {
            //sumWhereTest(false, false); // fails since message to binary matrix is updated first
            SumWhere(true, false);
            //sumWhereTest(false, true);
            //sumWhereTest(true, true);
        }

        private void SumWhere(bool initialiseBinary, bool initialiseContinuous)
        {
            Rand.Restart(1234);
            int N = 20;
            int D = 10;
            int K = 1;
            var trueZ = new bool[D][];
            var trueA = new Vector[N];
            for (int i = 0; i < D; i++)
            {
                trueZ[i] = new bool[K];
                for (int j = 0; j < K; j++)
                    trueZ[i][j] = Rand.Double() < .5;
            }
            for (int i = 0; i < N; i++)
            {
                trueA[i] = Vector.Zero(K);
                trueA[i].SetToFunction(trueA[i], u => Rand.Normal());
            }
            var data = new double[D, N];
            for (int i = 0; i < D; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    data[i, j] = .1 * Rand.Normal(); //noise
                    for (int k1 = 0; k1 < K; k1++)
                        data[i, j] += trueZ[i][k1] ? trueA[j][k1] : 0.0;
                }
            }

            var d = new Range(D);
            var n = new Range(N);
            var k = new Range(K);
            var z = Variable.Array(Variable.Array<bool>(k), d).Named("z");
            z[d][k] = Variable.Bernoulli(.5).ForEach(d, k);
            if (initialiseBinary)
            {
                var init = trueZ.Select(v => v.Select(b => Bernoulli.FromLogOdds(0)).ToArray()).ToArray();
                z.InitialiseTo(Distribution<bool>.Array(init));
            }
            var a = Variable.Array<Vector>(n).Named("a");
            a[n] = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(K), PositiveDefiniteMatrix.Identity(K)).ForEach(n);
            if (initialiseContinuous)
                a.InitialiseTo(Distribution<Vector>.Array(trueA.Select(v => new VectorGaussian(Vector.Zero(K), PositiveDefiniteMatrix.Identity(K))).ToArray()));
            var x = Variable.Array<double>(d, n).Named("x");
            x[d, n] = Variable.SumWhere(z[d], a[n]);
            var y = Variable.Array<double>(d, n).Named("y");
            y[d, n] = Variable.GaussianFromMeanAndPrecision(x[d, n], 10);
            y.ObservedValue = data;
            if (false)
            {
                for (int i = 0; i < D; i++)
                {
                    for (int j = 0; j < N; j++)
                    {
                        Console.Write(data[i, j].ToString("G2") + " ");
                    }
                    Console.WriteLine();
                }
            }
            var ie = new InferenceEngine(new VariationalMessagePassing());
            var apost = ie.Infer<DistributionArray<VectorGaussian>>(a);
            double xy = 0, x2 = 0, y2 = 0;
            for (int i = 0; i < N; i++)
            {
                xy += apost[i].GetMean()[0] * trueA[i][0];
                x2 += apost[i].GetMean()[0] * apost[i].GetMean()[0];
                y2 += trueA[i][0] * trueA[i][0];
            }
            var zpost = ie.Infer<DistributionRefArray<DistributionStructArray<Bernoulli, bool>, bool[]>>(z);
            for (int i = 0; i < D; i++)
            {
                Console.WriteLine("True Z: {0} Inferred {1}", trueZ[i][0], zpost[i][0]);
                Assert.True(Math.Abs((trueZ[i][0] ? 1.0 : 0.0) - zpost[i][0].GetMean()) < 1e-4);
            }

            Assert.True(xy / Math.Sqrt(x2 * y2) > .99);
        }

        [Fact]
        public void UnderlyingPhenotype()
        {
            int Nint = 100;
            int[] numberObservedDimensions = new int[] { 1, 2, 3 };
            Range N = new Range(Nint);
            var dataMatrix = new int[numberObservedDimensions.Length][,];
            Range D1 = new Range(numberObservedDimensions.Length).Named("D1");
            var varNumObs = Variable.Constant(numberObservedDimensions, D1).Named("numObs");
            Range D2 = new Range(varNumObs[D1]).Named("D2");
            var numCategories = new int[][] { new int[] { 2 }, new int[] { 2, 3 }, new int[] { 1, 3, 4 } };
            // generate data
            for (int i = 0; i < numberObservedDimensions.Length; i++)
            {
                dataMatrix[i] = new int[Nint, numberObservedDimensions[i]];
                for (int j = 0; j < numberObservedDimensions[i]; j++)
                {
                    var p = Dirichlet.Sample(Vector.Constant(numCategories[i][j], 1.0), Vector.Zero(numCategories[i][j]));
                    for (int n = 0; n < Nint; n++)
                    {
                        dataMatrix[i][n, j] = Rand.Sample(p);
                    }
                }
            }
            // model
            var varNumCategories = Variable.Constant(numCategories, D1, D2).Named("numCategories");
            var CategoriesRange = new Range(varNumCategories[D1][D2]).Named("cats");
            var probsTrue = Variable.Array(Variable.Array<Vector>(D2), D1).Named("probsT");
            probsTrue[D1][D2] = Variable.DirichletUniform(CategoriesRange);
            var probsFalse = Variable.Array(Variable.Array<Vector>(D2), D1).Named("probsF");
            probsFalse[D1][D2] = Variable.DirichletUniform(CategoriesRange);
            var data = Variable.Array(Variable.Array<int>(N, D2), D1).Named("data");
            data.ObservedValue = dataMatrix;
            var truePheno = Variable.Array<bool>(N, D1).Named("true");
            truePheno[N, D1] = Variable.Bernoulli(.5).ForEach(N, D1);
            using (Variable.ForEach(N))
            using (Variable.ForEach(D1))
            {
                using (Variable.If(truePheno[N, D1]))
                using (Variable.ForEach(D2))
                    data[D1][N, D2] = Variable.Discrete(probsTrue[D1][D2]);
                using (Variable.IfNot(truePheno[N, D1]))
                using (Variable.ForEach(D2))
                    data[D1][N, D2] = Variable.Discrete(probsFalse[D1][D2]);
            }
            var ie = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(ie.Infer(probsTrue));
        }

        [Fact]
        public void StudentDistribution()
        {
            // generated using: tsamp( 4[mean], .25[variance], 1[dof], 20[nData] )
            VariableArray<double> data = Variable.Constant<double>(new double[]
                {
                    4.12, 3.33,
                    4.37, 7.20, 3.69, 5.74, 4.54, 2.88, 1.19,
                    4.89, 3.84, 4.74, 4.34, 4.25, 4.90, 4.42,
                    4.42, 3.21, 4.00, 3.76
                });
            data.Named("data");
            Range item = data.Range;
            Variable<double> mu
                = Variable<double>.GaussianFromMeanAndPrecision(0, 0.01).Named("mu");
            Variable<double> lambda
                = Variable<double>.GammaFromShapeAndScale(1, 1).Named("lambda");
            double dof = 1;
            using (Variable.ForEach(item))
            {
                Variable<double> s = Variable<double>.GammaFromShapeAndScale(dof / 2, 2 / dof).Named("s");
                data[item] = Variable.GaussianFromMeanAndPrecision(mu, s * lambda);
            }
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine("mu=" + ie.Infer(mu));
            Gamma lambdaPost = ie.Infer<Gamma>(lambda);
            Console.WriteLine("lambda=" + lambdaPost.GetMean() + " +/- " + Math.Sqrt(lambdaPost.GetVariance()));
        }

        [Fact]
        public void DiscreteArray()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<Vector> pi = Variable.Dirichlet(new double[] { 1, 1, 1, 1 }).Named("pi");
            Range item = new Range(4).Named("item");
            VariableArray<int> x = Variable.Constant<int>(new int[] { 2, 1, 2, 3 }, item);
            Variable.ConstrainEqual(x[item], Variable.Discrete(pi).ForEach(item));
            block.CloseBlock();

            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            double[] VibesSuffStats = new double[] { -2.59285714209419, -1.59285714206543, -1.09285714194003, -1.59285714206543 };

            VmpTests.TestDirichletMoments(ie, pi, VibesSuffStats);
            VmpTests.TestEvidence(ie, evidence, -6.0402546);
        }


        [Fact]
        public void ProductArray()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Range item = new Range(2).Named("item");

            VariableArray<double> a = Variable.Array<double>(item).Named("a");
            a[item] = Variable.GaussianFromMeanAndPrecision(2, 10).ForEach(item);

            Variable<double> b = Variable.GaussianFromMeanAndPrecision(5, 10).Named("b");
            VariableArray<double> abNoisy = Variable.Constant<double>(new double[] { 7, 19 }, item).Named("abConstant");
            Variable.ConstrainEqual(abNoisy[item], Variable.GaussianFromMeanAndPrecision((a[item] * b).Named("ab"), 10).Named("abNoisy"));
            block.CloseBlock();

            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.NumberOfIterations = 1000;

            double[][] aVibes = new double[2][];
            aVibes[0] = new double[] { 1.26963041938737, 1.61503541347205 };
            aVibes[1] = new double[] { 3.34074502502446, 11.16365133386395 };

            VmpTests.TestGaussianArrayMoments(ie, a, aVibes);
            VmpTests.TestGaussianMoments(ie, b, 5.61458213179377, 31.53079010036498);
            VmpTests.TestEvidence(ie, evidence, -18.250566);
        }

        [Fact]
        public void sum_VarOutsidePlate()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            double[] data = { 7, 11 };

            Variable<double> b = Variable.GaussianFromMeanAndPrecision(1, 1);
            Range nData = new Range(2).Named("K");
            VariableArray<double> xNoisy = Variable.Constant<double>(data, nData);
            VariableArray<double> y = Variable.Array<double>(nData).Named("y");
            y[nData] = Variable.GaussianFromMeanAndPrecision(5, 1).ForEach(nData);
            xNoisy[nData] = Variable.GaussianFromMeanAndPrecision(b + y[nData], 10);
            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.NumberOfIterations = 70;

            double[][] yVibes = new double[2][];
            yVibes[0] = new double[] { 4.14957410613514, 17.30987435321637 };
            yVibes[1] = new double[] { 7.78593774249878, 60.71173562097611 };
            VmpTests.TestGaussianMoments(ie, b, 2.93547054826956, 8.66460638737703);
            VmpTests.TestGaussianArrayMoments(ie, y, yVibes);
            VmpTests.TestEvidence(ie, evidence, -9.995038);
        }

        [Fact]
        public void Sum3ElementArray()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            int nItems = 3;
            Range item = new Range(nItems).Named("item");
            VariableArray<double> a = Variable.Array<double>(item).Named("a");
            VariableArray<double> c = Variable.Constant<double>(new double[] { 1, 2, 3 }, item);
            a[item] = Variable.GaussianFromMeanAndVariance(c[item], 1);
            Variable<double> x = Variable.Sum(a).Named("x");

            Variable<double> xNoisy = Variable.Observed<double>(5).Named("xNoisy");
            Variable.ConstrainEqual<double>(xNoisy, Variable.GaussianFromMeanAndPrecision(x, 10));
            block.CloseBlock();

            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.NumberOfIterations = 500;

            double[][] vibesResult = new double[nItems][];
            vibesResult[0] = new double[] { 0.67740400765708, 0.54978528049896 };
            vibesResult[1] = new double[] { 1.67743250780634, 2.90468890915456 };
            vibesResult[2] = new double[] { 2.67742308377088, 7.25950346041824 };

            VmpTests.TestGaussianArrayMoments(ie, a, vibesResult);
            VmpTests.TestGaussianMoments(ie, x, 5.03225959923429, 25.59636394681296);
            VmpTests.TestEvidence(ie, evidence, -3.5257792);

            //bound: -3.5257792
            // Console.WriteLine(ie.Infer(a));
        }

        [Fact]
        public void SumTest()
        {
            Gaussian[] priors = new Gaussian[]
                {
                    new Gaussian(1.2, 3.4),
                    new Gaussian(5.6, 7.8)
                };
            Range item = new Range(2).Named("item");
            VariableArray<Gaussian> priorsVar = Variable.Constant(priors, item).Named("priors");
            VariableArray<double> x = Variable.Array<double>(item).Named("x");
            x[item] = Variable.Random<double, Gaussian>(priorsVar[item]);
            Variable<double> sum = Variable.Sum(x).Named("sum");
            Variable.ConstrainEqualRandom(sum, new Gaussian(2.1, 4.3));
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Gaussian[] xExpectedArray = new Gaussian[]
                {
                    new Gaussian(0.169032258064516, 1.898701298701299),
                    new Gaussian(3.234838709677419, 2.771900826446281)
                };
            object xActual = engine.Infer(x);
            Console.WriteLine(StringUtil.JoinColumns("x = ", xActual));
            IDistribution<double[]> xExpected = Distribution<double>.Array(xExpectedArray);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
        }

        // This is a test of vmp.UseDerivMessages functionality
        [Fact]
        [Trait("Category", "OpenBug")]
        public void SumOfProductTest2()
        {
            Gaussian[] priors;
            for (int trial = 0; trial < 3; trial++)
            {
                if (trial == 1)
                    priors = Util.ArrayInit(4, i => new Gaussian(i, i + 1));
                else if (trial == 2)
                    priors = Util.ArrayInit(4, i => new Gaussian(1, 20));
                else
                {
                    // should converge in 1 iteration
                    priors = new Gaussian[4];
                    priors[0] = new Gaussian(1, 1);
                    priors[1] = Gaussian.PointMass(10);
                    priors[2] = new Gaussian(1, 1);
                    priors[3] = Gaussian.PointMass(10);
                }
                Gaussian sumLike = new Gaussian(2.1, 4.3);
                var xExpected = SumOfProductPosterior(priors, sumLike);
                var xActual = SumOfProductPosterior2(priors, sumLike);
                Console.WriteLine(StringUtil.JoinColumns("x = ", xActual, " should be ", xExpected));
                Console.WriteLine("error = {0}", xExpected.MaxDiff(xActual));
                Assert.True(xExpected.MaxDiff(xActual) < 1e-4);
            }
        }

        public IDistribution<double[]> SumOfProductPosterior(Gaussian[] priors, Gaussian sumLike)
        {
            Variable<double> a = Variable.Random(priors[0]).Named("a");
            Variable<double> b = Variable.Random(priors[1]).Named("b");
            Variable<double> c = Variable.Random(priors[2]).Named("c");
            Variable<double> d = Variable.Random(priors[3]).Named("d");
            Variable<double> ab = (a * b).Named("ab");
            Variable<double> cd = (c * d).Named("cd");
            Variable<double> sum = (ab + cd).Named("sum");
            Variable.ConstrainEqualRandom(sum, sumLike);
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            //engine.NumberOfIterations = 500;
            Gaussian[] result = new Gaussian[4];
            result[0] = engine.Infer<Gaussian>(a);
            result[1] = engine.Infer<Gaussian>(b);
            result[2] = engine.Infer<Gaussian>(c);
            result[3] = engine.Infer<Gaussian>(d);
            return Distribution<double>.Array(result);
        }

        public IDistribution<double[]> SumOfProductPosterior2(Gaussian[] priors, Gaussian sumLike)
        {
            Range group = new Range(2).Named("group");
            VariableArray<double> a = Variable.Array<double>(group).Named("a");
            a.SetTo(Variable<double>.Random(Distribution<double>.Array(new Gaussian[] { priors[0], priors[2] })));
            VariableArray<double> b = Variable.Array<double>(group).Named("b");
            b.SetTo(Variable<double>.Random(Distribution<double>.Array(new Gaussian[] { priors[1], priors[3] })));
            VariableArray<double> products = Variable.Array<double>(group).Named("products");
            products[group] = a[group] * b[group];
            Variable<double> sum = Variable.Sum(products).Named("sum");
            Variable.ConstrainEqualRandom(sum, sumLike);
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing()
            {
                UseDerivMessages = true
            });
            //engine.NumberOfIterations = 25;
            Gaussian[] aPost = engine.Infer<Gaussian[]>(a);
            Gaussian[] bPost = engine.Infer<Gaussian[]>(b);
            Gaussian[] result = new Gaussian[4];
            result[0] = aPost[0];
            result[1] = bPost[0];
            result[2] = aPost[1];
            result[3] = bPost[1];
            return Distribution<double>.Array(result);
        }

        // This is a test of vmp.UseDerivMessages functionality
        // This test only passes with vmp.UseDerivMessages = true
        [Fact]
        [Trait("Category", "OpenBug")]
        public void SumTest2()
        {
            // Test that sum of sums is equal to single sum
            //Gaussian[] priors = Util.ArrayInit(4, i => new Gaussian(i, i+1));
            Gaussian[] priors = Util.ArrayInit(4, i => new Gaussian(0, 20));
            Gaussian sumLike = new Gaussian(2.1, 4.3);
            var xExpected = SumArrayPosterior(priors, sumLike);
            var xActual = SumOfSumArrayPosterior(priors, sumLike);
            Console.WriteLine(StringUtil.JoinColumns("x = ", xActual, " should be ", xExpected));
            Assert.True(xExpected.MaxDiff(xActual) < 1e-4);
        }

        public IDistribution<double[]> SumArrayPosterior(Gaussian[] priors, Gaussian sumLike)
        {
            Range item = new Range(priors.Length).Named("item");
            VariableArray<Gaussian> priorsVar = Variable.Constant(priors, item).Named("priors");
            VariableArray<double> x = Variable.Array<double>(item).Named("x");
            x[item] = Variable.Random<double, Gaussian>(priorsVar[item]);
            Variable<double> sum = Variable.Sum(x).Named("sum");
            Variable.ConstrainEqualRandom(sum, sumLike);
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.NumberOfIterations = 2;
            return engine.Infer<IDistribution<double[]>>(x);
        }

        public IDistribution<double[]> SumOfSumArrayPosterior(Gaussian[] priors, Gaussian sumLike)
        {
            Range item = new Range(priors.Length).Named("item");
            VariableArray<Gaussian> priorsVar = Variable.Constant(priors, item).Named("priors");
            VariableArray<double> x = Variable.Array<double>(item).Named("x");
            x[item] = Variable.Random<double, Gaussian>(priorsVar[item]);
            Range piece = new Range(2).Named("piece");
            int[] lengths = new int[] { priors.Length / 2, priors.Length - (priors.Length / 2) };
            VariableArray<int> lengthsVar = Variable.Constant(lengths, piece).Named("lengths");
            Range item2 = new Range(lengthsVar[piece]).Named("item2");
            int[][] indices = new int[2][];
            indices[0] = Util.ArrayInit(lengths[0], i => i);
            indices[1] = Util.ArrayInit(lengths[1], i => i + lengths[0]);
            var indicesVar = Variable.Constant(indices, piece, item2).Named("indices");
            var x2 = Variable.Array(Variable.Array<double>(item2), piece).Named("x2");
            x2[piece] = Variable.Subarray(x, indicesVar[piece]);
            var sums = Variable.Array<double>(piece).Named("sums");
            sums[piece] = Variable.Sum(x2[piece]);
            Variable<double> sum = Variable.Sum(sums).Named("sum");
            Variable.ConstrainEqualRandom(sum, sumLike);
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing()
            {
                UseDerivMessages = true
            });
            // this test is only useful with a small number of iterations, otherwise it trivially passes
            engine.NumberOfIterations = 4;
            return engine.Infer<IDistribution<double[]>>(x);
        }

        [Fact]
        public void MatrixMultiplyMustBeStochasticError2()

        {
            Assert.Throws<CompilationFailedException>(() =>
            {

                Variable<bool> b = Variable.Bernoulli(0.1).Named("b");
                Range r1 = new Range(2).Named("r1");
                Range r2 = new Range(2).Named("r2");
                Range r3 = new Range(2).Named("r3");
                VariableArray2D<double> w = Variable.Array<double>(r1, r2).Named("w");
                VariableArray2D<double> x = Variable.Array<double>(r2, r3).Named("x");
                x.ObservedValue = new double[,] { { 1, 2 }, { 3, 4 } };
                using (Variable.If(b))
                {
                    w[r1, r2] = Variable.GaussianFromMeanAndVariance(1, 2).ForEach(r1, r2);
                }
                using (Variable.IfNot(b))
                {
                    w[r1, r2] = Variable.GaussianFromMeanAndVariance(3, 4).ForEach(r1, r2);
                }
                VariableArray2D<double> y = Variable.MatrixMultiply(w, x).Named("y");
                VariationalMessagePassing vmp = new VariationalMessagePassing();
                //vmp.UseGateExitRandom = true;  // will succeed if this is set
                InferenceEngine engine = new InferenceEngine(vmp);
                Console.WriteLine(engine.Infer(y));

            });
        }

        // Fails because w is used by Gate.Exit and therefore has 2 uses, making it invalid as an argument to MatrixMultiply
        [Fact]
        [Trait("Category", "OpenBug")]
        public void MatrixMultiplyGateExit()
        {
            Variable<bool> b = Variable.Bernoulli(0.1).Named("b");
            Range r1 = new Range(2).Named("r1");
            Range r2 = new Range(2).Named("r2");
            Range r3 = new Range(2).Named("r3");
            VariableArray2D<double> w = Variable.Array<double>(r1, r2).Named("w");
            VariableArray2D<double> x = Variable.Array<double>(r2, r3).Named("x");
            x.ObservedValue = new double[,] { { 1, 2 }, { 3, 4 } };
            VariableArray2D<double> y = Variable.Array<double>(r1, r3).Named("y");
            using (Variable.If(b))
            {
                w[r1, r2] = Variable.GaussianFromMeanAndVariance(1, 2).ForEach(r1, r2);
                y.SetTo(Variable.MatrixMultiply(w, x));
            }
            using (Variable.IfNot(b))
            {
                w[r1, r2] = Variable.GaussianFromMeanAndVariance(3, 4).ForEach(r1, r2);
                y.SetTo(Variable.MatrixMultiply(w, x));
            }
            VariationalMessagePassing vmp = new VariationalMessagePassing();
            //vmp.UseGateExitRandom = true;  // will succeed if this is set
            InferenceEngine engine = new InferenceEngine(vmp);
            Console.WriteLine(engine.Infer(y));
        }

        [Fact]
        public void Regression()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> m = Variable.GaussianFromMeanAndPrecision(2, 1).Named("m");
            Variable<double> prec = Variable.GammaFromShapeAndScale(1, 1).Named("prec");
            Range dim = new Range(2).Named("dim");
            VariableArray<double> w = Variable.Array<double>(dim).Named("w");
            w[dim] = Variable.GaussianFromMeanAndPrecision(m, prec).ForEach(dim);
            VariableArray<double> x = Variable.Array<double>(dim).Named("x");
            x[dim] = Variable.GaussianFromMeanAndPrecision(2.0, 0.5).ForEach(dim);
            VariableArray<double> ytmp = Variable.Array<double>(dim).Named("ytmp");
            ytmp[dim] = w[dim] * x[dim];
            Variable<double> y = Variable.Sum(ytmp).Named("y");
            Variable<double> yNoisy = Variable.GaussianFromMeanAndPrecision(y, 10).Named("yNoisy");
            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing()
            {
                UseDerivMessages = true,
                DefaultNumberOfIterations = 60
            });
            x.ObservedValue = new double[] { 1, 2 };
            yNoisy.ObservedValue = 7.9;

            VmpTests.TestGaussianMoments(ie, m, 2.45865638280071, 6.29467433998947);
            VmpTests.TestGammaMoments(ie, prec, 1.50253816663965, 0.13679294373106);
            double[][] wVibesResult = new double[2][];
            wVibesResult[0] = new double[] { 2.56041992634439, 6.64268753447734 };
            wVibesResult[1] = new double[] { 2.66214626326932, 7.11111763903348 };
            VmpTests.TestGaussianArrayMoments(ie, w, wVibesResult);
            VmpTests.TestGaussianMoments(ie, y, 7.88471240184059, 62.35200664105416);
            VmpTests.TestEvidence(ie, evidence, -6.546374);
        }

        [Fact]
        [Trait("Category", "BadTest")] // this test is bad because it passes or fails at random
        public void SoftmaxRegression()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Range dim = new Range(2).Named("dim");
            VariableArray<double> w = Variable.Array<double>(dim).Named("w");
            w[dim] = Variable.GaussianFromMeanAndPrecision(1.2, 0.4).ForEach(dim);
            Range item = new Range(4).Named("item");
            var x = Variable.Array(Variable.Array<double>(dim), item).Named("x");
            x[item][dim] = Variable.GaussianFromMeanAndPrecision(2.0, 0.5).ForEach(item, dim);
            var wx = Variable.Array(Variable.Array<double>(dim), item).Named("wx");
            wx[item][dim] = w[dim] * x[item][dim];
            var sum = Variable.Array<double>(item).Named("sum");
            sum[item] = Variable.Sum(wx[item]);
            Range outcome = new Range(2).Named("outcome");
            var y = Variable.Array<int>(item).Named("y");
            var probs = Variable.Array(Variable.Array<double>(outcome), item).Named("probs");
            using (Variable.ForEach(item))
            {
                //probs[item][0] = Variable.GaussianFromMeanAndVariance(0, 0); 
                probs[item][0] = Variable<double>.Random(Variable.Observed(Gaussian.PointMass(0)));
                //probs[item][0] = 0; 
                probs[item][1] = Variable.Copy(sum[item]);

                y[item] = Variable.DiscreteFromLogProbs(probs[item]);
            }
            block.CloseBlock();
            Rand.Restart(1);
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.NumberOfIterations = 1000;
            ie.ShowProgress = false;
            ie.Compiler.GivePriorityTo(typeof(LogisticOp_SJ99));
            ie.ModelName = "SoftmaxRegression";
            double oldStep = LogisticOp_SJ99.global_step;
            try
            {
                LogisticOp_SJ99.global_step = .3;
                ie.Compiler.GivePriorityTo(typeof(SoftmaxOp_KM11_Sparse));

                x.ObservedValue = new double[][]
                    {
                        new double[] {1, 2},
                        new double[] {-3, 4},
                        new double[] {5, -6},
                        new double[] {-7, -8}
                    };
                y.ObservedValue = new int[] { 1, 0, 1, 0 };

                // exact mean and variance of the posterior
                Gaussian[] wExact = new Gaussian[]
                    {
                        new Gaussian(2.61298, 1.2818),
                        new Gaussian(0.553222, 0.876026)
                    };
                // most of the inaccuracy here comes from the factorized approximation on w.

                var ca = ie.GetCompiledInferenceAlgorithm(w, evidence);
                for (int i = 0; i < ie.NumberOfIterations; i++)
                {
                    ca.Update(1);
                    var wActual = ca.Marginal<Gaussian[]>(w.NameInGeneratedCode);
                    //double evActual = ie.Infer<Bernoulli>(evidence).LogOdds;
                    //Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                    //Console.WriteLine(StringUtil.JoinColumns("w = ", wActual, " should be ", wExpected));
                    Console.WriteLine(System.Linq.Enumerable.Range(0, 4).Select(o => "{" + o + "} ").Aggregate((p, q) => p + q), wActual[0].GetMean(), wActual[0].GetVariance(),
                                      wActual[1].GetMean(), wActual[1].GetVariance());
                }

                var wActualFinal = ca.Marginal(w.NameInGeneratedCode);
                double evActual = ca.Marginal<Bernoulli>(evidence.NameInGeneratedCode).LogOdds;

                IDistribution<double[]> wExpected;
                double evExpected;
                LogisticRegression(ie, out wExpected, out evExpected);

                Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-4);
                Console.WriteLine(StringUtil.JoinColumns("w = ", wActualFinal, " should be ", wExpected));
                Assert.True(wExpected.MaxDiff(wActualFinal) < 1e-6);
            }
            finally
            {
                LogisticOp_SJ99.global_step = oldStep;
            }
        }

        [Fact]
        [Trait("Category", "BadTest")] // this test is bad because it passes or fails at random
        public void SoftmaxRegressionAlternative()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Range dim = new Range(2).Named("dim");
            Range outcome = new Range(2).Named("outcome");
            var wPrior = new Gaussian[2][]
                {
                    new Gaussian[] {Gaussian.PointMass(0), Gaussian.PointMass(0)},
                    new Gaussian[] {Gaussian.FromMeanAndPrecision(1.2, 0.4), Gaussian.FromMeanAndPrecision(1.2, 0.4)}
                };
            var wPriorVar = Variable.Observed(wPrior, outcome, dim).Named("wPriorVar");
            var w = Variable.Array(Variable.Array<double>(dim), outcome).Named("w");
            w[outcome][dim] = Variable<double>.Random(wPriorVar[outcome][dim]);
            Range item = new Range(4).Named("item");
            var x = Variable.Array(Variable.Array<double>(dim), item).Named("x");
            x[item][dim] = Variable.GaussianFromMeanAndPrecision(2.0, 0.5).ForEach(item, dim);
            var y = Variable.Array<int>(item).Named("y");
            var probs = Variable.Array(Variable.Array<double>(outcome), item).Named("probs");
            using (Variable.ForEach(item))
            {
                var wx = Variable.Array(Variable.Array<double>(dim), outcome).Named("wx");
                wx[outcome][dim] = w[outcome][dim] * x[item][dim];
                probs[item][outcome] = Variable.Sum(wx[outcome]);
                y[item] = Variable.DiscreteFromLogProbs(probs[item]);
            }
            block.CloseBlock();
            Rand.Restart(1);
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.NumberOfIterations = 1000;
            ie.ShowProgress = false;
            // does not work consistently with ParallelLoops=true
            ie.Compiler.UseParallelForLoops = false;
            ie.Compiler.GivePriorityTo(typeof(LogisticOp_SJ99));
            ie.Compiler.GivePriorityTo(typeof(SoftmaxOp_KM11_Sparse));
            ie.ModelName = "SoftmaxRegressionAlternative";
            x.ObservedValue = new double[][]
                {
                    new double[] {1, 2},
                    new double[] {-3, 4},
                    new double[] {5, -6},
                    new double[] {-7, -8}
                };
            y.ObservedValue = new int[] { 1, 0, 1, 0 };

            // exact mean and variance of the posterior
            Gaussian[] wExact = new Gaussian[]
                {
                    new Gaussian(2.61298, 1.2818),
                    new Gaussian(0.553222, 0.876026)
                };
            // most of the inaccuracy here comes from the factorized approximation on w.

            IDistribution<double[]> wExpected;
            double evExpected;
            LogisticRegression(ie, out wExpected, out evExpected);
            var ca = ie.GetCompiledInferenceAlgorithm(w, evidence, probs);
            for (int i = 0; i < ie.NumberOfIterations; i++)
            {
                ca.Update(1);
                if (false)
                {
                    var wActual = new DistributionStructArray<Gaussian, double>(ca.Marginal<Gaussian[][]>(w.NameInGeneratedCode)[1]);
                    Console.WriteLine(System.Linq.Enumerable.Range(0, 4).Select(o => "{" + o + "} ").Aggregate((p, q) => p + q), wActual[0].GetMean(), wActual[0].GetVariance(),
                                      wActual[1].GetMean(), wActual[1].GetVariance());
                }
                if (false)
                {
                    var probsPost = ca.Marginal<Gaussian[][]>(probs.NameInGeneratedCode);
                    for (int j = 0; j < 4; j++)
                    {
                        Console.Write(probsPost[j][1].GetMean() / Math.Sqrt(probsPost[j][1].GetVariance()) + " ");
                        //Console.Write(probsPost[j][1].GetMean() + " ");
                        //Console.Write(probsPost[j][1].GetVariance() + " ");
                    }
                    Console.WriteLine();
                }
            }
            var wActualFinal = new DistributionStructArray<Gaussian, double>(ca.Marginal<Gaussian[][]>(w.NameInGeneratedCode)[1]);
            double evActual = ca.Marginal<Bernoulli>(evidence.NameInGeneratedCode).LogOdds;
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-4);
            Console.WriteLine(StringUtil.JoinColumns("w = ", wActualFinal, " should be ", wExpected));
            Assert.True(wExpected.MaxDiff(wActualFinal) < 1e-2);
        }

        private void LogisticRegression(InferenceEngine ie, out IDistribution<double[]> wPost, out double logZ)
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Range dim = new Range(2).Named("dim");
            VariableArray<double> w = Variable.Array<double>(dim).Named("w");
            w[dim] = Variable.GaussianFromMeanAndPrecision(1.2, 0.4).ForEach(dim);
            Range item = new Range(4).Named("item");
            var x = Variable.Array(Variable.Array<double>(dim), item).Named("x");
            x[item][dim] = Variable.GaussianFromMeanAndPrecision(2.0, 0.5).ForEach(item, dim);
            var wx = Variable.Array(Variable.Array<double>(dim), item).Named("wx");
            wx[item][dim] = w[dim] * x[item][dim];
            var sum = Variable.Array<double>(item).Named("sum");
            sum[item] = Variable.Sum(wx[item]);
            var y = Variable.Array<bool>(item).Named("y");
            y[item] = Variable.BernoulliFromLogOdds(sum[item]);
            block.CloseBlock();

            x.ObservedValue = new double[][]
                {
                    new double[] {1, 2},
                    new double[] {-3, 4},
                    new double[] {5, -6},
                    new double[] {-7, -8}
                };
            y.ObservedValue = new bool[] { true, false, true, false };

            wPost = ie.Infer<IDistribution<double[]>>(w);
            logZ = ie.Infer<Bernoulli>(evidence).LogOdds;
        }

        // This test illustrates a convergence issue with NCVMP and parallel schedules.
        // If you remove the damping from BernoulliFromLogOddsOp, the marginal of w will never
        // converge, but continually oscillate.
        [Fact]
        public void LogisticRegressionQuadrature()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Range dim = new Range(2).Named("dim");
            VariableArray<double> w = Variable.Array<double>(dim).Named("w");
            w[dim] = Variable.GaussianFromMeanAndPrecision(1.2, 0.4).ForEach(dim);
            Range item = new Range(4).Named("item");
            var x = Variable.Array(Variable.Array<double>(dim), item).Named("x");
            x[item][dim] = Variable.GaussianFromMeanAndPrecision(2.0, 0.5).ForEach(item, dim);
            var wx = Variable.Array(Variable.Array<double>(dim), item).Named("wx");
            wx[item][dim] = w[dim] * x[item][dim];
            var sum = Variable.Array<double>(item).Named("sum");
            sum[item] = Variable.Sum(wx[item]);
            var y = Variable.Array<bool>(item).Named("y");
            y[item] = Variable.BernoulliFromLogOdds(sum[item]);
            //y[item] = Variable.Bernoulli(Variable.Logistic(sum[item]));
            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.Compiler.GivePriorityTo(typeof(LogisticOp));
            ie.NumberOfIterations = 100;
            ie.ShowProgress = false;
            //ie.Compiler.UnrollLoops = true;
            ie.Compiler.FreeMemory = false;
            ie.ModelName = "LogisticRegressionQuadrature";

            x.ObservedValue = new double[][]
                {
                    new double[] {1, 2},
                    new double[] {-3, 4},
                    new double[] {5, -6},
                    new double[] {-7, -8}
                };
            y.ObservedValue = new bool[] { true, false, true, false };

            Gaussian[] wExpectedArray = new Gaussian[] { new Gaussian(2.659, 0.8202), new Gaussian(0.37, 0.552) };
            IDistribution<double[]> wExpected = Distribution<double>.Array(wExpectedArray);

            var ca = ie.GetCompiledInferenceAlgorithm(w);
            // Uncomment this to use a hand-written compiled algorithm that uses sequential updates.
            // This should converge without damping.
            //var ca = new LogisticRegressionQuadrature_VMP();
            //ca.y = y.ObservedValue;
            //ca.x = x.ObservedValue;
            object wActual;
            if (true)
            {
                ca.Execute(100);
                wActual = ca.Marginal(w.NameInGeneratedCode);
                Console.WriteLine(StringUtil.JoinColumns("w = ", wActual, " should be ", wExpected));
                Assert.True(wExpected.MaxDiff(wActual) < 3e-4);
            }
            ca.Reset();
            for (int i = 0; i < ie.NumberOfIterations; i++)
            {
                ca.Update(1);
                IList<Gaussian> wPost = ca.Marginal<IList<Gaussian>>(w.NameInGeneratedCode);
                Console.WriteLine(StringUtil.CollectionToString(wPost.Select(g => g.GetMean()), " "));
            }
            wActual = ca.Marginal(w.NameInGeneratedCode);
            Console.WriteLine(StringUtil.JoinColumns("w = ", wActual, " should be ", wExpected));
            Assert.True(wExpected.MaxDiff(wActual) < 3e-4);
        }

        // Sometimes fails with ParallelLoops because LogisticOp_SJ99 has randomness
        [Trait("Category", "OpenBug")]
        [Fact]
        public void LogisticRegression_SJ99()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Range dim = new Range(2).Named("dim");
            VariableArray<double> w = Variable.Array<double>(dim).Named("w");
            w[dim] = Variable.GaussianFromMeanAndPrecision(1.2, 0.4).ForEach(dim);
            Range item = new Range(4).Named("item");
            var x = Variable.Array(Variable.Array<double>(dim), item).Named("x");
            x[item][dim] = Variable.GaussianFromMeanAndPrecision(2.0, 0.5).ForEach(item, dim);
            var wx = Variable.Array(Variable.Array<double>(dim), item).Named("wx");
            wx[item][dim] = w[dim] * x[item][dim];
            var sum = Variable.Array<double>(item).Named("sum");
            sum[item] = Variable.Sum(wx[item]);
            var y = Variable.Array<bool>(item).Named("y");
            y[item] = Variable.BernoulliFromLogOdds(sum[item]);
            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.Compiler.GivePriorityTo(typeof(LogisticOp_SJ99));
            ie.NumberOfIterations = 1000;
            ie.ShowProgress = false;
            //ie.Compiler.UnrollLoops = true;
            ie.ModelName = "LogisticRegression_SJ99";
            ie.Compiler.FreeMemory = false;

            x.ObservedValue = new double[][]
                {
                    new double[] {1, 2},
                    new double[] {-3, 4},
                    new double[] {5, -6},
                    new double[] {-7, -8}
                };
            y.ObservedValue = new bool[] { true, false, true, false };

            Gaussian[] wExpectedArray = new Gaussian[] { new Gaussian(2.61, 0.366), new Gaussian(0.3703, 0.2519) };
            IDistribution<double[]> wExpected = Distribution<double>.Array(wExpectedArray);

            Rand.Restart(12347);
            var ca = ie.GetCompiledInferenceAlgorithm(w);
            //var ca = new LogisticRegression_SJ99_VMP();
            //ca.y = y.ObservedValue;
            //ca.x = x.ObservedValue;
            object wActual;
            if (true)
            {
                ca.Execute(1000);
                wActual = ca.Marginal(w.NameInGeneratedCode);
                Console.WriteLine(StringUtil.JoinColumns("w = ", wActual, " should be ", wExpected));
                Console.WriteLine("error = {0}", wExpected.MaxDiff(wActual));
                //Assert.True(wExpected.MaxDiff(wActual) < 2e-3);
            }
            if (false)
            {
                ca.Execute(999);
                wActual = ca.Marginal(w.NameInGeneratedCode);
                Console.WriteLine(StringUtil.JoinColumns("w = ", wActual));
            }
            if (false)
            {
                ca.Reset();
                for (int i = 0; i < ie.NumberOfIterations; i++)
                {
                    ca.Update(1);
                    IList<Gaussian> wPost = ca.Marginal<IList<Gaussian>>(w.NameInGeneratedCode);
                    Console.WriteLine(StringUtil.CollectionToString(wPost.Select(g => g.GetMean()), " "));
                }
            }
        }

        /// <summary>
        /// Unrolled version of LogisticRegression
        /// </summary>
        [Fact]
        public void LogisticRegressionUnrolled()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> w1 = Variable.GaussianFromMeanAndPrecision(1.2, 0.4).Named("w1");
            Variable<double> w2 = Variable.GaussianFromMeanAndPrecision(1.2, 0.4).Named("w2");
            Range item = new Range(4).Named("item");
            var x1 = Variable.Array<double>(item).Named("x1");
            x1[item] = Variable.GaussianFromMeanAndPrecision(2.0, 0.5).ForEach(item);
            var x2 = Variable.Array<double>(item).Named("x2");
            x2[item] = Variable.GaussianFromMeanAndPrecision(2.0, 0.5).ForEach(item);
            var wx = Variable.Array<double>(item).Named("wx");
            wx[item] = (w1 * x1[item]).Named("wx1") + (w2 * x2[item]).Named("wx2");
            var y = Variable.Array<bool>(item).Named("y");
            y[item] = Variable.BernoulliFromLogOdds(wx[item]);
            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.NumberOfIterations = 500;

            x1.ObservedValue = new double[] { 1, -3, 5, -7 };
            x2.ObservedValue = new double[] { 2, 4, -6, -8 };
            y.ObservedValue = new bool[] { true, false, true, false };

            //ie.Compiler.GivePriorityTo(typeof(BernoulliFromLogOddsOp_JJ96));
            ie.Compiler.GivePriorityTo(typeof(LogisticOp_JJ96));
            VmpTests.TestGaussianMoments(ie, w1, 2.390537703233513, 5.975568523864546);
            VmpTests.TestGaussianMoments(ie, w2, 0.573187406316556, 0.500807124106724);
            VmpTests.TestEvidence(ie, evidence, -84.0191);
        }

        [Fact]
        public void LogisticRegressionTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Range dim = new Range(2).Named("dim");
            VariableArray<double> w = Variable.Array<double>(dim).Named("w");
            w[dim] = Variable.GaussianFromMeanAndPrecision(1.2, 0.4).ForEach(dim);
            Range item = new Range(4).Named("item");
            var x = Variable.Array(Variable.Array<double>(dim), item).Named("x");
            x[item][dim] = Variable.GaussianFromMeanAndPrecision(2.0, 0.5).ForEach(item, dim);
            var wx = Variable.Array(Variable.Array<double>(dim), item).Named("wx");
            wx[item][dim] = w[dim] * x[item][dim];
            var sum = Variable.Array<double>(item).Named("sum");
            sum[item] = Variable.Sum(wx[item]);
            var y = Variable.Array<bool>(item).Named("y");
            y[item] = Variable.BernoulliFromLogOdds(sum[item]);
            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.NumberOfIterations = 500;

            x.ObservedValue = new double[][]
                {
                    new double[] {1, 2},
                    new double[] {-3, 4},
                    new double[] {5, -6},
                    new double[] {-7, -8}
                };
            y.ObservedValue = new bool[] { true, false, true, false };

            // exact mean and variance of the posterior
            Gaussian[] wExact = new Gaussian[]
                {
                    new Gaussian(2.61298, 1.2818),
                    new Gaussian(0.553222, 0.876026)
                };
            // most of the inaccuracy here comes from the factorized approximation on w.

            ie.Compiler.GivePriorityTo(typeof(LogisticOp_JJ96));
            double[][] wVibesResult = new double[2][];
            wVibesResult[0] = new double[] { 2.390537703233513, 5.975568523864546 };
            wVibesResult[1] = new double[] { 0.573187406316556, 0.500807124106724 };
            VmpTests.TestGaussianArrayMoments(ie, w, wVibesResult);
            VmpTests.TestEvidence(ie, evidence, -84.0191);
        }

        //[Fact]
        internal void LogisticRegressionKnowles()
        {
            int P = 8, N = 200;
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Range dim = new Range(P).Named("dim");
            VariableArray<double> w = Variable.Array<double>(dim).Named("w");
            w[dim] = Variable.GaussianFromMeanAndPrecision(1.2, 0.4).ForEach(dim);
            Range item = new Range(N).Named("item");
            var x = Variable.Array(Variable.Array<double>(dim), item).Named("x");
            x[item][dim] = Variable.GaussianFromMeanAndPrecision(2.0, 0.5).ForEach(item, dim);
            var wx = Variable.Array(Variable.Array<double>(dim), item).Named("wx");
            wx[item][dim] = w[dim] * x[item][dim];
            var sum = Variable.Array<double>(item).Named("sum");
            sum[item] = Variable.Sum(wx[item]);
            var y = Variable.Array<bool>(item).Named("y");
            y[item] = Variable.BernoulliFromLogOdds(sum[item]);
            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            //ie.Compiler.GivePriorityTo(typeof(LogisticOp_JJ96));
            ie.Compiler.GivePriorityTo(typeof(LogisticOp_SJ99));
            //ie.NumberOfIterations = 500;

            Rand.Restart(2);
            double[][] xTrue = new double[N][];
            double[] wTrue = new double[P];
            bool[] yObs = new bool[N];
            double[] g = new double[N];
            for (int i = 0; i < P; i++)
                wTrue[i] = Rand.Normal(0, 1);
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
                yObs[j] = (Rand.Binomial(1, MMath.Logistic(g[j])) == 1);
            }
            y.ObservedValue = yObs;
            x.ObservedValue = xTrue;
            DistributionArray<Gaussian> G = ie.Infer<DistributionArray<Gaussian>>(w);
            double G_mean, G_var;

            double logl = 0.0;
            for (int i = 0; i < P; i++)
            {
                G[i].GetMeanAndVariance(out G_mean, out G_var);
                logl += G[i].GetLogProb(wTrue[i]);
                Console.WriteLine("True w: " + wTrue[i] + " inferred: " + G_mean + " +/- " + Math.Sqrt(G_var));
            }
            Console.WriteLine("log P(W_true | D) = " + logl);
        }

        [Fact]
        public void LogisticRegression2()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> m = Variable.GaussianFromMeanAndPrecision(1.2, 3.4).Named("m");
            Variable<double> prec = Variable.GammaFromShapeAndRate(5.0, 6.0).Named("prec");
            Range dim = new Range(2).Named("dim");
            VariableArray<double> w = Variable.Array<double>(dim).Named("w");
            w[dim] = Variable.GaussianFromMeanAndPrecision(m, prec).ForEach(dim);
            Range item = new Range(4).Named("item");
            var x = Variable.Array(Variable.Array<double>(dim), item).Named("x");
            x[item][dim] = Variable.GaussianFromMeanAndPrecision(2.0, 0.5).ForEach(item, dim);
            var wx = Variable.Array(Variable.Array<double>(dim), item).Named("wx");
            wx[item][dim] = w[dim] * x[item][dim];
            var sum = Variable.Array<double>(item).Named("sum");
            sum[item] = Variable.Sum(wx[item]);
            var y = Variable.Array<bool>(item).Named("y");
            y[item] = Variable.BernoulliFromLogOdds(sum[item]);
            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.NumberOfIterations = 500;

            x.ObservedValue = new double[][]
                {
                    new double[] {1, 2},
                    new double[] {-3, 4},
                    new double[] {5, -6},
                    new double[] {-7, -8}
                };
            y.ObservedValue = new bool[] { true, false, true, false };

            ie.Compiler.GivePriorityTo(typeof(LogisticOp_JJ96));
            VmpTests.TestGaussianMoments(ie, m, 1.247372733311744, 1.749611105200646);
            VmpTests.TestGammaMoments(ie, prec, 0.881679573457745, -0.211568384956627);
            double[][] wVibesResult = new double[2][];
            wVibesResult[0] = new double[] { 2.003961443928190, 4.207214268270652 };
            wVibesResult[1] = new double[] { 0.673466369104049, 0.583451188365762 };
            VmpTests.TestGaussianArrayMoments(ie, w, wVibesResult);
            VmpTests.TestEvidence(ie, evidence, -83.91957);
        }

        //[Fact]
        internal void LogisticRegression3()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            //IfBlock block = Variable.If(evidence);
            Variable<Vector> w = Variable.VectorGaussianFromMeanAndPrecision(Vector.FromArray(1.2, 1.2),
                                                                             new PositiveDefiniteMatrix(new double[,] { { 0.4, 0.0 }, { 0.0, 0.4 } }));
            Range item = new Range(4).Named("item");
            var x = Variable.Array<Vector>(item).Named("x");
            var wx = Variable.Array<double>(item).Named("wx");
            wx[item] = Variable.InnerProduct(w, x[item]);
            var y = Variable.Array<bool>(item).Named("y");
            y[item] = Variable.BernoulliFromLogOdds(wx[item]);
            //block.CloseBlock();
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());

            x.ObservedValue = new Vector[]
                {
                    Vector.FromArray(new double[] {1, 2}),
                    Vector.FromArray(new double[] {-3, 4}),
                    Vector.FromArray(new double[] {5, -6}),
                    Vector.FromArray(new double[] {-7, -8})
                };
            y.ObservedValue = new bool[] { true, false, true, false };

            if (false)
            {
                // can't duplicate this model in Vibes
                double[][] wVibesResult = new double[2][];
                wVibesResult[0] = new double[] { 2.390537703233513, 5.975568523864546 };
                wVibesResult[1] = new double[] { 0.573187406316556, 0.500807124106724 };
                VmpTests.TestVectorGaussianDiagonalMoments(ie, w, wVibesResult);
                VmpTests.TestEvidence(ie, evidence, -84.0191);
            }
        }

        [Fact]
        // Bayesian PCA as in Chris Bishop's book
        public void BayesianPCA()
        {
            Rand.Restart(12347);
            double[,] dataIn = MatlabReader.ReadMatrix(new double[400, 64], Path.Combine(TestUtils.DataFolderPath, "pca.txt"));
            VariableArray2D<double> data = Variable.Observed(dataIn).Named("data");

            Range N = data.Range0.Named("N");
            Range D = data.Range1.Named("D");
            Range d = new Range(10).Named("d");

            Variable<Gaussian> priorMu = Variable.New<Gaussian>().Named("PriorMu");
            Variable<Gamma> priorPrec = Variable.New<Gamma>().Named("PriorPrec");
            priorMu.ObservedValue = Gaussian.FromMeanAndPrecision(0.0, 1.0);
            priorPrec.ObservedValue = Gamma.FromShapeAndScale(2.0, 1 / 2.0);

            // Mixing matrix
            VariableArray2D<double> W = Variable.Array<double>(d, D).Named("W");
            // Initialize the marginal to break symmetry
            W.InitialiseTo(BioTests.RandomGaussianArray(d.SizeAsInt, D.SizeAsInt));

            VariableArray<double> Alpha = Variable.Array<double>(d).Named("Alpha");
            Alpha[d] = Variable.GammaFromShapeAndScale(2, 0.5).ForEach(d);
            W[d, D] = Variable.GaussianFromMeanAndPrecision(0, Alpha[d]).ForEach(D);

            // Latent variables
            VariableArray2D<double> Z = Variable.Array<double>(N, d).Named("Z");
            Z[N, d] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(N, d);

            // Multiply
            VariableArray2D<double> ZtimesW = Variable.MatrixMultiply(Z, W).Named("ZTimesW");

            // Bias
            VariableArray<double> mu = Variable.Array<double>(D).Named("mu");
            mu[D] = Variable.Random<double, Gaussian>(priorMu).ForEach(D);
            VariableArray2D<double> ZWplusMu = Variable.Array<double>(N, D).Named("ZWplusMu");
            ZWplusMu[N, D] = ZtimesW[N, D] + mu[D];

            // Observation noise
            VariableArray<double> pi = Variable.Array<double>(D).Named("pi");
            pi[D] = Variable.Random<double, Gamma>(priorPrec).ForEach(D);

            // The observation
            data[N, D] = Variable.GaussianFromMeanAndPrecision(ZWplusMu[N, D], pi[D]);

            // Inference
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.ShowProgress = true;
            engine.ShowTimings = true;
            //engine.ShowFactorGraph = true;
            //engine.BrowserMode = BrowserMode.Always;
            engine.NumberOfIterations = 5;
            // Fails with unrolling since MatrixMultiply assumes the array is updated in parallel
            engine.Compiler.UnrollLoops = false;

            DistributionArray2D<Gaussian> wMarginal = engine.Infer<DistributionArray2D<Gaussian>>(W);
            DistributionArray2D<Gaussian> zMarginal = engine.Infer<DistributionArray2D<Gaussian>>(Z);
            DistributionArray<Gaussian> muMarginal = engine.Infer<DistributionArray<Gaussian>>(mu);
            DistributionArray<Gamma> piMarginal = engine.Infer<DistributionArray<Gamma>>(pi);

            //BioTests.WriteMatrix(wMarginal.ToArray(), @"..\..\bpcaresultsW.txt");
            //BioTests.WriteMatrix(zMarginal.ToArray(), @"..\..\bpcaresultsZ.txt");

            // Reconstruct
            DistributionArray2D<Gaussian> productMarginal = MatrixMultiplyOp.MatrixMultiplyAverageLogarithm(zMarginal, wMarginal, null);
            double error = 0;
            for (int i = 0; i < productMarginal.GetLength(0); i++)
            {
                for (int j = 0; j < productMarginal.GetLength(1); j++)
                {
                    error += Math.Abs(productMarginal[i, j].GetMean() + muMarginal[j].GetMean() - dataIn[i, j]);
                }
            }
            error /= productMarginal.Count;
            // error = 0.118088499533868
            Console.WriteLine("error = {0}", error);
            Assert.True(error < 0.15);
        }

        [Fact]
        public void BayesianPartialVectorPCA()
        {
            Rand.Restart(12347);
            double[,] dataIn = MatlabReader.ReadMatrix(new double[400, 64], Path.Combine(TestUtils.DataFolderPath, "pca.txt"));
            VariableArray2D<double> data = Variable.Observed(dataIn).Named("data");

            Range N = data.Range0.Named("N");
            Range D = data.Range1.Named("D");
            Range d = new Range(10).Named("d");

            Variable<Gaussian> priorMu = Variable.New<Gaussian>().Named("PriorMu");
            Variable<Gamma> priorPrec = Variable.New<Gamma>().Named("PriorPrec");
            priorMu.ObservedValue = Gaussian.FromMeanAndPrecision(0.0, 1.0);
            priorPrec.ObservedValue = Gamma.FromShapeAndScale(2.0, 1 / 2.0);

            // Mixing matrix
            VariableArray<Vector> W = Variable.Array<Vector>(D).Named("W");
            // Initialize the marginal to break symmetry
            W.InitialiseTo(BioTests.RandomGaussianVectorArray(D.SizeAsInt, d.SizeAsInt));
            var Alpha = Variable.WishartFromShapeAndScale(2, PositiveDefiniteMatrix.IdentityScaledBy(d.SizeAsInt, .5)).Named("Alpha");
            W[D] = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(d.SizeAsInt), Alpha).ForEach(D);

            // Latent variables
            var Z = Variable.Array(Variable.Array<double>(d), N).Named("Z");
            Z[N][d] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(N, d);

            // Multiply
            var ZtimesW = Variable.Array<double>(N, D).Named("ZTimesW");
            ZtimesW[N, D] = Variable.InnerProduct(Z[N], W[D]);

            // Bias
            VariableArray<double> mu = Variable.Array<double>(D).Named("mu");
            mu[D] = Variable<double>.Random(priorMu).ForEach(D);
            mu.InitialiseTo(new GaussianArray(D.SizeAsInt, i => Gaussian.PointMass(0)));
            VariableArray2D<double> ZWplusMu = Variable.Array<double>(N, D).Named("ZWplusMu");
            ZWplusMu[N, D] = ZtimesW[N, D] + mu[D];

            // Observation noise
            VariableArray<double> pi = Variable.Array<double>(D).Named("pi");
            pi[D] = Variable.Random<double, Gamma>(priorPrec).ForEach(D);
            // must initialize pi since if it is updated too early it will be too small and spoil the results
            pi.InitialiseTo(Distribution<double>.Array(Util.ArrayInit(D.SizeAsInt, i => priorPrec.ObservedValue)));

            // The observation
            data[N, D] = Variable.GaussianFromMeanAndPrecision(ZWplusMu[N, D], pi[D]);

            // Inference
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.ShowProgress = true;
            engine.ShowTimings = true;
            engine.Compiler.InitialisationAffectsSchedule = true;
            engine.NumberOfIterations = 4;
            if (!engine.Compiler.OptimiseInferenceCode)
                engine.NumberOfIterations++;
            var wMarginal = engine.Infer<DistributionRefArray<VectorGaussian, Vector>>(W);
            var zMarginal = engine.Infer<DistributionArray<DistributionStructArray<Gaussian, double>>>(Z);
            DistributionArray<Gaussian> muMarginal = engine.Infer<DistributionArray<Gaussian>>(mu);
            DistributionArray<Gamma> piMarginal = engine.Infer<DistributionArray<Gamma>>(pi);

            //BioTests.WriteMatrix(wMarginal.ToArray(), @"..\..\bVpcaresultsW.txt");
            //BioTests.WriteMatrix(zMarginal.ToArray(), @"..\..\bVpcaresultsZ.txt");

            // Reconstruct

            double error = 0;
            for (int i = 0; i < N.SizeAsInt; i++)
            {
                for (int j = 0; j < D.SizeAsInt; j++)
                {
                    Gaussian productMarginal =
                        InnerProductPartialCovarianceOp.XAverageLogarithm(
                            zMarginal[i],
                            wMarginal[j], wMarginal[j].GetMean(), wMarginal[j].GetVariance());
                    error += Math.Abs(productMarginal.GetMean() + muMarginal[j].GetMean() - dataIn[i, j]);
                }
            }
            error /= (N.SizeAsInt * D.SizeAsInt);
            Console.WriteLine("error = {0}", error);
            Assert.True(error < 0.15);
        }

        [Fact]
        public void BayesianVectorPCA()
        {
            Rand.Restart(12347);
            double[,] dataIn = MatlabReader.ReadMatrix(new double[400, 64], Path.Combine(TestUtils.DataFolderPath, "pca.txt"));
            VariableArray2D<double> data = Variable.Observed(dataIn).Named("data");

            Range N = data.Range0.Named("N");
            Range D = data.Range1.Named("D");
            Range d = new Range(10).Named("d");

            Variable<Gaussian> priorMu = Variable.New<Gaussian>().Named("PriorMu");
            Variable<Gamma> priorPrec = Variable.New<Gamma>().Named("PriorPrec");
            priorMu.ObservedValue = Gaussian.FromMeanAndPrecision(0.0, 1.0);
            priorPrec.ObservedValue = Gamma.FromShapeAndScale(2.0, 1 / 2.0);

            // Mixing matrix
            VariableArray<Vector> W = Variable.Array<Vector>(D).Named("W");
            // Initialize the marginal to break symmetry
            W.InitialiseTo(BioTests.RandomGaussianVectorArray(D.SizeAsInt, d.SizeAsInt));
            var Alpha = Variable.WishartFromShapeAndScale(2, PositiveDefiniteMatrix.IdentityScaledBy(d.SizeAsInt, .5)).Named("Alpha");
            W[D] = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(d.SizeAsInt), Alpha).ForEach(D);

            // Latent variables
            var Z = Variable.Array<Vector>(N).Named("Z");
            Z[N] = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(d.SizeAsInt), PositiveDefiniteMatrix.Identity(d.SizeAsInt)).ForEach(N);

            // Multiply
            var ZtimesW = Variable.Array<double>(N, D).Named("ZTimesW");
            ZtimesW[N, D] = Variable.InnerProduct(Z[N], W[D]);

            // Bias
            VariableArray<double> mu = Variable.Array<double>(D).Named("mu");
            mu[D] = Variable.Random<double, Gaussian>(priorMu).ForEach(D);
            VariableArray2D<double> ZWplusMu = Variable.Array<double>(N, D).Named("ZWplusMu");
            ZWplusMu[N, D] = ZtimesW[N, D] + mu[D];

            // Observation noise
            VariableArray<double> pi = Variable.Array<double>(D).Named("pi");
            pi[D] = Variable.Random<double, Gamma>(priorPrec).ForEach(D);

            // The observation
            data[N, D] = Variable.GaussianFromMeanAndPrecision(ZWplusMu[N, D], pi[D]);

            // Inference
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.ShowProgress = true;
            engine.ShowTimings = true;
            //engine.ShowFactorGraph = true;
            //engine.BrowserMode = BrowserMode.Always;
            engine.NumberOfIterations = 10;
            var wMarginal = engine.Infer<DistributionRefArray<VectorGaussian, Vector>>(W);
            var zMarginal = engine.Infer<DistributionRefArray<VectorGaussian, Vector>>(Z);
            DistributionArray<Gaussian> muMarginal = engine.Infer<DistributionArray<Gaussian>>(mu);
            DistributionArray<Gamma> piMarginal = engine.Infer<DistributionArray<Gamma>>(pi);

            //BioTests.WriteMatrix(wMarginal.ToArray(), @"..\..\bVpcaresultsW.txt");
            //BioTests.WriteMatrix(zMarginal.ToArray(), @"..\..\bVpcaresultsZ.txt");

            // Reconstruct
            double error = 0;
            for (int i = 0; i < N.SizeAsInt; i++)
            {
                for (int j = 0; j < D.SizeAsInt; j++)
                {
                    Gaussian productMarginal = InnerProductOp.InnerProductAverageLogarithm(
                        zMarginal[i].GetMean(),
                        zMarginal[i].GetVariance(),
                        wMarginal[j].GetMean(),
                        wMarginal[j].GetVariance());
                    error += Math.Abs(productMarginal.GetMean() + muMarginal[j].GetMean() - dataIn[i, j]);
                }
            }
            error /= (N.SizeAsInt * D.SizeAsInt);
            Console.WriteLine("error = {0}", error);
            Assert.True(error < 0.15);
        }

        [Fact]
        public void BinaryPCA()
        {
            Rand.Restart(12347);
            double[,] dataIn = MatlabReader.ReadMatrix(new double[400, 64], Path.Combine(TestUtils.DataFolderPath, "pca.txt"));
            var binData = new bool[400, 64];
            for (int i = 0; i < 400; i++)
            {
                for (int j = 0; j < 64; j++)
                {
                    binData[i, j] = dataIn[i, j] > .1;
                }
            }
            VariableArray2D<bool> data = Variable.Observed(binData).Named("data");

            Range N = data.Range0.Named("N");
            Range D = data.Range1.Named("D");
            Range d = new Range(10).Named("d");

            Variable<Gaussian> priorMu = Variable.New<Gaussian>().Named("PriorMu");
            Variable<Gamma> priorPrec = Variable.New<Gamma>().Named("PriorPrec");
            priorMu.ObservedValue = Gaussian.FromMeanAndPrecision(0.0, 1.0);
            priorPrec.ObservedValue = Gamma.FromShapeAndScale(2.0, 1 / 2.0);

            // Mixing matrix
            VariableArray<Vector> W = Variable.Array<Vector>(D).Named("W");
            // Initialize the marginal to break symmetry
            W.InitialiseTo(BioTests.RandomGaussianVectorArray(D.SizeAsInt, d.SizeAsInt));
            var Alpha = Variable.WishartFromShapeAndScale(2, PositiveDefiniteMatrix.IdentityScaledBy(d.SizeAsInt, .5)).Named("Alpha");
            W[D] = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(d.SizeAsInt), Alpha).ForEach(D);

            // Latent variables
            var Z = Variable.Array<Vector>(N).Named("Z");
            Z[N] = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(d.SizeAsInt), PositiveDefiniteMatrix.Identity(d.SizeAsInt)).ForEach(N);

            // Multiply
            var ZtimesW = Variable.Array<double>(N, D).Named("ZTimesW");
            ZtimesW[N, D] = Variable.InnerProduct(Z[N], W[D]);

            // Bias
            VariableArray<double> mu = Variable.Array<double>(D).Named("mu");
            mu[D] = Variable.Random<double, Gaussian>(priorMu).ForEach(D);
            VariableArray2D<double> ZWplusMu = Variable.Array<double>(N, D).Named("ZWplusMu");
            ZWplusMu[N, D] = ZtimesW[N, D] + mu[D];

            // The observation
            data[N, D] = Variable.BernoulliFromLogOdds(ZWplusMu[N, D]);

            // Inference
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.ShowProgress = true;
            engine.ShowTimings = true;
            //engine.ShowFactorGraph = true;
            //engine.BrowserMode = BrowserMode.Always;
            engine.NumberOfIterations = 10;
            var wMarginal = engine.Infer<DistributionRefArray<VectorGaussian, Vector>>(W);
            var zMarginal = engine.Infer<DistributionRefArray<VectorGaussian, Vector>>(Z);
            DistributionArray<Gaussian> muMarginal = engine.Infer<DistributionArray<Gaussian>>(mu);
            //DistributionArray<Gamma> piMarginal = engine.Infer<DistributionArray<Gamma>>(pi);
        }

        [Fact]
        public void VectorGaussianWithLoops()
        {
            // Model
            Vector[] data = { Vector.FromArray(new double[] { 1, 2 }), Vector.FromArray(new double[] { 3, 4 }) };
            Variable<Vector> mean = Variable.VectorGaussianFromMeanAndVariance(Vector.Zero(2), PositiveDefiniteMatrix.IdentityScaledBy(2, 100.0)).Named("mean");
            mean.ObservedValue = Vector.Zero(2);
            Variable<PositiveDefiniteMatrix> precision = Variable.WishartFromShapeAndScale(2.0, PositiveDefiniteMatrix.IdentityScaledBy(2, 100.0)).Named("precision");
            VariableArray<Vector> dataVar = Variable.Constant(data);
            Range N = dataVar.Range;
            dataVar[N] = Variable.VectorGaussianFromMeanAndPrecision(mean, precision).ForEach(N);

            Wishart precisionExpected = Wishart.FromShapeAndScale(2.0, PositiveDefiniteMatrix.IdentityScaledBy(2, 100.0));
            for (int i = 0; i < data.Length; i++)
            {
                PositiveDefiniteMatrix rate = new PositiveDefiniteMatrix(precisionExpected.Dimension, precisionExpected.Dimension);
                rate.SetToSumWithOuter(rate, 0.5, data[i], data[i]);
                precisionExpected.SetToProduct(precisionExpected, Wishart.FromShapeAndRate(0.5 * (2 + 2), rate));
            }

            // Inference
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Wishart precisionActual = engine.Infer<Wishart>(precision);
            Console.WriteLine(StringUtil.JoinColumns(engine.Algorithm.ShortName, " Marginal for precision = ", precisionActual));
            Assert.True(precisionExpected.MaxDiff(precisionActual) < 1e-8);

            engine = new InferenceEngine(new ExpectationPropagation());
            precisionActual = engine.Infer<Wishart>(precision);
            Console.WriteLine(StringUtil.JoinColumns(engine.Algorithm.ShortName, " Marginal for precision = ", precisionActual));
            Assert.True(precisionExpected.MaxDiff(precisionActual) < 1e-8);
            //Console.WriteLine("Marginal for mean = " +  engine.Infer(mean));

            // Validation
            //VectorGaussian meanExpected = VectorGaussian.FromMeanAndVariance(5.96, 0.6613);
            //VectorGaussian meanActual = engine.Infer<VectorGaussian>(mean);
            //Assert.True(meanExpected.MaxDiff(meanActual) < 1e-4);
        }

        [Fact]
        public void SimpleGaussianWithLoops()
        {
            // Model
            double[] data = { 5, 7 }; //, 9, 11, 17, 41, 32 };
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100).Named("mean");
            Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1).Named("precision");
            //precision.AddAttribute(new Output());
            VariableArray<double> dataVar = Variable.Constant(data);
            Range N = dataVar.Range;
            dataVar[N] = Variable.GaussianFromMeanAndPrecision(mean, precision).ForEach(N);

            // Inference
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine("Marginal for mean = " + engine.Infer(mean));
            Console.WriteLine("Marginal for precision = " + engine.Infer(precision));
            //Console.WriteLine("Output message for precision = " + precision.GetOutputMessage<Gamma>());

            // Validation
            Gaussian meanExpected = Gaussian.FromMeanAndVariance(5.9603207170807826, 0.66132138200164436);
            Gaussian meanActual = engine.Infer<Gaussian>(mean);
            Assert.True(meanExpected.MaxDiff(meanActual) < 1e-4);
            Gamma precisionExpected = Gamma.FromShapeAndRate(2, 2.6628958274937107);
            Gamma precisionActual = engine.Infer<Gamma>(precision);
            Assert.True(precisionExpected.MaxDiff(precisionActual) < 2e-4);
        }

        [Fact]
        public void VectorGaussianWithCompoundWishart()
        {
            // Sample data
            Vector[] data = new Vector[] { Vector.FromArray(new double[] { 1, 2 }), Vector.FromArray(new double[] { 3, 4 }) };

            // Expected results
            VectorGaussian meanExpected = VectorGaussian.FromMeanAndVariance(
                Vector.FromArray(new double[] { 1.536, 2.309 }),
                new PositiveDefiniteMatrix(new double[,] { { 22.79, 0.2672 }, { 0.2672, 22.84 } }));
            Wishart precisionExpected = Wishart.FromShapeAndScale(
                2,
                new PositiveDefiniteMatrix(new double[,] { { 0.008472, -0.0001283 }, { -0.0001283, 0.008446 } }));

            // Model
            Variable<Vector> mean = Variable.VectorGaussianFromMeanAndPrecision(
                Vector.Zero(2),
                PositiveDefiniteMatrix.IdentityScaledBy(2, 0.01)).Named("mean");
            Variable<PositiveDefiniteMatrix> rate = Variable.WishartFromShapeAndRate(
                10,
                PositiveDefiniteMatrix.IdentityScaledBy(2, 0.1)).Named("rate");
            Variable<PositiveDefiniteMatrix> precision = Variable.WishartFromShapeAndRate(1, rate).Named("precision");

            Range n = new Range(data.Length).Named("n");
            VariableArray<Vector> x = Variable.Array<Vector>(n).Named("x");
            x[n] = Variable.VectorGaussianFromMeanAndPrecision(mean, precision).ForEach(n);
            x.ObservedValue = data;


            // Inference VMP
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine("[VMP] Marginal for mean =\n" + engine.Infer(mean));
            Console.WriteLine("[VMP] Marginal for precision =\n" + engine.Infer(precision));

            // Validation VMP
            VectorGaussian meanActual = engine.Infer<VectorGaussian>(mean);
            Assert.True(meanExpected.MaxDiff(meanActual) < 1e-3);
            Wishart precisionActual = engine.Infer<Wishart>(precision);
            Assert.True(precisionExpected.MaxDiff(precisionActual) < 1e-2);
        }

        [Fact]
        public void SimpleGaussianWithCompoundGamma()
        {
            // Sample data
            double[] data = new double[] { 5, 7 };

            // Expected results
            Gaussian meanExpected = Gaussian.FromMeanAndVariance(5.957, 0.722);
            Gamma precisionExpected = Gamma.FromShapeAndRate(2, 2.90866783);

            // Model
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100).Named("mean");
            Variable<double> rate = Variable.GammaFromShapeAndRate(1, 1).Named("rate");
            Variable<double> precision = Variable.GammaFromShapeAndRate(1, rate).Named("precision");

            Range n = new Range(data.Length).Named("n");
            VariableArray<double> x = Variable.Array<double>(n).Named("x");
            x[n] = Variable.GaussianFromMeanAndPrecision(mean, precision).ForEach(n);
            x.ObservedValue = data;


            // Inference VMP
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine("[VMP] Marginal for mean = " + engine.Infer(mean));
            Console.WriteLine("[VMP] Marginal for precision = " + engine.Infer(precision));

            // Validation VMP
            Gaussian meanActual = engine.Infer<Gaussian>(mean);
            Assert.True(meanExpected.MaxDiff(meanActual) < 1e-3);
            Gamma precisionActual = engine.Infer<Gamma>(precision);
            Assert.True(precisionExpected.MaxDiff(precisionActual) < 1e-3);

            // Inference EP
            engine = new InferenceEngine(new ExpectationPropagation());
            engine.Compiler.GivePriorityTo(typeof(GammaFromShapeAndRateOp_Laplace));
            Console.WriteLine("[EP] Marginal for mean = " + engine.Infer(mean));
            Console.WriteLine("[EP] Marginal for precision = " + engine.Infer(precision));

            // Validation EP
            //meanActual = engine.Infer<Gaussian>(mean);
            //Assert.True(meanExpected.MaxDiff(meanActual) < 1e-3);
            //precisionActual = engine.Infer<Gamma>(precision);
            //Assert.True(precisionExpected.MaxDiff(precisionActual) < 1e-3);
        }
    }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif
}
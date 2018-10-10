// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#define coupled
#define gated
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Xunit;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Compiler;

namespace Microsoft.ML.Probabilistic.Tests
{
    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
    using BernoulliArray = DistributionStructArray<Bernoulli, bool>;
    using BernoulliArrayArray = DistributionRefArray<DistributionStructArray<Bernoulli, bool>, bool[]>;
    using DirichletArray = DistributionRefArray<Dirichlet, Vector>;
    using GaussianArray = DistributionStructArray<Gaussian, double>;

    /// <summary>
    /// Tests for the modelling API
    /// </summary>
    public class ModelTests
    {
        /// <summary>
        /// Tests that C# compiler errors produce a CompilationFailedException.
        /// </summary>
        [Fact]
        public void GeneratedSourceError()
        {
            double pTrue = 0.7;
            bool[] data = Util.ArrayInit(100, i => (Rand.Double() < pTrue));
            Variable<double> p = Variable.Beta(1, 1).Named("p");
            Range item = new Range(data.Length);
            VariableArray<bool> x = Variable.Array<bool>(item).Named("x");
            x[item] = Variable.Bernoulli(p).ForEach(item);
            x.ObservedValue = data;
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.WriteSourceFiles = true;
            engine.Compiler.UseExistingSourceFiles = true;
            engine.ModelName = "src_" + DateTime.Now.ToString("MM_dd_yy_HH_mm_ss_ff");
            Directory.CreateDirectory(engine.Compiler.GeneratedSourceFolder);
            string sourcePath = Path.Combine(engine.Compiler.GeneratedSourceFolder, engine.ModelName + "_EP.cs");
            File.WriteAllText(sourcePath, "invalid");
            Assert.Throws<CompilationFailedException>(() => { engine.Infer<Beta>(p); });
        }

        /// <summary>
        /// Test that hoisting works correctly for a jagged array with weakly-relevant outer dimension.
        /// </summary>
        [Fact]
        public void HoistingTest()
        {
            Range outer = new Range(2).Named("outer");
            var innerSizes = Variable.Constant(new int[] { 3, 2 }, outer).Named("innerSizes");
            Range inner = new Range(innerSizes[outer]).Named("inner");
            var array = Variable.Array(Variable.Array<int>(inner), outer).Named("array");
            using (Variable.ForEach(outer))
            {
                using (var innerBlock = Variable.ForEach(inner))
                {
                    array[outer][inner] = Variable.DiscreteUniform(innerBlock.Index + 1);
                }
            }
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(array));
        }

        /// <summary>
        /// Test that SumWhere works when arguments are array elements.
        /// </summary>
        [Fact]
        public void SumWhereTest()
        {
            var outer = new Range(2);
            var inner = new Range(2);
            var x = Variable.Array<double>(inner);
            x[inner] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(inner);
            var z = Variable.Array(Variable.Array<bool>(inner), outer);
            z[outer][inner] = Variable.Bernoulli(.5).ForEach(outer, inner);
            var y = Variable.Array<double>(outer);
            y[outer] = Variable.SumWhere(z[outer], x);
            Variable.ConstrainEqualRandom(y[outer], new Gaussian(2.0, .5));
            var ie = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(ie.Infer(z));
        }

        /// <summary>
        /// Test that variables observed to impossible values throw an exception.
        /// </summary>
        [Fact]
        public void ObserveDeterministicViolationError()
        {
            Assert.Throws<ConstraintViolatedException>(() =>
            {
                var b = Variable.Observed(true).Named("b");
                var c = (!b).Named("c");
                c.ObservedValue = true;
                InferenceEngine engine = new InferenceEngine();
                Console.WriteLine(engine.Infer(c));
            });
        }

        /// <summary>
        /// Test that variables observed to impossible values throw an exception.
        /// </summary>
        [Fact]
        public void ObserveDeterministicViolationError2()
        {
            Assert.Throws<ConstraintViolatedException>(() =>
            {
                var b = Variable.Observed(true).Named("b");
                var c = (!b).Named("c");
                Variable.ConstrainTrue(c);
                InferenceEngine engine = new InferenceEngine();
                Console.WriteLine(engine.Infer(c));
            });
        }

        /// <summary>
        /// Test that constraints on observed variables are enforced in the generated code.
        /// </summary>
        [Fact]
        public void ObservedConstraintViolationError()
        {
            Assert.Throws<ConstraintViolatedException>(() =>
            {
                var c = Variable.Constant(2).Named("c");
                var t = Variable.Observed(1).Named("t");
                Variable.ConstrainTrue(t > c);
                InferenceEngine engine = new InferenceEngine();
                Console.WriteLine(engine.Infer(t));
            });
        }

        /// <summary>
        /// Test that the size of a range can refer to a constant.
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")]
        public void RangeWithConstantSizeTest()
        {
            var counts = Variable.Constant(new int[] { 2, 3 }).Named("counts");
            var range = new Range(counts[0]).Named("range");
            var array = Variable.Array<bool>(range).Named("array");
            array.ObservedValue = new bool[2];
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(array));
        }

        /// <summary>
        /// Test for a meaningful error message when a range depends on an inner range.
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")]
        public void JaggedRangesError()
        {
            Assert.Throws<CompilationFailedException>(() =>
            {
                var outer = new Range(2).Named("outer");
                var counts = Variable.Observed(new int[] { 2, 3 }, outer).Named("counts");
                var inner = new Range(counts[outer]).Named("inner");
                var array = Variable.Array(Variable.Array<Vector>(outer), inner).Named("array");
                array.ObservedValue = new Vector[2][];
                var engine = new InferenceEngine();
                Console.WriteLine(engine.Infer(array));
            });
        }

        /// <summary>
        /// Test that Variable.Factor contains the appropriate overloads
        /// </summary>
        [Fact]
        public void FactorFromMethodTest()
        {
            var y = Variable<double>.Factor<double, double>(Factor.GammaFromShapeAndRate, 1, 1);
            var x = Variable<bool>.Factor<double, double, double>(Factor.IsBetween, 1, 1, 1);
        }

        [Fact]
        public void LoopMerging2dTest()
        {
            Range r = new Range(2);
            Range r2 = new Range(2);
            var array = Variable.Array<bool>(r, r2).Named("array");
            array[r, r2] = Variable.Bernoulli(0.1).ForEach(r, r2);
            using (var fb2 = Variable.ForEach(r2))
            {
                using (var fb = Variable.ForEach(r))
                {
                    Variable.ConstrainTrue(array[fb2.Index, fb.Index]);
                }
            }
            InferenceEngine engine = new InferenceEngine();
            var arrayActual = engine.Infer<IArray2D<Bernoulli>>(array);
            var arrayExpected = Bernoulli.PointMass(true);
            for (int i = 0; i < arrayActual.Count; i++)
            {
                Assert.True(arrayExpected.MaxDiff(arrayActual[i]) < 1e-10);
            }
        }

        [Fact]
        public void DuplicateRangeError()

        {
            Assert.Throws<CompilationFailedException>(() =>
            {

                Range r = new Range(2).Named("r");
                var kernel = Variable.Array(Variable.Array<bool>(r), r).Named("kernel");
                //kernel[r][r] = Variable.Bernoulli(0.1).ForEach(r);
                kernel.ObservedValue = Util.ArrayInit(r.SizeAsInt, i => Util.ArrayInit(r.SizeAsInt, j => false));
                InferenceEngine engine = new InferenceEngine();
                Console.WriteLine(engine.Infer(kernel));

            });
        }

        // Fails with 'compiler limit exceeded'
        //[Fact]
        internal void LargeArrayConstantTest()
        {
            var array = Variable.Constant(Util.ArrayInit(10000000, i => false));
            InferenceEngine engine = new InferenceEngine();
            engine.Infer(array);
        }

        // This test should throw an exception explaining that r2 depends on r1 so it cannot be used with r1Clone.
        // Currently it fails when compiling the generated code.
        [Fact]
        [Trait("Category", "OpenBug")]
        public void CloneRangeWithBadInnerRangeError()
        {
            Assert.Throws<InvalidOperationException>(() =>
            {

                var r1 = (new Range(1)).Named("r1");
                var len = (Variable.Observed<int>(new[] { 1 }, r1)).Named("len");
                var r2 = (new Range(len[r1])).Named("r2");
                var v = (Variable.Array(Variable.Array<double>(r2), r1)).Named("var1");
                v[r1][r2] = Variable.GaussianFromMeanAndPrecision(0.0, 1.0).ForEach(r1, r2);
#if true
                var r1Clone = (r1.Clone()).Named("r1Clone");
                var constraint = (Variable.Array(Variable.Array<double>(r2), r1Clone)).Named("constraint");
#else
                var constraint = (Variable.Array(Variable.Array<double>(r2), r1)).Named("constraint");
#endif
                constraint.ObservedValue = new[] { new[] { 1.0 } };
                using (Variable.ForEach(r1))
                {
                    Variable.ConstrainEqual<double>(constraint[r1][r2], Variable.GaussianFromMeanAndPrecision(v[r1][r2], 1.0));
                }
                var e = new InferenceEngine(new ExpectationPropagation());
                e.Infer(v);

            });
        }

        [Fact]
        public void RandomPointMassTest()
        {
            var v = Variable.Random(new PointMass<double>(42.0)).Named("v");
            var x = Variable.GaussianFromMeanAndVariance(v, 1).Named("x");
            Variable.ConstrainPositive(x);
            var engine = new InferenceEngine();
            //engine.OptimiseForVariables = new List<IVariable>() { x, v };
            Console.WriteLine(engine.Infer(x));
            Console.WriteLine(engine.Infer(v));
        }

        /// <summary>
        /// Message transform used to fail, saying that Gaussian is not assignable from double.
        /// </summary>
        [Fact]
        public void MixtureOfPointMassesTest()
        {
            const double SelectorProbTrue = 0.3;

            Variable<double> v = Variable.New<double>().Named("v");
            Variable<bool> c = Variable.Bernoulli(SelectorProbTrue).Named("c");
            using (Variable.If(c))
            {
                v.SetTo(Variable.Random(Gaussian.PointMass(1)));
            }
            using (Variable.IfNot(c))
            {
                v.SetTo(Variable.Random(Gaussian.PointMass(2)));
            }

            InferenceEngine engine = new InferenceEngine();
            Gaussian vPosterior = engine.Infer<Gaussian>(v);
        }

        [Fact]
        public void NullConstantTest()
        {
            var evidence = Variable.Bernoulli(0.5).Named("evidence");
            var block = Variable.If(evidence);
            Range item = new Range(0).Named("item");
            var array = Variable.Array<bool>(item).Named("array");
            array.ObservedValue = null;
            array.IsReadOnly = true;
            var c = Variable.AllTrue(array).Named("c");
            var array2 = Variable.Array<bool>(item).Named("array2");
            array2[item] = !array[item];
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            engine.OptimiseForVariables = new IVariable[] { evidence };
            Console.WriteLine(engine.Infer(evidence));
        }

        // scheduler duplicates work if KeepFresh=true
        internal void NevenaTest()
        {
            int dimX = 2;
            int numX = 1;
            int numN = 2;
            int numU = 2;
            int dimU = 2;

            // # variables
            var nX = Variable.Observed<int>(numX).Named("nX");
            Range D = new Range(nX).Named("D");
            var nU = Variable.Observed<int>(numU).Named("nU");
            Range rG = new Range(nU).Named("rG");

            // # observations
            var nN = Variable.Observed<int>(numN).Named("nN");
            Range N = new Range(nN).Named("N");

            // variable dimension
            var dX = Variable.Observed<int>(dimX).Named("dX");
            Range rX = new Range(dX).Named("rX");
            var dU = Variable.Observed<int>(dimU).Named("dU");
            Range rU = new Range(dU).Named("rU");


            // Latent variables

            // U latent var definition & init
            var U = Variable.Array(Variable.Array<int>(N), rG).Named("U");
            U[rG][N] = Variable.DiscreteUniform(rU).ForEach(rG, N);

            // G structure var definition & init
            var G = Variable.Array<int>(D).Named("G");
            G[D] = Variable.DiscreteUniform(rG).ForEach(D);

            // pT: prior for T 
            var priorObs = new Dirichlet[numX][];
            for (int i = 0; i < numX; i++)
            {
                priorObs[i] = new Dirichlet[dimU];
                for (int j = 0; j < dimU; j++)
                    priorObs[i][j] = Dirichlet.Symmetric(dimX, 1 / (double)dimU / (double)dimX);
            }
            var pT = Variable.Array(Variable.Array<Dirichlet>(rU), D).Named("pT");
            pT.ObservedValue = priorObs;

            // T: cpt for X conditioned on U
            var T = Variable.Array(Variable.Array<Vector>(rU), D).Named("T");

            // X: observed variables
            var X = Variable.Array(Variable.Array<int>(N), D).Named("X");
            X.ObservedValue = Util.ArrayInit(numX, d => Util.ArrayInit(numN, n => n));

            // Model
            using (Variable.ForEach(D))
            {
                T[D][rU] = Variable<Vector>.Random(pT[D][rU]);
                T[D].SetValueRange(rX);
                X[D].SetValueRange(rX);


                using (Variable.Switch(G[D]))
                using (Variable.ForEach(N))
                using (Variable.Switch(U[G[D]][N]))
                    X[D][N] = Variable.Discrete(T[D][U[G[D]][N]]);
            }

            T.AddAttribute(new DivideMessages(false));
            U.AddAttribute(new DivideMessages(false));
            G.AddAttribute(new DivideMessages(false));

            // Random init

            var initU = new Discrete[numU][];
            for (int i = 0; i < numU; i++)
            {
                initU[i] = new Discrete[numN];
                for (int j = 0; j < numN; j++)
                    initU[i][j] = Discrete.PointMass(Rand.Int(dimU), dimU);
            }
            U.InitialiseTo(Distribution<int>.Array(initU));

            var initG = new Discrete[numX];
            for (int i = 0; i < numX; i++)
                initG[i] = Discrete.PointMass(Rand.Int(numU), numU);
            G.InitialiseTo(Distribution<int>.Array(initG));


            // Inference

            // ie.Compiler.WriteSourceFiles = false;
            //   ie.ModelName = "OneParent24";

            var ie = new InferenceEngine(new ExpectationPropagation());
            //ie.ShowSchedule = true;

            for (int iter = 1; iter < 100; iter++)
            {
                ie.NumberOfIterations = iter;
                var Gout = ie.Infer<Discrete[]>(G);


                for (int i = 0; i < numX; i++)
                {
                    double[] probs = Gout[i].GetProbs().ToArray();
                    for (int j = 0; j < probs.Length; j++)
                        Console.Write("  {0:F2}", probs[j]);
                }
                Console.WriteLine();
            }
        }

        /// <summary>
        /// Fails with the wrong error message.
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")]
        public void DefinitionBeforeDeclarationError()
        {
            Assert.Throws<CompilationFailedException>(() =>
            {

                var r = new Range(2);

                // execute the body
                var b1 = Variable.ForEach(r);
                var r2 = new Range(2);
                var item = Variable.Array<int>(r2);
                item[r2].SetTo(Variable.DiscreteUniform(3).ForEach(r2));
                b1.CloseBlock();

                // figure out item range and create the result array with correct ranges
                var rItem = item.Range;
                var proto = Variable.Array<int>(rItem);
                var x = Variable.Array<int>(proto, r).Named("x");

                // assign the result
                var b2 = Variable.ForEach(r);
                x[r].SetTo(item);
                //x[r].SetTo(Variable.Copy(item));
                b2.CloseBlock();

                var ie = new InferenceEngine();
                ie.ShowMsl = true;
                var D = ie.Infer(x);
                Console.WriteLine("x = {0}", D);

            });
        }

        [Fact]
        public void NullVariableError()
        {
            Variable<double> x;
            try
            {
                x = Variable.Random<double, Gamma>(null);
            }
            catch (ArgumentNullException ex)
            {
                Console.WriteLine("Exception correctly thrown: " + ex.Message);
                // Exception correctly thrown
                return;
            }
            Assert.True(false, "Null variable was not detected.");
        }

        [Fact]
        public void NullVariableError2()
        {
            Variable<bool> x;
            try
            {
                x = Variable.Bernoulli(null);
            }
            catch (ArgumentNullException ex)
            {
                Console.WriteLine("Exception correctly thrown: " + ex.Message);
                // Exception correctly thrown
                return;
            }
            Assert.True(false, "Null variable was not detected.");
        }

        [Fact]
        public void NullVariableError3()
        {
            Variable<bool> x = null;
            try
            {
                Variable.ConstrainTrue(x);
            }
            catch (ArgumentNullException ex)
            {
                Console.WriteLine("Exception correctly thrown: " + ex.Message);
                // Exception correctly thrown
                return;
            }
            Assert.True(false, "Null variable was not detected.");
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void GibbsArrayUsedTwiceInFactorTest()
        {
            ArrayUsedTwiceInFactor(new GibbsSampling());
        }

        [Fact]
        public void ArrayUsedTwiceInFactorTest()
        {
            ArrayUsedTwiceInFactor(new ExpectationPropagation());
            ArrayUsedTwiceInFactor(new VariationalMessagePassing());
        }

        private void ArrayUsedTwiceInFactor(IAlgorithm algorithm)
        {
            Range item = new Range(2);
            VariableArray<double> x = Variable.Array<double>(item).Named("x");
            x[item] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(item);
            var y = x[0] + x[1];
            y.Name = "y";
            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = algorithm;
            double tolerance = 1e-8;
            Gaussian yActual = engine.Infer<Gaussian>(y);
            Gaussian yExpected = new Gaussian(0, 2);
            Console.WriteLine("y = {0} should be {1}", yActual, yExpected);
            Assert.True(yExpected.MaxDiff(yActual) < tolerance);
        }

        [Fact]
        public void VectorFromArrayError()

        {
            Assert.Throws<CompilationFailedException>(() =>
            {

                Range r = new Range(3);
                var probs = Variable.Array<double>(r).Named("probs");
                var b0 = Variable.Bernoulli(0.5).Named("b0");
                using (Variable.If(b0))
                {
                    probs[0] = 1.0;
                }
                using (Variable.IfNot(b0))
                {
                    probs[0] = 0.0;
                }
                probs[1] = 1.0;
                probs[2] = 0.0;
                var probsVec = Variable.Vector(probs);
                var x = Variable.Discrete(probsVec);
                InferenceEngine engine = new InferenceEngine();
                Console.WriteLine(engine.Infer(x));

            });
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        internal void PoissonRegressionTest()
        {
            Rand.Restart(0);
            Vector wTrue = Vector.FromArray(10, -1);
            int n = 10;
            Vector[] xData = Util.ArrayInit(n, j => Vector.FromArray(Rand.Double(), Rand.Double()));
            int[] yData = Util.ArrayInit(n, j => Rand.Poisson(System.Math.Exp(wTrue.Inner(xData[j]))));
            double wVariance = 1;

            if (false)
            {
                // test on data from Matt Wand
                //TODO: change path
                using (StreamReader reader = new StreamReader(@"c:\users\minka\Desktop\TMinkaChk.csv"))
                {
                    bool gotHeader = false;
                    List<Vector> xList = new List<Vector>();
                    List<int> yList = new List<int>();
                    while (!reader.EndOfStream)
                    {
                        string line = reader.ReadLine();
                        if (!gotHeader)
                        {
                            gotHeader = true;
                            continue;
                        }
                        string[] fields = line.Split(',');
                        int yi = int.Parse(fields[0]);
                        yList.Add(yi);
                        Vector xi = Vector.FromArray(Util.ArrayInit(fields.Length, j => (j == 0) ? 1.0 : double.Parse(fields[j])));
                        xList.Add(xi);
                    }
                    n = yList.Count;
                    yData = yList.ToArray();
                    xData = xList.ToArray();
                }
                wVariance = 1000;
            }

            Range i = new Range(yData.Length).Named("i");
            int dim = xData[0].Count;
            Variable<Vector> w = Variable.VectorGaussianFromMeanAndVariance(Vector.Zero(dim), PositiveDefiniteMatrix.IdentityScaledBy(dim, wVariance)).Named("w");
            // must initialise to a reasonable variance since ExpOp explodes with large variance
            if (wVariance > 1)
                //w.InitialiseTo(new VectorGaussian(Vector.Zero(dim), PositiveDefiniteMatrix.Identity(dim)));
                w.InitialiseTo(new VectorGaussian(wTrue, PositiveDefiniteMatrix.Identity(dim)));
            VariableArray<Vector> x = Variable.Array<Vector>(i).Named("x");
            x.ObservedValue = xData;
            VariableArray<int> y = Variable.Array<int>(i).Named("y");
            y[i] = Variable.Poisson(Variable.Exp(Variable.InnerProduct(w, x[i]).Named("c")).Named("d"));
            y.ObservedValue = yData;

            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine("EP");
            Console.WriteLine(engine.Infer<VectorGaussian>(w));
            //engine.ShowFactorGraph = true;
            // VMP is very similar to EP in this model because there is only one stochastic variable
            engine.Algorithm = new VariationalMessagePassing();
            engine.NumberOfIterations = 1000;
            ExpOp.UseRandomDamping = false;
            Console.WriteLine("VMP");
            //engine.Compiler.GivePriorityTo(typeof(ExpOp_Laplace));  // modified Laplace is identical to EP with n=100
            // works with either ExpectationPropagation (default) or VariationalMessagePassing
            //engine.Algorithm = new VariationalMessagePassing();
            // EP result should be:
            // VectorGaussian(1.852 -0.6617, 0.01701  -0.01764)
            //                               -0.01764 0.03279
            // VMP result should be:
            // VectorGaussian(1.852 -0.6617, 0.017    -0.01763)
            //                               -0.01763 0.03278
            if (false)
            {
                for (int iter = 1; iter < 1000; iter++)
                {
                    engine.NumberOfIterations = iter;
                    Console.WriteLine(StringUtil.JoinColumns(iter.ToString(), ": ", engine.Infer<VectorGaussian>(w)));
                }
            }
            VectorGaussian wActual = engine.Infer<VectorGaussian>(w);
            Console.WriteLine(wActual);
            Console.WriteLine("wTrue = {0}", wTrue);
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        // model provided by Jan Luts for testing
        internal void NormalZeroMixtureTest()
        {
            double sigsq_beta = 1e2;
            Variable<double> beta0 = Variable.GaussianFromMeanAndVariance(0.0, sigsq_beta).Named("beta0");
            Variable<double> nu = Variable.GaussianFromMeanAndVariance(0.0, sigsq_beta).Named("nu");

            double rho = 0.5;
            double A = 0.01;
            double B = 0.01;
            Variable<bool> gamma = Variable.Bernoulli(rho);
            Variable<double> beta1 = Variable.New<double>();
            using (Variable.IfNot(gamma))
            {
                beta1.SetTo(Variable<double>.Random(Gaussian.PointMass(0)));
            }
            using (Variable.If(gamma))
            {
                //beta1.SetTo(Variable.GaussianFromMeanAndVariance(0.0, sigsq_beta));
                beta1.SetTo(Variable.Copy(nu));
            }
            Variable<double> tau = Variable.GammaFromShapeAndScale(A, 1 / B).Named("tau");

            Range item = new Range(2);
            VariableArray<double> meany = Variable.Array<double>(item).Named("meany");
            VariableArray<double> x = Variable.Array<double>(item).Named("x");
            VariableArray<double> y = Variable.Array<double>(item).Named("y");
            meany[item] = beta0 + beta1 * x[item];
            y[item] = Variable.GaussianFromMeanAndPrecision(meany[item], tau);

            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = new GibbsSampling();
            //engine.Algorithm = new VariationalMessagePassing();
            x.ObservedValue = new double[] { 1, 2 };
            y.ObservedValue = new double[] { 1, 2 };
            Console.WriteLine(engine.Infer(beta0));
            Console.WriteLine(engine.Infer(beta1));
        }

        [Fact]
        public void RatingModelTest()
        {
            double[][] ratings = new double[5][]
                {
                    new double[] {4, 5, 5},
                    new double[] {4, 2, 1},
                    new double[] {3, 2, 4},
                    new double[] {4, 4},
                    new double[] {2, 1, 3, 5}
                };
            int[][] items = new int[5][]
                {
                    new int[] {0, 2, 3},
                    new int[] {0, 1, 2},
                    new int[] {0, 2, 3},
                    new int[] {0, 1},
                    new int[] {0, 1, 2, 3}
                };

            Range Y = new Range(4).Named("Y");
            Range K = new Range(2).Named("K");
            var Means = Variable.Array<double>(Y, K).Named("Means");
            Means[Y, K] = Variable.GaussianFromMeanAndVariance(5, 5).ForEach(Y, K);
            var Precs = Variable.Array<double>(Y, K).Named("Precs");
            Precs[Y, K] = Variable.GammaFromShapeAndScale(1, 1).ForEach(Y, K);

            Range U = new Range(items.Length);
            VariableArray<int> RatedItemsCount = Variable.Observed(Util.ArrayInit(items.Length, i => items[i].Length), U);
            Range RatedItems = new Range(RatedItemsCount[U]);
            var Ratings = Variable.Array(Variable.Array<double>(RatedItems), U).Named("Ratings");
            Ratings.ObservedValue = ratings;
            var Items = Variable.Array(Variable.Array<int>(RatedItems), U).Named("Items");
            Items.ObservedValue = items;
            var Theta = Variable.Array<Vector>(U);

            // For each user
            using (Variable.ForEach(U))
            {
                Theta[U] = Variable.DirichletUniform(K);
                // For each rating of that user
                using (Variable.ForEach(RatedItems))
                {
                    // Choose a topic
                    var attitude = Variable.Discrete(Theta[U]).Named("attitude");
                    using (Variable.Switch(attitude))
                    {
                        Ratings[U][RatedItems] = Variable.GaussianFromMeanAndPrecision(Means[Items[U][RatedItems], attitude], Precs[Items[U][RatedItems], attitude]);
                    }
                }
            }

            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(Theta));
            Console.WriteLine(engine.Infer(Means));
        }

        [Fact]
        public void ArrayFromVectorTest()
        {
            var q = new Range(2).Named("q");
            var n = new Range(3).Named("n");
            var f = Variable.Array<Vector>(q).Named("f");
            f[q] = Variable.VectorGaussianFromMeanAndVariance(Vector.Zero(n.SizeAsInt), PositiveDefiniteMatrix.Identity(n.SizeAsInt)).ForEach(q);
            var s = Variable.Array(Variable.Array<double>(n), q).Named("s");
            s[q] = Variable.ArrayFromVector(f[q], n);
        }

        /// <summary>
        /// Has same problem as CloneRangeSwitchTest
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")]
        public void CloneRangeSwitchTest2()
        {
            Range i = new Range(2).Named("i");
            Range j = i.Clone().Named("j");
            var n = Variable.Array<int>(i).Named("n");
            Range vi = new Range(n[i]).Named("vi");

            var A = Variable.Array<Vector>(i).Named("A");
            var index = Variable.Array<int>(i).Named("index");
            var bools = Variable.Array<bool>(i).Named("bools");
            // Fails because the inner Dirichlet is implicitly indexed by Site_i
            using (Variable.ForEach(i))
            {
                A[i] = Variable.DirichletUniform(vi);
                index[i] = Variable.Discrete(A[i]);
                bools[i] = Variable.Bernoulli(0.1);
            }

            n.ObservedValue = new int[] { 4, 3 };
            var obs = Variable.Observed(new Vector[] { Vector.Constant(4, 0.25), Vector.Constant(3, 1.0 / 3) }, j);
            //using (Variable.ForEach(i))
            {
                using (Variable.ForEach(j))
                {
                    using (Variable.Switch(index[j]))
                    {
                        Variable.ConstrainTrue(bools[j]);
                    }
                }
            }

            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(A));
        }

        /// <summary>
        /// This tests passes but it demonstrates a compiler problem that happens to be hidden by PruningTransform.
        /// index[j] has a different dimension for each j, but Hoisting creates a single variable to hold messages for all j.
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")]
        public void CloneRangeSwitchTest()
        {
            Range i = new Range(2).Named("i");
            Range j = i.Clone().Named("j");
            var n = Variable.Array<int>(i).Named("n");
            Range vi = new Range(n[i]).Named("vi");

            var index = Variable.Array<int>(i).Named("index");
            var bools = Variable.Array<bool>(i).Named("bools");
            using (Variable.ForEach(i))
            {
                index[i] = Variable.DiscreteUniform(vi);
                bools[i] = Variable.Bernoulli(0.1);
            }

            n.ObservedValue = new int[] { 4, 3 };
            using (Variable.ForEach(j))
            {
                using (Variable.Switch(index[j]))
                {
                    Variable.ConstrainTrue(bools[j]);
                }
            }

            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(index));
        }

        /// <summary>
        /// Test for a bug found in HoistingTransform
        /// </summary>
        [Fact]
        public void CloneDependentRangeTest()
        {
            Range game = new Range(3).Named("game");
            VariableArray<int> teamsPerGame = Variable.Array<int>(game).Named("teamsPerGame");
            teamsPerGame.ObservedValue = new[] { 2, 2, 3 };
            Range team = new Range(teamsPerGame[game]).Named("team");
            Range teamClone = team.Clone();
            var b = Variable.Bernoulli(0.6);
            using (Variable.ForEach(game))
            {
                using (ForEachBlock outer = Variable.ForEach(team))
                {
                    using (ForEachBlock inner = Variable.ForEach(teamClone))
                    {
                        using (Variable.If(inner.Index != outer.Index))
                        {
                            Variable.ConstrainEqualRandom(b, new Bernoulli(0.4));
                        }
                    }
                }
            }
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(b));
        }

        internal void IntChainTest()
        {
            Range J = new Range(3).Named("J");
            Range N = new Range(2).Named("N");
            var objectNumber = Variable.Array<int>(N).Named("objectNumber");
            var objectNumberPrior = Variable.Array<Vector>(N).Named("objectNumberPrior");
            objectNumberPrior.ObservedValue = Util.ArrayInit(N.SizeAsInt, i => Vector.Constant(J.SizeAsInt, 1.0));
            using (var f = Variable.ForEach(N))
            {
                using (Variable.If(f.Index == 0))
                {
                    objectNumber[N] = Variable.Discrete(J, objectNumberPrior[N]).Named("objNumber"); // todo: make non-uniform
                }
                using (Variable.If(f.Index > 0))
                {
                    var b = objectNumber[f.Index - 1] > 0;
                    using (Variable.If(b))
                    {
                        objectNumber[N] = Variable.Discrete(J, objectNumberPrior[N]).Named("objNumber"); // todo: make non-uniform
                    }
                    using (Variable.IfNot(b))
                    {
                        objectNumber[N] = Variable.Discrete(J, objectNumberPrior[N]).Named("objNumber"); // todo: make non-uniform
                    }
                }
            }
            Variable.ConstrainEqualRandom(objectNumber[0], new Discrete(0.1, 0.2, 0.3));
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(objectNumber));
        }

        [Fact]
        public void ModuloTest()
        {
            Range unwrappedValue = new Range(4).Named("unwrappedValue");
            Variable<int> unwrapped = Variable.DiscreteUniform(unwrappedValue).Named("unwrapped");
            Range wrappedValue = new Range(2).Named("wrappedValue");
            Variable<int> wrapped = Variable.New<int>().Named("wrapped");
            VariableArray<int> modulo2 = Variable.Observed(new int[] { 0, 1, 0, 1 }, unwrappedValue).Named("modulo2");
            modulo2.SetValueRange(wrappedValue);
            using (Variable.Switch(unwrapped))
            {
                wrapped.SetTo(modulo2[unwrapped]);
            }
            wrapped.ObservedValue = 1;
            InferenceEngine engine = new InferenceEngine();
            engine.NumberOfIterations = 1;
            var unwrappedPost1 = engine.Infer(unwrapped);
            engine.NumberOfIterations = 2;
            Console.WriteLine(engine.Infer(unwrapped));
            // test resetting inference
            engine.NumberOfIterations = 1;
            var unwrappedPost2 = engine.Infer<Diffable>(unwrapped);
            Assert.True(unwrappedPost2.MaxDiff(unwrappedPost1) < 1e-10);
        }

        [Fact]
        public void ModuloTest2()
        {
            Range unwrappedValue = new Range(4).Named("unwrappedValue");
            Variable<int> unwrapped = Variable.DiscreteUniform(unwrappedValue).Named("unwrapped");
            Variable<int> wrapped = Variable.New<int>().Named("wrapped");
            Variable<bool> lessThan2 = (unwrapped < 2);
            using (Variable.If(lessThan2))
            {
                wrapped.SetTo(Variable.Copy(unwrapped));
            }
            using (Variable.IfNot(lessThan2))
            {
                // integer subtraction is not implemented
                //wrapped.SetTo(unwrapped-2);
                wrapped.SetTo(unwrapped + (-2));
            }
            wrapped.ObservedValue = 1;
            InferenceEngine engine = new InferenceEngine();
            engine.NumberOfIterations = 1;
            var unwrappedPost1 = engine.Infer(unwrapped);
            engine.NumberOfIterations = 2;
            Console.WriteLine(engine.Infer(unwrapped));
            // test resetting inference
            engine.NumberOfIterations = 1;
            var unwrappedPost2 = engine.Infer<Diffable>(unwrapped);
            Assert.True(unwrappedPost2.MaxDiff(unwrappedPost1) < 1e-10);
        }

        [Fact]
        public void DeepJaggedArrayTest()
        {
            int depth = 5;
            Range[] ranges = new Range[depth];
            for (int i = 0; i < depth; i++)
            {
                ranges[i] = new Range(2);
            }
            var a = Variable<bool>.Array(ranges[0]);
            var b = Variable.Array(a, ranges[1]);
            var c = Variable.Array(b, ranges[2]);
            var d = Variable.Array(c, ranges[3]);
            var e = Variable.Array(d, ranges[4]);
            var f = Variable.Array(d, ranges[4]);
            e[ranges[4]][ranges[3]][ranges[2]][ranges[1]][ranges[0]] = Variable.Bernoulli(0.1).ForEach(ranges[4], ranges[3], ranges[2], ranges[1], ranges[0]);
            f[ranges[4]][ranges[3]][ranges[2]][ranges[1]][ranges[0]] =
                Variable.Bernoulli(0.1).ForEach(ranges[4]).ForEach(ranges[3]).ForEach(ranges[2]).ForEach(ranges[1]).ForEach(ranges[0]);
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(e));
            Console.WriteLine(engine.Infer(f));
        }

        [Fact]
        public void DeepJaggedArrayTest2()
        {
            int depth = 5;
            Range[] ranges = new Range[depth];
            for (int i = 0; i < depth; i++)
            {
                ranges[i] = new Range(2);
            }
            var a = Variable<bool>.Array(ranges[0]);
            var b = Variable.Array(a, ranges[1]);
            var c = Variable.Array(b, ranges[2]);
            var d = Variable.Array(c, ranges[3]);
            var e = Variable.Array(d, ranges[4]);
            Stack<ForEachBlock> blocks = new Stack<ForEachBlock>();
            for (int i = depth - 1; i >= 0; i--)
            {
                blocks.Push(Variable.ForEach(ranges[i]));
            }
            e[ranges[4]][ranges[3]][ranges[2]][ranges[1]][ranges[0]] = Variable.Bernoulli(0.1);
            while (blocks.Count > 0)
            {
                blocks.Pop().CloseBlock();
            }
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(e));
        }

        [Fact]
        public void ArrayUsedAtManyDepths()
        {
            double boolsPrior = 0.1;
            double boolsLike = 0.15;
            double boolsLike2 = 0.2;
            double boolsLike3 = 0.25;
            double boolsLike4 = 0.3;
            double boolsLike5 = 0.35;
            Range outer = new Range(2).Named("outer");
            VariableArray<int> middleSizes = Variable.Constant(new int[] { 2, 3 }, outer).Named("middleSizes");
            Range middle = new Range(middleSizes[outer]).Named("middle");
            var innerSizes = Variable.Constant(new int[][] { new int[] { 1, 2 }, new int[] { 1, 2, 3 } }, outer, middle).Named("innerSizes");
            Range inner = new Range(innerSizes[outer][middle]).Named("inner");
            var bools = Variable.Array(Variable.Array(Variable.Array<bool>(inner), middle), outer).Named("bools");
            using (Variable.ForEach(outer))
            {
                using (Variable.ForEach(middle))
                {
                    using (Variable.ForEach(inner))
                    {
                        bools[outer][middle][inner] = Variable.Bernoulli(boolsPrior);
                        Variable.ConstrainEqualRandom(bools[outer][middle][inner], new Bernoulli(boolsLike));
                    }
                }
            }
            Variable.ConstrainEqualRandom(bools,
                                          (Sampleable<bool[][][]>)Distribution<bool>.Array(Util.ArrayInit(outer.SizeAsInt, i => Util.ArrayInit(middleSizes.ObservedValue[i],
                                                                                                                                                j =>
                                                                                                                                                Util.ArrayInit(
                                                                                                                                                    innerSizes.ObservedValue[i]
                                                                                                                                                        [j],
                                                                                                                                                    k =>
                                                                                                                                                    new Bernoulli(boolsLike2))))));
            VariableArray<int> innerIndices = Variable.Constant(new int[] { 0 }).Named("innerIndices");
            var bools_inner = Variable.Subarray(bools[0][0], innerIndices).Named("bools_inner");
            Variable.ConstrainEqualRandom(bools_inner[innerIndices.Range], new Bernoulli(boolsLike3));
            if (true)
            {
                VariableArray<int> middleIndices = Variable.Constant(new int[] { 0 }).Named("middleIndices");
                middleIndices.SetValueRange(middle);
                var bools_middle = Variable.Subarray(bools[0], middleIndices).Named("bools_middle");
                //Range inner2 = new Range(innerSizes[0][middleIndices[middleIndices.Range]]).Named("inner2");
                Range inner2 = bools_middle[bools_middle.Range].Range;
                Variable.ConstrainEqualRandom(bools_middle[middleIndices.Range][inner2], new Bernoulli(boolsLike4));
            }
            if (true)
            {
                VariableArray<int> outerIndices = Variable.Constant(new int[] { 0 }).Named("outerIndices");
                outerIndices.SetValueRange(outer);
                var bools_outer = Variable.Subarray(bools, outerIndices).Named("bools_outer");
                //Range middle3 = new Range(middleSizes[outerIndices[outerIndices.Range]]);
                //Range inner3 = new Range(innerSizes[outerIndices[outerIndices.Range]][middle3]);
                Range middle3 = bools_outer[bools_outer.Range].Range;
                Range inner3 = bools_outer[bools_outer.Range][middle3].Range;
                Variable.ConstrainEqualRandom(bools_outer[outerIndices.Range][middle3][inner3], new Bernoulli(boolsLike5));
            }

            InferenceEngine engine = new InferenceEngine();
            double z = boolsPrior * boolsLike * boolsLike2 + (1 - boolsPrior) * (1 - boolsLike) * (1 - boolsLike2);
            double boolsPost = boolsPrior * boolsLike * boolsLike2 / z;
            double z000 = boolsPrior * boolsLike * boolsLike2 * boolsLike3 + (1 - boolsPrior) * (1 - boolsLike) * (1 - boolsLike2) * (1 - boolsLike3);
            double boolsPost000 = boolsPrior * boolsLike * boolsLike2 * boolsLike3 / z000;
            //double z00 = boolsPrior*boolsLike
            Bernoulli[][][] boolsExpectedArray = new Bernoulli[outer.SizeAsInt][][];
            for (int i = 0; i < boolsExpectedArray.Length; i++)
            {
                boolsExpectedArray[i] = new Bernoulli[middleSizes.ObservedValue[i]][];
                for (int j = 0; j < boolsExpectedArray[i].Length; j++)
                {
                    boolsExpectedArray[i][j] = new Bernoulli[innerSizes.ObservedValue[i][j]];
                    for (int k = 0; k < boolsExpectedArray[i][j].Length; k++)
                    {
                        double probTrue = boolsPrior * boolsLike * boolsLike2;
                        double probFalse = (1 - boolsPrior) * (1 - boolsLike) * (1 - boolsLike2);
                        if (i == 0)
                        {
                            probTrue *= boolsLike5;
                            probFalse *= 1 - boolsLike5;
                            if (j == 0)
                            {
                                probTrue *= boolsLike4;
                                probFalse *= 1 - boolsLike4;
                                if (k == 0)
                                {
                                    probTrue *= boolsLike3;
                                    probFalse *= 1 - boolsLike3;
                                }
                            }
                        }
                        boolsExpectedArray[i][j][k] = new Bernoulli(probTrue / (probTrue + probFalse));
                    }
                }
            }
            IDistribution<bool[][][]> boolsExpected = Distribution<bool>.Array(boolsExpectedArray);
            object boolsActual = engine.Infer(bools);
            Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsActual, " should be ", boolsExpected));
            Assert.True(boolsExpected.MaxDiff(boolsActual) < 1e-10);
        }

        [Fact]
        public void ArrayUsedAtManyDepths2()
        {
            double boolsPrior = 0.1;
            double boolsLike = 0.15;
            double boolsLike2 = 0.2;
            double boolsLike3 = 0.25;
            double boolsLike4 = 0.3;
            double boolsLike5 = 0.35;
            Range outer = new Range(2).Named("outer");
            VariableArray<int> middleSizes = Variable.Constant(new int[] { 2, 3 }, outer).Named("middleSizes");
            Range middle = new Range(middleSizes[outer]).Named("middle");
            var innerSizes = Variable.Constant(new int[][]
                {
                    new int[] {1, 2}, new int[] {1, 2, 3}
                }, outer, middle).Named("innerSizes");
            Range inner = new Range(innerSizes[outer][middle]).Named("inner");
            var bools = Variable.Array(Variable.Array(Variable.Array<bool>(inner), middle), outer).Named("bools");
            using (Variable.ForEach(outer))
            {
                using (Variable.ForEach(middle))
                {
                    using (Variable.ForEach(inner))
                    {
                        bools[outer][middle][inner] = Variable.Bernoulli(boolsPrior);
                        Variable.ConstrainEqualRandom(bools[outer][middle][inner], new Bernoulli(boolsLike));
                    }
                }
            }
            Variable.ConstrainEqualRandom(bools,
                                          (Sampleable<bool[][][]>)Distribution<bool>.Array(Util.ArrayInit(outer.SizeAsInt, i => Util.ArrayInit(middleSizes.ObservedValue[i],
                                                                                                                                                j =>
                                                                                                                                                Util.ArrayInit(
                                                                                                                                                    innerSizes.ObservedValue[i]
                                                                                                                                                        [j],
                                                                                                                                                    k =>
                                                                                                                                                    new Bernoulli(boolsLike2))))));
            VariableArray<int> innerIndices = Variable.Constant(new int[] { 0 }).Named("innerIndices");
            var bools_inner = Variable.GetItems(bools[0][0], innerIndices).Named("bools_inner");
            Variable.ConstrainEqualRandom(bools_inner[innerIndices.Range], new Bernoulli(boolsLike3));
            if (true)
            {
                VariableArray<int> middleIndices = Variable.Constant(new int[] { 0 }).Named("middleIndices");
                middleIndices.SetValueRange(middle);
                var bools_middle = Variable.GetItems(bools[0], middleIndices).Named("bools_middle");
                //Range inner2 = new Range(innerSizes[0][middleIndices[middleIndices.Range]]).Named("inner2");
                Range inner2 = bools_middle[bools_middle.Range].Range;
                Variable.ConstrainEqualRandom(bools_middle[middleIndices.Range][inner2], new Bernoulli(boolsLike4));
            }
            if (true)
            {
                VariableArray<int> outerIndices = Variable.Constant(new int[] { 0 }).Named("outerIndices");
                outerIndices.SetValueRange(outer);
                var bools_outer = Variable.GetItems(bools, outerIndices).Named("bools_outer");
                //Range middle3 = new Range(middleSizes[outerIndices[outerIndices.Range]]);
                //Range inner3 = new Range(innerSizes[outerIndices[outerIndices.Range]][middle3]);
                Range middle3 = bools_outer[bools_outer.Range].Range;
                Range inner3 = bools_outer[bools_outer.Range][middle3].Range;
                Variable.ConstrainEqualRandom(bools_outer[outerIndices.Range][middle3][inner3], new Bernoulli(boolsLike5));
            }

            InferenceEngine engine = new InferenceEngine();
            double z = boolsPrior * boolsLike * boolsLike2 + (1 - boolsPrior) * (1 - boolsLike) * (1 - boolsLike2);
            double boolsPost = boolsPrior * boolsLike * boolsLike2 / z;
            double z000 = boolsPrior * boolsLike * boolsLike2 * boolsLike3 + (1 - boolsPrior) * (1 - boolsLike) * (1 - boolsLike2) * (1 - boolsLike3);
            double boolsPost000 = boolsPrior * boolsLike * boolsLike2 * boolsLike3 / z000;
            //double z00 = boolsPrior*boolsLike
            Bernoulli[][][] boolsExpectedArray = new Bernoulli[outer.SizeAsInt][][];
            for (int i = 0; i < boolsExpectedArray.Length; i++)
            {
                boolsExpectedArray[i] = new Bernoulli[middleSizes.ObservedValue[i]][];
                for (int j = 0; j < boolsExpectedArray[i].Length; j++)
                {
                    boolsExpectedArray[i][j] = new Bernoulli[innerSizes.ObservedValue[i][j]];
                    for (int k = 0; k < boolsExpectedArray[i][j].Length; k++)
                    {
                        double probTrue = boolsPrior * boolsLike * boolsLike2;
                        double probFalse = (1 - boolsPrior) * (1 - boolsLike) * (1 - boolsLike2);
                        if (i == 0)
                        {
                            probTrue *= boolsLike5;
                            probFalse *= 1 - boolsLike5;
                            if (j == 0)
                            {
                                probTrue *= boolsLike4;
                                probFalse *= 1 - boolsLike4;
                                if (k == 0)
                                {
                                    probTrue *= boolsLike3;
                                    probFalse *= 1 - boolsLike3;
                                }
                            }
                        }
                        boolsExpectedArray[i][j][k] = new Bernoulli(probTrue / (probTrue + probFalse));
                    }
                }
            }
            IDistribution<bool[][][]> boolsExpected = Distribution<bool>.Array(boolsExpectedArray);
            object boolsActual = engine.Infer(bools);
            Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsActual, " should be ", boolsExpected));
            Assert.True(boolsExpected.MaxDiff(boolsActual) < 1e-10);
        }

        [Fact]
        public void ArrayUsedAtManyDepths3()
        {
            var evidence = Variable.Bernoulli(0.5).Named("evidence");
            var block = Variable.If(evidence);

            // Define the range over features
            var featureCount = Variable.Observed(3).Named("FeatureCount");
            var featureRange = new Range(featureCount).Named("FeatureRange");

            // Define the range over classes
            var classCount = Variable.Observed(2).Named("ClassCount");
            var classRange = new Range(classCount).Named("ClassRange");

            // Define the weights
            var Weights = Variable.Array(Variable.Array<double>(featureRange), classRange).Named("Weights");

            // The distributions over weights
            Weights[classRange][featureRange] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(classRange, featureRange);

            // we apply two constraints, at different levels of indexing
            Variable.ConstrainEqualRandom(Weights[classRange][featureRange], new Gaussian(0, 1));
            Variable.ConstrainEqualRandom(Weights[classRange], new GaussianArray(featureCount.ObservedValue, i => new Gaussian(0, 1)));

            block.CloseBlock();

            Gaussian prior = new Gaussian(0, 1);
            double evExpected = prior.GetLogAverageOf(prior) + (prior * prior).GetLogAverageOf(prior);
            evExpected *= featureCount.ObservedValue * classCount.ObservedValue;

            var engine = new InferenceEngine();
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.Equal(evExpected, evActual, 1e-8);
        }

        [Fact]
        public void JaviOliver()
        {
            int ObservedNode = 2; // = 1 (Node1 is observed) or = 2 (Node2 is observed)

            Range k = new Range(2).Named("k");
            Range d = new Range(3).Named("d");


            ///////////////////////
            // Node 1
            ///////////////////////

            // Mixture component means
            var means1 = Variable.Array(Variable.Array<double>(d), k);
            means1[k][d] = Variable.GaussianFromMeanAndPrecision(2.0, 0.01).ForEach(k, d);

            VariableArray<Vector> vmeans1 = Variable.Array<Vector>(k).Named("vmeans1");
            vmeans1[k] = Variable.Vector(means1[k]);

            // Mixture component precisions
            VariableArray<PositiveDefiniteMatrix> precs = Variable.Array<PositiveDefiniteMatrix>(k).Named("precs");

            precs[k] = Variable.WishartFromShapeAndScale(100.0, PositiveDefiniteMatrix.IdentityScaledBy(d.SizeAsInt, 0.01)).ForEach(k);
            //precs.InitialiseTo(Distribution<PositiveDefiniteMatrix>.Array(Util.ArrayInit(k.SizeAsInt, i => Wishart.FromShapeAndScale(100.0, PositiveDefiniteMatrix.IdentityScaledBy(d.SizeAsInt, 0.01)))));

            // Mixture weights 
            Variable<Vector> weights = Variable.Dirichlet(k, new double[] { 1, 1 }).Named("weights");

            // Create a variable array which will hold the data
            Range n = new Range(100).Named("n");
            VariableArray<Vector> data = Variable.Array<Vector>(n).Named("x");

            // Create latent indicator variable for each data point
            VariableArray<int> z = Variable.Array<int>(n).Named("z");

            // The mixture of Gaussians model
            using (Variable.ForEach(n))
            {
                z[n] = Variable.Discrete(weights);
                using (Variable.Switch(z[n]))
                {
                    data[n] = Variable.VectorGaussianFromMeanAndPrecision(vmeans1[z[n]], precs[z[n]]);
                }
            }

            // Attach some generated data
            if (ObservedNode == 1)
                data.ObservedValue = GenerateData(n.SizeAsInt);

            // Initialise messages randomly so as to break symmetry
            Discrete[] zinit = new Discrete[n.SizeAsInt];
            for (int i = 0; i < zinit.Length; i++)
                zinit[i] = Discrete.PointMass(Rand.Int(k.SizeAsInt), k.SizeAsInt);
            z.InitialiseTo(Distribution<int>.Array(zinit));


            /////////////////////////////
            // Node 2
            /////////////////////////////

            // Mixture component means
            var means2 = Variable.Array(Variable.Array<double>(d), k).Named("mean2");

            // Set the relation among variables
            // means2[k][d] = means1[k][d] + 1.0;      // This works!

            means2[0][0] = means1[0][0] + 1.0; // This not...
            means2[0][1] = means1[0][1] + 1.0;
            means2[0][2] = means1[0][2] + 1.0;

            means2[1][0] = means1[1][0] + 1.0;
            means2[1][1] = means1[1][1] + 1.0;
            means2[1][2] = means1[1][2] + 1.0;


            VariableArray<Vector> vmeans2 = Variable.Array<Vector>(k).Named("vmeans2");
            vmeans2[k] = Variable.Vector(means2[k]);


            // Mixture component precisions
            VariableArray<PositiveDefiniteMatrix> precs2 = Variable.Array<PositiveDefiniteMatrix>(k).Named("precs2");
            precs2[k] = Variable.WishartFromShapeAndScale(100.0, PositiveDefiniteMatrix.IdentityScaledBy(d.SizeAsInt, 0.01)).ForEach(k);

            // Mixture weights 
            Variable<Vector> weights2 = Variable.Dirichlet(k, new double[] { 1, 1 }).Named("weights2");

            // Create a variable array which will hold the data
            Range n2 = new Range(100).Named("n2");
            VariableArray<Vector> data2 = Variable.Array<Vector>(n2).Named("x2");

            // Create latent indicator variable for each data point
            VariableArray<int> z2 = Variable.Array<int>(n2).Named("z2");

            // The mixture of Gaussians model
            using (Variable.ForEach(n2))
            {
                z2[n2] = Variable.Discrete(weights2);
                using (Variable.Switch(z2[n2]))
                {
                    data2[n2] = Variable.VectorGaussianFromMeanAndPrecision(vmeans2[z2[n2]], precs2[z2[n2]]);
                }
            }

            // Attach some generated data
            if (ObservedNode == 2)
                data2.ObservedValue = GenerateData(n2.SizeAsInt);

            // Initialise messages randomly so as to break symmetry
            Discrete[] zinit2 = new Discrete[n2.SizeAsInt];
            for (int i = 0; i < zinit2.Length; i++)
                zinit2[i] = Discrete.PointMass(Rand.Int(k.SizeAsInt), k.SizeAsInt);
            z2.InitialiseTo(Distribution<int>.Array(zinit2));


            /////////////////////////////


            // The inference
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.NumberOfIterations = 50;
            //ie.Compiler.GenerateInMemory = false;
            //ie.Compiler.WriteSourceFiles = true;
            //ie.Compiler.IncludeDebugInformation = true;

            Console.WriteLine("Dist over pi=" + ie.Infer(weights));
            Console.WriteLine("Dist over means=\n" + ie.Infer(vmeans1));
            Console.WriteLine("Dist over precs=\n" + ie.Infer(precs));

            Console.WriteLine("Dist over pi=" + ie.Infer(weights2));
            Console.WriteLine("Dist over means=\n" + ie.Infer(vmeans2));
            Console.WriteLine("Dist over precs=\n" + ie.Infer(precs2));
            IList<Wishart> precs2Actual = ie.Infer<IList<Wishart>>(precs2);
            double[] shapeExpected = new double[] { 116, 133 };
            int j = (precs2Actual[0].Shape < precs2Actual[1].Shape) ? 0 : 1;
            Assert.True(System.Math.Abs(precs2Actual[0].Shape - shapeExpected[j]) < 1);
            Assert.True(System.Math.Abs(precs2Actual[1].Shape - shapeExpected[1 - j]) < 1);
        }


        /// <summary>
        /// Generates a data set from a particular true model.
        /// </summary>
        public Vector[] GenerateData(int nData)
        {
            Vector trueM1 = Vector.FromArray(2.0, 3.0, 4.0);
            Vector trueM2 = Vector.FromArray(7.0, 5.0, 5.0);

            double[] auxP1 =
                {
                    3.0, 0.2, 0.2,
                    0.2, 2.0, 0.3,
                    0.2, 0.3, 2.5
                };
            double[] auxP2 =
                {
                    2.0, 0.4, 0.4,
                    0.4, 1.5, 0.1,
                    0.4, 0.1, 2.0
                };

            Matrix MatrixP1 = new Matrix(3, 3, auxP1);
            Matrix MatrixP2 = new Matrix(3, 3, auxP2);

            PositiveDefiniteMatrix trueP1 = new PositiveDefiniteMatrix(MatrixP1);
            PositiveDefiniteMatrix trueP2 = new PositiveDefiniteMatrix(MatrixP2);

            VectorGaussian trueVG1 = VectorGaussian.FromMeanAndPrecision(trueM1, trueP1);
            VectorGaussian trueVG2 = VectorGaussian.FromMeanAndPrecision(trueM2, trueP2);

            double truePi = 0.6;
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

        [Fact]
        public void ConstantAndStochasticArrayElementsError()
        {
            try
            {
                Range item = new Range(2).Named("item");
                var x = Variable.Array<double>(item).Named("x");
                x[0] = Variable.Constant(0.0);
                x[1] = Variable.GaussianFromMeanAndPrecision(0, 1);
                InferenceEngine engine = new InferenceEngine();
                object xActual = engine.Infer(x);
                IDistribution<double[]> xExpected = Distribution<double>.Array(new Gaussian[] { Gaussian.PointMass(0), Gaussian.FromMeanAndVariance(0, 1) });
                Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
            }
            catch (CompilationFailedException ex)
            {
                Console.WriteLine("Correctly threw exception: " + ex);
            }
        }

        [Fact]
        public void VariableUsedTwiceInFactorError()

        {
            Assert.Throws<CompilationFailedException>(() =>
            {

                Variable<bool> x = Variable.Bernoulli(0.1).Named("x");
                Variable<bool> y = x & x;
                InferenceEngine engine = new InferenceEngine();
                Console.WriteLine("y = {0}", engine.Infer(y));

            });
        }

        [Fact]
        public void LiteralIndexingLocalConditionTest()
        {
            double bPrior = 0.2;
            double bLike = 0.6;
            Range i = new Range(2).Named("i");
            Range j = new Range(2).Named("j");
            VariableArray2D<bool> bools = Variable.Array<bool>(i, j).Named("bools");
            bools[i, j] = Variable.Bernoulli(bPrior).ForEach(i, j);
            VariableArray2D<bool> condition = Variable.Array<bool>(i, j).Named("condition");
            condition.ObservedValue = new bool[,] { { false, false }, { false, true } };
            using (ForEachBlock fbi = Variable.ForEach(i))
            {
                using (Variable.If(fbi.Index > 0))
                {
                    using (ForEachBlock fbj = Variable.ForEach(j))
                    {
                        using (Variable.If(condition[fbi.Index, fbj.Index]))
                        {
                            Variable.ConstrainEqualRandom(bools[fbi.Index - 1, 1], new Bernoulli(bLike));
                        }
                    }
                }
            }
            //Variable.ConstrainEqualRandom(bools[1, 1], new Bernoulli(bLike));
            InferenceEngine engine = new InferenceEngine();
            double sumT = bPrior * bLike + (1 - bPrior) * (1 - bLike);
            double Z = sumT;
            double bExpected01 = bPrior * (bLike) / Z;
            IDistribution<bool[,]> bExpected = Distribution<bool>.Array(new Bernoulli[,]
                {
                    {new Bernoulli(bPrior), new Bernoulli(bExpected01)},
                    {new Bernoulli(bPrior), new Bernoulli(bPrior)},
                });
            object bActual = engine.Infer(bools);
            Console.WriteLine(StringUtil.JoinColumns("b = ", bActual, " should be ", bExpected));
            Assert.True(bExpected.MaxDiff(bActual) < 1e-10);
        }

        [Fact]
        public void LiteralIndexingLocalConditionTest2()
        {
            double bPrior = 0.2;
            double bLike = 0.6;
            Range i = new Range(2).Named("i");
            Range j = new Range(2).Named("j");
            VariableArray2D<bool> bools = Variable.Array<bool>(i, j).Named("bools");
            bools[i, j] = Variable.Bernoulli(bPrior).ForEach(i, j);
            VariableArray2D<bool> condition = Variable.Array<bool>(i, j).Named("condition");
            condition.ObservedValue = new bool[,] { { false, false }, { false, true } };
            using (ForEachBlock fbi = Variable.ForEach(i))
            {
                using (ForEachBlock fbj = Variable.ForEach(j))
                {
                    using (Variable.If(fbi.Index > 0))
                    {
                        using (Variable.If(condition[fbi.Index, fbj.Index]))
                        {
                            Variable.ConstrainEqualRandom(bools[fbi.Index - 1, 1], new Bernoulli(bLike));
                        }
                    }
                }
            }
            //Variable.ConstrainEqualRandom(bools[1, 1], new Bernoulli(bLike));
            InferenceEngine engine = new InferenceEngine();
            double sumT = bPrior * bLike + (1 - bPrior) * (1 - bLike);
            double Z = sumT;
            double bExpected01 = bPrior * (bLike) / Z;
            IDistribution<bool[,]> bExpected = Distribution<bool>.Array(new Bernoulli[,]
                {
                    {new Bernoulli(bPrior), new Bernoulli(bExpected01)},
                    {new Bernoulli(bPrior), new Bernoulli(bPrior)},
                });
            object bActual = engine.Infer(bools);
            Console.WriteLine(StringUtil.JoinColumns("b = ", bActual, " should be ", bExpected));
            Assert.True(bExpected.MaxDiff(bActual) < 1e-10);
        }

        [Fact]
        public void ReplicationLocalConditionTest()
        {
            double bPrior = 0.2;
            double bLike = 0.6;
            Range i = new Range(2).Named("i");
            Range j = new Range(3).Named("j");
            Range i2 = new Range(2).Named("i2");
            VariableArray<bool> bools = Variable.Array<bool>(i2).Named("bools");
            bools[i2] = Variable.Bernoulli(bPrior).ForEach(i2);
            VariableArray2D<bool> condition = Variable.Array<bool>(i, j).Named("condition");
            condition.ObservedValue = new bool[,] { { true, false, false }, { true, true, true } };
            using (ForEachBlock fbi = Variable.ForEach(i))
            {
                using (ForEachBlock fbj = Variable.ForEach(j))
                {
                    using (ForEachBlock fbi2 = Variable.ForEach(i2))
                    {
                        using (Variable.If(fbi2.Index == 0))
                        {
                            using (Variable.If(fbi.Index < 1))
                            {
                                using (Variable.If(condition[fbi.Index, fbj.Index]))
                                {
                                    Variable.ConstrainEqualRandom(bools[fbi.Index], new Bernoulli(bLike));
                                }
                            }
                        }
                    }
                }
            }
            InferenceEngine engine = new InferenceEngine();
            double sumT = bPrior * bLike + (1 - bPrior) * (1 - bLike);
            double Z = sumT;
            double bExpected01 = bPrior * (bLike) / Z;
            IDistribution<bool[]> bExpected = Distribution<bool>.Array(new Bernoulli[]
                {
                    new Bernoulli(bExpected01), new Bernoulli(bPrior),
                });
            object bActual = engine.Infer(bools);
            Console.WriteLine(StringUtil.JoinColumns("b = ", bActual, " should be ", bExpected));
            Assert.True(bExpected.MaxDiff(bActual) < 1e-10);
        }

        internal void CardTest()
        {
            int numPlayers = 3;
            int numCards = 6;
            int cardsPerPlayer = numCards / numPlayers;

            Variable<bool>[][] hasCard = new Variable<bool>[numPlayers][];
            for (int p = 0; p < numPlayers; p++)
            {
                hasCard[p] = new Variable<bool>[numCards];
                for (int c = 0; c < numCards; c++)
                {
                    hasCard[p][c] = Variable.Bernoulli(1.0 / numPlayers);
                }
            }

            // Each player has a fixed number of cards
            for (int p = 0; p < numPlayers; p++)
            {
                Variable<int> playerCards = Variable.DiscreteUniform(1);
                //Variable<int> playerCards = Variable.Observed<int>(0);

                for (int c = 0; c < numCards; c++)
                {
                    Variable<int> nextSum = Variable.New<int>();

                    using (Variable.If(hasCard[p][c]))
                        nextSum.SetTo(playerCards + 1);
                    using (Variable.IfNot(hasCard[p][c]))
                        nextSum.SetTo(playerCards + 0);

                    playerCards = nextSum;
                }
                Variable.ConstrainTrue(playerCards == cardsPerPlayer);
            }

            // Every card is with one player
            for (int c = 0; c < numCards; c++)
            {
                Variable<int> numPlayersWithCard = Variable.DiscreteUniform(1);

                // Count how many players have this card
                for (int p = 0; p < numPlayers; p++)
                {
                    Variable<int> nextSum = Variable.New<int>();

                    using (Variable.If(hasCard[p][c]))
                        nextSum.SetTo(numPlayersWithCard + 1);
                    using (Variable.IfNot(hasCard[p][c]))
                        nextSum.SetTo(numPlayersWithCard + 0);

                    numPlayersWithCard = nextSum;
                }

                // Must be equal to 1
                Variable.ConstrainTrue(numPlayersWithCard == 1);
            }

            InferenceEngine ie = new InferenceEngine();
            Console.WriteLine("Card distribution:\n{0}", ie.Infer(hasCard[0][0]));
        }

        internal void CardTest2()
        {
            int numPlayers = 3;
            int numCards = 6;
            int cardsPerPlayer = numCards / numPlayers;

            Variable<bool>[][] hasCard = new Variable<bool>[numPlayers][];
            for (int p = 0; p < numPlayers; p++)
            {
                hasCard[p] = new Variable<bool>[numCards];
                for (int c = 0; c < numCards; c++)
                {
                    hasCard[p][c] = Variable.Bernoulli(1.0 / numPlayers);
                }
            }

            // Each player has a fixed number of cards
            for (int p = 0; p < numPlayers; p++)
            {
                Variable<int> playerCards = TrueCount(hasCard[p], cardsPerPlayer);
                Variable.ConstrainTrue(playerCards == cardsPerPlayer);
            }

            // Run!
            InferenceEngine ie = new InferenceEngine();
            Console.WriteLine("Card distribution:\n{0}", ie.Infer(hasCard[0][0]));
        }

        public Variable<int> TrueCount(IList<Variable<bool>> bools, int maximumCount)
        {
            int threshold = maximumCount + 1;
            Range r = new Range(threshold + 1);
            Variable<int> count = Variable.Observed(0);
            count.SetValueRange(r);
            foreach (Variable<bool> b in bools)
            {
                Variable<int> nextCount = Variable.New<int>();
                using (Variable.If(b))
                {
                    for (int i = 0; i <= threshold; i++)
                    {
                        using (Variable.Case(count, i))
                        {
                            nextCount.SetTo(System.Math.Min(i + 1, threshold));
                        }
                    }
                }
                using (Variable.IfNot(b))
                {
                    nextCount.SetTo(Variable.Copy(count));
                }
                count = nextCount;
            }
            return count;
        }

        [Fact]
        public void CardTest3()
        {
            int numPlayers = 4;
            int numCards = 6;
            int cardsPerPlayer = numCards / numPlayers;

            Variable<bool>[][] hasCard = new Variable<bool>[numPlayers][];
            for (int p = 0; p < numPlayers; p++)
            {
                hasCard[p] = new Variable<bool>[numCards];
                for (int c = 0; c < numCards; c++)
                {
                    hasCard[p][c] = Variable.Bernoulli(1.0 / numPlayers);
                }
            }

            // Each player has a fixed number of cards
            //Variable<int>[] playerCards = new Variable<int>[numPlayers];
            for (int p = 0; p < numPlayers; p++)
            {
                Variable<int> playerCards = Variable.DiscreteUniform(1);

                // Count how many cards player p has
                for (int c = 0; c < numCards; c++)
                {
                    playerCards.Name = "playerCards" + p + "_" + c;
                    //playerCards.AddAttribute(new DivideMessages(false));
                    Variable<int> nextSum = Variable.New<int>();

                    var mustIncrement = hasCard[p][c] & playerCards < cardsPerPlayer + 1;
                    using (Variable.If(mustIncrement))
                        nextSum.SetTo(playerCards + 1);
                    using (Variable.IfNot(mustIncrement))
                        nextSum.SetTo(playerCards + 0);

                    playerCards = nextSum;
                }
                playerCards.Name = "playerCards" + p;

                // Must be correct
                Variable.ConstrainEqualRandom(playerCards == cardsPerPlayer, new Bernoulli(0.9999));
            }

            // Every card is with one player
            for (int c = 0; c < numCards; c++)
            {
                Variable<int> numPlayersWithCard = Variable.DiscreteUniform(1);

                // Count how many players have this card
                for (int p = 0; p < numPlayers; p++)
                {
                    numPlayersWithCard.Name = "numPlayersWithCard" + c + "_" + p;
                    //numPlayersWithCard.AddAttribute(new DivideMessages(false));
                    Variable<int> nextSum = Variable.New<int>();

                    var mustIncrement = hasCard[p][c] & numPlayersWithCard < 2;
                    using (Variable.If(mustIncrement))
                        nextSum.SetTo(numPlayersWithCard + 1);
                    using (Variable.IfNot(mustIncrement))
                        nextSum.SetTo(numPlayersWithCard + 0);

                    numPlayersWithCard = nextSum;
                }
                numPlayersWithCard.Name = "numPlayersWithCard" + c;

                // Must be equal to 1
                Variable.ConstrainEqualRandom(numPlayersWithCard == 1, new Bernoulli(0.9999));
            }

            // Player 1 has card 1
            hasCard[1][1].ObservedValue = true;

            InferenceEngine ie = new InferenceEngine();
            // What is the probability of me having card 0?
            Console.WriteLine("Card distribution:\n{0}", ie.Infer(hasCard[0][0]));
        }

        [Fact]
        public void LocalIndexTest()
        {
            Range item = new Range(2).Named("item");
            VariableArray<bool> x = Variable.Array<bool>(item).Named("x");
            VariableArray<bool> y = Variable.Array<bool>(item).Named("y");
            using (Variable.ForEach(item))
            {
                x[item] = Variable.Bernoulli(0.1);
                Variable<int> index = Variable.Constant(0) + Variable.Constant(0);
                index.Name = "index";
                y[item] = !x[index];
            }

            InferenceEngine engine = new InferenceEngine();
            object yActual = engine.Infer(y);
            IDistribution<bool[]> yExpected = Distribution<bool>.Array(new Bernoulli[] { new Bernoulli(0.9), new Bernoulli(0.9) });
            Console.WriteLine(StringUtil.JoinColumns("y = ", yActual, " should be ", yExpected));
            Assert.True(yExpected.MaxDiff(yActual) < 1e-10);
        }

        [Fact]
        public void BuildTwiceTest()
        {
            Variable<double> const1 = Variable.Constant(1.0).Named("const1");
            Variable<double> const2 = Variable.Constant(1.0).Named("const2");
            Variable<double> zero = Variable.Constant(0.0).Named("zero");
            Variable<double> x = Variable.GaussianFromMeanAndPrecision(zero, const1).Named("x");
            //Variable<double> y = Variable.GaussianFromMeanAndPrecision(x, const2).Named("y");
            InferenceEngine engine = new InferenceEngine();
            Gaussian xPost = engine.Infer<Gaussian>(x);
            Variable<double> z = Variable.GaussianFromMeanAndPrecision(zero, const2).Named("z");
            Variable.ConstrainPositive(z - x);
            InferenceEngine engine2 = new InferenceEngine();
            Gaussian xPost2 = engine2.Infer<Gaussian>(z);
        }

        [Fact]
        public void CloneRangeFibonacciTest()
        {
            int n = 10;
            Range row = new Range(n);
            VariableArray<double> x = Variable.Array<double>(row).Named("x");
            x[row] = Variable.GaussianFromMeanAndPrecision(0, 0).ForEach(row);
            Variable.ConstrainEqual(x[0], 1);
            Variable.ConstrainEqual(x[1], 1);
            using (ForEachBlock fb1 = Variable.ForEach(row))
            {
                using (ForEachBlock fb2 = Variable.ForEach(row.Clone()))
                {
                    using (ForEachBlock fb3 = Variable.ForEach(row.Clone()))
                    {
                        using (Variable.If((fb2.Index == fb1.Index + 1) & (fb3.Index == fb2.Index + 1)))
                        {
                            Variable.ConstrainEqual(x[fb3.Index], x[fb1.Index] + x[fb2.Index]);
                        }
                    }
                }
            }
            InferenceEngine engine = new InferenceEngine();
            IList<Gaussian> xActual = engine.Infer<IList<Gaussian>>(x);
            Console.WriteLine(xActual);
            Assert.True(xActual[9].Point == 55);
        }

        // This test only works if you disable the check for duplicate definitions.
        //[Fact]
        internal void CloneRangeFibonacciTest2()
        {
            int n = 10;
            Range row = new Range(n);
            VariableArray<double> x = Variable.Array<double>(row).Named("x");
            x[0] = Variable.Constant(1.0);
            x[1] = Variable.Constant(1.0);
            using (ForEachBlock fb1 = Variable.ForEach(row))
            {
                using (ForEachBlock fb2 = Variable.ForEach(row.Clone()))
                {
                    using (ForEachBlock fb3 = Variable.ForEach(row.Clone()))
                    {
                        using (Variable.If((fb2.Index == fb1.Index + 1) & (fb3.Index == fb2.Index + 1)))
                        {
                            x[fb3.Index] = x[fb1.Index] + x[fb2.Index];
                        }
                    }
                }
            }
            InferenceEngine engine = new InferenceEngine();
            double[] xActual = engine.Infer<double[]>(x);
            Console.WriteLine(StringUtil.ArrayToString(xActual));
            Assert.True(xActual[9] == 55);
        }

        [Fact]
        public void CloneRangeBlockModelTest()
        {
            CloneRangeBlockModel(true);
            CloneRangeBlockModel(false);
        }

        // TODO: Lack of optimization: z2_cond_z1_B is not being substituted by CopyProp because the rhs is a different array type (Discrete[])
        private void CloneRangeBlockModel(bool initialise)
        {
            // Observed interaction matrix
            var YObs = new bool[5][];
            bool useDataSet1 = true;
            if (useDataSet1)
            {
                YObs[0] = new bool[] { false, true, true, false, false };
                YObs[1] = new bool[] { true, false, true, false, false };
                YObs[2] = new bool[] { true, true, false, false, false };
                YObs[3] = new bool[] { false, false, false, false, true };
                YObs[4] = new bool[] { false, false, false, true, false };
            }
            else
            {
                YObs[0] = new bool[] { true, true, true, false, false };
                YObs[1] = new bool[] { true, true, true, false, false };
                YObs[2] = new bool[] { true, true, true, false, false };
                YObs[3] = new bool[] { false, false, false, true, true };
                YObs[4] = new bool[] { false, false, false, true, true };
            }
            int K = 2; // Number of blocks
            int N = YObs.Length; // Number of nodes

            // Ranges
            Range p = new Range(N).Named("p"); // Range for initiator
            Range q = p.Clone().Named("q"); // Range for receiver
            Range kp = new Range(K).Named("kp"); // Range for initiator group membership
            Range kq = kp.Clone().Named("kq"); // Range for receiver group membership

            // The model
            var Y = Variable.Array(Variable.Array<bool>(q), p); // Interaction matrix
            var pi = Variable.Array<Vector>(p).Named("pi"); // Block-membership probability vector
            pi[p] = Variable.DirichletUniform(kp).ForEach(p);
            var B = Variable.Array<double>(kp, kq).Named("B"); // Link probability matrix
            B[kp, kq] = Variable.Beta(1, 1).ForEach(kp, kq);

            using (Variable.ForEach(p))
            {
                using (Variable.ForEach(q))
                {
                    var z1 = Variable.Discrete(pi[p]).Named("z1"); // Draw initiator membership indicator
                    var z2 = Variable.Discrete(pi[q]).Named("z2"); // Draw receiver membership indicator
                    z2.SetValueRange(kq);
                    using (Variable.Switch(z1))
                    using (Variable.Switch(z2))
                        Y[p][q] = Variable.Bernoulli(B[z1, z2]); // Sample interaction value
                }
            }

            if (initialise)
            {
                // Initialise to break symmetry
                Rand.Restart(12347);
                var piInit = new Dirichlet[N];
                for (int i = 0; i < N; i++)
                {
                    Vector v = Vector.Zero(K);
                    for (int j = 0; j < K; j++)
                        v[j] = 1 + Rand.Double();
                    piInit[i] = new Dirichlet(v);
                }
                pi.InitialiseTo((DirichletArray)Distribution<Vector>.Array(piInit));
            }
            //B.InitialiseTo((DistributionArray2D<Beta,double>)Distribution<double>.Array(new Beta[,] { { new Beta(10,1), new Beta(1,10) }, { new Beta(1,10), new Beta(10,1) } }));

            // Hook up the data
            Y.ObservedValue = YObs;

            // Infer
            var engine = new InferenceEngine(new VariationalMessagePassing());
            var posteriorPi = engine.Infer<IList<Dirichlet>>(pi);
            var posteriorB = engine.Infer(B);
            var piExpected =
                Distribution<Vector>.Array(new Dirichlet[] { new Dirichlet(6.334, 5.666), new Dirichlet(6.333, 5.667), new Dirichlet(6.345, 5.655), new Dirichlet(4.291, 7.709), new Dirichlet(4.302, 7.698) });
            Console.WriteLine(StringUtil.JoinColumns("pi = ", posteriorPi, " should be ", piExpected));
            Console.WriteLine(StringUtil.JoinColumns("B = ", posteriorB));
            int indexOfMax1 = posteriorPi[0].PseudoCount.IndexOfMaximum();
            Assert.Equal(posteriorPi[1].PseudoCount.IndexOfMaximum(), indexOfMax1);
            Assert.Equal(posteriorPi[2].PseudoCount.IndexOfMaximum(), indexOfMax1);
            int indexOfMax2 = posteriorPi[3].PseudoCount.IndexOfMaximum();
            Assert.Equal(posteriorPi[4].PseudoCount.IndexOfMaximum(), indexOfMax2);
        }

        [Fact]
        public void CopyLocalVariable()
        {
            Range item = new Range(2).Named("item");
            VariableArray<bool> array = Variable.Array<bool>(item).Named("array");
            using (Variable.ForEach(item))
            {
                Variable<bool> b = Variable.Bernoulli(0.1).Named("b");
                Variable<bool> c = Variable.Copy(b).Named("c");
                array[item] = Variable.Copy(c);
            }
            InferenceEngine engine = new InferenceEngine();
            IDistribution<bool[]> arrayExpected = Distribution<bool>.Array(new Bernoulli[] { new Bernoulli(0.1), new Bernoulli(0.1) });
            object arrayActual = engine.Infer(array);
            Console.WriteLine(StringUtil.JoinColumns("array = ", arrayActual, " should be ", arrayExpected));
            Assert.True(arrayExpected.MaxDiff(arrayActual) < 1e-10);
        }

        [Fact]
        public void JaggedRandomTest()
        {
            JaggedRandom(new ExpectationPropagation());
            JaggedRandom(new VariationalMessagePassing());
        }

        [Fact]
        public void GibbsJaggedRandomTest()
        {
            JaggedRandom(new GibbsSampling());
        }

        private void JaggedRandom(IAlgorithm algorithm)
        {
            Range outer = new Range(2).Named("outer");
            Range inner = new Range(1).Named("inner");
            var priors = Variable.Array<BernoulliArray>(outer).Named("priors");
            priors.ObservedValue = Util.ArrayInit(outer.SizeAsInt, i => new BernoulliArray(inner.SizeAsInt, j => new Bernoulli(0.1 * (j + i + 1))));
            var array = Variable.Array(Variable.Array<bool>(inner), outer).Named("array");
            array[outer].SetTo(Variable<bool[]>.Random(priors[outer]));
            InferenceEngine engine = new InferenceEngine(algorithm);
            object actual = engine.Infer(array);
            BernoulliArrayArray expected = new BernoulliArrayArray(priors.ObservedValue);
            Console.WriteLine(StringUtil.JoinColumns("array = ", actual, " should be ", expected));
            Assert.True(expected.MaxDiff(actual) < 1e-10);
        }

        [Fact]
        public void RangeUsedAsArgumentInsideForEach()
        {
            Range item = new Range(2).Named("item");
            VariableArray<int> x = Variable.Array<int>(item).Named("x");
            using (Variable.ForEach(item))
            {
                x[item] = Variable.DiscreteUniform(item);
            }

            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine("x = {0}", engine.Infer(x));
        }

        [Fact]
        public void FriendGraph()
        {
            Range uRange = new Range(1).Named("u");
            ;
            Range fVaryRange = new Range(1).Named("fVary");
            Range sRange = new Range(1).Named("s");
            Range eRange = new Range(1).Named("e");
            Range cRange = new Range(1).Named("c");
            var edgeLabel2 = Variable.Array<int>(eRange).Named("edgeLabel2");
            edgeLabel2[eRange] = Variable.Discrete(1.0).ForEach(eRange);
            var cArray = Variable.Array<bool>(cRange).Named("cArray");
            cArray[cRange] = Variable.Bernoulli(0.3).ForEach(cRange);

            var edgeIndexData = Variable.Constant(new int[][] { new int[] { 0 } }, uRange, fVaryRange).Named("edgeIndexData");
            var f = Variable.Array(Variable.Array<int>(sRange), uRange).Named("f");
            var psi = Variable.Array<Vector>(uRange).Named("psi");
            psi.ObservedValue = new Vector[] { Vector.FromArray(1.0) };
            using (Variable.ForEach(uRange))
            {
                using (Variable.ForEach(sRange))
                {
                    f[uRange][sRange] = Variable.Discrete(psi[uRange]).Attrib(new ValueRange(fVaryRange));
                    var currentFriend = f[uRange][sRange];
                    using (Variable.Switch(currentFriend))
                    {
                        var currentEdge = (edgeIndexData[uRange][currentFriend]).Attrib(new ValueRange(eRange));
                        using (Variable.Switch(currentEdge))
                        {
                            var currentLabel = ((edgeLabel2[currentEdge])).Attrib(new ValueRange(cRange));
                            using (Variable.Switch(currentLabel))
                            {
                                Variable.ConstrainTrue(cArray[currentLabel]);
                            }
                        }
                    }
                }
            }

            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(edgeLabel2));
        }

        [Fact]
        public void SetToError()
        {
            Assert.Throws<InvalidOperationException>(() =>
            {
                Variable<bool> temp = Variable.Bernoulli(0.1).Named("temp");
                Variable<bool> x = Variable.New<bool>().Named("x");
                x.SetTo(temp);
                InferenceEngine engine = new InferenceEngine();
                Console.WriteLine(engine.Infer(temp)); // illegal

            });
        }

        [Fact]
        public void SetToConstraintError()
        {
            Assert.Throws<InvalidOperationException>(() =>
            {
                Variable<bool> temp = Variable.Bernoulli(0.1).Named("temp");
                Variable.ConstrainTrue(temp);
                Variable<bool> x = Variable.New<bool>().Named("x");
                x.SetTo(temp); // illegal
                InferenceEngine engine = new InferenceEngine();
                Console.WriteLine(engine.Infer(x));

            });
        }

        [Fact]
        public void SetToConstraintError2()

        {
            Assert.Throws<InvalidOperationException>(() =>
            {

                var R = new Range(2);
                var array = Variable.Array<double>(R);
                using (Variable.ForEach(R))
                {
                    var x = Variable.GaussianFromMeanAndPrecision(0.0, 1.0);
                    Variable.ConstrainTrue(x > 0);
                    array[R] = x;
                }
                var ie = new InferenceEngine();
                // ie.ShowFactorGraph = true;
                // ie.ShowMsl = true;
                Console.WriteLine("array = " + ie.Infer(array));

            });
        }

        [Fact]
        public void SetToError2()

        {
            Assert.Throws<InvalidOperationException>(() =>
            {

                Variable<bool> a = Variable.Bernoulli(1).Named("a");
                Variable<bool> x = Variable.New<bool>().Named("x");
                Variable<bool> temp = Variable.New<bool>().Named("temp");
                using (Variable.If(a))
                {
                    temp.SetTo(Variable.Bernoulli(0.1));
                    x.SetTo(temp);
                }
                using (Variable.IfNot(a))
                {
                    x.SetTo(Variable.Bernoulli(0.2));
                }
                InferenceEngine engine = new InferenceEngine();
                Console.WriteLine(engine.Infer(temp)); // illegal

            });
        }

        [Fact]
        public void SetToTest()
        {
            Variable<bool> temp = Variable.Bernoulli(0.5);
            Variable<bool> b = Variable.Bernoulli(0.1);
            Variable<bool> x = Variable.New<bool>();
            Variable.ConstrainTrue(temp);
            // this should not fail since Variable.Copy is inserted automatically
            using (Variable.If(b))
                x.SetTo(temp);
            using (Variable.IfNot(b))
                x.SetTo(!temp);
            var engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(temp));
        }

        [Fact]
        public void ImplicitForEachTest()
        {
            Range r = new Range(2);
            var v = Variable.Constant(true);
            var va = Variable.Array<bool>(r);
            using (Variable.ForEach(r))
            {
                va[r] = v;
            }
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(va));
        }

        [Fact]
        public void DiscreteVariableSizeError2()
        {
            Assert.Throws<InvalidOperationException>(() =>
            {
                Variable<int> size = Variable.New<int>().Named("size");
                Range item = new Range(size).Named("item");
                Variable<int> index = Variable.Discrete(item, 0.2, 0.8).Named("index");
                size.ObservedValue = 3;

                InferenceEngine engine = new InferenceEngine();
                Discrete indexActual = engine.Infer<Discrete>(index);

            });
        }

        [Fact]
        public void DiscreteVariableSizeError()
        {
            Assert.Throws<ArgumentException>(() =>
            {
                Variable<int> index = Variable.Discrete(new Range(3), 0.2, 0.8).Named("index");

                InferenceEngine engine = new InferenceEngine();
                Discrete indexActual = engine.Infer<Discrete>(index);

            });
        }

        [Fact]
        public void DiscreteValueRangeError()

        {
            Assert.Throws<ArgumentException>(() =>
            {

                Range i = new Range(2);
                Variable<Vector> probs = Variable.Dirichlet(i, new double[] { 1, 1 }).Named("probs");
                Range j = new Range(2);
                Variable<int> index = Variable.Discrete(j, probs).Named("index");

                InferenceEngine engine = new InferenceEngine();
                Discrete indexActual = engine.Infer<Discrete>(index);

            });
        }

        [Fact]
        public void DiscreteVariableSizeTest()
        {
            Variable<int> size = Variable.New<int>().Named("size");
            Range item = new Range(size).Named("item");
            Variable<int> index = Variable.DiscreteUniform(item).Named("index");
            size.ObservedValue = 3;

            InferenceEngine engine = new InferenceEngine();
            Discrete indexActual = engine.Infer<Discrete>(index);
            Discrete indexExpected = Discrete.Uniform(size.ObservedValue);
            Console.WriteLine("index = {0} should be {1}", indexActual, indexExpected);
            Assert.True(indexExpected.MaxDiff(indexActual) < 1e-10);
        }

        [Fact]
        public void DiscreteSizeError()

        {
            Assert.Throws<ArgumentException>(() =>
            {

                Variable<int> index = Variable.Discrete(6).Named("index");

                InferenceEngine engine = new InferenceEngine();
                Discrete indexActual = engine.Infer<Discrete>(index);
                Discrete indexExpected = Discrete.Uniform(6);
                Console.WriteLine("index = {0} should be {1}", indexActual, indexExpected);
                Assert.True(indexExpected.MaxDiff(indexActual) < 1e-10);

            });
        }

        [Fact]
        public void DirichletVariableSizeTest()
        {
            Variable<int> size = Variable.New<int>().Named("size");
            Range item = new Range(size).Named("item");
            Variable<Vector> probs = Variable.DirichletUniform(item).Named("probs");
            size.ObservedValue = 3;

            InferenceEngine engine = new InferenceEngine();
            Dirichlet probsActual = engine.Infer<Dirichlet>(probs);
            Dirichlet probsExpected = Dirichlet.Uniform(size.ObservedValue);
            Console.WriteLine("probs = {0} should be {1}", probsActual, probsExpected);
            Assert.True(probsExpected.MaxDiff(probsActual) < 1e-10);
        }

        [Fact]
        public void SetToArrayError()

        {
            Assert.Throws<ArgumentException>(() =>
            {

                Range item = new Range(1).Named("item");
                VariableArray<double> probs = Variable.Constant(new double[] { 0.3 }, item).Named("probs");
                Variable<bool> x = Variable.New<bool>().Named("x");
                x.SetTo(Variable.Bernoulli(probs[item]));
                InferenceEngine engine = new InferenceEngine();
                Console.WriteLine(engine.Infer(x));

            });
        }

        [Fact]
        public void JaggedForEachBlockError()

        {
            Assert.Throws<InvalidOperationException>(() =>
            {

                Range outer = new Range(3).Named("outer");
                VariableArray<int> innerSizes = Variable.Constant(new int[] { 1, 2, 3 }, outer).Named("innerSizes");
                Range inner = new Range(innerSizes[outer]).Named("inner");
                var bools = Variable.Array<bool>(Variable.Array<bool>(inner), outer).Named("bools");
                using (Variable.ForEach(inner))
                {
                    using (Variable.ForEach(outer))
                    {
                        bools[outer][inner] = Variable.Bernoulli(0.3);
                    }
                }
                InferenceEngine engine = new InferenceEngine();
                Console.WriteLine(engine.Infer(bools));

            });
        }

        [Fact]
        public void JaggedForEachBlockError2()

        {
            Assert.Throws<InvalidOperationException>(() =>
            {

                Range outer = new Range(3).Named("outer");
                VariableArray<int> innerSizes = Variable.Constant(new int[] { 1, 2, 3 }, outer).Named("innerSizes");
                Range inner = new Range(innerSizes[outer]).Named("inner");
                var bools = Variable.Array<bool>(Variable.Array<bool>(inner), outer).Named("bools");
                using (Variable.ForEach(inner))
                {
                    bools[outer][inner] = Variable.Bernoulli(0.3).ForEach(outer);
                }
                InferenceEngine engine = new InferenceEngine();
                Console.WriteLine(engine.Infer(bools));

            });
        }

        [Fact]
        public void JaggedForEachError()

        {
            Assert.Throws<InvalidOperationException>(() =>
            {

                Range outer = new Range(3).Named("outer");
                VariableArray<int> innerSizes = Variable.Constant(new int[] { 1, 2, 3 }, outer).Named("innerSizes");
                Range inner = new Range(innerSizes[outer]).Named("inner");
                var bools = Variable.Array<bool>(inner).Named("bools");
                bools[inner] = Variable.Bernoulli(0.3).ForEach(inner);
                InferenceEngine engine = new InferenceEngine();
                Console.WriteLine(engine.Infer(bools));

            });
        }

        [Fact]
        public void RandomToGivenToConstantToGivenToRandom()
        {
            Variable<double> prob = Variable.Beta(2, 1).Named("prob");
            Variable<bool> b = Variable.Bernoulli(prob).Named("b");
            InferenceEngine engine = new InferenceEngine();
            engine.ShowMsl = true;
            Bernoulli bActual = engine.Infer<Bernoulli>(b);
            Bernoulli bExpected = new Bernoulli(2.0 / 3);
            Assert.True(bExpected.MaxDiff(bActual) < 1e-10);

            prob.ObservedValue = 0.4;
            bActual = engine.Infer<Bernoulli>(b);
            bExpected = new Bernoulli(prob.ObservedValue);
            Assert.True(bExpected.MaxDiff(bActual) < 1e-10);

            prob.IsReadOnly = true;
            bActual = engine.Infer<Bernoulli>(b);
            bExpected = new Bernoulli(prob.ObservedValue);
            Assert.True(bExpected.MaxDiff(bActual) < 1e-10);

            prob.IsReadOnly = false;
            prob.ObservedValue = 0.3;
            bActual = engine.Infer<Bernoulli>(b);
            bExpected = new Bernoulli(prob.ObservedValue);
            Assert.True(bExpected.MaxDiff(bActual) < 1e-10);

            prob.ClearObservedValue();
            bActual = engine.Infer<Bernoulli>(b);
            bExpected = new Bernoulli(2.0 / 3);
            Assert.True(bExpected.MaxDiff(bActual) < 1e-10);
        }

        [Fact]
        public void RedefineForEachError()
        {
            Assert.Throws<InvalidOperationException>(() =>
            {
                Range r = new Range(1).Named("r");
                Variable<bool> bools = Variable.Bernoulli(0.1).ForEach(r).Named("bools");
                bools.SetTo(Variable.Bernoulli(0.2).ForEach(r).Named("bools2"));
                InferenceEngine engine = new InferenceEngine();
                //engine.BrowserMode = BrowserMode.Always;
                engine.Infer<DistributionArray<Bernoulli>>(bools);

            });
        }

        [Fact]
        public void ObserveForEach()
        {
            Range r = new Range(1).Named("r");
            Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(r).Named("x");
            x.ArrayVariable.ObservedValue = new double[] { 3 };
            VariableArray<double> y = Variable.Array<double>(r).Named("y");
            y[r] = Variable.GaussianFromMeanAndVariance(x, 1);
            InferenceEngine engine = new InferenceEngine();
            DistributionArray<Gaussian> yActual = engine.Infer<DistributionArray<Gaussian>>(y);
            IDistribution<double[]> yExpected = Distribution<double>.Array(new Gaussian[] { new Gaussian(3, 1) });
            Assert.True(yExpected.MaxDiff(yActual) < 1e-10);
        }

        [Fact]
        public void ObserveForEachBlockError()

        {
            Assert.Throws<NullReferenceException>(() =>
            {

                Range r = new Range(1).Named("r");
                Variable<double> x;
                using (Variable.ForEach(r))
                {
                    x = Variable.GaussianFromMeanAndVariance(0, 1).Named("x");
                }
                x.ArrayVariable.ObservedValue = new double[] { 3 };
                VariableArray<double> y = Variable.Array<double>(r).Named("y");
                y[r] = Variable.GaussianFromMeanAndVariance(x, 1).ForEach(r);
                InferenceEngine engine = new InferenceEngine();
                DistributionArray<Gaussian> yActual = engine.Infer<DistributionArray<Gaussian>>(y);
                IDistribution<double[]> yExpected = Distribution<double>.Array(new Gaussian[] { new Gaussian(3, 1) });
                Assert.True(yExpected.MaxDiff(yActual) < 1e-10);

            });
        }

        [Fact]
        public void ObserveInsideForEachBlock()
        {
            Range r = new Range(1).Named("r");
            Variable<double> x;
            using (Variable.ForEach(r))
            {
                x = Variable.New<double>().Named("x");
                x.ObservedValue = 3;
            }
            //x.IsReadOnly = true;
            VariableArray<double> y = Variable.Array<double>(r).Named("y");
            y[r] = Variable.GaussianFromMeanAndVariance(x, 1).ForEach(r);
            InferenceEngine engine = new InferenceEngine();
            DistributionArray<Gaussian> yActual = engine.Infer<DistributionArray<Gaussian>>(y);
            IDistribution<double[]> yExpected = Distribution<double>.Array(new Gaussian[] { new Gaussian(3, 1) });
            Assert.True(yExpected.MaxDiff(yActual) < 1e-10);
        }

        [Fact]
        public void ObserveInsideForEachBlock2()
        {
            Range r = new Range(1).Named("r");
            Variable<double> x;
            using (Variable.ForEach(r))
            {
                x = Variable.Observed(3.0).Named("x");
            }
            //x.IsReadOnly = true;
            VariableArray<double> y = Variable.Array<double>(r).Named("y");
            y[r] = Variable.GaussianFromMeanAndVariance(x, 1).ForEach(r);
            InferenceEngine engine = new InferenceEngine();
            DistributionArray<Gaussian> yActual = engine.Infer<DistributionArray<Gaussian>>(y);
            IDistribution<double[]> yExpected = Distribution<double>.Array(new Gaussian[] { new Gaussian(3, 1) });
            Assert.True(yExpected.MaxDiff(yActual) < 1e-10);
        }

        [Fact]
        public void ObserveOutsideForEachBlock()
        {
            Range r = new Range(1).Named("r");
            Variable<double> x;
            using (Variable.ForEach(r))
            {
                x = Variable.GaussianFromMeanAndVariance(0, 1).Named("x");
            }
            x.ObservedValue = 3;
            //x.IsReadOnly = true;
            VariableArray<double> y = Variable.Array<double>(r).Named("y");
            y[r] = Variable.GaussianFromMeanAndVariance(x, 1).ForEach(r);
            InferenceEngine engine = new InferenceEngine();
            DistributionArray<Gaussian> yActual = engine.Infer<DistributionArray<Gaussian>>(y);
            IDistribution<double[]> yExpected = Distribution<double>.Array(new Gaussian[] { new Gaussian(3, 1) });
            Assert.True(yExpected.MaxDiff(yActual) < 1e-10);
        }

        [Fact]
        public void ObserveForEachError()
        {
            Assert.Throws<InvalidOperationException>(() =>
            {
                Range r = new Range(1).Named("r");
                Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(r).Named("x");
                x.ObservedValue = 3;
                VariableArray<double> y = Variable.Array<double>(r).Named("y");
                y[r] = Variable.GaussianFromMeanAndVariance(x, 1);
                InferenceEngine engine = new InferenceEngine();
                engine.Infer<DistributionArray<Gaussian>>(y);

            });
        }

        [Fact]
        public void CircularDefinitionError()
        {
            Assert.Throws<CompilationFailedException>(() =>
            {
                Variable<bool> x = Variable.New<bool>().Named("x");
                Variable<bool> y = !x;
                y.Named("y");
                Variable<bool> z = !y;
                x.SetTo(z);

                InferenceEngine engine = new InferenceEngine();
                Console.WriteLine(engine.Infer(x));

            });
        }

        [Fact]
        public void RedefineConstantError()
        {
            Assert.Throws<InvalidOperationException>(() =>
            {
                Variable<int> size = Variable.Constant<int>(4).Named("size");
                Range r = new Range(size).Named("r");
                VariableArray<double> x = Variable.Array<double>(r).Named("x");
                x[r] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(r);
                size.ObservedValue = 3;
                InferenceEngine engine = new InferenceEngine();
                //engine.BrowserMode = BrowserMode.Always;
                DistributionArray<Gaussian> xActual = engine.Infer<DistributionArray<Gaussian>>(x);

            });
        }

        [Fact]
        public void GivenNotDefinedError()
        {
            Assert.Throws<InferCompilerException>(() =>
            {
                Variable<int> size = Variable.New<int>().Named("size");
                Range r = new Range(size).Named("r");
                VariableArray<double> x = Variable.Array<double>(r).Named("x");
                x[r] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(r);
                InferenceEngine engine = new InferenceEngine();
                //engine.BrowserMode = BrowserMode.Always;
                DistributionArray<Gaussian> xActual = engine.Infer<DistributionArray<Gaussian>>(x);

            });
        }

        [Trait("Category", "OpenBug")]
        [Fact]
        public void RedefineVariableError()
        {
            Assert.Throws<InferCompilerException>(() =>
            {

                var x = Variable.New<bool>();
                var c1 = Variable.Bernoulli(0.1).Named("c1");
                using (Variable.If(c1))
                {
                    x.SetTo(true);
                }
                using (Variable.IfNot(c1))
                {
                    x.SetTo(false);
                }
                var c2 = Variable.Bernoulli(0.2).Named("c2");
                using (Variable.If(c2))
                {
                    x.SetTo(true);
                }
                using (Variable.IfNot(c2))
                {
                    x.SetTo(false);
                }
                InferenceEngine engine = new InferenceEngine();
                var xActual = engine.Infer(x);

            });
        }

        [Fact]
        public void RedefineArrayError()

        {
            Assert.Throws<InvalidOperationException>(() =>
            {

                Range r = new Range(1).Named("r");
                VariableArray<double> array = Variable.Constant(new double[] { 1.2 }).Named("array");
                VariableArray<int> indices = Variable.Constant(new int[] { 0 }).Named("indices");
                VariableArray<double> x = Variable.GetItems(array, indices).Named("x");
                // redefinition of x
                x[r] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(r);
                InferenceEngine engine = new InferenceEngine();
                //engine.BrowserMode = BrowserMode.Always;
                DistributionArray<Gaussian> xActual = engine.Infer<DistributionArray<Gaussian>>(x);

            });
        }

        [Trait("Category", "OpenBug")]
        [Fact]
        public void RedefineArrayError2()
        {
            Assert.Throws<InferCompilerException>(() =>
            {
                Range timeRange = new Range(2);
                var sum = Variable.Array<double>(timeRange).Named("sum");
                var x = Variable.Array<double>(timeRange).Named("x");

                // for each time step
                using (var block = Variable.ForEach(timeRange))
                {
                    var t = block.Index;

                    using (Variable.If(t == 0))
                    {
                        sum[t] = Variable.GaussianFromMeanAndVariance(0, 0);
                        x[t] = Variable.GaussianFromMeanAndVariance(0, 1.0);
                    }
                    using (Variable.If(t > 0))
                    {
                        sum[t] = sum[t - 1] + x[t - 1];
                    }

                    // order iff below minimum
                    Variable<bool> mustOrder = Variable.Bernoulli(0.1);
                    using (Variable.If(mustOrder))
                    {
                        x[t].SetTo(0.0);
                    }
                    using (Variable.IfNot(mustOrder))
                    {
                        x[t].SetTo(0.0);
                    }
                }    // end ForEach block      

                InferenceEngine ie = new InferenceEngine();
                var xActual = ie.Infer<IList<Gaussian>>(x);
            });
        }

        [Fact]
        public void ArrayNotDefinedError()
        {
            Assert.Throws<InferCompilerException>(() =>
            {
                Range r = new Range(1).Named("r");
                VariableArray<double> x = Variable.Array<double>(r).Named("x");
                InferenceEngine engine = new InferenceEngine();
                //engine.BrowserMode = BrowserMode.Always;
                DistributionArray<Gaussian> xActual = engine.Infer<DistributionArray<Gaussian>>(x);

            });
        }

        [Fact]
        public void ArrayOfGivenLength()
        {
            Variable<int> size = Variable.New<int>().Named("size");
            Range r = new Range(size).Named("r");
            VariableArray<double> x = Variable.Array<double>(r).Named("x");
            x[r] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(r);
            InferenceEngine engine = new InferenceEngine();
            //engine.BrowserMode = BrowserMode.Always;
            size.ObservedValue = 1;
            DistributionArray<Gaussian> xActual = engine.Infer<DistributionArray<Gaussian>>(x);
            IDistribution<double[]> xExpected = Distribution<double>.Array(size.ObservedValue,
                                                                           delegate (int i)
                                                                           {
                                                                               return new Gaussian(0, 1);
                                                                           });
            Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);
            Assert.True(xActual.MaxDiff(xExpected) < 1e-10);
        }

        [Fact]
        public void ArrayOfConstantLength()
        {
            Variable<int> size = Variable.Constant<int>(1).Named("size");
            Range r = new Range(size).Named("r");
            VariableArray<double> x = Variable.Array<double>(r).Named("x");
            x[r] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(r);
            InferenceEngine engine = new InferenceEngine();
            //engine.BrowserMode = BrowserMode.Always;
            DistributionArray<Gaussian> xActual = engine.Infer<DistributionArray<Gaussian>>(x);
            IDistribution<double[]> xExpected = Distribution<double>.Array(size.ObservedValue, i => new Gaussian(0, 1));
            Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);
            Assert.True(xActual.MaxDiff(xExpected) < 1e-10);
        }

        [Fact]
        public void Array2DSizeError()
        {
            Range r = new Range(1).Named("r");
            Range r2 = new Range(2).Named("r2");
            VariableArray2D<bool> y = Variable.Array<bool>(r, r2).Named("y");
            VariableArray2D<bool> x = Variable.Array<bool>(r, r2).Named("x");
            x[r, r2] = !y[r, r2];
            InferenceEngine engine = new InferenceEngine();
            y.ObservedValue = Util.ArrayInit(1, 1, (i, j) => true);
            try
            {
                object xActual = engine.Infer(x);
                Assert.True(false, "Did not throw exception");
                IDistribution<bool[]> xExpected = Distribution<bool>.Array(r.SizeAsInt, i => new Bernoulli(0));
                Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);
                Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with " + ex);
            }
        }

        [Fact]
        public void ArraySizeError()
        {
            Range r = new Range(1).Named("r");
            VariableArray<bool> y = Variable.Array<bool>(r).Named("y");
            VariableArray<bool> x = Variable.Array<bool>(r).Named("x");
            x[r] = !y[r];
            InferenceEngine engine = new InferenceEngine();
            y.ObservedValue = Util.ArrayInit(2, i => true);
            try
            {
                object xActual = engine.Infer(x);
                Assert.True(false, "Did not throw exception");
                IDistribution<bool[]> xExpected = Distribution<bool>.Array(r.SizeAsInt, i => new Bernoulli(0));
                Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);
                Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with " + ex);
            }
        }

        [Fact]
        public void ArraySizeError2()
        {
            Variable<int> size = Variable.Constant<int>(1).Named("size");
            Range r = new Range(size).Named("r");
            VariableArray<bool> y = Variable.Array<bool>(r).Named("y");
            VariableArray<bool> x = Variable.Array<bool>(r).Named("x");
            x[r] = !y[r];
            InferenceEngine engine = new InferenceEngine();
            y.ObservedValue = Util.ArrayInit(2, i => true);
            try
            {
                object xActual = engine.Infer(x);
                Assert.True(false, "Did not throw exception");
                IDistribution<bool[]> xExpected = Distribution<bool>.Array(size.ObservedValue, i => new Bernoulli(0));
                Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);
                Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with " + ex);
            }
        }

        [Fact]
        public void ArraySizeError3()
        {
            Variable<int> size = Variable.Observed(1).Named("size");
            Range r = new Range(size).Named("r");
            VariableArray<bool> y = Variable.Array<bool>(r).Named("y");
            VariableArray<bool> x = Variable.Array<bool>(r).Named("x");
            x[r] = !y[r];
            InferenceEngine engine = new InferenceEngine();
            y.ObservedValue = Util.ArrayInit(2, i => true);
            try
            {
                object xActual = engine.Infer(x);
                Assert.True(false, "Did not throw exception");
                IDistribution<bool[]> xExpected = Distribution<bool>.Array(size.ObservedValue, i => new Bernoulli(0));
                Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);
                Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with " + ex);
            }
        }

        [Fact]
        [Trait("Category", "BadTest")]
        public void ImplicitArrayOfGivenLength()
        {
            Variable<int> size = Variable.New<int>().Named("size");
            Range r = new Range(size).Named("r");
            Variable<double> x;
            using (Variable.ForEach(r))
            {
                x = Variable.GaussianFromMeanAndVariance(0, 1).Named("x");
            }
            InferenceEngine engine = new InferenceEngine();
            //engine.BrowserMode = BrowserMode.Always;
            size.ObservedValue = 1;
            DistributionArray<Gaussian> xActual = engine.Infer<DistributionArray<Gaussian>>(x);
            IDistribution<double[]> xExpected = Distribution<double>.Array(size.ObservedValue,
                                                                           delegate (int i)
                                                                           {
                                                                               return new Gaussian(0, 1);
                                                                           });
            Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);
            Assert.True(xActual.MaxDiff(xExpected) < 1e-10);
        }

        [Fact]
        [Trait("Category", "BadTest")]
        public void ImplicitArrayOfConstantLength()
        {
            Variable<int> size = Variable.Constant<int>(1).Named("size");
            Range r = new Range(size).Named("r");
            Variable<double> x;
            using (Variable.ForEach(r))
            {
                x = Variable.GaussianFromMeanAndVariance(0, 1).Named("x");
            }
            InferenceEngine engine = new InferenceEngine();
            //engine.BrowserMode = BrowserMode.Always;
            DistributionArray<Gaussian> xActual = engine.Infer<DistributionArray<Gaussian>>(x);
            IDistribution<double[]> xExpected = Distribution<double>.Array(size.ObservedValue,
                                                                           delegate (int i)
                                                                           {
                                                                               return new Gaussian(0, 1);
                                                                           });
            Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);
            Assert.True(xActual.MaxDiff(xExpected) < 1e-10);
        }

        [Fact]
        public void VariableOfGenericType()
        {
            Variable<List<double>> a = Variable.Constant(new List<double>(new double[] { 5 }));
            Variable<double> y = Variable<double>.Factor(Factor.GetItem, a, 0);
            Variable<double> x = Variable.GaussianFromMeanAndVariance(y, 1);
            InferenceEngine engine = new InferenceEngine();
            Gaussian xActual = engine.Infer<Gaussian>(x);
            Gaussian xExpected = new Gaussian(5, 1);
            Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);
            Assert.True(xActual.MaxDiff(xExpected) < 1e-10);
        }

        [Fact]
        public void RandomArray()
        {
            Bernoulli[] boolsPriorArray = new Bernoulli[] { new Bernoulli(0.2) };
            Variable<bool[]> bools = Variable<bool[]>.Random(Distribution<bool>.Array(boolsPriorArray));
            InferenceEngine engine = new InferenceEngine();
            Bernoulli[] boolsActual = engine.Infer<Bernoulli[]>(bools);
            Assert.True(boolsPriorArray[0].MaxDiff(boolsActual[0]) < 1e-10);
        }

        [Fact]
        public void ForEachBooleanTest()
        {
            VariableArray<double> probTrue = Variable.Constant(new double[] { 0.1, 0.8 }).Named("probTrue");
            Range n = probTrue.Range;
            VariableArray<bool> noisyBools = Variable.Array<bool>(n).Named("noisyBools");
            noisyBools[n] = Variable.Bernoulli(probTrue[n]).Named("bools") == Variable.Bernoulli(0.75).ForEach(n);
            VariableArray<bool> sameNoise = Variable.Array<bool>(n).Named("sameNoise");
            sameNoise[n] = Variable.Bernoulli(probTrue[n]).Named("sameNoiseBools") == Variable.Bernoulli(0.75);

            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            Console.WriteLine("noisyBools = ");
            Console.WriteLine(engine.Infer(noisyBools));
            Console.WriteLine("sameNoise = ");
            Console.WriteLine(engine.Infer(sameNoise));
        }

        [Fact]
        public void ForEachConstantError()
        {
            Assert.Throws<ArgumentException>(() =>
            {
                Range n = new Range(1).Named("n");
                VariableArray<bool> bools = Variable.Array<bool>(n).Named("bools");
                bools[n] = Variable.Bernoulli(Variable.Constant(0.9).ForEach(n));

                InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
                DistributionArray<Bernoulli> d = engine.Infer<DistributionArray<Bernoulli>>(bools);
                Console.WriteLine("bools = ");
                Console.WriteLine(d);
                //Assert.True(d[0].MaxDiff(new Bernoulli(0.9)) < 1e-10);

            });
        }

        [Fact]
        public void ForEachGivenTest()
        {
            Range item = new Range(1).Named("item");
            Variable<double> p = Variable.New<double>().Named("p").ForEach(item);
            VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
            bools[item] = Variable.Bernoulli(p);

            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            p.ArrayVariable.ObservedValue = new double[] { 0.9 };
            DistributionArray<Bernoulli> d = engine.Infer<DistributionArray<Bernoulli>>(bools);
            Console.WriteLine("bools = ");
            Console.WriteLine(d);
            Assert.True(d[0].MaxDiff(new Bernoulli(0.9)) < 1e-10);
        }

        [Fact]
        public void ForEachGivenInitialError()
        {
            Assert.Throws<ArgumentException>(() =>
            {
                Variable<double> p = Variable.Observed(0.9).Named("p");
                Range item = new Range(1).Named("item");
                VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
                bools[item] = Variable.Bernoulli(p.ForEach(item));

                InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
                DistributionArray<Bernoulli> d = engine.Infer<DistributionArray<Bernoulli>>(bools);
                Console.WriteLine("bools = ");
                Console.WriteLine(d);
                //Assert.True(d[0].MaxDiff(new Bernoulli(0.9)) < 1e-10);

            });
        }

        [Fact]
        public void ForEachGivenNotSetTest()
        {
            Assert.Throws<InvalidOperationException>(() =>
            {
                Variable<double> p = Variable.New<double>().Named("p");
                Range item = new Range(1).Named("item");
                VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
                bools[item] = Variable.Bernoulli(p.ForEach(item));

                InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
                p.ObservedValue = 0.9; // must use p.Array.Value
                Console.WriteLine(engine.Infer(bools));

            });
        }

        [Fact]
        public void ForEachGivenReplicationError()
        {
            Assert.Throws<ArgumentException>(() =>
            {
                Variable<double> p = Variable.Observed(0.9).Named("p");
                Variable<int> size = Variable.New<int>().Named("size");
                Range n = new Range(size).Named("n");
                VariableArray<bool> bools = Variable.Array<bool>(n).Named("bools");
                bools[n] = Variable.Bernoulli(p.ForEach(n));

                InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
                size.ObservedValue = 1;
                Console.WriteLine(engine.Infer(bools));

            });
        }

        [Fact]
        public void ArrayIndexingErrors()
        {
            Range i = new Range(2).Named("i");
            Range n = new Range(2).Named("n");

            try
            {
                VariableArray<bool> bools = Variable.Array<bool>(n).Named("bools");
                bools[n] = Variable.Bernoulli(0.5);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            try
            {
                VariableArray<bool> bools = Variable.Array<bool>(n).Named("bools");
                bools[n] = Variable.Bernoulli(0.5).ForEach(i);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            try
            {
                VariableArray<bool> bools = Variable.Array<bool>(n).Named("bools");
                bools[n] = Variable.Bernoulli(0.5).ForEach(i, n);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            try
            {
                VariableArray<bool> bools = Variable.Array<bool>(n).Named("bools");
                bools[i] = Variable.Bernoulli(0.5).ForEach(i);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            try
            {
                VariableArray<bool> bools = Variable.Array<bool>(n).Named("bools");
                bools[i] = Variable.Bernoulli(0.5).ForEach(n);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            if (true)
            {
                VariableArray<bool> bools = Variable.Array<bool>(n).Named("bools");
                bools[n] = Variable.Bernoulli(0.5).ForEach(n);
            }
            try
            {
                VariableArray<bool> constBools = Variable.Constant(new bool[] { true, false }, n).Named("constBools");
                constBools[n] = Variable.Bernoulli(0.5);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            try
            {
                VariableArray<bool> constBools = Variable.Constant(new bool[] { true, false }, n).Named("constBools");
                constBools[n] = Variable.Bernoulli(0.5).ForEach(i);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            try
            {
                VariableArray<bool> constBools = Variable.Constant(new bool[] { true, false }, n).Named("constBools");
                constBools[n] = Variable.Bernoulli(0.5).ForEach(i, n);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            try
            {
                VariableArray<bool> constBools = Variable.Constant(new bool[] { true, false }, n).Named("constBools");
                constBools[i] = Variable.Bernoulli(0.5).ForEach(i);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            try
            {
                VariableArray<bool> constBools = Variable.Constant(new bool[] { true, false }, n).Named("constBools");
                constBools[i] = Variable.Bernoulli(0.5).ForEach(n);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            if (true)
            {
                VariableArray<bool> constBools = Variable.Constant(new bool[] { true, false }, n).Named("constBools");
                constBools[n] = Variable.Bernoulli(0.5).ForEach(n);
            }

            try
            {
                VariableArray<bool> givenBools = Variable.Observed(new bool[] { true, false }, n).Named("givenBools");
                givenBools[n] = Variable.Bernoulli(0.5);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            try
            {
                VariableArray<bool> givenBools = Variable.Observed(new bool[] { true, false }, n).Named("givenBools");
                givenBools[n] = Variable.Bernoulli(0.5).ForEach(i);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            try
            {
                VariableArray<bool> givenBools = Variable.Observed(new bool[] { true, false }, n).Named("givenBools");
                givenBools[n] = Variable.Bernoulli(0.5).ForEach(i, n);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            try
            {
                VariableArray<bool> givenBools = Variable.Observed(new bool[] { true, false }, n).Named("givenBools");
                givenBools[i] = Variable.Bernoulli(0.5).ForEach(i);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            try
            {
                VariableArray<bool> givenBools = Variable.Observed(new bool[] { true, false }, n).Named("givenBools");
                givenBools[i] = Variable.Bernoulli(0.5).ForEach(n);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            if (true)
            {
                VariableArray<bool> givenBools = Variable.Observed(new bool[] { true, false }, n).Named("givenBools");
                givenBools[n] = Variable.Bernoulli(0.5).ForEach(n);
            }
        }

        [Fact]
        public void Array2dIndexingErrors()
        {
            Range i = new Range(2).Named("i");
            Range n = new Range(2).Named("n");
            Range m = new Range(2).Named("m");

            try
            {
                VariableArray2D<bool> bools = Variable.Array<bool>(n, m).Named("bools");
                bools[n, m] = Variable.Bernoulli(0.5);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            try
            {
                VariableArray2D<bool> bools = Variable.Array<bool>(n, m).Named("bools");
                bools[n, m] = Variable.Bernoulli(0.5).ForEach(i);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            try
            {
                VariableArray2D<bool> bools = Variable.Array<bool>(n, m).Named("bools");
                bools[n, m] = Variable.Bernoulli(0.5).ForEach(i, n);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            try
            {
                VariableArray2D<bool> bools = Variable.Array<bool>(n, m).Named("bools");
                bools[n, m] = Variable.Bernoulli(0.5).ForEach(i, n, m);
                Assert.True(false, "Did not throw exception");
            }
            catch (NotSupportedException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            try
            {
                VariableArray2D<bool> bools = Variable.Array<bool>(n, m).Named("bools");
                bools[i, m] = Variable.Bernoulli(0.5).ForEach(i, m);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            try
            {
                VariableArray2D<bool> bools = Variable.Array<bool>(n, m).Named("bools");
                bools[i, m] = Variable.Bernoulli(0.5).ForEach(n, m);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            if (true)
            {
                VariableArray2D<bool> bools = Variable.Array<bool>(n, m).Named("bools");
                bools[n, m] = Variable.Bernoulli(0.5).ForEach(n, m);
            }

            bool[,] init2d = new bool[,] { { true, false }, { true, false } };
            try
            {
                VariableArray2D<bool> constBools = Variable.Constant(init2d, n, m).Named("constBools");
                constBools[n, m] = Variable.Bernoulli(0.5);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            try
            {
                VariableArray2D<bool> constBools = Variable.Constant(init2d, n, m).Named("constBools");
                constBools[n, m] = Variable.Bernoulli(0.5).ForEach(i);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            try
            {
                VariableArray2D<bool> constBools = Variable.Constant(init2d, n, m).Named("constBools");
                constBools[n, m] = Variable.Bernoulli(0.5).ForEach(i, n);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            try
            {
                VariableArray2D<bool> constBools = Variable.Constant(init2d, n, m).Named("constBools");
                constBools[n, m] = Variable.Bernoulli(0.5).ForEach(i, n, m);
                Assert.True(false, "Did not throw exception");
            }
            catch (NotSupportedException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            try
            {
                VariableArray2D<bool> constBools = Variable.Constant(init2d, n, m).Named("constBools");
                constBools[i, m] = Variable.Bernoulli(0.5).ForEach(i, m);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            try
            {
                VariableArray2D<bool> constBools = Variable.Constant(init2d, n, m).Named("constBools");
                constBools[i, m] = Variable.Bernoulli(0.5).ForEach(n, m);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            if (true)
            {
                VariableArray2D<bool> constBools = Variable.Constant(init2d, n, m).Named("constBools");
                constBools[n, m] = Variable.Bernoulli(0.5).ForEach(n, m);
            }

            try
            {
                VariableArray2D<bool> givenBools = Variable.Observed(init2d, n, m).Named("givenBools");
                givenBools[n, m] = Variable.Bernoulli(0.5);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            try
            {
                VariableArray2D<bool> givenBools = Variable.Observed(init2d, n, m).Named("givenBools");
                givenBools[n, m] = Variable.Bernoulli(0.5).ForEach(i);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            try
            {
                VariableArray2D<bool> givenBools = Variable.Observed(init2d, n, m).Named("givenBools");
                givenBools[n, m] = Variable.Bernoulli(0.5).ForEach(i, n);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            try
            {
                VariableArray2D<bool> givenBools = Variable.Observed(init2d, n, m).Named("givenBools");
                givenBools[n, m] = Variable.Bernoulli(0.5).ForEach(i, n, m);
                Assert.True(false, "Did not throw exception");
            }
            catch (NotSupportedException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            try
            {
                VariableArray2D<bool> givenBools = Variable.Observed(init2d, n, m).Named("givenBools");
                givenBools[i, m] = Variable.Bernoulli(0.5).ForEach(i, m);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            try
            {
                VariableArray2D<bool> givenBools = Variable.Observed(init2d, n, m).Named("givenBools");
                givenBools[i, m] = Variable.Bernoulli(0.5).ForEach(n, m);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("Correctly failed with exception: " + ex);
            }
            if (true)
            {
                VariableArray2D<bool> givenBools = Variable.Observed(init2d, n, m).Named("givenBools");
                givenBools[n, m] = Variable.Bernoulli(0.5).ForEach(n, m);
            }
        }

        [Fact]
        public void Array2dIndexing()
        {
            Range n = new Range(2).Named("n");
            Range m = new Range(2).Named("m");

            VariableArray<bool> bools = Variable.Array<bool>(n).Named("bools");
            bools[n] = Variable.Bernoulli(0.9).ForEach(n);
            bool[] init = new bool[] { true, false };
            VariableArray<bool> constBools = Variable.Constant<bool>(init, n).Named("constBools");
            VariableArray<bool> givenBools = Variable.Observed(init, n).Named("givenBools");

            VariableArray2D<bool> bools2d = Variable.Array<bool>(n, m).Named("bools2d");
            bools2d[n, m] = (!bools[n]).ForEach(m);
            VariableArray2D<bool> bools2dFromConst = Variable.Array<bool>(n, m).Named("bools2dFromConst");
            bools2dFromConst[n, m] = (!constBools[n]).ForEach(m);
            VariableArray2D<bool> bools2dFromGiven = Variable.Array<bool>(n, m).Named("bools2dFromGiven");
            bools2dFromGiven[n, m] = (!givenBools[n]).ForEach(m);

            VariableArray<bool> boolsFromConst = Variable.Array<bool>(n).Named("boolsFromConst");
            boolsFromConst[n] = Variable.Bernoulli(0.9).ForEach(n);
            bool[,] init2d = new bool[,] { { true, true }, { false, false } };
            VariableArray2D<bool> constBools2d = Variable.Constant<bool>(init2d, n, m);
            constBools2d[n, m] = (!boolsFromConst[n]).ForEach(m);

            VariableArray<bool> boolsFromGiven = Variable.Array<bool>(n).Named("boolsFromGiven");
            boolsFromGiven[n] = Variable.Bernoulli(0.9).ForEach(n);
            VariableArray2D<bool> givenBools2d = Variable.Observed<bool>(init2d, n, m).Named("givenBools2d");
            givenBools2d[n, m] = (!boolsFromGiven[n]).ForEach(m);

            InferenceEngine engine = new InferenceEngine();
            DistributionArray<Bernoulli> bools_Marginal = engine.Infer<DistributionArray<Bernoulli>>(bools);
            bools_Marginal.ForEach(delegate (Bernoulli item)
            {
                Assert.True(item.MaxDiff(new Bernoulli(0.9)) < 1e-10);
            });
#if false
    // cannot infer derived constants or givens
            DistributionArray<Bernoulli> bools2dFromConst_Marginal = engine.Infer<DistributionArray<Bernoulli>>(bools2dFromConst);
            for (int i = 0; i < bools2dFromConst_Marginal.GetLength(0); i++) {
                for (int j = 0; j < bools2dFromConst_Marginal.GetLength(1); j++) {
                    Assert.True(bools2dFromConst_Marginal[i, j].MaxDiff(Bernoulli.PointMass(!init[i])) < 1e-10);
                }
            }
            DistributionArray<Bernoulli> bools2dFromGiven_Marginal = engine.Infer<DistributionArray<Bernoulli>>(bools2dFromGiven);
            for (int i = 0; i < bools2dFromGiven_Marginal.GetLength(0); i++) {
                for (int j = 0; j < bools2dFromGiven_Marginal.GetLength(1); j++) {
                    Assert.True(bools2dFromGiven_Marginal[i, j].MaxDiff(Bernoulli.PointMass(!init[i])) < 1e-10);
                }
            }
#endif
            DistributionArray<Bernoulli> boolsFromConst_Marginal = engine.Infer<DistributionArray<Bernoulli>>(boolsFromConst);
            for (int i = 0; i < boolsFromConst_Marginal.Count; i++)
            {
                Assert.True(boolsFromConst_Marginal[i].MaxDiff(Bernoulli.PointMass(!init[i])) < 1e-10);
            }
            DistributionArray<Bernoulli> boolsFromGiven_Marginal = engine.Infer<DistributionArray<Bernoulli>>(boolsFromGiven);
            for (int i = 0; i < boolsFromGiven_Marginal.Count; i++)
            {
                Assert.True(boolsFromGiven_Marginal[i].MaxDiff(Bernoulli.PointMass(!init[i])) < 1e-10);
            }
        }

        [Fact]
        public void DuplicateForEachError()

        {
            Assert.Throws<InvalidOperationException>(() =>
            {

                Range item = new Range(2).Named("item");
                VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
                using (Variable.ForEach(item))
                {
                    using (Variable.ForEach(item))
                    {
                        bools[item] = Variable.Bernoulli(0.1);
                    }
                }

            });
        }

        [Fact]
        public void DuplicateForEachError2()

        {
            Assert.Throws<InvalidOperationException>(() =>
            {

                Range item = new Range(2).Named("item");
                VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
                using (Variable.ForEach(item))
                {
                    bools[item] = Variable.Bernoulli(0.1).ForEach(item);
                }

            });
        }

        [Fact]
        public void DanglingBlockError()

        {
            Assert.Throws<InvalidOperationException>((Action)(() =>
            {

                Range item = new Range(2).Named("item");
                try
                {
                    ForEachBlock block = Variable.ForEach(item);
                    Variable<bool> b = Variable.Bernoulli(0.5);
                    InferenceEngine engine = new InferenceEngine();
                    engine.Infer(b);
                }
                finally
                {
                    StatementBlock.CloseAllBlocks();
                }

            }));
        }

        [Fact]
        public void ForEachBlockLhsError()

        {
            Assert.Throws<ArgumentException>(() =>
            {

                Range item = new Range(2).Named("item");
                Range feature = new Range(2).Named("feature");
                VariableArray<bool> b = Variable.Array<bool>(feature);
                using (Variable.ForEach(item))
                {
                    b[feature] = Variable.Bernoulli(0.5).ForEach(feature);
                }

            });
        }

        [Fact]
        public void ForEachBlockLocalArrayTest()
        {
            Range item = new Range(2).Named("item");
            Range feature = new Range(2).Named("feature");
            VariableArray2D<bool> a = Variable.Array<bool>(item, feature);
            using (Variable.ForEach(item))
            {
                VariableArray<bool> b = Variable.Array<bool>(feature);
                b[feature] = Variable.Bernoulli(0.9).ForEach(feature);
                a[item, feature] = !b[feature];
            }
            InferenceEngine engine = new InferenceEngine();
            DistributionArray2D<Bernoulli> actual = engine.Infer<DistributionArray2D<Bernoulli>>(a);
            Bernoulli expected = new Bernoulli(0.1);
            foreach (Bernoulli dist in actual)
            {
                Assert.True(dist.MaxDiff(expected) < 1e-10);
            }
        }


        // This functionality is no longer supported.
        //[Fact]
        internal void ForEachBlockVariableArrayTest()
        {
            Range item = new Range(1).Named("item");
            Range feature = new Range(1).Named("feature");
            VariableArray2D<bool> b = Variable.Array<bool>(item, feature).Named("b");
            VariableArray<double> p;
            using (Variable.ForEach(item))
            {
                p = Variable.Array<double>(feature).Named("p");
                b[item, feature] = Variable.Bernoulli(p[feature]);
            }
            InferenceEngine engine = new InferenceEngine();
            p.ArrayVariable.ObservedValue = new double[][] { new double[] { 0.9 } };
            DistributionArray2D<Bernoulli> actual = engine.Infer<DistributionArray2D<Bernoulli>>(b);
            IDistribution<bool[,]> expected = Distribution<bool>.Array(new Bernoulli[,] { { new Bernoulli(0.9) } });
            Assert.True(expected.MaxDiff(actual) < 1e-10);
        }

        [Fact]
        [Trait("Category", "BadTest")]
        public void ForEachSetToTest()
        {
            Range item = new Range(1).Named("item");
            Variable<bool> x;
            using (Variable.ForEach(item))
            {
                Variable<bool> c = Variable.Bernoulli(0.1).Named("c");
                x = Variable.New<bool>().Named("x");
                using (Variable.If(c))
                {
                    x.SetTo(Variable.Bernoulli(0.3));
                }
                using (Variable.IfNot(c))
                {
                    x.SetTo(Variable.Bernoulli(0.4));
                }
            }
            InferenceEngine engine = new InferenceEngine();
            DistributionArray<Bernoulli> actual = engine.Infer<DistributionArray<Bernoulli>>(x);
            IDistribution<bool[]> expected = Distribution<bool>.Array(new Bernoulli[] { new Bernoulli(0.1 * 0.3 + 0.9 * 0.4) });
            Assert.True(expected.MaxDiff(actual) < 1e-10);
        }

        [Fact]
        public void ForEachBlockTest()
        {
            Range n = new Range(2).Named("n");
            Range m = new Range(2).Named("m");

            bool[] init = new bool[] { true, false };
            bool[,] init2d = new bool[,] { { true, true }, { false, false } };

            VariableArray<bool> bools = Variable.Array<bool>(n).Named("bools");
            VariableArray<bool> constBools = Variable.Constant<bool>(init, n).Named("constBools");
            VariableArray<bool> givenBools = Variable.Observed(init, n).Named("givenBools");
            VariableArray<bool> boolsFromConst = Variable.Array<bool>(n).Named("boolsFromConst");
            VariableArray<bool> boolsFromGiven = Variable.Array<bool>(n).Named("boolsFromGiven");

            VariableArray2D<bool> bools2d = Variable.Array<bool>(n, m).Named("bools2d");
            VariableArray2D<bool> bools2dFromConst = Variable.Array<bool>(n, m).Named("bools2dFromConst");
            VariableArray2D<bool> bools2dFromGiven = Variable.Array<bool>(n, m).Named("bools2dFromGiven");
            VariableArray2D<bool> constBools2d = Variable.Constant<bool>(init2d, n, m);
            VariableArray2D<bool> givenBools2d = Variable.Observed<bool>(init2d, n, m).Named("givenBools2d");

            using (Variable.ForEach(n))
            {
                bools[n] = Variable.Bernoulli(0.9);
                boolsFromConst[n] = Variable.Bernoulli(0.9);
                boolsFromGiven[n] = Variable.Bernoulli(0.9);
                using (Variable.ForEach(m))
                {
                    bools2d[n, m] = !bools[n];
                }
                bools2dFromConst[n, m] = (!constBools[n]).ForEach(m);
            }
            using (Variable.ForEach(m))
            {
                bools2dFromGiven[n, m] = (!givenBools[n]);
                constBools2d[n, m] = (!boolsFromConst[n]);
                givenBools2d[n, m] = (!boolsFromGiven[n]);
            }

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.ReturnCopies = false;
            DistributionArray<Bernoulli> bools_Marginal = engine.Infer<DistributionArray<Bernoulli>>(bools);
            bools_Marginal.ForEach(delegate (Bernoulli item)
            {
                Assert.True(item.MaxDiff(new Bernoulli(0.9)) < 1e-10);
            });
            DistributionArray2D<Bernoulli> bools2dFromConst_Marginal = engine.Infer<DistributionArray2D<Bernoulli>>(bools2dFromConst);
            for (int i = 0; i < bools2dFromConst_Marginal.GetLength(0); i++)
            {
                for (int j = 0; j < bools2dFromConst_Marginal.GetLength(1); j++)
                {
                    Assert.True(bools2dFromConst_Marginal[i, j].MaxDiff(Bernoulli.PointMass(!init[i])) < 1e-10);
                }
            }
            DistributionArray2D<Bernoulli> bools2dFromGiven_Marginal = engine.Infer<DistributionArray2D<Bernoulli>>(bools2dFromGiven);
            for (int i = 0; i < bools2dFromGiven_Marginal.GetLength(0); i++)
            {
                for (int j = 0; j < bools2dFromGiven_Marginal.GetLength(1); j++)
                {
                    Assert.True(bools2dFromGiven_Marginal[i, j].MaxDiff(Bernoulli.PointMass(!init[i])) < 1e-10);
                }
            }
            DistributionArray<Bernoulli> boolsFromConst_Marginal = engine.Infer<DistributionArray<Bernoulli>>(boolsFromConst);
            for (int i = 0; i < boolsFromConst_Marginal.Count; i++)
            {
                Assert.True(boolsFromConst_Marginal[i].MaxDiff(Bernoulli.PointMass(!init[i])) < 1e-10);
            }
            DistributionArray<Bernoulli> boolsFromGiven_Marginal = engine.Infer<DistributionArray<Bernoulli>>(boolsFromGiven);
            for (int i = 0; i < boolsFromGiven_Marginal.Count; i++)
            {
                Assert.True(boolsFromGiven_Marginal[i].MaxDiff(Bernoulli.PointMass(!init[i])) < 1e-10);
            }
        }

        /// This test now does not compile, as desired, since Variable.Random 
        /// has additional static type checks which pick up on this error.
        /*[Fact]
    public void RandomRequiresDistributionError()
    {
        try {
            Given<double> d = Variable.Given<double>("d", 0.0);
            Variable<double> x = Variable.Random<double,double>(d);
            Assert.True(false, "Did not throw exception");
        } catch (Exception ex) { Console.WriteLine("Correctly failed with exception: "+ex); }
        //InferenceEngine engine = new InferenceEngine();
        //engine.BrowserMode = BrowserMode.Never;
        //Console.WriteLine(engine.Infer<Gaussian>(x));
    }*/
        /// <summary>
        /// Checks to ensure that using variable names that are not valid identifiers doesn't cause errors.
        /// </summary>
        [Fact]
        public void VariableNamingTest()
        {
            InferenceEngine engine = new InferenceEngine();
            string[] names =
                {
                    "this", "a,b", "00hello", "test name", "hello!world", "fred0", "   x   ",
                    "x`~!@#$%^&*()-+=[]\\{}|;:'\",./<>?", "", "for", "typeof"
                };
            foreach (string name in names)
            {
                Variable<double> mean = Variable.Observed(5.6).Named("given" + name);
                Variable<double> prec = Variable.Constant(1.0).Named("const" + name);
                Variable<double> x = Variable.GaussianFromMeanAndPrecision(mean, prec).Named(name);
                Console.WriteLine("result = {0}", engine.Infer(x));
            }
        }

        [Fact]
        public void ModelNamingTest()
        {
            InferenceEngine engine = new InferenceEngine();
            string[] names =
                {
                    "this", "a,b", "00hello", "test name", "hello!world", "fred0", "   x   ",
                    "x`~!@#$%^&*()-+=[]\\{}|;:'\",./<>?", "", "for", "typeof"
                };
            foreach (string name in names)
            {
                engine.ModelName = name;
                var b = Variable<bool>.Bernoulli(0.5);
                Console.WriteLine(engine.Infer(b));
            }
        }

        [Fact]
        public void RangeNamingTest()
        {
            InferenceEngine engine = new InferenceEngine();
            string[] names =
                {
                    "this", "a,b", "00hello", "test name", "hello!world", "fred0", "   i   ",
                    "x`~!@#$%^&*()-+=[]\\{}|;:'\",./<>?", "", "for", "typeof"
                };
            foreach (string name in names)
            {
                Range item = new Range(1).Named(name);
                var x = Variable.Array<bool>(item).Named("x");
                x[item] = Variable.Bernoulli(0.1).ForEach(item);
                Console.WriteLine(engine.Infer(x));
            }
        }

        [Fact]
        public void RangeNameClashTest()
        {
            Assert.Throws<InferCompilerException>(() =>
            {
                Range item = new Range(2).Named("item");
                Range item2 = new Range(3).Named("item");
                VariableArray2D<bool> bools = Variable.Array<bool>(item, item2).Named("bools");
                bools[item, item2] = Variable.Bernoulli(0.1).ForEach(item, item2);
                InferenceEngine engine = new InferenceEngine();
                engine.Infer(bools);

            });
        }

        [Fact]
        public void RangeNameClashTest2()
        {
            Assert.Throws<InferCompilerException>(() =>
            {
                Range item = new Range(2).Named("item");
                VariableArray<bool> bools = Variable.Array<bool>(item).Named("item");
                bools[item] = Variable.Bernoulli(0.1).ForEach(item);
                InferenceEngine engine = new InferenceEngine();
                engine.Infer(bools);

            });
        }

        [Fact]
        public void RangeNameClashTest3()

        {
            Assert.Throws<InferCompilerException>(() =>
            {

                Range item = new Range(2).Named("emptyString");
                Range item2 = new Range(3).Named("");
                VariableArray2D<bool> bools = Variable.Array<bool>(item, item2).Named("bools");
                bools[item, item2] = Variable.Bernoulli(0.1).ForEach(item, item2);
                InferenceEngine engine = new InferenceEngine();
                engine.Infer(bools);

            });
        }

        /*public void MyTest()
    {
        Range i = new Range(10);
        VariableArray<double> array = Variable.Array<double>(i);            
    }*/

        [Fact]
        public void BooleanConstantTest()
        {
            VariableArray<bool> const0 = Variable.Constant(new bool[] { false, true }).Named("c0");
            Range n = const0.Range.Named("n");
            VariableArray<bool> const1 = Variable.Array<bool>(n).Named("c1");
            const1[n] = !const0[n];
            VariableArray<bool> x = Variable.Array<bool>(n).Named("x");
            x[n] = Variable.Bernoulli(0.5).ForEach(n);
            VariableArray<bool> y = Variable.Array<bool>(n).Named("y");
            y[n] = x[n] | const1[n];

            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            //engine.BrowserMode = BrowserMode.Always;
            Console.WriteLine(engine.Infer(y));
            DistributionArray<Bernoulli> yActual = engine.Infer<DistributionArray<Bernoulli>>(y);
            IDistribution<bool[]> yExpected = Distribution<bool>.Array(new Bernoulli[]
                {
                    new Bernoulli(1), new Bernoulli(0.5)
                });
            Assert.True(yExpected.MaxDiff(yActual) < 1e-10);
        }

        [Fact]
        public void ArrayOperatorTest()
        {
            Range item = new Range(2).Named("item");
            Range feature = new Range(3).Named("feature");
            VariableArray<double> array = Variable.Array<double>(item).Named("array");
            array[item] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(item);
            VariableArray2D<double> array2D = Variable.Constant(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } }, item, feature).Named("array2D");
            VariableArray2D<double> product = Variable.Array<double>(item, feature).Named("product");
            product[item, feature] = array[item] * array2D[item, feature];
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(product));
        }

        /// <summary>
        /// Tests when building MSL that the uses of a variable come after its definition.
        /// </summary>
        [Fact]
        public void DeclarationOrderTest()
        {
            Variable<bool> x = Variable.Bernoulli(0.1).Named("x");
            Variable<bool> y = (!x).Named("y");
            Variable<bool> z = (x | y).Named("z");
            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            //engine.ShowSchedule = true;
            Bernoulli yActual = engine.Infer<Bernoulli>(y);
            Bernoulli yExpected = new Bernoulli(0.9);
            Console.WriteLine("y = {0}", yActual);
            Assert.True(yExpected.MaxDiff(yActual) < 1e-10);
        }

        public static T[] Replicate<T>(T value, int length)
        {
            T[] result = new T[length];
            for (int i = 0; i < length; i++)
            {
                result[i] = value;
            }
            return result;
        }

        [Fact]
        public void ClickChainTest()
        {
            try
            {
                // observation[rank][user] 
                //bool[][] observation = { new bool[] { false }, new bool[] { false }, new bool[] { true } };
                bool[][] observation = new bool[2][];
                observation[0] = Replicate(true, 12);
                observation[1] = Replicate(false, 12);

                int nRanks = observation.Length;
                int nUsers = observation[0].Length;
                Range user = new Range(nUsers).Named("user");

                Variable<bool>[] rel = new Variable<bool>[nRanks];
                Variable<double>[] summary = new Variable<double>[nRanks];
                // examine[rank][user]
                VariableArray<bool>[] examine = new VariableArray<bool>[nRanks];
                VariableArray<bool>[] click = new VariableArray<bool>[nRanks];

                // user independent variables
                for (int rank = 0; rank < nRanks; rank++)
                {
                    summary[rank] = Variable.Beta(1, 1).Named("Summary" + rank);
                    rel[rank] = (Variable.Bernoulli(summary[rank]) == Variable.Bernoulli(0.8)).Named("Rel" + rank);
                }
                //user specific variables
                for (int rank = 0; rank < nRanks; rank++)
                {
                    examine[rank] = Variable.Array<bool>(user).Named("Examine" + rank);
                    click[rank] = Variable.Constant(observation[rank], user).Named("Click" + rank);
                    if (rank == 0)
                    {
                        examine[rank][user] = Variable.Bernoulli(1).ForEach(user);
                        //Variable.ConstrainEqual(examine[rank][user], true);
                    }
                    else
                    {
                        using (Variable.ForEach(user))
                        {
                            Variable<bool> examineIfClick = Variable.New<bool>().Named("examineIfClick" + rank);
                            using (Variable.If(rel[rank - 1]))
                                examineIfClick.SetTo(Variable.Bernoulli(0.2));
                            using (Variable.IfNot(rel[rank - 1]))
                                examineIfClick.SetTo(Variable.Bernoulli(0.9));
                            examine[rank][user] = examine[rank - 1][user] &
                                                  ((!click[rank - 1][user] & Variable.Bernoulli(.8))
                                                   | (click[rank - 1][user] & examineIfClick));
                        }
                    }
                    click[rank][user] = examine[rank][user] & Variable.Bernoulli(summary[rank]).ForEach(user);
                }

                InferenceEngine ie = new InferenceEngine();
                //ie.BrowserMode = BrowserMode.Always;
                for (int i = 0; i < nRanks; i++)
                {
                    //  Console.WriteLine("Relevance of document " +i +": "+ ie.Infer(rel));
                    Console.WriteLine("Relevance of document {0}: {1}", i, ie.Infer(rel[i]));
                }
            }
            catch (CompilationFailedException ex)
            {
                Console.WriteLine("Correctly threw " + ex);
            }
        }

        /// <summary>
        /// Click chain with unrolled users
        /// </summary>
        [Fact]
        public void ClickChainTest2()
        {
            // observation[rank][user] 
            //bool[][] observation = { new bool[] { false }, new bool[] { false }, new bool[] { true } };
            bool[][] observation = new bool[2][];
            observation[0] = Replicate(true, 12);
            observation[1] = Replicate(false, 12);

            int nRanks = observation.Length;
            int nUsers = observation[0].Length;

            Variable<bool>[] rel = new Variable<bool>[nRanks];
            Variable<double>[] summary = new Variable<double>[nRanks];
            // examine[rank][user]
            Variable<bool>[][] examine = new Variable<bool>[nRanks][];
            Variable<bool>[][] click = new Variable<bool>[nRanks][];

            // user independent variables
            for (int rank = 0; rank < nRanks; rank++)
            {
                summary[rank] = Variable.Beta(1, 1).Named("Summary" + rank);
                rel[rank] = (Variable.Bernoulli(summary[rank]) == Variable.Bernoulli(0.8)).Named("Rel" + rank);
            }
            //user specific variables
            for (int rank = 0; rank < nRanks; rank++)
            {
                examine[rank] = new Variable<bool>[nUsers];
                click[rank] = new Variable<bool>[nUsers];
                for (int user = 0; user < nUsers; user++)
                {
                    click[rank][user] = Variable.Constant(observation[rank][user]).Named("Click" + rank + user);

                    if (rank == 0)
                    {
                        examine[rank][user] = Variable.Bernoulli(1);
                        //Variable.ConstrainEqual(examine[rank][user], true);
                    }
                    else
                    {
                        Variable<bool> examineIfClick = Variable.New<bool>().Named("examineIfClick" + rank + user);
                        using (Variable.If(rel[rank - 1]))
                            examineIfClick.SetTo(Variable.Bernoulli(0.2));
                        using (Variable.IfNot(rel[rank - 1]))
                            examineIfClick.SetTo(Variable.Bernoulli(0.9));
                        examine[rank][user] = examine[rank - 1][user] &
                                              ((!click[rank - 1][user] & Variable.Bernoulli(.8))
                                               | (click[rank - 1][user] & examineIfClick));
                    }
                    examine[rank][user].Name = "Examine" + rank + user;
                    Variable.ConstrainEqual(click[rank][user], examine[rank][user] & Variable.Bernoulli(summary[rank]));
                }
            }

            InferenceEngine ie = new InferenceEngine();
            //ie.BrowserMode = BrowserMode.Always;
            for (int i = 0; i < nRanks; i++)
            {
                //  Console.WriteLine("Relevance of document " +i +": "+ ie.Infer(rel));
                Console.WriteLine("Relevance of document {0}: {1}", i, ie.Infer(rel[i]));
            }
        }

        [Fact]
        public void BetaConstructionFromMeanAndVarianceTest()
        {
            var b = Variable.BetaFromMeanAndVariance(Variable.Constant(0.5), Variable.Constant(1.0));
            Console.WriteLine(new InferenceEngine().Infer(b));
        }

        [Fact]
        public void BetaConstructionTest()
        {
            var b = Variable.Beta(Variable.Constant(1.0), Variable.Constant(1.0));
            Console.WriteLine(new InferenceEngine().Infer(b));
        }

        [Fact]
        public void GammaConstructionFromMeanAndVarianceTest()
        {
            var g = Variable.GammaFromMeanAndVariance(Variable.Constant(0.5), Variable.Constant(1.0));
            Console.WriteLine(new InferenceEngine().Infer(g));
        }

        [Fact]
        public void GammaConstructionFromShapeAndScaleTest()
        {
            var g = Variable.GammaFromShapeAndScale(Variable.Constant(1.0), Variable.Constant(1.0));
            Console.WriteLine(new InferenceEngine().Infer(g));
        }

        [Fact]
        public void WishartConstructionFromShapeAndScaleTest()
        {
            var w = Variable.WishartFromShapeAndScale(Variable.Constant(1.0), Variable.Constant(PositiveDefiniteMatrix.Identity(5)));
            Console.WriteLine(new InferenceEngine().Infer(w));
        }

        [Fact]
        public void WishartConstructionFromShapeAndRateTest()
        {
            var w = Variable.WishartFromShapeAndRate(Variable.Constant(1.0), Variable.Constant(PositiveDefiniteMatrix.Identity(5)));
            Console.WriteLine(new InferenceEngine().Infer(w));
        }

        [Fact]
        public void VectorGaussianConstructionTest()
        {
            var mean = Vector.Zero(10);
            var cov = PositiveDefiniteMatrix.Identity(mean.Count);
            var v = Variable.VectorGaussianFromMeanAndVariance(Variable.Constant(mean), Variable.Constant(cov));
            Console.WriteLine(new InferenceEngine().Infer(v));
        }

        [Fact]
        public void CoupledHmmTest()
        {
            Diffable previous = default(Diffable);
            for (int i = 0; i < 3; i++)
            {
                Diffable result = CoupledHmmHelper(i);
                Console.WriteLine(result);
                if (i == 0)
                    previous = result;
                else
                    Assert.True(previous.MaxDiff(result) < 1e-10);
            }
        }

        private Diffable CoupledHmmHelper(int modelType)
        {
            // Emissions for two sites at 3 time steps (single sequence)
            int[][] emissions = new int[][]
                {
                    new int[] {1, 2}, new int[] {0, 1}, new int[] {1, 2}
                };

            CoupledHmm model = new CoupledHmm();
            model.CreateModel(emissions.Length, modelType);

            // Observe the data:
            for (int t = 0; t < emissions.Length; t++)
                model.X[t].ObservedValue = emissions[t];

            // Number of state values and output values per site:
            int[] numStatesPerSite = new int[] { 4, 3 };
            int[] numOutputsPerSite = new int[] { 2, 3 };
            int numSites = numStatesPerSite.Length;
            model.Sites.ObservedValue = numSites;
            model.StatesPerSite.ObservedValue = numStatesPerSite;
            model.OutputsPerSite.ObservedValue = numOutputsPerSite;
            model.Aprior.ObservedValue = Util.ArrayInit(numSites, site => Dirichlet.Uniform(numStatesPerSite[site]));
            model.engine.Compiler.OptimiseInferenceCode = false;

            return model.engine.Infer<Diffable>(model.A);
        }

        private class CoupledHmm
        {
            public Variable<int> Sites;
            public VariableArray<int> StatesPerSite;
            public VariableArray<int> OutputsPerSite;
            public VariableArray<VariableArray<VariableArray<Vector>, Vector[][]>, Vector[][][]> A;
            public VariableArray<Dirichlet> Aprior;
            public VariableArray<VariableArray<Vector>, Vector[][]> C;
            public VariableArray<int>[] Z;
            public VariableArray<int>[] X;
            public InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());

            public void CreateModel(int numTimeSteps, int type = 0)
            {
                // Number of sites (observed)
                Sites = Variable.New<int>().Named("Sites");
                // Range for current site
                Range Site_i = new Range(Sites).Named("Site_i");
                // Range for influnecing site
                Range Site_j = Site_i.Clone().Named("Site_j");
                // Number of discrete states per site (observed)
                StatesPerSite = Variable.Array<int>(Site_i).Named("StatesPerSite");
                // Range for state in current site
                Range State_i = new Range(StatesPerSite[Site_i]).Named("State_i"); // State for current site
                // Range for state in influencing site
                Range State_j = new Range(StatesPerSite[Site_j]).Named("State_j");
                //Range State_j = State_i.Clone().Named("State_j");
                // Number of discrete outputs per site (observed)
                OutputsPerSite = Variable.Array<int>(Site_i).Named("OutputsPerSite");
                // Range for output in current site
                Range Output = new Range(OutputsPerSite[Site_i]).Named("Output");

                // Each (current site, influencing site) pair have a set of probability
                // vectors which determine the state transition from the influencing site
                // to the current site.
                A = Variable.Array(Variable.Array(Variable.Array<Vector>(State_j), Site_j), Site_i).Named("A");
                Aprior = Variable.Array<Dirichlet>(Site_i).Named("Aprior");
                if (type == 2)
                {
                    using (Variable.ForEach(Site_i))
                    using (Variable.ForEach(Site_j))
                    using (Variable.ForEach(State_j))
                        A[Site_i][Site_j][State_j] = Variable.DirichletUniform(State_i);
                }
                else if (type == 1)
                {
                    using (Variable.ForEach(Site_i))
                        A[Site_i][Site_j][State_j] = Variable.DirichletUniform(State_i).ForEach(Site_j, State_j);
                }
                else
                {
                    using (Variable.ForEach(Site_i))
                    using (Variable.ForEach(Site_j))
                    using (Variable.ForEach(State_j))
                        A[Site_i][Site_j][State_j] = Variable<Vector>.Random(Aprior[Site_i]);
                    A.SetValueRange(State_i);
                }

                // The probability vectors determining the emission variable for the current site
                // conditional on the current state at the current sit
                C = Variable.Array(Variable.Array<Vector>(State_i), Site_i).Named("C");
                using (Variable.ForEach(Site_i))
                    //using (Variable.ForEach(State_i))
                    C[Site_i][State_i] = Variable.DirichletUniform(Output).ForEach(State_i);

                // The influence matrix - determines the influencing site given the current site.
                // Priors should be non-uniform, typically favouring the current site.
                VariableArray<Vector> Influence = Variable.Array<Vector>(Site_i).Named("Influence");
                Influence[Site_i] = Variable.DirichletUniform(Site_j).ForEach(Site_i);

                // Z are the states, X the emissions
                // The VariableArray is over sites
                Z = new VariableArray<int>[numTimeSteps];
                X = new VariableArray<int>[numTimeSteps];
                for (int t = 0; t < numTimeSteps; t++)
                {
                    Z[t] = Variable.Array<int>(Site_i).Named("Z" + t);
                    X[t] = Variable.Array<int>(Site_i).Named("X" + t);
                }

                Z[0][Site_i] = Variable.DiscreteUniform(State_i);
                using (Variable.ForEach(Site_i))
                using (Variable.Switch(Z[0][Site_i]))
                    X[0][Site_i] = Variable.Discrete(C[Site_i][Z[0][Site_i]]);

                for (int t = 1; t < numTimeSteps; t++)
                {
                    using (Variable.ForEach(Site_i))
                    {
                        // Infuencing site is randomly chosen from the influence matrix
                        var influencingSite = Variable.Discrete(Influence[Site_i]).Named("InfSite" + t);
                        //influencingSite.SetValueRange(Site_j);
                        // Switch on the influencing site
                        using (Variable.Switch(influencingSite))
                        {
                            // Switch on the influencing state
                            // Must make a copy so we don't change Z's ValueRange
                            var influencingState = Variable.Copy(Z[t - 1][influencingSite]).Named("InfState" + t);
                            influencingState.SetValueRange(State_j);
                            using (Variable.Switch(influencingState))
                            {
                                // Switch on the current state
                                Z[t][Site_i] = Variable.Discrete(A[Site_i][influencingSite][influencingState]);
                                // Switch on the current state
                                using (Variable.Switch(Z[t][Site_i]))
                                {
                                    // Emission
                                    X[t][Site_i] = Variable.Discrete(C[Site_i][Z[t][Site_i]]);
                                }
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Unrolled HMM with gated CPT. If 'coupled' and/or 'gated' are not defined
        /// this model compiles fine. If they are both defined, the scheduler chokes
        /// </summary>
        private class HMMWithGatedP
        {
#if gated
            public Variable<bool> useScheme1;
#endif

            // CPT for state transition scheme 1
            public Variable<double> P1_z1T_cond_z1T;
            public Variable<double> P1_z1T_cond_z1F;
            public Variable<double> P1_z2T_cond_z2T;
            public Variable<double> P1_z2T_cond_z2F;

            // CPT for state transition scheme 2
            public Variable<double> P2_z1T_cond_z1T;
            public Variable<double> P2_z1T_cond_z1F;
#if coupled
            public Variable<double> P2_z2T_cond_z1T_z2T;
            public Variable<double> P2_z2T_cond_z1T_z2F;
            public Variable<double> P2_z2T_cond_z1F_z2T;
            public Variable<double> P2_z2T_cond_x1F_z2F;
#else
            public Variable<double> P2_z2T_cond_z2T;
            public Variable<double> P2_z2T_cond_z2F;
#endif

            // Emissions for the GP
            public Variable<double> R_x1T_cond_z1T;
            public Variable<double> R_x1T_cond_z1F;
            public Variable<double> R_x2T_cond_z2T;
            public Variable<double> R_x2T_cond_z2F;

            // State variables
            public Variable<bool>[] z1;
            public Variable<bool>[] z2;

            // Emission variables
            public Variable<bool>[] x1;
            public Variable<bool>[] x2;

            // Use EP for HMM for accuracy
            public InferenceEngine engine = new InferenceEngine();

            public HMMWithGatedP(int numTimeSteps)
            {
#if gated
                useScheme1 = Variable<bool>.Bernoulli(0.5).Named("useScheme1");
#endif
                // CPTs for state transition scheme1
                P1_z1T_cond_z1T = Variable.Beta(1, 1).Named("P1_z1T_cond_z1T");
                P1_z1T_cond_z1F = Variable.Beta(1, 1).Named("P1_z1T_cond_z1F");
                P1_z2T_cond_z2T = Variable.Beta(1, 1).Named("P1_z2T_cond_z2T");
                P1_z2T_cond_z2F = Variable.Beta(1, 1).Named("P1_z2T_cond_z2F");
                P1_z1T_cond_z1T.ObservedValue = 0.5;
                P1_z1T_cond_z1F.ObservedValue = 0.5;
                P1_z2T_cond_z2T.ObservedValue = 0.5;
                P1_z2T_cond_z2F.ObservedValue = 0.5;

                // CPTs for state transition scheme2
                P2_z1T_cond_z1T = Variable.Beta(1, 1).Named("P2_z1T_cond_z1T");
                P2_z1T_cond_z1F = Variable.Beta(1, 1).Named("P2_z1T_cond_z1F");
                P2_z1T_cond_z1T.ObservedValue = 0.5;
                P2_z1T_cond_z1F.ObservedValue = 0.5;
#if coupled
                P2_z2T_cond_z1T_z2T = Variable.Beta(1, 1).Named("P2_z2T_cond_z1T_z2T");
                P2_z2T_cond_z1T_z2F = Variable.Beta(1, 1).Named("P2_z2T_cond_z1T_z2F");
                P2_z2T_cond_z1F_z2T = Variable.Beta(1, 1).Named("P2_z2T_cond_z1F_z2T");
                P2_z2T_cond_x1F_z2F = Variable.Beta(1, 1).Named("P2_z2T_cond_x1F_z2F");
                P2_z2T_cond_z1T_z2T.ObservedValue = 0.5;
                P2_z2T_cond_z1T_z2F.ObservedValue = 0.5;
                P2_z2T_cond_z1F_z2T.ObservedValue = 0.5;
                P2_z2T_cond_x1F_z2F.ObservedValue = 0.5;
#else
                P2_z2T_cond_z2T = Variable.Beta(1, 1).Named("P2_z2T_cond_z2T");
                P2_z2T_cond_z2F = Variable.Beta(1, 1).Named("P2_z2T_cond_z2F");
#endif

                // CPTs for GP emissions
                R_x1T_cond_z1T = Variable.Beta(1, 1).Named("R_x1T_cond_z1T");
                R_x1T_cond_z1F = Variable.Beta(1, 1).Named("R_x1T_cond_z1F");
                R_x2T_cond_z2T = Variable.Beta(1, 1).Named("R_x2T_cond_z2T");
                R_x2T_cond_z2F = Variable.Beta(1, 1).Named("R_x2T_cond_z2F");
                R_x1T_cond_z1T.ObservedValue = 0.5;
                R_x1T_cond_z1F.ObservedValue = 0.5;
                R_x2T_cond_z2T.ObservedValue = 0.5;
                //R_x2T_cond_z2F.ObservedValue = 0.5;

                z1 = new Variable<bool>[numTimeSteps];
                z2 = new Variable<bool>[numTimeSteps];
                x1 = new Variable<bool>[numTimeSteps];
                x2 = new Variable<bool>[numTimeSteps];

                for (int t = 0; t < numTimeSteps; t++)
                {
                    z1[t] = Variable.New<bool>().Named("z1_" + t);
                    //z1[t].InitialiseTo(new Bernoulli(0.5));
                    //if(t==0) z1[t].AddAttribute(new InitialiseBackward());
                    z2[t] = Variable.New<bool>().Named("z2_" + t);
                    //z2[t].InitialiseTo(new Bernoulli(0.5));
                    //if(t==0) z2[t].AddAttribute(new InitialiseBackward());
                    x1[t] = Variable.New<bool>().Named("x1_" + t);
                    x2[t] = Variable.New<bool>().Named("x2_" + t);
                }

                for (int t = 0; t < numTimeSteps; t++)
                {
                    if (t == 0)
                    {
                        z1[t].SetTo(Variable.Bernoulli(.5));
                        z2[t].SetTo(Variable.Bernoulli(.5));
                    }
                    else
                    {
#if gated
                        using (Variable.If(useScheme1))
                        {
                            FromSingleParent(z1[t], z1[t - 1], P1_z1T_cond_z1T, P1_z1T_cond_z1F);
                            FromSingleParent(z2[t], z2[t - 1], P1_z2T_cond_z2T, P1_z2T_cond_z2F);
                        }

                        using (Variable.IfNot(useScheme1))
                        {
#endif
                            FromSingleParent(z1[t], z1[t - 1], P2_z1T_cond_z1T, P2_z1T_cond_z1F);
#if coupled
                            FromTwoParents(z2[t], z1[t - 1], z2[t - 1],
                                           P2_z2T_cond_z1T_z2T, P2_z2T_cond_z1T_z2F,
                                           P2_z2T_cond_z1F_z2T, P2_z2T_cond_x1F_z2F);
#else
                            FromSingleParent(z2[t], z2[t - 1], P2_z2T_cond_z2T, P2_z2T_cond_z2F);
#endif

#if gated
                        }
#endif
                    }

                    // Emissions
                    FromSingleParent(x1[t], z1[t], R_x1T_cond_z1T, R_x1T_cond_z1F);
                    FromSingleParent(x2[t], z2[t], R_x2T_cond_z2T, R_x2T_cond_z2F);
                }

                engine = new InferenceEngine();
                engine.Compiler.ShowProgress = true;
                engine.ShowProgress = false;
                engine.Compiler.GivePriorityTo(typeof(ReplicateOp_NoDivide));
                //engine.Compiler.OptimiseInferenceCode = false;
            }

            public void Train()
            {
                int numTimeSteps = z1.Length;

                for (int t = 0; t < numTimeSteps; t++)
                {
                    x1[t].ObservedValue = true;
                    x2[t].ObservedValue = true;
                }

                engine.NumberOfIterations = 1;

                // Variables to infer
                IList<IVariable> variablesToInfer = new List<IVariable>()
                    {
#if gated
                        useScheme1,
#endif
                        P1_z1T_cond_z1T,
                        P1_z1T_cond_z1F,
                        P1_z2T_cond_z2T,
                        P1_z2T_cond_z2F,
                        P2_z1T_cond_z1T,
                        P2_z1T_cond_z1F,
#if coupled
                        P2_z2T_cond_z1T_z2T,
                        P2_z2T_cond_z1T_z2F,
                        P2_z2T_cond_z1F_z2T,
                        P2_z2T_cond_x1F_z2F,
#else
                    P2_z2T_cond_z2T,
                    P2_z2T_cond_z2F,
#endif
                        R_x1T_cond_z1T,
                        R_x1T_cond_z1F,
                    };

                for (int t = 0; t < numTimeSteps; t++)
                {
                    variablesToInfer.Add(z1[t]);
                    variablesToInfer.Add(z2[t]);
                }

                engine.GetCompiledInferenceAlgorithm(variablesToInfer.ToArray());
            }

            public static T[][] Transpose<T>(T[][] array)
            {
                int numRows = array.Length;
                int numCols = array[0].Length;
                T[][] result = new T[numCols][];

                for (int j = 0; j < numCols; j++)
                {
                    result[j] = new T[numRows];
                    for (int i = 0; i < numRows; i++)
                    {
                        result[j][i] = array[i][j];
                    }
                }

                return result;
            }
        }

        [Fact]
        public void HMMWithGatedPTest()
        {
            // Fails because RepairSchedule causes schedule length to explode
            // seems related to KeepFresh (attached to cases arrays and selectors)
            //DependencyAnalysisTransform.KeepFresh = false;
            // numTimeSteps=2: schedule should have 84 nodes after repair
            //HMMWithGatedP model = new HMMWithGatedP(2);
            HMMWithGatedP model = new HMMWithGatedP(12);
            model.Train();
        }

        private static void FromSingleParent(
            Variable<bool> result,
            Variable<bool> condition,
            Variable<double> condTrueProb,
            Variable<double> condFalseProb)
        {
            using (Variable.If(condition))
            {
                result.SetTo(Variable.Bernoulli(condTrueProb));
            }
            using (Variable.IfNot(condition))
            {
                result.SetTo(Variable.Bernoulli(condFalseProb));
            }
        }

        private static void FromTwoParents(
            Variable<bool> result,
            Variable<bool> condition1,
            Variable<bool> condition2,
            Variable<double> condTrueTrueProb,
            Variable<double> condTrueFalseProb,
            Variable<double> condFalseTrueProb,
            Variable<double> condFalseFalseProb)
        {
            using (Variable.If(condition1))
            {
                using (Variable.If(condition2))
                {
                    result.SetTo(Variable.Bernoulli(condTrueTrueProb));
                }
                using (Variable.IfNot(condition2))
                {
                    result.SetTo(Variable.Bernoulli(condTrueFalseProb));
                }
            }
            using (Variable.IfNot(condition1))
            {
                using (Variable.If(condition2))
                {
                    result.SetTo(Variable.Bernoulli(condFalseTrueProb));
                }
                using (Variable.IfNot(condition2))
                {
                    result.SetTo(Variable.Bernoulli(condFalseFalseProb));
                }
            }
        }

        [Fact]
        public void HMMWithObservationScore()
        {
            var numChildren = 2;
            var numClusters = 2;
            var numTimes = 2;

            var n = new Range(numChildren).Named("n");
            var k = new Range(numClusters).Named("k");
            var t = new Range(numTimes).Named("t");

            var z = Variable.Array(Variable.Array<bool>(t), n).Named("z");
            var x = Variable.Array(Variable.Array<bool>(t), n).Named("x");
            var P_T = Variable.Array<double>(k).Named("PT");
            var P_F = Variable.Array<double>(k).Named("PF");
            P_T[k] = Variable.Beta(1, 1).ForEach(k);
            P_F[k] = Variable.Beta(1, 1).ForEach(k);
            var cluster = Variable.Array<int>(n).Named("cluster");
            var mixingCoeffs = Variable.DirichletSymmetric(k, 0.1).Named("Pi");

            using (Variable.ForEach(n))
            {
                cluster[n] = Variable.Discrete(mixingCoeffs);

                using (Variable.Switch(cluster[n]))
                {
                    using (ForEachBlock tBlock = Variable.ForEach(t))
                    {
                        using (Variable.If(tBlock.Index == 0))
                        {
                            z[n][tBlock.Index] = Variable.Bernoulli(0.5);
                        }

                        using (Variable.If(tBlock.Index > 0))
                        {
                            FromSingleParent(
                                z[n][tBlock.Index],
                                z[n][tBlock.Index - 1],
                                P_T[cluster[n]],
                                P_F[cluster[n]]);
                        }
                    }
                }

                // Score-based observation model
                using (Variable.ForEach(t))
                {
                    using (Variable.If(z[n][t]))
                    {
                        x[n][t] = Variable.GaussianFromMeanAndPrecision(-1, 1.0).Named("scoreT") > 0;
                    }

                    using (Variable.IfNot(z[n][t]))
                    {
                        x[n][t] = Variable.GaussianFromMeanAndPrecision(1, 1.0).Named("scoreF") > 0;
                    }
                }
            }

            var engine = new InferenceEngine();
            // This prevents a serial schedule when UseSerialSchedules=true
            //engine.Compiler.GivePriorityTo(typeof (ReplicateOp_NoDivide));
            x.ObservedValue = Util.ArrayInit(numChildren, i => Util.ArrayInit(numTimes, j => Bernoulli.Sample(0.5)));
            engine.Infer(z);
        }
    }
}
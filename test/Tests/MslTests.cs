// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Xunit;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using GaussianArray = Microsoft.ML.Probabilistic.Distributions.DistributionStructArray<Microsoft.ML.Probabilistic.Distributions.Gaussian, double>;
using GaussianArray2D = Microsoft.ML.Probabilistic.Distributions.DistributionStructArray2D<Microsoft.ML.Probabilistic.Distributions.Gaussian, double>;

namespace Microsoft.ML.Probabilistic.Tests
{
    using Assert = Xunit.Assert;
    using Microsoft.ML.Probabilistic.Utilities;
    using Microsoft.ML.Probabilistic.Compiler;
    using Microsoft.ML.Probabilistic.Algorithms;
    using Microsoft.ML.Probabilistic.Models;

    public abstract class MslTestsBase
    {
        internal abstract void MethodOverrideModel(double[] prec);
    }


    public class MslTests : MslTestsBase
    {
        /// <summary>
        /// Dummy property to test that properties in MSL are handled correctly.
        /// </summary>
        public int MyProp
        {
            get
            {
                return 0;
            }
            set
            {
                Console.WriteLine(value);
            }
        }

        /// <summary>
        /// Test compiling a method that overrides an inherited method.
        /// </summary>
        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MethodOverrideTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(MethodOverrideModel, new double[] { 10.0 });
            ca.Execute(engine.NumberOfIterations);
        }

        internal override void MethodOverrideModel(double[] prec)
        {
            double x = Factor.Random(new Gaussian(0, 1));
            double[] y = new double[prec.Length];
            for (int i = 0; i < prec.Length; i++)
            {
                y[i] = Factor.Gaussian(x, prec[i]);
            }
            InferNet.Infer(x, nameof(x), QueryTypes.Marginal);
            InferNet.Infer(y, nameof(y), QueryTypes.Marginal);
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MethodInAnotherFileTest()
        {
            Assert.Throws<CompilationFailedException>(() =>
            {
                InferenceEngine engine = new InferenceEngine();
                engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
                engine.Compiler.RequiredQuality = Factors.Attributes.QualityBand.Unknown;
                var ca = engine.Compiler.Compile(MethodInAnotherFileModel);
                ca.Execute(engine.NumberOfIterations);
            });
        }

        private static void MethodInAnotherFileModel()
        {
            // Could also use EpTests.IsPositive
            bool b = Factor.Bernoulli(0.5);
            FactorManagerTests.MissingBufferMethodsFailureFactorAndOp.Factor(b);
        }

        /// <summary>
        /// Test compiling a model method containing checked arithmetic.
        /// </summary>
        [Fact]
        [Trait("Category", "CsoftModel")]
        public void CheckedArithmeticTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(CheckedArithmeticModel);
            ca.Execute(engine.NumberOfIterations);
        }

        private void CheckedArithmeticModel()
        {
            int x = 1;
            int y = 2;
            int z = checked(x + y);
            //int z = Factor.Plus(x, y);
            InferNet.Infer(z, nameof(z), QueryTypes.Marginal);
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void ParityCheckTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            double[] y = Util.ArrayInit(6, i => 1.0);
            double variance = 1;
            var ca = engine.Compiler.Compile(ParityCheckModel, y, variance);
            ca.Execute(engine.NumberOfIterations);
        }

        private void ParityCheckModel(double[] y, double Variance)
        {
            bool[] x = new bool[6];
            int[] Intermediate = new int[6];
            //InferNet.Infer(y, nameof(y));
            Bernoulli vBernoulli1 = Bernoulli.FromLogOdds(0);
            Discrete vDiscrete0 = new Discrete(DenseVector.FromArray(new double[2] {0.5, 0.5}));
            for (int index3 = 0; index3 < 6; index3++)
            {
                x[index3] = Factor.Random<bool>(vBernoulli1);
                Intermediate[index3] = Factor.Random<int>(vDiscrete0);
                if (x[index3])
                {
                    Constrain.Equal<int>(Intermediate[index3], 1);
                }
                if (!x[index3])
                {
                    Constrain.Equal<int>(Intermediate[index3], 0);
                }
                if (Intermediate[index3] == 0)
                {
                    y[index3] = Factor.GaussianFromMeanAndVariance(1, Variance);
                }
                if (Intermediate[index3] == 1)
                {
                    y[index3] = Factor.GaussianFromMeanAndVariance(-1, Variance);
                }
            }
            bool vbool23 = x[3] == x[2];
            bool vbool24 = !vbool23;
            bool vbool25 = x[4] == vbool24;
            bool vbool26 = !vbool25;
            Constrain.Equal<bool>(vbool26, false);
            bool vbool31 = x[3] == x[0];
            bool vbool32 = !vbool31;
            bool vbool33 = x[4] == vbool32;
            bool vbool34 = !vbool33;
            Constrain.Equal<bool>(vbool34, false);
            bool vbool39 = x[3] == x[0];
            bool vbool40 = !vbool39;
            bool vbool41 = x[5] == vbool40;
            bool vbool42 = !vbool41;
            Constrain.Equal<bool>(vbool42, false);
            bool vbool47 = x[1] == x[0];
            bool vbool48 = !vbool47;
            bool vbool49 = x[4] == vbool48;
            bool vbool50 = !vbool49;
            Constrain.Equal<bool>(vbool50, false);
            bool vbool55 = x[4] == x[0];
            bool vbool56 = !vbool55;
            bool vbool57 = x[5] == vbool56;
            bool vbool58 = !vbool57;
            Constrain.Equal<bool>(vbool58, false);
            bool vbool63 = x[2] == x[1];
            bool vbool64 = !vbool63;
            bool vbool65 = x[3] == vbool64;
            bool vbool66 = !vbool65;
            Constrain.Equal<bool>(vbool66, false);
            InferNet.Infer(x, "x", QueryTypes.Marginal);
            InferNet.Infer(Intermediate, "Intermediate", QueryTypes.Marginal);
            InferNet.Infer(vbool23, "vbool23", QueryTypes.Marginal);
            InferNet.Infer(vbool24, "vbool24", QueryTypes.Marginal);
            InferNet.Infer(vbool25, "vbool25", QueryTypes.Marginal);
            InferNet.Infer(vbool26, "vbool26", QueryTypes.Marginal);
            InferNet.Infer(vbool31, "vbool31", QueryTypes.Marginal);
            InferNet.Infer(vbool32, "vbool32", QueryTypes.Marginal);
            InferNet.Infer(vbool33, "vbool33", QueryTypes.Marginal);
            InferNet.Infer(vbool34, "vbool34", QueryTypes.Marginal);
            InferNet.Infer(vbool39, "vbool39", QueryTypes.Marginal);
            InferNet.Infer(vbool40, "vbool40", QueryTypes.Marginal);
            InferNet.Infer(vbool41, "vbool41", QueryTypes.Marginal);
            InferNet.Infer(vbool42, "vbool42", QueryTypes.Marginal);
            InferNet.Infer(vbool47, "vbool47", QueryTypes.Marginal);
            InferNet.Infer(vbool48, "vbool48", QueryTypes.Marginal);
            InferNet.Infer(vbool49, "vbool49", QueryTypes.Marginal);
            InferNet.Infer(vbool50, "vbool50", QueryTypes.Marginal);
            InferNet.Infer(vbool55, "vbool55", QueryTypes.Marginal);
            InferNet.Infer(vbool56, "vbool56", QueryTypes.Marginal);
            InferNet.Infer(vbool57, "vbool57", QueryTypes.Marginal);
            InferNet.Infer(vbool58, "vbool58", QueryTypes.Marginal);
            InferNet.Infer(vbool63, "vbool63", QueryTypes.Marginal);
            InferNet.Infer(vbool64, "vbool64", QueryTypes.Marginal);
            InferNet.Infer(vbool65, "vbool65", QueryTypes.Marginal);
            InferNet.Infer(vbool66, "vbool66", QueryTypes.Marginal);
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void TwoCoinsTest()
        {
            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(TwoCoinsModel);
            ca.Execute(1);
            Console.WriteLine("Marginal for bothHeads=" + ca.Marginal("bothHeads"));
        }

        private void TwoCoinsModel()
        {
            bool firstCoinHeads = Factor.Bernoulli(0.5);
            bool secondCoinHeads = Factor.Bernoulli(0.5);
            bool bothHeads = firstCoinHeads & secondCoinHeads;
            InferNet.Infer(bothHeads, nameof(bothHeads));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void TrueSkillTest()
        {
            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(TrueSkillModel);
            ca.Execute(10);
            Console.WriteLine("Marginal for skill1=" + ca.Marginal("skill1"));
            Console.WriteLine("Marginal for skill2=" + ca.Marginal("skill2"));
            Console.WriteLine("Marginal for skill3=" + ca.Marginal("skill3"));
            // results are compared to the online rank calculator: (0% draw)
            // http://research.microsoft.com/mlp/trueskill/RankCalculator.aspx
            Gaussian[] skillPosts =
                {
                    Gaussian.FromMeanAndVariance(25.592, 6.255*6.255),
                    Gaussian.FromMeanAndVariance(23.611, 6.322*6.322),
                    Gaussian.FromMeanAndVariance(20.165, 7.039*7.039)
                };
            for (int i = 0; i < 3; i++)
            {
                Gaussian newSkill = ca.Marginal<Gaussian>("skill" + (i + 1));
                Assert.True(skillPosts[i].MaxDiff(newSkill) < 0.01);
            }
        }

        private void TrueSkillModel()
        {
            double oldskill1 = Factor.Random(Gaussian.FromMeanAndVariance(15, 8*8));
            double oldskill2 = Factor.Random(Gaussian.FromMeanAndVariance(25, 9*9));
            double oldskill3 = Factor.Random(Gaussian.FromMeanAndVariance(35, 10*10));

            double tau = 144;
            double skill1 = Factor.Gaussian(oldskill1, tau);
            double skill2 = Factor.Gaussian(oldskill2, tau);
            double skill3 = Factor.Gaussian(oldskill3, tau);

            double beta = 0.0576;
            double perf1 = Factor.Gaussian(skill1, beta);
            double perf2 = Factor.Gaussian(skill2, beta);
            double perf3 = Factor.Gaussian(skill3, beta);

            bool h12 = Factor.IsPositive(Factor.Difference(perf1, perf2));
            Constrain.Equal(true, h12);
            bool h23 = Factor.IsPositive(Factor.Difference(perf2, perf3));
            Constrain.Equal(true, h23);


            InferNet.Infer(skill1, nameof(skill1), QueryTypes.Marginal);
            InferNet.Infer(skill2, nameof(skill2), QueryTypes.Marginal);
            InferNet.Infer(skill3, nameof(skill3), QueryTypes.Marginal);
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void TrueSkillTest2()
        {
            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            int[] games_Item1 = new int[] {0};
            int[] games_Item2 = new int[] {1};
            int[] outcome = new int[] {0};
            var ca = engine.Compiler.Compile(TrueSkillModel2, games_Item1, games_Item2, outcome);
            ca.Execute(10);
        }

        private void TrueSkillModel2(int[] games_Item1, int[] games_Item2, int[] outcome)
        {
            double[] skills = new double[5];
            double[] GaussianFromMeanAndPrecisionSample0 = new double[1];
            double[] GaussianFromMeanAndPrecisionSample1 = new double[1];
            double[] inputSequence_Item1 = new double[1];
            for (int range3 = 0; range3 < 5; range3++)
            {
                double vdouble2;
                vdouble2 = Factor.Gaussian(25, 0.014399999999999998);
                skills[range3] = Clone.Copy<double>(vdouble2);
            }
            for (int range2 = 0; range2 < 1; range2++)
            {
                double vdouble8 = Factor.Gaussian(skills[games_Item1[range2]], 0.014399999999999998);
                GaussianFromMeanAndPrecisionSample0[range2] = Clone.Copy<double>(vdouble8);
                double vdouble14 = Factor.Gaussian(skills[games_Item2[range2]], 0.014399999999999998);
                GaussianFromMeanAndPrecisionSample1[range2] = Clone.Copy<double>(vdouble14);
                double vdouble18 = Factor.Difference(GaussianFromMeanAndPrecisionSample0[range2], GaussianFromMeanAndPrecisionSample1[range2]);
                inputSequence_Item1[range2] = Clone.Copy<double>(vdouble18);
                bool vbool0 = outcome[range2] == 1;
                if (vbool0)
                {
                    double vdouble23 = Factor.Difference(inputSequence_Item1[range2], 10);
                    bool vbool1 = Factor.IsPositive(vdouble23);
                    Constrain.Equal<bool>(true, vbool1);
                }
                if (!vbool0)
                {
                    bool vbool3 = outcome[range2] == -1;
                    if (vbool3)
                    {
                        double vdouble26 = Factor.Difference(0.0, 10.0);
                        double vdouble27 = Factor.Difference(vdouble26, inputSequence_Item1[range2]);
                        bool vbool4 = Factor.IsPositive(vdouble27);
                        Constrain.Equal<bool>(true, vbool4);
                    }
                    if (!vbool3)
                    {
                        double vdouble30 = Factor.Difference(0.0, 10.0);
                        double vdouble31 = Factor.Difference(inputSequence_Item1[range2], vdouble30);
                        bool vbool6 = Factor.IsPositive(vdouble31);
                        Constrain.Equal<bool>(true, vbool6);
                        double vdouble33 = Factor.Difference(10, inputSequence_Item1[range2]);
                        bool vbool8 = Factor.IsPositive(vdouble33);
                        Constrain.Equal<bool>(true, vbool8);
                    }
                }
            }
            InferNet.Infer(skills, "skills", QueryTypes.Marginal);
            InferNet.Infer(GaussianFromMeanAndPrecisionSample0, "GaussianFromMeanAndPrecisionSample0", QueryTypes.Marginal);
            InferNet.Infer(GaussianFromMeanAndPrecisionSample1, "GaussianFromMeanAndPrecisionSample1", QueryTypes.Marginal);
            InferNet.Infer(inputSequence_Item1, "inputSequence_Item1", QueryTypes.Marginal);
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void DirichletVariableSizeTest()
        {
            int dimension = 3;
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(DirichletVariableSizeModel, dimension);
            ca.Execute(engine.NumberOfIterations);
            Dirichlet cActual = ca.Marginal<Dirichlet>("c");
            Dirichlet cExpected = Dirichlet.Uniform(dimension);
            Console.WriteLine("c=" + cActual + " should be " + cExpected);
            Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
        }

        private void DirichletVariableSizeModel(int dimension)
        {
            Dirichlet prior = Dirichlet.Uniform(dimension);
            Vector c = Factor.Random(prior);
            InferNet.Infer(c, nameof(c), QueryTypes.Marginal);
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void InitialiseToIgnoredTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(InitialiseToIgnoredModel);
            ca.Execute(1);
            DistributionArray<Bernoulli> cActual = ca.Marginal<DistributionArray<Bernoulli>>("c");
            Console.WriteLine("c=" + cActual[0] + " should be 0.8");
            Assert.True(System.Math.Abs(cActual[0].GetProbTrue() - 0.8) < 1e-10);
        }

        private void InitialiseToIgnoredModel()
        {
            int n = 1;
            Bernoulli[] init = new Bernoulli[n];
            for (int j = 0; j < n; j++)
            {
                init[j] = new Bernoulli(0.1);
            }
            bool[] b = new bool[n];
            bool[] c = new bool[n];
            for (int i = 0; i < n; i++)
            {
                b[i] = Factor.Bernoulli(0.2);
                c[i] = !b[i];
            }
            Attrib.InitialiseTo(b, Distribution<bool>.Array(init));
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(c, nameof(c));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void PropertyIndexerTest()
        {
            int size1 = 1;
            int size2 = 1;
            Bernoulli[,] priorsArray = new Bernoulli[size1,size2];
            priorsArray[0, 0] = new Bernoulli(0.1);
            IArray2D<Bernoulli> priors = (IArray2D<Bernoulli>) Distribution<bool>.Array(priorsArray);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(PropertyIndexerModel, priors, size1, size2);
            ca.Execute(engine.NumberOfIterations);
        }

        private void PropertyIndexerModel(IArray2D<Bernoulli> priors, int size1, int size2)
        {
            bool[,] array = new bool[size1,size2];
            for (int i = 0; i < size1; i++)
            {
                for (int j = 0; j < size2; j++)
                {
                    array[i, j] = Factor.Random<bool>(priors[i, j]);
                }
            }
            InferNet.Infer(array, nameof(array));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void TwoDimStructArrayTest()
        {
            int size1 = 1;
            int size2 = 1;
            Gaussian[,] priors = new Gaussian[size1,size2];
            priors[0, 0] = new Gaussian(0, 1);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(TwoDimStructArrayModel, priors, size1, size2);
            ca.Execute(engine.NumberOfIterations);
        }

        private void TwoDimStructArrayModel(Gaussian[,] priors, int size1, int size2)
        {
            double[,] array = new double[size1,size2];
            for (int i = 0; i < size1; i++)
            {
                for (int j = 0; j < size2; j++)
                {
                    array[i, j] = Factor.Random<double>(priors[i, j]);
                }
            }
            InferNet.Infer(array, nameof(array));
        }

        internal void GetItemsTest()
        {
            int[] indices = new int[] {0, 0};

            InferenceEngine engine = new InferenceEngine();
            var ca = engine.Compiler.Compile(GetItemsModel, indices);
            ca.Execute(engine.NumberOfIterations);

            DistributionArray<Gaussian> xPost = ca.Marginal<DistributionArray<Gaussian>>("x");
            DistributionArray<Gaussian> arrayPost = ca.Marginal<DistributionArray<Gaussian>>("array");

            Console.WriteLine("x:");
            Console.WriteLine(xPost);
            Console.WriteLine("array:");
            Console.WriteLine(arrayPost);

            for (int i = 0; i < indices.Length; i++)
            {
                Assert.True(xPost[i].MaxDiff(arrayPost[indices[i]]) < 1e-10);
            }
        }

        private void GetItemsModel(IReadOnlyList<int> indices)
        {
            double[] c = new double[] {1, 2, 3, 4};
            double[] array = new double[4];
            for (int i = 0; i < 4; i++)
            {
                array[i] = Gaussian.Sample(c[i], 1);
            }
            double[] x = new double[indices.Count];
            x = Collection.GetItems(array, indices);
            InferNet.Infer(array, nameof(array));
            InferNet.Infer(x, nameof(x));
        }

        internal void GivenIndexingTest()
        {
            InferenceEngine engine = new InferenceEngine();
            var ca = engine.Compiler.Compile(GivenIndexingModel, 2, 0);
            ca.Execute(engine.NumberOfIterations);
        }

        private void GivenIndexingModel(int size, int index)
        {
            bool[] x = new bool[size];
            for (int i = 0; i < size; i++)
            {
                x[i] = Bernoulli.Sample(0.5);
            }
            // bool y = Factor.Not(x_item_uses[index][ x_use_lookup[(index,0)] ]);
            // setup code: x_use_lookup[(index,0)] = x_usecount[index]++;
            bool y = !x[index]; // label = 0
            bool[] a = new bool[size];
            for (int j = 0; j < size; j++)
            {
                // handled by FullIndexingTransform
                a[j] = !x[index]; // label = (2,j)
            }
            InferNet.Infer(y, nameof(y));
        }

        internal void GivenIndexing2Test()
        {
            InferenceEngine engine = new InferenceEngine();
            var ca = engine.Compiler.Compile(GivenIndexing2Model, 2, new int[] {0});
            ca.Execute(engine.NumberOfIterations);
        }

        private void GivenIndexing2Model(int size, int[] index)
        {
            bool[] x = new bool[size];
            //bool[,] y = new bool[size,size];
            for (int i = 0; i < size; i++)
            {
                x[i] = Bernoulli.Sample(0.5);
                //for (int j = 0; j < size; j++) {
                //  y[i,j] = Factor.Not(x[i]);
                //}
            }
            bool[] y = new bool[index.Length];
            for (int j = 0; j < index.Length; j++)
            {
                // assume 0 <= index[j] < x.Length
                // y[j] = x_item_uses[index[j]][x_use_index[index[j]]++] ??
                // y[j] = x_item_uses[index[j]][ x_lookup[(index[j],0,j)] ]
                // hash key is (nth use of array with given index, loop indices)
                // copy 'for' loops into setup code
                // setup: x_lookup[index[j]][(0,j)] = x_usecount[index[j]]++;
                // x_usecount
                y[j] = x[index[j]];
                // z[j] = x_item_uses[index[j]][ x_lookup[index[j]] [(1,i,j)] ]
                //z[j] = x[index[j]];
                // there are always index.Length messages, we just don't know to which item they go.
                // x_givenIndex_F[index[j]],x_givenIndex_B[index[j]] works fine
                // but we need to tie them together using a fancy UsesEqualDef:
                // x_givenIndex = Factor.GetItems(x,index)
                // prior to the channel transform, NestedIndexingTransform
                // x[i,index[j]] = x_givenIndex[i,j]  where x_givenIndex = Factor.GetItems(x,:,index)
                // x[index[i],index2[j]] = x_givenIndex[i,j] where x_givenIndex = Factor.GetItems(x,index,index2)
                // the downside is that every different way of indexing x creates a new copy of the whole array.
                // we want Factor.GetItems to not count as a use of x.
            }
            // x_item_uses[0][x_use_index[0]++]
            //z = x[0];
            InferNet.Infer(y, nameof(y));
        }

        internal void ReplicateNdTest()
        {
            InferenceEngine engine = new InferenceEngine();
            var ca = engine.Compiler.Compile(ReplicateNdModel, 2);
            ca.Execute(engine.NumberOfIterations);
        }

        private void ReplicateNdModel(int size)
        {
            // bool x;
            // bool[] x_uses = new bool[size+1+size];
            bool x = Bernoulli.Sample(0.5);
            bool[] y = new bool[size];
            bool[] z = new bool[size];
            bool[,] w = new bool[size,size];
            bool[] v = new bool[size];
            for (int i = 0; i < size; i++)
            {
                // interleaved:
                // y[i] = Factor.Not(x_uses[0+2*i]);
                // z[i] = Factor.Not(x_uses[1+2*i]);
                // not interleaved:
                // y[i] = Factor.Not(x_uses[0+i]);
                // z[i] = Factor.Not(x_uses[size+i]);
                y[i] = !x;
                z[i] = !x;
                for (int j = 0; j < size; j++)
                {
                    // x_uses[size+size+size*i+j]
                    w[i, j] = !x;
                }
                // x counter is now size+size+size*size
            }
            for (int k = 0; k < 2; k++)
            {
                // y[i] = Factor.Not(x_uses[2*size+i]);
                v[k] = !x;
            }
            InferNet.Infer(y, nameof(y));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void LiteralIndexingTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(LiteralIndexingModel);
            ca.Execute(engine.NumberOfIterations);
            var xActual = ca.Marginal<Gaussian>("x");
            var xExpected = Gaussian.FromMeanAndVariance(0, 2);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
        }

        private void LiteralIndexingModel()
        {
            double[] array = new double[2];
            array[0] = Gaussian.Sample(0.0, 1.0);
            array[1] = Gaussian.Sample(0.0, 1.0);

            // array_uses[use][item]
            //Factor.Sum(array);
            // each array has array_uses[use][item] and array_item_uses[item][use]
            // for constant index, replace with array_item_uses, else replace with array_uses.
            // double[][] array_items_uses = Factor.ReplicateJagged(array_uses[0],new int[] {2,1,0});
            // array_items_uses[item][use]
            // array_items_uses[0][0]
            double x = Gaussian.Sample(array[0], 1.0);
            // array_uses[1][0]
            // array_items_uses[0][1]
            double z = Gaussian.Sample(array[0], 1.0);
            //bool b = Factor.BernoulliFromBoolean(true, array);
            //InferNet.Infer(b, nameof(b));
            // array_uses[2][1]
            double y = Gaussian.Sample(array[1], 1.0);
            InferNet.Infer(x, nameof(x));
            InferNet.Infer(y, nameof(y));
            InferNet.Infer(z, nameof(z));
            InferNet.Infer(array, nameof(array));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void LiteralIndexing2Test()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(LiteralIndexing2Model);
            ca.Execute(engine.NumberOfIterations);
            Gaussian xExpected = new Gaussian(1.0, 1/2.0 + 1/9.0);
            Gaussian xActual = ca.Marginal<Gaussian>("x");
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
            Gaussian yExpected = new Gaussian(7.0, 1/8.0 + 1/11.0);
            Gaussian yActual = ca.Marginal<Gaussian>("y");
            Console.WriteLine("y = {0} should be {1}", yActual, yExpected);
            Assert.True(yExpected.MaxDiff(yActual) < 1e-10);
        }

        private void LiteralIndexing2Model()
        {
            double[][] array = new double[2][];
            for (int i = 0; i < 2; i++)
            {
                array[i] = new double[2];
            }
            array[0][0] = Gaussian.Sample(1.0, 2.0);
            array[0][1] = Gaussian.Sample(3.0, 4.0);
            array[1][0] = Gaussian.Sample(5.0, 6.0);
            array[1][1] = Gaussian.Sample(7.0, 8.0);
            // array_uses[use][item]
            //Factor.Sum(array);
            // each array has array_uses[use][item] and array_item_uses[item][use]
            // for constant index, replace with array_item_uses, else replace with array_uses.
            // double[][] array_items_uses = Factor.ReplicateJagged(array_uses[0],new int[] {2,1,0});
            // array_items_uses[item][use]
            // array_items_uses[0][0]
            double x = Gaussian.Sample(array[0][0], 9.0);
            // array_uses[1][0]
            // array_items_uses[0][1]
            double z = Gaussian.Sample(array[0][1], 10.0);
            //bool b = Factor.BernoulliFromBoolean(true, array);
            //InferNet.Infer(b, nameof(b));
            // array_uses[2][1]
            double y = Gaussian.Sample(array[1][1], 11.0);
            double y2 = Gaussian.Sample(array[1][1], 12.0);
            InferNet.Infer(x, nameof(x));
            InferNet.Infer(y, nameof(y));
            InferNet.Infer(y2, nameof(y2));
            InferNet.Infer(z, nameof(z));
            InferNet.Infer(array, nameof(array));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void LiteralIndexing3Test()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(LiteralIndexing3);
            ca.Execute(engine.NumberOfIterations);
        }

        private void LiteralIndexing3()
        {
            double[] y = new double[2];
            for (int j = 0; j < 2; j++)
            {
                y[j] = Factor.Gaussian(0, 1);
            }
            bool[] b = new bool[3];
            for (int i = 0; i < 3; i++)
            {
                b[i] = Factor.IsPositive(y[0]);
            }
            double[] x = new double[2];
            x[0] = Factor.Gaussian(0, 1);
            x[1] = Factor.Gaussian(x[0], 1);
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(x, nameof(x));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void DeclarationInLoopTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(DeclarationInLoopModel);
            ca.Execute(engine.NumberOfIterations);
        }

        private void DeclarationInLoopModel()
        {
            double[,] y = new double[2,3];
            double[] mean = new double[2];
            for (int i = 0; i < 2; i++)
            {
                mean[i] = Factor.Random(new Gaussian(0.0, 1.0));
                for (int j = 0; j < 3; j++)
                {
                    double x = Factor.Gaussian(mean[i], 1.0);
                    y[i, j] = Factor.Gaussian(x, 1);
                }
            }
            InferNet.Infer(y, nameof(y));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void ParameterAsFactorArgumentTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(ParameterAsFactorArgumentModel, new double[] {10.0});
            ca.Execute(engine.NumberOfIterations);
        }

        private void ParameterAsFactorArgumentModel(double[] prec)
        {
            double x = Factor.Random(new Gaussian(0, 1));
            double[] y = new double[prec.Length];
            for (int i = 0; i < prec.Length; i++)
            {
                y[i] = Factor.Gaussian(x, prec[i]);
            }
            InferNet.Infer(x, nameof(x));
            InferNet.Infer(y, nameof(y));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void JaggedArraysTest()
        {
            double[][] data = new double[2][];
            data[0] = new double[] {0, 1, 2};
            data[1] = new double[] {5, 6, 7, 8, 9};
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(JaggedArraysModel, data);
            ca.Execute(engine.NumberOfIterations);
            Console.WriteLine("means=" + ca.Marginal("means"));
            Console.WriteLine("precs=" + ca.Marginal("precs"));
        }

        internal static void JaggedArraysModel(double[][] data)
        {
            double[] means = new double[data.Length];
            double[] precs = new double[data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                means[i] = Factor.Random(new Gaussian(0, 100));
                precs[i] = Factor.Random(new Gamma(0.1, 0.1));
                for (int j = 0; j < data[i].Length; j++)
                {
                    data[i][j] = Factor.Gaussian(means[i], precs[i]);
                }
            }
            InferNet.Infer(means, nameof(means));
            InferNet.Infer(precs, nameof(precs));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void JaggedArrays2Test()
        {
            double[][] data = new double[2][];
            data[0] = new double[] {0, 1, 2};
            data[1] = new double[] {5, 6, 7, 8, 9};
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(JaggedArrays2Model, data);
            ca.Execute(engine.NumberOfIterations);
            Console.WriteLine("means=" + ca.Marginal("means"));
            Console.WriteLine("precs=" + ca.Marginal("precs"));
        }

        private void JaggedArrays2Model(double[][] data)
        {
            double[] means = new double[data.Length];
            double[] precs = new double[data.Length];
            double[][] x = new double[data.Length][];
            for (int i = 0; i < data.Length; i++)
            {
                x[i] = new double[data[i].Length];
                means[i] = Factor.Random(new Gaussian(0, 100));
                precs[i] = Factor.Random(new Gamma(0.1, 0.1));
                for (int j = 0; j < data[i].Length; j++)
                {
                    x[i][j] = Factor.Gaussian(means[i], precs[i]);
                    Constrain.Equal(data[i][j], x[i][j]);
                }
            }
            InferNet.Infer(means, nameof(means));
            InferNet.Infer(precs, nameof(precs));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void JaggedArrayTest()
        {
            int[] sizes = new int[] {2, 3};
            Gaussian xPrior = new Gaussian(1.2, 3.4);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(JaggedArrayModel, sizes, xPrior);
            ca.Execute(engine.NumberOfIterations);

            DistributionArray<GaussianArray> dist = ca.Marginal<DistributionArray<GaussianArray>>("x");
            Console.WriteLine("x = ");
            Console.WriteLine(dist);
            Gaussian xPost = xPrior*IsPositiveOp.XAverageConditional(true, xPrior);
            for (int i = 0; i < sizes.Length; i++)
            {
                for (int j = 0; j < sizes[i]; j++)
                {
                    Assert.True(dist[i][j].MaxDiff(xPost) < 1e-10);
                }
            }
        }

        private void JaggedArrayModel(int[] sizes, Gaussian xPrior)
        {
            double[][] x = new double[sizes.Length][];
            for (int i = 0; i < sizes.Length; i++)
            {
                x[i] = new double[sizes[i]];
                for (int j = 0; j < sizes[i]; j++)
                {
                    x[i][j] = Factor.Random(xPrior);
                    bool h = Factor.IsPositive(x[i][j]);
                    Constrain.Equal(true, h);
                }
            }
            InferNet.Infer(x, nameof(x));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void JaggedReplicationTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(JaggedReplicationModel);
            ca.Execute(1);
            object marg = ca.Marginal("b");
            Console.WriteLine("Posterior over b=" + marg);
        }

        private void JaggedReplicationModel()
        {
            int[][] sizes = new int[][] {new int[] {1, 2}, new int[] {2, 3, 4}};
            double[] b = new double[sizes.Length];
            double[,] b2 = new double[4,sizes.Length];
            double d = Factor.Gaussian(0, 1);
            for (int i = 0; i < sizes.Length; i++)
            {
                b[i] = Factor.Gaussian(d, 1);
                double[] c = new double[sizes[i].Length];
                for (int j = 0; j < sizes[i].Length; j++)
                {
                    c[j] = Factor.Random(new Gamma(1, 1));
                    double a = Factor.Gaussian(b[i], c[j]);
                    double a2 = Factor.Gaussian(d, 0);
                    for (int k = 0; k < sizes[i][j]; k++)
                    {
                        double z = Factor.Gaussian(0, c[j]);
                    }
                }
                for (int k2 = 0; k2 < 4; k2++)
                {
                    b2[k2, i] = Factor.Gaussian(d, 1);
                    for (int j2 = 0; j2 < sizes[i].Length; j2++)
                    {
                        double a3 = Factor.Gaussian(b2[k2, i], 1);
                        Constrain.Equal(a3, 2.0);
                    }
                }
            }
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(b2, nameof(b2));
            InferNet.Infer(d, nameof(d));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void JaggedReplication2Test()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(JaggedReplication2Model, 1, 1, 1);
            ca.Execute(10);
            DistributionArray<GaussianArray2D> array123Actual = ca.Marginal<DistributionArray<GaussianArray2D>>("array123");
            Console.WriteLine("Posterior over array123=" + array123Actual);
            Assert.True(array123Actual[0][0, 0].MaxDiff(new Gaussian(4, 1.0/2 + 1.0/4)) < 1e-10);
        }

        private void JaggedReplication2Model(int size1, int size2, int size3)
        {
            double[][,] array123 = new double[size1][,];
            double[][] array12 = new double[size1][];
            double[] array3 = new double[size3];
            for (int k1 = 0; k1 < size3; k1++)
            {
                array3[k1] = Factor.Gaussian(1, 2);
            }
            for (int i = 0; i < size1; i++)
            {
                array123[i] = new double[size2,size3];
                array12[i] = new double[size2];
                for (int j = 0; j < size2; j++)
                {
                    array12[i][j] = Factor.Gaussian(3, 4);
                    for (int k = 0; k < size3; k++)
                    {
                        array123[i][j, k] = array12[i][j] + array3[k];
                    }
                }
            }
            InferNet.Infer(array123, nameof(array123));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void JaggedReplication3Test()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(JaggedReplication3Model, 1, 2, 3);
            ca.Execute(10);
            DistributionArray<GaussianArray2D> array123Actual = ca.Marginal<DistributionArray<GaussianArray2D>>("array123");
            Console.WriteLine("Posterior over array123=");
            Console.WriteLine(array123Actual);
            Assert.True(array123Actual[0][0, 0].MaxDiff(new Gaussian(4, 1.0/2 + 1.0/4)) < 1e-10);
        }

        private void JaggedReplication3Model(int size1, int size2, int size3)
        {
            double[][,] array123 = new double[size1][,];
            double[][] array31 = new double[size3][];
            double[,] matrix31 = new double[size3,size1];
            for (int k1 = 0; k1 < size3; k1++)
            {
                array31[k1] = new double[size1];
                for (int i1 = 0; i1 < size1; i1++)
                {
                    array31[k1][i1] = Factor.Gaussian(1, 2);
                    matrix31[k1, i1] = Factor.Gaussian(3, 4);
                }
            }
            for (int i = 0; i < size1; i++)
            {
                array123[i] = new double[size2,size3];
                for (int j = 0; j < size2; j++)
                {
                    for (int k = 0; k < size3; k++)
                    {
                        array123[i][j, k] = array31[k][i] + matrix31[k, i];
                    }
                }
            }
            InferNet.Infer(array123, nameof(array123));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void JaggedReplication4Test()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(JaggedReplication4Model, 1, 2, 3, 4);
            ca.Execute(10);
            DistributionArray2D<GaussianArray2D> array1234Actual = ca.Marginal<DistributionArray2D<GaussianArray2D>>("array1234");
            Console.WriteLine("Posterior over array1234=" + array1234Actual);
            Assert.True(array1234Actual[0, 0][0, 0].MaxDiff(new Gaussian(4, 1.0/2 + 1.0/4)) < 1e-10);
        }

        private void JaggedReplication4Model(int size1, int size2, int size3, int size4)
        {
            double[,][,] array1234 = new double[size1,size2][,]; //,size3,size4];
            double[][] array41 = new double[size4][];
            double[,] matrix41 = new double[size4,size1];
            for (int l1 = 0; l1 < size4; l1++)
            {
                array41[l1] = new double[size1];
                for (int i1 = 0; i1 < size1; i1++)
                {
                    array41[l1][i1] = Factor.Gaussian(1, 2);
                    matrix41[l1, i1] = Factor.Gaussian(3, 4);
                }
            }
            for (int i = 0; i < size1; i++)
            {
                for (int j = 0; j < size2; j++)
                {
                    array1234[i, j] = new double[size3,size4];
                    for (int k = 0; k < size3; k++)
                    {
                        for (int l = 0; l < size4; l++)
                        {
                            array1234[i, j][k, l] = array41[l][i] + matrix41[l, i];
                        }
                    }
                }
            }
            InferNet.Infer(array1234, nameof(array1234));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void JaggedLiteralIndexingTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(JaggedLiteralIndexingModel);
            ca.Execute(engine.NumberOfIterations);
            DistributionArray<GaussianArray> dist = ca.Marginal<DistributionArray<GaussianArray>>("b2");
            Console.WriteLine("b2 = ");
            Console.WriteLine(dist);
            Gaussian xPrior = new Gaussian(0, 1);
            Gaussian xPost = xPrior*IsPositiveOp.XAverageConditional(true, xPrior);
            for (int i = 0; i < 2; i++)
            {
                Assert.True(dist[0][i].MaxDiff(xPost) < 1e-10);
            }
            for (int i = 0; i < 3; i++)
            {
                Assert.True(dist[1][i].MaxDiff(xPrior) < 1e-10);
            }
        }

        private void JaggedLiteralIndexingModel()
        {
            int[] sizes = new int[2] {2, 3};
            double[][] b2 = new double[2][];
            for (int i = 0; i < sizes.Length; i++)
            {
                b2[i] = new double[sizes[i]];
                for (int j = 0; j < sizes[i]; j++)
                {
                    b2[i][j] = Factor.Random(new Gaussian(0, 1));
                }
            }
            for (int j2 = 0; j2 < sizes[0]; j2++)
            {
                bool h = Factor.IsPositive(b2[0][j2]);
                Constrain.Equal(true, h);
            }
            InferNet.Infer(b2, nameof(b2));
        }

        private void JaggedLiteralIndexing2Model()
        {
            int[][] sizes = new int[][] {new int[] {1, 2}, new int[] {2, 3, 4}};
            double[] b = new double[2];
            double d = Factor.Gaussian(0, 1);
            double[][] b2 = new double[2][];
            for (int i = 0; i < sizes.Length; i++)
            {
                b[i] = Factor.Gaussian(d, 1);
                double[] c = new double[sizes[i].Length];
                b2[i] = new double[sizes[i].Length];
                for (int j = 0; j < sizes[i].Length; j++)
                {
                    c[j] = Factor.Random(new Gamma(1, 1));
                    b2[i][j] = Factor.Random(new Gaussian(0, 1));
                    double a = Factor.Gaussian(b[i], c[j]);
                    double a2 = Factor.Gaussian(d, 0);
                    for (int k = 0; k < sizes[i][j]; k++)
                    {
                        double z = Factor.Gaussian(0, c[j]);
                    }
                }
            }
            for (int j2 = 0; j2 < sizes[0].Length; j2++)
            {
                double f = Factor.Gaussian(b2[0][j2], 1.0);
            }
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(b2, nameof(b2));
            InferNet.Infer(d, nameof(d));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void ConstrainEqualArrayTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(ConstrainEqualArrayModel, new double[3], Util.ArrayInit(6, i => Vector.Zero(3)), new bool[6]);
            ca.Execute(engine.NumberOfIterations);
        }

        private void ConstrainEqualArrayModel(double[] vdouble__0, Vector[] vVector__0, bool[] vbool__0)
        {
            Vector vVector0 = Vector.FromArray(vdouble__0);
            InferNet.Infer(vVector0, "vVector0", QueryTypes.Marginal);
            PositiveDefiniteMatrix vPositiveDefiniteMatrix0 = new PositiveDefiniteMatrix(new double[3,3] {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
            Vector weights = VectorGaussian.SampleFromMeanAndVariance(vVector0, vPositiveDefiniteMatrix0);
            InferNet.Infer(weights, "weights", QueryTypes.Marginal);
            double[] GaussianFromMeanAndVarianceSample = new double[6];
            for (int index2 = 0; index2 < 6; index2++)
            {
                double vdouble2 = Vector.InnerProduct(vVector__0[index2], weights);
                GaussianFromMeanAndVarianceSample[index2] = Factor.GaussianFromMeanAndVariance(vdouble2, 0.1);
            }
            InferNet.Infer(GaussianFromMeanAndVarianceSample, "GaussianFromMeanAndVarianceSample", QueryTypes.Marginal);
            bool[] vbool__1 = new bool[6];
            for (int index5 = 0; index5 < 6; index5++)
            {
                vbool__1[index5] = Factor.IsPositive(GaussianFromMeanAndVarianceSample[index5]);
            }
            InferNet.Infer(vbool__1, "vbool__1", QueryTypes.Marginal);
            Constrain.Equal<bool[]>(vbool__0, vbool__1);
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void ConstrainEqualVectorArrayTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            engine.Algorithm = new VariationalMessagePassing();
            var ca = engine.Compiler.Compile(ConstrainEqualVectorArrayModel, new double[2], Util.ArrayInit(4, i => Vector.Zero(2)));
            ca.Execute(engine.NumberOfIterations);
        }

        private void ConstrainEqualVectorArrayModel(double[] vdouble__1, Vector[] vVector__1)
        {
            Vector var0 = Vector.FromArray(vdouble__1);
            InferNet.Infer(var0, "var0", QueryTypes.Marginal);
            PositiveDefiniteMatrix var1 = new PositiveDefiniteMatrix(new double[2,2] {{0.01, 0}, {0, 0.01}});
            Vector[] means0 = new Vector[2];
            for (int index4 = 0; index4 < 2; index4++)
            {
                means0[index4] = Factor.VectorGaussian(var0, var1);
            }
            InferNet.Infer(means0, "means0", QueryTypes.Marginal);
            Dirichlet vDirichlet0 = new Dirichlet(DenseVector.FromArray(new double[2] {1, 1}));
            Vector weights0 = Factor.Random<Vector>(vDirichlet0);
            InferNet.Infer(weights0, "weights0", QueryTypes.Marginal);
            int[] z = new int[4];
            InferNet.Infer(z, "z", QueryTypes.Marginal);
            Vector[] vVector__2 = new Vector[4];
            PositiveDefiniteMatrix[] precs0 = new PositiveDefiniteMatrix[2];
            for (int index2 = 0; index2 < 2; index2++)
            {
                precs0[index2] = Wishart.SampleFromShapeAndScale(100, var1);
            }
            InferNet.Infer(precs0, "precs0", QueryTypes.Marginal);
            for (int index5 = 0; index5 < 4; index5++)
            {
                z[index5] = Factor.Discrete(weights0);
                for (int index1 = 0; index1 < 2; index1++)
                {
                    if (z[index5] == index1)
                    {
                        vVector__2[index5] = Factor.VectorGaussian(means0[z[index5]], precs0[z[index5]]);
                    }
                }
            }
            InferNet.Infer(vVector__2, "vVector__2", QueryTypes.Marginal);
            Constrain.Equal<Vector[]>(vVector__1, vVector__2);
        }
    }
}
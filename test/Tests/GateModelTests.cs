// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;
using Xunit;

namespace Microsoft.ML.Probabilistic.Tests
{
#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
    using BernoulliArray = DistributionStructArray<Bernoulli, bool>;
    using BernoulliArrayArray = DistributionRefArray<DistributionStructArray<Bernoulli, bool>, bool[]>;
    using Range = Microsoft.ML.Probabilistic.Models.Range;

    public class GateModelTests
    {
        /// <summary>
        /// Tests that GateAnalysisTransform does not introduce a performance bug.
        /// The bug can only be seen in the generated source.
        /// </summary>
        [Fact]
        public void GateAnalysisTest()
        {
            Range item = new Range(2);
            var array = Variable.Array<double>(item);
            Range otherItem = item.Clone();
            using (var block = Variable.ForEach(item))
            {
                array[item] = Variable.GaussianFromMeanAndVariance(0, 1);
                using (var otherBlock = Variable.ForEach(otherItem))
                {
                    using (Variable.If(block.Index > otherBlock.Index))
                    {
                        Variable.ConstrainTrue(array[item] > array[otherItem]);
                    }
                }
            }

            InferenceEngine engine = new InferenceEngine();
            engine.Infer(array);
        }

        [Fact]
        public void ForEachIfObservedConstrainTrueTest()
        {
            Range item = new Range(2).Named("item");
            VariableArray<bool> array1 = Variable.Observed(new[] { false, true }, item).Named("array1");
            VariableArray<bool> array2 = Variable.Observed(new[] { false, true }, item).Named("array2");
            VariableArray<bool> array3 = Variable.Observed(new[] { false, true }, item).Named("array3");
            var b = Variable.Bernoulli(0.5).Named("b");
            using (Variable.ForEach(item))
            {
                using (Variable.If(array1[item]))
                {
                    using (Variable.If(array2[item]))
                    {
                        using (Variable.If(array3[item]))
                        {
                            Variable.ConstrainTrue(b);
                        }
                    }
                }
            }

            InferenceEngine engine = new InferenceEngine();
            Bernoulli bActual = engine.Infer<Bernoulli>(b);
            Bernoulli bExpected = new Bernoulli(1.0);
            Assert.True(bExpected.MaxDiff(bActual) < 1e-10);
        }

        // TODO: This model generates inefficient code, since the generated array x_i__F has the same value for all i
        internal void ForEachGateExitTest()
        {
            Range item = new Range(2).Named("i");
            VariableArray<bool> b = Variable.Array<bool>(item).Named("b");
            VariableArray<bool> x = Variable.Array<bool>(item).Named("x");
            using (Variable.ForEach(item))
            {
                b[item] = Variable.Bernoulli(0.5);
                using (Variable.If(b[item]))
                {
                    x[item] = Variable.Bernoulli(0.1);
                }
                using (Variable.IfNot(b[item]))
                {
                    x[item] = Variable.Bernoulli(0.2);
                }
            }
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(x));
        }

        // test CasesOp when one branch is uniform
        [Fact]
        public void UniformBranchTest()
        {
            double p = 0.4;
            double cPrior = 0.1;
            Variable<bool> c = Variable.Bernoulli(cPrior).Named("c");
            using (Variable.If(c))
            {
                Variable.ConstrainTrue(Variable.Bernoulli(p));
            }
            using (Variable.IfNot(c))
            {
                var x = Variable.Bernoulli(p);
            }
            InferenceEngine engine = new InferenceEngine();
            Bernoulli cActual = engine.Infer<Bernoulli>(c);
            double sumCondT = p;
            double sumCondF = 1;
            double Z = cPrior * sumCondT + (1 - cPrior) * sumCondF;
            double cPost = cPrior * sumCondT / Z;
            Bernoulli cExpected = new Bernoulli(cPost);
            Console.WriteLine("c = {0} (should be {1})", cActual, cExpected);
            Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
        }

        [Fact]
        public void LoopIndexGateTest()
        {
            Range r = new Range(3).Named("r");
            var array = Variable.Array<double>(r).Named("array");
            var index = Variable.Observed<int>(1).Named("index");
            using (var block = Variable.ForEach(r))
            {
                var isIndex = (block.Index == index);

                using (Variable.If(isIndex))
                {
                    array[r] =
                        Variable.GaussianFromMeanAndVariance(0.0, 2.0);
                }

                using (Variable.IfNot(isIndex))
                {
                    array[r] =
                        Variable.GaussianFromMeanAndVariance(0.0, 1.0);
                }
            }
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(array));
        }

        /// <summary>
        /// Tests argument asymmetry in the condition of a gate.
        /// </summary>
        [Fact]
        public void FlipGateConditionTest()
        {
            var sample = Variable.Random(new Gaussian(0, 1)).Named("sample");

            var n = new Range(1).Named("n");
            var index = Variable.Array<int>(n).Named("index");
            index.ObservedValue = new[] { 0 };

            using (Variable.ForEach(n))
            {
#if true
                // Fails!
                using (Variable.If(0 == index[n]))
#else
                // Works!
                using (Variable.If(index[n] == 0)) 
#endif
                {
                    Variable.ConstrainPositive(sample);
                }
            }

            var engine = new InferenceEngine();
            engine.Infer(sample);
        }

        [Fact]
        public void ReplicateWithConditionedIndexTest()
        {
            var count1 = Variable.New<int>();
            count1.Name = nameof(count1);
            var count2 = Variable.New<int>();
            count2.Name = nameof(count2);
            var range1 = new Range(count1);
            range1.Name = nameof(range1);
            var range2 = new Range(count2);
            range2.Name = nameof(range2);

            var matrix = Variable.Array(Variable.Array<double>(range2), range1);
            matrix.Name = nameof(matrix);
            matrix[range1][range2] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(range2).ForEach(range1);

            var evidence = Variable.Bernoulli(0.5);
            evidence.Name = nameof(evidence);

            using (Variable.If(evidence))
            {
                using (var b1 = Variable.ForEach(range1))
                {
                    using (Variable.If(b1.Index < count1 & b1.Index < count2))
                    {
                        using (var b2 = Variable.ForEach(range2))
                        {
                            using (Variable.If(b2.Index < count1 & b2.Index < count2))
                            {
                                var diagonalElement = (b2.Index == b1.Index);
                                diagonalElement.Name = nameof(diagonalElement);

                                using (Variable.If(diagonalElement))
                                {
                                    Variable.ConstrainEqual(matrix[range1][range2], 1);
                                }

                                using (Variable.IfNot(diagonalElement))
                                {
                                    Variable.ConstrainEqual(matrix[range1][range2], 0);
                                }
                            }
                        }
                    }
                }
            }

            count1.ObservedValue = 4;
            count2.ObservedValue = 20;

            Console.WriteLine(new InferenceEngine().Infer(matrix));
        }

        [Fact]
        public void CaseLoopIndexTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Range rows = new Range(3).Named("i");
            VariableArray<double> states = Variable.Array<double>(rows).Named("states");
            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.Case(rowBlock.Index, 0))
                {
                    //using (Variable.If(rowBlock.Index == 0))
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(0, 1000);
                }
                using (Variable.IfNot(rowBlock.Index == 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                }
            }
            block.CloseBlock();
            InferenceEngine engine = new InferenceEngine();
            var statesActual = engine.Infer<IList<Gaussian>>(states);
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Gaussian state2Expected = new Gaussian(1, 1);
            Assert.True(state2Expected.MaxDiff(statesActual[2]) < 1e-10);
        }

        [Fact]
        public void CaseLoopIndexTest2()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Range rows = new Range(3).Named("i");
            VariableArray<double> states = Variable.Array<double>(rows).Named("states");
            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.Case(rowBlock.Index, 0))
                {
                    //using (Variable.If(rowBlock.Index == 0)) {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(0, 1000);
                }
                using (Variable.If(rowBlock.Index > 0))
                {
                    Variable<int> rowMinusOne = rowBlock.Index - 1;
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowMinusOne], 1);
                }
            }
            block.CloseBlock();
            InferenceEngine engine = new InferenceEngine();
            var statesActual = engine.Infer<IList<Gaussian>>(states);
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Assert.True(MMath.AbsDiff(0, evActual) < 1e-10);
            Gaussian state2Expected = new Gaussian(0, 1002);
            Assert.True(state2Expected.MaxDiff(statesActual[2]) < 1e-10);
        }

        [Fact]
        public void CaseLoopIndexTest3()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Range rows = new Range(4).Named("i");
            VariableArray<double> states = Variable.Array<double>(rows).Named("states");
            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.Case(rowBlock.Index, 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(0, 1000);
                }
                using (Variable.Case(rowBlock.Index, 1))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1000);
                }
                using (Variable.If(rowBlock.Index > 1))
                {
                    Variable<int> rowMinusOne = rowBlock.Index - 1;
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowMinusOne], 1);
                }
            }
            block.CloseBlock();
            InferenceEngine engine = new InferenceEngine();
            var statesActual = engine.Infer<IList<Gaussian>>(states);
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Assert.True(MMath.AbsDiff(0, evActual) < 1e-10);
            Gaussian state3Expected = new Gaussian(1, 1002);
            Assert.True(state3Expected.MaxDiff(statesActual[3]) < 1e-10);
        }

        // Fails with error: Cannot a define a variable more than once in the same condition context
        //[Fact]
        internal void CaseLoopIndexBroken()
        {
            Range rows = new Range(3).Named("i");
            VariableArray<double> states = Variable.Array<double>(rows).Named("states");
            states[0] = Variable.GaussianFromMeanAndVariance(0, 1000);
            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index > 0))
                {
                    Variable<int> rowMinusOne = rowBlock.Index - 1;
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowMinusOne], 1);
                }
            }
            InferenceEngine engine = new InferenceEngine();
            var result = engine.Infer(states);
            Console.WriteLine("Result: {0}", result);
        }

        [Fact]
        public void MixtureLogisticRegression()
        {
            double[] xdata = new double[] { 3, 4, 2, 2 };
            bool[] ydata = new bool[] { true, true, false, false };

            int numN = xdata.Length;
            int numK = 2;
            Range n = new Range(numN).Named("n");
            Range k = new Range(numK).Named("k");

            var p = Variable.DirichletUniform(k).Named("p");
            var c = Variable.Array<int>(n).Named("c");
            var b = Variable.Array<double>(k).Named("b");
            var x = Variable.Observed<double>(xdata, n).Named("x");
            var y = Variable.Observed<bool>(ydata, n).Named("y");

            b[k] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(k);

            using (Variable.ForEach(n))
            {
                c[n] = Variable.Discrete(p);
                using (Variable.Switch(c[n]))
                {
                    var dot = x[n] * b[c[n]];
                    dot.Name = "dot";
                    var dotNoisy = Variable.GaussianFromMeanAndPrecision(dot, 100).Named("dotNoisy");
                    var logist = Variable.Logistic(dot).Named("logist");
                    y[n] = Variable.Bernoulli(logist);
                }
            }
            c.InitialiseTo(Distribution<int>.Array(Util.ArrayInit(numN, i => Discrete.PointMass(Rand.Int(numK), numK))));
            //b.InitialiseTo(Distribution<double>.Array(Util.ArrayInit(numK, i => new Gaussian(i, 1))));
            var engine = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(StringUtil.JoinColumns("b = ", engine.Infer(b)));
            Console.WriteLine(StringUtil.JoinColumns("c = ", engine.Infer(c)));
        }

        [Fact]
        public void MurphySprinklerTest()
        {
            // cloudy 
            Variable<bool> cloudy = Variable.Bernoulli(0.5);

            // sprinkler
            Variable<bool> sprinkler = Variable.New<bool>();
            using (Variable.If(cloudy)) sprinkler.SetTo(Variable.Bernoulli(0.1));
            using (Variable.IfNot(cloudy)) sprinkler.SetTo(Variable.Bernoulli(0.5));
            // rain
            Variable<bool> rain = Variable.New<bool>();
            using (Variable.If(cloudy)) rain.SetTo(Variable.Bernoulli(0.8));
            using (Variable.IfNot(cloudy)) rain.SetTo(Variable.Bernoulli(0.2));
            // wet grass
            Variable<bool> wetGrass = Variable.New<bool>();
            using (Variable.If(sprinkler))
            {
                using (Variable.If(rain)) wetGrass.SetTo(Variable.Bernoulli(0.99));
                using (Variable.IfNot(rain)) wetGrass.SetTo(Variable.Bernoulli(0.9));
            }
            using (Variable.IfNot(sprinkler))
            {
                using (Variable.If(rain)) wetGrass.SetTo(Variable.Bernoulli(0.9));
                using (Variable.IfNot(rain)) wetGrass.SetTo(Variable.Bernoulli(0.0));
            }
            // Observations and inference
            InferenceEngine ie = new InferenceEngine();
            ie.ShowProgress = false;

            wetGrass.ObservedValue = true;
            Assert.Equal(0.7841, ie.Infer<Bernoulli>(rain).GetProbTrue(), 1e-4);
            Assert.Equal(0.404, ie.Infer<Bernoulli>(sprinkler).GetProbTrue(), 1e-4);
            cloudy.ObservedValue = false;
            Assert.Equal(0.3443, ie.Infer<Bernoulli>(rain).GetProbTrue(), 1e-4);
            Assert.Equal(0.8361, ie.Infer<Bernoulli>(sprinkler).GetProbTrue(), 1e-4);
        }

        [Fact]
        public void IfObservedArrayExitTest()
        {
            Range n = new Range(2).Named("n");
            VariableArray<bool> c = Variable.Array<bool>(n).Named("c");
            //c[n] = Variable.Bernoulli(0.1).ForEach(n);
            c.ObservedValue = new bool[] { true, false };
            VariableArray<bool> x = Variable.Array<bool>(n).Named("x");
            VariableArray<bool> y = Variable.Array<bool>(n).Named("y");
            using (Variable.ForEach(n))
            {
                using (Variable.If(c[n]))
                {
                    x[n] = Variable.Bernoulli(0.2);
                }
                using (Variable.IfNot(c[n]))
                {
                    x[n] = Variable.Bernoulli(0.3);
                }
                y[n] = Variable.Copy(x[n]);
                Variable.ConstrainEqualRandom(y[n], new Bernoulli(0.4));
                Variable.ConstrainEqualRandom(x[n], new Bernoulli(0.7));
            }
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(x));
        }

        // Requires schedule to be sequential
        [Fact]
        public void VocabularyTest()
        {
            Range V = new Range(10).Named("v"); // vocabulary size
            Range K = new Range(2).Named("k"); // number of latent words

            var latentWords = Variable.Array<int>(K).Named("latentWords");
            latentWords[K] = Variable.DiscreteUniform(V).ForEach(K);

            int[] wordsData = new int[] { 0, 0, 0, 1 };
            var Words = Variable.Observed(wordsData).Named("words");
            Words.Range.Named("i");
            Words.Range.AddAttribute(new Sequential());

            using (Variable.ForEach(Words.Range))
            {
                var sel = Variable.DiscreteUniform(K).Named("selector");
                using (Variable.Switch(sel))
                {
                    Words[Words.Range] = Variable.Copy(latentWords[sel]);
                }
            }
            //latentWords.InitialiseTo(Distribution<int>.Array(new Discrete[] { Discrete.PointMass(0,10), Discrete.PointMass(1,10) }));

            var engine = new InferenceEngine();
            var dists = engine.Infer(latentWords);

            Console.WriteLine(dists);
        }

        [Fact]
        public void GateExitPointMassTest()
        {
            GateExitPointMass(true);
            GateExitPointMass(false);
        }

        private void GateExitPointMass(bool obs)
        {
            Variable<bool> b = Variable.Bernoulli(0.1).Named("b");
            Variable<bool> x = Variable.New<bool>().Named("x");
            using (Variable.If(b))
            {
                x.SetTo(Variable.Random(Bernoulli.PointMass(true)));
            }
            using (Variable.IfNot(b))
            {
                x.SetTo(Variable.Random(Bernoulli.PointMass(false)));
            }
            Variable.ConstrainEqualRandom(x, Bernoulli.PointMass(obs));
            InferenceEngine engine = new InferenceEngine();
            Bernoulli bActual = engine.Infer<Bernoulli>(b);
            Console.WriteLine(bActual);
            if (obs)
                Assert.True(double.IsPositiveInfinity(bActual.LogOdds));
            else
                Assert.True(double.IsNegativeInfinity(bActual.LogOdds));
        }

        [Fact]
        public void GateExitPointMassTest2()
        {
            Range n = new Range(2);
            var topics = Variable.Array<int>(n);
            var theta = new double[] { 0.6, 0.4 };
            VariableArray<double> oneDoubleIndices = Variable.Array<double>(n);
            using (Variable.ForEach(n))
            {
                topics[n] = Variable.Discrete(theta);
                Variable<bool> isOne = topics[n] == 1;
                using (Variable.If(isOne))
                {
                    oneDoubleIndices[n] = 1.0;
                }
                using (Variable.IfNot(isOne))
                {
                    oneDoubleIndices[n] = 0.0;
                }
            }
            //oneDoubleIndices.AddAttribute(new MarginalPrototype(new Gaussian()));
            Variable<double> oneCount = Variable.Sum(oneDoubleIndices);

            InferenceEngine engine = new InferenceEngine();
            Gaussian actual = engine.Infer<Gaussian>(oneCount);
            Gaussian expected = new Gaussian(0.8, 0.48);
            Assert.True(expected.MaxDiff(actual) < 1e-4);
        }

        [Fact]
        public void GatedPointMassTest()
        {
            // bPrior is chosen so that the product of 2 of them leads to a point mass
            Bernoulli bPrior = Bernoulli.FromLogOdds(400);
            int nSubjects = 2;
            int nChoices = 2;

            Range subject = new Range(nSubjects).Named("subject");
            Range choice = new Range(nChoices).Named("choice");

            var x = Variable.DiscreteUniform(nChoices);

            using (Variable.ForEach(subject))
            {
                var b = Variable.Random(bPrior);
                using (Variable.If(b))
                    Variable.ConstrainEqual(1, x);
            }

            double logProbX0 = nSubjects * bPrior.GetLogProbFalse();
            double logProbX1 = 0.0;
            double Z = MMath.LogSumExp(logProbX0, logProbX1);

            InferenceEngine engine = new InferenceEngine();
            engine.NumberOfIterations = 10;
            //subject.AddAttribute(new Sequential());
            //engine.Compiler.UseSerialSchedules = true;
            //engine.Algorithm = new VariationalMessagePassing();
            var xPost = engine.Infer<Discrete>(x);
            Discrete xExpected = new Discrete(System.Math.Exp(logProbX0 - Z), System.Math.Exp(logProbX1 - Z));
            Console.WriteLine("x = {0} should be {1}", xPost, xExpected);
            Assert.True(xExpected.MaxDiff(xPost) < 1e-10);
        }

        [Fact]
        public void GateEnterPointMassTest()
        {
            Variable<bool> b = Variable.Bernoulli(0.1).Named("b");
            Variable<Bernoulli> xPrior = Variable.New<Bernoulli>();
            Variable<bool> x = Variable<bool>.Random(xPrior).Named("x");
            using (Variable.If(b))
            {
                Variable.ConstrainEqualRandom(x, Bernoulli.PointMass(true));
            }
            using (Variable.IfNot(b))
            {
                Variable.ConstrainEqualRandom(x, Bernoulli.PointMass(false));
            }
            InferenceEngine engine = new InferenceEngine();
            for (int trial = 0; trial < 2; trial++)
            {
                xPrior.ObservedValue = Bernoulli.PointMass(trial == 0);
                Bernoulli bActual = engine.Infer<Bernoulli>(b);
                Console.WriteLine(bActual);
                if (trial == 0)
                {
                    Assert.True(double.IsPositiveInfinity(bActual.LogOdds));
                }
                else
                {
                    Assert.True(double.IsNegativeInfinity(bActual.LogOdds));
                }
            }
        }

        [Fact]
        public void GateEnterPointMassTest2()
        {
            Range i = new Range(2);
            Variable<int> b = Variable.Discrete(i, 0.1, 0.9).Named("b");
            Variable<Bernoulli> xPrior = Variable.New<Bernoulli>();
            Variable<bool> x = Variable<bool>.Random(xPrior).Named("x");
            VariableArray<Bernoulli> dists = Variable.Constant(new Bernoulli[] { Bernoulli.PointMass(true), Bernoulli.PointMass(false) }, i).Named("dists");
            using (Variable.Switch(b))
            {
                Variable.ConstrainEqualRandom(x, dists[b]);
            }
            InferenceEngine engine = new InferenceEngine();
            for (int trial = 0; trial < 2; trial++)
            {
                xPrior.ObservedValue = Bernoulli.PointMass(trial == 0);
                Discrete bActual = engine.Infer<Discrete>(b);
                Console.WriteLine(bActual);
                if (trial == 0)
                {
                    Assert.True(bActual[0] == 1.0);
                }
                else
                {
                    Assert.True(bActual[1] == 1.0);
                }
            }
        }

        [Fact]
        public void SwitchPointMassGaussianTest()
        {
            double bSwitch = SwitchPointMassGaussian(true);
            double bNoSwitch = SwitchPointMassGaussian(false);
            Console.WriteLine($"switch = {bSwitch} expected = {bNoSwitch}");
            Assert.Equal(bSwitch, bNoSwitch, 1e-4);
        }

        internal double SwitchPointMassGaussian(bool useSwitch)
        {
            Gaussian productLike = new Gaussian(4, 5);
            double Amean = 1;
            double Avariance = 3;
            Variable<double> B = Variable.GaussianFromMeanAndVariance(0, 1000);
            B.AddAttribute(new PointEstimate());
            B.InitialiseTo(Gaussian.PointMass(0));
            if (useSwitch)
            {
                double[] nodes = new double[5];
                double[] weights = new double[5];
                Quadrature.GaussianNodesAndWeights(0, 1, nodes, weights);
                Range item = new Range(weights.Length);
                Variable<int> selector = Variable.Discrete(item, weights);
                VariableArray<double> nodesArray = Variable.Constant(nodes, item);
                using (Variable.Switch(selector))
                {
                    Variable<double> Atmp = Variable.GaussianFromMeanAndVariance(nodesArray[selector] + Amean, Avariance - 1);
                    var product = Atmp * B;
                    Variable.ConstrainEqualRandom(product, productLike);
                }
            }
            else
            {
                Variable<double> A = Variable.GaussianFromMeanAndVariance(Amean, Avariance);
                var product = A * B;
                Variable.ConstrainEqualRandom(product, productLike);
            }
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.InitialisationAffectsSchedule = true;
            return engine.Infer<Gaussian>(B).Point;
        }

        [Fact]
        public void SwitchPointMassGammaTest()
        {
            double bSwitch = SwitchPointMassGamma(true);
            double bNoSwitch = SwitchPointMassGamma(false);
            Console.WriteLine($"switch = {bSwitch} expected = {bNoSwitch}");
            Assert.Equal(bSwitch, bNoSwitch, 1e-4);
        }

        internal double SwitchPointMassGamma(bool useSwitch)
        {
            Gaussian productLike = new Gaussian(4, 0.5);
            double Amean = 1;
            double Avariance = 3;
            Variable<double> B = Variable.GammaFromShapeAndRate(1, 1);
            B.AddAttribute(new PointEstimate());
            B.InitialiseTo(Gamma.PointMass(1));
            if (useSwitch)
            {
                double[] nodes = new double[5];
                double[] weights = new double[5];
                Quadrature.GaussianNodesAndWeights(0, 1, nodes, weights);
                Range item = new Range(weights.Length);
                Variable<int> selector = Variable.Discrete(item, weights);
                VariableArray<double> nodesArray = Variable.Constant(nodes, item);
                using (Variable.Switch(selector))
                {
                    Variable<double> Atmp = Variable.GaussianFromMeanAndVariance(nodesArray[selector] + Amean, Avariance - 1);
                    var product = Variable.GaussianFromMeanAndPrecision(Atmp, B);
                    Variable.ConstrainEqualRandom(product, productLike);
                }
            }
            else
            {
                Variable<double> A = Variable.GaussianFromMeanAndVariance(Amean, Avariance);
                var product = Variable.GaussianFromMeanAndPrecision(A, B);
                Variable.ConstrainEqualRandom(product, productLike);
            }
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.InitialisationAffectsSchedule = true;
            return engine.Infer<Gamma>(B).Point;
        }

        [Fact]
        public void MultiplierOffsetTest()
        {
            double noSwitch = MultiplierOffset(false);
            double withSwitch = MultiplierOffset(true);
            Assert.Equal(noSwitch, withSwitch, 1e-4);
        }

        /// <summary>
        /// This model converges extremely slowly with RpropBufferData.EnsureConvergence=true and useSwitch=false.
        /// </summary>
        /// <param name="useSwitch"></param>
        internal double MultiplierOffset(bool useSwitch)
        {
            Variable<double> offset = Variable.GaussianFromMeanAndVariance(0, 1000);
            offset.Name = nameof(offset);
            offset.AddAttribute(new PointEstimate());
            offset.InitialiseTo(Gaussian.PointMass(0));
            //offset.ObservedValue = 0;
            Variable<double> multiplier = Variable.GaussianFromMeanAndVariance(1, 1);
            multiplier.Name = nameof(multiplier);
            multiplier.AddAttribute(new PointEstimate());
            multiplier.InitialiseTo(Gaussian.PointMass(1));
            //multiplier.ObservedValue = 0.5;
            Variable<double> precision = Variable.GammaFromShapeAndRate(1, 1);
            precision.Name = nameof(precision);
            precision.AddAttribute(new PointEstimate());
            precision.InitialiseTo(Gamma.PointMass(1));
            Range item = new Range(2);
            VariableArray<double> performance = Variable.Array<double>(item);
            performance.Name = nameof(performance);
            VariableArray<double> data = Variable.Array<double>(item);
            data.Name = nameof(data);
            using (Variable.ForEach(item))
            {
                performance[item] = Variable.GaussianFromMeanAndVariance(1, 0.25);
                int bucketCount = 5;
                Range bucket = new Range(bucketCount);
                bucket.Name = nameof(bucket);
                var performanceBucket = Variable.DiscreteUniform(bucket);
                performanceBucket.Name = nameof(performanceBucket);
                var lowerBounds = Variable.Constant(new[] { double.NegativeInfinity, 0.2, 0.4, 0.6, 0.8 }, bucket);
                var upperBounds = Variable.Constant(new[] { 0.2, 0.4, 0.6, 0.8, double.PositiveInfinity }, bucket);
                var mean = offset;
                var product = performance[item] * multiplier;
                product.Name = "scaledPerformance";
                SwitchBlock block = null;
                if (useSwitch)
                {
                    block = Variable.Switch(performanceBucket);
                    Variable.ConstrainBetween(product, lowerBounds[performanceBucket], upperBounds[performanceBucket]);
                }
                mean += product;
                if (useSwitch)
                {
                    //Variable.ConstrainBetween(mean, lowerBounds[performanceBucket], upperBounds[performanceBucket]);
                }
                mean.Name = "EventCountMean";
                data[item] = Variable.GaussianFromMeanAndPrecision(mean, precision);
                if (block != null)
                    block.CloseBlock();
            }
            data.ObservedValue = Util.ArrayInit(item.SizeAsInt, i => (double)i/item.SizeAsInt);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.InitialisationAffectsSchedule = true;
            engine.Compiler.GivePriorityTo(typeof(GaussianProductOp_PointB));
            engine.Compiler.GivePriorityTo(typeof(GaussianOp_PointPrecision));
            engine.ShowProgress = false;
            //engine.NumberOfIterations = 100;
            var offsetActual = engine.Infer<Gaussian>(offset).Point;
            var multiplierActual = engine.Infer<Gaussian>(multiplier).Point;
            var precisionActual = engine.Infer<Gamma>(precision).Point;
            //Console.WriteLine($"offset = {offsetActual}");
            //Console.WriteLine($"multiplier = {multiplierActual}");
            //Console.WriteLine($"precision = {precisionActual}");
            return precisionActual;
        }

        // This test requires DependencyAnalysisTransform.KeepFresh=true (when useMinCut=false)
        // otherwise GateEnterPartialOp<bool>.ValueAverageConditional will multiply two point masses
        // y_item1_use_uses_F[1] and x_item1_use_uses_F[1]
        [Fact]
        [Trait("Category", "ModifiesGlobals")]
        public void GateEnterOnePointMassTest3()
        {
            using (TestUtils.TemporarilyUnrollLoops)
            {
                GateEnterOnePointMassTest2();
            }
        }

        [Fact]
        public void GateEnterOnePointMassTest2()
        {
            Range i = new Range(2).Named("i");
            Range j = new Range(2).Named("j");
            //Variable<int> b = Variable.Discrete(i, 0.1, 0.9).Named("b");
            VariableArray<int> b = Variable.Array<int>(i).Named("b");
            b[i] = Variable.Discrete(j, 0.1, 0.9).ForEach(i);
            Variable<Bernoulli> xPrior = Variable.New<Bernoulli>();
            //Variable<bool> x = Variable<bool>.Random(xPrior).Named("x");
            VariableArray<Bernoulli> dists = Variable.Constant(new Bernoulli[] { Bernoulli.PointMass(true), Bernoulli.PointMass(false) }, i).Named("dists");
            VariableArray<bool> x = Variable.Array<bool>(i).Named("x");
            x[i] = Variable<bool>.Random(dists[i]);
            VariableArray<bool> y = Variable.Array<bool>(j).Named("y");
            VariableArray<Bernoulli> dists2 = Variable.Constant(new Bernoulli[] { Bernoulli.PointMass(true), Bernoulli.PointMass(false) }, j).Named("dists2");
            y[j] = Variable<bool>.Random(dists2[j]);
            using (Variable.ForEach(i))
            {
                using (Variable.Switch(b[i]))
                {
                    Variable.ConstrainEqual(x[i], y[b[i]]);
                }
            }
            xPrior.ObservedValue = Bernoulli.PointMass(true);
            InferenceEngine engine = new InferenceEngine();
            object bActual = engine.Infer(b);
            Console.WriteLine(bActual);
        }

        [Fact]
        public void GateEnterOnePointMassTest()
        {
            Range i = new Range(2);
            Variable<int> b = Variable.Discrete(i, 0.1, 0.9).Named("b");
            VariableArray<bool> x = Variable.Array<bool>(i).Named("x");
            x[i] = Variable.Random(Bernoulli.PointMass(true)).ForEach(i);
            VariableArray<Bernoulli> dists = Variable.Constant(new Bernoulli[] { Bernoulli.PointMass(true), Bernoulli.PointMass(false) }, i).Named("dists");
            using (Variable.Switch(b))
            {
                Variable.ConstrainEqualRandom(x[b], dists[b]);
            }
            InferenceEngine engine = new InferenceEngine();
            Discrete bActual = engine.Infer<Discrete>(b);
            Console.WriteLine(bActual);
            Assert.True(bActual[0] == 1.0);
        }

        [Fact]
        public void GatedLiteralIndexingTest()
        {
            Range item = new Range(1).Named("item");
            VariableArray<bool> x = Variable.Array<bool>(item).Named("x");
            Variable<bool> b = Variable.New<bool>().Named("b");
            double xPrior = 0.1;
            double xLike = 0.3;
            using (Variable.If(b))
            {
                // use x[0]
                x[item] = Variable.Bernoulli(xPrior).ForEach(item);
                Variable.ConstrainEqualRandom(x[0], new Bernoulli(xLike));
            }
            using (Variable.IfNot(b))
            {
                // def x[0]
                x[0] = Variable.Bernoulli(xPrior);
                Variable.ConstrainEqualRandom(x[0], new Bernoulli(xLike));
            }
            InferenceEngine engine = new InferenceEngine();
            double z = xPrior * xLike + (1 - xPrior) * (1 - xLike);
            double xPost = xPrior * xLike / z;
            for (int trial = 0; trial < 2; trial++)
            {
                IDistribution<bool[]> xExpected = Distribution<bool>.Array(new Bernoulli[] { new Bernoulli(xPost) });
                if (trial == 0)
                {
                    b.ObservedValue = true;
                }
                else
                {
                    b.ObservedValue = false;
                }
                object xActual = engine.Infer(x);
                Console.WriteLine(StringUtil.JoinColumns("x = ", xActual, " should be ", xExpected));
                Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
            }
        }

        [Fact]
        public void CloseAllBlocksTest()
        {
            try
            {
                BadNestingError();
            }
            catch (InvalidOperationException) { }
            Variable.CloseAllBlocks();
            GatedConstrainEqualRandom();
        }

        [Fact]
        public void EnterLocalConditionalTest()
        {
            double cPrior = 0.3;
            double bPrior = 0.2;
            double bLike = 0.6;
            Range i = new Range(2).Named("i");
            Range j = new Range(2).Named("j");
            VariableArray2D<bool> bools = Variable.Array<bool>(i, j).Named("bools");
            bools[i, j] = Variable.Bernoulli(bPrior).ForEach(i, j);
            VariableArray<bool> c = Variable.Array<bool>(i).Named("c");
            using (ForEachBlock fbi = Variable.ForEach(i))
            {
                c[i] = Variable.Bernoulli(cPrior);
                using (Variable.If(c[i]))
                {
                    using (Variable.If(fbi.Index > 0))
                    {
                        using (ForEachBlock fbj = Variable.ForEach(j))
                        {
                            using (Variable.If(fbi.Index + fbj.Index > 1))
                            {
                                Variable.ConstrainEqualRandom(bools[fbi.Index - 1, j], new Bernoulli(bLike));
                            }
                        }
                    }
                }
            }
            InferenceEngine engine = new InferenceEngine();
            double sumT = bPrior * bLike + (1 - bPrior) * (1 - bLike);
            double Z = cPrior * sumT + (1 - cPrior);
            double cExpected1 = cPrior * sumT / Z;
            double bExpected01 = bPrior * (cPrior * bLike + (1 - cPrior)) / Z;
            IDistribution<bool[]> cExpected = Distribution<bool>.Array(new Bernoulli[] { new Bernoulli(cPrior), new Bernoulli(cExpected1) });
            IDistribution<bool[,]> bExpected = Distribution<bool>.Array(new Bernoulli[,]
                {
                    {new Bernoulli(bPrior), new Bernoulli(bExpected01)},
                    {new Bernoulli(bPrior), new Bernoulli(bPrior)},
                });
            object cActual = engine.Infer(c);
            object bActual = engine.Infer(bools);
            Console.WriteLine(StringUtil.JoinColumns("c = ", cActual, " should be ", cExpected));
            Console.WriteLine(StringUtil.JoinColumns("b = ", bActual, " should be ", bExpected));
            Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
            Assert.True(bExpected.MaxDiff(bActual) < 1e-10);
        }

        [Fact]
        public void IfOrderTest()
        {
            Gaussian result1 = FadingChannelSimplest();
            Console.WriteLine("Result A: {0}", result1);
            Gaussian result2 = FadingChannelSimplestFlipped();
            Console.WriteLine("Result B: {0}", result2);
            Assert.True(result1.MaxDiff(result2) < 1e-10);
        }

        public Gaussian FadingChannelSimplest()
        {
            Variable<double> state = new Variable<double>();
            state = Variable.GaussianFromMeanAndVariance(1, 5);
            Variable<double> observation = new Variable<double>();
            Variable<bool> symbol = Variable.Bernoulli(.5).Named("symbol");

            using (Variable.If(symbol))
            {
                observation.SetTo(Variable.GaussianFromMeanAndVariance(state, .001));
            }
            using (Variable.IfNot(symbol))
            {
                observation.SetTo(Variable.GaussianFromMeanAndVariance(-state, .001));
            }
            observation.ObservedValue = 1;

            InferenceEngine engine = new InferenceEngine();
            engine.ModelName = "FadingChannelSimplest";
            return engine.Infer<Gaussian>(state);
        }

        public Gaussian FadingChannelSimplestFlipped()
        {
            Variable<double> state = new Variable<double>();
            state = Variable.GaussianFromMeanAndVariance(1, 5);
            Variable<double> observation = new Variable<double>();
            Variable<bool> symbol = Variable.Bernoulli(.5).Named("symbol");

            using (Variable.If(symbol))
            {
                observation.SetTo(Variable.GaussianFromMeanAndVariance(0 - state, .001));
            }
            using (Variable.IfNot(symbol))
            {
                observation.SetTo(Variable.GaussianFromMeanAndVariance(state, .001));
            }
            observation.ObservedValue = 1;

            InferenceEngine engine = new InferenceEngine();
            engine.ModelName = "FadingChannelSimplestFlipped";
            return engine.Infer<Gaussian>(state);
        }

        [Fact]
        public void CasePriorObserved()
        {
            var prior = Variable.Observed(Discrete.Uniform(1)).Named("prior");
            var val = Variable.Random<int, Discrete>(prior).Named("val");
            var outv = Variable.New<bool>().Named("out");
            using (Variable.Case(val, 0))
            {
                outv.SetTo(Variable.Bernoulli(1.0));
            }
            Console.WriteLine(new InferenceEngine().Infer(outv));
        }

        [Fact]
        public void MissingSwitchFactorError()
        {
            Assert.Throws<CompilationFailedException>(() =>
            {
                Range item = new Range(2);
                VariableArray<bool> x = Variable.Array<bool>(item).Named("x");
                x[item] = Variable.Bernoulli(0.1).ForEach(item);
                Variable<int> index = Variable.DiscreteUniform(item).Named("index");
                Variable.ConstrainTrue(!x[index]);

                InferenceEngine engine = new InferenceEngine();
                Console.WriteLine(engine.Infer(index));

            });
        }

        [Fact]
        public void MissingSwitchConstraintError()
        {
            Assert.Throws<CompilationFailedException>(() =>
            {
                Range item = new Range(2);
                VariableArray<bool> x = Variable.Array<bool>(item).Named("x");
                x[item] = Variable.Bernoulli(0.1).ForEach(item);
                Variable<int> index = Variable.DiscreteUniform(item).Named("index");
                Variable.ConstrainTrue(x[index]);

                InferenceEngine engine = new InferenceEngine();
                Console.WriteLine(engine.Infer(index));

            });
        }

        [Fact]
        public void MissingSwitchInGateError()
        {
            Assert.Throws<CompilationFailedException>(() =>
            {
                Range item = new Range(2);
                VariableArray<bool> x = Variable.Array<bool>(item).Named("x");
                x[item] = Variable.Bernoulli(0.1).ForEach(item);
                Variable<int> index = Variable.DiscreteUniform(item).Named("index");
                Variable<bool> b = Variable.Bernoulli(0.2).Named("b");
                using (Variable.If(b))
                {
                    Variable.ConstrainTrue(x[index]);
                }

                InferenceEngine engine = new InferenceEngine();
                Console.WriteLine(engine.Infer(index));
                Assert.True(false, "Did not throw exception");
            });
        }

        [Fact]
        public void MissingSwitchObservedError()
        {
            Assert.Throws<CompilationFailedException>(() =>
            {

                Range item = new Range(2);
                VariableArray<bool> x = Variable.Array<bool>(item).Named("x");
                x.ObservedValue = new bool[] { true, false };
                Variable<int> index = Variable.DiscreteUniform(item).Named("index");
                Variable.ConstrainTrue(x[index]);

                InferenceEngine engine = new InferenceEngine();
                Console.WriteLine(engine.Infer(index));

            });
        }

        [Fact]
        public void ForEachSwitchTest()

        {
            Assert.Throws<CompilationFailedException>(() =>
            {

                int n = 2;
                Range item = new Range(n).Named("item");
                double xPrior = 0.2;
                Variable<bool> x = Variable.Bernoulli(xPrior).Named("x");
                Bernoulli[] priorsArray = new Bernoulli[]
                {
                    new Bernoulli(0.1),
                    new Bernoulli(0.2),
                    new Bernoulli(0.4)
                };
                Range prior = new Range(priorsArray.Length);
                VariableArray<Bernoulli> priors = Variable.Observed(priorsArray, prior).Named("priors");
                double[] indexProbs = new double[] { 0.1, 0.3, 0.6 };
                Variable<int> index = Variable.Discrete(prior, indexProbs).Named("index");
                using (Variable.ForEach(item))
                {
                    using (Variable.Switch(index))
                    {
                        Variable.ConstrainEqualRandom(x, priors[index]);
                    }
                }

                InferenceEngine engine = new InferenceEngine();
                double xLike = 0.0;
                for (int i = 0; i < priorsArray.Length; i++)
                {
                    xLike += priorsArray[i].GetProbTrue() * indexProbs[i];
                }
                double sumCondT = xPrior * System.Math.Pow(xLike, n);
                double sumCondF = (1 - xPrior) * System.Math.Pow(1 - xLike, n);
                Bernoulli xActual = engine.Infer<Bernoulli>(x);
                Bernoulli xExpected = new Bernoulli(sumCondT / (sumCondT + sumCondF));
                Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
                Assert.True(xExpected.MaxDiff(xActual) < 1e-10);

            });
        }

        [Fact]
        public void ForEachCaseTest()
        {
            Assert.Throws<CompilationFailedException>(() =>
            {

                double bPrior = 0.1;
                Variable<int> b = Variable.Discrete(bPrior, 1 - bPrior).Named("b");
                int n = 2;
                Range item = new Range(n).Named("item");
                double xPrior = 0.2;
                Variable<bool> x = Variable.Bernoulli(xPrior).Named("x");
                double xCondT = 0.3;
                double xCondF = 0.7;
                using (Variable.ForEach(item))
                {
                    using (Variable.Case(b, 0))
                    {
                        Variable.ConstrainEqualRandom(x, new Bernoulli(xCondT));
                    }
                    using (Variable.Case(b, 1))
                    {
                        Variable.ConstrainEqualRandom(x, new Bernoulli(xCondF));
                    }
                }
                InferenceEngine engine = new InferenceEngine();
                double sumCondTT = xPrior * System.Math.Pow(xCondT, n);
                double sumCondTF = (1 - xPrior) * System.Math.Pow(1 - xCondT, n);
                double sumCondFT = xPrior * System.Math.Pow(xCondF, n);
                double sumCondFF = (1 - xPrior) * System.Math.Pow(1 - xCondF, n);
                double sumCondT = sumCondTT + sumCondTF;
                double sumCondF = sumCondFT + sumCondFF;
                double z = bPrior * sumCondT + (1 - bPrior) * sumCondF;
                Discrete bActual = engine.Infer<Discrete>(b);
                double bProb0 = bPrior * sumCondT / z;
                Discrete bExpected = new Discrete(bProb0, 1 - bProb0);
                Console.WriteLine("b = {0} should be {1}", bActual, bExpected);
                Assert.True(bExpected.MaxDiff(bActual) < 1e-10);

            });
        }

        [Fact]
        public void ForEachCaseTest2()

        {
            Assert.Throws<CompilationFailedException>(() =>
            {

                double bPrior = 0.1;
                Variable<int> b = Variable.Discrete(bPrior, 1 - bPrior).Named("b");
                int n = 2;
                Range item = new Range(n).Named("item");
                double xPrior = 0.2;
                Variable<bool> x = Variable.Bernoulli(xPrior).Named("x");
                double xCondT = 0.3;
                double xCondF = 0.4;
                using (Variable.ForEach(item))
                {
                    using (Variable.Case(b, 0))
                    {
                        Variable.ConstrainEqualRandom(x, new Bernoulli(xCondT));
                    }
                }
                using (Variable.Case(b, 1))
                {
                    Variable.ConstrainEqualRandom(x, new Bernoulli(xCondF));
                }
                InferenceEngine engine = new InferenceEngine();
                double sumCondT = xPrior * System.Math.Pow(xCondT, n) + (1 - xPrior) * System.Math.Pow(1 - xCondT, n);
                double sumCondF = xPrior * xCondF + (1 - xPrior) * (1 - xCondF);
                double z = bPrior * sumCondT + (1 - bPrior) * sumCondF;
                Discrete bActual = engine.Infer<Discrete>(b);
                double bProb0 = bPrior * sumCondT / z;
                Discrete bExpected = new Discrete(bProb0, 1 - bProb0);
                Console.WriteLine("b = {0} should be {1}", bActual, bExpected);
                Assert.True(bExpected.MaxDiff(bActual) < 1e-10);

            });
        }

        // should throw an exception that Case(b) must be outside of ForEach(item)
        [Fact]
        [Trait("Category", "OpenBug")]
        public void ForEachCaseTest3()
        {
            Assert.Throws<CompilationFailedException>(() =>
            {

                double bPrior = 0.1;
                Variable<int> b = Variable.Discrete(bPrior, 1 - bPrior).Named("b");
                int n = 2;
                Range item = new Range(n).Named("item");
                double xPrior = 0.2;
                Variable<bool> x = Variable.Bernoulli(xPrior).Named("x");
                double xCondT = 0.3;
                double xCondF = 0.7;
                var c = Variable.Array<bool>(item).Named("c");
                c[item] = Variable.Bernoulli(1.0).ForEach(item);
                using (Variable.ForEach(item))
                {
                    using (Variable.If(c[item]))
                    {
                        using (Variable.Case(b, 0))
                        {
                            Variable.ConstrainEqualRandom(x, new Bernoulli(xCondT));
                        }
                        using (Variable.Case(b, 1))
                        {
                            Variable.ConstrainEqualRandom(x, new Bernoulli(xCondF));
                        }
                    }
                }
                InferenceEngine engine = new InferenceEngine();
                double sumCondTT = xPrior * System.Math.Pow(xCondT, n);
                double sumCondTF = (1 - xPrior) * System.Math.Pow(1 - xCondT, n);
                double sumCondFT = xPrior * System.Math.Pow(xCondF, n);
                double sumCondFF = (1 - xPrior) * System.Math.Pow(1 - xCondF, n);
                double sumCondT = sumCondTT + sumCondTF;
                double sumCondF = sumCondFT + sumCondFF;
                double z = bPrior * sumCondT + (1 - bPrior) * sumCondF;
                Discrete bActual = engine.Infer<Discrete>(b);
                double bProb0 = bPrior * sumCondT / z;
                Discrete bExpected = new Discrete(bProb0, 1 - bProb0);
                Console.WriteLine("b = {0} should be {1}", bActual, bExpected);
                Assert.True(bExpected.MaxDiff(bActual) < 1e-10);

            });
        }

        // Test that compiler throws due to unexpected nesting of containers.  
        // To compile such models, we would need another transform to reorder containers.
        // Also see:
        //(new BlogTests()).Handedness2ForEach();
        //(new ModelTests()).ClickChainTest();
        [Fact]
        public void ForEachIfRandomTest()
        {
            Assert.Throws<CompilationFailedException>(() =>
            {

                double bPrior = 0.1;
                Variable<bool> b = Variable.Bernoulli(bPrior).Named("b");
                int n = 2;
                Range item = new Range(n).Named("item");
                double xPrior = 0.2;
                Variable<bool> x = Variable.Bernoulli(xPrior).Named("x");
                double xCondT = 0.3;
                using (Variable.ForEach(item))
                {
                    using (Variable.If(b))
                    {
                        Variable.ConstrainEqualRandom(x, new Bernoulli(xCondT));
                    }
                }
                double xCondF = 0.4;
                using (Variable.IfNot(b))
                {
                    Variable.ConstrainEqualRandom(x, new Bernoulli(xCondF));
                }
                InferenceEngine engine = new InferenceEngine();
                double sumCondT = xPrior * System.Math.Pow(xCondT, n) + (1 - xPrior) * System.Math.Pow(1 - xCondT, n);
                double sumCondF = xPrior * xCondF + (1 - xPrior) * (1 - xCondF);
                double z = bPrior * sumCondT + (1 - bPrior) * sumCondF;
                Bernoulli bActual = engine.Infer<Bernoulli>(b);
                Bernoulli bExpected = new Bernoulli(bPrior * sumCondT / z);
                Console.WriteLine("b = {0} should be {1}", bActual, bExpected);
                Assert.True(bExpected.MaxDiff(bActual) < 1e-10);

            });
        }

        [Fact]
        public void ForEachIfRandomTest2()
        {
            Assert.Throws<CompilationFailedException>(() =>
            {

                int L = 1;

                Variable<int> Kcount = Variable<int>.New<int>().Named("Kcount");
                Kcount.ObservedValue = 1;

                Variable<int> Ncount = Variable<int>.New<int>().Named("Ncount");
                Ncount.ObservedValue = 1;

                Range K = new Range(Kcount).Named("K");
                Range N = new Range(Ncount).Named("N");

                // Define per positions parameters
                var thetas = new VariableArray<Vector>[L]; // Transition matrix
                var emissions = new VariableArray<double>[L]; // emission matrix

                for (int l = 0; l < L; l++)
                {
                    var theta_prior_var = Variable.Array<Dirichlet>(K).Named("ThetaPrior" + l);
                    theta_prior_var.ObservedValue = new Dirichlet[] { };

                    thetas[l] = Variable.Array<Vector>(K).Attrib(new ValueRange(K)).Named("Theta" + l);
                    thetas[l][K] = Variable.Random<Vector, Dirichlet>(theta_prior_var[K]);
                }

                for (int l = 0; l < L; l++)
                {
                    emissions[l] = Variable.Array<double>(K).Named("emission" + l);
                }

                // Define per position variables
                var hs = new VariableArray<int>[L + 1]; // Hidden unit            
                var vsi = new VariableArray<bool>[L];

                var pointmass_on_0_prior = Variable<Discrete>.New<Discrete>().Named("PointMassOn0");
                pointmass_on_0_prior.ObservedValue = Discrete.Uniform(0);

                for (int l = 0; l < L + 1; l++)
                {
                    hs[l] = Variable.Array<int>(N).Named("H" + (l - 1)).Attrib(new ValueRange(K));
                }
                hs[0][N] = Variable<int>.Random(pointmass_on_0_prior).ForEach(N);

                Variable<int> TwoKcount = Variable<int>.New<int>().Named("TwoKcount");
                TwoKcount.ObservedValue = 2;
                Range TwoK = new Range(TwoKcount).Named("TwoK");

                Range S = new Range(2);

                VariableArray<int>[] LociAtAncestor = new VariableArray<int>[L];
                VariableArray<Beta> EarlyMutationDirichletOptions =
                Variable.Observed(new Beta[] { new Beta(200, 2), new Beta(2, 200) }, S).Named("EarlyMutationDirichletOptions");

                Variable<bool>[] EarlyMutation = new Variable<bool>[L];

                for (int l = 0; l < L; l++)
                {
                    EarlyMutation[l] = Variable.Bernoulli(0.5).Named("EarlyMutation" + l);
                    using (Variable.If(EarlyMutation[l]))
                    {
                        LociAtAncestor[l] = Variable.Array<int>(K).Named("LociAtAncestor" + l);

                        using (Variable.ForEach(K))
                        {
                            LociAtAncestor[l][K] = //Variable.DiscreteUniform(S);
                            Variable.Discrete(S, Discrete.Uniform(2).GetProbs());
                            using (Variable.Switch(LociAtAncestor[l][K]))
                            {
                                emissions[l][K] = Variable<double>.Random(EarlyMutationDirichletOptions[LociAtAncestor[l][K]]);
                            }
                        }
                    }
                }

                VariableArray<VariableArray<Beta>, Beta[][]> NoEarlyMutationOptions =
                Variable.Array(Variable.Array<Beta>(K), TwoK).Named("NoEarlyMutationOptions");
                NoEarlyMutationOptions.ObservedValue = new Beta[][] { };

                var AncestorWithMutation = new Variable<int>[L];

                for (int l = 0; l < L; l++)
                {
                    using (Variable.IfNot(EarlyMutation[l]))
                    {
                        AncestorWithMutation[l] = Variable.DiscreteUniform(TwoK).Named("AncestorWithMutation" + l);
                        using (Variable.Switch(AncestorWithMutation[l]))
                        {
                            emissions[l][K] = Variable.Random<double, Beta>(
                            NoEarlyMutationOptions[AncestorWithMutation[l]][K]);
                        }
                    }
                }
                using (Variable.ForEach(N))
                {
                    for (int l = 0; l < L; l++)
                    {
                        using (Variable.If(EarlyMutation[l]))
                        {
                            using (Variable.Switch(hs[l][N]))
                            {
                                hs[l + 1][N] = Variable.Discrete(thetas[l][hs[l][N]]);
                                ;
                            }
                        }
                        using (Variable.IfNot(EarlyMutation[l]))
                        {
                            using (Variable.Switch(hs[l][N]))
                            {
                                hs[l + 1][N] = Variable.Discrete(thetas[l][hs[l][N]]);
                            }
                        }
                    }
                }


                InferenceEngine engine = new InferenceEngine();
                engine.Infer(hs[1]);

            });
        }

        // Test that implicit 'for' loops are nested correctly inside 'if' blocks
        [Fact]
        public void IfForEachTest()
        {
            double Rho = 0.5;
            Range featureRange = new Range(2).Named("featureRange");
            Range dataRange = new Range(2).Named("dataRange");
            VariableArray<VariableArray<double>, double[][]> w = Variable.Array(Variable.Array<double>(featureRange), dataRange);
            VariableArray<Gaussian> spike = Variable.Array<Gaussian>(featureRange).Named("spike");
            VariableArray<Gaussian> slab = Variable.Array<Gaussian>(featureRange).Named("slab");
            VariableArray<double> output = Variable.Array<double>(dataRange).Named("output");
            VariableArray<VariableArray<double>, double[][]> input = Variable.Array(Variable.Array<double>(featureRange), dataRange);
            VariableArray<bool> gamma = Variable.Array<bool>(featureRange).Named("gamma");

            gamma[featureRange] = Variable.Bernoulli(Rho).ForEach(featureRange);
            using (Variable.ForEach(featureRange))
                spike[featureRange] = Gaussian.FromMeanAndVariance(0, 0);
            using (Variable.ForEach(featureRange))
                slab[featureRange] = Gaussian.FromMeanAndVariance(0, 1);

            using (Variable.ForEach(featureRange))
            {
                using (Variable.If(gamma[featureRange]))
                {
                    // spike is the true distribution for w
                    w[dataRange][featureRange] = Variable<double>.Random(spike[featureRange]).ForEach(dataRange);
                }
                using (Variable.IfNot(gamma[featureRange]))
                {
                    // slab is the true distribution for w
                    w[dataRange][featureRange] = Variable<double>.Random(slab[featureRange]).ForEach(dataRange);
                }
            }

            using (Variable.ForEach(dataRange))
            {
                var tmp = Variable<double>.Array(featureRange);
                tmp[featureRange] = w[dataRange][featureRange] * input[dataRange][featureRange];
                output[dataRange] = Variable<double>.Sum(tmp);
            }

            output.ObservedValue = Util.ArrayInit(2, i => 0.0);
            input.ObservedValue = Util.ArrayInit(2, i => Util.ArrayInit(2, j => 0.0));

            InferenceEngine ie = new InferenceEngine();
            var gammaPosterior = ie.Infer<Bernoulli[]>(gamma);
            Console.WriteLine(gammaPosterior);
        }

        [Fact]
        public void Wolfgang()
        {
            Range K = new Range(2).Named("K");
            Range S = new Range(2).Named("S");
            Range dirichletFCount = new Range(2).Named("dirichletFCount");
            Range vr = new Range(2).Named("vr");

            double cPrior = 0.1;
            Variable<bool> c = Variable.Bernoulli(cPrior).Named("c");
            VariableArray<Vector> probs = Variable.Array<Vector>(K).Named("probs");

            VariableArray<Dirichlet> dirichletT = Variable.Array<Dirichlet>(S).Named("dirichletT");
            dirichletT.ObservedValue = new Dirichlet[]
                {
                    new Dirichlet(200, 2),
                    new Dirichlet(2, 200)
                };
            VariableArray<Dirichlet> dirichletT2 = Variable.Array<Dirichlet>(K).Named("dirichletT2");
            dirichletT2.ObservedValue = new Dirichlet[]
                {
                    new Dirichlet(200, 2),
                    new Dirichlet(2, 200)
                };
            var dirichletF = Variable.Array(Variable.Array<Dirichlet>(K), dirichletFCount).Named("dirichletF");
            dirichletF.ObservedValue =
                new Dirichlet[][]
                    {
                        new Dirichlet[]
                            {
                                Dirichlet.PointMass(0.2, 0.8),
                                Dirichlet.PointMass(0.8, 0.2)
                            },
                        new Dirichlet[]
                            {
                                Dirichlet.Uniform(2),
                                Dirichlet.Uniform(2)
                            }
                    };

            using (Variable.If(c))
            {
                if (false)
                {
                    probs[K] = Variable<Vector>.Random(dirichletT2[K]);
                }
                else
                {
                    VariableArray<int> switchT = Variable.Array<int>(K).Named("switchT");
                    using (Variable.ForEach(K))
                    {
                        switchT[K] = Variable.DiscreteUniform(S);
                        using (Variable.Switch(switchT[K]))
                        {
                            probs[K] = Variable<Vector>.Random(dirichletT[switchT[K]]);
                        }
                    }
                }
            }
            using (Variable.IfNot(c))
            {
                if (true)
                {
                    VariableArray<int> switchT = Variable.Array<int>(K).Named("switchT2");
                    using (Variable.ForEach(K))
                    {
                        switchT[K] = Variable.DiscreteUniform(S);
                        using (Variable.Switch(switchT[K]))
                        {
                            probs[K] = Variable<Vector>.Random(dirichletT[switchT[K]]);
                        }
                    }
                }
                else
                {
                    Variable<int> switchF = Variable.DiscreteUniform(dirichletFCount).Named("switchF");
                    using (Variable.Switch(switchF))
                    {
                        using (Variable.ForEach(K))
                        {
                            probs[K] = Variable<Vector>.Random(dirichletF[switchF][K]);
                        }
                    }
                }
            }

            VariableArray<int> obs = Variable.Observed(new int[] { 0, 0 }, K).Named("obs");
            using (Variable.ForEach(K))
            {
                obs[K] = Variable.Discrete(probs[K]);
            }

            var ie = new InferenceEngine(new VariationalMessagePassing());
            Bernoulli cExpected = new Bernoulli(cPrior);
            Bernoulli cActual = ie.Infer<Bernoulli>(c);
            Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
            Assert.True(cExpected.MaxDiff(cActual) < 1e-4);
        }

        [Fact]
        public void Wolfgang2()
        {
            double cPrior = 0.1;
            Variable<bool> c = Variable.Bernoulli(cPrior).Named("c");

            Range K = new Range(2).Named("K");
            Range S = new Range(2).Named("S");
            Range vr = new Range(2).Named("vr");

            VariableArray<Vector> probs = Variable.Array<Vector>(K).Named("probs");

            VariableArray<Dirichlet> dirichletT = Variable.Array<Dirichlet>(S).Named("dirichletT");
            dirichletT.ObservedValue = new Dirichlet[]
                {
                    new Dirichlet(200, 2),
                    new Dirichlet(2, 200)
                };

            using (Variable.If(c))
            {
                if (true)
                {
                    using (Variable.ForEach(K))
                    {
                        Variable<int> switchT = Variable.DiscreteUniform(S).Named("switchT");
                        using (Variable.Switch(switchT))
                        {
                            probs[K] = Variable.Random<Vector, Dirichlet>(dirichletT[switchT]);
                        }
                    }
                }
                else
                {
                    VariableArray<int> switchT = Variable.Array<int>(K).Named("switchT");
                    using (Variable.ForEach(K))
                    {
                        switchT[K] = Variable.DiscreteUniform(S);
                        using (Variable.Switch(switchT[K]))
                        {
                            probs[K] = Variable.Random<Vector, Dirichlet>(dirichletT[switchT[K]]);
                        }
                    }
                }
            }
            using (Variable.IfNot(c))
            {
                if (true)
                {
                    using (Variable.ForEach(K))
                    {
                        Variable<int> switchT2 = Variable.DiscreteUniform(S).Named("switchT2");
                        using (Variable.Switch(switchT2))
                        {
                            probs[K] = Variable.Random<Vector, Dirichlet>(dirichletT[switchT2]);
                        }
                    }
                }
                else
                {
                    VariableArray<int> switchT2 = Variable.Array<int>(K).Named("switchT2");
                    using (Variable.ForEach(K))
                    {
                        switchT2[K] = Variable.DiscreteUniform(S);
                        using (Variable.Switch(switchT2[K]))
                        {
                            probs[K] = Variable.Random<Vector, Dirichlet>(dirichletT[switchT2[K]]);
                        }
                    }
                }
            }

            VariableArray<int> obs = Variable.Observed(new int[] { 0, 0 }, K).Named("obs");
            using (Variable.ForEach(K))
            {
                obs[K] = Variable.Discrete(probs[K]);
            }

            var ie = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(ie.Infer(probs));
            Bernoulli cExpected = new Bernoulli(cPrior);
            Bernoulli cActual = ie.Infer<Bernoulli>(c);
            Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
            Assert.True(cExpected.MaxDiff(cActual) < 1e-4);
        }

        [Fact]
        public void Wolfgang3()
        {
            Range Kr = new Range(5).Named("K");
            int L = 3; // L = 1 or L = 2 works fine

            var H = new Variable<int>[L];
            var Hobs = new Variable<Discrete>[L];
            var R = new Variable<bool>[L - 1];
            var Rbeta = new Variable<Bernoulli>[L - 1];

            H[0] = Variable.DiscreteUniform(Kr).Named("H0");

            for (int l = 0; l < L; l++)
            {
                if (l > 0)
                {
                    //Rbeta[l-1] 
                    //Rbeta[l-1] = Variable.Observed(new Bernoulli()).Named("Rbeta" + l);
                    //R[l-1] = Variable.Random<bool, Bernoulli>(Rbeta[l-1]).Named("R" + l);
                    R[l - 1] = Variable.Bernoulli(0.5);

                    H[l] = Variable.New<int>().Named("H" + l);

                    using (Variable.If(R[l - 1]))
                    {
                        H[l].SetTo(Variable.Copy(H[l - 1]));
                    }
                    using (Variable.IfNot(R[l - 1]))
                    {
                        H[l].SetTo(Variable.DiscreteUniform(Kr));
                    }
                }
                if (l == L - 1 || l == L - 2)
                {
                    // removing either condition will stop it crashing
                    Hobs[l] = Variable.Observed(Discrete.Uniform(Kr.SizeAsInt)).Named("Hobs" + l);
                    Variable.ConstrainEqualRandom(H[l], Hobs[l]);
                }
            }
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing()); // EP is fine
            //ie.BrowserMode = BrowserMode.Never;  // removing this gives a COMException when trying to display browser
            Console.WriteLine(ie.Infer(H[L - 1]));
        }

        [Fact]
        public void Laura()
        {
            double[] alphaPhi = new double[0];
            int cNum = 0;
            InferenceEngine inferenceEngine = new InferenceEngine();

            var uRangeVar = Variable.New<int>().Named("uRangeVar");
            uRangeVar.ObservedValue = 0;
            Range uRange = (new Range(uRangeVar)).Named("uRange");

            var fLengths = Variable.Array<int>(uRange).Named("fLengths");
            fLengths.ObservedValue = new int[0];
            Range fVaryRange = (new Range(fLengths[uRange])).Named("fVariRange");


            var friendshipLengths = Variable.New<int>().Named("friendshipLengths");
            friendshipLengths.ObservedValue = 0;
            Range friendshipRange = (new Range(friendshipLengths)).Named("friendshipRange");

            var sLengths = Variable.Array<int>(uRange).Named("sLengths");
            sLengths.ObservedValue = new int[0];
            Range sRange = (new Range(sLengths[uRange])).Named("sRange");
            Range cRange = (new Range(cNum)).Named("cRange");

            var ufToE = Variable.Array(Variable.Array<int>(fVaryRange), uRange).Named("ufToE").Attrib(new ValueRange(friendshipRange));
            ufToE.ObservedValue = new int[0][];

            var songData = Variable.Array(Variable.Array<int>(sRange), uRange).Named("songData");
            songData.ObservedValue = new int[0][];

            var songVocabRangeVar = Variable.New<int>().Named("songVocabRangeVar");
            songVocabRangeVar.ObservedValue = 0;
            Range songVocabRange = new Range(songVocabRangeVar).Named("songVocabRange");

            var backgroundDistr = Variable.New<Vector>();
            backgroundDistr.ObservedValue = Vector.Zero(0);


            var phi = (Variable.Array<Vector>(cRange)).Named("phi");
            using (Variable.ForEach(cRange))
            {
                phi[cRange] = Variable.Dirichlet(alphaPhi);
            }

            var alphaPsi = (Variable.Array<Dirichlet>(uRange)).Named("alphaPsi");
            alphaPsi.ObservedValue = new Dirichlet[0];

            var psi = (Variable.Array<Vector>(uRange)).Named("psi").Attrib(new ValueRange(fVaryRange));
            var epsilon = (Variable.Array<double>(uRange)).Named("epsilon");
            using (Variable.ForEach(uRange))
            {
                psi[uRange] = Variable.Random<Vector, Dirichlet>(alphaPsi[uRange]);
                epsilon[uRange] = Variable.Beta(1, 4).Named("epsilon");
            }

            var f = Variable.Array(Variable.Array<int>(sRange), uRange).Named("f");
            var isBack = Variable.Array(Variable.Array<bool>(sRange), uRange).Named("isBack");

            var edgeLabel = (Variable.Array<int>(friendshipRange)).Named("edgeLabel");

            Variable<Vector> communityDistr = Variable.DirichletUniform(cRange).Named("communityDistr");

            using (Variable.ForEach(friendshipRange))
            {
                edgeLabel[friendshipRange] = Variable.Discrete(communityDistr);
            }

            using (Variable.ForEach(uRange))
            {
                using (Variable.ForEach(sRange))
                {
                    isBack[uRange][sRange] = Variable.Bernoulli(epsilon[uRange]);
                    using (Variable.If(isBack[uRange][sRange]))
                    {
                        songData[uRange][sRange] = Variable.Discrete(backgroundDistr);
                        // ============================================
                        // this triggers the issue
                        f[uRange][sRange] = Variable.Discrete(fVaryRange);
                        //
                        // this works
                        //f[uRange][sRange] = Variable.Discrete(trash[uRange]).Attrib(new ValueRange(fVaryRange));
                        // ============================================
                    }
                    using (Variable.IfNot(isBack[uRange][sRange]))
                    {
                        f[uRange][sRange] = Variable.Discrete(psi[uRange]).Attrib(new ValueRange(fVaryRange));

                        Variable<int> currentFriend = f[uRange][sRange];
                        using (Variable.Switch(currentFriend))
                        {
                            Variable<int> friendshipIdx = ufToE[uRange][currentFriend];
                            using (Variable.Switch(friendshipIdx))
                            {
                                Variable<int> currentCommunity = edgeLabel[friendshipIdx];
                                using (Variable.Switch(currentCommunity))
                                {
                                    songData[uRange][sRange] = Variable.Discrete(phi[currentCommunity]);
                                }
                            }
                        }
                    }
                }
            }

            inferenceEngine.Infer(phi);
        }

        [Fact]
        public void ArrayDefinedManyWaysInGate()
        {
            double aPrior = 0.15;
            Variable<bool> PriorType = Variable.Bernoulli(aPrior).Named("PriorType");
            Range item = new Range(2).Named("item");
            VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
            IDistribution<bool[]> boolsPrior1 = Distribution<bool>.Array(new Bernoulli[] { new Bernoulli(0.1), new Bernoulli(0.2) });
            Bernoulli boolsLike = new Bernoulli(0.4);
            using (Variable.If(PriorType))
            {
                bools.SetTo(Variable.Random(boolsPrior1));
                Variable.ConstrainEqualRandom(bools[item], boolsLike);
            }
            using (Variable.IfNot(PriorType))
            {
                bools[item] = Variable.Bernoulli(0.3).ForEach(item);
                Variable.ConstrainEqualRandom(bools, (Sampleable<bool[]>)boolsPrior1);
            }
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(bools));

            BernoulliArray prior = (BernoulliArray)boolsPrior1;
            double sumCondT = 0.0;
            for (int i = 0; i < prior.Count; i++)
            {
                sumCondT += prior[i].GetLogAverageOf(boolsLike);
            }
            sumCondT = System.Math.Exp(sumCondT);
            double sumCondF = 0.0;
            Bernoulli boolsLike2 = new Bernoulli(0.3);
            for (int i = 0; i < prior.Count; i++)
            {
                sumCondF += prior[i].GetLogAverageOf(boolsLike2);
            }
            sumCondF = System.Math.Exp(sumCondF);
            double z = aPrior * sumCondT + (1 - aPrior) * sumCondF;
            Bernoulli aExpected = new Bernoulli(aPrior * sumCondT / z);
            Bernoulli aActual = engine.Infer<Bernoulli>(PriorType);
            Console.WriteLine("a = {0} should be {1}", aActual, aExpected);
            Assert.True(aExpected.MaxDiff(aActual) < 1e-10);
        }

        internal void ArrayDefinedInGate()
        {
            double aPrior = 0.15;
            Variable<bool> PriorType = Variable.Bernoulli(aPrior).Named("PriorType");
            Range item = new Range(2).Named("item");
            VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
            IDistribution<bool[]> boolsPrior1 = Distribution<bool>.Array(new Bernoulli[] { new Bernoulli(0.1), new Bernoulli(0.2) });
            Bernoulli boolsLike = new Bernoulli(0.4);
            using (Variable.If(PriorType))
            {
                bools[item] = Variable.Bernoulli(0.3).ForEach(item);
                Variable.ConstrainEqualRandom(bools[item], boolsLike);
            }
            using (Variable.IfNot(PriorType))
            {
                bools[item] = Variable.Bernoulli(0.3).ForEach(item);
                Variable.ConstrainEqualRandom(bools[item], boolsLike);
            }
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(bools));

            BernoulliArray prior = (BernoulliArray)boolsPrior1;
            double sumCondT = 0.0;
            for (int i = 0; i < prior.Count; i++)
            {
                sumCondT += prior[i].GetLogAverageOf(boolsLike);
            }
            sumCondT = System.Math.Exp(sumCondT);
            double sumCondF = 0.0;
            Bernoulli boolsLike2 = new Bernoulli(0.3);
            for (int i = 0; i < prior.Count; i++)
            {
                sumCondF += prior[i].GetLogAverageOf(boolsLike2);
            }
            sumCondF = System.Math.Exp(sumCondF);
            double z = aPrior * sumCondT + (1 - aPrior) * sumCondF;
            Bernoulli aExpected = new Bernoulli(aPrior * sumCondT / z);
            Bernoulli aActual = engine.Infer<Bernoulli>(PriorType);
            //Console.WriteLine("a = {0} should be {1}", aActual, aExpected);
            //Assert.True(aExpected.MaxDiff(aActual) < 1e-10);
        }

        [Fact]
        public void ArrayUsedManyTimesInGateTest()
        {
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    for (int k = 0; k < 2; k++)
                    {
                        for (int k2 = 0; k2 < 2; k2++)
                        {
                            ArrayUsedManyTimesInGate(i > 0, j > 0, k > 0, k2 > 0);
                        }
                    }
                }
            }
        }

        private void ArrayUsedManyTimesInGate(bool includeIndex0, bool includeSwitch, bool includeData, bool includeData2)
        {
            Console.WriteLine("{0} {1} {2} {3}", includeIndex0, includeSwitch, includeData, includeData2);
            Range item = new Range(2).Named("item");
            VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
            double bPrior = 0.3;
            bools[item] = Variable.Bernoulli(bPrior).ForEach(item);
            double cPrior = 0.1;
            Variable<bool> c = Variable.Bernoulli(cPrior).Named("c");
            Variable<int> index0 = Variable.Observed(0).Named("index0");
            Variable<int> index0c = Variable.Constant(0).Named("index0c");
            Variable<int> index0s = Variable.Observed(0).Named("index0s").Attrib(new ValueRange(item));
            double data0 = 0.2;
            double data0c = 0.4;
            double data0s = 0.25;
            double[] data = new double[] { 0.4, 0.7 };
            Bernoulli[] dataArray = new Bernoulli[data.Length];
            for (int i = 0; i < dataArray.Length; i++)
            {
                dataArray[i] = new Bernoulli(data[i]);
            }
            double[] data2 = new double[] { 0.6, 0.8 };
            Bernoulli[] dataArray2 = new Bernoulli[data2.Length];
            for (int i = 0; i < dataArray2.Length; i++)
            {
                dataArray2[i] = new Bernoulli(data2[i]);
            }
            double[] data3 = new double[] { 0.9, 0.65 };
            Bernoulli[] dataArray3 = new Bernoulli[data3.Length];
            for (int i = 0; i < dataArray3.Length; i++)
            {
                dataArray3[i] = new Bernoulli(data3[i]);
            }
            VariableArray<Bernoulli> dataVar = Variable.Observed(dataArray2, item).Named("dataVar");
            using (Variable.If(c))
            {
                if (includeIndex0)
                {
                    Variable.ConstrainEqualRandom(bools[index0], new Bernoulli(data0));
                    Variable.ConstrainEqualRandom(bools[index0c], new Bernoulli(data0c));
                }
                if (includeSwitch)
                {
                    using (Variable.Switch(index0s))
                    {
                        Variable.ConstrainEqualRandom(bools[index0s], new Bernoulli(data0s));
                    }
                }
                if (includeData)
                {
                    Variable.ConstrainEqualRandom(bools, (Sampleable<bool[]>)Distribution<bool>.Array(dataArray));
                }
                if (includeData2)
                {
                    using (Variable.ForEach(item))
                    {
                        Variable.ConstrainEqualRandom(bools[item], dataVar[item]);
                    }
                }
            }

            InferenceEngine engine = new InferenceEngine();
            Bernoulli[] boolsExpectedArray = new Bernoulli[item.SizeAsInt];
            double p0TCondT = bPrior;
            double p0FCondT = (1 - bPrior);
            if (includeData)
            {
                p0TCondT *= data[0];
                p0FCondT *= (1 - data[0]);
            }
            if (includeData2)
            {
                p0TCondT *= data2[0];
                p0FCondT *= (1 - data2[0]);
            }
            if (includeIndex0)
            {
                p0TCondT *= data0 * data0c;
                p0FCondT *= (1 - data0) * (1 - data0c);
            }
            if (includeSwitch)
            {
                p0TCondT *= data0s;
                p0FCondT *= (1 - data0s);
            }
            double sum0CondT = p0TCondT + p0FCondT;
            double p1TCondT = bPrior;
            double p1FCondT = (1 - bPrior);
            if (includeData)
            {
                p1TCondT *= data[1];
                p1FCondT *= (1 - data[1]);
            }
            if (includeData2)
            {
                p1TCondT *= data2[1];
                p1FCondT *= (1 - data2[1]);
            }
            double sum1CondT = p1TCondT + p1FCondT;
            double z = cPrior * (sum0CondT * sum1CondT) + (1 - cPrior);
            Bernoulli cExpected = new Bernoulli(cPrior * (sum0CondT * sum1CondT) / z);
            boolsExpectedArray[0] = new Bernoulli((cPrior * p0TCondT * sum1CondT + (1 - cPrior) * bPrior) / z);
            boolsExpectedArray[1] = new Bernoulli((cPrior * sum0CondT * p1TCondT + (1 - cPrior) * bPrior) / z);
            IDistribution<bool[]> boolsExpected = Distribution<bool>.Array(boolsExpectedArray);
            object boolsActual = engine.Infer(bools);
            Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsActual, " should be ", boolsExpected));
            Bernoulli cActual = engine.Infer<Bernoulli>(c);
            Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
            Assert.True(boolsExpected.MaxDiff(boolsActual) < 1e-10);
            Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
        }

        [Fact]
        public void IfNotObservedSwitchRandomTest()
        {
            Range Kout = new Range(3).Named("Kout");

            VariableArray<Vector> emissions = Variable.Array<Vector>(Kout).Named("Emissions");
            emissions[Kout] = Variable.Dirichlet(Kout, new double[] { 1.0, 2.0, 3.0 }).ForEach(Kout);

            Variable<int> H = Variable.Discrete(Kout, 0.5, 0.5, 0.0).Named("H");
            Variable<int> V = Variable.Observed<int>(0).Named("V");
            Variable<bool> IsMissingV = Variable.Observed(true).Named("IsMissingV");

            using (Variable.IfNot(IsMissingV))
            {
                using (Variable.Switch(H))
                {
                    V.SetTo(Variable.Discrete(emissions[H]));
                }
            }

            var engine = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(engine.Infer(H));
        }

        [Fact]
        public void SwitchForEachSwitchTest()
        {
            Range Kin = new Range(2).Named("Kin");
            Range Parents = new Range(2).Named("Parents");
            Range Phases = new Range(2).Named("Phases");

            VariableArray<Vector> theta = Variable.Array<Vector>(Kin).Named("Theta");
            theta[Kin] = Variable.Dirichlet(new double[] { 1.0, 2.0 }).ForEach(Kin);

            VariableArray<int> ModelBoundaryLeft = Variable.Array<int>(Parents).Named("ModelBoundaryLeft").Attrib(new ValueRange(Kin));
            ModelBoundaryLeft[Parents] = Variable.Discrete(0.1, 0.2).ForEach(Parents);

            VariableArray<int> H = Variable.Array<int>(Parents).Named("H");

            Variable<int> Phase = Variable.Discrete(0.5, 0.5).Named("Phase").Attrib(new ValueRange(Phases));

            using (Variable.Switch(Phase))
            {
                using (Variable.ForEach(Parents))
                {
                    using (Variable.Switch(ModelBoundaryLeft[Parents]))
                    {
                        H[Parents] = Variable.Discrete(theta[ModelBoundaryLeft[Parents]]);
                    }
                }
            }

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(engine.Infer(H));
        }

        // Tests a model where a backward message only exists for certain conditions.
        [Fact]
        public void IfObservedSwitchRandomTest()
        {
            Range Phases = new Range(2).Named("Phases");
            Variable<Vector> H = Variable.Dirichlet(new double[] { 0.5, 0.5 }).Named("H");

            Variable<int> V = Variable.Observed(1);

            Variable<int> Phase = Variable.Discrete(Phases, 1.0, 1.0).Named("Phase");
            Variable<bool> isMissing = Variable.Observed(true);

            using (Variable.IfNot(isMissing))
            {
                using (Variable.Switch(Phase))
                {
                    V.SetTo(Variable.Discrete(H));
                }
            }
            var engine = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(engine.Infer(H));
        }

        [Fact]
        public void GatedArrayCopyTest()
        {
            Range item = new Range(2).Named("item");
            VariableArray<bool> x = Variable.Array<bool>(item).Named("x");
            x[item] = Variable.Bernoulli(0.2).ForEach(item);
            VariableArray<bool> xCopy = Variable.Array<bool>(item).Named("xCopy");
            Variable<bool> c = Variable.Bernoulli(0.1).Named("c");
            using (Variable.If(c))
            {
                xCopy.SetTo(Variable.Copy(x));
            }
            using (Variable.IfNot(c))
            {
                xCopy.SetTo(Variable.Copy(x));
            }
            var engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(xCopy));
        }

        [Fact]
        public void SwitchArrayCopyTest()
        {
            Range item = new Range(2).Named("item");
            VariableArray<bool> x = Variable.Array<bool>(item).Named("x");
            x[item] = Variable.Bernoulli(0.2).ForEach(item);            
            VariableArray<bool> xCopy = Variable.Array<bool>(item).Named("xCopy");
            Range switchValues = new Range(2).Named("switchValues");
            Variable<int> switchVar = Variable.Discrete(switchValues, 1.0, 1.0).Named("switchVar");
            using (Variable.Switch(switchVar))
            {
                xCopy.SetTo(Variable.Copy(x));
            }
            var engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(xCopy));
        }

        // DepthCloningTransform requires two passes on this model.
        [Fact]
        public void SwitchDeepArrayCopyTest()
        {
            Range outer = new Range(2).Named("outer");
            Range inner = new Range(2).Named("inner");
            var x = Variable.Array(Variable.Array<bool>(inner), outer).Named("x");
            x[outer][inner] = Variable.Bernoulli(0.2).ForEach(outer, inner);
            Range switchValues = new Range(2).Named("switchValues");
            Variable<int> switchVar = Variable.Discrete(switchValues, 1.0, 1.0).Named("switchVar");
            var dists = Variable.Array<Bernoulli>(switchValues).Named("dists");
            dists.ObservedValue = new[] { new Bernoulli(0.1), new Bernoulli(0.3) };
            var allTrue = Variable.Array<bool>(outer).Named("allTrue");
            using (Variable.Switch(switchVar))
            {
                var xCopy = Variable.Array(Variable.Array<bool>(inner), outer).Named("xCopy");
                xCopy.SetTo(Variable.Copy(x));
                Variable.ConstrainEqualRandom(xCopy[outer][inner], dists[switchVar]);
                allTrue[outer] = Variable.AllTrue(xCopy[outer]);
                Variable.ConstrainEqualRandom(allTrue[outer], new Bernoulli(0.1));
            }
            var engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(allTrue));
        }

        [Fact]
        public void MarkovModel()
        {
            int T = 10;

            Variable<bool>[] ca = new Variable<bool>[T];
            for (int i = 0; i < T; i++)
            {
                ca[i] = Variable.Bernoulli(0.2).Named("ca" + i);
            }

            Variable<double>[] a = new Variable<double>[T];
            a[0] = Variable.GammaFromMeanAndVariance(6.0, 0.1).Named("a0");
            for (int i = 1; i < T; i++)
            {
                a[i] = Variable<double>.New<double>().Named("a" + i);
            }

            for (int i = 1; i < T; i++)
            {
                using (Variable.If(ca[i]))
                {
                    a[i].SetTo(Variable.Copy(a[i - 1]));
                }
                using (Variable.IfNot(ca[i]))
                {
                    a[i].SetTo(Variable.GammaFromMeanAndVariance(1.0, 1.0));
                }
            }

            InferenceEngine ie = new InferenceEngine();
            //ie.NumberOfIterations = 1000;
            for (int i = 1; i < T; i++)
            {
                Bernoulli caExpected = new Bernoulli(0.2);
                Bernoulli caActual = ie.Infer<Bernoulli>(ca[i]);
                //Console.WriteLine("ca[{0}] = {1}", i, caActual);
                Assert.True(caExpected.MaxDiff(caActual) < 1e-10);
            }
            Gamma prior = Gamma.FromMeanAndVariance(1.0, 1.0);
            Gamma[] aExpected = new Gamma[T];
            aExpected[0] = Gamma.FromMeanAndVariance(6.0, 0.1);
            Bernoulli[] cases = new Bernoulli[2];
            CasesOp.CasesAverageConditional(new Bernoulli(0.2), cases);
            for (int i = 1; i < T; i++)
            {
                aExpected[i] = GateExitOp<double>.ExitAverageConditional<Gamma>(Gamma.Uniform(), cases,
                                                                                new Gamma[] { aExpected[i - 1], prior }, new Gamma());
            }
            for (int i = 0; i < T; i++)
            {
                Gamma aActual = ie.Infer<Gamma>(a[i]);
                Console.WriteLine("a[{0}] = {1} should be {2}", i, aActual, aExpected[i]);
                Assert.True(aExpected[i].MaxDiff(aActual) < 1e-3);
            }
        }

        [Fact]
        public void MarkovModelError()

        {
            Assert.Throws<CompilationFailedException>(() =>
            {

                int T = 3;

                Variable<bool>[] ca = new Variable<bool>[T];
                for (int i = 0; i < T; i++)
                {
                    ca[i] = Variable.Bernoulli(0.2).Named("ca" + i);
                }

                Variable<double>[] a = new Variable<double>[T];
                a[0] = Variable.GammaFromMeanAndVariance(6.0, 0.1).Named("a0");
                for (int i = 1; i < T; i++)
                {
                    a[i] = Variable<double>.New<double>().Named("a" + i);
                }

                for (int i = 1; i < T; i++)
                {
                    using (Variable.If(ca[i]))
                    {
                        Variable.ConstrainEqual(a[i], a[i - 1]);
                    }
                    using (Variable.IfNot(ca[i]))
                    {
                        a[i].SetTo(Variable.GammaFromMeanAndVariance(1.0, 1.0));
                    }
                }

                InferenceEngine ie = new InferenceEngine();
                for (int i = 1; i < T; i++)
                {
                    Console.WriteLine(ie.Infer<Gamma>(a[i]));
                }

            });
        }

        [Fact]
        public void ReplicateWithLocalCondition()
        {
            Variable<bool> c = Variable.Bernoulli(0.2).Named("c");
            Range item = new Range(2).Named("item");
            using (Variable.ForEach(item))
            {
                Variable<bool> b = Variable.Bernoulli(0.1).Named("b");
                using (Variable.If(b))
                {
                    Variable.ConstrainTrue(c);
                }
            }

            InferenceEngine engine = new InferenceEngine();
            Bernoulli cActual = engine.Infer<Bernoulli>(c);
            // p(c,b1,b2) = 0.2^c 0.8^(1-c) prod_k 0.1^b[k] 0.9^(1-b[k]) 0^(b[k](1-c))
            double z = 0.2 + 0.8 * 0.9 * 0.9;
            Bernoulli cExpected = new Bernoulli(0.2 / z);
            Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
            Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
        }

        [Fact]
        public void GatedBernoulli2()
        {
            Beta prior = new Beta(1, 100);
            Variable<double> p = Variable.Random(prior).Named("p");
            double pF = 0.2;
            double cPrior = 0.1;
            Range item = new Range(1).Named("item");
            Variable<bool> c = Variable.Bernoulli(cPrior).Named("c");
            VariableArray<bool> x = Variable.Array<bool>(item).Named("x");
            x.ObservedValue = new bool[] { false };
            using (Variable.If(c))
            {
                x[item] = Variable.Bernoulli(p).ForEach(item);
            }
            using (Variable.IfNot(c))
            {
                x[item] = Variable.Bernoulli(pF).ForEach(item);
            }
            InferenceEngine engine = new InferenceEngine();
            Bernoulli cActual = engine.Infer<Bernoulli>(c);
            double sumCondT = 1.0;
            double sumCondF = 1.0;
            for (int i = 0; i < x.ObservedValue.Length; i++)
            {
                sumCondT *= x.ObservedValue[i] ? prior.GetMean() : (1 - prior.GetMean());
                sumCondF *= x.ObservedValue[i] ? pF : (1 - pF);
            }
            double Z = cPrior * sumCondT + (1 - cPrior) * sumCondF;
            double cPost = cPrior * sumCondT / Z;
            Bernoulli cExpected = new Bernoulli(cPost);
            Console.WriteLine("c = {0} (should be {1})", cActual, cExpected);
            Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
        }

        [Fact]
        public void GatedBernoulli()
        {
            Beta prior = new Beta(1, 100);
            Variable<double> p = Variable.Random(prior).Named("p");
            double pF = 1e-10;
            double cPrior = 0.1;
            Variable<bool> c = Variable.Bernoulli(cPrior).Named("c");
            Variable<bool> x = Variable.Constant(false).Named("x");
            using (Variable.If(c))
            {
                Variable.ConstrainEqual(x, Variable.Bernoulli(p));
            }
            using (Variable.IfNot(c))
            {
                Variable.ConstrainEqual(x, Variable.Bernoulli(pF));
            }
            InferenceEngine engine = new InferenceEngine();
            Bernoulli cActual = engine.Infer<Bernoulli>(c);
            double sumCondT = x.ObservedValue ? prior.GetMean() : (1 - prior.GetMean());
            double sumCondF = x.ObservedValue ? pF : (1 - pF);
            double Z = cPrior * sumCondT + (1 - cPrior) * sumCondF;
            double cPost = cPrior * sumCondT / Z;
            Bernoulli cExpected = new Bernoulli(cPost);
            Console.WriteLine("c = {0} (should be {1})", cActual, cExpected);
            Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
        }

        [Fact]
        public void ForEachIfObservedConstrainEqualRandomTest()
        {
            double pT = 0.1;
            double pF = 0.5;
            Range item = new Range(2);
            VariableArray<bool> c = Variable.Array<bool>(item).Named("c");
            c.ObservedValue = new bool[] { true, false };
            double xPrior = 0.5;
            VariableArray<bool> x = Variable.Array<bool>(item).Named("x");
            x[item] = Variable.Bernoulli(xPrior).ForEach(item);
            using (Variable.ForEach(item))
            {
                using (Variable.If(c[item]))
                {
                    Variable.ConstrainEqualRandom(x[item], new Bernoulli(pT));
                }
                using (Variable.IfNot(c[item]))
                {
                    Variable.ConstrainEqualRandom(x[item], new Bernoulli(pF));
                }
            }
            InferenceEngine engine = new InferenceEngine();
            var xActual = engine.Infer(x);
            var xExpected = new BernoulliArray(item.SizeAsInt);
            xExpected[0] = new Bernoulli(xPrior * pT / (xPrior * pT + (1 - xPrior) * (1 - pT)));
            xExpected[1] = new Bernoulli(xPrior * pF / (xPrior * pF + (1 - xPrior) * (1 - pF)));
            Console.WriteLine("x = {0} (should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
        }

        [Fact]
        public void GatedBernoulliWithRepeat()
        {
            Beta prior = new Beta(3, 2);
            Variable<double> p = Variable.Random(prior).Named("p");
            double pF = 0.2; // 1e-10;
            double cPrior = 0.1;
            Variable<bool> c = Variable.Bernoulli(cPrior).Named("c");
            Variable<bool> x = Variable.Constant(false).Named("x");
            // do not make these counts too large or they will induce multiple fixed points
            int countTrue = 2;
            int countFalse = 3;
            using (Variable.If(c))
            {
                using (Variable.Repeat(countTrue))
                {
                    Variable.ConstrainEqual(x, Variable.Bernoulli(p));
                }
            }
            using (Variable.IfNot(c))
            {
                using (Variable.Repeat(countFalse))
                {
                    Variable.ConstrainEqual(x, Variable.Bernoulli(pF));
                }
            }

            Variable<bool> c2 = Variable.Bernoulli(cPrior).Named("c2");
            Variable<double> p2 = Variable.Random(prior).Named("p2");
            using (Variable.If(c2))
            {
                for (int i = 0; i < countTrue; i++)
                    Variable.ConstrainEqual(x, Variable.Bernoulli(p2));
            }
            using (Variable.IfNot(c2))
            {
                for (int i = 0; i < countFalse; i++)
                    Variable.ConstrainEqual(x, Variable.Bernoulli(pF));
            }

            InferenceEngine engine = new InferenceEngine();
            for (int iter = 0; iter < 2; iter++)
            {
                if (iter == 0) engine.Algorithm = new VariationalMessagePassing();
                Bernoulli cActual = engine.Infer<Bernoulli>(c);
                Bernoulli cExpected = engine.Infer<Bernoulli>(c2);
                Console.WriteLine("c = {0} (should be {1})", cActual, cExpected);
                Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
            }
        }

        [Fact]
        public void GatedRepeatBeta()
        {
            GenericGatedRepeat(new Beta(2, 3), new Beta(4, 5));
        }

        [Fact]
        public void GatedRepeatDirichlet()
        {
            GenericGatedRepeat(new Dirichlet(2, 3), new Dirichlet(4, 5));
        }

        [Fact]
        public void GatedRepeatDiscrete()
        {
            GenericGatedRepeat(new Discrete(2, 3), new Discrete(4, 5));
        }

        [Fact]
        public void GatedRepeatGamma()
        {
            GenericGatedRepeat(new Gamma(2, 3), new Gamma(4, 5));
        }

        [Fact]
        public void GatedRepeatWishart()
        {
            GenericGatedRepeat(new Wishart(2, 3), new Wishart(4, 5));
        }

        [Fact]
        public void GatedRepeatTruncatedGaussian()
        {
            GenericGatedRepeat(new TruncatedGaussian(2, 3, 0, 1), new TruncatedGaussian(4, 5, 0, 1));
        }

        [Fact]
        public void GatedRepeatBernoulliArray()
        {
            GenericGatedRepeat(new BernoulliArray(2, i => new Bernoulli(i * 0.1)), new BernoulliArray(2, i => new Bernoulli(i * 0.2)));
        }

        [Fact]
        public void GatedRepeatBernoulliArrayArray()
        {
            GenericGatedRepeat(new BernoulliArrayArray(2, j => new BernoulliArray(2, i => new Bernoulli(i * j * 0.1))),
                               new BernoulliArrayArray(2, j => new BernoulliArray(2, i => new Bernoulli(i * j * 0.2))));
        }

        private void GenericGatedRepeat<T>(IDistribution<T> prior, Sampleable<T> likelihood)
        {
            Variable<T> param = Variable.Random(prior).Named("param");
            double cPrior = 0.1;
            Variable<bool> c = Variable.Bernoulli(cPrior).Named("c");
            int count = 3;
            using (Variable.If(c))
            {
                using (Variable.Repeat(count))
                {
                    Variable.ConstrainEqualRandom(param, likelihood);
                }
            }

            Variable<bool> c2 = Variable.Bernoulli(cPrior).Named("c2");
            Variable<T> param2 = Variable.Random(prior).Named("param2");
            using (Variable.If(c2))
            {
                for (int i = 0; i < count; i++)
                {
                    Variable.ConstrainEqualRandom(param2, likelihood);
                }
            }

            InferenceEngine engine = new InferenceEngine();
            for (int iter = 0; iter < 2; iter++)
            {
                if (iter == 1) engine.Algorithm = new VariationalMessagePassing();
                Bernoulli cActual = engine.Infer<Bernoulli>(c);
                Bernoulli cExpected = engine.Infer<Bernoulli>(c2);
                Console.WriteLine("c = {0} (should be {1})", cActual, cExpected);
                Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
            }
        }

        [Fact]
        public void GatedBernoulliWithNestedRepeat()
        {
            Beta prior = new Beta(3, 2);
            Variable<double> p = Variable.Random(prior).Named("p");
            double cPrior = 0.1;
            Variable<bool> c = Variable.Bernoulli(cPrior).Named("c");
            Variable<bool> x = Variable.Constant(false).Named("x");
            int count1 = 3;
            int count2 = 2;
            using (Variable.If(c))
            {
                using (Variable.Repeat(count1))
                {
                    using (Variable.Repeat(count2))
                    {
                        Variable.ConstrainEqual(x, Variable.Bernoulli(p));
                    }
                }
            }

            Variable<bool> c2 = Variable.Bernoulli(cPrior).Named("c2");
            Variable<double> p2 = Variable.Random(prior).Named("p2");
            using (Variable.If(c2))
            {
                for (int i = 0; i < count1 * count2; i++)
                    Variable.ConstrainEqual(x, Variable.Bernoulli(p2));
            }

            InferenceEngine engine = new InferenceEngine();
            for (int iter = 0; iter < 2; iter++)
            {
                if (iter == 1) engine.Algorithm = new VariationalMessagePassing();
                Bernoulli cActual = engine.Infer<Bernoulli>(c);
                Bernoulli cExpected = engine.Infer<Bernoulli>(c2);
                Console.WriteLine("c = {0} (should be {1})", cActual, cExpected);
                Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
            }
        }

        [Fact]
        public void GateExitConstraintTest()
        {
            double priorB = 0.1;
            double priorX = 0.2;
            double pXCondT = 0.3;
            double pXCondT2 = 0.6;
            double pXCondF = 0.4;

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<bool> b = Variable.Bernoulli(priorB).Named("b");
            Variable<bool> x = Variable.New<bool>().Named("x");
            using (Variable.If(b))
            {
                x.SetTo(Variable.Bernoulli(pXCondT));
                Variable.ConstrainEqualRandom(x, new Bernoulli(pXCondT2));
            }
            using (Variable.IfNot(b))
            {
                x.SetTo(Variable.Bernoulli(pXCondF));
            }
            Variable.ConstrainEqualRandom(x, new Bernoulli(priorX));
            block.CloseBlock();
            InferenceEngine engine = new InferenceEngine();
            double tolerance = 1e-10;

            // p(x,b) =propto (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x)
            //                [(pT)^x (1-pT)^(1-x) (pT2)^x (1-pT2)^(1-x)]^b [(pF)^x (1-pF)^(1-x)]^(1-b)
            double sumXCondT = priorX * pXCondT * pXCondT2 + (1 - priorX) * (1 - pXCondT) * (1 - pXCondT2);
            double sumXCondF = priorX * pXCondF + (1 - priorX) * (1 - pXCondF);
            double Z = priorB * sumXCondT + (1 - priorB) * sumXCondF;
            double postB = priorB * sumXCondT / Z;
            double postX = priorX * (priorB * pXCondT * pXCondT2 + (1 - priorB) * pXCondF) / Z;

            Bernoulli bActual = engine.Infer<Bernoulli>(b);
            Bernoulli xActual = engine.Infer<Bernoulli>(x);
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("b = {0} (should be {1})", bActual, postB);
            Console.WriteLine("x = {0} (should be {1})", xActual, postX);
            Console.WriteLine("evidence = {0} (should be {1})", System.Math.Exp(evActual), Z);
            Assert.True(System.Math.Abs(bActual.GetProbTrue() - postB) < tolerance);
            Assert.True(System.Math.Abs(xActual.GetProbTrue() - postX) < tolerance);
            Assert.True(MMath.AbsDiff(evActual, System.Math.Log(Z), 1e-10) < tolerance);
        }

        [Fact]
        public void GateExitTest()
        {
            double priorB = 0.1;
            double pXCondT = 0.3;
            double pXCondT2 = 0.6;
            double pXCondF = 0.4;

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<bool> b = Variable.Bernoulli(priorB).Named("b");
            Variable<bool> x = Variable.New<bool>().Named("x");
            using (Variable.If(b))
            {
                x.SetTo(Variable.Bernoulli(pXCondT));
                Variable.ConstrainEqualRandom(x, new Bernoulli(pXCondT2));
            }
            using (Variable.IfNot(b))
            {
                x.SetTo(Variable.Bernoulli(pXCondF));
            }
            block.CloseBlock();
            InferenceEngine engine = new InferenceEngine();
            double tolerance = 1e-10;

            // p(x,b) =propto (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x)
            //                [(pT)^x (1-pT)^(1-x) (pT2)^x (1-pT2)^(1-x)]^b [(pF)^x (1-pF)^(1-x)]^(1-b)
            double sumXCondT = pXCondT * pXCondT2 + (1 - pXCondT) * (1 - pXCondT2);
            double sumXCondF = 1;
            double Z = priorB * sumXCondT + (1 - priorB) * sumXCondF;
            double postB = priorB * sumXCondT / Z;
            double postX = (priorB * pXCondT * pXCondT2 + (1 - priorB) * pXCondF) / Z;

            Bernoulli bActual = engine.Infer<Bernoulli>(b);
            Bernoulli xActual = engine.Infer<Bernoulli>(x);
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("b = {0} (should be {1})", bActual, postB);
            Console.WriteLine("x = {0} (should be {1})", xActual, postX);
            Console.WriteLine("evidence = {0} (should be {1})", System.Math.Exp(evActual), Z);
            Assert.True(System.Math.Abs(bActual.GetProbTrue() - postB) < tolerance);
            Assert.True(System.Math.Abs(xActual.GetProbTrue() - postX) < tolerance);
            Assert.True(MMath.AbsDiff(evActual, System.Math.Log(Z), 1e-10) < tolerance);
        }

        [Fact]
        public void GateExitTest2()
        {
            double priorB = 0.1;
            double pXCondT = 0.3;
            double pXCondF = 0.4;

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<bool> b = Variable.Bernoulli(priorB).Named("b");
            Variable<bool> x = Variable.New<bool>().Named("x");
            using (Variable.If(b))
            {
                x.SetTo(Variable.Bernoulli(pXCondT));
            }
            using (Variable.IfNot(b))
            {
                x.SetTo(Variable.Bernoulli(pXCondF));
            }
            block.CloseBlock();
            InferenceEngine engine = new InferenceEngine();
            double tolerance = 1e-10;

            // p(x,b) =propto (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x)
            //                [(pT)^x (1-pT)^(1-x) (pT2)^x (1-pT2)^(1-x)]^b [(pF)^x (1-pF)^(1-x)]^(1-b)
            double sumXCondT = 1;
            double sumXCondF = 1;
            double Z = priorB * sumXCondT + (1 - priorB) * sumXCondF;
            double postB = priorB * sumXCondT / Z;
            double postX = (priorB * pXCondT + (1 - priorB) * pXCondF) / Z;

            Bernoulli bActual = engine.Infer<Bernoulli>(b);
            Bernoulli xActual = engine.Infer<Bernoulli>(x);
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("b = {0} (should be {1})", bActual, postB);
            Console.WriteLine("x = {0} (should be {1})", xActual, postX);
            Console.WriteLine("evidence = {0} (should be {1})", System.Math.Exp(evActual), Z);
            Assert.True(System.Math.Abs(bActual.GetProbTrue() - postB) < tolerance);
            Assert.True(System.Math.Abs(xActual.GetProbTrue() - postX) < tolerance);
            Assert.True(MMath.AbsDiff(evActual, System.Math.Log(Z), 1e-10) < tolerance);
        }

        [Fact]
        public void GateExitConstrainTrueTest()
        {
            double priorB = 0.1;
            double pXCondT = 0.3;
            double pXCondF = 0.4;

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<bool> b = Variable.Bernoulli(priorB).Named("b");
            Variable<bool> x = Variable.New<bool>().Named("x");
            using (Variable.If(b))
            {
                x.SetTo(Variable.Bernoulli(pXCondT));
            }
            using (Variable.IfNot(b))
            {
                x.SetTo(Variable.Bernoulli(pXCondF));
            }
            Variable.ConstrainTrue(x);
            block.CloseBlock();
            InferenceEngine engine = new InferenceEngine();
            double tolerance = 1e-10;

            double Z = priorB * pXCondT + (1 - priorB) * pXCondF;
            double postB = priorB * pXCondT / Z;
            double postX = (priorB * pXCondT + (1 - priorB) * pXCondF) / Z;

            Bernoulli bActual = engine.Infer<Bernoulli>(b);
            Bernoulli xActual = engine.Infer<Bernoulli>(x);
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("b = {0} (should be {1})", bActual, postB);
            Console.WriteLine("x = {0} (should be {1})", xActual, postX);
            Console.WriteLine("evidence = {0} (should be {1})", System.Math.Exp(evActual), Z);
            Assert.True(System.Math.Abs(bActual.GetProbTrue() - postB) < tolerance);
            Assert.True(System.Math.Abs(xActual.GetProbTrue() - postX) < tolerance);
            Assert.True(MMath.AbsDiff(evActual, System.Math.Log(Z), 1e-10) < tolerance);
        }

        [Fact]
        public void BarrenEvidenceTest()
        {
            double priorB = 0.1;
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<bool> b = Variable.Bernoulli(priorB).Named("b");
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            double tolerance = 1e-10;
            double postB = priorB;
            double Z = 1.0;
            Bernoulli bActual = engine.Infer<Bernoulli>(b);
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("b = {0} (should be {1})", bActual, postB);
            Console.WriteLine("evidence = {0} (should be {1})", System.Math.Exp(evActual), Z);
            Assert.True(System.Math.Abs(bActual.GetProbTrue() - postB) < tolerance);
            Assert.True(MMath.AbsDiff(evActual, System.Math.Log(Z), 1e-10) < tolerance);
        }

        [Fact]
        public void GatedGivenSetToRandom()
        {
            Range item = new Range(1).Named("item");
            VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
            Variable<bool> c = Variable.Bernoulli(0.5).Named("c");
            using (Variable.If(c))
            {
                bools[item] = Variable.Bernoulli(0.1).ForEach(item);
            }
            bools.ObservedValue = new bool[] { true };
            Bernoulli cExpected = new Bernoulli(0.1 / (0.1 + 1));
            InferenceEngine engine = new InferenceEngine();
            Bernoulli cActual = engine.Infer<Bernoulli>(c);
            Console.WriteLine("c = {0} (should be {1})", cActual, cExpected);
            Assert.True(cExpected.MaxDiff(cActual) < 1e-10);

            engine = new InferenceEngine(new VariationalMessagePassing());
            cActual = engine.Infer<Bernoulli>(c);
            Console.WriteLine("c = {0} (should be {1})", cActual, cExpected);
            Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
        }

        [Fact]
        public void GatedGivenSetToRandom2()
        {
            Range item = new Range(1).Named("item");
            VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
            Variable<bool> c = Variable.Bernoulli(0.5).Named("c");
            Variable<bool> c2 = Variable.Bernoulli(0.5).Named("c2");
            using (Variable.If(c))
            {
                using (Variable.If(c2))
                {
                    bools[item] = Variable.Bernoulli(0.1).ForEach(item);
                }
                using (Variable.IfNot(c2))
                {
                    bools[item] = Variable.Bernoulli(0.5).ForEach(item);
                }
            }
            using (Variable.IfNot(c))
            {
                using (Variable.If(c2))
                {
                    bools[item] = Variable.Bernoulli(0.2).ForEach(item);
                }
                using (Variable.IfNot(c2))
                {
                    bools[item] = Variable.Bernoulli(0.5).ForEach(item);
                }
            }
            Variable.ConstrainTrue(bools[item]);
            InferenceEngine engine = new InferenceEngine();
            engine.ModelName = "GatedGivenSetToRandom2";
            double sumT = 0.5 * (0.1 + 0.5);
            double sumF = 0.5 * (0.2 + 0.5);
            Bernoulli cExpected = new Bernoulli(sumT / (sumT + sumF));
            Bernoulli cActual = engine.Infer<Bernoulli>(c);
            Console.WriteLine("c = {0} (should be {1})", cActual, cExpected);
            Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
        }

        [Fact]
        public void GatedGivenSetToRandom3()
        {
            Range item = new Range(1).Named("item");
            VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
            Variable<bool> c = Variable.Bernoulli(0.5).Named("c");
            Variable<double> p1 = Variable.Beta(1, 9).Named("p1");
            Variable<double> p2 = Variable.Beta(2, 8).Named("p2");
            using (Variable.If(c))
            {
                bools[item] = Variable.Bernoulli(p1).ForEach(item);
            }
            using (Variable.IfNot(c))
            {
                bools[item] = Variable.Bernoulli(p2).ForEach(item);
            }
            InferenceEngine engine = new InferenceEngine();
            bools.ObservedValue = new bool[] { true };
            Bernoulli cExpected = new Bernoulli(0.1 / (0.1 + 0.2));
            Bernoulli cActual = engine.Infer<Bernoulli>(c);
            Console.WriteLine("c = {0} (should be {1})", cActual, cExpected);
            Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
        }

        [Fact]
        public void GatedConstrainEqualRandom()
        {
            Range item = new Range(1).Named("item");
            VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
            Variable<bool> c = Variable.Bernoulli(0.5).Named("c");
            using (Variable.If(c))
            {
                Variable.ConstrainEqualRandom(bools[item], new Bernoulli(0.1));
            }
            bools.ObservedValue = new bool[] { true };
            Bernoulli cExpected = new Bernoulli(0.1 / (0.1 + 1));
            InferenceEngine engine = new InferenceEngine();
            Bernoulli cActual = engine.Infer<Bernoulli>(c);
            Console.WriteLine("c = {0} (should be {1})", cActual, cExpected);
            Assert.True(cExpected.MaxDiff(cActual) < 1e-10);

            engine = new InferenceEngine(new VariationalMessagePassing());
            cActual = engine.Infer<Bernoulli>(c);
            Console.WriteLine("c = {0} (should be {1})", cActual, cExpected);
            Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
        }

        [Fact]
        public void GatedArrayReplication()
        {
            Range item = new Range(2).Named("item");
            Range itemRep = new Range(3).Named("itemRep");
            VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
            bools[item] = Variable.Bernoulli(0.3).ForEach(item);
            Variable<bool> c = Variable.Bernoulli(0.2).Named("c");
            using (Variable.If(c))
            {
                using (Variable.ForEach(itemRep))
                {
                    Variable.ConstrainEqualRandom(bools[item], new Bernoulli(0.1));
                }
            }
            double sumCondC = 0.3 * System.Math.Pow(0.1, itemRep.SizeAsInt) + (1 - 0.3) * System.Math.Pow(1 - 0.1, itemRep.SizeAsInt);
            sumCondC = System.Math.Pow(sumCondC, item.SizeAsInt);
            double Z = 0.2 * sumCondC + (1 - 0.2);
            Bernoulli cExpected = new Bernoulli(0.2 * sumCondC / Z);
            InferenceEngine engine = new InferenceEngine();
            Bernoulli cActual = engine.Infer<Bernoulli>(c);
            Console.WriteLine("c = {0} (should be {1})", cActual, cExpected);
            Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
        }

        [Fact]
        public void GatedArrayReplication2()
        {
            Range item = new Range(2).Named("item");
            Range itemRep = new Range(3).Named("itemRep");
            VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
            double bPrior = 0.2;
            double cPrior = 0.1;
            double bCondT = 0.3;
            double bCondF = 0.4;
            bools[item] = Variable.Bernoulli(bPrior).ForEach(item);
            Variable<bool> c = Variable.Bernoulli(cPrior).Named("c");
            using (Variable.If(c))
            {
                using (Variable.ForEach(itemRep))
                {
                    Variable.ConstrainEqualRandom(bools[item], new Bernoulli(bCondT));
                }
            }
            using (Variable.IfNot(c))
            {
                using (Variable.ForEach(itemRep))
                {
                    Variable.ConstrainEqualRandom(bools[item], new Bernoulli(bCondF));
                }
            }
            double sumCondT = bPrior * System.Math.Pow(bCondT, itemRep.SizeAsInt) + (1 - bPrior) * System.Math.Pow(1 - bCondT, itemRep.SizeAsInt);
            sumCondT = System.Math.Pow(sumCondT, item.SizeAsInt);
            double sumCondF = bPrior * System.Math.Pow(bCondF, itemRep.SizeAsInt) + (1 - bPrior) * System.Math.Pow(1 - bCondF, itemRep.SizeAsInt);
            sumCondF = System.Math.Pow(sumCondF, item.SizeAsInt);
            double Z = cPrior * sumCondT + (1 - cPrior) * sumCondF;
            Bernoulli cExpected = new Bernoulli(cPrior * sumCondT / Z);
            InferenceEngine engine = new InferenceEngine();
            Bernoulli cActual = engine.Infer<Bernoulli>(c);
            Console.WriteLine("c = {0} (should be {1})", cActual, cExpected);
            Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
        }

        [Fact]
        public void GatedArrayReplication2D()
        {
            Range item = new Range(2).Named("item");
            Range item2 = new Range(3).Named("item2");
            Range itemRep = new Range(3).Named("itemRep");
            VariableArray2D<bool> bools = Variable.Array<bool>(item, item2).Named("bools");
            bools[item, item2] = Variable.Bernoulli(0.3).ForEach(item, item2);
            Variable<bool> c = Variable.Bernoulli(0.2).Named("c");
            using (Variable.If(c))
            {
                using (Variable.ForEach(itemRep))
                {
                    Variable.ConstrainEqualRandom(bools[item, item2], new Bernoulli(0.1));
                }
            }
            double sumCondC = 0.3 * System.Math.Pow(0.1, itemRep.SizeAsInt) + (1 - 0.3) * System.Math.Pow(1 - 0.1, itemRep.SizeAsInt);
            sumCondC = System.Math.Pow(sumCondC, item.SizeAsInt * item2.SizeAsInt);
            double Z = 0.2 * sumCondC + (1 - 0.2);
            Bernoulli cExpected = new Bernoulli(0.2 * sumCondC / Z);
            InferenceEngine engine = new InferenceEngine();
            Bernoulli cActual = engine.Infer<Bernoulli>(c);
            Console.WriteLine("c = {0} (should be {1})", cActual, cExpected);
            Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
        }

        [Fact]
        public void GatedArrayReplication2D2()
        {
            Range item = new Range(2).Named("item");
            Range item2 = new Range(3).Named("item2");
            VariableArray<int> itemRepN = Variable.Constant(new int[] { 2, 3 }, item).Named("itemRepN");
            Range itemRep = new Range(itemRepN[item]).Named("itemRep");
            VariableArray2D<bool> bools = Variable.Array<bool>(item, item2).Named("bools");
            bools[item, item2] = Variable.Bernoulli(0.3).ForEach(item, item2);
            Variable<bool> c = Variable.Bernoulli(0.2).Named("c");
            using (Variable.If(c))
            {
                using (Variable.ForEach(item))
                {
                    using (Variable.ForEach(itemRep))
                    {
                        Variable.ConstrainEqualRandom(bools[item, item2], new Bernoulli(0.1));
                    }
                }
            }
            double sumCondC = 1.0;
            for (int i = 0; i < 2; i++)
            {
                sumCondC *= 0.3 * System.Math.Pow(0.1, itemRepN.ObservedValue[i]) + (1 - 0.3) * System.Math.Pow(1 - 0.1, itemRepN.ObservedValue[i]);
            }
            sumCondC = System.Math.Pow(sumCondC, item2.SizeAsInt);
            double Z = 0.2 * sumCondC + (1 - 0.2);
            Bernoulli cExpected = new Bernoulli(0.2 * sumCondC / Z);
            InferenceEngine engine = new InferenceEngine();
            Bernoulli cActual = engine.Infer<Bernoulli>(c);
            Console.WriteLine("c = {0} (should be {1})", cActual, cExpected);
            Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
        }

        [Fact]
        public void GatedJaggedArrayReplication()
        {
            Range item = new Range(2).Named("item");
            VariableArray<int> sizes = Variable.Constant(new int[] { 2, 3 }, item).Named("sizes");
            Range item2 = new Range(sizes[item]).Named("item2");
            Range itemRep = new Range(3).Named("itemRep");
            var bools = Variable.Array<bool>(Variable.Array<bool>(item2), item).Named("bools");
            bools[item][item2] = Variable.Bernoulli(0.3).ForEach(item, item2);
            Variable<bool> c = Variable.Bernoulli(0.2).Named("c");
            using (Variable.If(c))
            {
                using (Variable.ForEach(itemRep))
                {
                    Variable.ConstrainEqualRandom(bools[item][item2], new Bernoulli(0.1));
                }
            }
            double sumCondC = 0.3 * System.Math.Pow(0.1, itemRep.SizeAsInt) + (1 - 0.3) * System.Math.Pow(1 - 0.1, itemRep.SizeAsInt);
            sumCondC = System.Math.Pow(sumCondC, sizes.ObservedValue[0] + sizes.ObservedValue[1]);
            double Z = 0.2 * sumCondC + (1 - 0.2);
            Bernoulli cExpected = new Bernoulli(0.2 * sumCondC / Z);
            InferenceEngine engine = new InferenceEngine();
            Bernoulli cActual = engine.Infer<Bernoulli>(c);
            Console.WriteLine("c = {0} (should be {1})", cActual, cExpected);
            Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
        }

        // This test correctly fails to compile.
#if false
        [Fact]
        public void SwitchDefinitionError()
        {
            try {
                Range item = new Range(3).Named("item");
                VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
                Variable<int> index = Variable.Discrete(0.1, 0.5, 0.4).Named("index");
                using (Variable.Switch(index)) {
                    bools[index] = Variable.Bernoulli(0.3);
                }
                InferenceEngine engine = new InferenceEngine();
                DistributionArray<Bernoulli> boolsActual = engine.Infer<DistributionArray<Bernoulli>>(bools);
                Console.WriteLine("bools = {0}", boolsActual);
                Assert.True(false, "Did not throw exception");
            } catch (ArgumentException ex) { Console.WriteLine("Correctly failed with exception: " + ex); }
        }
#endif

        [Fact]
        public void SwitchInSwitchTest()
        {
            SwitchInSwitch(new ExpectationPropagation());
        }

        [Fact]
        public void GibbsSwitchInSwitchTest()
        {
            SwitchInSwitch(new GibbsSampling());
        }

        private void SwitchInSwitch(IAlgorithm algorithm)
        {
            Range item = new Range(3).Named("item");
            double[] indexPrior = { 0.1, 0.5, 0.4 };
            Variable<int> index = Variable.Discrete(item, indexPrior).Named("index");
            Range item2 = new Range(3).Named("item2");
            VariableArray<bool> bools = Variable.Array<bool>(item2).Named("bools");
            VariableArray<double> bPrior = Variable.Constant(new double[] { 0.3, 0.6, 0.7 }, item2).Named("bPrior");
            using (Variable.ForEach(item2))
            {
                bools[item2] = Variable.Bernoulli(bPrior[item2]);
            }
            VariableArray<int> indices = Variable.Array<int>(item).Named("indices");
            VariableArray<Vector> priors =
                Variable.Constant(new Vector[] { Vector.FromArray(1.0, 0, 0), Vector.FromArray(0, 1, 0), Vector.FromArray(0, 0, 1) }, item).Named("priors");
            indices[item] = Variable.Discrete(priors[item]);
            using (Variable.Switch(index))
            {
                Variable<int> index2 = indices[index].Attrib(new ValueRange(item2));
                using (Variable.Switch(index2))
                {
                    Variable.ConstrainTrue(bools[index2]);
                }
            }
            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = algorithm;
            double tolerance = 1e-10;
            if (algorithm is GibbsSampling)
            {
                Rand.Restart(12347);
                tolerance = 1e-2;
                engine.NumberOfIterations = 100000;
                engine.ShowProgress = false;
            }
            //indices.InitialiseTo(Distribution<int>.Array(Util.ArrayInit(item.SizeAsInt, i => Discrete.PointMass(i, item2.SizeAsInt))));
            // p(index,bools) = p0(index) p0(bools) delta(bools[index]=T)
            double[] boolsPost = new double[item2.SizeAsInt];
            double[] indexPost = new double[item.SizeAsInt];
            double z = 0;
            for (int j = 0; j < indexPrior.Length; j++)
            {
                z += indexPrior[j] * bPrior.ObservedValue[j];
            }
            for (int i = 0; i < indexPrior.Length; i++)
            {
                indexPost[i] = indexPrior[i] * bPrior.ObservedValue[i] / z;
            }
            Discrete indexExpected = new Discrete(indexPost);
            for (int i = 0; i < boolsPost.Length; i++)
            {
                for (int j = 0; j < indexPrior.Length; j++)
                {
                    boolsPost[i] += bPrior.ObservedValue[i] * indexPrior[j] * (i == j ? 1.0 : bPrior.ObservedValue[j]);
                }
                boolsPost[i] /= z;
            }
            Bernoulli[] boolsExpectedArray = new Bernoulli[boolsPost.Length];
            for (int i = 0; i < boolsExpectedArray.Length; i++)
            {
                boolsExpectedArray[i] = new Bernoulli(boolsPost[i]);
            }
            Diffable boolsExpected = Distribution<bool>.Array(boolsExpectedArray);
            object boolsActual = engine.Infer(bools);
            Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsActual, " should be ", boolsExpected));
            Discrete indexActual = engine.Infer<Discrete>(index);
            Console.WriteLine("index = {0} should be {1}", indexActual, indexExpected);
            Assert.True(boolsExpected.MaxDiff(boolsActual) < tolerance);
            Assert.True(indexExpected.MaxDiff(indexActual) < tolerance);
        }

        [Fact]
        public void SwitchRandomSwitchDeterministicTest()
        {
            Range item = new Range(3).Named("item");
            Variable<int> index = Variable.Discrete(item, 0.1, 0.5, 0.4).Named("index");
            Range item2 = new Range(3).Named("item2");
            VariableArray<bool> bools = Variable.Array<bool>(item2).Named("bools");
            using (Variable.ForEach(item2))
            {
                bools[item2] = Variable.Bernoulli(0.3);
            }
            VariableArray<int> indices = Variable.Constant(new int[] { 0, 1, 2 }, item).Named("indices");
            using (Variable.Switch(index))
            {
                // index2 is a deterministic function of index.
                Variable<int> index2 = indices[index].Attrib(new ValueRange(item2));
                using (Variable.Switch(index2))
                {
                    Variable.ConstrainTrue(bools[index2]);
                }
            }
            InferenceEngine engine = new InferenceEngine();
            double[] boolsPost = new double[item.SizeAsInt];
            boolsPost[0] = 0.1 + 0.5 * 0.3 + 0.4 * 0.3;
            boolsPost[1] = 0.1 * 0.3 + 0.5 + 0.4 * 0.3;
            boolsPost[2] = 0.1 * 0.3 + 0.5 * 0.3 + 0.4;
            Bernoulli[] boolsExpectedArray = new Bernoulli[boolsPost.Length];
            for (int i = 0; i < boolsExpectedArray.Length; i++)
            {
                boolsExpectedArray[i] = new Bernoulli(boolsPost[i]);
            }
            Diffable boolsExpected = Distribution<bool>.Array(boolsExpectedArray);
            object boolsActual = engine.Infer(bools);
            Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsActual, " should be ", boolsExpected));
            Assert.True(boolsExpected.MaxDiff(boolsActual) < 1e-10);
        }

        [Fact]
        public void SwitchRandomSwitchDeterministicTest2()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Range item = new Range(3).Named("item");
            Variable<int> index = Variable.Discrete(item, 0.1, 0.5, 0.4).Named("index");
            Range item2 = new Range(3).Named("item2");
            VariableArray<int> indices = Variable.Constant(new int[] { 0, 1, 2 }, item).Named("indices");
            Variable<bool> b = Variable.Bernoulli(0.3).Named("b");
            using (Variable.Switch(index))
            {
                // index2 is a deterministic function of index.
                Variable<int> index2 = indices[index].Attrib(new ValueRange(item2));
                using (Variable.Switch(index2))
                {
                    Variable.ConstrainEqualRandom(b, new Bernoulli(0.4));
                }
            }
            block.CloseBlock();
            InferenceEngine engine = new InferenceEngine();
            Bernoulli bActual = engine.Infer<Bernoulli>(b);
            double z = 0.3 * 0.4 + (1 - 0.3) * (1 - 0.4);
            Bernoulli bExpected = new Bernoulli(0.3 * 0.4 / z);
            Console.WriteLine("b = {0} should be {1}", bActual, bExpected);
            Assert.True(bExpected.MaxDiff(bActual) < 1e-10);
            double evExpected = System.Math.Log(z);
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-10);
        }

        [Fact]
        public void SwitchRandomIfDeterministicTest()
        {
            Range item = new Range(3).Named("item");
            Variable<int> index = Variable.Discrete(item, 0.1, 0.5, 0.4).Named("index");
#if true
            VariableArray<bool> c = Variable.Constant(new bool[] { false, true, false }, item).Named("c");
#else
            VariableArray<bool> c = Variable.Array<bool>(item).Named("c");
            var priors = Variable.Constant(new double[] { 0.0, 1.0, 0.0 }, item).Named("priors");
            c[item] = Variable.Bernoulli(priors[item]);
#endif
            Variable<bool> b = Variable.Bernoulli(0.3).Named("b");
            using (Variable.Switch(index))
            {
                using (Variable.If(c[index]))
                {
                    Variable.ConstrainTrue(b);
                }
            }
            // p(b|i) = 0.3^b 0.7^(1-b) 0^(1-b)(i==1)
            // Z = 0.3 + 0.7*0.5
            InferenceEngine engine = new InferenceEngine();
            //engine.Compiler.GenerateInMemory = false;
            Bernoulli bActual = engine.Infer<Bernoulli>(b);
            double Z = 0.3 + 0.7 * 0.5;
            Bernoulli bExpected = new Bernoulli(0.3 / Z);
            Console.WriteLine("b = {0} (should be {1})", bActual, bExpected);
            Discrete indexActual = engine.Infer<Discrete>(index);
            Discrete indexExpected = new Discrete(0.1 / Z, 0.5 * 0.3 / Z, 0.4 / Z);
            Console.WriteLine("index = {0} (should be {1})", indexActual, indexExpected);
            Assert.True(bExpected.MaxDiff(bActual) < 1e-10);
            Assert.True(indexExpected.MaxDiff(indexActual) < 1e-10);
        }

        [Fact]
        public void SwitchConstantConditionInLoopTest()
        {
            Range item = new Range(3).Named("item");
            Variable<int> index = Variable.Discrete(item, 0.1, 0.5, 0.4).Named("index");
            Range item2 = new Range(3).Named("item2");
            VariableArray<bool> bools = Variable.Array<bool>(item2).Named("bools");
            using (Variable.ForEach(item2))
            {
                bools[item2] = Variable.Bernoulli(0.3);
            }
            VariableArray<int> indices = Variable.Constant(new int[] { 0, 1, 2 }, item).Named("indices");
            using (Variable.ForEach(item))
            {
                // index2 is a deterministic function of index.
                // however, index2 is still a random value.
                Variable<int> index2 = indices[item].Attrib(new ValueRange(item2));
                using (Variable.Switch(index2))
                {
                    Variable.ConstrainTrue(bools[index2]);
                }
            }
            InferenceEngine engine = new InferenceEngine();
            double[] boolsPost = new double[item.SizeAsInt];
            boolsPost[0] = 1.0;
            boolsPost[1] = 1.0;
            boolsPost[2] = 1.0;
            Bernoulli[] boolsExpectedArray = new Bernoulli[boolsPost.Length];
            for (int i = 0; i < boolsExpectedArray.Length; i++)
            {
                boolsExpectedArray[i] = new Bernoulli(boolsPost[i]);
            }
            Diffable boolsExpected = Distribution<bool>.Array(boolsExpectedArray);
            object boolsActual = engine.Infer(bools);
            Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsActual, " should be ", boolsExpected));
            Assert.True(boolsExpected.MaxDiff(boolsActual) < 1e-10);
        }

        [Fact]
        public void SwitchTest()
        {
            Range item = new Range(3).Named("item");
            VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
            double[] indexPrior = new double[] { 0.1, 0.5, 0.4 };
            Variable<int> index = Variable.Discrete(item, indexPrior).Named("index");
            double bPrior = 0.3;
            using (Variable.ForEach(item))
            {
                bools[item] = Variable.Bernoulli(bPrior);
            }
            double[] bData = new double[] { 0.2, 0.6, 0.7 };
            //double[] bData = new double[] { 0.0, 1.0, 0.0 };
            VariableArray<Bernoulli> data = Variable.Constant(new Bernoulli[] { new Bernoulli(bData[0]), new Bernoulli(bData[1]), new Bernoulli(bData[2]) }, item).Named("data");
            using (Variable.Switch(index))
            {
                Variable.ConstrainEqualRandom(bools[index], data[index]);
            }
            InferenceEngine engine = new InferenceEngine();
            // model is p(bools,index) = p0(bools) p0(index) prod_k (bData[k]^bools[k] (1-bData[k])^(1-bools[k]))^delta(index=k)
            // z_k = bPrior*bData[k] + (1-bPrior)*(1-bData[k])
            // z = sum_k p0(index=k) z_k
            // p(bools[i]=1) = (p0(index=i) p0(bools[i]=1) bData[i] + 
            //       sum_(j!=i) p0(index=j) p0(bools[i]=1) z_j)/z
            double[] zk = new double[indexPrior.Length];
            double z = 0.0;
            for (int i = 0; i < indexPrior.Length; i++)
            {
                zk[i] = bPrior * bData[i] + (1 - bPrior) * (1 - bData[i]);
                z += indexPrior[i] * zk[i];
            }
            Discrete indexExpected = Discrete.Uniform(3);
            Vector probs = indexExpected.GetWorkspace();
            for (int i = 0; i < indexPrior.Length; i++)
            {
                probs[i] = indexPrior[i] * zk[i] / z;
            }
            indexExpected.SetProbs(probs);
            object indexActual = engine.Infer(index);
            Console.WriteLine("index = {0} should be {1}", indexActual, indexExpected);
            Assert.True(indexExpected.MaxDiff(indexActual) < 1e-4);
            double[] boolsPost = new double[item.SizeAsInt];
            for (int i = 0; i < boolsPost.Length; i++)
            {
                boolsPost[i] = bPrior * (indexPrior[i] * bData[i] + (z - indexPrior[i] * zk[i])) / z;
            }
            Bernoulli[] boolsExpectedArray = new Bernoulli[boolsPost.Length];
            for (int i = 0; i < boolsExpectedArray.Length; i++)
            {
                boolsExpectedArray[i] = new Bernoulli(boolsPost[i]);
            }
            Diffable boolsExpected = Distribution<bool>.Array(boolsExpectedArray);
            object boolsActual = engine.Infer(bools);
            Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsActual, " should be ", boolsExpected));
            Assert.True(boolsExpected.MaxDiff(boolsActual) < 1e-10);
        }

        [Fact]
        public void SwitchTest2()
        {
            Range item = new Range(3).Named("item");
            VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
            double[] indexPrior = new double[] { 0.1, 0.5, 0.4 };
            Variable<int> index = Variable.Discrete(item, indexPrior).Named("index");
            double bPrior = 0.3;
            using (Variable.ForEach(item))
            {
                bools[item] = Variable.Bernoulli(bPrior);
            }
            double[] bData = new double[] { 0.2, 0.6, 0.7 };
            VariableArray<Bernoulli> data = Variable.Constant(new Bernoulli[] { new Bernoulli(bData[0]), new Bernoulli(bData[1]), new Bernoulli(bData[2]) }, item).Named("data");
            double[] bData2 = new double[] { 0.8, 0.9, 0.2 };
            VariableArray<Bernoulli> data2 =
                Variable.Constant(new Bernoulli[] { new Bernoulli(bData2[0]), new Bernoulli(bData2[1]), new Bernoulli(bData2[2]) }, item).Named("data2");
            using (Variable.Switch(index))
            {
                Variable.ConstrainEqualRandom(bools[index], data[index]);
                Variable.ConstrainEqualRandom(bools[index], data2[index]);
            }
            InferenceEngine engine = new InferenceEngine();
            // model is p(bools,index) = p0(bools) p0(index) prod_k ((bData[k] bData2[k])^bools[k] ((1-bData)(1-bData2))^(1-bools[k])^delta(index=k)
            // z_k = bPrior bData[k] bData2[k] + (1-bPrior)(1-bData[k])(1-bData2[k])
            // z = sum_k p0(index=k) z_k
            // p(bools[i]=1) = p0(bools[i]=1) (p0(index=i) bData[i] bData2[i] + sum_(j!=i) p0(index=j) z_j)/z
            double[] zk = new double[indexPrior.Length];
            double z = 0.0;
            for (int i = 0; i < indexPrior.Length; i++)
            {
                zk[i] = bPrior * bData[i] * bData2[i] + (1 - bPrior) * (1 - bData[i]) * (1 - bData2[i]);
                z += indexPrior[i] * zk[i];
            }
            double[] boolsPost = new double[item.SizeAsInt];
            for (int i = 0; i < boolsPost.Length; i++)
            {
                boolsPost[i] = bPrior * (indexPrior[i] * bData[i] * bData2[i] + (z - indexPrior[i] * zk[i])) / z;
            }
            Bernoulli[] boolsExpectedArray = new Bernoulli[boolsPost.Length];
            for (int i = 0; i < boolsExpectedArray.Length; i++)
            {
                boolsExpectedArray[i] = new Bernoulli(boolsPost[i]);
            }
            Diffable boolsExpected = Distribution<bool>.Array(boolsExpectedArray);
            object boolsActual = engine.Infer(bools);
            Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsActual, " should be ", boolsExpected));
            Assert.True(boolsExpected.MaxDiff(boolsActual) < 1e-10);
        }

        [Fact]
        public void IfSwitchTest()
        {
            Range item = new Range(3).Named("item");
            VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
            double[] indexPrior = new double[] { 0.1, 0.5, 0.4 };
            Variable<int> index = Variable.Discrete(item, indexPrior).Named("index");
            double bPrior = 0.3;
            using (Variable.ForEach(item))
            {
                bools[item] = Variable.Bernoulli(bPrior);
            }
            double[] bData = new double[] { 0.2, 0.6, 0.7 };
            //double[] bData = new double[] { 0.0, 1.0, 0.0 };
            VariableArray<Bernoulli> data = Variable.Constant(new Bernoulli[] { new Bernoulli(bData[0]), new Bernoulli(bData[1]), new Bernoulli(bData[2]) }, item).Named("data");
            using (var fb = Variable.ForEach(item))
            {
                using (Variable.If(index == fb.Index))
                {
                    Variable.ConstrainEqualRandom(bools[index], data[index]);
                }
            }
            InferenceEngine engine = new InferenceEngine();
            // model is p(bools,index) = p0(bools) p0(index) prod_k (bData[k]^bools[k] (1-bData[k])^(1-bools[k]))^delta(index=k)
            // z_k = bPrior*bData[k] + (1-bPrior)*(1-bData[k])
            // z = sum_k p0(index=k) z_k
            // p(bools[i]=1) = (p0(index=i) p0(bools[i]=1) bData[i] + 
            //       sum_(j!=i) p0(index=j) p0(bools[i]=1) z_j)/z
            double[] zk = new double[indexPrior.Length];
            double z = 0.0;
            for (int i = 0; i < indexPrior.Length; i++)
            {
                zk[i] = bPrior * bData[i] + (1 - bPrior) * (1 - bData[i]);
                z += indexPrior[i] * zk[i];
            }
            Discrete indexExpected = Discrete.Uniform(3);
            Vector probs = indexExpected.GetWorkspace();
            for (int i = 0; i < indexPrior.Length; i++)
            {
                probs[i] = indexPrior[i] * zk[i] / z;
            }
            indexExpected.SetProbs(probs);
            object indexActual = engine.Infer(index);
            Console.WriteLine("index = {0} should be {1}", indexActual, indexExpected);
            Assert.True(indexExpected.MaxDiff(indexActual) < 1e-4);
            double[] boolsPost = new double[item.SizeAsInt];
            for (int i = 0; i < boolsPost.Length; i++)
            {
                boolsPost[i] = bPrior * (indexPrior[i] * bData[i] + (z - indexPrior[i] * zk[i])) / z;
            }
            Bernoulli[] boolsExpectedArray = new Bernoulli[boolsPost.Length];
            for (int i = 0; i < boolsExpectedArray.Length; i++)
            {
                boolsExpectedArray[i] = new Bernoulli(boolsPost[i]);
            }
            Diffable boolsExpected = Distribution<bool>.Array(boolsExpectedArray);
            object boolsActual = engine.Infer(bools);
            Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsActual, " should be ", boolsExpected));
            Assert.True(boolsExpected.MaxDiff(boolsActual) < 1e-10);
        }

        [Fact]
        public void SwitchRedundantForEachError()

        {
            Assert.Throws<InvalidOperationException>(() =>
            {

                Range item = new Range(3).Named("item");
                VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
                Variable<int> index = Variable.Discrete(item, 0.1, 0.5, 0.4).Named("index");
                using (Variable.ForEach(item))
                {
                    bools[item] = Variable.Bernoulli(0.3);
                    using (Variable.Switch(index))
                    {
                        Variable.ConstrainTrue(bools[index]);
                    }
                }
                InferenceEngine engine = new InferenceEngine();
                // p(bools) sum_k p(index=k) delta(bools[k]=1)
                double[] b = new double[3];
                b[0] = 0.1 + 0.5 * 0.3 + 0.4 * 0.3;
                b[1] = 0.1 * 0.3 + 0.5 + 0.4 * 0.3;
                b[2] = 0.1 * 0.3 + 0.5 * 0.3 + 0.4;
                Bernoulli[] boolsExpectedArray = new Bernoulli[3];
                for (int i = 0; i < boolsExpectedArray.Length; i++)
                {
                    boolsExpectedArray[i] = new Bernoulli(b[i]);
                }
                IDistribution<bool[]> boolsExpected = Distribution<bool>.Array(boolsExpectedArray);
                DistributionArray<Bernoulli> boolsActual = engine.Infer<DistributionArray<Bernoulli>>(bools);
                Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsActual, " should be ", boolsExpected));
                Assert.True(boolsExpected.MaxDiff(boolsActual) < 1e-10);

            });
        }

        [Fact]
        public void SwitchRedundantForEachError2()

        {
            Assert.Throws<InvalidOperationException>(() =>
            {

                Range item = new Range(3).Named("item");
                VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
                Variable<int> index = Variable.Discrete(item, 0.1, 0.5, 0.4).Named("index");
                using (Variable.ForEach(item))
                {
                    bools[item] = Variable.Bernoulli(0.3);
                }
                using (Variable.Switch(index))
                {
                    using (Variable.ForEach(item))
                    {
                        Variable.ConstrainTrue(bools[item]);
                    }
                }
                InferenceEngine engine = new InferenceEngine();
                // p(bools) sum_k p(index=k) delta(bools[k]=1)
                double[] b = new double[3];
                b[0] = 0.1 + 0.5 * 0.3 + 0.4 * 0.3;
                b[1] = 0.1 * 0.3 + 0.5 + 0.4 * 0.3;
                b[2] = 0.1 * 0.3 + 0.5 * 0.3 + 0.4;
                Bernoulli[] boolsExpectedArray = new Bernoulli[3];
                for (int i = 0; i < boolsExpectedArray.Length; i++)
                {
                    boolsExpectedArray[i] = new Bernoulli(b[i]);
                }
                IDistribution<bool[]> boolsExpected = Distribution<bool>.Array(boolsExpectedArray);
                DistributionArray<Bernoulli> boolsActual = engine.Infer<DistributionArray<Bernoulli>>(bools);
                Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsActual, " should be ", boolsExpected));
                Assert.True(boolsExpected.MaxDiff(boolsActual) < 1e-10);

            });
        }

        // Fails with error "Range 'item' is already open in a ForEach or Switch block"
        [Fact]
        [Trait("Category", "OpenBug")]
        public void ForEachSwitchTest2()
        {
            Range item = new Range(2).Named("item");
            VariableArray<bool> x = Variable.Array<bool>(item).Named("x");
            VariableArray<bool> y = Variable.Array<bool>(item).Named("y");
            using (Variable.ForEach(item))
            {
                x[item] = Variable.Bernoulli(0.1);
            }
            using (Variable.ForEach(item))
            {
                Variable<int> index = Variable.DiscreteUniform(item).Named("index");
                using (Variable.Switch(index))
                {
                    y[item] = !x[index];
                }
            }

            InferenceEngine engine = new InferenceEngine();
            Bernoulli[] yExpectedArray = new Bernoulli[2];
            for (int i = 0; i < yExpectedArray.Length; i++)
            {
                yExpectedArray[i] = new Bernoulli(0.9);
            }
            object yActual = engine.Infer(y);
            IDistribution<bool[]> yExpected = Distribution<bool>.Array(yExpectedArray);
            Assert.True(yExpected.MaxDiff(yActual) < 1e-10);
        }

        [Fact]
        public void ForEachSwitchTest3()
        {
            Range item = new Range(2).Named("item");
            Range val = new Range(2).Named("val");
            VariableArray<bool> x = Variable.Array<bool>(val).Named("x");
            VariableArray<bool> y = Variable.Array<bool>(item).Named("y");
            var indices = Variable.Observed(new int[] { 0, 0 }, item);
            indices.SetValueRange(val);
            using (Variable.ForEach(val))
            {
                x[val] = Variable.Bernoulli(0.1);
            }
            using (Variable.ForEach(item))
            {
                Variable<int> index = Variable.Copy(indices[item]).Named("index");
                using (Variable.Switch(index))
                {
                    y[item] = !x[index];
                }
            }

            InferenceEngine engine = new InferenceEngine();
            Bernoulli[] yExpectedArray = new Bernoulli[2];
            for (int i = 0; i < yExpectedArray.Length; i++)
            {
                yExpectedArray[i] = new Bernoulli(0.9);
            }
            object yActual = engine.Infer(y);
            IDistribution<bool[]> yExpected = Distribution<bool>.Array(yExpectedArray);
            Assert.True(yExpected.MaxDiff(yActual) < 1e-10);
        }

        // Check for exception on invalid model.
        [Fact]
        public void SwitchUsesRangeError()

        {
            Assert.Throws<ArgumentException>(() =>
            {

                Range item = new Range(3).Named("item");
                VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
                Variable<int> index = Variable.Discrete(item, 0.1, 0.5, 0.4).Named("index");
                using (Variable.ForEach(item))
                {
                    bools[item] = Variable.Bernoulli(0.3);
                }
                using (Variable.Switch(index))
                {
                    Variable.ConstrainTrue(bools[item]);
                }
                InferenceEngine engine = new InferenceEngine();
                DistributionArray<Bernoulli> boolsActual = engine.Infer<DistributionArray<Bernoulli>>(bools);

            });
        }

        [Fact]
        public void CaseExitDeepJaggedArrayTest2()
        {
            Discrete aPriorDist = new Discrete(0.3, 0.7);
            Range outer = new Range(3).Named("outer");
#if false
            VariableArray<int> middleSizes = Variable.Constant(new int[] { 2, 2, 2 }, outer).Named("middleSizes");
            Range middle = new Range(middleSizes[outer]).Named("inner");
#else
            Range middle = new Range(2).Named("middle");
#endif
#if false
            var innerSizes = Variable.Constant(new int[][] { new int[] { 2, 2 }, new int[] { 2, 2 }, new int[] { 2, 3 } }, outer, middle).Named("innerSizes");
            Range inner = new Range(innerSizes[outer][middle]).Named("inner");
#else
            Range inner = new Range(2).Named("inner");
#endif
            Discrete[][] aPriorArray = new Discrete[outer.SizeAsInt][];
            for (int i = 0; i < aPriorArray.Length; i++)
            {
                aPriorArray[i] = new Discrete[middle.SizeAsInt];
                for (int j = 0; j < aPriorArray[i].Length; j++)
                {
                    aPriorArray[i][j] = aPriorDist;
                }
            }
            var aPrior = Variable.Constant(aPriorArray, outer, middle).Named("aPrior");
            var a = Variable.Array(Variable.Array(Variable.Array<int>(inner), middle), outer).Named("a");
            double[] priorIndex = new double[] { 0.4, 0.6 };
            Range item = new Range(1).Named("item");
            VariableArray<int> index = Variable.Array<int>(item).Named("index");
            index[item] = Variable.Discrete(middle, priorIndex).ForEach(item);
            using (Variable.ForEach(outer))
            {
                using (Variable.ForEach(middle))
                {
                    using (Variable.ForEach(inner))
                    {
                        a[outer][middle][inner] = Variable.Random<int, Discrete>(aPrior[outer][middle]);
                    }
                }
            }
            var b = Variable.Array(Variable.Array(Variable.Array<int>(inner), item), outer).Named("b");
            using (Variable.ForEach(item))
            {
                for (int c = 0; c < middle.SizeAsInt; c++)
                {
                    using (Variable.Case(index[item], c))
                    {
                        using (Variable.ForEach(outer))
                        {
                            b[outer][item][inner] = Variable.Copy(a[outer][index[item]][inner]);
                        }
                    }
                }
            }

            InferenceEngine engine = new InferenceEngine();
            Discrete[][][] bExpectedArray = new Discrete[outer.SizeAsInt][][];
            for (int i = 0; i < bExpectedArray.Length; i++)
            {
                bExpectedArray[i] = new Discrete[item.SizeAsInt][];
                for (int j = 0; j < bExpectedArray[i].Length; j++)
                {
                    bExpectedArray[i][j] = new Discrete[inner.SizeAsInt];
                    for (int k = 0; k < bExpectedArray[i][j].Length; k++)
                    {
                        bExpectedArray[i][j][k] = aPriorDist;
                    }
                }
            }
            var bExpected = Distribution<int>.Array(bExpectedArray);
            object bActual = engine.Infer(b);
            Console.WriteLine(StringUtil.JoinColumns("b = ", bActual, " should be ", bExpected));
            Assert.True(bExpected.MaxDiff(bActual) < 1e-10);
        }

        [Fact]
        public void CaseExitDeepJaggedArrayTest()
        {
            Discrete aPriorDist = new Discrete(0.3, 0.7);
            Range outer = new Range(3).Named("outer");
#if false
            VariableArray<int> middleSizes = Variable.Constant(new int[] { 2, 2, 2 }, outer).Named("middleSizes");
            Range middle = new Range(middleSizes[outer]).Named("inner");
#else
            Range middle = new Range(2).Named("middle");
#endif
#if false
            var innerSizes = Variable.Constant(new int[][] { new int[] { 2, 2 }, new int[] { 2, 2 }, new int[] { 2, 3 } }, outer, middle).Named("innerSizes");
            Range inner = new Range(innerSizes[outer][middle]).Named("inner");
#else
            Range inner = new Range(2).Named("inner");
#endif
            Discrete[][] aPriorArray = new Discrete[outer.SizeAsInt][];
            for (int i = 0; i < aPriorArray.Length; i++)
            {
                aPriorArray[i] = new Discrete[middle.SizeAsInt];
                for (int j = 0; j < aPriorArray[i].Length; j++)
                {
                    aPriorArray[i][j] = aPriorDist;
                }
            }
            var aPrior = Variable.Constant(aPriorArray, outer, middle).Named("aPrior");
            var a = Variable.Array(Variable.Array(Variable.Array<int>(inner), middle), outer).Named("a");
            double[] priorIndex = new double[] { 0.1, 0.5, 0.4 };
            Range item = new Range(1).Named("item");
            VariableArray<int> index = Variable.Array<int>(item).Named("index");
            index[item] = Variable.Discrete(outer, priorIndex).ForEach(item);
            using (Variable.ForEach(outer))
            {
                using (Variable.ForEach(middle))
                {
                    using (Variable.ForEach(inner))
                    {
                        a[outer][middle][inner] = Variable.Random<int, Discrete>(aPrior[outer][middle]);
                    }
                }
            }
            var b = Variable.Array(Variable.Array(Variable.Array<int>(inner), middle), item).Named("b");
            using (Variable.ForEach(item))
            {
                for (int c = 0; c < outer.SizeAsInt; c++)
                {
                    using (Variable.Case(index[item], c))
                    {
                        b[item][middle][inner] = Variable.Copy(a[index[item]][middle][inner]);
                    }
                }
            }

            InferenceEngine engine = new InferenceEngine();
            Discrete[][][] bExpectedArray = new Discrete[item.SizeAsInt][][];
            for (int i = 0; i < bExpectedArray.Length; i++)
            {
                bExpectedArray[i] = new Discrete[middle.SizeAsInt][];
                for (int j = 0; j < bExpectedArray[i].Length; j++)
                {
                    bExpectedArray[i][j] = new Discrete[inner.SizeAsInt];
                    for (int k = 0; k < bExpectedArray[i][j].Length; k++)
                    {
                        bExpectedArray[i][j][k] = aPriorDist;
                    }
                }
            }
            var bExpected = Distribution<int>.Array(bExpectedArray);
            object bActual = engine.Infer(b);
            Console.WriteLine(StringUtil.JoinColumns("b = ", bActual, " should be ", bExpected));
            Assert.True(bExpected.MaxDiff(bActual) < 1e-10);
        }

        [Fact]
        public void CaseExitJaggedArrayTest()
        {
            Discrete aPriorDist = new Discrete(0.3, 0.7);
            Range outer = new Range(3).Named("outer");
#if false
            VariableArray<int> innerSizes = Variable.Constant(new int[] { 2, 2, 2 }, outer).Named("innerSizes");
            Range inner = new Range(innerSizes[outer]).Named("inner");
#else
            Range inner = new Range(2).Named("inner");
#endif
            Discrete[] aPriorArray = new Discrete[outer.SizeAsInt];
            for (int i = 0; i < aPriorArray.Length; i++)
            {
                aPriorArray[i] = aPriorDist;
            }
            var aPrior = Variable.Constant(aPriorArray, outer).Named("aPrior");
            var a = Variable.Array(Variable.Array<int>(inner), outer).Named("a");
            double[] priorIndex = new double[] { 0.1, 0.5, 0.4 };
            Range item = new Range(1).Named("item");
            VariableArray<int> index = Variable.Array<int>(item).Named("index");
            index[item] = Variable.Discrete(outer, priorIndex).ForEach(item);
            using (Variable.ForEach(outer))
            {
                using (Variable.ForEach(inner))
                {
                    a[outer][inner] = Variable.Random<int, Discrete>(aPrior[outer]);
                }
            }
            var b = Variable.Array(Variable.Array<int>(inner), item).Named("b");
            using (Variable.ForEach(item))
            {
                for (int c = 0; c < outer.SizeAsInt; c++)
                {
                    using (Variable.Case(index[item], c))
                    {
                        b[item][inner] = Variable.Copy(a[index[item]][inner]);
                    }
                }
            }

            InferenceEngine engine = new InferenceEngine();
            Discrete[][] bExpectedArray = new Discrete[item.SizeAsInt][];
            for (int i = 0; i < bExpectedArray.Length; i++)
            {
                bExpectedArray[i] = new Discrete[inner.SizeAsInt];
                for (int j = 0; j < bExpectedArray[i].Length; j++)
                {
                    bExpectedArray[i][j] = aPriorDist;
                }
            }
            var bExpected = Distribution<int>.Array(bExpectedArray);
            object bActual = engine.Infer(b);
            Console.WriteLine(StringUtil.JoinColumns("b = ", bActual, " should be ", bExpected));
            Assert.True(bExpected.MaxDiff(bActual) < 1e-10);
        }


        [Fact]
        public void SwitchLocalVariableTest2()
        {
            Discrete aPriorDist = new Discrete(0.3, 0.7);
            Range outer = new Range(3).Named("outer");
            //VariableArray<int> innerSizes = Variable.Constant(new int[] { 2, 2, 2 }, outer).Named("innerSizes");
            //Range inner = new Range(innerSizes[outer]).Named("inner");
            Range inner = new Range(2).Named("inner");
            Discrete[] aPriorArray = new Discrete[outer.SizeAsInt];
            for (int i = 0; i < aPriorArray.Length; i++)
            {
                aPriorArray[i] = aPriorDist;
            }
            var aPrior = Variable.Constant(aPriorArray, outer).Named("aPrior");
            var a = Variable.Array(Variable.Array<int>(inner), outer).Named("a");
            double[] priorIndex = new double[] { 0.1, 0.5, 0.4 };
            Range item = new Range(1).Named("item");
            VariableArray<int> index = Variable.Array<int>(item).Named("index");
            index[item] = Variable.Discrete(outer, priorIndex).ForEach(item);
            using (Variable.ForEach(outer))
            {
                using (Variable.ForEach(inner))
                {
                    a[outer][inner] = Variable.Random<int, Discrete>(aPrior[outer]);
                }
            }
            var b = Variable.Array(Variable.Array<int>(inner), item).Named("b");
            var temp = Variable.Array(Variable.Array<int>(inner), item).Named("temp");
            using (Variable.ForEach(item))
            {
                using (Variable.Switch(index[item]))
                {
                    temp[item][inner] = Variable.Copy(a[index[item]][inner]);
                }
            }
            b.SetTo(temp);

            InferenceEngine engine = new InferenceEngine();
            Discrete[][] bExpectedArray = new Discrete[item.SizeAsInt][];
            for (int i = 0; i < bExpectedArray.Length; i++)
            {
                bExpectedArray[i] = new Discrete[inner.SizeAsInt];
                for (int j = 0; j < bExpectedArray[i].Length; j++)
                {
                    bExpectedArray[i][j] = aPriorDist;
                }
            }
            var bExpected = Distribution<int>.Array(bExpectedArray);
            object bActual = engine.Infer(b);
            Console.WriteLine(StringUtil.JoinColumns("b = ", bActual, " should be ", bExpected));
            Assert.True(bExpected.MaxDiff(bActual) < 1e-10);
        }

        [Fact]
        public void SwitchExitJaggedArrayTest2()
        {
            Discrete aPriorDist = new Discrete(0.3, 0.7);
            Range outer = new Range(3).Named("outer");
            //VariableArray<int> innerSizes = Variable.Constant(new int[] { 2, 2, 2 }, outer).Named("innerSizes");
            //Range inner = new Range(innerSizes[outer]).Named("inner");
            Range inner = new Range(2).Named("inner");
            Discrete[] aPriorArray = new Discrete[outer.SizeAsInt];
            for (int i = 0; i < aPriorArray.Length; i++)
            {
                aPriorArray[i] = aPriorDist;
            }
            var aPrior = Variable.Constant(aPriorArray, outer).Named("aPrior");
            var a = Variable.Array(Variable.Array<int>(inner), outer).Named("a");
            double[] priorIndex = new double[] { 0.1, 0.5, 0.4 };
            Range item = new Range(1).Named("item");
            VariableArray<int> index = Variable.Array<int>(item).Named("index");
            index[item] = Variable.Discrete(outer, priorIndex).ForEach(item);
            using (Variable.ForEach(outer))
            {
                using (Variable.ForEach(inner))
                {
                    a[outer][inner] = Variable.Random<int, Discrete>(aPrior[outer]);
                }
            }
            var b = Variable.Array(Variable.Array<int>(inner), item).Named("b");
            using (Variable.ForEach(item))
            {
                using (Variable.Switch(index[item]))
                {
                    b[item][inner] = Variable.Copy(a[index[item]][inner]);
                }
            }

            InferenceEngine engine = new InferenceEngine();
            Discrete[][] bExpectedArray = new Discrete[item.SizeAsInt][];
            for (int i = 0; i < bExpectedArray.Length; i++)
            {
                bExpectedArray[i] = new Discrete[inner.SizeAsInt];
                for (int j = 0; j < bExpectedArray[i].Length; j++)
                {
                    bExpectedArray[i][j] = aPriorDist;
                }
            }
            var bExpected = Distribution<int>.Array(bExpectedArray);
            object bActual = engine.Infer(b);
            Console.WriteLine(StringUtil.JoinColumns("b = ", bActual, " should be ", bExpected));
            Assert.True(bExpected.MaxDiff(bActual) < 1e-10);
        }

        [Fact]
        public void SwitchExitJaggedArrayTest()
        {
            Discrete aPriorDist = new Discrete(0.3, 0.7);
            Range outer = new Range(3).Named("outer");
            //VariableArray<int> innerSizes = Variable.Constant(new int[] { 2, 2, 2 }, outer).Named("innerSizes");
            //Range inner = new Range(innerSizes[outer]).Named("inner");
            Range inner = new Range(2).Named("inner");
            Discrete[] aPriorArray = new Discrete[outer.SizeAsInt];
            for (int i = 0; i < aPriorArray.Length; i++)
            {
                aPriorArray[i] = aPriorDist;
            }
            var aPrior = Variable.Constant(aPriorArray, outer).Named("aPrior");
            var a = Variable.Array(Variable.Array<int>(inner), outer).Named("a");
            double[] priorIndex = new double[] { 0.1, 0.5, 0.4 };
            Variable<int> index = Variable.Discrete(outer, priorIndex).Named("index");
            using (Variable.ForEach(outer))
            {
                using (Variable.ForEach(inner))
                {
                    a[outer][inner] = Variable.Random<int, Discrete>(aPrior[outer]);
                }
            }
            VariableArray<int> b = Variable.Array<int>(inner).Named("b");
            using (Variable.Switch(index))
            {
                b[inner] = Variable.Copy(a[index][inner]);
            }

            InferenceEngine engine = new InferenceEngine();
            Discrete[] bExpectedArray = new Discrete[inner.SizeAsInt];
            for (int i = 0; i < bExpectedArray.Length; i++)
            {
                bExpectedArray[i] = aPriorDist;
            }
            var bExpected = Distribution<int>.Array(bExpectedArray);
            object bActual = engine.Infer(b);
            Console.WriteLine(StringUtil.JoinColumns("b = ", bActual, " should be ", bExpected));
            Assert.True(bExpected.MaxDiff(bActual) < 1e-10);
        }

        [Fact]
        public void SwitchJaggedArrayTest()
        {
            double boolsPrior = 0.3;
            Range outer = new Range(3).Named("outer");
            VariableArray<int> innerSizes = Variable.Constant(new int[] { 1, 2, 3 }, outer).Named("innerSizes");
            Range inner = new Range(innerSizes[outer]).Named("inner");
            //Range inner = new Range(1).Named("inner");
            var bools = Variable.Array<bool>(Variable.Array<bool>(inner), outer).Named("bools");
            double[] priorIndex = new double[] { 0.1, 0.5, 0.4 };
            Variable<int> index = Variable.Discrete(outer, priorIndex).Named("index");
            using (Variable.ForEach(outer))
            {
                using (Variable.ForEach(inner))
                {
                    bools[outer][inner] = Variable.Bernoulli(boolsPrior);
                }
            }
            using (Variable.Switch(index))
            {
                Variable.ConstrainTrue(bools[index][inner]);
            }
            InferenceEngine engine = new InferenceEngine();
            // model is p(bools,index) = p0(bools) p0(index) prod_k delta(bools[k][*]=1 if index=k)
            // p(bools[i]=1) = (p0(index=i) p0(bools[i][*]=1) + 
            //       sum_(j!=i) p0(index=j) p0(bools[j][*]=1) p0(bools[i]=1))/z
            // z = sum_i p0(index=i) p0(bools[i][*]=1)
            Vector postIndex = Vector.Zero(outer.SizeAsInt);
            for (int i = 0; i < postIndex.Count; i++)
            {
                double zIndex = System.Math.Pow(boolsPrior, innerSizes.ObservedValue[i]);
                postIndex[i] = priorIndex[i] * zIndex;
            }
            postIndex.Scale(1.0 / postIndex.Sum());
            double[] boolsPost = new double[outer.SizeAsInt];
            for (int i = 0; i < boolsPost.Length; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < postIndex.Count; j++)
                {
                    sum += postIndex[j] * ((i == j) ? 1.0 : boolsPrior);
                }
                boolsPost[i] = sum;
            }

            Discrete indexExpected = new Discrete(postIndex);
            Discrete indexActual = engine.Infer<Discrete>(index);
            Console.WriteLine("index = {0} should be {1}", indexActual, indexExpected);
            Assert.True(indexExpected.MaxDiff(indexActual) < 1e-10);

            Bernoulli[][] boolsExpectedArray = new Bernoulli[outer.SizeAsInt][];
            for (int i = 0; i < boolsExpectedArray.Length; i++)
            {
                boolsExpectedArray[i] = new Bernoulli[innerSizes.ObservedValue[i]];
                for (int j = 0; j < boolsExpectedArray[i].Length; j++)
                {
                    boolsExpectedArray[i][j] = new Bernoulli(boolsPost[i]);
                }
            }
            var boolsExpected = Distribution<bool>.Array(boolsExpectedArray);
            object boolsActual = engine.Infer(bools);
            Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsActual, " should be ", boolsExpected));
            Assert.True(boolsExpected.MaxDiff(boolsActual) < 1e-10);
        }

        [Fact]
        public void SwitchJaggedArrayRedundantForEachError()

        {
            Assert.Throws<InvalidOperationException>(() =>
            {

                Range outer = new Range(3).Named("outer");
                Range item = new Range(1).Named("item");
                var bools = Variable.Array<bool>(Variable.Array<bool>(item), outer).Named("bools");
                Variable<int> index = Variable.Discrete(outer, 0.1, 0.5, 0.4).Named("index");
                using (Variable.ForEach(outer))
                {
                    using (Variable.ForEach(item))
                    {
                        bools[outer][item] = Variable.Bernoulli(0.3);
                    }
                    using (Variable.Switch(index))
                    {
                        Variable.ConstrainTrue(bools[index][item]);
                    }
                }
                InferenceEngine engine = new InferenceEngine();
                object boolsActual = engine.Infer(bools);
                Console.WriteLine(boolsActual);

            });
        }

        [Fact]
        public void SwitchJaggedArray2DTest()
        {
            Range outer = new Range(3).Named("outer");
            Range outer2 = new Range(2).Named("outer2");
            VariableArray<int> innerSizes = Variable.Constant(new int[] { 1, 2, 3 }, outer).Named("innerSizes");
            Range inner = new Range(innerSizes[outer]).Named("inner");
            //Range inner = new Range(1).Named("inner");
            var bools = Variable.Array<bool>(Variable.Array<bool>(inner), outer, outer2).Named("bools");
            double[] priorIndex = new double[] { 0.1, 0.5, 0.4 };
            Variable<int> index = Variable.Discrete(outer, priorIndex).Named("index");
            double boolsPrior = 0.3;
            using (Variable.ForEach(outer))
            {
                using (Variable.ForEach(inner))
                {
                    bools[outer, outer2][inner] = Variable.Bernoulli(boolsPrior).ForEach(outer2);
                }
            }
            using (Variable.Switch(index))
            {
                Variable.ConstrainTrue(bools[index, outer2][inner]);
            }
            InferenceEngine engine = new InferenceEngine();

            // model is p(bools,index) = p0(bools) p0(index) prod_k delta(bools[k][*]=1 if index=k)
            // p(bools[i]=1) = (p0(index=i) p0(bools[i][*]=1) + 
            //       sum_(j!=i) p0(index=j) p0(bools[j][*]=1) p0(bools[i]=1))/z
            // z = sum_i p0(index=i) p0(bools[i][*]=1)
            Vector postIndex = Vector.Zero(outer.SizeAsInt);
            for (int i = 0; i < postIndex.Count; i++)
            {
                double zIndex = System.Math.Pow(boolsPrior, outer2.SizeAsInt * innerSizes.ObservedValue[i]);
                postIndex[i] = priorIndex[i] * zIndex;
            }
            postIndex.Scale(1.0 / postIndex.Sum());
            double[] boolsPost = new double[outer.SizeAsInt];
            for (int i = 0; i < boolsPost.Length; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < postIndex.Count; j++)
                {
                    sum += postIndex[j] * ((i == j) ? 1.0 : boolsPrior);
                }
                boolsPost[i] = sum;
            }

            Discrete indexExpected = new Discrete(postIndex);
            Discrete indexActual = engine.Infer<Discrete>(index);
            Console.WriteLine("index = {0} should be {1}", indexActual, indexExpected);
            Assert.True(indexExpected.MaxDiff(indexActual) < 1e-10);

            Bernoulli[,][] boolsExpectedArray = new Bernoulli[outer.SizeAsInt, outer2.SizeAsInt][];
            for (int i = 0; i < boolsExpectedArray.GetLength(0); i++)
            {
                for (int j = 0; j < boolsExpectedArray.GetLength(1); j++)
                {
                    boolsExpectedArray[i, j] = new Bernoulli[innerSizes.ObservedValue[i]];
                    for (int k = 0; k < boolsExpectedArray[i, j].Length; k++)
                    {
                        boolsExpectedArray[i, j][k] = new Bernoulli(boolsPost[i]);
                    }
                }
            }
            IDistribution<bool[,][]> boolsExpected = Distribution<bool>.Array(boolsExpectedArray);
            object boolsActual = engine.Infer(bools);
            Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsActual, " should be ", boolsExpected));
            Assert.True(boolsExpected.MaxDiff(boolsActual) < 1e-10);
        }

        // DistributionArray3D not supported yet
        [Fact]
        [Trait("Category", "OpenBug")]
        public void SwitchJaggedArray3DTest()
        {
            Range outer = new Range(3).Named("outer");
            Range outer2 = new Range(2).Named("outer2");
            Range outer3 = new Range(2).Named("outer3");
            VariableArray<int> innerSizes = Variable.Constant(new int[] { 1, 2, 3 }, outer).Named("innerSizes");
            Range inner = new Range(innerSizes[outer]).Named("inner");
            //Range inner = new Range(1).Named("inner");
            var bools = Variable.Array<bool>(Variable.Array<bool>(inner), outer, outer2, outer3).Named("bools");
            double[] priorIndex = new double[] { 0.1, 0.5, 0.4 };
            Variable<int> index = Variable.Discrete(outer, priorIndex).Named("index");
            double boolsPrior = 0.3;
            using (Variable.ForEach(outer))
            {
                using (Variable.ForEach(inner))
                {
                    bools[outer, outer2, outer3][inner] = Variable.Bernoulli(boolsPrior).ForEach(outer2, outer3);
                }
            }
            using (Variable.Switch(index))
            {
                Variable.ConstrainTrue(bools[index, outer2, outer3][inner]);
            }

            InferenceEngine engine = new InferenceEngine();
            // model is p(bools,index) = p0(bools) p0(index) prod_k delta(bools[k][*]=1 if index=k)
            // p(bools[i]=1) = (p0(index=i) p0(bools[i][*]=1) + 
            //       sum_(j!=i) p0(index=j) p0(bools[j][*]=1) p0(bools[i]=1))/z
            // z = sum_i p0(index=i) p0(bools[i][*]=1)
            Vector postIndex = Vector.Zero(outer.SizeAsInt);
            for (int i = 0; i < postIndex.Count; i++)
            {
                double zIndex = System.Math.Pow(boolsPrior, outer2.SizeAsInt * outer3.SizeAsInt * innerSizes.ObservedValue[i]);
                postIndex[i] = priorIndex[i] * zIndex;
            }
            postIndex.Scale(1.0 / postIndex.Sum());
            double[] boolsPost = new double[outer.SizeAsInt];
            for (int i = 0; i < boolsPost.Length; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < postIndex.Count; j++)
                {
                    sum += postIndex[j] * ((i == j) ? 1.0 : boolsPrior);
                }
                boolsPost[i] = sum;
            }

            Discrete indexExpected = new Discrete(postIndex);
            Discrete indexActual = engine.Infer<Discrete>(index);
            Console.WriteLine("index = {0} should be {1}", indexActual, indexExpected);
            Assert.True(indexExpected.MaxDiff(indexActual) < 1e-10);

            Bernoulli[,,][] boolsExpectedArray = new Bernoulli[outer.SizeAsInt, outer2.SizeAsInt, outer3.SizeAsInt][];
            for (int i = 0; i < boolsExpectedArray.GetLength(0); i++)
            {
                for (int j = 0; j < boolsExpectedArray.GetLength(1); j++)
                {
                    for (int k = 0; k < boolsExpectedArray.GetLength(2); k++)
                    {
                        boolsExpectedArray[i, j, k] = new Bernoulli[innerSizes.ObservedValue[i]];
                        for (int m = 0; m < boolsExpectedArray[i, j, k].Length; m++)
                        {
                            boolsExpectedArray[i, j, k][m] = new Bernoulli(boolsPost[i]);
                        }
                    }
                }
            }
            IDistribution<bool[,,][]> boolsExpected = null;
            //IDistribution<bool[,,][]> boolsExpected = Distribution<bool>.Array(boolsExpectedArray);
            object boolsActual = engine.Infer(bools);
            Console.WriteLine("bools = {0} (should be {1})", boolsActual, boolsExpected);
            Assert.True(boolsExpected.MaxDiff(boolsActual) < 1e-10);
        }

        [Fact]
        public void SwitchDeepJaggedArrayTest()
        {
            double boolsPrior = 0.3;
            Range outer = new Range(2).Named("outer");
            VariableArray<int> middleSizes = Variable.Constant(new int[] { 2, 3 }, outer).Named("middleSizes");
            Range middle = new Range(middleSizes[outer]).Named("middle");
            var innerSizes = Variable.Constant(new int[][]
                {
                    new int[] {1, 2}, new int[] {1, 2, 3}
                }, outer, middle).Named("innerSizes");
            Range inner = new Range(innerSizes[outer][middle]).Named("inner");
            //Range inner = new Range(1).Named("inner");
            var bools = Variable.Array(Variable.Array(Variable.Array<bool>(inner), middle), outer).Named("bools");
            double[] priorIndex = new double[] { 0.1, 0.9 };
            Variable<int> index = Variable.Discrete(outer, priorIndex).Named("index");
            if (false)
            {
                using (Variable.ForEach(outer))
                {
                    using (Variable.ForEach(middle))
                    {
                        using (Variable.ForEach(inner))
                        {
                            bools[outer][middle][inner] = Variable.Bernoulli(boolsPrior);
                        }
                    }
                }
            }
            else
            {
                bools[outer][middle][inner] = Variable.Bernoulli(boolsPrior).ForEach(outer, middle, inner);
            }
            using (Variable.Switch(index))
            {
                Variable.ConstrainTrue(bools[index][middle][inner]);
            }
            InferenceEngine engine = new InferenceEngine();
            // model is p(bools,index) = p0(bools) p0(index) prod_k delta(bools[k][*]=1 if index=k)
            // p(bools[i]=1) = (p0(index=i) p0(bools[i][*]=1) + 
            //       sum_(j!=i) p0(index=j) p0(bools[j][*]=1) p0(bools[i]=1))/z
            // z = sum_i p0(index=i) p0(bools[i][*]=1)
            Vector postIndex = Vector.Zero(outer.SizeAsInt);
            for (int i = 0; i < postIndex.Count; i++)
            {
                int size = 1;
                for (int j = 0; j < middleSizes.ObservedValue[i]; j++)
                {
                    size += innerSizes.ObservedValue[i][j];
                }
                double zIndex = System.Math.Pow(boolsPrior, size);
                postIndex[i] = priorIndex[i] * zIndex;
            }
            postIndex.Scale(1.0 / postIndex.Sum());
            double[] boolsPost = new double[outer.SizeAsInt];
            for (int i = 0; i < boolsPost.Length; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < postIndex.Count; j++)
                {
                    sum += postIndex[j] * ((i == j) ? 1.0 : boolsPrior);
                }
                boolsPost[i] = sum;
            }

            Discrete indexExpected = new Discrete(postIndex);
            Discrete indexActual = engine.Infer<Discrete>(index);
            Console.WriteLine("index = {0} should be {1}", indexActual, indexExpected);
            Assert.True(indexExpected.MaxDiff(indexActual) < 1e-10);

            Bernoulli[][][] boolsExpectedArray = new Bernoulli[outer.SizeAsInt][][];
            for (int i = 0; i < boolsExpectedArray.Length; i++)
            {
                boolsExpectedArray[i] = new Bernoulli[middleSizes.ObservedValue[i]][];
                for (int j = 0; j < boolsExpectedArray[i].Length; j++)
                {
                    boolsExpectedArray[i][j] = new Bernoulli[innerSizes.ObservedValue[i][j]];
                    for (int k = 0; k < boolsExpectedArray[i][j].Length; k++)
                    {
                        boolsExpectedArray[i][j][k] = new Bernoulli(boolsPost[i]);
                    }
                }
            }
            IDistribution<bool[][][]> boolsExpected = Distribution<bool>.Array(boolsExpectedArray);
            object boolsActual = engine.Infer(bools);
            Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsActual, " should be ", boolsExpected));
            Assert.True(boolsExpected.MaxDiff(boolsActual) < 1e-10);
        }

        [Fact]
        public void SwitchWithReplicateTest()
        {
            Range item = new Range(1).Named("item");
            VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
            using (Variable.ForEach(item))
            {
                bools[item] = Variable.Bernoulli(0.3);
                Variable<int> index = Variable.Discrete(new Range(3), 0.1, 0.5, 0.4).Named("index");
                using (Variable.Switch(index))
                {
                    Variable.ConstrainTrue(bools[item]);
                }
            }
            InferenceEngine engine = new InferenceEngine();
            IDistribution<bool[]> boolsExpected = Distribution<bool>.Array(new Bernoulli[] { new Bernoulli(1) });
            DistributionArray<Bernoulli> boolsActual = engine.Infer<DistributionArray<Bernoulli>>(bools);
            Console.WriteLine("bools = {0} (should be {1})", boolsActual, boolsExpected);
            Assert.True(boolsExpected.MaxDiff(boolsActual) < 1e-10);
        }

        [Fact]
        public void SwitchNotIndexedByConditionTest()
        {
            double bPrior = 0.3;
            Variable<bool> b = Variable.Bernoulli(bPrior).Named("b");
            double[] indexPrior = { 0.1, 0.5, 0.4 };
            Range item = new Range(indexPrior.Length);
            Variable<int> index = Variable.Discrete(item, indexPrior).Named("index");
            double[] likes = { 0.2, 0.6, 0.7 };
            Bernoulli[] bernoullis = Array.ConvertAll(likes, prob => new Bernoulli(prob));
            VariableArray<Bernoulli> likelihood = Variable.Observed(bernoullis, item).Named("likelihood");
            double[] likes2 = { 0.2, 0.6, 0.7 };
            Bernoulli[] bernoullis2 = Array.ConvertAll(likes2, prob => new Bernoulli(prob));
            VariableArray<Bernoulli> likelihood2 = Variable.Observed(bernoullis2, item).Named("likelihood2");
            using (Variable.Switch(index))
            {
                Variable.ConstrainEqualRandom(b, likelihood[index]);
                Variable.ConstrainEqualRandom(b, likelihood2[index]);
            }
            InferenceEngine engine = new InferenceEngine();
            double sumT = 0.0;
            double sumF = 0.0;
            for (int i = 0; i < indexPrior.Length; i++)
            {
                sumT += indexPrior[i] * bPrior * likes[i] * likes2[i];
                sumF += indexPrior[i] * (1 - bPrior) * (1 - likes[i]) * (1 - likes2[i]);
            }
            double z = sumT + sumF;
            Bernoulli bExpected = new Bernoulli(sumT / z);
            Bernoulli bActual = engine.Infer<Bernoulli>(b);
            Console.WriteLine("b = {0} (should be {1})", bActual, bExpected);
            Assert.True(bExpected.MaxDiff(bActual) < 1e-10);
        }

        [Fact]
        public void SwitchNotIndexedByConditionTest2()
        {
            double bPrior = 0.3;
            Range outer = new Range(2);
            VariableArray<bool> b = Variable.Array<bool>(outer).Named("b");
            b[outer] = Variable.Bernoulli(bPrior).ForEach(outer);
            Variable<int> indexObs = Variable.New<int>().Named("zero");
            double[] indexPrior = { 0.1, 0.5, 0.4 };
            Range item = new Range(indexPrior.Length);
            Variable<int> index = Variable.Discrete(item, indexPrior).Named("index");
            double[] likes = { 0.2, 0.6, 0.7 };
            Bernoulli[] bernoullis = Array.ConvertAll(likes, prob => new Bernoulli(prob));
            VariableArray<Bernoulli> likelihood = Variable.Observed(bernoullis, item).Named("likelihood");
            double[] likes2 = { 0.2, 0.6, 0.7 };
            Bernoulli[] bernoullis2 = Array.ConvertAll(likes2, prob => new Bernoulli(prob));
            VariableArray<Bernoulli> likelihood2 = Variable.Observed(bernoullis2, item).Named("likelihood2");
            using (Variable.Switch(index))
            {
                Variable.ConstrainEqualRandom(b[0], likelihood[index]);
                Variable.ConstrainEqualRandom(b[indexObs], likelihood2[index]);
            }
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.Compiled += (sender, e) =>
            {
                // check for the inefficient replication warning
                Assert.Equal(1, e.Warnings.Count);
            };
            for (int trial = 0; trial < 2; trial++)
            {
                indexObs.ObservedValue = trial;
                // Z = sum_index sum_b p(index) p(b[0]) p(b[1]) f(b[0],like[index]) f(b[zero],like2[index])
                // p(b[0]|D) = p(b[0]) sum_index sum_b[1] p(index) f(b[0],like[index]) f(b[zero],like2[index])
                double[] sum0_cond_index = new double[indexPrior.Length];
                double[] sum1_cond_index = new double[indexPrior.Length];
                double[] sumT = new double[outer.SizeAsInt];
                double z = 0;
                for (int i = 0; i < indexPrior.Length; i++)
                {
                    double sum0T, sum1T;
                    if (indexObs.ObservedValue == 0)
                    {
                        sum0T = bPrior * likes[i] * likes2[i];
                        sum1T = bPrior;
                        sum0_cond_index[i] = sum0T + (1 - bPrior) * (1 - likes[i]) * (1 - likes2[i]);
                        sum1_cond_index[i] = 1;
                    }
                    else
                    {
                        sum0T = bPrior * likes[i];
                        sum1T = bPrior * likes2[i];
                        sum0_cond_index[i] = sum0T + (1 - bPrior) * (1 - likes[i]);
                        sum1_cond_index[i] = sum1T + (1 - bPrior) * (1 - likes2[i]);
                    }
                    z += indexPrior[i] * sum0_cond_index[i] * sum1_cond_index[i];
                    sumT[0] += indexPrior[i] * sum0T * sum1_cond_index[i];
                    sumT[1] += indexPrior[i] * sum1T * sum0_cond_index[i];
                }
                IDistribution<bool[]> bExpected = Distribution<bool>.Array(new Bernoulli[] { new Bernoulli(sumT[0] / z), new Bernoulli(sumT[1] / z) });
                object bActual = engine.Infer(b);
                Console.WriteLine(StringUtil.JoinColumns("b = ", bActual, " should be ", bExpected));
                Assert.True(bExpected.MaxDiff(bActual) < 1e-10);
            }
        }

        [Fact]
        public void SwitchUsesConditionTest()
        {
            double[] probs = new double[] { 0.1, 0.5, 0.4 };
            int n = probs.Length;
            Range item = new Range(n).Named("item");
            VariableArray<int> array = Variable.Array<int>(item).Named("array");
            array[item] = Variable.DiscreteUniform(n).ForEach(item);
            Variable<int> index = Variable.Discrete(item, probs).Named("index");
            using (Variable.Switch(index))
            {
                Variable.ConstrainEqual(array[index], index);
            }
            InferenceEngine engine = new InferenceEngine();
            Discrete[] arrayExpectedArray = new Discrete[n];
            for (int i = 0; i < arrayExpectedArray.Length; i++)
            {
                double[] p = new double[n];
                double c = (1 - probs[i]) / n;
                for (int j = 0; j < p.Length; j++)
                {
                    p[j] = (i == j) ? (probs[i] + c) : c;
                }
                arrayExpectedArray[i] = new Discrete(p);
            }
            IDistribution<int[]> arrayExpected = Distribution<int>.Array(arrayExpectedArray);
            DistributionArray<Discrete> arrayActual = engine.Infer<DistributionArray<Discrete>>(array);
            Console.WriteLine(Utilities.StringUtil.JoinColumns("array = ", arrayActual, " should be ", arrayExpected));
            Assert.True(arrayExpected.MaxDiff(arrayActual) < 1e-10);
        }

        [Fact]
        public void SwitchMixedWithCaseError()

        {
            Assert.Throws<CompilationFailedException>(() =>
            {

                double[] probs = new double[] { 0.1, 0.5, 0.4 };
                int n = probs.Length;
                Range item = new Range(n).Named("item");
                VariableArray<int> array = Variable.Array<int>(item).Named("array");
                array[item] = Variable.DiscreteUniform(n).ForEach(item);
                Variable<int> index = Variable.Discrete(item, probs).Named("index");
                using (Variable.Switch(index))
                {
                    Variable.ConstrainEqual(array[index], index);
                }
                if (true)
                {
                    using (Variable.Case(index, 0))
                    {
                        Variable.ConstrainEqual(array[index], index);
                    }
                }
                InferenceEngine engine = new InferenceEngine();
                DistributionArray<Discrete> arrayActual = engine.Infer<DistributionArray<Discrete>>(array);
                Console.WriteLine(arrayActual);

            });
        }

        // This model is very slow to mix under Gibbs sampling, while EP is exact
        [Fact]
        public void OnePointMixtureGaussianTest()
        {
            double psi = 10;
            int K = 2;
            Gaussian like = new Gaussian(27, 1);

            Variable<double>[] m = new Variable<double>[K];
            for (int i = 0; i < K; i++)
            {
                m[i] = Variable.GaussianFromMeanAndVariance(i, psi).Named("m" + i);
            }
            Vector probs = Vector.Constant(K, 1.0 / (double)K);
            Variable<int> c = Variable.Discrete(probs).Named("c");
            for (int i = 0; i < K; i++)
            {
                using (Variable.Case(c, i))
                {
                    Variable.ConstrainEqualRandom<double, Gaussian>(m[i], like);
                }
            }

            InferenceEngine ie = new InferenceEngine();
            //ie.Algorithm = new GibbsSampling();
            Discrete cActual = ie.Infer<Discrete>(c);
            double[] logProbs = new double[K];
            for (int i = 0; i < K; i++)
            {
                logProbs[i] = (new Gaussian(i, psi)).GetLogAverageOf(like);
            }
            double logZ = MMath.LogSumExp(logProbs);
            Discrete cExpected = new Discrete(new double[] { System.Math.Exp(logProbs[0] - logZ), System.Math.Exp(logProbs[1] - logZ) });
            double error = cExpected.MaxDiff(cActual);
            Console.WriteLine("c = {0} should be {1} (error = {2})", cActual, cExpected, error);
            Assert.True(error < 1e-5);

            Gaussian[] mExpected = new Gaussian[K];
            for (int i = 0; i < K; i++)
            {
                Gaussian prior = new Gaussian(i, psi);
                Gaussian post = new Gaussian();
                post.SetToSum(cExpected[i], prior * like, 1 - cExpected[i], prior);
                if (GateEnterOp<double>.ForceProper)
                {
                    post.SetMeanAndVariance(post.GetMean(), psi);
                }
                mExpected[i] = post;
            }

            Gaussian[] mpost = new Gaussian[K];
            for (int i = 0; i < K; i++)
            {
                mpost[i] = ie.Infer<Gaussian>(m[i]);
                error = mpost[i].MaxDiff(mExpected[i]);
                Console.WriteLine("mpost[{0}] = {1} (should be {2}) and  error = {3}", i, mpost[i], mExpected[i], error);
                Assert.True(error < 1e-5);
            }
        }

        [Fact]
        public void OnePointMixtureGaussianInferPrecisionTest()
        {
            int K = 2;

            InferenceEngine ie = new InferenceEngine(); //new VariationalMessagePassing());

            Variable<Gaussian>[] mprior = new Variable<Gaussian>[K];
            Variable<double>[] m = new Variable<double>[K];
            Variable<Gamma>[] precprior = new Variable<Gamma>[K];
            Variable<double>[] prec = new Variable<double>[K];
            for (int i = 0; i < K; i++)
            {
                precprior[i] = Variable.Observed(Gamma.FromMeanAndVariance(1, 1)).Named("precprior" + i);
                prec[i] = Variable<double>.Random(precprior[i]).Named("prec" + i);
            }
            for (int i = 0; i < K; i++)
            {
                mprior[i] = Variable.Observed(new Gaussian(i, 10)).Named("mprior" + i);
            }
            Vector probs = Vector.Constant(K, 1.0 / (double)K);
            Variable<int> c = Variable.Discrete(probs).Named("c");
            for (int i = 0; i < K; i++)
            {
                using (Variable.Case(c, i))
                {
                    Variable<double> mprev = Variable<double>.Random(mprior[i]);
                    m[i] = Variable.GaussianFromMeanAndPrecision(mprev, prec[i]).Named("m" + i);
                    Variable.ConstrainEqualRandom(m[i], Variable.Constant(new Gaussian(27, 1)));
                }
            }
            Console.WriteLine(ie.Infer<Discrete>(c));
            for (int i = 0; i < K; i++)
            {
                Console.WriteLine(ie.Infer<Gaussian>(m[i]));
                Console.WriteLine(ie.Infer<Gamma>(prec[i]));
            }
        }

        [Fact]
        public void IfExitObservedConditionTest()
        {
            Variable<bool> b = Variable.New<bool>().Named("b");
            Variable<bool> c = Variable.New<bool>().Named("c");
            using (Variable.If(b))
            {
                c.SetTo(Variable.Bernoulli(0.1));
            }
            using (Variable.IfNot(b))
            {
                c.SetTo(Variable.Bernoulli(0.2));
            }
            InferenceEngine engine = new InferenceEngine();
            Bernoulli cExpected;
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial % 2 == 0)
                {
                    b.ObservedValue = true;
                    cExpected = new Bernoulli(0.1);
                }
                else
                {
                    b.ObservedValue = false;
                    cExpected = new Bernoulli(0.2);
                }
                Bernoulli cActual = engine.Infer<Bernoulli>(c);
                Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
                Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
            }
        }

        [Fact]
        public void IfExitObservedConditionTest2()
        {
            Variable<bool> b = Variable.New<bool>().Named("b");
            Variable<bool> c = Variable.New<bool>().Named("c");
            using (Variable.If(b))
            {
                c.SetTo(Variable.Bernoulli(0.125));
            }
            using (Variable.IfNot(b))
            {
                c.SetTo(Variable.Bernoulli(0.25));
            }
            var d = !c;
            d.Name = nameof(d);
            InferenceEngine engine = new InferenceEngine();
            Bernoulli dExpected;
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial % 2 == 0)
                {
                    b.ObservedValue = true;
                    dExpected = new Bernoulli(1 - 0.125);
                }
                else
                {
                    b.ObservedValue = false;
                    dExpected = new Bernoulli(1 - 0.25);
                }
                Bernoulli dActual = engine.Infer<Bernoulli>(d);
                Console.WriteLine("d = {0} should be {1}", dActual, dExpected);
                Assert.True(dExpected.MaxDiff(dActual) < 1e-10);
            }
        }

        [Fact]
        public void IfRandomIfObservedIfRandomConditionTest()
        {
            Variable<bool> a = Variable.Bernoulli(0.2).Named("a");
            Variable<bool> b = Variable.New<bool>().Named("b");
            Variable<bool> c = Variable.Bernoulli(0.1).Named("c");
            Variable<bool> d = Variable.Bernoulli(0.3).Named("d");
            using (Variable.If(a))
            {
                using (Variable.If(b))
                {
                    using (Variable.If(c))
                    {
                        Variable.ConstrainTrue(d);
                    }
                }
            }
            InferenceEngine engine = new InferenceEngine();
            Bernoulli dExpected, cExpected, aExpected;
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial == 0)
                {
                    b.ObservedValue = true;
                    // p(a,b,c) = 0.2^a 0.8^(1-a) 0.1^c 0.9^(1-c) 0.3^d 0.7^(1-d) 0^(abc(1-d))
                    double z = 0.2 * 0.1 * 0.3 + 0.2 * 0.9 + 0.8;
                    dExpected = new Bernoulli(0.3 / z);
                    cExpected = new Bernoulli(0.1 * (0.2 * 0.3 + 0.8) / z);
                    aExpected = new Bernoulli(0.2 * (0.1 * 0.3 + 0.9) / z);
                }
                else
                {
                    b.ObservedValue = false;
                    dExpected = new Bernoulli(0.3);
                    cExpected = new Bernoulli(0.1);
                    aExpected = new Bernoulli(0.2);
                }
                Bernoulli dActual = engine.Infer<Bernoulli>(d);
                Console.WriteLine("d = {0} should be {1}", dActual, dExpected);
                Assert.True(dExpected.MaxDiff(dActual) < 1e-10);
                Bernoulli cActual = engine.Infer<Bernoulli>(c);
                Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
                Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
                Bernoulli aActual = engine.Infer<Bernoulli>(a);
                Console.WriteLine("a = {0} should be {1}", aActual, aExpected);
                Assert.True(aExpected.MaxDiff(aActual) < 1e-10);
            }
        }

        [Fact]
        public void IfRandomIfObservedConditionTest()
        {
            double aPrior = 0.2;
            Variable<bool> a = Variable.Bernoulli(aPrior).Named("a");
            Variable<bool> b = Variable.New<bool>().Named("b");
            Variable<bool> c = Variable.Bernoulli(0.1).Named("c");
            using (Variable.If(a))
            {
                using (Variable.If(b))
                {
                    Variable.ConstrainTrue(c);
                }
            }
            InferenceEngine engine = new InferenceEngine();
            Bernoulli cExpected, aExpected;
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial == 0)
                {
                    b.ObservedValue = true;
                    // p(a,b,c) = 0.2^a 0.8^(1-a) 0.1^c 0.9^(1-c) 0^(ab(1-c))
                    double z = aPrior * 0.1 + (1 - aPrior);
                    cExpected = new Bernoulli(0.1 / z);
                    aExpected = new Bernoulli(aPrior * 0.1 / z);
                }
                else
                {
                    b.ObservedValue = false;
                    cExpected = new Bernoulli(0.1);
                    aExpected = new Bernoulli(aPrior);
                }
                Bernoulli cActual = engine.Infer<Bernoulli>(c);
                Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
                Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
                Bernoulli aActual = engine.Infer<Bernoulli>(a);
                Console.WriteLine("a = {0} should be {1}", aActual, aExpected);
                Assert.True(aExpected.MaxDiff(aActual) < 1e-10);
            }
        }

        [Fact]
        public void IfRandomIfObservedConditionTest2()
        {
            double cPrior = 0.1;
            double aPrior = 0.2;
            Variable<bool> a = Variable.Bernoulli(aPrior).Named("a");
            Variable<bool> b = Variable.New<bool>().Named("b");
            Variable<bool> c = Variable.Bernoulli(cPrior).Named("c");
            using (Variable.If(a))
            {
                using (Variable.If(b))
                {
                    Variable.ConstrainTrue(c);
                }
                Variable.ConstrainEqualRandom(c, new Bernoulli(0.3));
            }
            InferenceEngine engine = new InferenceEngine();
            Bernoulli cExpected, aExpected;
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial == 0)
                {
                    b.ObservedValue = true;
                    // p(a,b,c) = 0.2^a 0.8^(1-a) 0.1^c 0.9^(1-c) 0^(ab(1-c)) 0.3^ac 0.7^a(1-c)
                    double z = aPrior * (cPrior * 0.3) + (1 - aPrior);
                    cExpected = new Bernoulli(cPrior * (aPrior * 0.3 + (1 - aPrior)) / z);
                    aExpected = new Bernoulli(aPrior * cPrior * 0.3 / z);
                }
                else
                {
                    b.ObservedValue = false;
                    double zCondA = (cPrior * 0.3 + (1 - cPrior) * (1 - 0.3));
                    double z = aPrior * zCondA + (1 - aPrior);
                    cExpected = new Bernoulli(cPrior * (aPrior * 0.3 + (1 - aPrior)) / z);
                    aExpected = new Bernoulli(aPrior * zCondA / z);
                }
                Bernoulli cActual = engine.Infer<Bernoulli>(c);
                Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
                Bernoulli aActual = engine.Infer<Bernoulli>(a);
                Console.WriteLine("a = {0} should be {1}", aActual, aExpected);
                Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
                Assert.True(aExpected.MaxDiff(aActual) < 1e-10);
            }
        }

        [Fact]
        public void IfRandomIfObservedConditionTest3()
        {
            double aPrior = 0.2;
            double cPrior = 0.1;
            Variable<bool> a = Variable.Bernoulli(aPrior).Named("a");
            Variable<bool> b = Variable.New<bool>().Named("b");
            Variable<bool> c = Variable.Bernoulli(cPrior).Named("c");
            using (Variable.If(a))
            {
                using (Variable.If(b))
                {
                    Variable.ConstrainTrue(c);
                }
                using (Variable.IfNot(b))
                {
                    Variable.ConstrainFalse(c);
                }
            }
            InferenceEngine engine = new InferenceEngine();
            Bernoulli cExpected, aExpected;
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial == 0)
                {
                    b.ObservedValue = true;
                    // p(a,b,c) = 0.2^a 0.8^(1-a) 0.1^c 0.9^(1-c) 0^(ab(1-c)) 0^(a(1-b)c)
                    double z = aPrior * cPrior + (1 - aPrior);
                    cExpected = new Bernoulli(cPrior / z);
                    aExpected = new Bernoulli(aPrior * cPrior / z);
                }
                else
                {
                    b.ObservedValue = false;
                    double z = aPrior * (1 - cPrior) + (1 - aPrior);
                    cExpected = new Bernoulli(cPrior * (1 - aPrior) / z);
                    aExpected = new Bernoulli(aPrior * (1 - cPrior) / z);
                }
                Bernoulli cActual = engine.Infer<Bernoulli>(c);
                Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
                Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
                Bernoulli aActual = engine.Infer<Bernoulli>(a);
                Console.WriteLine("a = {0} should be {1}", aActual, aExpected);
                Assert.True(aExpected.MaxDiff(aActual) < 1e-10);
            }
        }

        [Fact]
        public void IfRandomIfObservedConditionTest4()
        {
            double aPrior = 0.2;
            double cPrior = 0.1;
            Variable<bool> a = Variable.Bernoulli(aPrior).Named("a");
            Variable<bool> b = Variable.New<bool>().Named("b");
            Variable<bool> c = Variable.Bernoulli(cPrior).Named("c");
            Variable<bool> d = Variable.New<bool>().Named("d");
            using (Variable.If(a))
            {
                using (Variable.If(b))
                {
                    Variable.ConstrainTrue(c);
                }
                using (Variable.If(d))
                {
                    Variable.ConstrainEqualRandom(c, new Bernoulli(0.3));
                }
            }
            InferenceEngine engine = new InferenceEngine();
            Bernoulli cExpected, aExpected;
            for (int trial = 0; trial < 4; trial++)
            {
                if (trial == 0)
                {
                    b.ObservedValue = true;
                    d.ObservedValue = true;
                    // p(a,b,c) = 0.2^a 0.8^(1-a) 0.1^c 0.9^(1-c) 0^(a(1-c)) 0.3^(ac)
                    double z = aPrior * cPrior * 0.3 + (1 - aPrior);
                    cExpected = new Bernoulli(cPrior * (aPrior * 0.3 + (1 - aPrior)) / z);
                    aExpected = new Bernoulli(aPrior * cPrior * 0.3 / z);
                }
                else if (trial == 1)
                {
                    b.ObservedValue = true;
                    d.ObservedValue = false;
                    // p(a,b,c) = 0.2^a 0.8^(1-a) 0.1^c 0.9^(1-c) 0^(a(1-c))
                    double z = aPrior * cPrior + (1 - aPrior);
                    cExpected = new Bernoulli(cPrior / z);
                    aExpected = new Bernoulli(aPrior * cPrior / z);
                }
                else if (trial == 2)
                {
                    b.ObservedValue = false;
                    d.ObservedValue = true;
                    // p(a,b,c) = 0.2^a 0.8^(1-a) 0.1^c 0.9^(1-c) 0.3^(ac) 0.7^(a(1-c))
                    double z = aPrior * (cPrior * 0.3 + (1 - cPrior) * 0.7) + (1 - aPrior);
                    cExpected = new Bernoulli(cPrior * (aPrior * 0.3 + (1 - aPrior)) / z);
                    aExpected = new Bernoulli(aPrior * (cPrior * 0.3 + (1 - cPrior) * 0.7) / z);
                }
                else
                {
                    // trial == 3
                    b.ObservedValue = false;
                    d.ObservedValue = false;
                    // p(a,b,c) = 0.2^a 0.8^(1-a) 0.1^c 0.9^(1-c)
                    double z = 1.0;
                    cExpected = new Bernoulli(cPrior / z);
                    aExpected = new Bernoulli(aPrior / z);
                }
                Bernoulli cActual = engine.Infer<Bernoulli>(c);
                Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
                Bernoulli aActual = engine.Infer<Bernoulli>(a);
                Console.WriteLine("a = {0} should be {1}", aActual, aExpected);
                Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
                Assert.True(aExpected.MaxDiff(aActual) < 1e-10);
            }
        }

        [Fact]
        public void IfRandomIfObservedConditionTest5()
        {
            double aPrior = 0.2;
            double cPrior = 0.1;
            double cLike1T = 0.25;
            double cLike1F = 0.3;
            double cLike2T = 0.35;
            double cLike2F = 0.4;
            Variable<bool> a = Variable.Bernoulli(aPrior).Named("a");
            Variable<bool> b = Variable.New<bool>().Named("b");
            Variable<bool> b2 = Variable.New<bool>().Named("b2");
            Variable<bool> c = Variable.Bernoulli(cPrior).Named("c");
            using (Variable.If(a))
            {
                using (Variable.If(b))
                {
                    Variable.ConstrainEqualRandom(c, new Bernoulli(cLike1T));
                }
                using (Variable.IfNot(b))
                {
                    Variable.ConstrainEqualRandom(c, new Bernoulli(cLike1F));
                }
                using (Variable.If(b2))
                {
                    Variable.ConstrainEqualRandom(c, new Bernoulli(cLike2T));
                }
                using (Variable.IfNot(b2))
                {
                    Variable.ConstrainEqualRandom(c, new Bernoulli(cLike2F));
                }
            }
            InferenceEngine engine = new InferenceEngine();
            Bernoulli cExpected, aExpected;
            for (int trial = 0; trial < 4; trial++)
            {
                if (trial == 0)
                {
                    b.ObservedValue = true;
                    b2.ObservedValue = true;
                    double sumT = cPrior * cLike1T * cLike2T + (1 - cPrior) * (1 - cLike1T) * (1 - cLike2T);
                    double z = aPrior * sumT + (1 - aPrior);
                    cExpected = new Bernoulli(cPrior * (aPrior * cLike1T * cLike2T + (1 - aPrior)) / z);
                    aExpected = new Bernoulli(aPrior * sumT / z);
                }
                else if (trial == 1)
                {
                    b.ObservedValue = true;
                    b2.ObservedValue = false;
                    double sumT = cPrior * cLike1T * cLike2F + (1 - cPrior) * (1 - cLike1T) * (1 - cLike2F);
                    double z = aPrior * sumT + (1 - aPrior);
                    cExpected = new Bernoulli(cPrior * (aPrior * cLike1T * cLike2F + (1 - aPrior)) / z);
                    aExpected = new Bernoulli(aPrior * sumT / z);
                }
                else if (trial == 2)
                {
                    b.ObservedValue = false;
                    b2.ObservedValue = true;
                    double sumT = cPrior * cLike1F * cLike2T + (1 - cPrior) * (1 - cLike1F) * (1 - cLike2T);
                    double z = aPrior * sumT + (1 - aPrior);
                    cExpected = new Bernoulli(cPrior * (aPrior * cLike1F * cLike2T + (1 - aPrior)) / z);
                    aExpected = new Bernoulli(aPrior * sumT / z);
                }
                else
                {
                    b.ObservedValue = false;
                    b2.ObservedValue = false;
                    double sumT = cPrior * cLike1F * cLike2F + (1 - cPrior) * (1 - cLike1F) * (1 - cLike2F);
                    double z = aPrior * sumT + (1 - aPrior);
                    cExpected = new Bernoulli(cPrior * (aPrior * cLike1F * cLike2F + (1 - aPrior)) / z);
                    aExpected = new Bernoulli(aPrior * sumT / z);
                }
                Bernoulli cActual = engine.Infer<Bernoulli>(c);
                Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
                Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
                Bernoulli aActual = engine.Infer<Bernoulli>(a);
                Console.WriteLine("a = {0} should be {1}", aActual, aExpected);
                Assert.True(aExpected.MaxDiff(aActual) < 1e-10);
            }
        }

        /// <summary>
        /// Fails because c_marginal_F is never assigned when b.ObservedValue = false.
        /// Not clear what the correct behavior should be.
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")]
        public void InferVariableInsideDeterministicGate()
        {
            double aPrior = 0.2;
            double cPrior = 0.1;
            Variable<bool> a = Variable.Bernoulli(aPrior).Named("a");
            Variable<bool> b = Variable.New<bool>().Named("b");
            Variable<int> c;
            using (Variable.If(b))
            {
                c = Variable.Discrete(cPrior, 1 - cPrior).Named("c");
            }
            using (Variable.If(a))
            {
                using (Variable.If(b))
                {
                    Variable.ConstrainEqualRandom(c, new Discrete(1.0, 0.0));
                }
            }
            InferenceEngine engine = new InferenceEngine();
            Bernoulli aExpected;
            Discrete cExpected;
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial == 1)
                {
                    b.ObservedValue = true;
                    // p(a,b,c) = 0.2^a 0.8^(1-a) 0.1^c 0.9^(1-c) 0^(ab(1-c))
                    double z = aPrior * cPrior + (1 - aPrior);
                    cExpected = new Discrete(cPrior / z, (1 - cPrior) * (1 - aPrior) / z);
                    aExpected = new Bernoulli(aPrior * cPrior / z);
                }
                else
                {
                    b.ObservedValue = false;
                    cExpected = new Discrete(0.5, 0.5);
                    aExpected = new Bernoulli(aPrior);
                }
                Discrete cActual = engine.Infer<Discrete>(c);
                Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
                Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
                Bernoulli aActual = engine.Infer<Bernoulli>(a);
                Console.WriteLine("a = {0} should be {1}", aActual, aExpected);
                Assert.True(aExpected.MaxDiff(aActual) < 1e-10);
            }
        }

        [Fact]
        public void IfObservedCopyTest()
        {
            Variable<bool> b = Variable.New<bool>().Named("b");
            Variable<bool> c = Variable.Bernoulli(0.1).Named("c");
            Variable<bool> x = Variable.New<bool>().Named("x");
            using (Variable.If(b))
            {
                x.SetTo(Variable.Copy(c));
            }
            using (Variable.IfNot(b))
            {
                x.SetTo(!c);
            }
            Variable<bool> y = (!x).Named("y");
            InferenceEngine engine = new InferenceEngine();
            Bernoulli yExpected;
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial % 2 == 0)
                {
                    b.ObservedValue = true;
                    yExpected = new Bernoulli(0.9);
                }
                else
                {
                    b.ObservedValue = false;
                    yExpected = new Bernoulli(0.1);
                }
                Bernoulli yActual = engine.Infer<Bernoulli>(y);
                Console.WriteLine("y = {0} should be {1}", yActual, yExpected);
                Assert.True(yExpected.MaxDiff(yActual) < 1e-10);
            }
        }

        [Fact]
        public void IfObservedIfRandomConditionTest()
        {
            Variable<bool> a = Variable.Bernoulli(0.2).Named("a");
            Variable<bool> b = Variable.New<bool>().Named("b");
            Variable<bool> c = Variable.Bernoulli(0.1).Named("c");
            using (Variable.If(b))
            {
                using (Variable.If(a))
                {
                    Variable.ConstrainTrue(c);
                }
            }
            InferenceEngine engine = new InferenceEngine();
            Bernoulli cExpected, aExpected;
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial == 0)
                {
                    b.ObservedValue = true;
                    // p(a,b,c) = 0.2^a 0.8^(1-a) 0.1^c 0.9^(1-c) 0^(ab(1-c))
                    double z = 0.2 * 0.1 + 0.8;
                    cExpected = new Bernoulli(0.1 / z);
                    aExpected = new Bernoulli(0.2 * 0.1 / z);
                }
                else
                {
                    b.ObservedValue = false;
                    cExpected = new Bernoulli(0.1);
                    aExpected = new Bernoulli(0.2);
                }
                Bernoulli cActual = engine.Infer<Bernoulli>(c);
                Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
                Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
                Bernoulli aActual = engine.Infer<Bernoulli>(a);
                Console.WriteLine("a = {0} should be {1}", aActual, aExpected);
                Assert.True(aExpected.MaxDiff(aActual) < 1e-10);
            }
        }

        [Fact]
        public void IfObservedIfObservedConditionTest()
        {
            Variable<bool> a = Variable.New<bool>().Named("a");
            Variable<bool> b = Variable.New<bool>().Named("b");
            Variable<bool> c = Variable.Bernoulli(0.1).Named("c");
            using (Variable.If(a))
            {
                using (Variable.If(b))
                {
                    Variable.ConstrainTrue(c);
                }
            }
            InferenceEngine engine = new InferenceEngine();
            Bernoulli cExpected;
            for (int aTrial = 0; aTrial < 2; aTrial++)
            {
                a.ObservedValue = (aTrial == 0);
                for (int bTrial = 0; bTrial < 2; bTrial++)
                {
                    b.ObservedValue = (bTrial == 0);
                    if (a.ObservedValue && b.ObservedValue)
                    {
                        cExpected = new Bernoulli(1);
                    }
                    else
                    {
                        cExpected = new Bernoulli(0.1);
                    }
                    Bernoulli cActual = engine.Infer<Bernoulli>(c);
                    Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
                    Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
                }
            }
        }

        [Fact]
        public void IfObservedConditionArrayTest2()
        {
            Range i = new Range(2).Named("i");
            VariableArray<bool> b = Variable.Observed(new bool[] { true, false }, i).Named("b");
            //VariableArray<bool> b = Variable.Array<bool>(i).Named("b");
            Variable<bool> c = Variable.Bernoulli(0.1).Named("c");
            using (Variable.ForEach(i))
            {
                //b[i] = Variable.Bernoulli(0.3);
                using (Variable.If(b[i]))
                {
                    Variable.ConstrainEqualRandom(c, new Bernoulli(0.2));
                }
            }
            InferenceEngine engine = new InferenceEngine();
            Bernoulli cExpected = new Bernoulli(0.1 * 0.2 / (0.1 * 0.2 + 0.9 * 0.8));
            object cActual = engine.Infer(c);
            Console.WriteLine(StringUtil.JoinColumns("c = ", cActual, " should be ", cExpected));
            Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
        }

        [Fact]
        public void IfObservedConditionArrayTest()
        {
            Range i = new Range(2).Named("i");
            VariableArray<bool> b = Variable.Observed(new bool[] { true, false }, i).Named("b");
            VariableArray<bool> c = Variable.Array<bool>(i).Named("c");
            using (Variable.ForEach(i))
            {
                c[i] = Variable.Bernoulli(0.1);
                using (Variable.If(b[i]))
                {
                    Variable.ConstrainTrue(c[i]);
                }
            }
            InferenceEngine engine = new InferenceEngine();
            IDistribution<bool[]> cExpected = Distribution<bool>.Array(new Bernoulli[2]
                {
                    new Bernoulli(1), new Bernoulli(0.1)
                });
            object cActual = engine.Infer(c);
            Console.WriteLine(StringUtil.JoinColumns("c = ", cActual, " should be ", cExpected));
            Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
        }

        [Fact]
        public void IfObservedConstrainTrueTest()
        {
            Variable<bool> b = Variable.New<bool>().Named("b");
            Variable<bool> c = Variable.Bernoulli(0.1).Named("c");
            using (Variable.If(b))
            {
                Variable.ConstrainTrue(c);
            }
            InferenceEngine engine = new InferenceEngine();
            Bernoulli cExpected;
            for (int iter = 0; iter < 2; iter++)
            {
                for (int trial = 0; trial < 2; trial++)
                {
                    if (trial == 0)
                    {
                        b.ObservedValue = true;
                        cExpected = new Bernoulli(1);
                    }
                    else
                    {
                        b.ObservedValue = false;
                        cExpected = new Bernoulli(0.1);
                    }
                    Bernoulli cActual = engine.Infer<Bernoulli>(c);
                    Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
                    Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
                }
            }
        }

        [Fact]
        public void CaseObservedConditionTest()
        {
            Variable<int> b = Variable.New<int>().Named("b");
            Bernoulli cPrior = new Bernoulli(0.1);
            Variable<bool> c = Variable.Bernoulli(0.1).Named("c");
            VariableArray<Bernoulli> cLike = Variable.Observed(new Bernoulli[] { new Bernoulli(0.2), new Bernoulli(0.3) }).Named("cLike");
            using (Variable.Case(b, 0))
            {
                Variable.ConstrainEqualRandom(c, cLike[0]);
            }
            using (Variable.Case(b, 1))
            {
                Variable.ConstrainEqualRandom(c, cLike[1]);
            }
            InferenceEngine engine = new InferenceEngine();
            Bernoulli cExpected;
            for (int iter = 0; iter < 2; iter++)
            {
                for (int trial = 0; trial < 3; trial++)
                {
                    b.ObservedValue = trial;
                    if (trial == 0)
                    {
                        cExpected = cPrior * cLike.ObservedValue[0];
                    }
                    else if (trial == 1)
                    {
                        cExpected = cPrior * cLike.ObservedValue[1];
                    }
                    else
                    {
                        cExpected = cPrior;
                    }
                    Bernoulli cActual = engine.Infer<Bernoulli>(c);
                    Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
                    Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
                }
            }
        }

        [Fact]
        public void CaseObservedConditionTest2()
        {
            Variable<int> b = Variable.New<int>().Named("b");
            Bernoulli cPrior = new Bernoulli(0.1);
            VariableArray<Bernoulli> cLike = Variable.Observed(new Bernoulli[] { new Bernoulli(0.2), new Bernoulli(0.3) }).Named("cLike");
            Range item = new Range((b+1).Named("bPlus1")).Named("item");
            VariableArray<bool> c = Variable.Array<bool>(item).Named("c");
            c[item] = Variable.Bernoulli(0.1).ForEach(item);
            using (Variable.Case(b, 0))
            {
                Variable.ConstrainEqualRandom(c[0], cLike[0]);
            }
            using (Variable.Case(b, 1))
            {
                Variable.ConstrainEqualRandom(c[1], cLike[1]);
            }
            InferenceEngine engine = new InferenceEngine();
            BernoulliArray cExpected;
            for (int iter = 0; iter < 2; iter++)
            {
                for (int trial = 0; trial < 3; trial++)
                {
                    b.ObservedValue = trial;
                    if (trial == 0)
                    {
                        cExpected = new BernoulliArray(new Bernoulli[] { cPrior * cLike.ObservedValue[0] });
                    }
                    else if (trial == 1)
                    {
                        cExpected = new BernoulliArray(new Bernoulli[] { cPrior, cPrior * cLike.ObservedValue[1] });
                    }
                    else
                    {
                        cExpected = new BernoulliArray(new Bernoulli[] { cPrior, cPrior, cPrior });
                    }
                    IList<Bernoulli> cActual = engine.Infer<IList<Bernoulli>>(c);
                    Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
                    Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
                }
            }
        }

        [Fact]
        public void IfObservedConstrainTrueElseTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<bool> b = Variable.New<bool>().Named("b");
            Variable<bool> c = Variable.Bernoulli(0.1).Named("c");
            using (Variable.If(b))
            {
                Variable.ConstrainTrue(c);
            }
            using (Variable.IfNot(b))
            {
                Variable.ConstrainFalse(c);
            }
            block.CloseBlock();
            InferenceEngine engine = new InferenceEngine();
            for (int trial = 0; trial < 2; trial++)
            {
                double evExpected;
                Bernoulli cExpected;
                if (trial % 2 == 0)
                {
                    b.ObservedValue = true;
                    cExpected = new Bernoulli(1);
                    evExpected = System.Math.Log(0.1);
                }
                else
                {
                    b.ObservedValue = false;
                    cExpected = new Bernoulli(0);
                    evExpected = System.Math.Log(0.9);
                }
                Bernoulli cActual = engine.Infer<Bernoulli>(c);
                Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
                Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-10);
            }
        }

        [Fact]
        public void IfObservedConstrainEqualRandomTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<bool> b = Variable.New<bool>().Named("b");
            double cPrior = 0.9;
            Variable<bool> c = Variable.Bernoulli(cPrior).Named("c");
            double cLike1 = 0.1, cLike2 = 0.2, cLike3 = 0.3, cLike4 = 0.4, cLike5 = 0.6;
            using (Variable.If(b))
            {
                Variable.ConstrainEqualRandom(c, new Bernoulli(cLike1));
                Variable.ConstrainEqualRandom(c, new Bernoulli(cLike2));
            }
            using (Variable.IfNot(b))
            {
                Variable.ConstrainEqualRandom(c, new Bernoulli(cLike3));
                Variable.ConstrainEqualRandom(c, new Bernoulli(cLike4));
                Variable.ConstrainEqualRandom(c, new Bernoulli(cLike5));
            }
            block.CloseBlock();
            InferenceEngine engine = new InferenceEngine();
            for (int trial = 0; trial < 2; trial++)
            {
                double evExpected;
                Bernoulli cExpected;
                if (trial % 2 == 0)
                {
                    b.ObservedValue = true;
                    double z = cPrior * cLike1 * cLike2 + (1 - cPrior) * (1 - cLike1) * (1 - cLike2);
                    cExpected = new Bernoulli(cPrior * cLike1 * cLike2 / z);
                    evExpected = System.Math.Log(z);
                }
                else
                {
                    b.ObservedValue = false;
                    double z = cPrior * cLike3 * cLike4 * cLike5 + (1 - cPrior) * (1 - cLike3) * (1 - cLike4) * (1 - cLike5);
                    cExpected = new Bernoulli(cPrior * cLike3 * cLike4 * cLike5 / z);
                    evExpected = System.Math.Log(z);
                }
                Bernoulli cActual = engine.Infer<Bernoulli>(c);
                Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
                Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-10);
            }
        }

        [Fact]
        public void IfNotObservedConditionTest()
        {
            Variable<bool> b = Variable.New<bool>().Named("b");
            Variable<bool> c = Variable.Bernoulli(0.1).Named("c");
            using (Variable.IfNot(b))
            {
                Variable.ConstrainTrue(c);
            }
            InferenceEngine engine = new InferenceEngine();
            Bernoulli cExpected;
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial % 2 == 0)
                {
                    b.ObservedValue = false;
                    cExpected = new Bernoulli(1);
                }
                else
                {
                    b.ObservedValue = true;
                    cExpected = new Bernoulli(0.1);
                }
                Bernoulli cActual = engine.Infer<Bernoulli>(c);
                Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
                Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
            }
        }

        [Fact]
        public void SwitchObservedConditionGaussianMeansTest()
        {
            int nComponents = 2;
            Range component = new Range(nComponents).Named("component");
            Variable<int> nItemsObs = Variable.New<int>().Named("nItem");
            Range itemObs = new Range(nItemsObs).Named("item");
            VariableArray<double> means = Variable.Array<double>(component).Named("means");
            means[component] = Variable.GaussianFromMeanAndVariance(0, 10).ForEach(component);

            VariableArray<double> x = Variable.Array<double>(itemObs).Named("x");
            VariableArray<int> xLabel = Variable.Array<int>(itemObs).Named("xLabel");

            using (Variable.ForEach(itemObs))
            {
                xLabel[itemObs] = Variable.Discrete(component, new double[] { .5, .5 });
                using (Variable.Switch(xLabel[itemObs]))
                {
                    x[itemObs] = Variable.GaussianFromMeanAndPrecision(means[xLabel[itemObs]], 1);
                }
            }

            x.ObservedValue = new double[] { -1, 4 };
            xLabel.ObservedValue = new int[] { 0, 1 };
            nItemsObs.ObservedValue = x.ObservedValue.Length;

            InferenceEngine ie = new InferenceEngine();
            Gaussian[] meansExpectedArray = new Gaussian[2];
            Gaussian prior = new Gaussian(0, 10);
            for (int i = 0; i < 2; i++)
            {
                meansExpectedArray[i] = prior * (new Gaussian(x.ObservedValue[i], 1));
            }
            IDistribution<double[]> meansExpected = Distribution<double>.Array(meansExpectedArray);
            object meansActual = ie.Infer(means);
            Console.WriteLine(StringUtil.JoinColumns("means = ", meansActual, " should be ", meansExpected));
            Assert.True(meansExpected.MaxDiff(meansActual) < 1e-10);
        }

        [Fact]
        public void SwitchExitObservedConditionTest()
        {
            Range item = new Range(2);
            VariableArray<double> probs = Variable.Constant(new double[] { 0.1, 0.2 }, item).Named("probs");
            VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
            bools[item] = Variable.Bernoulli(probs[item]);
            Variable<int> index = Variable.DiscreteUniform(item).Named("index");
            Variable<bool> c = Variable.New<bool>().Named("c");
            using (Variable.Switch(index))
            {
                c.SetTo(!bools[index]);
            }
            double q = 0.4;
            Variable.ConstrainEqualRandom(c, new Bernoulli(q));
            index.ObservedValue = 1;

            InferenceEngine engine = new InferenceEngine();
            double prior = 1 - probs.ObservedValue[index.ObservedValue];
            double z = prior * q + (1 - prior) * (1 - q);
            Bernoulli cExpected = new Bernoulli(prior * q / z);
            Bernoulli cActual = engine.Infer<Bernoulli>(c);
            Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
            Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
        }

        [Fact]
        public void SwitchExitObservedConditionTest2()
        {
            Range item = new Range(2);
            VariableArray<double> probs = Variable.Constant(new double[] { 0.1, 0.2 }, item).Named("probs");
            VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
            bools[item] = Variable.Bernoulli(probs[item]);
            Variable<int> index = Variable.DiscreteUniform(item).Named("index");
            Variable<int> indexCopy = Variable.Copy(index).Named("indexCopy");
            Variable<bool> c = Variable.New<bool>().Named("c");
            using (Variable.Switch(indexCopy))
            {
                c.SetTo(!bools[indexCopy]);
            }
            double q = 0.4;
            Variable.ConstrainEqualRandom(c, new Bernoulli(q));
            index.ObservedValue = 1;

            InferenceEngine engine = new InferenceEngine();
            double prior = 1 - probs.ObservedValue[index.ObservedValue];
            double z = prior * q + (1 - prior) * (1 - q);
            Bernoulli cExpected = new Bernoulli(prior * q / z);
            Bernoulli cActual = engine.Infer<Bernoulli>(c);
            Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
            Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
        }

        [Fact]
        public void SwitchExitObservedConditionTest3()
        {
            Range item = new Range(2);
            VariableArray<double> probs = Variable.Constant(new double[] { 0.1, 0.2 }, item).Named("probs");
            VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
            bools[item] = Variable.Bernoulli(probs[item]);
            Variable<int> index = Variable.DiscreteUniform(item).Named("index");
            Variable<int> indexCopy = Variable.Copy(index).Named("indexCopy");
            Range item2 = new Range(1);
            VariableArray<bool> c = Variable.Array<bool>(item2).Named("c");
            using (Variable.ForEach(item2))
            {
                using (Variable.Switch(indexCopy))
                {
                    c[item2] = !bools[indexCopy];
                }
            }
            double q = 0.4;
            Variable.ConstrainEqualRandom(c[item2], new Bernoulli(q));
            index.ObservedValue = 1;

            InferenceEngine engine = new InferenceEngine();
            double prior = 1 - probs.ObservedValue[index.ObservedValue];
            double z = prior * q + (1 - prior) * (1 - q);
            var cExpected = new BernoulliArray(item2.SizeAsInt, i => new Bernoulli(prior * q / z));
            var cActual = engine.Infer<IList<Bernoulli>>(c);
            Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
            Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void SwitchAllZeroTest()
        {
            Range item = new Range(2);
            VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
            bools.ObservedValue = new bool[] { true, false };
            Variable<int> index = Variable.DiscreteUniform(item).Named("index");
            Variable<bool> x = Variable.New<bool>().Named("x");
            using (Variable.Switch(index))
            {
                x.SetTo(bools[index]);
            }
            var y = (!x).Named("y");
            y.ObservedValue = false;

            InferenceEngine engine = new InferenceEngine();
            for (int iter = 0; iter < 2; iter++)
            {
                if (iter == 0) engine.Algorithm = new VariationalMessagePassing();
                else engine.Algorithm = new ExpectationPropagation();
                var indexActual = engine.Infer<Discrete>(index);
                Discrete indexExpected = Discrete.PointMass(0, item.SizeAsInt);
                Console.WriteLine("index = {0} should be {1}", indexActual, indexExpected);
                Assert.True(MMath.AbsDiff(indexExpected[0], indexActual[0], 1e-6) < 1e-10);
            }
        }

        [Fact]
        public void IfRandomForEachSwitchRandomTest()
        {
            Range item = new Range(2).Named("item");
            Range item2 = new Range(2).Named("item2");

            VariableArray<Vector> x = Variable.Array<Vector>(item).Named("x");
            x[item] = Variable.Dirichlet(item, Vector.FromArray(0.3, 0.7)).ForEach(item);

            VariableArray<int> indices = Variable.Array<int>(item2).Named("indices").Attrib(new ValueRange(item));
            indices[item2] = Variable.Discrete(0.2, 0.8).ForEach(item2);

            VariableArray<int> y = Variable.Array<int>(item2).Named("y");

            double cPrior = 0.1;
            Variable<bool> c = Variable.Bernoulli(cPrior).Named("c");

            using (Variable.If(c))
            {
                using (Variable.ForEach(item2))
                {
                    using (Variable.Switch(indices[item2]))
                    {
                        y[item2] = Variable.Discrete(x[indices[item2]]);
                    }
                }
            }
            using (Variable.IfNot(c))
            {
                using (Variable.ForEach(item2))
                {
                    using (Variable.Switch(indices[item2]))
                    {
                        y[item2] = Variable.Discrete(x[indices[item2]]);
                    }
                }
            }
            Discrete[] yExpectedArray = new Discrete[item2.SizeAsInt];
            for (int i = 0; i < yExpectedArray.Length; i++)
            {
                yExpectedArray[i] = new Discrete((0.2 * 0.3 + 0.8 * 0.3), (0.2 * 0.7 + 0.8 * 0.7));
            }
            IDistribution<int[]> yExpected = Distribution<int>.Array(yExpectedArray);
            InferenceEngine engine = new InferenceEngine();
            object yActual = engine.Infer(y);
            Console.WriteLine(StringUtil.JoinColumns("y = ", yActual, " should be ", yExpected));
            Assert.True(yExpected.MaxDiff(yActual) < 1e-10);
        }

        [Fact]
        public void IfRandomSwitchConstantTest()
        {
            Variable<bool> a = Variable.Bernoulli(0.2).Named("a");
            Range item = new Range(2);
            VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
            bools[item] = Variable.Bernoulli(0.1).ForEach(item);
            Variable<int> index = Variable.DiscreteUniform(item).Named("index");
            using (Variable.If(a))
            {
                using (Variable.Switch(index))
                {
                    Variable.ConstrainEqualRandom(bools[index], new Bernoulli(0.3));
                }
            }
            index.ObservedValue = 1;

            InferenceEngine engine = new InferenceEngine();
            double zCondA = 0.1 * 0.3 + (1 - 0.1) * (1 - 0.3);
            double z = 0.2 * zCondA + (1 - 0.2);
            Bernoulli aExpected = new Bernoulli(0.2 * zCondA / z);
            Bernoulli aActual = engine.Infer<Bernoulli>(a);
            Console.WriteLine("a = {0} should be {1}", aActual, aExpected);
            Assert.True(aExpected.MaxDiff(aActual) < 1e-10);
            IDistribution<bool[]> boolsExpected = Distribution<bool>.Array(2,
                                                                           delegate (int i)
                                                                               {
                                                                                   if (i == index.ObservedValue) return new Bernoulli((0.2 * 0.1 * 0.3 + (1 - 0.2) * 0.1) / z);
                                                                                   else return new Bernoulli(0.1);
                                                                               });
            DistributionArray<Bernoulli> boolsActual = engine.Infer<DistributionArray<Bernoulli>>(bools);
            Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsActual, " should be ", boolsExpected));
            Assert.True(boolsExpected.MaxDiff(boolsActual) < 1e-10);
        }

        [Fact]
        public void SwitchObservedConditionTest()
        {
            Range item = new Range(2);
            VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
            bools[item] = Variable.Bernoulli(0.1).ForEach(item);
            Variable<int> index = Variable.DiscreteUniform(item).Named("index");
            using (Variable.Switch(index))
            {
                Variable.ConstrainTrue(bools[index]);
            }
            index.ObservedValue = 1;

            InferenceEngine engine = new InferenceEngine();
            IDistribution<bool[]> boolsExpected = Distribution<bool>.Array(2,
                                                                           delegate (int i)
                                                                               {
                                                                                   if (i == index.ObservedValue) return Bernoulli.PointMass(true);
                                                                                   else return new Bernoulli(0.1);
                                                                               });
            DistributionArray<Bernoulli> boolsActual = engine.Infer<DistributionArray<Bernoulli>>(bools);
            Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsActual, " should be ", boolsExpected));
            Assert.True(boolsExpected.MaxDiff(boolsActual) < 1e-10);
        }

        [Fact]
        public void SwitchConstantConditionTest()
        {
            Range item = new Range(2);
            VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
            bools[item] = Variable.Bernoulli(0.1).ForEach(item);
            Variable<int> index = Variable.DiscreteUniform(item).Named("index");
            using (Variable.Switch(index))
            {
                Variable.ConstrainTrue(bools[index]);
            }
            index.ObservedValue = 1;
            index.IsReadOnly = true;

            InferenceEngine engine = new InferenceEngine();
            IDistribution<bool[]> boolsExpected = Distribution<bool>.Array(2,
                                                                           delegate (int i)
                                                                               {
                                                                                   if (i == index.ObservedValue) return Bernoulli.PointMass(true);
                                                                                   else return new Bernoulli(0.1);
                                                                               });
            DistributionArray<Bernoulli> boolsActual = engine.Infer<DistributionArray<Bernoulli>>(bools);
            Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsActual, " should be ", boolsExpected));
            Assert.True(boolsExpected.MaxDiff(boolsActual) < 1e-10);
        }

        [Fact]
        public void SwitchLocalVariableTest()
        {
            Range k = new Range(2).Named("k");
            double p0 = 0.1, p1 = 0.8;
            VariableArray<double> probs = Variable.Constant(new double[] { p0, p1 }, k).Named("probs");
            Variable<int> c = Variable.DiscreteUniform(k).Named("c");
            Variable<bool> x = Variable.New<bool>().Named("x");
            Variable<bool> temp;
            using (Variable.Switch(c))
            {
                temp = Variable.Bernoulli(probs[c]).Named("temp");
            }
            x.SetTo(temp);
            double pXCond = 0.2;
            Variable.ConstrainEqualRandom(x, new Bernoulli(pXCond));
            InferenceEngine engine = new InferenceEngine();
            Discrete actual = engine.Infer<Discrete>(c);
            double sumXCond0 = pXCond * p0 + (1 - pXCond) * (1 - p0);
            double sumXCond1 = pXCond * p1 + (1 - pXCond) * (1 - p1);
            double Z = sumXCond0 + sumXCond1;
            Discrete expected = new Discrete(sumXCond0 / Z, sumXCond1 / Z);
            Console.WriteLine("c = {0} (should be {1})", actual, expected);
            Assert.True(actual.MaxDiff(expected) < 1e-10);
        }

        [Fact]
        public void SwitchRedefinitionError()
        {
            try
            {
                Range k = new Range(2).Named("k");
                double p0 = 0.1, p1 = 0.8;
                VariableArray<double> probs = Variable.Constant(new double[] { p0, p1 }, k).Named("probs");
                Variable<int> c = Variable.DiscreteUniform(k).Named("c");
                Variable<bool> x = Variable.New<bool>().Named("x");
                x.SetTo(Variable.Bernoulli(0.1));
                using (Variable.Switch(c))
                {
                    x.SetTo(Variable.Bernoulli(probs[c]));
                }
                double pXCond = 0.2;
                Variable.ConstrainEqualRandom(x, new Bernoulli(pXCond));
                InferenceEngine engine = new InferenceEngine();
                Discrete actual = engine.Infer<Discrete>(c);
                double sumXCond0 = pXCond * p0 + (1 - pXCond) * (1 - p0);
                double sumXCond1 = pXCond * p1 + (1 - pXCond) * (1 - p1);
                double Z = sumXCond0 + sumXCond1;
                Discrete expected = new Discrete(sumXCond0 / Z, sumXCond1 / Z);
                Console.WriteLine("c = {0} (should be {1})", actual, expected);
                //Assert.True(actual.MaxDiff(expected) < 1e-10);
                Assert.True(false, "Did not throw exception");
            }
            catch (InvalidOperationException exn)
            {
                Console.WriteLine("Correctly threw exception: " + exn);
            }
        }

        [Fact]
        public void SwitchRedefinitionError2()

        {
            Assert.Throws<InvalidOperationException>(() =>
            {

                Range k = new Range(2).Named("k");
                double p0 = 0.1, p1 = 0.8;
                VariableArray<double> probs = Variable.Constant(new double[] { p0, p1 }, k).Named("probs");
                Variable<int> c = Variable.DiscreteUniform(k).Named("c");
                Variable<bool> x = Variable.New<bool>().Named("x");
                using (Variable.Switch(c))
                {
                    x.SetTo(Variable.Bernoulli(probs[c]));
                }
                Assert.True(x.IsDefined);
                x.SetTo(Variable.Bernoulli(0.1));
                double pXCond = 0.2;
                Variable.ConstrainEqualRandom(x, new Bernoulli(pXCond));
                InferenceEngine engine = new InferenceEngine();
                Discrete actual = engine.Infer<Discrete>(c);
                double sumXCond0 = pXCond * p0 + (1 - pXCond) * (1 - p0);
                double sumXCond1 = pXCond * p1 + (1 - pXCond) * (1 - p1);
                double Z = sumXCond0 + sumXCond1;
                Discrete expected = new Discrete(sumXCond0 / Z, sumXCond1 / Z);
                Console.WriteLine("c = {0} (should be {1})", actual, expected);
                //Assert.True(actual.MaxDiff(expected) < 1e-10);

            });
        }

        [Fact]
        public void ArrayDefinedInsideSwitchError()

        {
            Assert.Throws<InvalidOperationException>(() =>
            {

                Range inner = new Range(3).Named("inner");
                Range outer = new Range(2).Named("outer");
                Range middle = new Range(2).Named("middle");
                var y = Variable.Array(Variable.Array(Variable.Array<double>(inner), middle), outer).Named("y");
                var index = Variable.Array(Variable.Array<int>(inner), outer).Named("index");
                var x = Variable.Array<double>(middle).Named("x");
                x[middle] = Variable.GaussianFromMeanAndPrecision(0.0, 0.01).ForEach(middle);

                using (Variable.ForEach(outer))
                {
                    using (Variable.ForEach(inner))
                    {
                        index[outer][inner] = Variable.DiscreteUniform(middle);
                        using (Variable.Switch(index[outer][inner]))
                        {
                            y[outer][index[outer][inner]][inner] = x[index[outer][inner]];
                        }
                    }
                }

                InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
                ie.Infer(y);

            });
        }

        [Fact]
        public void SwitchExitConstraintIfObservedTest()
        {
            Range k = new Range(2).Named("k");
            double p0 = 0.1, p1 = 0.8;
            double pXCond = 0.2;
            VariableArray<double> probs = Variable.Constant(new double[] { p0, p1 }, k).Named("probs");
            VariableArray<double> probs2 = Variable.Constant(new double[] { p1, p0 }, k).Named("probs2");
            Variable<int> c = Variable.DiscreteUniform(k).Named("c");
            Variable<bool> x = Variable.New<bool>().Named("x");
            VariableArray<bool> b = Variable.Array<bool>(k).Named("b");
            using (Variable.Switch(c))
            {
                using (Variable.If(b[c]))
                {
                    x.SetTo(Variable.Bernoulli(probs[c]));
                    Variable.ConstrainEqualRandom(x, new Bernoulli(pXCond));
                }
                using (Variable.IfNot(b[c]))
                {
                    x.SetTo(Variable.Bernoulli(probs2[c]));
                }
            }
            b.ObservedValue = new bool[] { true, true };

            InferenceEngine engine = new InferenceEngine();
            Discrete actual = engine.Infer<Discrete>(c);
            double sumXCond0 = pXCond * p0 + (1 - pXCond) * (1 - p0);
            double sumXCond1 = pXCond * p1 + (1 - pXCond) * (1 - p1);
            double Z = sumXCond0 + sumXCond1;
            Discrete expected = new Discrete(sumXCond0 / Z, sumXCond1 / Z);
            Console.WriteLine("c = {0} (should be {1})", actual, expected);
            Assert.True(actual.MaxDiff(expected) < 1e-10);
        }

        [Fact]
        public void SwitchExitConstraintTest()
        {
            Range k = new Range(2).Named("k");
            double p0 = 0.1, p1 = 0.8;
            VariableArray<double> probs = Variable.Constant(new double[] { p0, p1 }, k).Named("probs");
            Variable<int> c = Variable.DiscreteUniform(k).Named("c");
            Variable<bool> x = Variable.New<bool>().Named("x");
            using (Variable.Switch(c))
            {
                x.SetTo(Variable.Bernoulli(probs[c]));
            }
            double pXCond = 0.2;
            Variable.ConstrainEqualRandom(x, new Bernoulli(pXCond));
            InferenceEngine engine = new InferenceEngine();
            Discrete actual = engine.Infer<Discrete>(c);
            double sumXCond0 = pXCond * p0 + (1 - pXCond) * (1 - p0);
            double sumXCond1 = pXCond * p1 + (1 - pXCond) * (1 - p1);
            double Z = sumXCond0 + sumXCond1;
            Discrete expected = new Discrete(sumXCond0 / Z, sumXCond1 / Z);
            Console.WriteLine("c = {0} (should be {1})", actual, expected);
            Assert.True(actual.MaxDiff(expected) < 1e-10);
        }

        [Fact]
        public void SwitchExitConstraintTest2()
        {
            Range k = new Range(2).Named("k");
            Range xr = new Range(2).Named("xr");
            Vector p0 = Vector.FromArray(0.1, 0.9);
            Vector p1 = Vector.FromArray(0.3, 0.7);
            Vector pXCond = Vector.FromArray(0.2, 0.8);
            VariableArray<Vector> probs = Variable.Constant(new Vector[] { p0, p1 }, k).Named("probs");
            probs.SetValueRange(xr);
            Variable<int> c = Variable.DiscreteUniform(k).Named("c");
            Variable<int> x = Variable.New<int>().Named("x");
            using (Variable.Switch(c))
            {
                x.SetTo(Variable.Discrete(probs[c]));
            }
            Variable.ConstrainEqualRandom(x, new Discrete(pXCond));
            InferenceEngine engine = new InferenceEngine();
            Discrete actual = engine.Infer<Discrete>(c);
            double sumXCond0 = p0.Inner(pXCond);
            double sumXCond1 = p1.Inner(pXCond);
            double Z = sumXCond0 + sumXCond1;
            Discrete expected = new Discrete(sumXCond0 / Z, sumXCond1 / Z);
            Console.WriteLine("c = {0} (should be {1})", actual, expected);
            Assert.True(actual.MaxDiff(expected) < 1e-10);
        }

        [Fact]
        public void SwitchExitConstraintTest3()
        {
            Range k = new Range(2).Named("k");
            Vector p0 = Vector.FromArray(0.1, 0.9);
            Vector p1 = Vector.FromArray(0.3, 0.7);
            Vector pXCond = Vector.FromArray(0.2, 0.8);
            VariableArray<Dirichlet> pprobs = Variable.Constant(new Dirichlet[] { new Dirichlet(p0), new Dirichlet(p1) }, k).Named("pprobs");
            VariableArray<Vector> probs = Variable.Array<Vector>(k).Named("probs");
            probs[k] = Variable.Random<Vector, Dirichlet>(pprobs[k]);
            Variable<int> c = Variable.DiscreteUniform(k).Named("c");
            Variable<int> x = Variable.New<int>().Named("x");
            using (Variable.Switch(c))
            {
                x.SetTo(Variable.Discrete(probs[c]));
            }
            Variable.ConstrainEqualRandom(x, new Discrete(pXCond));
            InferenceEngine engine = new InferenceEngine();
            Discrete actual = engine.Infer<Discrete>(c);
            double sumXCond0 = p0.Inner(pXCond);
            double sumXCond1 = p1.Inner(pXCond);
            double Z = sumXCond0 + sumXCond1;
            Discrete expected = new Discrete(sumXCond0 / Z, sumXCond1 / Z);
            Console.WriteLine("c = {0} (should be {1})", actual, expected);
            Assert.True(actual.MaxDiff(expected) < 1e-10);
        }

        [Fact]
        public void SwitchJaggedConstantArrayTest()
        {
            Range outer = new Range(2);
            VariableArray<int> lengths = Variable.Constant(new int[] { 2, 3 }, outer).Named("lengths");
            Range inner = new Range(lengths[outer]);
            var means = Variable.Constant(new double[][] { new double[] { 1, 2 }, new double[] { 3, 4, 5 } }, outer, inner);
            Variable<int> c = Variable.DiscreteUniform(outer);
            Variable<double> x = Variable.New<double>();
            using (Variable.Switch(c))
            {
                x.SetTo(Variable.GaussianFromMeanAndVariance(Variable.Sum(means[c]), 1));
            }
            InferenceEngine engine = new InferenceEngine();
            double m0 = 3;
            double m1 = 12;
            double mx = (m0 + m1) / 2;
            double diff = (m0 - m1);
            double vx = 1.0 + 0.5 * 0.5 * diff * diff;
            Gaussian expected = new Gaussian(mx, vx);
            Gaussian actual = engine.Infer<Gaussian>(x);
            Console.WriteLine("x = {0} (should be {1})", actual, expected);
            Assert.True(expected.MaxDiff(actual) < 1e-10);
        }

        [Fact]
        public void SwitchConstantArrayTest()
        {
            Range k = new Range(2);
            VariableArray<double> means = Variable.Constant<double>(new double[] { 1, 2 }, k);
            Variable<int> c = Variable.DiscreteUniform(k);
            Variable<double> x = Variable.New<double>();
            using (Variable.Switch(c))
            {
                x.SetTo(Variable.GaussianFromMeanAndVariance(means[c], 1));
            }
            InferenceEngine engine = new InferenceEngine();
            Gaussian expected = new Gaussian(1.5, 1.25);
            Gaussian actual = engine.Infer<Gaussian>(x);
            Console.WriteLine("x = {0} (should be {1})", actual, expected);
            Assert.True(expected.MaxDiff(actual) < 1e-10);
        }

        [Fact]
        public void SwitchGivenArrayTest()
        {
            Range k = new Range(2);
            VariableArray<double> means = Variable.Observed<double>(new double[] { 1, 2 }, k);
            Variable<int> c = Variable.DiscreteUniform(k);
            Variable<double> x = Variable.New<double>();
            using (Variable.Switch(c))
            {
                x.SetTo(Variable.GaussianFromMeanAndVariance(means[c], 1));
            }
            InferenceEngine engine = new InferenceEngine();
            Gaussian expected = new Gaussian(1.5, 1.25);
            Gaussian actual = engine.Infer<Gaussian>(x);
            Console.WriteLine("x = {0} (should be {1})", actual, expected);
            Assert.True(expected.MaxDiff(actual) < 1e-10);
        }

        [Fact]
        public void IfWithoutForEachError()

        {
            Assert.Throws<InvalidOperationException>(() =>
            {

                Range item = new Range(2).Named("item");
                VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
                bools[item] = Variable.Bernoulli(0.1).ForEach(item);
                using (Variable.If(bools[item]))
                {
                    Variable.ConstrainTrue(Variable.Bernoulli(0.5));
                }

            });
        }

        [Fact]
        public void DuplicateIfError()

        {
            Assert.Throws<InvalidOperationException>(() =>
            {

                Variable<bool> b = Variable.Bernoulli(0.5).Named("b");
                using (Variable.If(b))
                {
                    using (Variable.If(b))
                    {
                        Variable.ConstrainTrue(Variable.Bernoulli(0.5));
                    }
                }

            });
        }

        [Fact]
        public void BadNestingError()

        {
            Assert.Throws<InvalidOperationException>(() =>
            {

                Range item = new Range(2).Named("item");
                Variable<bool> b = Variable.Bernoulli(0.5).Named("b");
                try
                {
                    IfBlock block1 = Variable.If(b);
                    ForEachBlock block2 = Variable.ForEach(item);
                    block1.CloseBlock();
                    block2.CloseBlock();
                }
                finally
                {
                    StatementBlock.CloseAllBlocks();
                }

            });
        }

        [Fact]
        public void AssignmentInGateError()

        {
            Assert.Throws<InvalidOperationException>(() =>
            {

                double priorB = 0.1;
                double pXCondT = 0.3;
                double pXCondF = 0.4;

                Variable<bool> b = Variable.Bernoulli(priorB).Named("b");
                Variable<bool> x = Variable.New<bool>().Named("x");
                using (Variable.If(b))
                {
                    x = Variable.Bernoulli(pXCondT);
                    // x.SetTo(Variable.Bernoulli(pXCondT)); - this is the correct syntax
                }
                using (Variable.IfNot(b))
                {
                    x = Variable.Bernoulli(pXCondF);
                    // x.SetTo(Variable.Bernoulli(pXCondF)); - this is the correct syntax
                }
                Variable<bool> y = !x;

                InferenceEngine ie = new InferenceEngine();
                Bernoulli bDist = ie.Infer<Bernoulli>(b);
                Bernoulli xDist = ie.Infer<Bernoulli>(x);
                Bernoulli yDist = ie.Infer<Bernoulli>(y);

            });
        }

        [Fact]
        public void AssignmentInGateError2()

        {
            Assert.Throws<InvalidOperationException>(() =>
            {

                double priorB = 0.1;
                double pXCondT = 0.3;
                double pXCondF = 0.4;

                Range r = new Range(1).Named("r");
                Variable<bool> b = Variable.Bernoulli(priorB).Named("b");
                Variable<bool> x = Variable.New<bool>().Named("x");
                using (Variable.If(b))
                {
                    x = Variable.Bernoulli(pXCondT);
                    var a = Variable.Array<bool>(r).Named("a");
                    a[r] = x & Variable.Bernoulli(0.2).ForEach(r);
                }
                using (Variable.IfNot(b))
                {
                    x = Variable.Bernoulli(pXCondF);
                }
                Variable<bool> y = !x;

                InferenceEngine ie = new InferenceEngine();
                Bernoulli bDist = ie.Infer<Bernoulli>(b);
                Bernoulli xDist = ie.Infer<Bernoulli>(x);
                Bernoulli yDist = ie.Infer<Bernoulli>(y);

            });
        }

        [Fact]
        public void PartialDefinitionError()

        {
            Assert.Throws<CompilationFailedException>(() =>
            {

                double priorB = 0.1;
                double pXCondT = 0.3;

                Variable<bool> b = Variable.Bernoulli(priorB).Named("b");
                Variable<bool> x = Variable.New<bool>().Named("x");
                using (Variable.If(b))
                {
                    x.SetTo(Variable.Bernoulli(pXCondT));
                }
                Variable<bool> y = !x;

                InferenceEngine ie = new InferenceEngine();
                Bernoulli bDist = ie.Infer<Bernoulli>(b);
                Bernoulli xDist = ie.Infer<Bernoulli>(x);
                Bernoulli yDist = ie.Infer<Bernoulli>(y);

            });
        }

        [Fact]
        public void PartialDefinitionError2()

        {
            Assert.Throws<CompilationFailedException>(() =>
            {

                double priorB = 0.1;
                double pXCondT = 0.3;

                Variable<bool> b = Variable.Bernoulli(priorB).Named("b");
                Variable<bool> x = Variable.New<bool>().Named("x");
                using (Variable.If(b))
                {
                    x.SetTo(Variable.Bernoulli(pXCondT));
                }
                using (Variable.IfNot(b))
                {
                    Variable.ConstrainTrue(x);
                }
                Variable<bool> y = !x;

                InferenceEngine ie = new InferenceEngine();
                Bernoulli bDist = ie.Infer<Bernoulli>(b);
                Bernoulli xDist = ie.Infer<Bernoulli>(x);
                Bernoulli yDist = ie.Infer<Bernoulli>(y);

            });
        }

        [Fact]
        public void PartialDefinitionError3()

        {
            Assert.Throws<CompilationFailedException>(() =>
            {

                double priorB = 0.1;
                double pXCondT = 0.3;

                Variable<bool> b = Variable.Bernoulli(priorB).Named("b");
                Variable<bool> x = Variable.New<bool>().Named("x");
                using (Variable.IfNot(b))
                {
                    Variable.ConstrainTrue(x);
                }
                using (Variable.If(b))
                {
                    x.SetTo(Variable.Bernoulli(pXCondT));
                }
                Variable<bool> y = !x;

                InferenceEngine ie = new InferenceEngine();
                Bernoulli bDist = ie.Infer<Bernoulli>(b);
                Bernoulli xDist = ie.Infer<Bernoulli>(x);
                Bernoulli yDist = ie.Infer<Bernoulli>(y);

            });
        }

        [Fact]
        public void PartialDefinitionError4()

        {
            Assert.Throws<CompilationFailedException>(() =>
            {

                double priorB = 0.1;
                double pXCondT = 0.3;

                Variable<bool> b = Variable.Bernoulli(priorB).Named("b");
                Variable<bool> x = Variable.New<bool>().Named("x");
                using (Variable.IfNot(b))
                {
                    Variable.ConstrainTrue(x);
                }
                using (Variable.If(b))
                {
                    x.SetTo(Variable.Bernoulli(pXCondT));
                }
                Variable<bool> y = !x;

                InferenceEngine ie = new InferenceEngine();
                b.ObservedValue = false;
                Bernoulli bDist = ie.Infer<Bernoulli>(b);
                Bernoulli xDist = ie.Infer<Bernoulli>(x);
                Bernoulli yDist = ie.Infer<Bernoulli>(y);

            });
        }

        [Fact]
        public void NoDefinitionError()

        {
            Assert.Throws<InferCompilerException>(() =>
            {

                double priorB = 0.1;

                Variable<bool> b = Variable.Bernoulli(priorB).Named("b");
                Variable<bool> x = Variable.New<bool>().Named("x");
                using (Variable.IfNot(b))
                {
                    Variable.ConstrainTrue(x);
                }
                using (Variable.If(b))
                {
                    Variable.ConstrainFalse(x);
                }
                Variable<bool> y = !x;

                InferenceEngine ie = new InferenceEngine();
                Bernoulli bDist = ie.Infer<Bernoulli>(b);
                Bernoulli xDist = ie.Infer<Bernoulli>(x);
                Bernoulli yDist = ie.Infer<Bernoulli>(y);

            });
        }

        /// <summary>
        /// Check that partial definitions are allowed for observed variables.
        /// </summary>
        [Fact]
        public void PartialDefinitionTest()
        {
            double priorB = 0.1;
            double pXCondT = 0.3;

            Variable<bool> b = Variable.Bernoulli(priorB).Named("b");
            Variable<bool> x = Variable.New<bool>().Named("x");
            using (Variable.If(b))
            {
                x.SetTo(Variable.Bernoulli(pXCondT));
            }
            Variable<bool> y = !x;
            x.ObservedValue = true;

            InferenceEngine ie = new InferenceEngine();
            Bernoulli bActual = ie.Infer<Bernoulli>(b);
            Bernoulli bExpected = new Bernoulli(priorB * pXCondT / (priorB * pXCondT + (1 - priorB)));
            Assert.True(bExpected.MaxDiff(bActual) < 1e-10);
        }

        [Fact]
        public void UseBeforeDefError()
        {
            Assert.Throws<CompilationFailedException>(() =>
            {
                Variable<bool> x = Variable.New<bool>().Named("x");
                Variable.ConstrainTrue(x);
                x.SetTo(Variable.Bernoulli(0.3));

                InferenceEngine ie = new InferenceEngine();
                Bernoulli xDist = ie.Infer<Bernoulli>(x);

            });
        }

        [Fact]
        public void UseBeforeDefError2()

        {
            Assert.Throws<CompilationFailedException>(() =>
            {

                double priorB = 0.1;
                double pXCondT = 0.3;

                Variable<bool> b = Variable.Bernoulli(priorB).Named("b");
                Variable<bool> x = Variable.New<bool>().Named("x");
                using (Variable.IfNot(b))
                {
                    Variable.ConstrainTrue(x);
                    x.SetTo(Variable.Bernoulli(pXCondT));
                }
                using (Variable.If(b))
                {
                    x.SetTo(Variable.Bernoulli(pXCondT));
                }
                Variable<bool> y = !x;

                InferenceEngine ie = new InferenceEngine();
                Bernoulli bDist = ie.Infer<Bernoulli>(b);
                Bernoulli xDist = ie.Infer<Bernoulli>(x);
                Bernoulli yDist = ie.Infer<Bernoulli>(y);

            });
        }

        [Fact]
        public void LocalVariableError()

        {
            Assert.Throws<InvalidOperationException>(() =>
            {

                Variable<bool> b = Variable.Bernoulli(0.1).Named("b");
                Variable<bool> x;
                using (Variable.If(b))
                {
                    x = Variable.Bernoulli(0.2).Named("x");
                    Variable.ConstrainEqualRandom(x, new Bernoulli(0.3));
                }
                Variable<bool> y = !x;

                InferenceEngine engine = new InferenceEngine();
                object yActual = engine.Infer(y);

            });
        }

        [Fact]
        public void LocalObservedTest()
        {
            Variable<bool> b = Variable.Bernoulli(0.1).Named("b");
            Variable<bool> x;
            using (Variable.If(b))
            {
                x = Variable.Bernoulli(0.2).Named("x");
                x.ObservedValue = true;
                Variable.ConstrainEqualRandom(x, new Bernoulli(0.3));
            }
            Variable<bool> y = x & Variable.Bernoulli(0.4);

            InferenceEngine engine = new InferenceEngine();
            Bernoulli yActual = engine.Infer<Bernoulli>(y);
            Bernoulli yExpected = new Bernoulli(0.4);
            Console.WriteLine(yActual);
            Assert.True(yExpected.MaxDiff(yActual) < 1e-10);
        }

        internal void ErrorTest()
        {
            Range N = new Range(3).Named("dataCount");
            Range i = new Range(4).Named("i");
            VariableArray<double> mean = Variable.Array<double>(N);
            Variable<int> j = Variable.DiscreteUniform(3);
            //Variable<double> x = Variable.GaussianFromMeanAndVariance(mean[j], 1); - throws exception
            //Variable<double> x = Variable.GaussianFromMeanAndVariance(mean[i], 1); - throws exception
            var x = Variable.Array<double>(i);
            //x[i] = Variable.GaussianFromMeanAndVariance(mean[N], 1);  - throws exception
            //x[i] = Variable.GaussianFromMeanAndVariance(0, 1); - throws exception
        }

        [Fact]
        public void CaseExitTest()
        {
            double priorB = 0.1;
            VariableArray<double> pX = Variable.Observed(new double[] { 0.3, 0.4 }).Named("pX");
            Variable<int> b = Variable.Discrete(pX.Range, new double[] { priorB, 1 - priorB }).Named("b");
            Variable<bool> x = Variable.New<bool>().Named("x");
            using (Variable.Case(b, 0))
            {
                x.SetTo(Variable.Bernoulli(pX[0]));
            }
            using (Variable.Case(b, 1))
            {
                x.SetTo(Variable.Bernoulli(pX[1]));
            }

            InferenceEngine ie = new InferenceEngine();
            Discrete bDist = ie.Infer<Discrete>(b);
            Bernoulli xDist = ie.Infer<Bernoulli>(x);
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) [(pT)^x (1-pT)^(1-x)]^b [(pF)^x (1-pF)^(1-x)]^(1-b)
            double postB = priorB;
            double postX = priorB * pX.ObservedValue[0] + (1 - priorB) * pX.ObservedValue[1];
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            Assert.True(System.Math.Abs(bDist[0] - postB) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - postX) < 1e-4);
        }

        [Fact]
        public void IfCaseExitTest()
        {
            double priorB = 0.1;
            double pXCond0 = 0.3;
            double pXCond1 = 0.4;

            Variable<int> b = Variable.Discrete(new double[] { priorB, 1 - priorB }).Named("b");
            Variable<bool> x = Variable.New<bool>().Named("x");
            using (Variable.If(b == 0))
            {
                x.SetTo(Variable.Bernoulli(pXCond0));
            }
            using (Variable.If(b == 1))
            {
                x.SetTo(Variable.Bernoulli(pXCond1));
            }

            InferenceEngine ie = new InferenceEngine();
            Discrete bDist = ie.Infer<Discrete>(b);
            Bernoulli xDist = ie.Infer<Bernoulli>(x);
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) [(pT)^x (1-pT)^(1-x)]^b [(pF)^x (1-pF)^(1-x)]^(1-b)
            double postB = priorB;
            double postX = priorB * pXCond0 + (1 - priorB) * pXCond1;
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            Assert.True(System.Math.Abs(bDist[0] - postB) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - postX) < 1e-4);
        }

        [Fact]
        public void CaseExitTestFlipped()
        {
            double priorB = 0.1;
            double pXCond0 = 0.3;
            double pXCond1 = 0.4;

            Variable<int> b = Variable.Discrete(new double[] { priorB, 1 - priorB }).Named("b");
            Variable<bool> x = Variable.New<bool>().Named("x");
            using (Variable.Case(b, 1))
            {
                x.SetTo(Variable.Bernoulli(pXCond1));
            }
            using (Variable.Case(b, 0))
            {
                x.SetTo(Variable.Bernoulli(pXCond0));
            }

            InferenceEngine ie = new InferenceEngine();
            Discrete bDist = ie.Infer<Discrete>(b);
            Bernoulli xDist = ie.Infer<Bernoulli>(x);
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) [(pT)^x (1-pT)^(1-x)]^b [(pF)^x (1-pF)^(1-x)]^(1-b)
            double postB = priorB;
            double postX = priorB * pXCond0 + (1 - priorB) * pXCond1;
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            Assert.True(System.Math.Abs(bDist[0] - postB) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - postX) < 1e-4);
        }

        [Fact]
        public void IfExitTest()
        {
            double priorB = 0.1;
            double pXCondT = 0.3;
            double pXCondF = 0.4;

            Variable<bool> b = Variable.Bernoulli(priorB).Named("b");
            Variable<bool> x = Variable.New<bool>().Named("x");
            using (Variable.If(b))
            {
                x.SetTo(Variable.Bernoulli(pXCondT));
            }
            using (Variable.IfNot(b))
            {
                x.SetTo(Variable.Bernoulli(pXCondF));
            }

            InferenceEngine ie = new InferenceEngine();
            Bernoulli bDist = ie.Infer<Bernoulli>(b);
            Bernoulli xDist = ie.Infer<Bernoulli>(x);
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) [(pT)^x (1-pT)^(1-x)]^b [(pF)^x (1-pF)^(1-x)]^(1-b)
            double postB = priorB;
            double postX = priorB * pXCondT + (1 - priorB) * pXCondF;
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - postX) < 1e-4);
        }

        [Fact]
        public void IfExitTestFlipped()
        {
            double priorB = 0.1;
            double pXCondT = 0.3;
            double pXCondF = 0.4;

            Variable<bool> b = Variable.Bernoulli(priorB).Named("b");
            Variable<bool> x = Variable.New<bool>().Named("x");
            using (Variable.IfNot(b))
            {
                x.SetTo(Variable.Bernoulli(pXCondF));
            }
            using (Variable.If(b))
            {
                x.SetTo(Variable.Bernoulli(pXCondT));
            }
            InferenceEngine ie = new InferenceEngine();
            Bernoulli bDist = ie.Infer<Bernoulli>(b);
            Bernoulli xDist = ie.Infer<Bernoulli>(x);
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) [(pT)^x (1-pT)^(1-x)]^b [(pF)^x (1-pF)^(1-x)]^(1-b)
            double postB = priorB;
            double postX = priorB * pXCondT + (1 - priorB) * pXCondF;
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - postX) < 1e-4);
        }

        [Fact]
        public void GatedDeclarationModelTest()
        {
            double bPrior = 0.1;
            double xPrior = 0.4;
            double xCondT = 0.2;

            Variable<bool> b = Variable.Bernoulli(bPrior).Named("b");
            Variable<bool> x;
            using (Variable.If(b))
            {
                x = Variable.Bernoulli(xPrior);
                Variable.ConstrainEqualRandom(x, new Bernoulli(xCondT));
            }

            InferenceEngine ie = new InferenceEngine();
            Bernoulli bDist = ie.Infer<Bernoulli>(b);
            Bernoulli xDist = ie.Infer<Bernoulli>(x);

            // p(b) =propto (pb)^b (1-pb)^(1-b) [sum_x (px)^x (1-px)^(1-x) (pT)^x (1-pT)^(1-x)]^b
            double sumXCondT = xPrior * xCondT + (1 - xPrior) * (1 - xCondT);
            double Z = bPrior * sumXCondT + (1 - bPrior);
            double bPost = bPrior * sumXCondT / Z;
            double xPost = xPrior * xCondT / sumXCondT;
            Console.WriteLine("b = {0} (should be {1})", bDist, bPost);
            Console.WriteLine("x = {0} (should be {1})", xDist, xPost);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - bPost) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - xPost) < 1e-4);
        }

        [Fact]
        public void IfRandomIfRandomConditionTest()
        {
            double aPrior = 0.9;
            double bPrior = 0.1;
            double xPrior = 0.4;
            double xCondTT = 0.2;

            Variable<bool> a = Variable.Bernoulli(aPrior).Named("a");
            Variable<bool> b = Variable.Bernoulli(bPrior).Named("b");
            Variable<bool> x = Variable.Bernoulli(xPrior).Named("x");
            using (Variable.If(a))
            {
                using (Variable.If(b))
                {
                    Variable.ConstrainEqualRandom(x, new Bernoulli(xCondTT));
                }
            }

            InferenceEngine ie = new InferenceEngine();
            Bernoulli aDist = ie.Infer<Bernoulli>(a);
            Bernoulli bDist = ie.Infer<Bernoulli>(b);
            Bernoulli xDist = ie.Infer<Bernoulli>(x);

            double sumCondTT = xPrior * xCondTT + (1 - xPrior) * (1 - xCondTT);
            double sumCondT = bPrior * sumCondTT + (1 - bPrior);
            double Z = aPrior * bPrior * sumCondTT + (1 - aPrior) * bPrior + aPrior * (1 - bPrior) + (1 - aPrior) * (1 - bPrior);
            double aPost = aPrior * (bPrior * sumCondTT + (1 - bPrior)) / Z;
            double bPost = bPrior * (aPrior * sumCondTT + (1 - aPrior)) / Z;
            double xPost = xPrior * (aPrior * bPrior * xCondTT + (1 - aPrior) * bPrior + aPrior * (1 - bPrior) + (1 - aPrior) * (1 - bPrior)) / Z;

            Console.WriteLine("a = {0} (should be {1})", aDist, aPost);
            Console.WriteLine("b = {0} (should be {1})", bDist, bPost);
            Console.WriteLine("x = {0} (should be {1})", xDist, xPost);
            Assert.True(System.Math.Abs(aDist.GetProbTrue() - aPost) < 1e-4);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - bPost) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - xPost) < 1e-4);
        }

        // This test requires sequential because z is not initialized and parallel updates produce random permutations of the true solution.
        [Fact]
        public void MixtureOfThreeGaussians()
        {
            double[] data = { 0, 1, 2, 5, 6, 7, 10, 11, 12 };
            VariableArray<double> x = Variable.Constant(data).Named("x");
            Variable<double> mean1 = Variable.GaussianFromMeanAndPrecision(0, 0.1).Named("mean1");
            Variable<double> mean2 = Variable.GaussianFromMeanAndPrecision(5, 0.1).Named("mean2");
            Variable<double> mean3 = Variable.GaussianFromMeanAndPrecision(10, 0.1).Named("mean3");
            double prec = 10;
            Range i = x.Range.Named("i"); //.Attrib(new Sequential());
            VariableArray<int> z = Variable.Array<int>(i).Named("z");
            using (Variable.ForEach(i))
            {
                z[i] = Variable.DiscreteUniform(3);
                using (Variable.Case(z[i], 0))
                {
                    x[i] = Variable.GaussianFromMeanAndPrecision(mean1, prec);
                }
                using (Variable.Case(z[i], 1))
                {
                    x[i] = Variable.GaussianFromMeanAndPrecision(mean2, prec);
                }
                using (Variable.Case(z[i], 2))
                {
                    x[i] = Variable.GaussianFromMeanAndPrecision(mean3, prec);
                }
            }

            InferenceEngine ie = new InferenceEngine();
            i.AddAttribute(new Sequential());
            Gaussian mean1Actual = ie.Infer<Gaussian>(mean1);
            double mean1Expected = 1;
            Console.WriteLine("mean1 = {0} should be {1}", mean1Actual, mean1Expected);
            Gaussian mean2Actual = ie.Infer<Gaussian>(mean2);
            double mean2Expected = 6;
            Console.WriteLine("mean2 = {0} should be {1}", mean2Actual, mean2Expected);
            Gaussian mean3Actual = ie.Infer<Gaussian>(mean3);
            double mean3Expected = 11;
            Console.WriteLine("mean3 = {0} should be {1}", mean3Actual, mean3Expected);
            Assert.True(System.Math.Abs(mean1Expected - mean1Actual.GetMean()) < 0.2);
            Assert.True(System.Math.Abs(mean2Expected - mean2Actual.GetMean()) < 0.2);
            Assert.True(System.Math.Abs(mean3Expected - mean3Actual.GetMean()) < 0.2);
        }


        [Fact]
        public void MixtureOfManyGaussians()
        {
            // Set up data
            double[] data = { 0, 1, 2, 5, 6, 7, 10, 11, 12 };
            VariableArray<double> x = Variable.Constant(data).Named("x");
            Range i = x.Range;

            // Mixture components
            VariableArray<double> meanPriorMeans = Variable.Constant(new double[] { 1, 2, 3 }).Named("meanPriorMeans");
            Range N = meanPriorMeans.Range.Named("dataCount");
            VariableArray<double> mean = Variable.Array<double>(N).Named("mean");
            mean[N] = Variable.GaussianFromMeanAndPrecision(meanPriorMeans[N], 0.1);
            double prec = 1;

            VariableArray<int> j = Variable.Array<int>(i).Named("j");
            using (Variable.ForEach(i))
            {
                j[i] = Variable.DiscreteUniform(N);
                using (Variable.Switch(j[i]))
                {
                    x[i] = Variable.GaussianFromMeanAndPrecision(mean[j[i]], prec);
                }
            }
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine("mean=" + ie.Infer(mean));
        }


        // z(x) = a1.x+b1 if x+t >= 0, else = a2.x+b2
        // y(x) = z(x) + noise
        [Fact]
        public void GateModelPiecewiseLinear2()
        {
            double trueA1 = 1.2;
            double trueB1 = -0.2;
            double trueA2 = -0.7;
            double trueB2 = 0.3;
            double trueNoisePrec = 10.0;
            double trueT = 0.5;
            //-------------------------------------
            // The model
            //-------------------------------------
            Variable<int> nData = Variable.New<int>().Named("nData");
            Range n = new Range(nData).Named("dataCount");
            VariableArray<double> x = Variable.Array<double>(n).Named("x");
            VariableArray<double> y = Variable.Array<double>(n).Named("y");
            Variable<double> a1 = Variable.GaussianFromMeanAndVariance(0.0, 1.0).Named("a1");
            Variable<double> b1 = Variable.GaussianFromMeanAndVariance(0.0, 1.0).Named("b1");
            Variable<double> a2 = Variable.GaussianFromMeanAndVariance(0.0, 1.0).Named("a2");
            Variable<double> b2 = Variable.GaussianFromMeanAndVariance(0.0, 1.0).Named("b2");
            Variable<double> t = Variable.GaussianFromMeanAndVariance(0.0, 1.0).Named("t");
            Variable<double> noisePrec = Variable.Random(Gamma.PointMass(trueNoisePrec)).Named("noisePrec");
            using (Variable.ForEach(n))
            {
                var g = (x[n] > t);
                using (Variable.IfNot(g))
                {
                    var z = (a1 * x[n]).Named("a1*x") + b1;
                    z.Name = "z1";
                    y[n] = Variable.GaussianFromMeanAndPrecision(z, noisePrec);
                }
                using (Variable.If(g))
                {
                    var z = (a2 * x[n]).Named("a2*x") + b2;
                    z.Name = "z2";
                    y[n] = Variable.GaussianFromMeanAndPrecision(z, noisePrec);
                }
            }


            //-------------------------------------
            // Generate data from the model
            //-------------------------------------
            double trueStd = 1.0 / System.Math.Sqrt(trueNoisePrec);
            int numData = 1000;
            double[] xData = new double[numData];
            double[] yData = new double[numData];
            Rand.Restart(12347);
            for (int i = 0; i < numData / 2; i++)
            {
                // to the left
                double r = 5 * Rand.Double();
                r = trueT - r;
                xData[2 * i] = r;
                yData[2 * i] = trueA1 * r + trueB1 + Rand.Normal(0.0, trueStd);

                // to the right
                r = 5 * Rand.Double();
                r = trueT + r;
                xData[2 * i + 1] = r;
                yData[2 * i + 1] = trueA2 * r + trueB2 + Rand.Normal(0.0, trueStd);
            }

            //-------------------------------------
            // The inference
            //-------------------------------------
            nData.ObservedValue = numData;
            x.ObservedValue = xData;
            y.ObservedValue = yData;
            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            engine.NumberOfIterations = 50;
            double inferredA1 = engine.Infer<Gaussian>(a1).GetMean();
            double inferredB1 = engine.Infer<Gaussian>(b1).GetMean();
            double inferredA2 = engine.Infer<Gaussian>(a2).GetMean();
            double inferredB2 = engine.Infer<Gaussian>(b2).GetMean();
            double inferredT = engine.Infer<Gaussian>(t).GetMean();
            double inferredNoisePrec = engine.Infer<Gamma>(noisePrec).GetMean();

            Console.WriteLine("T = {0} should be {1}", inferredT, trueT);
            Console.WriteLine("A1 = {0} should be {1}", inferredA1, trueA1);
            Console.WriteLine("B1 = {0} should be {1}", inferredB1, trueB1);
            Console.WriteLine("A2 = {0} should be {1}", inferredA2, trueA2);
            Console.WriteLine("B2 = {0} should be {1}", inferredB2, trueB2);
            Console.WriteLine("noisePrec = {0} should be {1}", inferredNoisePrec, trueNoisePrec);

            double tolerance = 0.05;
            Assert.True(System.Math.Abs(trueA1 - inferredA1) < tolerance);
            Assert.True(System.Math.Abs(trueB1 - inferredB1) < tolerance);
            Assert.True(System.Math.Abs(trueA2 - inferredA2) < tolerance);
            Assert.True(System.Math.Abs(trueB2 - inferredB2) < tolerance);
            Assert.True(System.Math.Abs(trueT - inferredT) < tolerance);
            Assert.True(System.Math.Abs(trueNoisePrec - inferredNoisePrec) < tolerance);
        }

        internal void GateEnterPartialPointMassTest()
        {
            var prior = Variable.Observed(default(Beta));
            var p = Variable<double>.Random(prior).Named("p");
            p.AddAttribute(QueryTypes.MarginalDividedByPrior);
            var b = Variable.Bernoulli(1.0 / 3).Named("b");
            using (Variable.If(b))
            {
                Variable.ConstrainTrue(Variable.Bernoulli(p));
            }
            using (Variable.IfNot(b))
            {
                Variable.ConstrainFalse(Variable.Bernoulli(p));
            }

            InferenceEngine engine = new InferenceEngine();
            for (int i = 10; i <= 10; i++)
            {
                double s = System.Math.Exp(i);
                prior.ObservedValue = new Beta(s, 2 * s);
                if (i == 10)
                    prior.ObservedValue = Beta.PointMass(prior.ObservedValue.GetMean());
                Console.WriteLine("{0} {1}", prior.ObservedValue, engine.Infer(p, QueryTypes.MarginalDividedByPrior));
            }
        }

        // Mixture of two Gaussians where we learn the prior on the Bernoulli
        [Fact]
        public void GateModelMixtureOfTwoGaussiansTest()
        {
            GateModelMixtureOfTwoGaussians(new VariationalMessagePassing());
            GateModelMixtureOfTwoGaussians(new ExpectationPropagation());
            GateModelMixtureOfTwoGaussians(new GibbsSampling());
        }

        private void GateModelMixtureOfTwoGaussians(IAlgorithm algorithm)
        {
            // Restart the infer.NET random number generator
            Rand.Restart(12347);

            // Generate some data from the truth
            double trueM1 = 2.0;
            double trueP1 = 3.0;
            double trueM2 = 7.0;
            double trueP2 = 4.0;
            Gaussian trueG1 = Gaussian.FromMeanAndPrecision(trueM1, trueP1);
            Gaussian trueG2 = Gaussian.FromMeanAndPrecision(trueM2, trueP2);
            double truePi = 0.6;
            Bernoulli trueB = new Bernoulli(truePi);
            int totalSamples = 500;
            double[] data = new double[totalSamples];
            int samps1 = 0;
            int samps2 = 0;
            for (int j = 0; j < totalSamples; j++)
            {
                bool bSamp = trueB.Sample();
                if (!bSamp)
                {
                    data[j] = trueG1.Sample();
                    samps1++;
                }
                else
                {
                    data[j] = trueG2.Sample();
                    samps2++;
                }
            }

            VariableArray<double> x = Variable.Observed(data).Named("data");
            //x.QuoteInMSL = false;
            Variable<double> mean1 = Variable.GaussianFromMeanAndPrecision(1, 0.1).Named("mean1");
            Variable<double> mean2 = Variable.GaussianFromMeanAndPrecision(2, 0.1).Named("mean2");
            double precvar = 1.0;
            Variable<double> prec1 = Variable.GammaFromMeanAndVariance(1, precvar).Named("prec1");
            Variable<double> prec2 = Variable.GammaFromMeanAndVariance(2, precvar).Named("prec2");
            Variable<double> pi = Variable.Beta(1, 1).Named("pi");
            //mean1.AddAttribute(new PointEstimate());

            Range i = x.Range;
            VariableArray<bool> z = Variable.Array<bool>(i).Named("z");
            using (Variable.ForEach(i))
            {
                z[i] = Variable.Bernoulli(pi);
                using (Variable.IfNot(z[i]))
                {
                    x[i] = Variable.GaussianFromMeanAndPrecision(mean1, prec1);
                }
                using (Variable.If(z[i]))
                {
                    x[i] = Variable.GaussianFromMeanAndPrecision(mean2, prec2);
                }
            }

            InferenceEngine ie = new InferenceEngine(algorithm);
            if (algorithm is VariationalMessagePassing) ie.NumberOfIterations = 70;
            ie.ModelName = "MixtureOfTwoGaussians";
            Beta inferredPi = (Beta)ie.Infer(pi);
            Gaussian inferredM1 = (Gaussian)ie.Infer(mean1);
            Gaussian inferredM2 = (Gaussian)ie.Infer(mean2);
            Gamma inferredP1 = (Gamma)ie.Infer(prec1);
            Gamma inferredP2 = (Gamma)ie.Infer(prec2);

            double piMean = inferredPi.GetMean();
            if (piMean < 0.5)
            {
                piMean = 1 - piMean;
                Gaussian tmpGaussian = inferredM1;
                inferredM1 = inferredM2;
                inferredM2 = tmpGaussian;
                Gamma tmpGamma = inferredP1;
                inferredP1 = inferredP2;
                inferredP2 = tmpGamma;
            }
            double M1Mean = inferredM1.GetMean();
            double M2Mean = inferredM2.GetMean();
            double P1Mean = inferredP1.GetMean();
            double P2Mean = inferredP2.GetMean();
            // Set assertion tolerance at 2 times the inferred standard deviation
            double scale = 2.0;
            double piTol = scale * System.Math.Sqrt(inferredPi.GetVariance());
            double M1Tol = scale * System.Math.Sqrt(inferredM1.GetVariance());
            double M2Tol = scale * System.Math.Sqrt(inferredM2.GetVariance());
            double P1Tol = scale * System.Math.Sqrt(inferredP1.GetVariance());
            double P2Tol = scale * System.Math.Sqrt(inferredP2.GetVariance());

            Console.WriteLine("Comparing inferred values (+/- two sigma) with true values");
            Console.WriteLine("pi = " + piMean + "(+/-" + piTol + "), true value = " + truePi);
            Console.WriteLine("mean1 = " + M1Mean + "(+/-" + M1Tol + "), true value = " + trueM1);
            Console.WriteLine("mean2 = " + M2Mean + "(+/-" + M2Tol + "), true value = " + trueM2);
            Console.WriteLine("prec1 = " + P1Mean + "(+/-" + P1Tol + "), true value = " + trueP1);
            Console.WriteLine("prec2 = " + P2Mean + "(+/-" + P2Tol + "), true value = " + trueP2);

            Assert.True(System.Math.Abs(truePi - piMean) < piTol);
            Assert.True(System.Math.Abs(trueM1 - M1Mean) < M1Tol);
            Assert.True(System.Math.Abs(trueM2 - M2Mean) < M2Tol);
            Assert.True(System.Math.Abs(trueP1 - P1Mean) < P1Tol);
            Assert.True(System.Math.Abs(trueP2 - P2Mean) < P2Tol);
        }

        // Mixture of three Gaussians where we learn the prior on the Bernoulli
        [Fact]
        public void GateModelMixtureOfThreeGaussians()
        {
            Rand.Restart(0);
            Vector piTrue = Vector.FromArray(0.5, 0.3, 0.2);
            double[] means = new double[] { 1, 2, 3 };
            double precision = 10;
            double[] data = Util.ArrayInit(500, j =>
            {
                int zSample = Rand.Sample(piTrue);
                return Gaussian.Sample(means[zSample], precision);
            });
            VariableArray<double> x = Variable.Observed(data).Named("data");
            Variable<double> mean1 = Variable.GaussianFromMeanAndPrecision(means[0], precision).Named("mean1");
            Variable<double> mean2 = Variable.GaussianFromMeanAndPrecision(means[1], precision).Named("mean2");
            Variable<double> mean3 = Variable.GaussianFromMeanAndPrecision(means[2], precision).Named("mean3");
            Variable<Vector> pi = Variable.Dirichlet(new double[] { 1.0, 1.0, 1.0 }).Named("pi");
            //pi.AddAttribute(new PointEstimate());

            Range i = x.Range;
            VariableArray<int> z = Variable.Array<int>(i).Named("z");
            using (Variable.ForEach(i))
            {
                z[i] = Variable.Discrete(pi);
                using (Variable.Case(z[i], 0))
                {
                    x[i] = Variable.GaussianFromMeanAndPrecision(mean1, precision);
                }
                using (Variable.Case(z[i], 1))
                {
                    x[i] = Variable.GaussianFromMeanAndPrecision(mean2, precision);
                }
                using (Variable.Case(z[i], 2))
                {
                    x[i] = Variable.GaussianFromMeanAndPrecision(mean3, precision);
                }
            }

            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.ModelName = "MixtureOfThreeGaussians";
            var piPost = ie.Infer<Dirichlet>(pi);
            Console.WriteLine("pi = {0} mean = {1}", piPost, piPost.GetMean());
            Console.WriteLine("mean1=" + ie.Infer(mean1));
            Console.WriteLine("mean2=" + ie.Infer(mean2));
            Console.WriteLine("mean3=" + ie.Infer(mean3));
        }

        // Mixture of two VectorGaussians where we learn the prior on the Bernoulli. 
        [Fact]
        public void GateModelMixtureOfTwoVectorGaussiansTest()
        {
            GateModelMixtureOfTwoVectorGaussians(true);
            GateModelMixtureOfTwoVectorGaussians(false);
        }

        private void GateModelMixtureOfTwoVectorGaussians(bool mean1Obs)
        {
            // Restart the infer.NET random number generator
            Rand.Restart(12347);

            // Generate some data from the truth
            Vector trueM1 = Vector.FromArray(new double[] { 2.0, 3.0 });
            Vector trueM2 = Vector.FromArray(new double[] { 7.0, 5.0 });
            PositiveDefiniteMatrix trueP1 = new PositiveDefiniteMatrix(new double[,] { { 3.0, 0.2 }, { 0.2, 2.0 } });
            PositiveDefiniteMatrix trueP2 = new PositiveDefiniteMatrix(new double[,] { { 1.0, -0.4 }, { -0.4, 4.0 } });
            VectorGaussian trueVG1 = VectorGaussian.FromMeanAndPrecision(trueM1, trueP1);
            VectorGaussian trueVG2 = VectorGaussian.FromMeanAndPrecision(trueM2, trueP2);
            double truePi = 0.6;
            Bernoulli trueB = new Bernoulli(truePi);
            int totalSamples = 500;
            Vector[] data = new Vector[totalSamples];
            int samps1 = 0;
            int samps2 = 0;
            //Sampler<VectorGaussian, Vector> g1Sampler = trueVG1.SamplePrep();
            //Sampler<VectorGaussian, Vector> g2Sampler = trueVG2.SamplePrep();
            for (int j = 0; j < totalSamples; j++)
            {
                bool bSamp = trueB.Sample();
                if (!bSamp)
                {
                    data[j] = trueVG1.Sample();
                    samps1++;
                }
                else
                {
                    data[j] = trueVG2.Sample();
                    samps2++;
                }
            }

            VariableArray<Vector> x = Variable.Observed(data).Named("data");
            // priors must not be too broad or else the data will all go to one component
            Variable<Vector> mean1 = Variable.VectorGaussianFromMeanAndPrecision(Vector.Constant(2, 2.0), PositiveDefiniteMatrix.IdentityScaledBy(2, 0.1)).Named("mean1");
            Variable<Vector> mean2 = Variable.VectorGaussianFromMeanAndPrecision(Vector.Constant(2, 5.0), PositiveDefiniteMatrix.IdentityScaledBy(2, 0.1)).Named("mean2");
            Variable<PositiveDefiniteMatrix> prec1 = Variable.WishartFromShapeAndScale(1.0, PositiveDefiniteMatrix.Identity(2)).Named("prec1");
            Variable<PositiveDefiniteMatrix> prec2 = Variable.WishartFromShapeAndScale(1.0, PositiveDefiniteMatrix.Identity(2)).Named("prec2");
            Variable<double> pi = Variable.Beta(1, 1).Named("pi");

            Range i = x.Range;
            VariableArray<bool> z = Variable.Array<bool>(i).Named("z");
            using (Variable.ForEach(i))
            {
                z[i] = Variable.Bernoulli(pi);
                using (Variable.IfNot(z[i]))
                {
                    x[i] = Variable.VectorGaussianFromMeanAndPrecision(mean1, prec1);
                }
                using (Variable.If(z[i]))
                {
                    x[i] = Variable.VectorGaussianFromMeanAndPrecision(mean2, prec2);
                }
            }

            // Initialise parameters at their prior means
            mean1.InitialiseTo(VectorGaussian.FromMeanAndPrecision(Vector.Constant(2, 2.0), PositiveDefiniteMatrix.IdentityScaledBy(2, 1000)));
            mean2.InitialiseTo(VectorGaussian.FromMeanAndPrecision(Vector.Constant(2, 5.0), PositiveDefiniteMatrix.IdentityScaledBy(2, 1000)));
            prec1.InitialiseTo(Wishart.FromShapeAndScale(10.0, PositiveDefiniteMatrix.IdentityScaledBy(2, 0.1)));
            prec2.InitialiseTo(Wishart.FromShapeAndScale(10.0, PositiveDefiniteMatrix.IdentityScaledBy(2, 0.1)));

            if (mean1Obs)
            {
                mean1.ObservedValue = trueM1;
                //mean2.ObservedValue = trueM2;
                prec1.ObservedValue = trueP1;
                //prec2.ObservedValue = trueP2;
            }

            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.ModelName = "MixtureOfTwoVectorGaussians";
            Beta inferredPi = (Beta)ie.Infer(pi);
            VectorGaussian inferredM1 = (VectorGaussian)ie.Infer(mean1);
            VectorGaussian inferredM2 = (VectorGaussian)ie.Infer(mean2);
            Wishart inferredP1 = (Wishart)ie.Infer(prec1);
            Wishart inferredP2 = (Wishart)ie.Infer(prec2);

            double piMean = inferredPi.GetMean();
            if (false && piMean < 0.5)
            {
                piMean = 1 - piMean;
                VectorGaussian tmpGaussian = inferredM1;
                inferredM1 = inferredM2;
                inferredM2 = tmpGaussian;
                Wishart tmpGamma = inferredP1;
                inferredP1 = inferredP2;
                inferredP2 = tmpGamma;
            }
            Vector M1Mean = inferredM1.GetMean();
            Vector M2Mean = inferredM2.GetMean();
            PositiveDefiniteMatrix P1Mean = inferredP1.GetMean();
            PositiveDefiniteMatrix P2Mean = inferredP2.GetMean();
            int vCnt = M1Mean.Count;

            // Set assertion tolerance at 2 times the inferred standard deviation
            double scale = 2.0;
            double piTol = scale * System.Math.Sqrt(inferredPi.GetVariance());

            LowerTriangularMatrix M1Tol = new LowerTriangularMatrix(vCnt, vCnt);
            LowerTriangularMatrix M2Tol = new LowerTriangularMatrix(vCnt, vCnt);
            LowerTriangularMatrix P1Tol = new LowerTriangularMatrix(vCnt, vCnt);
            LowerTriangularMatrix P2Tol = new LowerTriangularMatrix(vCnt, vCnt);
            M1Tol.SetToCholesky(inferredM1.GetVariance());
            M2Tol.SetToCholesky(inferredM2.GetVariance());
            P1Tol.SetToCholesky(inferredP1.GetVariance());
            P2Tol.SetToCholesky(inferredP2.GetVariance());
            M1Tol.SetToProduct(M1Tol, scale);
            M2Tol.SetToProduct(M2Tol, scale);
            P1Tol.SetToProduct(P1Tol, scale);
            P2Tol.SetToProduct(P2Tol, scale);

            // Console output
            Debug.WriteLine("Comparing inferred values (+/- two sigma) with true values");
            Debug.WriteLine("pi = " + piMean + "(+/-" + piTol + "), true value = " + truePi);
            for (int j = 0; j < vCnt; j++)
                Debug.WriteLine("mean1[" + j + "] = " + M1Mean[j] + "(+/-" + M1Tol[j, j] + "), true value = " + trueM1[j]);
            for (int j = 0; j < vCnt; j++)
                Debug.WriteLine("mean2[" + j + "] = " + M2Mean[j] + "(+/-" + M2Tol[j, j] + "), true value = " + trueM2[j]);
            for (int j = 0; j < vCnt; j++)
                Debug.WriteLine("prec1[" + j + "," + j + "] = " + P1Mean[j, j] + "(+/-" + P1Tol[j, j] + "), true value = " + trueP1[j, j]);
            for (int j = 0; j < vCnt; j++)
                Debug.WriteLine("prec2[" + j + "," + j + "] = " + P2Mean[j, j] + "(+/-" + P2Tol[j, j] + "), true value = " + trueP2[j, j]);

            // Tolerance tests
            for (int j = 0; j < vCnt; j++)
                Assert.True(System.Math.Abs(trueM1[j] - M1Mean[j]) <= M1Tol[j, j]);
            for (int j = 0; j < vCnt; j++)
                Assert.True(System.Math.Abs(trueM2[j] - M2Mean[j]) <= M2Tol[j, j]);
            for (int j = 0; j < vCnt; j++)
                Assert.True(System.Math.Abs(trueP1[j, j] - P1Mean[j, j]) <= P1Tol[j, j]);
            for (int j = 0; j < vCnt; j++)
                Assert.True(System.Math.Abs(trueP2[j, j] - P2Mean[j, j]) <= P2Tol[j, j]);
        }

        // Nested mixture of mixtures of Gaussians. We want to learn all discrete parameters
        // and all Gaussian parameters
        [Fact]
        public void GateModelMixtureOfMixtureOfGaussians()
        {
            // Restart the infer.NET random number generator
            Rand.Restart(12347);

            // Generate some data from the truth
            int numMix1 = 3;
            int numMix2 = 2;
            double[][] trueParams = new double[numMix1][];
            trueParams[0] = new double[] { 0.7, 0.3 };
            trueParams[1] = new double[] { 0.1, 0.9 };
            trueParams[2] = new double[] { 0.4, 0.6 };
            Discrete[] trueDisc = new Discrete[numMix1];
            for (int j = 0; j < numMix1; j++)
                trueDisc[j] = new Discrete(trueParams[j]);

            double[] truePi = new double[] { 0.1, 0.4, 0.5 };
            Discrete truePiDist = new Discrete(truePi);

            double[,] trueMeans = new double[,] { { 2.0, 3.0 }, { -1.0, 4.0 }, { 3.0, -1.0 } };
            double[,] truePrecs = new double[,] { { 7.0, 2.0 }, { 4.0, 6.0 }, { 3.0, 4.0 } };
            Gaussian[,] trueG = new Gaussian[numMix1, numMix2];
            for (int j = 0; j < numMix1; j++)
                for (int k = 0; k < numMix2; k++)
                    trueG[j, k] = Gaussian.FromMeanAndPrecision(trueMeans[j, k], truePrecs[j, k]);

            int totalSamples = 500;
            double[] data = new double[totalSamples];
            for (int n = 0; n < totalSamples; n++)
            {
                int m1x = truePiDist.Sample();
                int m2x = trueDisc[m1x].Sample();
                data[n] = trueG[m1x, m2x].Sample();
            }

            Variable<int> nM1 = Variable.Constant<int>(numMix1);
            Variable<int> nM2 = Variable.Constant<int>(numMix2);
            VariableArray<double> x = Variable.Constant(data).Named("data");
            //x.QuoteInMSL = false;
            Range m1 = new Range(nM1).Named("m1");
            Range m2 = new Range(nM2).Named("m2");
            Range i = x.Range.Named("i");

            double[] prior1 = new double[nM1.ObservedValue];
            for (int j = 0; j < nM1.ObservedValue; j++)
                prior1[j] = 1.0;
            double[] prior2 = new double[nM2.ObservedValue];
            for (int j = 0; j < nM2.ObservedValue; j++)
                prior2[j] = 1.0;
            // Break symmetry
            double[,] priorM = new double[,] { { 0.01, 0.02 }, { 0.03, 0.04 }, { 0.05, 0.06 } };

            VariableArray<Vector> probs = Variable.Array<Vector>(m1).Named("Probs");
            Variable<Vector> pi = Variable.Dirichlet(m1, prior1).Named("pi");
            probs[m1] = Variable.Dirichlet(m2, prior2).ForEach(m1);
            VariableArray2D<double> means = Variable.Array<double>(m1, m2).Named("means");
            VariableArray2D<double> precs = Variable.Array<double>(m1, m2).Named("precs");
            VariableArray2D<double> priorMeans = Variable.Constant<double>(priorM, m1, m2).Named("priorMeans");
            means[m1, m2] = Variable.GaussianFromMeanAndPrecision(priorMeans[m1, m2], 0.1);
            precs[m1, m2] = Variable.GammaFromMeanAndVariance(1, 1.0).ForEach(m1, m2);

            VariableArray<int> z = Variable.Array<int>(i).Named("z");
            VariableArray2D<int> zz = Variable.Array<int>(m1, i).Named("zz");
            using (Variable.ForEach(i))
            {
                z[i] = Variable.Discrete(pi);
                zz[m1, i] = Variable.Discrete(probs[m1]);
                using (Variable.Switch(z[i]))
                {
                    using (Variable.Switch(zz[z[i], i]))
                    {
                        x[i] = Variable.GaussianFromMeanAndPrecision(means[z[i], zz[z[i], i]], precs[z[i], zz[z[i], i]]);
                    }
                }
            }

            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());

            Dirichlet inferredPi = ie.Infer<Dirichlet>(pi);
            DistributionArray<Dirichlet> inferredProbs = ie.Infer<DistributionArray<Dirichlet>>(probs);
            DistributionArray2D<Gaussian> inferredMeans = ie.Infer<DistributionArray2D<Gaussian>>(means);
            DistributionArray2D<Gamma> inferredPrecs = ie.Infer<DistributionArray2D<Gamma>>(precs);

            Console.WriteLine("pi = " + inferredPi);
            Console.WriteLine("probs = " + inferredProbs);
            Console.WriteLine("means = " + inferredMeans);
            Console.WriteLine("precs = " + inferredPrecs);
        }

        internal void GateInTwoContexts()
        {
            Variable<bool> b = Variable.Bernoulli(0.5);
            Variable<bool> c = Variable.Bernoulli(0.5);
            using (Variable.If(b))
            {
                Variable<double> x = Variable.GaussianFromMeanAndPrecision(0, 1);
            }
            using (Variable.If(c))
            {
                using (Variable.If(b))
                {
                    Variable<double> y = Variable.GaussianFromMeanAndPrecision(0, 1);
                }
            }
            InferenceEngine ie = new InferenceEngine();
            Console.WriteLine(ie.Infer(b));
        }

        [Fact]
        public void ArrayOfGatesTest()
        {
            Range item = new Range(2).Named("item");
            VariableArray<double> p = Variable.Constant(new double[] { 0.2, 0.3 }, item);
            VariableArray<bool> b = Variable.Array<bool>(item).Named("b");
            b[item] = Variable.Bernoulli(0.1).ForEach(item);
            using (Variable.ForEach(item))
            {
                using (Variable.If(b[item]))
                {
                    Variable.ConstrainTrue(Variable.Bernoulli(p[item]));
                }
            }

            InferenceEngine engine = new InferenceEngine();
            DistributionArray<Bernoulli> bDist = engine.Infer<DistributionArray<Bernoulli>>(b);
            for (int i = 0; i < bDist.Count; i++)
            {
                double sumCondT = p.ObservedValue[i];
                double bPost = 0.1 * sumCondT / (0.1 * sumCondT + 0.9);
                Assert.True(MMath.AbsDiff(bDist[i].GetProbTrue(), bPost, 1e-6) < 1e-6);
            }
        }

        [Fact]
        public void EnterArrayElementsTest()
        {
            Range item = new Range(2).Named("item");
            Range xitem = new Range(2).Named("xitem");
            VariableArray<bool> x = Variable.Array<bool>(xitem).Named("x");
            double xPrior = 0.3;
            x[xitem] = Variable.Bernoulli(xPrior).ForEach(xitem);
            VariableArray<int> indices = Variable.Array<int>(item).Named("indices");
            indices.ObservedValue = new int[] { 0, 1 };
            VariableArray<int> indices2 = Variable.Array<int>(item).Named("indices2");
            indices2.ObservedValue = new int[] { 0, 1 };
            VariableArray<bool> b = Variable.Array<bool>(item).Named("b");
            double bPrior = 0.1;
            double xLike = 0.4;
            using (Variable.ForEach(item))
            {
                b[item] = Variable.Bernoulli(bPrior);
                using (Variable.If(b[item]))
                {
                    Variable.ConstrainEqualRandom(x[indices[item]], new Bernoulli(xLike));
                    Variable.ConstrainEqualRandom(x[indices2[item]], new Bernoulli(xLike));
                }
            }

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.Compiled += (sender, e) =>
            {
                // check for the inefficient replication warning
                Assert.Equal(1, e.Warnings.Count);
            };
            DistributionArray<Bernoulli> bDist = engine.Infer<DistributionArray<Bernoulli>>(b);
            for (int i = 0; i < bDist.Count; i++)
            {
                double sumCondT = xPrior * xLike * xLike + (1 - xPrior) * (1 - xLike) * (1 - xLike);
                double bPost = bPrior * sumCondT / (bPrior * sumCondT + (1 - bPrior));
                Assert.True(MMath.AbsDiff(bDist[i].GetProbTrue(), bPost, 1e-6) < 1e-6);
            }
        }

        [Fact]
        public void EnterArrayElementsTest2()
        {
            Range item = new Range(2).Named("item");
            Range xitem = new Range(2).Named("xitem");
            VariableArray<bool> x = Variable.Array<bool>(xitem).Named("x");
            double xPrior = 0.3;
            x[xitem] = Variable.Bernoulli(xPrior).ForEach(xitem);
            VariableArray<int> indices = Variable.Array<int>(item).Named("indices");
            indices.ObservedValue = new int[] { 0, 1 };
            VariableArray<int> indices2 = Variable.Array<int>(item).Named("indices2");
            indices2.ObservedValue = new int[] { 1, 0 };
            VariableArray<bool> b = Variable.Array<bool>(item).Named("b");
            double bPrior = 0.1;
            double xLike = 0.4;
            using (Variable.ForEach(item))
            {
                b[item] = Variable.Bernoulli(bPrior);
                //Variable<bool> xindex2 = Variable.Copy(x[indices2[item]]);
                using (Variable.If(b[item]))
                {
                    Variable.ConstrainEqualRandom(x[indices[item]], new Bernoulli(xLike));
                    Variable.ConstrainEqualRandom(x[indices2[item]], new Bernoulli(xLike));
                    //Variable.ConstrainEqualRandom(xindex2, new Bernoulli(xLike));
                }
            }

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.Compiled += (sender, e) =>
            {
                // check for the inefficient replication warning
                Assert.Equal(1, e.Warnings.Count);
            };
            DistributionArray<Bernoulli> bDist = engine.Infer<DistributionArray<Bernoulli>>(b);
            for (int i = 0; i < bDist.Count; i++)
            {
                double sumCondTF = System.Math.Pow(xPrior * xLike + (1 - xPrior) * (1 - xLike), 2);
                double sumCondTT = System.Math.Pow(xPrior * xLike * xLike + (1 - xPrior) * (1 - xLike) * (1 - xLike), 2);
                double sumCondT = bPrior * sumCondTT + (1 - bPrior) * sumCondTF;
                double sumCondF = bPrior * sumCondTF + (1 - bPrior);
                double bPost = bPrior * sumCondT / (bPrior * sumCondT + (1 - bPrior) * sumCondF);
                // error = 2.285e-05
                Console.WriteLine("error = {0}", MMath.AbsDiff(bDist[i].GetProbTrue(), bPost, 1e-10).ToString("g4"));
                Assert.True(MMath.AbsDiff(bDist[i].GetProbTrue(), bPost, 1e-10) < 1e-4);
            }
        }

        [Fact]
        public void EnterArrayElementsTest3()
        {
            Range item = new Range(2).Named("item");
            Range xitem = new Range(2).Named("xitem");
            VariableArray<bool> x = Variable.Array<bool>(xitem).Named("x");
            double xPrior = 0.3;
            x[xitem] = Variable.Bernoulli(xPrior).ForEach(xitem);
            VariableArray<int> indices = Variable.Array<int>(item).Named("indices");
            indices.ObservedValue = new int[] { 0, 1 };
            VariableArray<int> indices2 = Variable.Array<int>(item).Named("indices2");
            indices2.ObservedValue = new int[] { 1, 0 };
            VariableArray<bool> b = Variable.Array<bool>(item).Named("b");
            double bPrior = 0.1;
            double xLike = 0.4;
            using (Variable.ForEach(item))
            {
                b[item] = Variable.Bernoulli(bPrior);
                using (Variable.If(b[item]))
                {
                    var index = Variable.Max(0, indices[item]);
                    index.Name = nameof(index);
                    Variable.ConstrainEqualRandom(x[index], new Bernoulli(xLike));
                }
            }

            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;
            engine.Compiler.Compiled += (sender, e) =>
            {
                // check for the inefficient replication warning
                Assert.Equal(1, e.Warnings.Count);
            };
            DistributionArray<Bernoulli> bDist = engine.Infer<DistributionArray<Bernoulli>>(b);
            for (int i = 0; i < bDist.Count; i++)
            {
                double sumCondT = xPrior * xLike + (1 - xPrior) * (1 - xLike);
                double bPost = bPrior * sumCondT / (bPrior * sumCondT + (1 - bPrior));
                Assert.True(MMath.AbsDiff(bDist[i].GetProbTrue(), bPost, 1e-10) < 1e-10);
            }

            b.ObservedValue = new bool[] { true, false };
            InferenceEngine engine2 = new InferenceEngine();
            engine2.ShowProgress = false;
            engine2.Compiler.Compiled += (sender, e) =>
            {
                Assert.Equal(0, e.Warnings.Count);
            };
            DistributionArray<Bernoulli> bDist2 = engine2.Infer<DistributionArray<Bernoulli>>(b);
        }

        internal void FairCoinTest()
        {
            VariableArray<bool> tosses = Variable.Constant(new bool[] { true, true, true, true, true });
            //ConstantArray<bool> tosses = Variable.Constant(new bool[] { true, false, true, false, true });
            Range i = tosses.Range;
            // Prior on being fair
            Variable<bool> isFair = Variable.Bernoulli(0.5);
            using (Variable.If(isFair))
            {
                tosses[i] = Variable.Bernoulli(0.5).ForEach(i);
            }
            using (Variable.IfNot(isFair))
            {
                Variable<double> probHeads = Variable.Beta(1, 1);
                tosses[i] = Variable.Bernoulli(probHeads).ForEach(i);
            }
            InferenceEngine ie = new InferenceEngine();
            Console.WriteLine("Probability coin is fair is " + ie.Infer(isFair));
        }

        internal void BinaryCPTTest()
        {
            VariableArray<bool> a = Variable.Constant(new bool[] { false, true, false, true, true });
            Range i = a.Range;
            VariableArray<bool> feature = Variable.Constant(new bool[] { true, false, true, false, false }, i);
            Variable<double> probTrue1 = Variable.Beta(1, 1);
            Variable<double> probTrue2 = Variable.Beta(1, 1);
            using (Variable.ForEach(i))
            {
                using (Variable.If(feature[i]))
                {
                    a[i] = Variable.Bernoulli(probTrue1);
                }
                using (Variable.IfNot(feature[i]))
                {
                    a[i] = Variable.Bernoulli(probTrue2);
                }
            }
            InferenceEngine ie = new InferenceEngine();
            Console.WriteLine("Dist over b=" + ie.Infer(a));
        }

        [Fact]
        public void SimpleImputationTest()
        {
            var n = new Range(2).Named("n");
            var z = Variable.DiscreteUniform(n).Named("z");
            var means = Variable.Array<double>(n).Named("means");
            means[n] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(n);
            var pred = Variable.New<double>().Named("pred");
            var isMissing = Variable.Observed<bool>(false).Named("isMissing");
            using (Variable.Switch(z))
            {
                using (Variable.IfNot(isMissing))
                    pred.SetTo(Variable.GaussianFromMeanAndPrecision(means[z], 1));
                using (Variable.If(isMissing))
                    pred.SetTo(Variable.GaussianFromMeanAndPrecision(0, 1));
            }
            var ie = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(ie.Infer(pred)); // throws exception
        }

        [Fact]
        public void ImputationTest()
        {
            Variable<double> mean1 = Variable.GaussianFromMeanAndVariance(0, 100).Named("m1");
            Variable<double> prec1 = Variable.GammaFromShapeAndScale(1, 1).Named("p1");
            Variable<double> mean2 = Variable.GaussianFromMeanAndVariance(0, 100).Named("m2");
            Variable<double> prec2 = Variable.GammaFromShapeAndScale(1, 1).Named("p2");
            VariableArray<double> data = Variable.Observed(new double[] { 11, 5, 11, 9, 7, 7, 6 });
            Range i = data.Range;

            VariableArray<bool> removed = Variable.Observed<bool>(
                new bool[] { false, false, false, true, false, true, true }, i).Named("removed");
            VariableArray<double> impData = Variable.Array<double>(i).Named("impData");

            using (Variable.ForEach(i))
            {
                var b = Variable.Bernoulli(0.5);
                using (Variable.IfNot(b))
                {
                    using (Variable.IfNot(removed[i]))
                    {
                        data[i] = Variable.GaussianFromMeanAndPrecision(mean1, prec1);
                        impData[i] = Variable.GaussianFromMeanAndPrecision(0, 100);
                    }
                    using (Variable.If(removed[i]))
                        impData[i] = Variable.GaussianFromMeanAndPrecision(mean1, prec1);
                }
                using (Variable.If(b))
                {
                    using (Variable.IfNot(removed[i]))
                    {
                        data[i] = Variable.GaussianFromMeanAndPrecision(mean2, prec2);
                        impData[i] = Variable.GaussianFromMeanAndPrecision(0, 100);
                    }
                    using (Variable.If(removed[i]))
                        impData[i] = Variable.GaussianFromMeanAndPrecision(mean2, prec2);
                }
            }
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.NumberOfIterations = 1;
            var mean1Expected = engine.Infer(mean1);
            engine.NumberOfIterations = engine.Algorithm.DefaultNumberOfIterations;
            Gaussian mean1Actual = engine.Infer<Gaussian>(mean1);
            Console.WriteLine("mean1 = {0}", mean1Actual);
            Gaussian mean2Actual = engine.Infer<Gaussian>(mean2);
            Console.WriteLine("mean2 = {0}", mean2Actual);

            // test resetting inference
            engine.NumberOfIterations = 1;
            var mean12 = engine.Infer<Diffable>(mean1);
            Assert.True(mean12.MaxDiff(mean1Expected) < 1e-10);
        }

        // Fixed by changes to VMP triggers.
        // Can fail due to parallel updates of mean and precision, which cause oscillation in VMP.
        // Can be fixed by lowering the scheduling cost of missing requirements.
        // However, that causes other tests to fail (ImputationTest, ResetTest3, TwoLoopScheduleTest2)
        [Fact]
        public void GatedSubarrayObservedTest()
        {
            Rand.Restart(12347);
            Variable<double> mean1 = Variable.GaussianFromMeanAndVariance(0, 100).Named("Mean1");
            Variable<double> prec1 = Variable.GammaFromShapeAndScale(1, 1).Named("Prec1");
            Variable<double> mean2 = Variable.GaussianFromMeanAndVariance(10, 100).Named("Mean2");
            Variable<double> prec2 = Variable.GammaFromShapeAndScale(1, 1).Named("Prec2");
            Range i = new Range(6).Named("i");
            VariableArray<double> samp = Variable.Array<double>(i).Named("Samp");
            VariableArray<bool> b = Variable.Array<bool>(i).Named("b");
            using (Variable.ForEach(i))
            {
                b[i] = Variable.Bernoulli(0.5);
                using (Variable.If(b[i]))
                    samp[i] = Variable.GaussianFromMeanAndPrecision(mean1, prec1);
                using (Variable.IfNot(b[i]))
                    samp[i] = Variable.GaussianFromMeanAndPrecision(mean2, prec2);
            }

            double[] data = new double[] { 5, 11, 3, 6, 1, 7 };
            int[] identityMap = new int[data.Length];
            for (int j = 0; j < identityMap.Length; j++) identityMap[j] = j;

            if (true)
            {
                VariableArray<int> subIndices = Variable.Constant(identityMap).Named("id");
                VariableArray<double> subdata = Variable.Subarray(samp, subIndices).Named("subdata");
                subdata.ObservedValue = data;
            }
            else
            {
                samp.ObservedValue = data;
            }
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            if (false)
            {
                mean1.AddAttribute(new TraceMessages());
                prec1.AddAttribute(new TraceMessages());
                mean2.AddAttribute(new TraceMessages());
                prec2.AddAttribute(new TraceMessages());
                b.AddAttribute(new TraceMessages());
                engine.NumberOfIterations = 10;
            }
            b.InitialiseTo(Distribution<bool>.Array(Util.ArrayInit(i.SizeAsInt, j => new Bernoulli(Rand.Double()))));
            //mean2.InitialiseTo(new Gaussian(10, 100));
            //prec1.InitialiseTo(new Gamma(1, 1));
            //prec2.InitialiseTo(new Gamma(1, 1));

            Gaussian margMean1 = engine.Infer<Gaussian>(mean1);
            Gaussian margMean2 = engine.Infer<Gaussian>(mean2);
            double smallerMean = System.Math.Min(margMean1.GetMean(), margMean2.GetMean());
            double biggerMean = System.Math.Max(margMean1.GetMean(), margMean2.GetMean());
            // Some crude checks
            Assert.True(MMath.AbsDiff(smallerMean, 3.2) < 0.5);
            Assert.True(MMath.AbsDiff(biggerMean, 7.5) < 0.5);
        }


        private Discrete MontyHall(bool hostObserved, int pick, int hostpick, IAlgorithm algorithm)
        {
            //------------------------
            // The model
            //------------------------
            // c represents the position of the car
            var c = Variable.DiscreteUniform(3).Named("Car");

            // p represents the pick. This will be observed
            var p = Variable.DiscreteUniform(3).Named("Pick");
            p.ObservedValue = pick;

            // h represents the host pick.
            var h = Variable.New<int>().Named("Host");
            if (hostObserved)
            {
                h.ObservedValue = hostpick;
            }

            // Whether the host is observed
            var hostIsObserved = Variable.Observed<bool>(hostObserved);

            for (int a = 0; a < 3; a++)
            {
                for (int b = 0; b < 3; b++)
                {
                    double[] probs = { 1, 1, 1 };
                    for (int ps = 0; ps < 3; ps++)
                    {
                        if (ps == a || ps == b)
                            probs[ps] = 0;
                    }

                    using (Variable.Case(p, a))
                    {
                        using (Variable.Case(c, b))
                        {
                            h.SetTo(Variable.Discrete(probs));
                        }
                    }
                }
            }

            using (Variable.If(hostIsObserved))
            {
                Variable.ConstrainFalse(h == c);
            }

            var engine = new InferenceEngine(algorithm);
            return engine.Infer<Discrete>(c);
        }

        [Fact]
        public void MontyHallTest()
        {
            MontyHall(true, 0, 1, new ExpectationPropagation());
            MontyHall(true, 0, 1, new GibbsSampling());
            MontyHall(false, 0, 1, new ExpectationPropagation());
            MontyHall(false, 0, 1, new GibbsSampling());
        }

        [Fact]
        public void LDARepeatBlockTest()
        {
            int numDocuments = 1;
            int sizeVocab = 1;
            int numTopics = 1;
            Dirichlet thetaPrior = Dirichlet.Uniform(numTopics);
            Dirichlet phiPrior = Dirichlet.Uniform(sizeVocab);

            Range D = new Range(numDocuments).Named("D");
            Range W = new Range(sizeVocab).Named("W");
            Range T = new Range(numTopics).Named("T");
            var numWordsInDoc = Variable.Array<int>(D).Named("NumWordsInDoc");
            Range WInD = new Range(numWordsInDoc[D]).Named("WInD");

            var Theta = Variable.Array<Vector>(D).Named("Theta");
            Theta.SetValueRange(T);
            Theta[D] = Variable<Vector>.Random(thetaPrior).ForEach(D);
            var Phi = Variable.Array<Vector>(T).Named("Phi");
            Phi.SetValueRange(W);
            Phi[T] = Variable<Vector>.Random(phiPrior).ForEach(T);
            var Words = Variable.Array(Variable.Array<int>(WInD), D).Named("Words");
            var WordCounts = Variable.Array(Variable.Array<double>(WInD), D).Named("WordCounts");
            using (Variable.ForEach(D))
            {
                using (Variable.ForEach(WInD))
                {
                    using (Variable.Repeat(WordCounts[D][WInD]))
                    {
                        var topic = Variable.Discrete(Theta[D]).Named("topic");
                        using (Variable.Switch(topic))
                            Words[D][WInD] = Variable.Discrete(Phi[topic]);
                    }
                }
            }

            numWordsInDoc.ObservedValue = new int[] { 1 };
            WordCounts.ObservedValue = new double[][] { new double[] { 1 } };
            Words.ObservedValue = new int[][] { new int[] { 0 } };

            var engine = new InferenceEngine();
            var postTheta = engine.Infer(Theta);
        }

        [Fact]
        public void FactorAnalysisMixtureTest()
        {
            // Mixture of FAs, Y = W*X + I for each components

            int numD = 1;
            int numN = 1;
            int numC = 3; // # classes
            int numK = 2; // # factors

            Range C = new Range(numC).Named("C");
            Range K = new Range(numK).Named("K");
            Range D = new Range(numD).Named("D");
            Range N = new Range(numN).Named("N");

            // Class membership
            var Cn = Variable.Array<int>(N).Named("Cn");
            Cn[N] = Variable.DiscreteUniform(numC).ForEach(N);
            using (Variable.ForEach(N))
                Cn[N].SetValueRange(C);


            // Precision for the mixing matrices
            var A = Variable.Array<PositiveDefiniteMatrix>(C).Named("A");
            A[C] = Variable.WishartFromShapeAndScale(100, PositiveDefiniteMatrix.IdentityScaledBy(numK, 0.01)).ForEach(C);

            // Mixing matrices
            var W = Variable.Array(Variable.Array<Vector>(D), C).Named("W");
            using (Variable.ForEach(C))
            using (Variable.ForEach(D))
                W[C][D] = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(numK), A[C]);
            //W[C][D] = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(numK), PositiveDefiniteMatrix.IdentityScaledBy(numK, 0.01));


            // Factor activations
            var X = Variable.Array(Variable.Array<double>(K), N).Named("X");
            X[N][K] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(N, K);


            // Noise precision
            var tau = Variable.Array<double>(D).Named("tau");
            tau[D] = Variable.GammaFromShapeAndScale(100, 0.01).ForEach(D);


            // Missing?
            var isMissing = Variable.Array(Variable.Array<bool>(D), N).Named("isMissing");
            isMissing.ObservedValue = Util.ArrayInit(numN, i => Util.ArrayInit(numD, j => false));

            // Data
            var Y = Variable.Array(Variable.Array<double>(D), N).Named("Y");
            Y.ObservedValue = Util.ArrayInit(numN, i => Util.ArrayInit(numD, j => 1.0));

            using (Variable.ForEach(N))
            using (Variable.ForEach(D))
            {
                using (Variable.IfNot(isMissing[N][D]))
                using (Variable.Switch(Cn[N]))
                    Y[N][D] = Variable.GaussianFromMeanAndPrecision(Variable.InnerProduct(X[N], W[Cn[N]][D]), tau[D]);
                using (Variable.If(isMissing[N][D]))
                    Y[N][D] = Variable.GaussianFromMeanAndPrecision(0, 1);
            }

            // Break symmetry
            /* var initCn = new Discrete[numN];
       for (int n = 0; n < numN; n++)
           initCn[n] = Discrete.PointMass(Rand.Int(numC), numC);
       Cn.InitialiseTo(Distribution<int>.Array(initCn));
  
    Gaussian[][] initX = new Gaussian[numN][];
    for (int i = 0; i < numN; i++)
    {
        initX[i] = new Gaussian[numK];
        for (int j = 0; j < numK; j++)
            initX[i][j] = Gaussian.FromMeanAndVariance(Rand.Normal(), 1);
    }
    X.InitialiseTo(Distribution<double>.Array(initX));*/


            // Run inference
            var ie = new InferenceEngine(new VariationalMessagePassing());
            var postCn = ie.Infer<Discrete[]>(Cn);
        }
    }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif
}
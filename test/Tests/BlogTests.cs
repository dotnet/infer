// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Distributions;
using Xunit;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Factors;
using Assert = Xunit.Assert;
using Microsoft.ML.Probabilistic.Distributions.Kernels;
using System.IO;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Serialization;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

    public enum Attitude
    {
        Happy,
        Unhappy,
        Quiet
    };

    public enum BabyAction
    {
        Smile,
        Cry,
        LookSilly
    };

    /// <summary>
    /// Tests of code put in the blog
    /// </summary>
    public class BlogTests
    {
        /// <summary>
        /// This test should give a warning for excessive memory use.
        /// </summary>
        internal static void TrueSkill2Test()
        {
            TrueSkillV2Batch model = new TrueSkillV2Batch(1, 1, 1, 1);
            model.NGames.ObservedValue = 1;
            model.PlayersInGames.ObservedValue = new[] { 2 };
            model.PrevGameIndices.ObservedValue = new int[][] { new[] { -1, -1 } };
            model.PrevGamePlayerIndices.ObservedValue = new int[][] { new[] { 0, 0 } };
            model.Scores.ObservedValue = new double[][] { new double[] { 1, 0 } };
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(model.BiasCoefficient));
        }

        class TrueSkillV2Batch
        {
            public Variable<int> NGames { get; }
            public VariableArray<int> PlayersInGames { get; }
            public VariableArray<VariableArray<int>, int[][]> PrevGameIndices { get; }
            public VariableArray<VariableArray<int>, int[][]> PrevGamePlayerIndices { get; }
            public VariableArray<VariableArray<double>, double[][]> Skills { get; }
            public VariableArray<VariableArray<double>, double[][]> Performances { get; }
            public VariableArray<VariableArray<double>, double[][]> Scores { get; }

            public VariableArray<VariableArray<double>, double[][]> Biases { get; }
            public Variable<double> BiasCoefficient { get; }
            private readonly double InitialSigma;

            public TrueSkillV2Batch(double dynamicsVariance, double performanceVariance, double drawMargin, double initialSigma)
            {
                InitialSigma = initialSigma;
                NGames = Variable.New<int>().Named("NGames");
                Range game = new Range(NGames).Named("game");
                PlayersInGames = Variable.Array<int>(game).Named("PlayersInGames");

                Range gamePlayer = new Range(PlayersInGames[game]).Named("gamePlayer");

                PrevGameIndices = Variable.Array(Variable.Array<int>(gamePlayer), game).Named("Previous Game Indices").Attrib(new DoNotInfer());
                PrevGamePlayerIndices = Variable.Array(Variable.Array<int>(gamePlayer), game).Named("Previous Game Player Indices").Attrib(new DoNotInfer());
                Skills = Variable.Array(Variable.Array<double>(gamePlayer), game).Named("Skills").Attrib(new DoNotInfer());

                Performances = Variable.Array(Variable.Array<double>(gamePlayer), game).Named("Performances").Attrib(new DoNotInfer());
                Scores = Variable.Array(Variable.Array<double>(gamePlayer), game).Named("Scores").Attrib(new DoNotInfer());

                Biases = Variable.Array(Variable.Array<double>(gamePlayer), game).Named("Biases").Attrib(new DoNotInfer());
                Biases[game][gamePlayer] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(game, gamePlayer);

                BiasCoefficient = Variable.GaussianFromMeanAndVariance(0, 1).Named("Bias Coefficient").Attrib(new PointEstimate());

                using (Variable.ForEach(game))
                {
                    using (var p = Variable.ForEach(gamePlayer))
                    {
                        var playersFirstGame = PrevGameIndices[game][gamePlayer] < 0;
                        using (Variable.If(playersFirstGame))
                        {
                            Skills[game][gamePlayer] = Variable.GaussianFromMeanAndVariance(0, InitialSigma);
                        }

                        using (Variable.IfNot(playersFirstGame))
                        {
                            ////var prevGame = Variable.Min(PrevGameIndices[game][gamePlayer], 0).Named("prevGame");
                            ////var prevGamePlayerIndex = Variable.Min(PrevGamePlayerIndices[game][gamePlayer], 0).Named("prevGamePlayerIndex");
                            var prevGame = PrevGameIndices[game][gamePlayer];
                            var prevGamePlayerIndex = PrevGamePlayerIndices[game][gamePlayer];
                            Skills[game][gamePlayer] = Variable.GaussianFromMeanAndVariance(
                                Skills[prevGame][prevGamePlayerIndex],
                                dynamicsVariance
                            );
                        }

                        Performances[game][gamePlayer] = Variable.GaussianFromMeanAndVariance(Skills[game][gamePlayer] + BiasCoefficient * Biases[game][gamePlayer], performanceVariance);

                        using (Variable.If(p.Index > 0))
                        {
                            var diff = (Performances[game][p.Index - 1] - Performances[game][p.Index]).Named("diff");

                            using (Variable.If(Scores[game][p.Index - 1] == Scores[game][p.Index]))
                            {
                                Variable.ConstrainBetween(diff, -drawMargin, drawMargin);
                            }

                            using (Variable.IfNot(Scores[game][p.Index - 1] == Scores[game][p.Index]))
                            {
                                Variable.ConstrainTrue(diff > drawMargin);
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Test of GaussianFromMeanAndVariance.
        /// </summary>
        [Fact]
        [Trait("Category","OpenBug")]
        public void JamieMarTest()
        {
            // This is based on: https://github.com/usptact/Infer.NET-BayesianRegression
            // Create synthetic data Y=B+10X1+20X2+error (uniform(0,20)), X1 and X2 Uniform[1,10]
            Vector[] xdata;
            double[] distress;
            List<double> lstEcuacion = new List<double>();
            lstEcuacion.Add(10);
            lstEcuacion.Add(20);
            Int32 intErrorMax = 20;
            Int32 intNumObservaciones = 25;
            xdata = new Vector[intNumObservaciones];
            distress = new double[intNumObservaciones];
            Int32 intNumFactores = lstEcuacion.Count;
            Random rnd = new Random();
            for (Int32 intI = 0; intI < intNumObservaciones; intI++)
            {
                double dblYtmp = 0;
                double[] dblXTmp = new double[lstEcuacion.Count + 1];
                dblXTmp[0] = 1;
                for (Int32 intFactor = 0; intFactor < lstEcuacion.Count; intFactor++)
                {
                    Int32 intFlag = rnd.Next(1, 10);
                    dblYtmp = dblYtmp + lstEcuacion[intFactor] * intFlag;
                    dblXTmp[intFactor + 1] = intFlag;
                }
                dblYtmp = dblYtmp + rnd.Next(0, intErrorMax);
                distress[intI] = dblYtmp;
                xdata[intI] = Vector.FromArray(dblXTmp.ToArray());
            }

            // define a prior distribution and attach that to "w" random variable
            VectorGaussian wPrior = new VectorGaussian(Vector.Zero(intNumFactores + 1), PositiveDefiniteMatrix.Identity(intNumFactores + 1));
            Variable<Vector> w = Variable.Random(wPrior);

            //Noise distribution mean of a Gamma distribution is shape * scale, and its variance is shape * scale * scale
            double shape = 1000;
            double scale = 0.0001;
            Gamma noiseDist = new Gamma(shape, scale);
            Variable<double> noise = Variable.Random(noiseDist);

            // set features "x" and observations "y" as observed in the model
            VariableArray<double> y = Variable.Observed(distress);
            Range n = y.Range;
            VariableArray<Vector> x = Variable.Observed(xdata, n);

            // define "y" statistically: Gaussian RV array. Mean is defined by dot-product of param vector "w" and the feature vector x[n]
            y[n] = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(w, x[n]), noise);

            // Training
            InferenceEngine engine = new InferenceEngine();
            engine.ModelName = "Regression";
            engine.Algorithm = new ExpectationPropagation();
            // engine.Algorithm = new VariationalMessagePassing();
            //engine.Algorithm = new GibbsSampling();

            // infer "w" posterior as a distribution
            VectorGaussian wPosterior = engine.Infer<VectorGaussian>(w);
            Gamma noisePosterior = engine.Infer<Gamma>(noise);
            Console.WriteLine("Distribution over w = \n" + wPosterior);
            Console.WriteLine("Distribution over noise = \n" + noisePosterior);
        }

        /// <summary>
        /// This test requires GateEnterPartial.ValueAverageConditional to have a [Proper] annotation.
        /// </summary>
        [Fact]
        public void CoalMiningTest()
        {
            int[] disaster_data = new int[] { 4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                                             3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                                             2, 2, 3, 4, 2, 1, 3, 0, 2, 1, 1, 1, 1, 3, 0, 0,
                                             1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                                             0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                                             3, 3, 1, 0, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                                             0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1 };

            Range n = new Range(disaster_data.Length).Named("n");

            var switchpoint = Variable.DiscreteUniform(n);
            switchpoint.Name = nameof(switchpoint);

            var early_rate = Variable.GammaFromShapeAndRate(2.0, 2.0).Named("early_rate");
            var late_rate = Variable.GammaFromShapeAndRate(2.0, 2.0).Named("late_rate");

            var data = Variable.Array<int>(n);

            using (ForEachBlock block = Variable.ForEach(n))
            {
                using (Variable.If(switchpoint > block.Index))
                    data[n] = Variable.Poisson(early_rate);
                using (Variable.IfNot(switchpoint > block.Index))
                    data[n] = Variable.Poisson(late_rate);
            }

            data.ObservedValue = disaster_data;

            InferenceEngine engine = new InferenceEngine();

            var switchpointMarginal = engine.Infer<Discrete>(switchpoint);
            //Console.WriteLine(switchpointMarginal);
        }


        /// <summary>
        /// Test a model where hoisting must not happen.
        /// This model has 2 convergence loops.
        /// a_selector_cases_uses_B is declared but unused.
        /// See https://social.microsoft.com/Forums/en-US/a76a904d-ae2e-4118-bec0-c928772d7ff7/solving-the-nested-number-guessing-problem-from-anglican-web-site?forum=infer.net
        /// </summary>
        [Fact]
        public void NumberGuessingTest()
        {
            var n = new Range(10).Named("n");
            var a = Variable.DiscreteUniform(n).Named("a");
            var b = Variable.DiscreteUniform(n).Named("b");
            using (Variable.Switch(a)) {
                Variable.ConstrainTrue(a + b > 8);
            }
            (a + b).ObservedValue = 13;
            var ie = new InferenceEngine();
            Console.WriteLine($"a={ie.Infer(a)} b={ie.Infer(b)}");
        }

        /// <summary>
        /// See https://social.microsoft.com/Forums/en-US/147663c2-cccd-4cd7-9395-d56d700a0522/bayesian-ab-testing-compilation-error?forum=infer.net
        /// </summary>
        [Fact]
        public void BetaComparisonTest()
        {
            // latent model variables
            Variable<double> aRate = Variable.Beta(1, 10).Named("aRate");
            Variable<double> bRate = Variable.Beta(1, 10).Named("bRate");

            Variable<bool> aIsBetter = Variable<bool>.Factor(Factor.IsGreaterThan, aRate, bRate);
            aIsBetter.Name = nameof(aIsBetter);
            Variable<bool> aGreaterThanConstant = Variable<bool>.Factor(Factor.IsGreaterThan, aRate, 0.01);
            aGreaterThanConstant.Name = nameof(aGreaterThanConstant);

            // observed variables
            int aTrialCount = 500;
            int bTrialCount = 700;

            Variable<int> aSuccessCount = Variable.Binomial(aTrialCount, aRate);
            Variable<int> bSuccessCount = Variable.Binomial(bTrialCount, bRate);

            aSuccessCount.ObservedValue = 10;
            bSuccessCount.ObservedValue = 15;

            // inference
            InferenceEngine engine = new InferenceEngine();

            Beta aRatePosterior = engine.Infer<Beta>(aRate);
            Beta bRatePosterior = engine.Infer<Beta>(bRate);
            Console.WriteLine($"a = {aRatePosterior}, b = {bRatePosterior}");

            Bernoulli aGreaterThanConstantPost = engine.Infer<Bernoulli>(aGreaterThanConstant);
            Console.WriteLine($"{aGreaterThanConstant.Name} = {aGreaterThanConstantPost}");

            Bernoulli aIsBetterPosterior = engine.Infer<Bernoulli>(aIsBetter);
            Console.WriteLine("P(A > B) =  " + aIsBetterPosterior);
        }

        [Fact]
        public void GammaComparisonTest()
        {
            // latent model variables
            Variable<double> aRate = Variable.GammaFromShapeAndRate(4, 1).Named("aRate");
            Variable<double> bRate = Variable.GammaFromShapeAndRate(3, 1).Named("bRate");

            Variable<bool> aIsBetter = Variable<bool>.Factor(Factor.IsGreaterThan, aRate, bRate);
            aIsBetter.Name = nameof(aIsBetter);
            Variable<bool> aGreaterThanConstant = Variable<bool>.Factor(Factor.IsGreaterThan, aRate, 0.2);
            aGreaterThanConstant.Name = nameof(aGreaterThanConstant);

            // observed variables
            double aShape = 5;
            double bShape = 7;

            Variable<double> aObs = Variable.GammaFromShapeAndRate(aShape, aRate);
            Variable<double> bObs = Variable.GammaFromShapeAndRate(bShape, bRate);

            aObs.ObservedValue = 10;
            bObs.ObservedValue = 15;

            // inference
            InferenceEngine engine = new InferenceEngine();

            Gamma aRatePosterior = engine.Infer<Gamma>(aRate);
            Gamma bRatePosterior = engine.Infer<Gamma>(bRate);
            Console.WriteLine($"a = {aRatePosterior}, b = {bRatePosterior}");

            Bernoulli aGreaterThanConstantPost = engine.Infer<Bernoulli>(aGreaterThanConstant);
            Console.WriteLine($"{aGreaterThanConstant.Name} = {aGreaterThanConstantPost}");

            Bernoulli aIsBetterPosterior = engine.Infer<Bernoulli>(aIsBetter);
            Console.WriteLine("P(A > B) =  " + aIsBetterPosterior);
        }

        [Fact]
        public void BetaIsGreaterThan_PointMassTest()
        {
            Variable<double> x = Variable.Beta(1, 2).Named("x");
            x.AddAttribute(new PointEstimate());
            x.InitialiseTo(Beta.PointMass(0.33));
            var c = 0.5;
            Variable<bool> xGreaterThanConstant = Variable<bool>.Factor(Factor.IsGreaterThan, x, c);
            xGreaterThanConstant.Name = nameof(xGreaterThanConstant);
            Variable.ConstrainTrue(xGreaterThanConstant);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.InitialisationAffectsSchedule = true;
            var xActual = engine.Infer<Beta>(x);
            var xExpected = Beta.PointMass(c);
            Assert.Equal(xExpected, xActual);
        }

        [Fact]
        public void BetaIsGreaterThan_PointMassTest2()
        {
            Variable<double> x = Variable.Beta(2, 1).Named("x");
            x.AddAttribute(new PointEstimate());
            x.InitialiseTo(Beta.PointMass(0.33));
            var c = 0.5;
            Variable<bool> xGreaterThanConstant = Variable<bool>.Factor(Factor.IsGreaterThan, c, x);
            xGreaterThanConstant.Name = nameof(xGreaterThanConstant);
            Variable.ConstrainTrue(xGreaterThanConstant);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.InitialisationAffectsSchedule = true;
            var xActual = engine.Infer<Beta>(x);
            var xExpected = Beta.PointMass(c);
            Assert.Equal(xExpected, xActual);
        }

        [Fact]
        public void GammaIsGreaterThan_PointMassTest()
        {
            Variable<double> x = Variable.GammaFromShapeAndRate(4, 1).Named("x");
            x.AddAttribute(new PointEstimate());
            x.InitialiseTo(Gamma.PointMass(4));
            var c = 50.0;
            Variable<bool> xGreaterThanConstant = Variable<bool>.Factor(Factor.IsGreaterThan, x, c);
            xGreaterThanConstant.Name = nameof(xGreaterThanConstant);
            Variable.ConstrainTrue(xGreaterThanConstant);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.InitialisationAffectsSchedule = true;
            var xActual = engine.Infer<Gamma>(x);
            var xExpected = Gamma.PointMass(c);
            Assert.Equal(xExpected, xActual);
        }

        [Fact]
        public void GammaIsGreaterThan_PointMassTest2()
        {
            Variable<double> x = Variable.GammaFromShapeAndRate(4, 1).Named("x");
            x.AddAttribute(new PointEstimate());
            x.InitialiseTo(Gamma.PointMass(4));
            var c = 0.5;
            Variable<bool> xGreaterThanConstant = Variable<bool>.Factor(Factor.IsGreaterThan, c, x);
            xGreaterThanConstant.Name = nameof(xGreaterThanConstant);
            Variable.ConstrainTrue(xGreaterThanConstant);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.InitialisationAffectsSchedule = true;
            var xActual = engine.Infer<Gamma>(x);
            var xExpected = Gamma.PointMass(c);
            Assert.Equal(xExpected, xActual);
        }

        /// <summary>
        /// Fails with "not converging"
        /// </summary>
        [Trait("Category", "OpenBug")]
        [Fact]
        public void PairProductTest()
        {
            var count = 5;
            var trueValues = new double[] { 1, 2, 3, 4, 5 };
            var distributions = Enumerable.Range(0, count).Select(i => Variable.GaussianFromMeanAndPrecision(Variable.GaussianFromMeanAndVariance(0, 100), Variable.GammaFromShapeAndScale(1, 1)).Named("v_" + i)).ToArray();

            for (int i = 0; i < count; i++)
            {
                for (int j = i + 1; j < count; j++)
                {
                    (Variable.GaussianFromMeanAndVariance(distributions[i] * distributions[j], Variable.GammaFromShapeAndScale(1, 1))).ObservedValue = trueValues[i] * trueValues[j];
                }
            }

            var engine1 = new InferenceEngine();
            for (int i = 0; i < count; i++)
            {
                Console.WriteLine(engine1.Infer(distributions[i]));
            }
        }

        // Tests StringDistribution as a replacement for Discrete in GuejonPuzzle
        // Fails because the output StringDistribution is not normalized.
        [Trait("Category", "OpenBug")]
        [Fact]
        public void GuejonPuzzleStrings()
        {
            string[] names = {"Rick","Steve","Jo","Bob","Simon","Peter"};
            string[][] addressData = { new string[] { "12", "25", "30" }, new string[] { "7", "25", "12" }, new string[] { "25", "9", "40" } };
            string[][] personData = { new string[] { "Rick", "Steve", "Jo" }, new string[] { "Rick", "Bob", "Jo" }, new string[] { "Rick", "Simon", "Peter" } };

            int numPeople = names.Length;
            Range person = new Range(numPeople).Named("person");
            VariableArray<string> addressOfPerson = Variable.Array<string>(person).Named("addressOfPerson");
            addressOfPerson[person] = Variable.StringUniform().ForEach(person);
            Range day = new Range(personData.Length).Named("day");
            VariableArray<int> instanceCount = Variable.Array<int>(day).Named("instanceCount");
            instanceCount.ObservedValue = Util.ArrayInit(personData.Length, d => personData[d].Length);
            Range instance = new Range(instanceCount[day]).Named("instance");
            VariableArray<Vector> personDistribution = Variable.Array<Vector>(day).Named("personDistribution");
            personDistribution.SetValueRange(person);
            personDistribution.ObservedValue = Util.ArrayInit(personData.Length, d => 
                Vector.FromArray(Util.ArrayInit(numPeople, i => personData[d].Contains(names[i]) ? 1.0 : 0.0), Sparsity.Sparse));
            var observedAddress = Variable.Array(Variable.Array<string>(instance), day).Named("observedAddress");
            observedAddress.ObservedValue = addressData;
            var selectedPersonArray = Variable.Array(Variable.Array<int>(instance), day).Named("selectedPerson");
            using (Variable.ForEach(day))
            {
                using (Variable.ForEach(instance))
                {
                    selectedPersonArray[day][instance] = Variable.Discrete(personDistribution[day]);
                    //var selectedPerson = Variable.Discrete(personDistribution[day]);
                    var selectedPerson = selectedPersonArray[day][instance];
                    using(Variable.Switch(selectedPerson)) 
                        observedAddress[day][instance] = Variable.Copy(addressOfPerson[selectedPerson]);
                }
            }
            if (false)
            {
                // Random initialization
                var selectedPersonInit = Variable.Array(Variable.Array<Discrete>(instance), day).Named("selectedPersonInit");
                selectedPersonInit.ObservedValue = Util.ArrayInit(personData.Length, d =>
                    Util.ArrayInit(personData[d].Length, i => new Discrete(Util.ArrayInit(numPeople, p => Rand.Double()))));
                selectedPersonArray[day][instance].InitialiseTo(selectedPersonInit[day][instance]);
            }

            InferenceEngine engine = new InferenceEngine();
            for (int iter = 1; iter < 10; iter++)
            {
                engine.NumberOfIterations = iter;
                Console.WriteLine("iter {0}", iter);
                Console.WriteLine(engine.Infer(selectedPersonArray));
                var inferredAddressOfPerson2 = engine.Infer<IList<StringDistribution>>(addressOfPerson);
                for (int i = 0; i < inferredAddressOfPerson2.Count; i++)
                {
                    Console.WriteLine("{0}: {1}", names[i], inferredAddressOfPerson2[i]);
                }
            }
            var inferredAddressOfPerson = engine.Infer<IList<StringDistribution>>(addressOfPerson);
            Assert.True(inferredAddressOfPerson[0].GetLogProb("25") > -0.5);
            Assert.True(inferredAddressOfPerson[1].GetLogProb("30") > -0.5);
            Assert.True(inferredAddressOfPerson[2].GetLogProb("12") > -0.5);
            Assert.True(inferredAddressOfPerson[3].GetLogProb("7") > -0.5);
            Assert.True(inferredAddressOfPerson[4].GetLogProb("9") > -1);
            Assert.True(inferredAddressOfPerson[4].GetLogProb("40") > -1);
            Assert.True(inferredAddressOfPerson[5].GetLogProb("9") > -1);
            Assert.True(inferredAddressOfPerson[5].GetLogProb("40") > -1);
        }

        // Loss of sparsity
        [Fact]
        public void GuejonPuzzle()
        {
            string[] names = { "Rick", "Steve", "Jo", "Bob", "Simon", "Peter" };
            int[][] addressData = { new int[] { 12, 25, 30 }, new int[] { 7, 25, 12 }, new int[] { 25, 9, 40 } };
            string[][] personData = { new string[] { "Rick", "Steve", "Jo" }, new string[] { "Rick", "Bob", "Jo" }, new string[] { "Rick", "Simon", "Peter" } };

            int numPeople = names.Length;
            int maxAddress = 100;
            Range person = new Range(numPeople).Named("person");
            VariableArray<int> addressOfPerson = Variable.Array<int>(person).Named("addressOfPerson");
            addressOfPerson[person] = Variable.DiscreteUniform(maxAddress).ForEach(person);
            Range day = new Range(personData.Length).Named("day");
            VariableArray<int> instanceCount = Variable.Array<int>(day).Named("instanceCount");
            instanceCount.ObservedValue = Util.ArrayInit(personData.Length, d => personData[d].Length);
            Range instance = new Range(instanceCount[day]).Named("instance");
            VariableArray<Vector> personDistribution = Variable.Array<Vector>(day).Named("personDistribution");
            personDistribution.SetValueRange(person);
            personDistribution.ObservedValue = Util.ArrayInit(personData.Length, d =>
                Vector.FromArray(Util.ArrayInit(numPeople, i => personData[d].Contains(names[i]) ? 1.0 : 0.0), Sparsity.Sparse));
            var observedAddress = Variable.Array(Variable.Array<int>(instance), day).Named("observedAddress");
            observedAddress.ObservedValue = addressData;
            using (Variable.ForEach(day))
            {
                using (Variable.ForEach(instance))
                {
                    var selectedPerson = Variable.Discrete(personDistribution[day]).Named("selectedPerson");
                    using (Variable.Switch(selectedPerson))
                        observedAddress[day][instance] = Variable.Copy(addressOfPerson[selectedPerson]);
                }
            }

            InferenceEngine engine = new InferenceEngine();
            var inferredAddressOfPerson = engine.Infer<IList<Discrete>>(addressOfPerson);
            for (int i = 0; i < inferredAddressOfPerson.Count; i++)
            {
                Console.Write("{0}: ", names[i]);
                var inferredAddress = inferredAddressOfPerson[i];
                for (int j = 0; j < inferredAddress.Dimension; j++)
                {
                    if (inferredAddress[j] > 1e-2)
                        Console.Write("{0} ", j);
                }
                Console.WriteLine();
            }
            Assert.True(inferredAddressOfPerson[0].GetLogProb(25) > -0.5);
            Assert.True(inferredAddressOfPerson[1].GetLogProb(30) > -0.5);
            Assert.True(inferredAddressOfPerson[2].GetLogProb(12) > -0.5);
            Assert.True(inferredAddressOfPerson[3].GetLogProb(7) > -0.5);
            Assert.True(inferredAddressOfPerson[4].GetLogProb(9) > -1);
            Assert.True(inferredAddressOfPerson[4].GetLogProb(40) > -1);
            Assert.True(inferredAddressOfPerson[5].GetLogProb(9) > -1);
            Assert.True(inferredAddressOfPerson[5].GetLogProb(40) > -1);
        }

        [Fact]
        public void BinomialGreaterThanTest()
        {
            int trialCount = 5;
            double p = 0.4;
            var xPrior = new Binomial(trialCount, p);
            int threshold = 3;
            Variable<int> x = Variable.Binomial(trialCount, p);
            Variable<bool> y = (x > threshold);
            InferenceEngine engine = new InferenceEngine();
            var yExpected = new Bernoulli(0.08704);
            var yActual = engine.Infer<Bernoulli>(y);
            Assert.True(yExpected.MaxDiff(yActual) < 1e-4);
            var yActual2 = IsGreaterThanOp.IsGreaterThanAverageConditional(xPrior, threshold);
            Assert.True(yExpected.MaxDiff(yActual2) < 1e-4);

            y.ObservedValue = true;
            var xExpected = Discrete.Uniform(trialCount+1);
            Vector probs = xExpected.GetWorkspace();
            for (int i = 0; i <= trialCount; i++)
            {
                if (i > threshold)
                    probs[i] = System.Math.Exp(xPrior.GetLogProb(i));
                else
                    probs[i] = 0;
            }
            xExpected.SetProbs(probs);
            var xActual = engine.Infer<Discrete>(x);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-4);
        }

        [Fact]
        public void BinomialLessThanTest()
        {
            int trialCount = 5;
            double p = 0.4;
            var xPrior = new Binomial(trialCount, p);
            int threshold = 3;
            Variable<int> x = Variable.Binomial(trialCount, p);
            Variable<bool> y = (x < threshold);
            InferenceEngine engine = new InferenceEngine();
            var yExpected = new Bernoulli(0.68256);
            var yActual = engine.Infer<Bernoulli>(y);
            Assert.True(yExpected.MaxDiff(yActual) < 1e-4);
            var yActual2 = IsGreaterThanOp.IsGreaterThanAverageConditional(threshold, xPrior);
            Assert.True(yExpected.MaxDiff(yActual2) < 1e-4);

            y.ObservedValue = true;
            var xExpected = Discrete.Uniform(trialCount + 1);
            Vector probs = xExpected.GetWorkspace();
            for (int i = 0; i <= trialCount; i++)
            {
                if (i < threshold)
                    probs[i] = System.Math.Exp(xPrior.GetLogProb(i));
                else
                    probs[i] = 0;
            }
            xExpected.SetProbs(probs);
            var xActual = engine.Infer<Discrete>(x);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-4);
        }

        [Fact]
        public void PoissonGreaterThanTest()
        {
            double mean = 3;
            Variable<int> x = Variable.Poisson(mean);
            Variable<bool> y = (x > 3);
            InferenceEngine engine = new InferenceEngine();
            var yExpected = new Bernoulli(0.352768111217769);
            var yActual = engine.Infer<Bernoulli>(y);
            Assert.True(yExpected.MaxDiff(yActual) < 1e-4);

            y.ObservedValue = true;
            var xExpected = new Poisson(4.905289626791834);
            var xActual = engine.Infer<Poisson>(x);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-4);
        }

        [Fact]
        public void PoissonLessThanTest()
        {
            double mean = 3;
            Variable<int> x = Variable.Poisson(mean);
            Variable<bool> y = (x < 3);
            InferenceEngine engine = new InferenceEngine();
            var yExpected = new Bernoulli(0.423190081126844);
            var yActual = engine.Infer<Bernoulli>(y);
            Assert.True(yExpected.MaxDiff(yActual) < 1e-4);

            y.ObservedValue = true;
            var xExpected = new Poisson(1.411764705882353);
            var xActual = engine.Infer<Poisson>(x);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-4);
        }

        // Example of inferring the size of a population.
        // Reference: A. Raftery, "Inference for the binomial N parameter: A hierarchical Bayes approach", Biometrika 1988
        // http://pluto.huji.ac.il/~galelidan/52558/Material/Raftery.pdf
        [Fact]
        public void BinomialTrialCountTest()
        {
            int[] impala = { 15, 20, 21, 23, 26 };
            int[] waterbuck = { 53, 57, 66, 67, 72 };
            var evidence = Variable.Bernoulli(0.5).Named("evidence");
            var block = Variable.If(evidence);
            // prior for theta and N is 1/N
            Variable<double> theta = Variable.Beta(1, 1).Named("theta");
            int maxN = 1000;
            Variable<int> N = Variable.Discrete(Util.ArrayInit(maxN, i => (i == 0) ? 0 : 1.0 / i));
            N.Name = "N";
            Variable<int> count = Variable.New<int>().Named("count");
            Range item = new Range(count).Named("item");
            VariableArray<int> x = Variable.Array<int>(item).Named("x");
            x[item] = Variable.Binomial(N, theta).ForEach(item);
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            for (int alg = 0; alg < 2; alg++)
            {
                if (alg == 1)
                    engine.Algorithm = new VariationalMessagePassing();
                Console.WriteLine();
                Console.WriteLine(engine.Algorithm.Name);
                engine.ShowProgress = false;
                for (int method = 0; method < 2; method++)
                {
                    for (int dataset = 0; dataset < 2; dataset++)
                    {
                        if (dataset == 0)
                        {
                            Console.WriteLine("impala");
                            x.ObservedValue = impala;
                        }
                        else
                        {
                            Console.WriteLine("waterbuck");
                            x.ObservedValue = waterbuck;
                        }
                        count.ObservedValue = x.ObservedValue.Length;
                        Discrete Npost;
                        if (method == 0)
                        {
                            Npost = engine.Infer<Discrete>(N);
                            Console.WriteLine("  theta = {0}", engine.Infer(theta));
                        }
                        else
                        {
                            Npost = Discrete.Uniform(maxN);
                            Vector probs = Npost.GetWorkspace();
                            for (int i = 0; i < maxN; i++)
                            {
                                N.ObservedValue = i;
                                double logprob = engine.Infer<Bernoulli>(evidence).LogOdds;
                                probs[i] = logprob;
                            }
                            double max = probs.Max();
                            probs.SetToFunction(probs, logprob => System.Math.Exp(logprob - max));
                            Npost.SetProbs(probs);
                        }
                        double Nmed = Npost.GetMedian();
                        double Nmode = Npost.GetMode();
                        Console.WriteLine("  N mode = {0}, median = {1}", Nmode, Nmed);
                        if (method == 1)
                        {
                            if (dataset == 0)
                                Assert.Equal(37, Nmode);
                            else
                                Assert.Equal(122, Nmode);
                        }
                    }
                    N.ClearObservedValue();
                }
            }
        }

        /// <summary>
        /// Fails because there is no Multinomial distribution implemented.
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")]
        public void MultinomialGreaterThanTest()
        {
            var trialOutcomes = new Range(2);
            var trialRange = new Range(5);
            var trial = Variable.New<int>();
            trial.Name = "trial";
            trial.ObservedValue = 5;
            trial.SetValueRange(trialRange);
            var pi = Variable.Dirichlet(trialOutcomes, new double[] { 1.0, 1.0 }).Named("pi");
            var md = Variable.Multinomial(trial, pi);
            md.SetValueRange(trialOutcomes);
            //md.ObservedValue = new int[] { 1, 2 };
            md.Name = "md";
            var ie = new InferenceEngine();
            var res = ie.Infer<Bernoulli>(md[0] > md[1]);
            Console.WriteLine(res);
        }

        // Learns a TrueSkill-type model where players are described by real vectors, and skill is a nonlinear function of the vector
        // Reference: E. Bonilla, S. Guo, and S. Sanner, "Gaussian Process Preference Elicitation", NIPS 2010
        // http://users.rsise.anu.edu.au/~ssanner/Papers/gppe_final.pdf
        // Reference: Wei Chu and Zoubin Ghahramani. "Preference learning with Gaussian processes", ICML 2005
        // http://www.gatsby.ucl.ac.uk/~chuwei/paper/gppl.pdf
        internal void GPTrueSkillTest()
        {
            Rand.Restart(0);
            int nGames = 100;
            int dims = 2;
            Vector[] data1 = Util.ArrayInit(nGames, i => Vector.FromArray(Util.ArrayInit(dims, j => Rand.Double())));
            Vector[] data2 = Util.ArrayInit(nGames, i => Vector.FromArray(Util.ArrayInit(dims, j => Rand.Double())));
            bool[] dataOutcome = new bool[nGames];
            for (int i = 0; i < nGames; i++)
            {
                double perf1 = data1[i][0]*data1[i][0] + data1[i][1]*data1[i][1] + Rand.Normal()/100;
                double perf2 = data2[i][0]*data2[i][0] + data2[i][1]*data2[i][1] + Rand.Normal()/100;
                dataOutcome[i] = (perf1 > perf2);
            }

            GaussianProcess gp = new GaussianProcess(new ConstantFunction(0), new SquaredExponential(-1));
            Variable<SparseGP> fprior = Variable.New<SparseGP>().Named("fprior");
            Vector[] basis = Util.ArrayInit(10, i => (i < nGames) ? data1[i] : data2[i - nGames]);
            fprior.ObservedValue = new SparseGP(new SparseGPFixed(gp, basis));
            Variable<IFunction> f = Variable<IFunction>.Random(fprior);
            Range game = new Range(nGames);
            VariableArray<Vector> vector1 = Variable.Array<Vector>(game);
            vector1.ObservedValue = data1;
            VariableArray<Vector> vector2 = Variable.Array<Vector>(game);
            vector2.ObservedValue = data2;
            VariableArray<bool> outcome = Variable.Array<bool>(game);
            outcome.ObservedValue = dataOutcome;
            using (Variable.ForEach(game))
            {
                var skill1 = Variable.FunctionEvaluate(f, vector1[game]);
                var perf1 = Variable.GaussianFromMeanAndPrecision(skill1, 1.0);
                var skill2 = Variable.FunctionEvaluate(f, vector2[game]);
                var perf2 = Variable.GaussianFromMeanAndPrecision(skill2, 1.0);
                outcome[game] = (perf1 > perf2);
            }
            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = true;
            var fPost = engine.Infer<SparseGP>(f);
            Console.WriteLine(fPost);
            int res = 100;
            double inc = 1.0/res;
            Vector v = Vector.Zero(dims);
            Matrix m = new Matrix(res, res);
            for (int i = 0; i < res; i++)
            {
                v[0] = i*inc;
                for (int j = 0; j < res; j++)
                {
                    v[1] = j*inc;
                    m[i, j] = fPost.EvaluateMean(v);
                }
            }
            //TODO: change path for cross platform using
            using (MatlabWriter writer = new MatlabWriter(@"..\..\..\Tests\gpTrueSkill.mat"))
            {
                writer.Write("m", m);
            }
            // in Matlab:
            // load gpTrueSkill.mat
            // imagesc(m)
        }



        public class GradingData
        {
            public int nWorkerN;
            public int nQuestionN;
            public int nResponseN;
            public int nChoiceN;
            public int[] worker;
            public int[] workerGradesN;
            public int[] question;
            public int[] responseV;
            public double[] unProbs;

            public GradingData(string fileNamePath)
            {
                Read(fileNamePath);
            }

            //Taken from Infer.NET site post
            public void Read(string path)
            {
                string fileName = path;
                StreamReader objReader = new StreamReader(fileName);

                Console.Out.WriteLine("Starting reading {0} file ...", fileName);
                string sLine = "";
                string[] ele = null;
                char[] SplitChar = new char[] {'\t'};
                List<string> LineList = new List<string>();
                while (sLine != null)
                {
                    sLine = objReader.ReadLine();

                    if (sLine != null && !sLine.Equals(""))
                        LineList.Add(sLine);
                }
                objReader.Close();

                nResponseN = LineList.Count;
                nWorkerN = 0;
                nQuestionN = 0;
                nChoiceN = 0;

                // build int[]
                worker = new int[nResponseN];
                question = new int[nResponseN];
                responseV = new int[nResponseN];
                for (int i = 0; i < nResponseN; i++)
                {
                    sLine = (string) (LineList[i]);
                    ele = sLine.Split(SplitChar, StringSplitOptions.RemoveEmptyEntries);

                    //This is my cosmos format (account puid, grader, response value)
                    question[i] = int.Parse(ele[0]);
                    worker[i] = int.Parse(ele[1]);
                    responseV[i] = int.Parse(ele[2]);

                    nWorkerN = System.Math.Max(nWorkerN, worker[i]);
                    nQuestionN = System.Math.Max(nQuestionN, question[i]);
                    nChoiceN = System.Math.Max(nChoiceN, responseV[i]);

                    // show the first 3 lines

                    if (i > 3)
                        continue;
                    Console.Out.WriteLine(string.Format("Account:{0} Grader:{1} Grade{2}", question[i], worker[i], responseV[i]));
                }
                // because the number is starting from 0
                nWorkerN++;
                nQuestionN++;
                nChoiceN++;
                Console.Out.WriteLine("...");
                Console.Out.WriteLine("Data loading Completed! Total:" + nResponseN + " grading events");

                //Calculating the value priors
                unProbs = new double[nChoiceN];
                for (int i = 0; i < nChoiceN; i++)
                    unProbs[i] = 0;

                for (int j = 0; j < nResponseN; j++)
                    unProbs[responseV[j]]++;


                //Grader review count
                workerGradesN = new int[nWorkerN];
                for (int i = 0; i < nResponseN; i++)
                    workerGradesN[worker[i]]++;
            }
        }

        //End GradingData class

        // Requires Scheduling2Transform.IgnoreOffsetRequirements = true.
        // Has similar problem as SerialTests.SumForwardBackwardTest
        [Fact]
        [Trait("Category", "OpenBug")]
        public void CoupledHmm4Test()
        {
            CoupledHmm4(new ExpectationPropagation());
            CoupledHmm4(new VariationalMessagePassing());
        }
        private void CoupledHmm4(IAlgorithm algorithm)
        {
            int numTimeSteps = 4;
            int numDimensions = 4;
            int numStates = 4;
            Range time = new Range(numTimeSteps).Named("time");
            Range dim = new Range(numDimensions).Named("dim");
            Range state = new Range(numStates).Named("state");
            Range prevState0 = state.Clone().Named("prevState0");
            Range prevState1 = state.Clone().Named("prevState1");
            Range prevState2 = state.Clone().Named("prevState2");
            Range prevState3 = state.Clone().Named("prevState3");
            // parameters
            Variable<Vector> Xprior = Variable.DirichletSymmetric(state, 1.0).Named("Xprior");
            var transitionProbs =
                Variable.Array(Variable.Array(Variable.Array(Variable.Array<Vector>(prevState3), prevState2), prevState1), prevState0).Named("transitionProbs");
            transitionProbs[prevState0][prevState1][prevState2][prevState3] = Variable.DirichletSymmetric(state, 1.0).ForEach(prevState0, prevState1, prevState2, prevState3);
            VariableArray<double> emissionMean = Variable.Array<double>(state).Named("emissionMean");
            emissionMean[state] = Variable.GaussianFromMeanAndPrecision(0, 0.01).ForEach(state);
            // hidden states and observations
            var X = Variable.Array(Variable.Array<int>(dim), time).Named("X");
            var S = Variable.Array(Variable.Array<double>(dim), time).Named("S");
            VariableArray<int> prevTime = Variable.Array<int>(time).Named("prevTime");
            prevTime.ObservedValue = System.Linq.Enumerable.Range(-1, numTimeSteps).ToArray();
            using (ForEachBlock fb = Variable.ForEach(time))
            {
                using (Variable.If(fb.Index == 0))
                {
                    X[time][dim] = Variable.Discrete(Xprior).ForEach(dim);
                }
                using (Variable.If(fb.Index > 0))
                {
                    //var prevX = X[prevTime[time]];
                    var prevX = X[fb.Index - 1];
                    var prevX0 = Variable.Copy(prevX[0]).Named("prevX0");
                    prevX0.SetValueRange(prevState0);
                    // this gives a nicer serial schedule
                    //prevX0.AddAttribute(new DivideMessages(false));
                    var prevX1 = Variable.Copy(prevX[1]).Named("prevX1");
                    prevX1.SetValueRange(prevState1);
                    var prevX2 = Variable.Copy(prevX[2]).Named("prevX2");
                    prevX2.SetValueRange(prevState2);
                    var prevX3 = Variable.Copy(prevX[3]).Named("prevX3");
                    prevX3.SetValueRange(prevState3);
                    using (Variable.Switch(prevX0))
                    using (Variable.Switch(prevX1))
                    using (Variable.Switch(prevX2))
                    using (Variable.Switch(prevX3))
                    using (Variable.ForEach(dim))
                    {
                        var prediction = Variable.Discrete(transitionProbs[prevX0][prevX1][prevX2][prevX3]);
                        X[time][dim] = prediction;
                    }
                }
                using (Variable.ForEach(dim))
                {
                    using (Variable.Switch(X[time][dim]))
                    {
                        S[time][dim] = Variable.GaussianFromMeanAndPrecision(emissionMean[X[time][dim]], 1.0);
                    }
                }
            }
            S.ObservedValue = Util.ArrayInit(numTimeSteps, t => Util.ArrayInit(numDimensions, d => t + d + 0.0));

            InferenceEngine engine = new InferenceEngine(algorithm);
            var xActual = engine.Infer<IReadOnlyList<IReadOnlyList<Discrete>>>(X);
            Console.WriteLine("X[0][0] = {0}", xActual[0][0]);
            Console.WriteLine("emissionMean:");
            Console.WriteLine(engine.Infer(emissionMean));
            var transitionProbsActual = engine.Infer<IReadOnlyList<IReadOnlyList<IReadOnlyList<IReadOnlyList<Dirichlet>>>>>(transitionProbs);
            Console.WriteLine("transitionProbs[0][0][0][0] = {0}", transitionProbsActual[0][0][0][0]);
        }

        // Fails similarly to CoupledHmm4Test
        [Fact]
        [Trait("Category", "OpenBug")]
        public void CoupledHmm3ExampleTest()
        {
            CoupledHmm3Example(new ExpectationPropagation());
            CoupledHmm3Example(new VariationalMessagePassing());
        }

        private void CoupledHmm3Example(IAlgorithm algorithm)
        {
            int numTimeSteps = 3;
            int numDimensions = 3;
            int numStates = 3;
            Range time = new Range(numTimeSteps).Named("time");
            Range dim = new Range(numDimensions).Named("dim");
            Range state = new Range(numStates).Named("state");
            Range prevState0 = state.Clone().Named("prevState0");
            Range prevState1 = state.Clone().Named("prevState1");
            Range prevState2 = state.Clone().Named("prevState2");
            // parameters
            Variable<Vector> Xprior = Variable.DirichletSymmetric(state, 1.0).Named("Xprior");
            var transitionProbs = Variable.Array(Variable.Array(Variable.Array<Vector>(prevState2), prevState1), prevState0).Named("transitionProbs");
            transitionProbs[prevState0][prevState1][prevState2] = Variable.DirichletSymmetric(state, 1.0).ForEach(prevState0, prevState1, prevState2);
            VariableArray<double> emissionMean = Variable.Array<double>(state).Named("emissionMean");
            emissionMean[state] = Variable.GaussianFromMeanAndPrecision(0, 0.01).ForEach(state);
            // hidden states and observations
            var X = Variable.Array(Variable.Array<int>(dim), time).Named("X");
            var S = Variable.Array(Variable.Array<double>(dim), time).Named("S");
            VariableArray<int> prevTime = Variable.Array<int>(time).Named("prevTime");
            prevTime.ObservedValue = System.Linq.Enumerable.Range(-1, numTimeSteps).ToArray();
            using (ForEachBlock fb = Variable.ForEach(time))
            {
                using (Variable.If(fb.Index == 0))
                {
                    X[time][dim] = Variable.Discrete(Xprior).ForEach(dim);
                }
                using (Variable.If(fb.Index > 0))
                {
                    //var prevX = X[prevTime[time]];
                    var prevX = X[fb.Index - 1];
                    var prevX0 = Variable.Copy(prevX[0]).Named("prevX0");
                    prevX0.SetValueRange(prevState0);
                    // this gives a nicer serial schedule
                    prevX0.AddAttribute(new DivideMessages(false));
                    var prevX1 = Variable.Copy(prevX[1]).Named("prevX1");
                    prevX1.SetValueRange(prevState1);
                    var prevX2 = Variable.Copy(prevX[2]).Named("prevX2");
                    prevX2.SetValueRange(prevState2);
                    using (Variable.Switch(prevX0))
                    using (Variable.Switch(prevX1))
                    using (Variable.Switch(prevX2))
                    using (Variable.ForEach(dim))
                    {
                        var prediction = Variable.Discrete(transitionProbs[prevX0][prevX1][prevX2]);
                        X[time][dim] = prediction;
                    }
                }
                using (Variable.ForEach(dim))
                {
                    using (Variable.Switch(X[time][dim]))
                    {
                        S[time][dim] = Variable.GaussianFromMeanAndPrecision(emissionMean[X[time][dim]], 1.0);
                    }
                }
            }
            S.ObservedValue = Util.ArrayInit(numTimeSteps, t => Util.ArrayInit(numDimensions, d => t + d + 0.0));

            InferenceEngine engine = new InferenceEngine(algorithm);
            // must build X first to avoid use-before-def problem
            var xActual = engine.Infer<IReadOnlyList<IReadOnlyList<Discrete>>>(X);
            Console.WriteLine("X[0][0] = {0}", xActual[0][0]);
            Console.WriteLine("emissionMean:");
            Console.WriteLine(engine.Infer(emissionMean));
            var transitionProbsActual = engine.Infer<IReadOnlyList<IReadOnlyList<IReadOnlyList<Dirichlet>>>>(transitionProbs);
            Console.WriteLine("transitionProbs[0][0][0] = {0}", transitionProbsActual[0][0][0]);
        }

        [Fact]
        public void CoupledHmm2Test()
        {
            CoupledHmm2(new ExpectationPropagation());
            CoupledHmm2(new VariationalMessagePassing());
        }

        private void CoupledHmm2(IAlgorithm algorithm)
        {
            int numTimeSteps = 100;
            int numDimensions = 3;
            int numStates = 3;
            Range time = new Range(numTimeSteps).Named("time");
            Range dim = new Range(numDimensions).Named("dim");
            Range state = new Range(numStates).Named("state");
            Range prevState0 = state.Clone().Named("prevState0");
            Range prevState1 = state.Clone().Named("prevState1");
            // parameters
            Variable<Vector> Xprior = Variable.DirichletSymmetric(state, 1.0).Named("Xprior");
            var transitionProbs = Variable.Array(Variable.Array<Vector>(prevState1), prevState0).Named("transitionProbs");
            transitionProbs[prevState0][prevState1] = Variable.DirichletSymmetric(state, 1.0).ForEach(prevState0, prevState1);
            VariableArray<double> emissionMean = Variable.Array<double>(state).Named("emissionMean");
            emissionMean[state] = Variable.GaussianFromMeanAndPrecision(0, 0.01).ForEach(state);
            // hidden states and observations
            var X = Variable.Array(Variable.Array<int>(dim), time).Named("X");
            var S = Variable.Array(Variable.Array<double>(dim), time).Named("S");
            VariableArray<int> prevTime = Variable.Array<int>(time).Named("prevTime");
            prevTime.ObservedValue = Enumerable.Range(-1, numTimeSteps).ToArray();
            using (ForEachBlock fb = Variable.ForEach(time))
            {
                using (Variable.If(fb.Index == 0))
                {
                    X[time][dim] = Variable.Discrete(Xprior).ForEach(dim);
                }
                using (Variable.If(fb.Index > 0))
                {
                    //var prevX = X[prevTime[time]];
                    var prevX = X[fb.Index - 1];
                    var prevX0 = Variable.Copy(prevX[0]).Named("prevX0");
                    prevX0.SetValueRange(prevState0);
                    // this gives a nicer serial schedule
                    prevX0.AddAttribute(new DivideMessages(false));
                    var prevX1 = Variable.Copy(prevX[1]).Named("prevX1");
                    prevX1.SetValueRange(prevState1);
                    using (Variable.Switch(prevX0))
                    using (Variable.Switch(prevX1))
                    using (Variable.ForEach(dim))
                    {
                        var prediction = Variable.Discrete(transitionProbs[prevX0][prevX1]);
                        X[time][dim] = prediction;
                    }
                }
                using (Variable.ForEach(dim))
                {
                    using (Variable.Switch(X[time][dim]))
                    {
                        S[time][dim] = Variable.GaussianFromMeanAndPrecision(emissionMean[X[time][dim]], 1.0);
                    }
                }
            }

            // generate data from the model
            Rand.Restart(0);
            S.ObservedValue = Util.ArrayInit(numTimeSteps, t => Util.ArrayInit(numDimensions, d => 0.0));
            double[] emissionMeanTrue = Util.ArrayInit(numStates, s => 10.0 * s);
            VariableArray<Gaussian> emissionMeanInit = Variable.Array<Gaussian>(state);
            emissionMean[state].InitialiseTo(emissionMeanInit[state]);
            emissionMeanInit.ObservedValue = emissionMeanTrue.Select(m => Gaussian.PointMass(m)).ToArray();
            Vector[][] transitionProbsTrue = Util.ArrayInit(numStates, s1 => Util.ArrayInit(numStates, s2 => {
                var v = Vector.FromArray(Util.ArrayInit(numStates, s => (s == s1) && (s != s2) ? 0.9 : 0.1 / (numStates - 1)));
                v.Scale(1.0 / v.Sum());
                return v;
                }));
            int[] Xtrue = new int[numDimensions];
            for (int t = 0; t < numTimeSteps; t++)
            {
                if (t > 0)
                {
                    int prevX0 = Xtrue[0];
                    int prevX1 = Xtrue[1];
                    for (int d = 0; d < numDimensions; d++)
                    {
                        Xtrue[d] = Rand.Sample(transitionProbsTrue[prevX0][prevX1]);
                    }
                }
                for (int d = 0; d < numDimensions; d++)
                {
                    S.ObservedValue[t][d] = Gaussian.Sample(emissionMeanTrue[Xtrue[d]], 1.0);
                }
            }

            InferenceEngine engine = new InferenceEngine(algorithm);
            engine.NumberOfIterations = 1;
            var transitionProbs1 = engine.Infer(transitionProbs);
            engine.NumberOfIterations = 20;
            // must build X first to avoid use-before-def problem
            var xActual = engine.Infer<IReadOnlyList<IReadOnlyList<Discrete>>>(X);
            //Console.WriteLine("X[0] = {0}", xActual[0]);
            //Console.WriteLine("emissionMean:");
            //Console.WriteLine(engine.Infer(emissionMean));
            var transitionProbsActual = engine.Infer<IReadOnlyList<IReadOnlyList<Dirichlet>>>(transitionProbs);
            //Console.WriteLine("transitionProbs = {0}", transitionProbsActual);
            Console.WriteLine("transitionProbs:");
            for (int s1 = 0; s1 < numStates; s1++)
            {
                for (int s2 = 0; s2 < numStates; s2++)
                {
                    var transitionMeanActual = transitionProbsActual[s1][s2].GetMean();
                    var transitionMeanExpected = transitionProbsTrue[s1][s2];
                    var error = transitionMeanExpected.MaxDiff(transitionMeanActual);
                    Console.WriteLine($"[{s1}][{s2}] {transitionMeanActual} should be {transitionMeanExpected} (error {error:f2})");
                    Assert.True(error < 0.2);
                }
            }
            // test resetting inference
            engine.NumberOfIterations = 1;
            var transitionProbs2 = engine.Infer<Diffable>(transitionProbs);
            Assert.True(transitionProbs2.MaxDiff(transitionProbs1) < 1e-10);
        }

        // Tests the ModelBuilder ordering bug that once existed
        [Fact]
        public void CoupledHmm3Test2()
        {
            int numTimeSteps = 3;
            int numDimensions = 3;
            int numStates = 3;
            Range time = new Range(numTimeSteps).Named("time");
            Range dim = new Range(numDimensions).Named("dim");
            Range state = new Range(numStates).Named("state");
            Range prevState0 = state.Clone().Named("prevState0");
            Range prevState1 = state.Clone().Named("prevState1");
            Range prevState2 = state.Clone().Named("prevState2");
            // parameters
            Variable<Vector> Xprior = Variable.DirichletSymmetric(state, 1.0).Named("Xprior");
            var transitionProbs = Variable.Array(Variable.Array(Variable.Array<Vector>(prevState2), prevState1), prevState0).Named("transitionProbs");
            transitionProbs[prevState0][prevState1][prevState2] = Variable.DirichletSymmetric(state, 1.0).ForEach(prevState0, prevState1, prevState2);
            // hidden states and observations
            var X = Variable.Array(Variable.Array<int>(dim), time).Named("X");
            VariableArray<int> prevTime = Variable.Array<int>(time).Named("prevTime");
            prevTime.ObservedValue = System.Linq.Enumerable.Range(-1, numTimeSteps).ToArray();
            using (ForEachBlock fb = Variable.ForEach(time))
            {
                using (Variable.If(fb.Index == 0))
                {
                    X[time][dim] = Variable.Discrete(Xprior).ForEach(dim);
                }
                using (Variable.If(fb.Index > 0))
                {
                    //var prevX = X[prevTime[time]];
                    var prevX = X[fb.Index - 1];
                    var prevX0 = Variable.Copy(prevX[0]).Named("prevX0");
                    prevX0.SetValueRange(prevState0);
                    var prevX1 = Variable.Copy(prevX[1]).Named("prevX1");
                    prevX1.SetValueRange(prevState1);
                    var prevX2 = Variable.Copy(prevX[2]).Named("prevX2");
                    prevX2.SetValueRange(prevState2);
                    using (Variable.Switch(prevX0))
                    using (Variable.Switch(prevX1))
                    using (Variable.Switch(prevX2))
                    using (Variable.ForEach(dim))
                    {
                        var prediction = Variable.Discrete(transitionProbs[prevX0][prevX1][prevX2]);
                        X[time][dim] = prediction;
                    }
                }
            }

            InferenceEngine engine = new InferenceEngine();
            // must infer transitionProbs first to get the use-before-def problem
            Console.WriteLine("transitionProbs:");
            Console.WriteLine(engine.Infer(transitionProbs));
            // build X first to avoid use-before-def problem
            Console.WriteLine("X:");
            Console.WriteLine(engine.Infer(X));
        }

        // Fails with error:
        // Internal: schedule splits group 38 at node 13 [38] prevX0_selector_cases_F[time][_iv]
        [Trait("Category", "OpenBug")]
        [Fact]
        public void HmmExample()
        {
            int numTimeSteps = 3;
            int numStates = 3;
            Range time = new Range(numTimeSteps).Named("time");
            Range state = new Range(numStates).Named("state");
            Range prevState0 = state.Clone().Named("prevState0");
            // parameters
            Variable<Vector> Xprior = Variable.DirichletSymmetric(state, 1.0).Named("Xprior");
            var transitionProbs = Variable.Array<Vector>(prevState0).Named("transitionProbs");
            transitionProbs[prevState0] = Variable.DirichletSymmetric(state, 1.0).ForEach(prevState0);
            VariableArray<double> emissionMean = Variable.Array<double>(state).Named("emissionMean");
            emissionMean[state] = Variable.GaussianFromMeanAndPrecision(0, 0.01).ForEach(state);
            // hidden states and observations
            var X = Variable.Array<int>(time).Named("X");
            var S = Variable.Array<double>(time).Named("S");
            VariableArray<int> prevTime = Variable.Array<int>(time).Named("prevTime");
            prevTime.ObservedValue = System.Linq.Enumerable.Range(-1, numTimeSteps).ToArray();
            using (ForEachBlock fb = Variable.ForEach(time))
            {
                using (Variable.If(fb.Index == 0))
                {
                    X[time] = Variable.Discrete(Xprior);
                }
                using (Variable.If(fb.Index > 0))
                {
                    //var prevX = X[prevTime[time]];
                    var prevX = X[fb.Index - 1];
                    var prevX0 = Variable.Copy(prevX).Named("prevX0");
                    prevX0.SetValueRange(prevState0);
                    using (Variable.Switch(prevX0))
                    {
                        var prediction = Variable.Discrete(transitionProbs[prevX0]);
                        X[time] = prediction;
                    }
                }
                using (Variable.Switch(X[time]))
                {
                    S[time] = Variable.GaussianFromMeanAndPrecision(emissionMean[X[time]], 1.0);
                }
            }
            S.ObservedValue = Util.ArrayInit(numTimeSteps, t => t + 0.0);

            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = new VariationalMessagePassing();
            Console.WriteLine("emissionMean:");
            Console.WriteLine(engine.Infer(emissionMean));
            Console.WriteLine("transitionProbs:");
            Console.WriteLine(engine.Infer(transitionProbs));
            Console.WriteLine("X:");
            Console.WriteLine(engine.Infer(X));
        }

        // Fails with Optimize=false due to flawed logic in repair for InitSchedule
        [Trait("Category", "OpenBug")]
        [Fact]
        public void CoupledHmm3UnrolledTest()
        {
            CoupledHmm3Unrolled(new VariationalMessagePassing());
            CoupledHmm3Unrolled(new ExpectationPropagation());
        }

        private void CoupledHmm3Unrolled(IAlgorithm algorithm)
        {
            int numTimeSteps = 3;
            int numDimensions = 3;
            int numStates = 3;
            Range state = new Range(numStates).Named("state");
            Range prevState0 = state.Clone().Named("prevState0");
            Range prevState1 = state.Clone().Named("prevState1");
            Range prevState2 = state.Clone().Named("prevState2");
            // parameters
            Variable<Vector> Xprior = Variable.DirichletSymmetric(state, 1.0).Named("Xprior");
            var transitionProbs = Variable.Array(Variable.Array(Variable.Array<Vector>(prevState2), prevState1), prevState0).Named("transitionProbs");
            transitionProbs[prevState0][prevState1][prevState2] = Variable.DirichletSymmetric(state, 1.0).ForEach(prevState0, prevState1, prevState2);
            VariableArray<double> emissionMean = Variable.Array<double>(state).Named("emissionMean");
            emissionMean[state] = Variable.GaussianFromMeanAndPrecision(0, 0.01).ForEach(state);
            // hidden states and observations
            Variable<int>[][] X = new Variable<int>[numTimeSteps][];
            Variable<double>[][] S = new Variable<double>[numTimeSteps][];
            for (int t = 0; t < numTimeSteps; t++)
            {
                X[t] = new Variable<int>[numDimensions];
                S[t] = new Variable<double>[numDimensions];
                for (int d = 0; d < numDimensions; d++)
                {
                    if (t == 0)
                    {
                        // generate from prior
                        X[t][d] = Variable.Discrete(Xprior);
                    }
                    else
                    {
                        // generate from previous timestep
                        X[t][d] = Variable.New<int>();
                        Variable<int> prevX0 = Variable.Copy(X[t - 1][0]).Named("prevX_" + t + "_" + d + "_0");
                        prevX0.SetValueRange(prevState0);
                        Variable<int> prevX1 = Variable.Copy(X[t - 1][1]).Named("prevX_" + t + "_" + d + "_1");
                        prevX1.SetValueRange(prevState1);
                        Variable<int> prevX2 = Variable.Copy(X[t - 1][2]).Named("prevX_" + t + "_" + d + "_2");
                        prevX2.SetValueRange(prevState2);
                        using (Variable.Switch(prevX0))
                        using (Variable.Switch(prevX1))
                        using (Variable.Switch(prevX2))
                            X[t][d].SetTo(Variable.Discrete(transitionProbs[prevX0][prevX1][prevX2]));
                    }
                    X[t][d].Name = "X_" + t + "_" + d;
                    X[t][d].InitialiseTo(Discrete.PointMass(Rand.Int(numStates), numStates));
                    // observations
                    S[t][d] = Variable.New<double>().Named("S_" + t + "_" + d);
                    using (Variable.Switch(X[t][d]))
                    {
                        S[t][d].SetTo(Variable.GaussianFromMeanAndPrecision(emissionMean[X[t][d]], 1.0));
                    }
                    S[t][d].ObservedValue = d;
                }
            }

            InferenceEngine engine = new InferenceEngine(algorithm);
            Console.WriteLine("emissionMean:");
            Console.WriteLine(engine.Infer(emissionMean));
            Console.WriteLine("transitionProbs:");
            Console.WriteLine(engine.Infer(transitionProbs));
            for (int t = 0; t < numTimeSteps; t++)
            {
                for (int d = 0; d < numDimensions; d++)
                {
                    Console.WriteLine("X[{0}][{1}] = {2}", t, d, engine.Infer(X[t][d]));
                }
            }
        }

        [Fact]
        public void CoupledHmm2UnrolledTest()
        {
            CoupledHmm2Unrolled(new ExpectationPropagation());
            CoupledHmm2Unrolled(new VariationalMessagePassing());
        }

        private void CoupledHmm2Unrolled(IAlgorithm algorithm)
        {
            int numTimeSteps = 3;
            int numDimensions = 2;
            int numStates = 3;
            Range state = new Range(numStates).Named("state");
            Range prevState0 = state.Clone().Named("prevState0");
            Range prevState1 = state.Clone().Named("prevState1");
            // parameters
            Variable<Vector> Xprior = Variable.DirichletSymmetric(state, 1.0).Named("Xprior");
            var transitionProbs = Variable.Array(Variable.Array<Vector>(prevState1), prevState0).Named("transitionProbs");
            transitionProbs[prevState0][prevState1] = Variable.DirichletSymmetric(state, 1.0).ForEach(prevState0, prevState1);
            VariableArray<double> emissionMean = Variable.Array<double>(state).Named("emissionMean");
            emissionMean[state] = Variable.GaussianFromMeanAndPrecision(0, 0.01).ForEach(state);
            // hidden states and observations
            Variable<int>[][] X = new Variable<int>[numTimeSteps][];
            Variable<double>[][] S = new Variable<double>[numTimeSteps][];
            for (int t = 0; t < numTimeSteps; t++)
            {
                X[t] = new Variable<int>[numDimensions];
                S[t] = new Variable<double>[numDimensions];
                for (int d = 0; d < numDimensions; d++)
                {
                    if (t == 0)
                    {
                        // generate from prior
                        X[t][d] = Variable.Discrete(Xprior);
                    }
                    else
                    {
                        // generate from previous timestep
                        X[t][d] = Variable.New<int>();
                        Variable<int> prevX0 = Variable.Copy(X[t - 1][0]).Named("prevX_" + t + "_" + d + "_0");
                        prevX0.SetValueRange(prevState0);
                        Variable<int> prevX1 = Variable.Copy(X[t - 1][1]).Named("prevX_" + t + "_" + d + "_1");
                        prevX1.SetValueRange(prevState1);
                        using (Variable.Switch(prevX0))
                        using (Variable.Switch(prevX1))
                            X[t][d].SetTo(Variable.Discrete(transitionProbs[prevX0][prevX1]));
                    }
                    X[t][d].Name = "X_" + t + "_" + d;
                    X[t][d].InitialiseTo(Discrete.PointMass(Rand.Int(numStates), numStates));
                    if (true)
                    {
                        S[t][d] = Variable.New<double>().Named("S_" + t + "_" + d);
                        using (Variable.Switch(X[t][d]))
                        {
                            S[t][d].SetTo(Variable.GaussianFromMeanAndPrecision(emissionMean[X[t][d]], 1.0));
                        }
                        S[t][d].ObservedValue = d;
                    }
                }
            }

            InferenceEngine engine = new InferenceEngine(algorithm);
            Console.WriteLine("emissionMean:");
            Console.WriteLine(engine.Infer(emissionMean));
            Console.WriteLine("transitionProbs:");
            Console.WriteLine(engine.Infer(transitionProbs));
            for (int t = 0; t < numTimeSteps; t++)
            {
                for (int d = 0; d < numDimensions; d++)
                {
                    Console.WriteLine("X[{0}][{1}] = {2}", t, d, engine.Infer(X[t][d]));
                }
            }
        }

        [Fact]
        public void CoupledHmm2UnrolledBinary()
        {
            int numTimeSteps = 4;
            int numDimensions = 2;
            // parameters
            Variable<double> Xprior = Variable.Beta(1, 1).Named("Xprior");
            // hidden states and observations
            Variable<bool>[][] X = new Variable<bool>[numTimeSteps][];
            Variable<bool>[][] S = new Variable<bool>[numTimeSteps][];
            for (int t = 0; t < numTimeSteps; t++)
            {
                X[t] = new Variable<bool>[numDimensions];
                S[t] = new Variable<bool>[numDimensions];
                for (int d = 0; d < numDimensions; d++)
                {
                    if (t == 0)
                    {
                        // generate from prior
                        X[t][d] = Variable.Bernoulli(Xprior);
                    }
                    else
                    {
                        // generate from previous timestep
                        X[t][d] = Variable.New<bool>();
                        var prevX0 = Variable.Copy(X[t - 1][0]).Named("prevX_" + t + "_" + d + "_0");
                        var prevX1 = Variable.Copy(X[t - 1][1]).Named("prevX_" + t + "_" + d + "_1");
                        FromTwoParents(X[t][d], prevX0, prevX1, 0.5, 0.5, 0.5, 0.5);
                    }
                    X[t][d].Name = "X_" + t + "_" + d;
                    if (true)
                    {
                        S[t][d] = Variable.New<bool>().Named("S_" + t + "_" + d);
                        FromSingleParent(S[t][d], X[t][d], 0.5, 0.5);
                        S[t][d].ObservedValue = (d > 0);
                    }
                }
            }

            InferenceEngine engine = new InferenceEngine();
            //Console.WriteLine("emissionMean:");
            //Console.WriteLine(engine.Infer(emissionMean));
            //Console.WriteLine("transitionProbs:");
            //Console.WriteLine(engine.Infer(transitionProbs));
            for (int t = 0; t < numTimeSteps; t++)
            {
                for (int d = 0; d < numDimensions; d++)
                {
                    Console.WriteLine("X[{0}][{1}] = {2}", t, d, engine.Infer(X[t][d]));
                }
            }
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
        public void VectorGaussianFromPrecisionDiagonalExample()
        {
            Range d = new Range(3);
            VariableArray<double> prec = Variable.Array<double>(d);
            prec[d] = Variable.GammaFromShapeAndRate(1, 1).ForEach(d);
            // this initialization helps convergence when there is no data
            prec.InitialiseTo(Distribution<double>.Array(Util.ArrayInit(3, i => new Gamma(1, 1))));
            Variable<Vector> x = VectorGaussianFromPrecisionDiagonal(prec).Named("x");
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            VectorGaussian actual = engine.Infer<VectorGaussian>(x);
            VectorGaussian expected = new VectorGaussian(Vector.Zero(3), PositiveDefiniteMatrix.Identity(3));
            Console.WriteLine(StringUtil.JoinColumns("x = ", actual, " should be ", expected));
            Assert.True(expected.MaxDiff(actual) < 1e-4);
        }

        public Variable<Vector> VectorGaussianFromPrecisionDiagonal(VariableArray<double> precisionDiagonal)
        {
            Range r = precisionDiagonal.Range;
            int dimension = r.SizeAsInt;
            Variable<Vector> x = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(dimension), PositiveDefiniteMatrix.IdentityScaledBy(dimension, 1e-20));
            using (var fb = Variable.ForEach(r))
            {
                Variable<double> xi = Variable.GetItem(x, fb.Index);
                var y = Variable.GaussianFromMeanAndPrecision(xi, precisionDiagonal[r]);
                y.ObservedValue = 0;
            }
            return x;
        }

        internal void PokerExample()
        {
            Range game = new Range(3);
            Range possibleAction = new Range(3);
            Range possibleHand = new Range(169);
            VariableArray<int> hand = Variable.Array<int>(game);
            VariableArray<Vector> actionProbs = Variable.Array<Vector>(possibleHand);
            actionProbs[possibleHand] = Variable.DirichletUniform(possibleAction).ForEach(possibleHand);
            VariableArray<int> action = Variable.Array<int>(game);
            using (Variable.ForEach(game))
            {
                hand[game] = Variable.DiscreteUniform(possibleHand);
                using (Variable.Switch(hand[game]))
                {
                    action[game] = Variable.Discrete(actionProbs[hand[game]]);
                }
            }

            InferenceEngine engine = new InferenceEngine();
            hand.ObservedValue = new int[] {4, 5, 6};
            action.ObservedValue = new int[] {2, 1, 0};
            Console.WriteLine(engine.Infer(actionProbs));

            Variable<int> newHand = Variable.DiscreteUniform(possibleHand);
            Variable<int> newAction = Variable.New<int>();
            using (Variable.Switch(newHand))
            {
                newAction.SetTo(Variable.Discrete(actionProbs[newHand]));
            }
            newAction.ObservedValue = 2;
            Console.WriteLine("predicted hand given action = {0}:", newAction.ObservedValue);
            Console.WriteLine(engine.Infer(newHand));
        }

        // this is 
        internal void SGuy()
        {
            Variable<int> a = Variable.Discrete(new double[] {0.5, 0.5});
            Variable<bool> b = Variable.New<bool>();
            Variable<bool> c = Variable.New<bool>();
            Variable<bool> d = Variable.New<bool>();
            using (Variable.Case(a, 0))
            {
                b.SetTo(true);
                c.SetTo(false);
                d = b | c;
            }
            using (Variable.Case(a, 1))
            {
                b.SetTo(false);
                c.SetTo(true);
                d = b | c;
            }
            InferenceEngine ie = new InferenceEngine();
            var post = ie.Infer<Bernoulli>(d);
            Console.WriteLine("b: " + post); // d = Bernoulli(1)
            Assert.True(post.GetMean() == 1.0);
        }


        [Fact]
        public void BabyTest()
        {
            // Observed actions
            BabyAction[] observedActions = new BabyAction[] {BabyAction.Smile, BabyAction.Smile, BabyAction.Cry, BabyAction.Smile, BabyAction.LookSilly};
            int[] acts = Array.ConvertAll(observedActions, act => (int) act);
            var actions = Variable.Observed(acts);

            // Conditional probabilities of actions given different attitudes
            var actionProbs = Variable.Observed(new Vector[]
                {
                    Vector.FromArray(0.6, 0.2, 0.2), // Happy
                    Vector.FromArray(0.2, 0.6, 0.2), // Unhappy
                    Vector.FromArray(0.4, 0.3, 0.3), // Quiet
                });

            // Model relating attitude to actions
            var attitude = Variable.DiscreteUniform(actionProbs.Range);
            var j = actions.Range;
            using (Variable.Switch(attitude))
            {
                actions[j] = Variable.Discrete(actionProbs[attitude]).ForEach(j);
            }

            // Inference of the posterior distribution over attitudes
            InferenceEngine ie = new InferenceEngine();
            Discrete posterior = ie.Infer<Discrete>(attitude);
            foreach (Attitude att in Enum.GetValues(typeof (Attitude)))
            {
                Console.WriteLine("Probability of {0} \t= {1}", att, posterior[(int) att]);
            }
        }


        internal void AffinityPropagation()
        {
            int N = 20;
            double[,] d = new double[N,2];
            for (int i = 0; i < N; i++)
            {
                d[i, 0] = Rand.Double();
                d[i, 1] = Rand.Double();
            }
            Discrete[] s = new Discrete[N];
            double alpha = -0.1;
            for (int i = 0; i < N; i++)
            {
                double[] probs = new double[N];
                for (int j = 0; j < N; j++) probs[j] = System.Math.Exp(-((d[i, 0] - d[j, 0])*(d[i, 0] - d[j, 0]) + (d[i, 1] - d[j, 1])*(d[i, 1] - d[j, 1])));
                probs[i] = System.Math.Exp(alpha);
                s[i] = new Discrete(probs);
            }
            Range nRange = new Range(N);
            Range nRange2 = new Range(N);
            VariableArray<int> c = Variable.Array<int>(nRange);
            VariableArray<Discrete> sprior = Variable.Observed(s, nRange);
            c[nRange] = Variable.Random<int, Discrete>(sprior[nRange]);
            int[] numbers = new int[N];
            for (int i = 0; i < N; i++) numbers[i] = i;
            VariableArray<int> nums = Variable.Observed(numbers, nRange);
            using (Variable.ForEach(nRange))
            {
                using (Variable.IfNot(c[nRange] == nums[nRange]))
                {
                    Variable.ConstrainFalse(c[nRange2] == nums[nRange]);
                }
            }
        }

        /// <summary>
        /// Inference on an undirected graph with binary unary and pairwise potentials.
        /// </summary>
        /// <remarks>
        /// Note that the generated model involves only 3 constants, instead of the 8 constants required by the tabular representation.
        /// </remarks>
        internal void BinaryPairwisePotentials()
        {
            double x1LogOdds = System.Math.Log(2.0/1.0); // first unary potential
            double x2LogOdds = System.Math.Log(4.0/3.0); // second unary potential
            double equalityLogOdds, aLogOdds, bLogOdds;
            BinaryPairwiseAsEquality(5, 6, 7, 8, out equalityLogOdds, out aLogOdds, out bLogOdds);

            Variable<bool> x1 = Variable.BernoulliFromLogOdds(x1LogOdds + aLogOdds); // combine the unary potentials into one prior distribution
            Variable<bool> x2 = Variable.BernoulliFromLogOdds(x2LogOdds + bLogOdds);
            Variable.ConstrainEqualRandom(x1 == x2, Bernoulli.FromLogOdds(equalityLogOdds)); // represent the pairwise potential as soft equality

            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine("x1 = {0}", engine.Infer(x1)); // x1 = Bernoulli(0.731)
            Console.WriteLine("x2 = {0}", engine.Infer(x2)); // x2 = Bernoulli(0.6069)
        }

        /// <summary>
        /// Represent a binary pairwise factor as a soft equality times unary potentials.
        /// </summary>
        /// <param name="pFF">Potential value for FF case, must be &gt; 0</param>
        /// <param name="pFT">Potential value for FT case, must be &gt; 0</param>
        /// <param name="pTF">Potential value for TF case, must be &gt; 0</param>
        /// <param name="pTT">Potential value for TT case, must be &gt; 0</param>
        /// <param name="equalityLogOdds">LogOdds of the soft equality</param>
        /// <param name="aLogOdds">LogOdds of first variable's unary potential</param>
        /// <param name="bLogOdds">LogOdds of second variable's unary potential</param>
        private static void BinaryPairwiseAsEquality(double pFF, double pFT, double pTF, double pTT, out double equalityLogOdds, out double aLogOdds, out double bLogOdds)
        {
            if (pFF <= 0) throw new ArgumentException("pFF <= 0");
            if (pFT <= 0) throw new ArgumentException("pFT <= 0");
            if (pTF <= 0) throw new ArgumentException("pTF <= 0");
            if (pTT <= 0) throw new ArgumentException("pTT <= 0");
            double logFF = System.Math.Log(pFF);
            double logFT = System.Math.Log(pFT);
            double logTF = System.Math.Log(pTF);
            double logTT = System.Math.Log(pTT);
            equalityLogOdds = 0.5*(logFF + logTT - logFT - logTF);
            aLogOdds = equalityLogOdds + logTF - logFF;
            bLogOdds = equalityLogOdds + logFT - logFF;
        }

        /// <summary>
        /// Another encoding of the model in BinaryPairwisePotentials
        /// </summary>
        internal void BinaryPairwisePotentials2()
        {
            // See forum thread http://community.research.microsoft.com/forums/t/4448.aspx
            double[] x1Prior = {1.0/3.0, 2.0/3.0};
            Variable<int> x1 = Variable.Discrete(x1Prior);
            double[] x2Prior = {3.0/7.0, 4.0/7.0};
            Variable<int> x2 = Variable.Discrete(x2Prior);

            if (false)
            {
                // this approach is not exact for BP
                Variable<bool> f1 = (x1 == 0);
                Variable<bool> t1 = (x1 == 1);
                Variable<bool> f2 = (x2 == 0);
                Variable<bool> t2 = (x2 == 1);

                Variable<bool> ff = (f1 & f2);
                Variable<bool> ft = (f1 & t2);
                Variable<bool> tf = (t1 & f2);
                Variable<bool> tt = (t1 & t2);

                Variable.ConstrainEqualRandom(ff, Bernoulli.FromLogOdds(System.Math.Log(5.0/1.0)));
                Variable.ConstrainEqualRandom(ft, Bernoulli.FromLogOdds(System.Math.Log(6.0/1.0)));
                Variable.ConstrainEqualRandom(tf, Bernoulli.FromLogOdds(System.Math.Log(7.0/1.0)));
                Variable.ConstrainEqualRandom(tt, Bernoulli.FromLogOdds(System.Math.Log(8.0/1.0)));
            }
            else if (false)
            {
                // this approach is exact for BP
                x1.SetValueRange(new Range(2));
                using (Variable.Switch(x1))
                {
                    Variable<bool> f1 = (x1 == 0);
                    Variable<bool> t1 = (x1 == 1);
                    Variable<bool> f2 = (x2 == 0);
                    Variable<bool> t2 = (x2 == 1);

                    Variable<bool> ff = (f1 & f2);
                    Variable<bool> ft = (f1 & t2);
                    Variable<bool> tf = (t1 & f2);
                    Variable<bool> tt = (t1 & t2);

                    Variable.ConstrainEqualRandom(ff, Bernoulli.FromLogOdds(System.Math.Log(5.0/1.0)));
                    Variable.ConstrainEqualRandom(ft, Bernoulli.FromLogOdds(System.Math.Log(6.0/1.0)));
                    Variable.ConstrainEqualRandom(tf, Bernoulli.FromLogOdds(System.Math.Log(7.0/1.0)));
                    Variable.ConstrainEqualRandom(tt, Bernoulli.FromLogOdds(System.Math.Log(8.0/1.0)));
                }
            }
            else if (false)
            {
                // this approach is exact for BP
                Variable<bool> ff = Variable.BernoulliFromLogOdds(System.Math.Log(5.0/1.0));
                Variable<bool> ft = Variable.BernoulliFromLogOdds(System.Math.Log(6.0/1.0));
                Variable<bool> tf = Variable.BernoulliFromLogOdds(System.Math.Log(7.0/1.0));
                Variable<bool> tt = Variable.BernoulliFromLogOdds(System.Math.Log(8.0/1.0));
                using (Variable.Case(x1, 0))
                {
                    using (Variable.Case(x2, 0))
                    {
                        Variable.ConstrainTrue(ff);
                        Variable.ConstrainFalse(ft);
                    }
                    using (Variable.Case(x2, 1))
                    {
                        Variable.ConstrainFalse(ff);
                        Variable.ConstrainTrue(ft);
                    }
                    Variable.ConstrainFalse(tf);
                    Variable.ConstrainFalse(tt);
                }
                using (Variable.Case(x1, 1))
                {
                    Variable.ConstrainFalse(ff);
                    Variable.ConstrainFalse(ft);
                    using (Variable.Case(x2, 0))
                    {
                        Variable.ConstrainTrue(tf);
                        Variable.ConstrainFalse(tt);
                    }
                    using (Variable.Case(x2, 1))
                    {
                        Variable.ConstrainFalse(tf);
                        Variable.ConstrainTrue(tt);
                    }
                }
            }
            else if (false)
            {
                // this approach is exact for BP
                Variable<int> x12 = Variable.New<int>();
                x12.SetValueRange(new Range(4));
                using (Variable.Case(x1, 0))
                {
                    using (Variable.Case(x2, 0))
                    {
                        x12.SetTo(0);
                    }
                    using (Variable.Case(x2, 1))
                    {
                        x12.SetTo(1);
                    }
                }
                using (Variable.Case(x1, 1))
                {
                    using (Variable.Case(x2, 0))
                    {
                        x12.SetTo(2);
                    }
                    using (Variable.Case(x2, 1))
                    {
                        x12.SetTo(3);
                    }
                }
                Variable.ConstrainEqualRandom(x12, new Discrete(5, 6, 7, 8));
            }
            else
            {
                Variable<int> x1x2 = Variable.Discrete(5, 6, 7, 8);
                using (Variable.Case(x1x2, 0))
                {
                    Variable.ConstrainEqual(x1, 0);
                    Variable.ConstrainEqual(x2, 0);
                }
                using (Variable.Case(x1x2, 1))
                {
                    Variable.ConstrainEqual(x1, 0);
                    Variable.ConstrainEqual(x2, 1);
                }
                using (Variable.Case(x1x2, 2))
                {
                    Variable.ConstrainEqual(x1, 1);
                    Variable.ConstrainEqual(x2, 0);
                }
                using (Variable.Case(x1x2, 3))
                {
                    Variable.ConstrainEqual(x1, 1);
                    Variable.ConstrainEqual(x2, 1);
                }
            }

            // EP on the loopy graph gives:
            // x1 = Discrete(0.3696 0.6304)
            // x2 = Discrete(0.4399 0.5601) 
            InferenceEngine engine = new InferenceEngine();
            //engine.Algorithm = new GibbsSampling();
            Discrete x1Expected = new Discrete(1 - 0.731, 0.731);
            Discrete x2Expected = new Discrete(1 - 0.6069, 0.6069);
            //x1Expected = new Discrete(x1Prior[0]*(5*x2Prior[0] + x2Prior[1]), x1Prior[1]);
            //x1Expected = new Discrete(x1Prior[0]*(5*x2Prior[0] + 6*x2Prior[1]), x1Prior[1]);
            Discrete x1Actual = engine.Infer<Discrete>(x1);
            Discrete x2Actual = engine.Infer<Discrete>(x2);
            Console.WriteLine("x1 = {0} should be {1}", x1Actual, x1Expected);
            Console.WriteLine("x2 = {0} should be {1}", x2Actual, x2Expected);
        }

        [Fact]
        public void BinaryPairwisePotentials3Test()
        {
            BinaryPairwisePotentials3(false, false, false);
            BinaryPairwisePotentials3(false, false, true);
        }

        // TODO: generated source has many optimization opportunities.
        private void BinaryPairwisePotentials3(bool xInit, bool yInit, bool zInit)
        {
            int length = 2;

            double[] xProbs = {0.4, 0.6};
            Variable<int> x = Variable.Discrete(xProbs).Named("x");
            double[] yProbs = {0.7, 0.3};
            Variable<int> y = Variable.Discrete(yProbs).Named("y");
            double[] zProbs = {0.8, 0.2};
            Variable<int> z = Variable.Discrete(zProbs).Named("z");

            Variable<int> xy = Variable.Discrete(1.0, 0.5, 0.5, 1.0).Named("xy");
            Variable<int> yz = Variable.Discrete(1.0, 0.5, 0.5, 1.0).Named("yz");
            Variable<int> zx = Variable.Discrete(1.0, 0.5, 0.5, 1.0).Named("zx");

            int pair_index = 0;

            for (int i = 0; i < length; i++)
            {
                for (int j = 0; j < length; j++, pair_index++)
                {
                    using (Variable.Case(xy, pair_index))
                    {
                        Variable.ConstrainEqual(x, i);
                        Variable.ConstrainEqual(y, j);
                    }
                    using (Variable.Case(yz, pair_index))
                    {
                        Variable.ConstrainEqual(y, i);
                        Variable.ConstrainEqual(z, j);
                    }
                    using (Variable.Case(zx, pair_index))
                    {
                        Variable.ConstrainEqual(z, i);
                        Variable.ConstrainEqual(x, j);
                    }
                }
            }
            if (xInit) x.InitialiseTo(new Discrete(zProbs));
            if (yInit) y.InitialiseTo(new Discrete(yProbs));
            if (zInit) z.InitialiseTo(new Discrete(zProbs));

            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            //engine.NumberOfIterations = 200;
            double xLogOdds = -0.430329845153877;
            double yLogOdds = -1.26916217135997;
            double zLogOdds = -1.6494367601758;
            Discrete xExpected = new Discrete(MMath.Logistic(-xLogOdds), MMath.Logistic(xLogOdds));
            Discrete yExpected = new Discrete(MMath.Logistic(-yLogOdds), MMath.Logistic(yLogOdds));
            Discrete zExpected = new Discrete(MMath.Logistic(-zLogOdds), MMath.Logistic(zLogOdds));
            Discrete xActual = engine.Infer<Discrete>(x);
            Discrete yActual = engine.Infer<Discrete>(y);
            Discrete zActual = engine.Infer<Discrete>(z);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Console.WriteLine("y = {0} should be {1}", yActual, yExpected);
            Console.WriteLine("z = {0} should be {1}", zActual, zExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-6);
            Assert.True(yExpected.MaxDiff(yActual) < 1e-6);
            Assert.True(zExpected.MaxDiff(zActual) < 1e-6);
        }

        [Fact]
        public void BinaryPairwisePotentials4()
        {
            double xLogOdds = System.Math.Log(0.6/0.4); // first unary potential
            double yLogOdds = System.Math.Log(0.3/0.7); // second unary potential
            double zLogOdds = System.Math.Log(0.2/0.8); // second unary potential
            double equalityLogOdds, aLogOdds, bLogOdds;
            BinaryPairwiseAsEquality(1, 0.5, 0.5, 1, out equalityLogOdds, out aLogOdds, out bLogOdds);

            Variable<bool> x = Variable.BernoulliFromLogOdds(xLogOdds + aLogOdds + bLogOdds); // combine the unary potentials into one prior distribution
            Variable<bool> y = Variable.BernoulliFromLogOdds(yLogOdds + bLogOdds + aLogOdds);
            Variable<bool> z = Variable.BernoulliFromLogOdds(zLogOdds + bLogOdds + aLogOdds);
            Variable.ConstrainEqualRandom(x == y, Bernoulli.FromLogOdds(equalityLogOdds)); // represent the pairwise potential as soft equality
            Variable.ConstrainEqualRandom(x == z, Bernoulli.FromLogOdds(equalityLogOdds)); // represent the pairwise potential as soft equality
            Variable.ConstrainEqualRandom(z == y, Bernoulli.FromLogOdds(equalityLogOdds)); // represent the pairwise potential as soft equality

            // the schedule has two loops (clockwise and counter-clockwise)
            InferenceEngine engine = new InferenceEngine();
            if (false)
            {
                x.InitialiseTo(new Bernoulli(0.1));
                y.InitialiseTo(new Bernoulli(0.1));
                z.InitialiseTo(new Bernoulli(0.1));
            }
            Bernoulli xExpected = Bernoulli.FromLogOdds(-0.430329845153877);
            Bernoulli yExpected = Bernoulli.FromLogOdds(-1.26916217135997);
            Bernoulli zExpected = Bernoulli.FromLogOdds(-1.6494367601758);
            Bernoulli xActual = engine.Infer<Bernoulli>(x);
            Bernoulli yActual = engine.Infer<Bernoulli>(y);
            Bernoulli zActual = engine.Infer<Bernoulli>(z);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Console.WriteLine("y = {0} should be {1}", yActual, yExpected);
            Console.WriteLine("z = {0} should be {1}", zActual, zExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-6);
            Assert.True(yExpected.MaxDiff(yActual) < 1e-6);
            Assert.True(zExpected.MaxDiff(zActual) < 1e-6);
            // x = Bernoulli(0.394)
            // y = Bernoulli(0.2194)
            // z = Bernoulli(0.1612)
        }

        [Fact]
        public void HowToMissingDataGaussian()
        {
            // Sample data from standard Gaussian
            double[] data = new double[100];
            for (int i = 0; i < data.Length; i++) data[i] = Rand.Normal(0, 1);


            // Create mean and precision random variables
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100).Named("mean");
            Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1).Named("precision");

            Range dataRange = new Range(data.Length);

            // Example pattern of missing data
            bool[] isMissing = new bool[100];
            for (int i = 0; i < isMissing.Length; i++) isMissing[i] = (i%2) == 0;
            VariableArray<bool> isMissingVar = Variable.Observed(isMissing, dataRange);

            VariableArray<double> x = Variable.Array<double>(dataRange);
            using (Variable.ForEach(dataRange))
            {
                using (Variable.IfNot(isMissingVar[dataRange]))
                {
                    x[dataRange] = Variable.GaussianFromMeanAndPrecision(mean, precision);
                }
            }
            x.ObservedValue = data;

            // Create an inference engine for VMP
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            // Retrieve the posterior distributions
            Console.WriteLine("mean=" + engine.Infer(mean));
            Console.WriteLine("prec=" + engine.Infer(precision));
        }

        [Fact]
        public void HowToMissingDataGaussian2()
        {
            // Sample data from standard Gaussian
            double[] data = new double[] {-1, 5.0, -1, 7.0, -1};

            // Create mean and precision random variables
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100).Named("mean");
            Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1).Named("precision");

            Range dataRange = new Range(data.Length);
            VariableArray<double> x = Variable.Array<double>(dataRange);
            using (Variable.ForEach(dataRange))
            {
                using (Variable.If(x[dataRange] != -1))
                {
                    x[dataRange] = Variable.GaussianFromMeanAndPrecision(mean, precision);
                }
            }
            x.ObservedValue = data;

            // Create an inference engine for VMP
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            // Retrieve the posterior distributions
            Console.WriteLine("mean=" + engine.Infer(mean));
            Console.WriteLine("prec=" + engine.Infer(precision));
        }

        internal void ClassicBayesNet()
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

            cloudy.Name = "cloudy";
            sprinkler.Name = "sprinkler";
            rain.Name = "rain";
            wetGrass.Name = "wetGrass";

            // Observations
            wetGrass.ObservedValue = true;
            InferenceEngine ie = new InferenceEngine();
            ie.ShowProgress = false;
            Console.WriteLine("P(rain      | grass is wet)=" + ie.Infer(rain));
            Console.WriteLine("P(sprinkler | grass is wet)=" + ie.Infer(sprinkler));
            cloudy.ObservedValue = false;
            Console.WriteLine("P(rain      | grass is wet, not cloudy)=" + ie.Infer(rain));
            Console.WriteLine("P(sprinkler | grass is wet, not cloudy)=" + ie.Infer(sprinkler));
        }

        internal void GetMixtureOfGammas(double[] data, int nComponents, out Vector weights, out Gamma[] components)
        {
            Range item = new Range(data.Length);
            Range component = new Range(nComponents);
            VariableArray<int> c = Variable.Array<int>(item).Named("c");
            VariableArray<double> x = Variable.Array<double>(item).Named("x");
            x.ObservedValue = data;
            Variable<Vector> probs = Variable.DirichletUniform(component).Named("probs");
            VariableArray<Gamma> gammas = Variable.Array<Gamma>(component).Named("gammas");
            gammas.ObservedValue = new Gamma[nComponents];
            for (int i = 0; i < nComponents; i++)
            {
                double shape = Rand.Double()*1;
                double rate = Rand.Double()*10;
                gammas.ObservedValue[i] = Gamma.FromShapeAndRate(shape, rate);
            }
            //VariableArray<double> shape = Variable.Array<double>(component).Named("shape");
            //VariableArray<double> rate = VariableArray
            using (Variable.ForEach(item))
            {
                c[item] = Variable.Discrete(probs);
                using (Variable.Switch(c[item]))
                {
                    //x[item] = Variable.GammaFromShapeAndRate(shape[c[item]], rate[c[item]]);
                    x[item] = Variable.Random<double, Gamma>(gammas[c[item]]);
                }
            }
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            weights = engine.Infer<Dirichlet>(probs).GetMean();
            components = new Gamma[nComponents];
            for (int i = 0; i < components.Length; i++)
            {
                //components[i] = Gamma.FromShapeAndRate(shape, rate);
                components[i] = gammas.ObservedValue[i];
            }
        }

        [Fact]
        public void HierarchicalNormalModel()
        {
            // Model and data from Gelman et al, "Bayesian Data Analysis", 2nd ed., sec 5.4

            Variable<double> mu = Variable.GaussianFromMeanAndVariance(0, 1e9);
            Variable<int> nItem = Variable.New<int>();
            Range item = new Range(nItem);
            VariableArray<double> theta = Variable.Array<double>(item);
            VariableArray<double> x = Variable.Array<double>(item);
            VariableArray<double> prec = Variable.Array<double>(item);

            // The book uses tau, but we're using tauInverse2.  This requires the prior on tau to be converted
            // into a prior on tauInverse2.
            // tauInverse2 = 1/tau^2   i.e. tau = tauInverse2^(-0.5) 
            // dtau = -0.5 tauInverse2^(-1.5)
            // p(tauInverse2) dtauInverse2 = p(tau=tauInverse2^(-0.5)) dtau
            // p(tauInverse2) = 0.5 tauInverse2^(-1.5) p(tau=tauInverse2^(-0.5))
            // in the book, p(tau) =propto 1, so
            //   p(tauInverse2) =propto tauInverse2^(-1.5)
            // which is an improper Gamma distribution.
            // If you limit tau to [0,tauMax] then tauInverse2 ranges from tauMax^(-1.5) to Infinity.
#if true
            // E[tau] = E[tauInverse2^(-0.5)] = Gamma(a-0.5)/Gamma(a)*sqrt(b)
            // If we want E[tau] = 5 then b = (E[tau]*Gamma(a)/Gamma(a-0.5))^2
            double shape = 0.25;
            double rate = System.Math.Pow(5* MMath.Gamma(shape) / MMath.Gamma(shape - 0.5), 2);
            Variable<double> tauInverse2 = Variable.GammaFromShapeAndScale(shape, 1/rate);
            theta[item] = Variable.GaussianFromMeanAndPrecision(mu, tauInverse2).ForEach(item);
            x[item] = Variable.GaussianFromMeanAndPrecision(theta[item], prec[item]);
#else
    // fit a mixture-of-gammas to the desired prior distribution
      Variable<double> tauInverse2 = Variable.New<double>().Named("tauInverse2");
      double[] samples = new double[1000];
      for (int i = 0; i < samples.Length; i++)
            {
              samples[i] = Math.Pow(Rand.Double()*30, -2);
            }
      Vector probs;
      Gamma[] components;
      GetMixtureOfGammas(samples, 100, out probs, out components);
      Console.WriteLine("weights = {0}", probs);
      Console.WriteLine("components = ");
      Console.WriteLine(StringUtil.ToString(components));
      Range component = new Range(components.Length);
      VariableArray<Gamma> gammas = Variable.Array<Gamma>(component);
      gammas.ObservedValue = components;
      Variable<int> c = Variable.Discrete(component,probs).Named("c");
      using(Variable.Switch(c)) {
        tauInverse2.SetTo(Variable.Random<double,Gamma>(gammas[c]));
        theta[item] = Variable.GaussianFromMeanAndPrecision(mu, tauInverse2).ForEach(item);
        x[item] = Variable.GaussianFromMeanAndPrecision(theta[item], prec[item]);
      }
#endif

            // Educational testing data from Rubin (1981)
            double[] xobs = new double[] {28, 8, -3, 7, -1, 1, 18, 12};
            double[] std = new double[] {15, 10, 16, 11, 9, 11, 10, 18};
            double[] precObs = new double[std.Length];
            for (int i = 0; i < std.Length; i++)
            {
                precObs[i] = 1.0/(std[i]*std[i]);
            }

            nItem.ObservedValue = xobs.Length;
            prec.ObservedValue = precObs;
            x.ObservedValue = xobs;

            // Posterior percentiles from the book
            double[] percentiles = new double[] {2.5, 25, 50, 75, 97.5};
            double[][] thetaPercentiles = new double[][]
                {
                    new double[] {-2, 7, 10, 16, 31},
                    new double[] {-5, 3, 8, 12, 23},
                    new double[] {-11, 2, 7, 11, 19},
                    new double[] {-7, 4, 8, 11, 21},
                    new double[] {-9, 1, 5, 10, 18},
                    new double[] {-7, 2, 6, 10, 28},
                    new double[] {-1, 7, 10, 15, 26},
                    new double[] {-6, 3, 8, 13, 33}
                };
            double[] tauPercentiles = new double[] {1, 3, 7, 12, 20};

            // Theta posteriors from VMP are close but not exact.
            // Tau posterior from VMP is quite far from truth.
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            if (false)
            {
                ie.Algorithm = new GibbsSampling();
                ie.NumberOfIterations = 10000;
                ie.ShowProgress = false;
            }
            Gamma tauInverse2Actual = ie.Infer<Gamma>(tauInverse2);
            int nSamples = 10000;
            IEnumerable<double> tauInverse2Samples = GetSamples(tauInverse2Actual, nSamples);
            IEnumerable<double> tauSamples = tauInverse2Samples.Select(s => 1.0/ System.Math.Sqrt(s));
            Console.WriteLine("tau: {0}^(-0.5)", tauInverse2Actual);
            Summary(tauSamples, percentiles, tauPercentiles);
            Gaussian muActual = ie.Infer<Gaussian>(mu);
            Console.WriteLine("mu: {0}", muActual);
            Summary(GetSamples(muActual, nSamples), percentiles, new double[0]);

            Gaussian[] thetas = ie.Infer<Gaussian[]>(theta);
            IList<double[]> thetaSamples = (ie.Algorithm is GibbsSampling)
                                               ? (IList<double[]>) ie.Infer(theta, QueryTypes.Samples)
                                               : null;
            double maxDiff = 0.0;
            for (int i = 0; i < thetas.Length; i++)
            {
                Console.WriteLine("theta{0}: {1}", i, thetas[i]);
                IEnumerable<double> thetaiSamples = (ie.Algorithm is GibbsSampling) ? thetaSamples[i] : GetSamples(thetas[i], nSamples);
                double diff = Summary(thetaiSamples, percentiles, thetaPercentiles[i]);
                maxDiff = System.Math.Max(maxDiff, diff);
            }
            Console.WriteLine("max error = {0}", maxDiff.ToString("g4"));
        }

        private static IEnumerable<T> GetSamples<T>(Sampleable<T> s, int howMany)
        {
            for (int i = 0; i < howMany; i++) yield return s.Sample();
        }

        /// <summary>
        /// Get the values below which certain percentages of the observations fall.
        /// </summary>
        /// <param name="data">Must be sorted.</param>
        /// <param name="percentiles">Numbers between 0 and 100.</param>
        /// <returns>The given percentiles of data.</returns>
        private static IEnumerable<double> GetPercentiles(IList<double> data, IEnumerable<double> percentiles)
        {
            int n = data.Count;
            foreach (double percentile in percentiles)
            {
                double pos = percentile/100*(n - 1) + 1;
                int lower = (int)System.Math.Floor(pos);
                int upper = (int)System.Math.Ceiling(pos);
                double pos_frac = pos - lower;
                yield return data[lower]*(1 - pos_frac) + data[upper]*pos_frac;
            }
        }

        private static double Summary(IEnumerable<double> samples, IEnumerable<double> percentiles, IEnumerable<double> truePercentiles)
        {
            List<double> data = new List<double>(samples);
            data.Sort();
            var actual = new List<double>(GetPercentiles(data, percentiles));
            Console.Write(" percentiles: ");
            foreach (double b in actual)
            {
                Console.Write("{0} ", System.Math.Round(b));
            }
            Console.WriteLine();
            Console.Write("   should be: ");
            foreach (double b in truePercentiles)
            {
                Console.Write("{0} ", System.Math.Round(b));
            }
            Console.WriteLine();
            double maxdiff = 0.0;
            int i = 0;
            foreach (double b in truePercentiles)
            {
                maxdiff = System.Math.Max(maxdiff, MMath.AbsDiff(b, actual[i++]));
            }
            return maxdiff;
        }

        [Fact]
        public void LinearRegressionTest()
        {
            Vector[] data = new Vector[]
                {
                    Vector.FromArray(1.0, -3), Vector.FromArray(1.0, -2.1),
                    Vector.FromArray(1.0, -1.3), Vector.FromArray(1.0, 0.5),
                    Vector.FromArray(1.0, 1.2), Vector.FromArray(1.0, 3.3),
                    Vector.FromArray(1.0, 4.4), Vector.FromArray(1.0, 5.5)
                };

            Range rows = new Range(data.Length);

            VariableArray<Vector> x = Variable.Constant(data, rows).Named("x");

            Variable<Vector> w = Variable.VectorGaussianFromMeanAndPrecision(
                Vector.FromArray(new double[] {0, 0}),
                PositiveDefiniteMatrix.Identity(2)).Named("w");
            VariableArray<double> y = Variable.Array<double>(rows);
            y[rows] = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(x[rows], w), 1.0);
            y.ObservedValue = new double[] {30, 45, 40, 80, 70, 100, 130, 110};
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            VectorGaussian postW = engine.Infer<VectorGaussian>(w);
            Console.WriteLine("Posterior over the weights: " + Environment.NewLine + postW);
        }


        internal void LinearRegressionPartiallyObservedTest()
        {
            Vector[] data = new Vector[]
                {
                    Vector.FromArray(1.0, -3), Vector.FromArray(1.0, -2.1),
                    Vector.FromArray(1.0, -1.3), Vector.FromArray(1.0, 0.5),
                    Vector.FromArray(1.0, 1.2), Vector.FromArray(1.0, 3.3),
                    Vector.FromArray(1.0, 4.4), Vector.FromArray(1.0, 5.5)
                };
            Range rows = new Range(data.Length);

            VariableArray<Vector> x = Variable.Constant(data, rows).Named("x");

            Variable<Vector> w = Variable.VectorGaussianFromMeanAndPrecision(
                Vector.FromArray(new double[] {0, 0}),
                PositiveDefiniteMatrix.Identity(2)).Named("w");
            VariableArray<double> y = Variable.Array<double>(rows);
            y[rows] = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(x[rows], w), 1.0);

            Variable<Matrix> m = Variable.Observed(new Matrix(new double[,] {{0, 1}}));
            Variable<Vector> wObs = Variable.MatrixTimesVector(m, w);
            wObs.ObservedValue = Vector.FromArray(10.0);
            y.ObservedValue = new double[] {30, 45, 40, 80, 70, 100, 130, 110};
            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            VectorGaussian postW = engine.Infer<VectorGaussian>(w);
            Console.WriteLine("Posterior over the weights: " + Environment.NewLine + postW);
        }

        internal void Heights()
        {
            // The probabilistic program
            // -------------------------
            Variable<double> heightMan = Variable.GaussianFromMeanAndVariance(177, 8*8);
            Variable<double> heightWoman = Variable.GaussianFromMeanAndVariance(164, 8*8);

            // The inference
            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;
            Console.WriteLine("P(man's height) = {0}", engine.Infer(heightMan));
            Console.WriteLine("P(woman's height) = {0}", engine.Infer(heightWoman));

            Variable<bool> isTaller = (heightWoman > heightMan);
            Console.WriteLine("P(isTaller) = {0}", engine.Infer(isTaller));
            isTaller.ObservedValue = true;
            Console.WriteLine("P(man's height|isTaller) = {0}", engine.Infer(heightMan));
            Console.WriteLine("P(woman's height|isTaller) = {0}", engine.Infer(heightWoman));
        }

        [Fact]
        public void Handedness()
        {
            bool[] studentData = {false, true, true, true, true, true, true, true, false, false};
            bool[] lecturerData = {false, true, true, true, true, true, true, true, true, true};

            // The probabilistic program
            // -------------------------

            int numStudents = studentData.Length;
            Range student = new Range(numStudents);
            VariableArray<bool> isRightHanded = Variable.Array<bool>(student);
            Variable<double> probRightHanded = Variable.Beta(0.72, 0.08);
            using (Variable.ForEach(student))
            {
                isRightHanded[student] = Variable.Bernoulli(probRightHanded);
            }
            isRightHanded.ObservedValue = studentData;

            // Inference queries about the program
            // -----------------------------------
            InferenceEngine engine = new InferenceEngine();
            //Console.WriteLine("isRightHanded = {0}", engine.Infer(isRightHanded));
            var probRightHandedExpected = new Beta(7.72, 3.08);
            var probRightHandedActual = engine.Infer<Beta>(probRightHanded);
            Assert.True(probRightHandedExpected.MaxDiff(probRightHandedActual) < 1e-4);
        }

        [Fact]
        public void Handedness2()
        {
            bool[] studentData = {false, true, true, true, true, true, true, true, false, false};
            bool[] lecturerData = {false, true, true, true, true, true, true, true, true, true};

            // The probabilistic program
            // -------------------------

            int numStudents = studentData.Length;
            Range student = new Range(numStudents);
            VariableArray<bool> isRightHanded = Variable.Array<bool>(student);
            Variable<bool> drawnFromGeneral = Variable.Bernoulli(0.5);
            using (Variable.If(drawnFromGeneral))
            {
                using (Variable.ForEach(student))
                {
                    isRightHanded[student] = Variable.Bernoulli(0.9);
                }
            }
            using (Variable.IfNot(drawnFromGeneral))
            {
                Variable<double> probRightHanded = Variable.Beta(0.72, 0.08);
                using (Variable.ForEach(student))
                {
                    isRightHanded[student] = Variable.Bernoulli(probRightHanded);
                }
            }
            isRightHanded.ObservedValue = studentData;

            // Inference queries about the program
            // -----------------------------------
            InferenceEngine engine = new InferenceEngine();
            //Console.WriteLine("isRightHanded = {0}", engine.Infer(isRightHanded));
            //Console.WriteLine("probRightHanded = {0}", engine.Infer(probRightHanded));
            Console.WriteLine("drawnFromGeneral = {0}", engine.Infer(drawnFromGeneral));
            Bernoulli drawnFromGeneralExpected = new Bernoulli(0.6955);
            Bernoulli drawnFromGeneralActual = engine.Infer<Bernoulli>(drawnFromGeneral);
            Assert.True(drawnFromGeneralExpected.MaxDiff(drawnFromGeneralActual) < 1e-4);
        }

        [Fact]
        public void Handedness2ForEach()
        {
            try
            {
                bool[] studentData = {false, true, true, true, true, true, true, true, false, false};
                bool[] lecturerData = {false, true, true, true, true, true, true, true, true, true};

                // The probabilistic program
                // -------------------------

                int numStudents = studentData.Length;
                Range student = new Range(numStudents);
                VariableArray<bool> isRightHanded = Variable.Array<bool>(student);
                Variable<bool> drawnFromGeneral = Variable.Bernoulli(0.5);
                Variable<double> probRightHanded = Variable.Beta(0.72, 0.08);
                using (Variable.ForEach(student))
                {
                    using (Variable.If(drawnFromGeneral))
                    {
                        isRightHanded[student] = Variable.Bernoulli(0.9);
                    }
                    using (Variable.IfNot(drawnFromGeneral))
                    {
                        isRightHanded[student] = Variable.Bernoulli(probRightHanded);
                    }
                }
                isRightHanded.ObservedValue = studentData;

                // Inference queries about the program
                // -----------------------------------
                InferenceEngine engine = new InferenceEngine();
                //Console.WriteLine("isRightHanded = {0}", engine.Infer(isRightHanded));
                //Console.WriteLine("probRightHanded = {0}", engine.Infer(probRightHanded));
                Console.WriteLine("drawnFromGeneral = {0}", engine.Infer(drawnFromGeneral));
                Bernoulli drawnFromGeneralExpected = new Bernoulli(0.6955);
                Bernoulli drawnFromGeneralActual = engine.Infer<Bernoulli>(drawnFromGeneral);
                Assert.True(drawnFromGeneralExpected.MaxDiff(drawnFromGeneralActual) < 1e-4);
                Assert.True(false, "Did not throw exception");
            }
            catch (CompilationFailedException ex)
            {
                Console.WriteLine("Correctly threw " + ex);
            }
        }

        [Fact]
        public void Handedness3()
        {
            bool[] studentData = {false, true, true, true, true, true, true, true, false, false};
            bool[] lecturerData = {false, true, true, true, true, true, true, true, true, true};

            // The probabilistic program
            // -------------------------

            int numStudents = studentData.Length;
            Range student = new Range(numStudents);
            VariableArray<bool> isRightHanded = Variable.Array<bool>(student);
            int numLecturers = lecturerData.Length;
            Range lecturer = new Range(numLecturers);
            VariableArray<bool> isRightHandedLec = Variable.Array<bool>(lecturer);
            Variable<bool> drawnFromSame = Variable.Bernoulli(0.5);
            using (Variable.If(drawnFromSame))
            {
                Variable<double> probRightHanded = Variable.Beta(0.72, 0.08);
                using (Variable.ForEach(student))
                {
                    isRightHanded[student] = Variable.Bernoulli(probRightHanded);
                }
                using (Variable.ForEach(lecturer))
                {
                    isRightHandedLec[lecturer] = Variable.Bernoulli(probRightHanded);
                }
            }
            using (Variable.IfNot(drawnFromSame))
            {
                Variable<double> probRightHanded = Variable.Beta(0.72, 0.08);
                using (Variable.ForEach(student))
                {
                    isRightHanded[student] = Variable.Bernoulli(probRightHanded);
                }
                Variable<double> probRightHandedLec = Variable.Beta(0.72, 0.08);
                using (Variable.ForEach(lecturer))
                {
                    isRightHandedLec[lecturer] = Variable.Bernoulli(probRightHandedLec);
                }
            }
            isRightHanded.ObservedValue = studentData;
            isRightHandedLec.ObservedValue = lecturerData;

            // Inference queries about the program
            // -----------------------------------
            InferenceEngine engine = new InferenceEngine();
            //Console.WriteLine("isRightHanded = {0}", engine.Infer(isRightHanded));
            //Console.WriteLine("probRightHanded = {0}", engine.Infer(probRightHanded));
            Console.WriteLine("drawnFromSame = {0}", engine.Infer(drawnFromSame));
            Bernoulli drawnFromSameExpected = new Bernoulli(0.7355);
            Bernoulli drawnFromSameActual = engine.Infer<Bernoulli>(drawnFromSame);
            Assert.True(drawnFromSameExpected.MaxDiff(drawnFromSameActual) < 1e-4);
        }

        [Fact]
        public void BayesNetExampleTest()
        {
            BayesNetExample(false);
        }

        [Fact]
        public void BayesNetExampleFast()
        {
            BayesNetExample(true);
        }


        private void BayesNetExample(bool fast)
        {
            int[] modelData = new int[] {0, 1, 2, 1, 3, 3, 1, 1, 2, 3, 3, 0};
            int[] ptData = new int[] {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2};
            int[] engineData = new int[] {0, 1, 2, 1, 3, 3, 1, 1, 2, 3, 3, 1};
            int[] typeData = new int[] {0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0};
            int numModels = 4;
            int numTypes = 2;
            int numPowerTrains = 3;
            int numEngines = 4;

            // Learn the parameters
            CarBayesNet bayesNet = new CarBayesNet();
            bayesNet.CreateModel(numModels, numTypes, numPowerTrains, numEngines);
            bayesNet.InferParameters(modelData, typeData, ptData, engineData);

            // Print out the CPT Dirichlets
            Console.WriteLine("\nCar model PT");
            Console.WriteLine(bayesNet.PT_M_Posterior);
            Console.WriteLine("\nPower Train PT");
            Console.WriteLine(bayesNet.PT_P_Posterior);
            Console.WriteLine("\nEngine CPT");
            Console.WriteLine(Distribution<Vector>.Array(bayesNet.CPT_E_Posterior));
            Console.WriteLine("\nType CPT:");
            Console.WriteLine(Distribution<Vector>.Array(bayesNet.CPT_T_Posterior));

            Assert.Equal(5.0, bayesNet.CPT_E_Posterior[1, 1].PseudoCount[1]);
            Assert.Equal(5.0, bayesNet.CPT_T_Posterior[1].PseudoCount[1]);

            // Query the model
            int[] modelTypes = System.Linq.Enumerable.Range(0, numModels).ToArray();

            var carTypesFromModelTypes = fast
                                             ? bayesNet.QueryCarTypesFromModelTypesFast(modelTypes)
                                             : bayesNet.QueryCarTypesFromModelTypes(modelTypes);
            Console.WriteLine("\nDistributions for car types from model types");
            for (int i = 0; i < numModels; i++)
                Console.WriteLine("Probability of type, given model {0}: {1}", i, carTypesFromModelTypes[i]);
        }
    }

    public class CarBayesNet
    {
        // Model variables
        public Variable<int> NumCases = Variable.New<int>();
        public VariableArray<int> CarModel;
        public VariableArray<int> CarType;
        public VariableArray<int> PowerTrain;
        public VariableArray<int> EngineType;
        // Probability tables and conditional probability tables
        public Variable<Vector> PT_M;
        public Variable<Vector> PT_P;
        public VariableArray2D<Vector> CPT_E;
        public VariableArray<Vector> CPT_T;
        // Priors
        public Variable<Dirichlet> PT_M_Prior = null;
        public Variable<Dirichlet> PT_P_Prior = null;
        public VariableArray2D<Dirichlet> CPT_E_Prior = null;
        public VariableArray<Dirichlet> CPT_T_Prior = null;
        // Inferred posteriors
        public Dirichlet PT_M_Posterior = null;
        public Dirichlet PT_P_Posterior = null;
        public Dirichlet[,] CPT_E_Posterior = null;
        public Dirichlet[] CPT_T_Posterior = null;
        // Inference engine
        public InferenceEngine InfEngine = new InferenceEngine() {ShowProgress = false};

        public void CreateModel(int numModels, int numTypes, int numPowerTrains, int numEngines)
        {
            NumCases = Variable.New<int>().Named("NumCases"); // Set this at run time
            // The ranges and value ranges
            Range n = new Range(NumCases).Named("n");
            Range m = new Range(numModels).Named("m");
            Range t = new Range(numTypes).Named("t");
            Range p = new Range(numPowerTrains).Named("p");
            Range e = new Range(numEngines).Named("e");

            CarModel = Variable.Array<int>(n).Named("CarModel");
            PowerTrain = Variable.Array<int>(n).Named("PowerTrain");
            // Define priors - these are made to be variables so they can be set at
            // run time without recompiling the model
            PT_M_Prior = Variable.New<Dirichlet>().Named("PT_M_Prior");
            PT_P_Prior = Variable.New<Dirichlet>().Named("PT_P_Prior");
            CPT_E_Prior = Variable.Array<Dirichlet>(m, p).Named("CPT_E_Prior");
            CPT_T_Prior = Variable.Array<Dirichlet>(e).Named("CPT_T_Prior");
            // Probability tables and conditional probability tables
            PT_M = Variable<Vector>.Random(PT_M_Prior).Named("PT_M");
            PT_P = Variable<Vector>.Random(PT_P_Prior).Named("PT_P");
            CPT_E = Variable.Array<Vector>(m, p).Named("CPT_E");
            CPT_E[m, p] = Variable<Vector>.Random(CPT_E_Prior[m, p]);
            CPT_T = Variable.Array<Vector>(e).Named("CPT_T");
            CPT_T[e] = Variable<Vector>.Random(CPT_T_Prior[e]);
            // Set the value ranges for the probability tables
            PT_M.SetValueRange(m);
            PT_P.SetValueRange(p);
            CPT_E.SetValueRange(e);
            CPT_T.SetValueRange(t);

            CarModel[n] = Variable.Discrete(PT_M).ForEach(n);
            PowerTrain[n] = Variable.Discrete(PT_P).ForEach(n);
            EngineType = AddChildFromTwoParents(CarModel, PowerTrain, CPT_E).Named("EngineType");
            CarType = AddChildFromOneParent(EngineType, CPT_T).Named("CarType");
        }

        public void InferParameters(int[] modelData, int[] typeData, int[] ptData, int[] engineData)
        {
            // Set the observations:
            NumCases.ObservedValue = modelData.Length;
            CarModel.ObservedValue = modelData;
            CarType.ObservedValue = typeData;
            PowerTrain.ObservedValue = ptData;
            EngineType.ObservedValue = engineData;

            // Set the uniform priors for the probability tables
            int numModels = CarModel.GetValueRange().SizeAsInt;
            int numPowerTrains = PowerTrain.GetValueRange().SizeAsInt;
            int numEngines = EngineType.GetValueRange().SizeAsInt;
            int numTypes = CarType.GetValueRange().SizeAsInt;
            Dirichlet[,] cpt_E_Prior = new Dirichlet[numModels,numPowerTrains];
            Dirichlet[] cpt_T_Prior = new Dirichlet[numEngines];
            for (int i = 0; i < numModels; i++)
                for (int j = 0; j < numPowerTrains; j++)
                    cpt_E_Prior[i, j] = Dirichlet.Uniform(numEngines);
            for (int i = 0; i < numEngines; i++)
                cpt_T_Prior[i] = Dirichlet.Uniform(numTypes);
            PT_M_Prior.ObservedValue = Dirichlet.Uniform(numModels);
            PT_P_Prior.ObservedValue = Dirichlet.Uniform(numPowerTrains);
            CPT_E_Prior.ObservedValue = cpt_E_Prior;
            CPT_T_Prior.ObservedValue = cpt_T_Prior;
            ClearObservedCPTs();

            // Run the inference
            InfEngine.OptimiseForVariables = new List<IVariable>() {PT_M, PT_P, CPT_E, CPT_T};
            PT_M_Posterior = InfEngine.Infer<Dirichlet>(PT_M);
            PT_P_Posterior = InfEngine.Infer<Dirichlet>(PT_P);
            CPT_E_Posterior = InfEngine.Infer<Dirichlet[,]>(CPT_E);
            CPT_T_Posterior = InfEngine.Infer<Dirichlet[]>(CPT_T);
        }

        // Use the means of the posteriors to directly set
        // the CPTs for run-time queries.
        public void SetCPTsFromPosteriorMeans()
        {
            PT_M.ObservedValue = GetMean(PT_M_Posterior);
            PT_P.ObservedValue = GetMean(PT_P_Posterior);
            CPT_E.ObservedValue = GetMean(CPT_E_Posterior);
            CPT_T.ObservedValue = GetMean(CPT_T_Posterior);
        }

        public void ClearObservedCPTs()
        {
            PT_M.ClearObservedValue();
            PT_P.ClearObservedValue();
            CPT_E.ClearObservedValue();
            CPT_T.ClearObservedValue();
        }

        public Vector GetMean(Dirichlet dist)
        {
            return dist.GetMean();
        }

        public Vector[] GetMean(Dirichlet[] dists)
        {
            Vector[] result = new Vector[dists.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = dists[i].GetMean();
            }
            return result;
        }

        public Vector[,] GetMean(Dirichlet[,] dists)
        {
            int rows = dists.GetLength(0);
            int cols = dists.GetLength(1);
            Vector[,] result = new Vector[rows,cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = dists[i, j].GetMean();
                }
            }
            return result;
        }

        public Discrete[] QueryCarTypesFromModelTypes(int[] models)
        {
            NumCases.ObservedValue = models.Length;
            CarModel.ObservedValue = models;
            CarType.ClearObservedValue();
            PowerTrain.ClearObservedValue();
            EngineType.ClearObservedValue();

            PT_M_Prior.ObservedValue = PT_M_Posterior;
            PT_P_Prior.ObservedValue = PT_P_Posterior;
            CPT_E_Prior.ObservedValue = CPT_E_Posterior;
            CPT_T_Prior.ObservedValue = CPT_T_Posterior;

            // Run the inference
            InfEngine.OptimiseForVariables = new List<IVariable>() {CarType};
            return InfEngine.Infer<Discrete[]>(CarType);
        }

        public Discrete[] QueryCarTypesFromModelTypesFast(int[] models)
        {
            NumCases.ObservedValue = models.Length;
            CarModel.ObservedValue = models;
            CarType.ClearObservedValue();
            PowerTrain.ClearObservedValue();
            EngineType.ClearObservedValue();

            SetCPTsFromPosteriorMeans();
            InfEngine.OptimiseForVariables = new List<IVariable>() {CarType};

            // Run the inference
            return InfEngine.Infer<Discrete[]>(CarType);
        }

        public static VariableArray<int> AddChildFromOneParent(
            VariableArray<int> parent, VariableArray<Vector> cpt)
        {
            var d = parent.Range;
            // data range
            var child = Variable.Array<int>(d);
            using (Variable.ForEach(d))
            using (Variable.Switch(parent[d]))
                child[d] = Variable.Discrete(cpt[parent[d]]);
            return child;
        }

        public static VariableArray<int> AddChildFromTwoParents(
            VariableArray<int> parent1, VariableArray<int> parent2, VariableArray2D<Vector> cpt)
        {
            var d = parent1.Range;
            // data range
            var child = Variable.Array<int>(d);
            using (Variable.ForEach(d))
            using (Variable.Switch(parent1[d]))
            using (Variable.Switch(parent2[d]))
                child[d] = Variable.Discrete(cpt[parent1[d], parent2[d]]);
            return child;
        }
    }

    public class CarBayesNet_Broken
    {
        // Model variables
        public Variable<int> NumCases = Variable.New<int>();
        public VariableArray<int> CarModel;
        public VariableArray<int> CarType;
        public VariableArray<int> PowerTrain;
        public VariableArray<int> EngineType;
        // Probability tables and conditional probability tables
        public Variable<Vector> PT_M;
        public Variable<Vector> PT_PT;
        public VariableArray2D<Vector> CPT_E;
        public VariableArray<Vector> CPT_T;
        // Priors
        public Variable<Dirichlet> PT_M_Prior = null;
        public Variable<Dirichlet> PT_PT_Prior = null;
        public Variable<IDistribution<Vector[,]>> CPT_E_Prior = null;
        public Variable<IDistribution<Vector[]>> CPT_T_Prior = null;
        // Inferred posteriors
        public Dirichlet PT_M_Posterior = null;
        public Dirichlet PT_PT_Posterior = null;
        public IArray2D<Dirichlet> CPT_E_Posterior = null;
        public IList<Dirichlet> CPT_T_Posterior = null;
        // Inference engine
        public InferenceEngine InfEngine = new InferenceEngine() {ShowProgress = false};

        public void CreateModel(int numModels, int numTypes, int numPowerTrains, int numEngines)
        {
            NumCases = Variable.New<int>().Named("NumCases"); // Set this at run time
            // The ranges and value ranges
            Range n = new Range(NumCases).Named("n");
            Range m = new Range(numModels).Named("m");
            Range t = new Range(numTypes).Named("t");
            Range pt = new Range(numPowerTrains).Named("pt");
            Range e = new Range(numEngines).Named("e");

            CarModel = Variable.Array<int>(n).Named("CarModel");
            CarType = Variable.Array<int>(n).Named("CarType");
            PowerTrain = Variable.Array<int>(n).Named("PowerTrain");
            EngineType = Variable.Array<int>(n).Named("EngineType");
            // Define priors - these are made to be variables so they can be set at
            // run time without recompiling the model
            PT_M_Prior = Variable.New<Dirichlet>().Named("PT_M_Prior");
            PT_PT_Prior = Variable.New<Dirichlet>().Named("PT_PT_Prior");
            CPT_E_Prior = Variable.New<IDistribution<Vector[,]>>().Named("CPT_E_Prior");
            CPT_T_Prior = Variable.New<IDistribution<Vector[]>>().Named("CPT_T_Prior");
            // Probability tables and conditional probability tables
            PT_M = Variable<Vector>.Random(PT_M_Prior).Named("PT_M");
            PT_PT = Variable<Vector>.Random(PT_PT_Prior).Named("PT_PT");
            CPT_E = Variable.Array<Vector>(m, pt).Named("CPT_E");
            CPT_E.SetTo(Variable<Vector[,]>.Random(CPT_E_Prior));
            CPT_T = Variable.Array<Vector>(e).Named("CPT_T");
            CPT_T.SetTo(Variable<Vector[]>.Random(CPT_T_Prior));
            // Set the value ranges for the probability tables
            PT_M.SetValueRange(m);
            PT_PT.SetValueRange(pt);
            CPT_E.SetValueRange(e);
            CPT_T.SetValueRange(t);

            CarModel[n] = Variable.Discrete(PT_M).ForEach(n);
            PowerTrain[n] = Variable.Discrete(PT_PT).ForEach(n);
            EngineType = AddChildFromTwoParents(CarModel, PowerTrain, CPT_E);
            CarType = AddChildFromOneParent(EngineType, CPT_T);
        }

        public void InferParameters(int[] modelData, int[] typeData, int[] ptData, int[] engineData)
        {
            // Set the observations:
            NumCases.ObservedValue = modelData.Length;
            CarModel.ObservedValue = modelData;
            CarType.ObservedValue = typeData;
            PowerTrain.ObservedValue = ptData;
            EngineType.ObservedValue = engineData;

            // Set the uniform priors for the probability tables
            int numModels = CarModel.GetValueRange().SizeAsInt;
            int numPowerTrains = PowerTrain.GetValueRange().SizeAsInt;
            int numEngines = EngineType.GetValueRange().SizeAsInt;
            int numTypes = CarType.GetValueRange().SizeAsInt;
            Dirichlet[,] cpt_E_Prior = new Dirichlet[numModels,numPowerTrains];
            Dirichlet[] cpt_T_Prior = new Dirichlet[numEngines];
            for (int i = 0; i < numModels; i++)
                for (int j = 0; j < numPowerTrains; j++)
                    cpt_E_Prior[i, j] = Dirichlet.Uniform(numEngines);
            for (int i = 0; i < numEngines; i++)
                cpt_T_Prior[i] = Dirichlet.Uniform(numTypes);
            PT_M_Prior.ObservedValue = Dirichlet.Uniform(numModels);
            PT_PT_Prior.ObservedValue = Dirichlet.Uniform(numPowerTrains);
            CPT_E_Prior.ObservedValue = Distribution<Vector>.Array(cpt_E_Prior);
            CPT_T_Prior.ObservedValue = Distribution<Vector>.Array(cpt_T_Prior);

            // Run the inference
            PT_M_Posterior = InfEngine.Infer<Dirichlet>(PT_M);
            PT_PT_Posterior = InfEngine.Infer<Dirichlet>(PT_PT);
            CPT_E_Posterior = InfEngine.Infer<IArray2D<Dirichlet>>(CPT_E);
            CPT_T_Posterior = InfEngine.Infer<IList<Dirichlet>>(CPT_T);
        }

        public Discrete[] QueryCarTypesFromModelTypes(int[] models)
        {
            NumCases.ObservedValue = models.Length;
            CarModel.ObservedValue = models;
            CarType.ClearObservedValue();
            PowerTrain.ClearObservedValue();
            EngineType.ClearObservedValue();

            PT_M_Prior.ObservedValue = PT_M_Posterior;
            PT_PT_Prior.ObservedValue = PT_PT_Posterior;
            CPT_E_Prior.ObservedValue = (IDistribution<Vector[,]>) CPT_E_Posterior;
            CPT_T_Prior.ObservedValue = (IDistribution<Vector[]>) CPT_T_Posterior;

            // Run the inference
            return InfEngine.Infer<Discrete[]>(CarType);
        }

        public static VariableArray<int> AddChildFromOneParent(
            VariableArray<int> parent, VariableArray<Vector> cpt)
        {
            var d = parent.Range;
            // data range
            var child = Variable.Array<int>(d);
            using (Variable.ForEach(d))
            using (Variable.Switch(parent[d]))
                child[d] = Variable.Discrete(cpt[parent[d]]);
            return child;
        }

        public static VariableArray<int> AddChildFromTwoParents(
            VariableArray<int> parent1, VariableArray<int> parent2, VariableArray2D<Vector> cpt)
        {
            var d = parent1.Range;
            // data range
            var child = Variable.Array<int>(d);
            using (Variable.ForEach(d))
            using (Variable.Switch(parent1[d]))
            using (Variable.Switch(parent2[d]))
                child[d] = Variable.Discrete(cpt[parent1[d], parent2[d]]);
            return child;
        }
    }


    public class LearningMusic
    {
        public void Johns()
        {
            var n = new Range(2000).Named("n");
            var k = new Range(6);
            double precision = 10.0;
            var values = Variable.Array<Vector>(n).Named("values");
            var means = Variable.VectorGaussianFromMeanAndPrecision(Vector.Constant(6, 1.0/6), PositiveDefiniteMatrix.IdentityScaledBy(6, 0.1));
            var precisions = PositiveDefiniteMatrix.IdentityScaledBy(6, precision);
            var vg = Variable.Array<Vector>(n).Named("vg");
            vg[n] = Variable.VectorGaussianFromMeanAndPrecision(means, precisions).ForEach(n);

            using (Variable.ForEach(n))
            {
                var ag = Variable.ArrayFromVector(vg[n], k);
                values[n] = Variable.Softmax(ag);
            }

            var obs = Variable.Array<Dirichlet>(n).Named("obs");
            Variable.ConstrainEqualRandom(values[n], obs[n]);
            var truth = new Dirichlet(0.1, 0.1, 0.5, 0.1, 0.1, 0.1);
            obs.ObservedValue = System.Linq.Enumerable.Range(0, n.SizeAsInt).Select(_ => Dirichlet.PointMass(truth.Sample())).ToArray();

            var engine = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(engine.Infer(means));
        }

        public void Test()
        {
            var n = new Range(4).Named("n");
            var k = new Range(6);
            var values = Variable.Array<Vector>(n).Named("values");
            var means = Variable.Array<double>(k).Named("means");
            means[k] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(k);
            var precisions = Variable.Array<double>(k).Named("precisions");
            precisions.ObservedValue = System.Linq.Enumerable.Range(0, k.SizeAsInt).Select(_ => 1.0).ToArray();
            var g = Variable.Array(Variable.Array<double>(k), n).Named("vg");
            g[n][k] = Variable.GaussianFromMeanAndPrecision(means[k], precisions[k]).ForEach(n);
            values[n] = Variable.Softmax(g[n]);

            var obs = Variable.Array<Dirichlet>(n).Named("obs");
            Variable.ConstrainEqualRandom(values[n], obs[n]);
            var truthd = new double[] {0.1, 0.1, 0.5, 0.1, 0.1, 0.1};
            double T = 1000;
            var truth = new Dirichlet(truthd.Select(o => o*T).ToArray());
            obs.ObservedValue = System.Linq.Enumerable.Range(0, n.SizeAsInt).Select(_ => Dirichlet.PointMass(truth.Sample())).ToArray();

            var engine = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(engine.Infer(means));
        }
    }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif
}
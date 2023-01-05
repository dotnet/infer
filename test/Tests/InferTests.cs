// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Xunit;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;
using System.Linq;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Factors.Attributes;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
    using BernoulliArray = DistributionStructArray<Bernoulli, bool>;
    using BernoulliArrayArray = DistributionRefArray<DistributionStructArray<Bernoulli, bool>, bool[]>;
    using GaussianArray = DistributionStructArray<Gaussian, double>;
    using GaussianArray2D = DistributionStructArray2D<Gaussian, double>;
    using GaussianArrayArray = DistributionRefArray<DistributionStructArray<Gaussian, double>, double[]>;
    using DirichletArray = DistributionRefArray<Dirichlet, Vector>;
    using DiscreteArray = DistributionRefArray<Discrete, int>;
    using DiscreteArrayArray = DistributionRefArray<DistributionRefArray<Discrete, int>, int[]>;


    public class InferTests
    {
        internal void DivisionTests()
        {
            // EP division tests
            (new DocumentationTests()).EvidenceExample();  // becomes iterative, also ConstrainTrue gets wrong message
            (new GatedFactorTests()).GatedConcatTest(); // division by partial pointmass
            (new GatedFactorTests()).GatedPlusIntRRCTest(); // divison by partial pointmass
        }

        internal void AccumulationTests()
        {
            // simple examples of Accumulation
            (new EpTests()).BernoulliBetaTest();
        }

        internal void Scheduling2Tests()
        {
            // Requires particular capacities in Scheduling2Transform.GetEdgeCost
            // probably because it changes the init schedule to something that breaks UseSerialSchedules
            // why does it need an init schedule at all? because MinCut doesn't handle Any constraints
            //(new ClickTest()).ClickModelSmall();

            // Scheduling2 tests
            //Scheduling2Transform.verbose = true;
            //SchedulingTransform.debug = true;
            //Scheduling2Transform.useFakeGraph = true;
            // Schedule has redundant operations due to poor choice of toposort followed by repair - need to integrate repair with toposort
            //(new GateModelTests()).MixtureOfThreeGaussians();
            // Ignores the initializer due to Divide - initializer doesn't propagate to marginal buffer
            // with Divide=false ignores due to poor schedule (2 back edges on a cycle when 1 would suffice)
            //(new MixtureTests()).PoissonMixtureTest();
            // Fails with sequential Sum operator
            //(new VmpArrayTests()).Sum3ElementArray();
            // Fails with unrolling because MatrixMultiply assumes the array is updated in parallel
            //(new VmpArrayTests()).FactorAnalysis();
            // Fails with unrolling because Blei06SoftmaxOp.XAverageLogarithm assumes the array is updated in parallel
            //(new VmpTests()).TestLinearTimeSoftmax();
            // Fails with unrolling because FastSumOp.ArrayAverageLogarithm assumes the array is updated in parallel
            //(new NonconjugateVMP2Tests()).LogisticRegressionTest();
            // Fails due to poor EP schedule - fixed by Scheduling2
            // 3,8,12 start with 2 reqs
            //(new EpTests()).GaussianPrecisionTest();
            //(new EpTests()).PlusTest();
            //PruningTransform.PruneUniformStmts = false;
            //(new ModelTests()).FadingGridTest();
            //(new PartialUnrollTests()).GridTest2();
            // no clones required
            // loops are not merged properly
            // forward/backward loops need to be rotated
            //(new PartialUnrollTests()).EndCoupledChainsTest();
            // old: iters = 1052, length = 72
            // new: iters = 1115, length = 67
            //(new BugsTests()).BugsRatsWithoutInitialisation();
            // BugsRatsWithInitialisation();
            // requires initialisation
            // old: iters = 4730, length = 48
            // new: iters = 6205, length = 32
            //(new BugsTests()).BugsDyes();
            // lots of cycles when division=false
            //(new EpTests()).LearningAGaussianEP();
            //(new TutorialTests()).LearningAGaussian();
            // requires lots of clones
            // most cycles are active - average matchCount = 12
            //(new ModelTests()).ClickChainTest2();
            // requires lots of clones
            // but only 94 cycles in a cover
            // most cycles are active - average matchCount = 14
            //(new ClickTest()).ClickModelSmall();
            // old: iters = 6, length = 33
            // new: iters = 3, length = 20
            //(new VmpArrayTests()).FactorAnalysis();
            // old: iters = 3, length = 34
            // new: iters = 2, length = 28
            // greedy: iters = 3, length = 20
            //(new VmpArrayTests()).BayesianPCA();
            // requires initialisation
            //(new VmpArrayTests()).BayesianPartialVectorPCA();
            // requires initialisation
            // 4 invalidates its own source
            //(new VmpArrayTests()).SumWhereTest();
            // requires initialisation
            //(new GateModelTests()).GatedSubarrayObservedTest();
            // These 3 tests are good examples of how to incorporate user-provided initialization into the schedule
            //(new VmpArrayTests()).BayesianVectorPCA();
            //(new VmpArrayTests()).BayesianPartialVectorPCA();
            //(new NonconjugateVMP2Tests()).MixtureOfMultivariateGaussiansWithHyperLearning();

            // r>2 schedules ///////////////////

            //(new GateModelTests()).GateModelMixtureOfTwoGaussians();
            // 13704 total cycles, 12233 cycles in a cover
            // average matchCount = 1.03, indicating most cycles are inactive
            //(new ModelTests()).JaviOliver();
            // 294 cycles in 8 groups
            // 121 -> 164 nodes
            // old: iters = 1172, length = 121
            // new, cycle cover: iters = 727, length = 160
            // new, all cycles: iters = 322, length = 168
            (new GibbsTests()).GibbsGrid3();

            // Perfect schedules  /////////////////////

            // Does not have triggers, no clones required
            //(new DiscreteTests()).BallCountingTest();
            // Does not have triggers, must clone, must respect fresh
            // 64 cycles in 64 incompatible groups
            // 1 back edge per cycle is not efficient
            // 24 -> 49 nodes
            //(new EpTests()).BernoulliBetaTest2();
            // Works even when triggers are ignored - must clone
            // 2 cycles in 2 groups
            //(new MslTests()).JaggedArrays2Test();
            // old: iters = 18, length = 16
            // new: iters = 17, length = 16
            //(new VmpTests()).ProductObservedTwiceTest();
            // Slow due to a large number of cycles and many clones required, when divide=false
            // If divide=true, same schedule as old - note that edge labels are consistent with either serial or parallel schedule
            // schedule is serial because of the way toposort breaks ties
            //(new BayesPointMachineTests()).BayesPointNoisyStep();
            // If divide=false, best schedule is 0,1,2,1 (1 back edge per cycle)
            //(new BayesPointMachineTests()).BayesPoint();

            // r=2 schedules //////////////////////////

            // Fails if you put shortest cycles first (too many back edges)
            //(new BlogTests()).BinaryPairwisePotentials3();
            // Fails with unrolling and mincut due to conservative counting of required edges
            // also BugsPumpModified, VocabularyTest, PoissonMixtureTest
            //(new InferTests()).VectorJaggedArrayTest();
            // Requires starting with right node
            //(new VmpTests()).VmpTruncatedGaussianTest2();
            //(new BayesPointMachineTests()).FactorizedRegression();
            // old: iters = 115, length = 44
            // new: iters = 132, length = 45
            //(new GibbsTests()).GibbsGrid2();
            // Must respect triggers and must clone
            // 2 cycles in 2 groups
            //(new GibbsTests()).GibbsScaledGaussian();
            // Must respect triggers and must clone
            // 2 cycles in 2 groups
            // old: iters = 7595, length = 19
            // new, all cycles: iters = 9407, length = 19
            // new, cycle cover: iters = 27493, length = 18
            // greedy: iters = 20673, length = 9
            //(new GibbsTests()).GibbsSimpleGaussian();
            // Works even when triggers are ignored - must clone
            // 8 cycles in 4 groups
            // solution for group 2 is: (15,10)(12,11)(0,19)(5,4)(8,3)
            // 20 -> 26 nodes
            // old: iters = 66, length = 34
            // new: iters = 69, length = 41
            //(new VmpTests()).MultipleProductTest();
            // old: iters = 35, length = 20
            // new: iters = 53, length = 24
            //(new VmpTests()).MultipleProduct2Test();

            //var t = new VmpArrayTests();
            //Action action = () => t.BayesianPCA();
            //Console.WriteLine("iters = {0}, length = {1}", TestUtils.FindMinimumNumberOfIterations(action), SchedulingTransform.LastScheduleLength);

            // KeepFresh tests
            //DependencyAnalysisTransform.KeepFresh = false;
            //(new GateModelTests()).VocabularyTest();
            //(new GateModelTests()).GateEnterOnePointMassTest2();
            //(new GateModelTests()).GateEnterOnePointMassTest3();
            //(new BlogTests()).BinaryPairwisePotentials3();
            //(new GateModelTests()).GibbsSwitchInSwitchTest();
            //(new GateModelTests()).GateEnterOnePointMassTest();
            //(new GateModelTests()).GatedSubarrayObservedTest();
            //(new GibbsTests()).GibbsGrid2();
            // Fails with KeepFresh=true - fixed by Scheduling2
            //(new ModelTests()).CoinRunLengths();
        }

        [Fact]
        public void DeterministicPowerTest()
        {
            foreach (var useMarginalPrototype in new[] { false, true })
            {
                var a = Variable.Observed(3.0);
                if (useMarginalPrototype)
                    a.AddAttribute(new MarginalPrototype(Gaussian.Uniform()));
                var b = Variable.Observed(2.0);
                var c = a ^ b;
                var engine = new InferenceEngine();
                var res = engine.Infer<Gaussian>(c);
                Assert.Equal(9, res.Point);
            }
        }

        [Fact]
        public void DeterministicMaxTest()
        {
            foreach (var useMarginalPrototype in new[] { false, true })
            {
                var a = Variable.Observed(3.0);
                if (useMarginalPrototype)
                    a.AddAttribute(new MarginalPrototype(Gaussian.Uniform()));
                var b = Variable.Observed(2.0);
                var c = Variable.Max(a, b);
                var engine = new InferenceEngine();
                var res = engine.Infer<Gaussian>(c);
                Assert.Equal(3, res.Point);
            }
        }

        [Fact]
        public void TraceAllMessagesTest()
        {
            Variable<double> x = Variable.GaussianFromMeanAndPrecision(0, 1);
            Variable.ConstrainPositive(x);
            Variable.ConstrainPositive(x);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.TraceAllMessages = true;
            engine.Infer(x);
        }

        [Fact]
        public void MarginalWrongDistributionError()
        {
            Assert.Throws<ArgumentException>(() =>
            {
                Range outer = new Range(2);
                Range inner = new Range(2);
                var array = Variable.Array(Variable.Array<double>(inner), outer).Named("array");
                array[outer][inner] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(outer, inner);

                InferenceEngine engine = new InferenceEngine();
                var ca = engine.GetCompiledInferenceAlgorithm(array);
                ca.Execute(1);
                var marginal = ca.Marginal<Gamma[][]>(array.Name);

            });
        }

        /// <summary>
        /// Failed due to having 2 iteration loops, and restarting the second loop whenever the first one runs.  Only happens when the loops are in different subroutines.
        /// Changed_numberOfIterationsDecreased_vdouble__0_vint0_iterationsDone is cleared by Changed_numberOfIterationsDecreased_vint0 which is run by Changed_numberOfIterationsDecreased
        /// Caused by sub.successors in IterativeProcessTransform
        /// Statements causing the dependency:
        ///         state_marginal_F[sessionOfPlayer] = DerivedVariableOp.MarginalAverageConditional<Gaussian>(state_use_B[sessionOfPlayer], state_sessionOfPlayer_0__uses_F[sessionOfPlayer][0], state_marginal_F[sessionOfPlayer]);
        ///         InferNet.Infer(state_marginal_F, "state", QueryTypes.Marginal)
        /// but its not in DependencyInformation
        /// edge (54,60) is added by ForEachWriteAfterRead, since it uses simplistic logic
        /// also the Infer statement is considered to be inside the second subroutine, because subroutines are defined by their parameter dependencies, not their statement dependencies
        /// Fixed by removing subroutine invalidations
        /// </summary>
        [Fact]
        public void SplitArrayTest()
        {
            var sessionCountOfPlayer = Variable.Observed(default(int));
            Range sessionOfPlayer = new Range(sessionCountOfPlayer).Named("sessionOfPlayer");
            var hoursPlayed = Variable.Observed(default(double[]), sessionOfPlayer);

            sessionCountOfPlayer.ObservedValue = 1;
            hoursPlayed.ObservedValue = new double[1];

            var initialStateMean = Variable.GaussianFromMeanAndPrecision(0, 1e-4).Named("initialStateMean");
            initialStateMean.AddAttribute(new PointEstimate());
            var initialStatePrecision = Variable.GammaFromShapeAndRate(1, 1).Named("initialStatePrecision");
            initialStatePrecision.AddAttribute(new PointEstimate());
            var stateNoisePrecision = Variable.GammaFromShapeAndRate(1, 1).Named("stateNoisePrecision");
            stateNoisePrecision.AddAttribute(new PointEstimate());

            var state = Variable.Array<double>(sessionOfPlayer).Named("state");
            using (var sessionBlock = Variable.ForEach(sessionOfPlayer))
            {
                var session = sessionBlock.Index;
                using (Variable.If(session == 0))
                {
                    state[sessionOfPlayer] = Variable.GaussianFromMeanAndPrecision(initialStateMean, initialStatePrecision);
                }
                using (Variable.If(session > 0))
                {
                    state[sessionOfPlayer] = Variable.GaussianFromMeanAndPrecision(0.0, stateNoisePrecision);
                    hoursPlayed[sessionOfPlayer] = Variable.Copy(state[sessionOfPlayer]);
                }
            }

            InferenceEngine engine = new InferenceEngine();
            int previousIteration = 0;
            engine.ProgressChanged += delegate (InferenceEngine sender, InferenceProgressEventArgs args)
            {
                if (args.Iteration < previousIteration)
                    throw new Exception();
                previousIteration = args.Iteration;
            };
            engine.Compiler.ReturnCopies = false;
            engine.Compiler.FreeMemory = false;
            engine.ShowProgress = false;
            for (int iter = 1; iter <= 10; iter++)
            {
                engine.NumberOfIterations = iter;
                var mean1Actual = engine.Infer(initialStateMean);
                var stateNoisePrecisionActual = engine.Infer<Gamma>(stateNoisePrecision).Point;
                //Console.WriteLine($"{iter} stateNoisePrecision = {stateNoisePrecisionActual}");
                var stateActual = engine.Infer<IList<Gaussian>>(state);
                //Console.WriteLine(stateActual);
            }
        }

        /// <summary>
        /// Tests that PointMassAnalysisTransform does not mark x as a ForwardPointMass
        /// </summary>
        [Fact]
        public void PointMassAnalysisTest()
        {
            Gaussian xPriorF = Gaussian.FromMeanAndVariance(2, 3);
            var b = Variable.Observed(false).Named("b");
            var x = Variable.New<double>().Named("x");
            using (Variable.If(b))
            {
                x.SetTo(1.0);
            }
            using (Variable.IfNot(b))
            {
                x.SetTo(Variable.Random(xPriorF));
            }
            Variable.ConstrainPositive(x);
            Variable.ConstrainPositive(x);

            InferenceEngine engine = new InferenceEngine();
            Gaussian x_uses_B0 = Gaussian.Uniform();
            Gaussian x_uses_B1 = Gaussian.Uniform();
            for (int iteration = 0; iteration < 100; iteration++)
            {
                x_uses_B0 = IsPositiveOp.XAverageConditional(true, xPriorF * x_uses_B1);
                x_uses_B1 = IsPositiveOp.XAverageConditional(true, xPriorF * x_uses_B0);
            }
            var xExpectedF = xPriorF * x_uses_B0 * x_uses_B1;
            var xExpectedT = Gaussian.PointMass(1.0);
            for (int trial = 0; trial < 2; trial++)
            {
                b.ObservedValue = (trial == 0);
                var xExpected = b.ObservedValue ? xExpectedT : xExpectedF;
                var xActual = engine.Infer(x);
                Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
                Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
            }
        }

        // Test that Discrete.Uniform is correctly understood as uniform when scheduling
        [Fact]
        public void DiscreteUniformTest()
        {
            CDARE.Test();
        }

        // Previously failed due to a bad init schedule.
        // Fixed by adding NoInit attribute to GaussianOp.SampleAverageConditional
        // Also GaussianProductVmpOp wasn't handling uniform arguments correctly.
        [Fact]
        public void RazinTest()
        {
            Func<Variable<double>, Variable<double>> testFunction = delegate (Variable<double> x)
             {
                 return (0.25 * x + 0.5 * x * x * x);
             };

            Range dataRange = new Range(2);
            Range featureRange = new Range(2);
            VariableArray<VariableArray<double>, double[][]> inputArr =
                Variable.Array(Variable.Array<double>(featureRange), dataRange);
            VariableArray<double> outputArray = Variable.Array<double>(dataRange).Named("outputArray");
            VariableArray<double> weightVar = Variable.Array<double>(featureRange).Named("weights");
            VariableArray<bool> selector = Variable.Array<bool>(featureRange).Named("selector");
            using (Variable.ForEach(featureRange))
                selector[featureRange] = Variable.Bernoulli(0.1);

            using (Variable.ForEach(featureRange))
            {
                using (Variable.If(selector[featureRange]))
                {
                    weightVar[featureRange].SetTo(
                        Variable.GaussianFromMeanAndPrecision(Variable.Random(Gaussian.FromMeanAndPrecision(1, 0.1)),
                        Variable.Random(Gamma.FromShapeAndScale(1, 99999)))
                        );
                }
                using (Variable.IfNot(selector[featureRange]))
                {
                    weightVar[featureRange].SetTo(Variable.GaussianFromMeanAndVariance(0, 0.0000001).Named("spike"));
                }
            }
            Variable<double> precision = Variable<double>.GammaFromShapeAndScale(1, 99999).Named("precision");
            using (Variable.ForEach(dataRange))
            {
                VariableArray<double> weightSumArray = Variable.Array<double>(featureRange).Named("weightSum");
                weightSumArray[featureRange] = (weightVar[featureRange] * inputArr[dataRange][featureRange]).Named("weightSumArray");
                // the effect from other nodes - in case of no perturbation
                Variable<double> linearEffect =
                    Variable.Sum(weightSumArray).Named("sum_weightSumArray");
                Variable<double> nonLinearEffect = testFunction(linearEffect).Named("nonLinearEffect");
                outputArray[dataRange] = Variable.GaussianFromMeanAndPrecision(nonLinearEffect, precision);
            }
            inputArr.ObservedValue = Util.ArrayInit(dataRange.SizeAsInt, i => Util.ArrayInit(featureRange.SizeAsInt, j => 1.0));
            outputArray.ObservedValue = Util.ArrayInit(dataRange.SizeAsInt, i => 1.0);
            InferenceEngine ie = new InferenceEngine();
            ie.Compiler.GivePriorityTo(typeof(GaussianProductOp_SHG09));
            var weightPosterior = ie.Infer(weightVar);
            Console.WriteLine(weightPosterior);
        }

        // Test that a good schedule is found for a sequential loop
        // the crucial reversal is (7,13)
        internal void DotaTest()
        {
            int playerCount = 2;
            int heroCount = 2;
            int matchCount = 2;
            int playersOnTeamCount = 2;
            double matchOutcomePrecision = 1;
            int[][] team0PlayersData = Util.ArrayInit(matchCount, i => Util.ArrayInit(playersOnTeamCount, j => 0));
            int[][] team1PlayersData = Util.ArrayInit(matchCount, i => Util.ArrayInit(playersOnTeamCount, j => 0));
            int[][] team0HeroesData = Util.ArrayInit(matchCount, i => Util.ArrayInit(playersOnTeamCount, j => 0));
            int[][] team1HeroesData = Util.ArrayInit(matchCount, i => Util.ArrayInit(playersOnTeamCount, j => 0));

            var playerTraitCount = Variable.Observed(1).Named("playerTraitCount");
            var performerTraitCount = Variable.Observed(1).Named("performerTraitCount");

            // define ranges
            var matchRange = new Range(matchCount).Named("matchRange");
            matchRange.AddAttribute(new Sequential());
            var playerRange = new Range(playerCount).Named("playerRange");
            var playerOnTeamRange = new Range(playersOnTeamCount).Named("playerOnTeamRange");
            var heroRange = new Range(heroCount).Named("heroRange");
            var playerTraitRange = new Range(playerTraitCount).Named("playerTraitRange");
            var performerTraitRange = new Range(performerTraitCount).Named("performerTraitRange");

            // player traits with priors
            var playerTraitsPriors = Variable.Array<Gaussian>(Variable.Array<Gaussian>(playerTraitRange), playerRange).Named("playerTraitsPriors");
            playerTraitsPriors.ObservedValue = Util.ArrayInit(playerCount, i => Util.ArrayInit(playerTraitCount.ObservedValue, j => new Gaussian(0, 1)));
            var playerTraits = Variable.Array<double>(Variable.Array<double>(playerTraitRange), playerRange).Named("playerTraits");
            playerTraits[playerRange][playerTraitRange] = Variable.Random<double, Gaussian>(playerTraitsPriors[playerRange][playerTraitRange]);

            // the outcome of matches
            var team0Won = Variable.Array<bool>(matchRange).Named("team0Won");
            team0Won.ObservedValue = Util.ArrayInit(matchCount, i => true);

            // observe players and heroes in each match
            VariableArray<VariableArray<int>, int[][]> team0Players = Variable.Observed(team0PlayersData, matchRange, playerOnTeamRange).Named("team0Players");
            VariableArray<VariableArray<int>, int[][]> team1Players = Variable.Observed(team1PlayersData, matchRange, playerOnTeamRange).Named("team1Players");

            // extract player traits for matches (Variable.Subarray needs to be done outside the matchRange ForEach loop)
            VariableArray<VariableArray<VariableArray<double>, double[][]>, double[][][]> team0PlayerTraits = Variable.Array(Variable.Array(Variable.Array<double>(playerTraitRange), playerOnTeamRange), matchRange).Named("team0PlayerTraits");
            VariableArray<VariableArray<VariableArray<double>, double[][]>, double[][][]> team1PlayerTraits = Variable.Array(Variable.Array(Variable.Array<double>(playerTraitRange), playerOnTeamRange), matchRange).Named("team1PlayerTraits");
            team0PlayerTraits[matchRange] = Variable.Subarray(playerTraits, team0Players[matchRange]);
            team1PlayerTraits[matchRange] = Variable.Subarray(playerTraits, team1Players[matchRange]);
            //team0PlayerTraits.AddAttribute(new DivideMessages(false));
            //team1PlayerTraits.AddAttribute(new DivideMessages(false));

            // compute performer traits by taking the product of playerTraits with H matrices
            VariableArray<VariableArray<VariableArray<double>, double[][]>, double[][][]> team0PerformerTraits = Variable.Array(Variable.Array(Variable.Array<double>(performerTraitRange), playerOnTeamRange), matchRange).Named("team0PerformerTraits");
            VariableArray<VariableArray<VariableArray<double>, double[][]>, double[][][]> team1PerformerTraits = Variable.Array(Variable.Array(Variable.Array<double>(performerTraitRange), playerOnTeamRange), matchRange).Named("team1PerformerTraits");
            var matrixMultiply0 = Variable.Array(Variable.Array(Variable.Array(Variable.Array<double>(playerTraitRange), performerTraitRange), playerOnTeamRange), matchRange).Named("matrixProductRunningSum0");
            var matrixMultiply1 = Variable.Array(Variable.Array(Variable.Array(Variable.Array<double>(playerTraitRange), performerTraitRange), playerOnTeamRange), matchRange).Named("matrixProductRunningSum1");
            VariableArray<VariableArray<double>, double[][]> team0Sum = Variable.Array(Variable.Array<double>(playerOnTeamRange), matchRange).Named("team0Sum");
            VariableArray<VariableArray<double>, double[][]> team1Sum = Variable.Array(Variable.Array<double>(playerOnTeamRange), matchRange).Named("team1Sum");
            using (Variable.ForEach(matchRange))
            {
                // this causes the bad schedule
                bool breakCode = true;
                if (breakCode)
                {
                    using (Variable.ForEach(playerOnTeamRange))
                    {
                        using (Variable.ForEach(performerTraitRange))
                        {
                            matrixMultiply0[matchRange][playerOnTeamRange][performerTraitRange][playerTraitRange].SetTo(team0PlayerTraits[matchRange][playerOnTeamRange][playerTraitRange]);
                            matrixMultiply1[matchRange][playerOnTeamRange][performerTraitRange][playerTraitRange].SetTo(team1PlayerTraits[matchRange][playerOnTeamRange][playerTraitRange]);
                        }
                    }
                }

                // compute team trait vectors by summing
                team0Sum[matchRange][playerOnTeamRange] = Variable.Sum(team0PlayerTraits[matchRange][playerOnTeamRange]);
                team1Sum[matchRange][playerOnTeamRange] = Variable.Sum(team1PlayerTraits[matchRange][playerOnTeamRange]);

                // add Gaussian noise and determine winner
                team0Won[matchRange] = Variable.GaussianFromMeanAndPrecision(Variable.Sum(team0Sum[matchRange]) - Variable.Sum(team1Sum[matchRange]), matchOutcomePrecision) > 0;
            }
            InferenceEngine engine = new InferenceEngine();
            engine.ShowSchedule = true;
            Console.WriteLine(engine.Infer(team0PlayerTraits));
        }

        /// <summary>
        /// Tests that no warning is given if a sequential loop doesn't require iteration.
        /// </summary>
        [Fact]
        public void SequentialNoWarningTest()
        {
            Range game = new Range(2).Named("game");
            game.AddAttribute(new Sequential());
            Range player = new Range(2).Named("player");
            var playerOfGame = Variable.Array<int>(game).Named("playerOfGame");
            playerOfGame.SetValueRange(player);
            playerOfGame.ObservedValue = new int[] { 0, 1 };
            var skill = Variable.Array<double>(player).Named("skill");
            skill[player] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(player);
            var outcome = Variable.Array<bool>(game).Named("outcome");
            using (Variable.ForEach(game))
            {
                var perf = Variable.GaussianFromMeanAndVariance(skill[playerOfGame[game]], 1);
                outcome[game] = (perf > 0);
            }

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.Compiled += delegate (ModelCompiler sender, ModelCompiler.CompileEventArgs e)
            {
                Assert.True(e.Warnings.Count == 0);
            };
            var outcomeActual = engine.Infer<IList<Bernoulli>>(outcome);
            Assert.True(outcomeActual[0].IsUniform());
        }

        [Fact]
        public void ConstrainBetweenTest()
        {
            Range item = new Range(2);
            var tPrior = Variable.Array<Gaussian>(item).Named("tPrior");
            tPrior.ObservedValue = new Gaussian[] { Gaussian.FromMeanAndVariance(-1, 1), Gaussian.FromMeanAndVariance(1, 1) };
            var t = Variable.Array<double>(item).Named("t");
            t[item] = Variable<double>.Random(tPrior[item]);
            var x = Variable.GaussianFromMeanAndPrecision(0, 1).Named("x");
            var index = Variable.Observed(0);
            index.SetValueRange(item);
            var indexPlus1 = (index + 1).Named("indexPlus1");
            Variable.ConstrainBetween(x, t[index], t[indexPlus1]);

            var engine = new InferenceEngine();
            var tActual = engine.Infer<IList<Gaussian>>(t);
            Gaussian[] tExpected = new Gaussian[2];
            Gaussian xPrior = Gaussian.FromMeanAndVariance(0, 1);
            tExpected[0] = tPrior.ObservedValue[0] *
                           IsBetweenGaussianOp.LowerBoundAverageConditional_Slow(Bernoulli.PointMass(true), xPrior, tPrior.ObservedValue[0], tPrior.ObservedValue[1]);
            tExpected[1] = tPrior.ObservedValue[1] *
                           IsBetweenGaussianOp.UpperBoundAverageConditional_Slow(Bernoulli.PointMass(true), xPrior, tPrior.ObservedValue[0], tPrior.ObservedValue[1]);
            for (int i = 0; i < tExpected.Length; i++)
            {
                Console.WriteLine("t[{0}] = {1} (should be {2})", i, tActual[i], tExpected[i]);
                Assert.True(tExpected[i].MaxDiff(tActual[i]) < 1e-10);
            }
        }

        [Fact]
        public void ConstrainBetweenTest2()
        {
            int n = 3;
            Range item = new Range(n);
            var tPrior = Variable.Array<Gaussian>(item).Named("tPrior");
            tPrior.ObservedValue = new Gaussian[] { Gaussian.FromMeanAndVariance(-1, 1), Gaussian.FromMeanAndVariance(0, 1), Gaussian.FromMeanAndVariance(1, 1) };
            var t = Variable.Array<double>(item).Named("t");
            t[item] = Variable<double>.Random(tPrior[item]);
            var x = Variable.GaussianFromMeanAndPrecision(0, 1).Named("x");
            var index = Variable.Random(Discrete.Uniform(n - 1));
            Range item1 = new Range(2);
            index.SetValueRange(item1);
            using (Variable.Switch(index))
            {
                var indexPlus0 = (index + 0).Named("indexPlus0");
                var indexPlus1 = (index + 1).Named("indexPlus1");
                Variable.ConstrainBetween(x, t[indexPlus0], t[indexPlus1]);
            }

            var engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(index));
            GaussianArray tExpected;
            ConstrainBetweenExact2(out tExpected);
            var tActual = engine.Infer<IList<Gaussian>>(t);
            Console.WriteLine(StringUtil.JoinColumns(tActual, " should be ", tExpected));
            Assert.True(tExpected.MaxDiff(tActual) < 1e-10);
        }

        private void ConstrainBetweenExact2(out GaussianArray tPost)
        {
            int n = 3;
            Variable<double>[] t = new Variable<double>[n];
            for (int i = 0; i < n; i++)
            {
                t[i] = Variable.GaussianFromMeanAndVariance(i - 1, 1);
            }
            var x = Variable.GaussianFromMeanAndPrecision(0, 1).Named("x");
            Variable<bool> index = Variable.Bernoulli(0.5);
            using (Variable.If(index))
            {
                Variable.ConstrainBetween(x, t[0], t[1]);
            }
            using (Variable.IfNot(index))
            {
                Variable.ConstrainBetween(x, t[1], t[2]);
            }
            var engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(index));
            tPost = new GaussianArray(n);
            for (int i = 0; i < n; i++)
            {
                tPost[i] = engine.Infer<Gaussian>(t[i]);
            }
        }

        // Test that multiple uses of an array within a loop are allocated different messages
        [Fact]
        public void ConstrainBetweenTest3()
        {
            int n = 3;
            Range item = new Range(n).Named("item");
            var tPrior = Variable.Array<Gaussian>(item).Named("tPrior");
            tPrior.ObservedValue = new Gaussian[] { Gaussian.FromMeanAndVariance(-1, 1), Gaussian.FromMeanAndVariance(0, 1), Gaussian.FromMeanAndVariance(1, 1) };
            var t = Variable.Array<double>(item).Named("t");
            t[item] = Variable<double>.Random(tPrior[item]);
            Range item1 = new Range(2).Named("item1");
            var x = Variable.Array<double>(item1).Named("x");
            x[item1] = Variable.GaussianFromMeanAndPrecision(0.5, 1).ForEach(item1);
            using (var fb = Variable.ForEach(item1))
            {
                //var indexPlus0 = fb.Index;
                var indexPlus0 = (fb.Index + 0).Named("indexPlus0");
                var indexPlus1 = (fb.Index + 1).Named("indexPlus1");
                Variable.ConstrainBetween(x[fb.Index], t[indexPlus0], t[indexPlus1]);
            }

            var engine = new InferenceEngine();
            var tActual = engine.Infer<IList<Gaussian>>(t);
            GaussianArray tExpected;
            ConstrainBetweenExact3(out tExpected);
            Console.WriteLine(StringUtil.JoinColumns(tActual, " should be ", tExpected));
            Assert.True(tExpected.MaxDiff(tActual) < 1e-8);
        }

        [Fact]
        public void ConstrainBetweenTest4()
        {
            int n = 3;
            Range item = new Range(n);
            var tPrior = Variable.Array<Gaussian>(item).Named("tPrior");
            tPrior.ObservedValue = new Gaussian[] { Gaussian.FromMeanAndVariance(-1, 1), Gaussian.FromMeanAndVariance(0, 1), Gaussian.FromMeanAndVariance(1, 1) };
            var t = Variable.Array<double>(item).Named("t");
            t[item] = Variable<double>.Random(tPrior[item]);
            Range item1 = new Range(2);
            var x = Variable.Array<double>(item1).Named("x");
            x[item1] = Variable.GaussianFromMeanAndPrecision(0.5, 1).ForEach(item1);
            using (var fb = Variable.ForEach(item))
            {
                using (Variable.If(fb.Index > 0))
                {
                    Variable.ConstrainBetween(x[fb.Index - 1], t[fb.Index - 1], t[fb.Index]);
                }
            }

            var engine = new InferenceEngine();
            var tActual = engine.Infer<IList<Gaussian>>(t);
            GaussianArray tExpected;
            ConstrainBetweenExact3(out tExpected);
            Console.WriteLine(StringUtil.JoinColumns(tActual, " should be ", tExpected));
            Assert.True(tExpected.MaxDiff(tActual) < 1e-8);
        }

        private void ConstrainBetweenExact3(out GaussianArray tPost)
        {
            int n = 3;
            Variable<double>[] t = new Variable<double>[n];
            for (int i = 0; i < n; i++)
            {
                t[i] = Variable.GaussianFromMeanAndVariance(i - 1, 1);
            }
            Variable<double>[] x = new Variable<double>[n - 1];
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = Variable.GaussianFromMeanAndPrecision(0.5, 1);
            }
            Variable.ConstrainBetween(x[0], t[0], t[1]);
            Variable.ConstrainBetween(x[1], t[1], t[2]);
            var engine = new InferenceEngine();
            tPost = new GaussianArray(n);
            for (int i = 0; i < n; i++)
            {
                tPost[i] = engine.Infer<Gaussian>(t[i]);
            }
        }

        [Fact]
        public void ConstrainBetweenTest5()
        {
            int n = 3;
            Range item = new Range(n);
            var tPrior = Variable.Array<Gaussian>(item).Named("tPrior");
            tPrior.ObservedValue = new Gaussian[] { Gaussian.FromMeanAndVariance(-1, 1), Gaussian.FromMeanAndVariance(0, 1), Gaussian.FromMeanAndVariance(1, 1) };
            var t = Variable.Array<double>(item).Named("t");
            t[item] = Variable<double>.Random(tPrior[item]);
            Range item1 = new Range(2);
            var x = Variable.Array<double>(item1).Named("x");
            x[item1] = Variable.GaussianFromMeanAndPrecision(0.5, 1).ForEach(item1);
            Variable.ConstrainBetween(x[0], t[0], t[1]);
            Variable.ConstrainBetween(x[1], t[1], t[2]);

            var engine = new InferenceEngine();
            var tActual = engine.Infer<IList<Gaussian>>(t);
            GaussianArray tExpected;
            ConstrainBetweenExact3(out tExpected);
            Console.WriteLine(StringUtil.JoinColumns(tActual, " should be ", tExpected));
            Assert.True(tExpected.MaxDiff(tActual) < 1e-8);
        }

        // Check that Compiler.FreeMemory=false option correctly resets messages
        [Fact]
        public void FreeMemoryConstrainBetweenTest()
        {
            IDistribution<double[]> tExpected1, tExpected2, tActual1, tActual2;
            FreeMemoryConstrainBetweenTest_Helper(false, out tExpected1, out tExpected2);
            FreeMemoryConstrainBetweenTest_Helper(true, out tActual1, out tActual2);
            Console.WriteLine(StringUtil.JoinColumns(tActual1, " should be ", tExpected1));
            Console.WriteLine(StringUtil.JoinColumns(tActual2, " should be ", tExpected2));
            Assert.True(tExpected1.MaxDiff(tActual1) < 1e-8);
            Assert.True(tExpected2.MaxDiff(tActual2) < 1e-8);
        }

        private void FreeMemoryConstrainBetweenTest_Helper(bool flip, out IDistribution<double[]> tPost1, out IDistribution<double[]> tPost2)
        {
            int n = 3;
            Range level = new Range(n);
            var tPrior = Variable.Array<Gaussian>(level).Named("tPrior");
            tPrior.ObservedValue = new Gaussian[] { Gaussian.FromMeanAndVariance(-1, 1), Gaussian.FromMeanAndVariance(0, 1), Gaussian.FromMeanAndVariance(1, 1) };
            var t = Variable.Array<double>(level).Named("t");
            t[level] = Variable<double>.Random(tPrior[level]);
            Range item = new Range(2);
            var x = Variable.Array<double>(item).Named("x");
            x.ObservedValue = new double[] { 2, 3 };
            var rating = Variable.Array<int>(item).Named("rating");
            using (Variable.ForEach(item))
            {
                var tNoisy = Variable.Array<double>(level).Named("tNoisy");
                tNoisy[level] = Variable.GaussianFromMeanAndVariance(t[level], 1);
                var thisRating = (rating[item] + 0).Named("thisRating");
                var nextRating = (rating[item] + 1).Named("nextRating");
                Variable.ConstrainBetween(x[item], tNoisy[thisRating], tNoisy[nextRating]);
            }

            var engine = new InferenceEngine();
            engine.Compiler.FreeMemory = false;
            engine.Compiler.ReturnCopies = true;
            if (flip)
            {
                // same as below but done in a different order
                rating.ObservedValue = new int[] { 1, 0 };
                tPost2 = engine.Infer<IDistribution<double[]>>(t);
                rating.ObservedValue = new int[] { 0, 1 };
                tPost1 = engine.Infer<IDistribution<double[]>>(t);
            }
            else
            {
                rating.ObservedValue = new int[] { 0, 1 };
                tPost1 = engine.Infer<IDistribution<double[]>>(t);
                rating.ObservedValue = new int[] { 1, 0 };
                tPost2 = engine.Infer<IDistribution<double[]>>(t);
            }
        }

        // Check that Compiler.FreeMemory=false option correctly resets messages
        [Fact]
        public void FreeMemoryConstrainBetweenTest2()
        {
            IDistribution<double[]> tExpected1, tExpected2, tActual1, tActual2;
            FreeMemoryConstrainBetweenTest2_Helper(false, out tExpected1, out tExpected2);
            FreeMemoryConstrainBetweenTest2_Helper(true, out tActual1, out tActual2);
            Console.WriteLine(StringUtil.JoinColumns(tActual1, " should be ", tExpected1));
            Console.WriteLine(StringUtil.JoinColumns(tActual2, " should be ", tExpected2));
            Assert.True(tExpected1.MaxDiff(tActual1) < 1e-8);
            Assert.True(tExpected2.MaxDiff(tActual2) < 1e-8);
        }

        private void FreeMemoryConstrainBetweenTest2_Helper(bool flip, out IDistribution<double[]> tPost1, out IDistribution<double[]> tPost2)
        {
            int n = 3;
            Range level = new Range(n);
            var tPrior = Variable.Array<Gaussian>(level).Named("tPrior");
            tPrior.ObservedValue = new Gaussian[] { Gaussian.FromMeanAndVariance(-1, 1), Gaussian.FromMeanAndVariance(0, 1), Gaussian.FromMeanAndVariance(1, 1) };
            var t = Variable.Array<double>(level).Named("t");
            t[level] = Variable<double>.Random(tPrior[level]);
            var rating = Variable.Observed(default(int)).Named("rating");
            var thisRating = (rating + 0).Named("thisRating");
            Variable.ConstrainPositive(t[thisRating]);

            var engine = new InferenceEngine();
            engine.Compiler.FreeMemory = false;
            engine.Compiler.ReturnCopies = true;
            if (flip)
            {
                // same as below but done in a different order
                rating.ObservedValue = 1;
                tPost2 = engine.Infer<IDistribution<double[]>>(t);
                rating.ObservedValue = 0;
                tPost1 = engine.Infer<IDistribution<double[]>>(t);
            }
            else
            {
                rating.ObservedValue = 0;
                tPost1 = engine.Infer<IDistribution<double[]>>(t);
                rating.ObservedValue = 1;
                tPost2 = engine.Infer<IDistribution<double[]>>(t);
            }
        }

        [Fact]
        public void ArrayConstraintsTest()
        {
            int n = 3;
            Range item = new Range(n);
            var tPrior = Variable.Array<Bernoulli>(item).Named("tPrior");
            tPrior.ObservedValue = new Bernoulli[] { new Bernoulli(0.1), new Bernoulli(0.2), new Bernoulli(0.3) };
            var t = Variable.Array<bool>(item).Named("t");
            t[item] = Variable<bool>.Random(tPrior[item]);
            var index = Variable.Random(Discrete.Uniform(2));
            Range item1 = new Range(n - 1);
            index.SetValueRange(item1);
            using (Variable.Switch(index))
            {
                var indexPlus0 = (index + 0).Named("indexPlus0");
                var indexPlus1 = (index + 1).Named("indexPlus1");
                var eq = (indexPlus0 == indexPlus1);
                // this conditional allows the compiler to perform additional optimizations.
                // in order to make this automatic, the compiler would need to detect such properties and propagate them through the code.
                using (Variable.IfNot(eq))
                {
                    Variable.ConstrainEqual(t[indexPlus0], t[indexPlus1]);
                }
            }

            BernoulliArray tExpected;
            ArrayConstraintsExact(out tExpected);

            var engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(index));
            var tActual = engine.Infer<IList<Bernoulli>>(t);
            Console.WriteLine(StringUtil.JoinColumns(tActual, " should be ", tExpected));
            Assert.True(tExpected.MaxDiff(tActual) < 1e-10);
        }

        private void ArrayConstraintsExact(out BernoulliArray tPost)
        {
            int n = 3;
            Variable<bool>[] t = new Variable<bool>[n];
            for (int i = 0; i < n; i++)
            {
                t[i] = Variable.Bernoulli((i + 1) / 10.0);
            }
            Variable<bool> index = Variable.Bernoulli(0.5);
            using (Variable.If(index))
            {
                Variable.ConstrainEqual(t[0], t[1]);
            }
            using (Variable.IfNot(index))
            {
                Variable.ConstrainEqual(t[1], t[2]);
            }

            var engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(index));
            tPost = new BernoulliArray(n);
            for (int i = 0; i < n; i++)
            {
                tPost[i] = engine.Infer<Bernoulli>(t[i]);
            }
        }

        internal void TwoCoinsBiased()
        {
            Variable<double> p = Variable.Beta(1, 1).Named("p");
            Variable<bool> firstCoin = Variable.Bernoulli(p).Named("firstCoin");
            Variable<bool> secondCoin = Variable.Bernoulli(p).Named("secondCoin");
            Variable<bool> bothHeads = (firstCoin & secondCoin).Named("bothHeads");
            bothHeads.ObservedValue = false;
            InferenceEngine ie = new InferenceEngine();
            ie.ShowProgress = true;
            if (true)
            {
                ie.Algorithm = new GibbsSampling();
                ie.NumberOfIterations = 20000;
            }
            // 0.3107 or 0.309
            //Beta.AllowImproperSum = true;
            Console.WriteLine(ie.Infer(firstCoin));
            Console.WriteLine(ie.Infer(p));
            // Gibbs: Bernoulli(0.252)
            //        Beta(1.103,1.844)[mean=0.3742]
        }

        // Test scheduling model
        internal void VmpChildScheduleTest()
        {
            int n = 3;
            Gaussian xPrior = new Gaussian(0, 1);
            Gaussian[] yLike = Util.ArrayInit(n, i => new Gaussian(1, 1));
            double yPrec = 1;
            Variable<double> x = Variable.Random(xPrior);
            Range r = new Range(n);
            VariableArray<double> y = Variable.Array<double>(r);
            y[r] = Variable.GaussianFromMeanAndPrecision(x, yPrec).ForEach(r);
            VariableArray<Gaussian> yLikeVar = Variable.Observed(yLike, r);
            Variable.ConstrainEqualRandom(y[r], yLikeVar[r]);
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            //engine.ShowSchedule = true;
            Gaussian xExpected = engine.Infer<Gaussian>(x);

            int[] numIters = new int[2];
            for (int trial = 0; trial < 2; trial++)
            {
                Gaussian[] xLike = new Gaussian[n];
                Gaussian xPost = xPrior;
                bool serial = (trial == 1);
                for (int iter = 0; iter < 50; iter++)
                {
                    xPost = Distribution.SetToProductWithAll(xPrior, xLike);
                    double diff = xExpected.MaxDiff(xPost);
                    if (diff < 1e-8)
                    {
                        numIters[trial] = iter;
                        Console.WriteLine("{0} iters", iter);
                        break;
                    }
                    for (int i = 0; i < n; i++)
                    {
                        Gaussian yPost = GaussianOp.SampleAverageLogarithm(xPost, yPrec) * yLike[i];
                        xLike[i] = GaussianOp.MeanAverageLogarithm(yPost, yPrec);
                        if (serial)
                        {
                            xPost = Distribution.SetToProductWithAll(xPrior, xLike);
                        }
                    }
                }
                Console.WriteLine(xPost);
            }
            Console.WriteLine("serial/parallel: {0}", ((double)numIters[1] / numIters[0]).ToString("g2"));
            // parallel is n+1 updates
            // serial is 2*n updates
            // if prior has same precision as data: ratio is 0.78 at n=2, 0.75 at n=3, 0.5 at n=1000
        }

        internal void VmpChildScheduleTest2()
        {
            int n = 1000;
            Bernoulli xPrior = Bernoulli.FromLogOdds(-3.0 * n / 200);
            Bernoulli[] yLike = Util.ArrayInit(n, i => new Bernoulli(0.6));
            Bernoulli yCondT = Bernoulli.FromLogOdds(1 / System.Math.Sqrt(n));
            Bernoulli yCondF = Bernoulli.FromLogOdds(-yCondT.LogOdds);
            Variable<bool> x = Variable.Random(xPrior);
            Range r = new Range(n);
            VariableArray<bool> y = Variable.Array<bool>(r);
            using (Variable.If(x))
            {
                y[r].SetTo(Variable.Random(yCondT).ForEach(r));
            }
            using (Variable.IfNot(x))
            {
                y[r].SetTo(Variable.Random(yCondF).ForEach(r));
            }
            VariableArray<Bernoulli> yLikeVar = Variable.Observed(yLike, r);
            Variable.ConstrainEqualRandom(y[r], yLikeVar[r]);
            var alg = new VariationalMessagePassing();
            InferenceEngine engine = new InferenceEngine(alg);
            alg.UseGateExitRandom = true;
            Bernoulli xExpected = engine.Infer<Bernoulli>(x);
            Console.WriteLine("xExpected = {0}", xExpected);

            int[] numIters = new int[2];
            for (int trial = 0; trial < 2; trial++)
            {
                Bernoulli[] xLike = new Bernoulli[n];
                Bernoulli xPost = xPrior;
                bool serial = (trial == 1);
                for (int iter = 0; iter < 50; iter++)
                {
                    xPost = Distribution.SetToProductWithAll(xPrior, xLike);
                    double diff = xExpected.MaxDiff(xPost);
                    if (diff < 1e-8)
                    {
                        numIters[trial] = iter;
                        Console.WriteLine("{0} iters", iter);
                        break;
                    }
                    for (int i = 0; i < n; i++)
                    {
                        Bernoulli yPost = (yCondT ^ xPost.GetProbTrue()) * (yCondF ^ xPost.GetProbFalse()) * yLike[i];
                        xLike[i] = Bernoulli.FromLogOdds(yPost.GetAverageLog(yCondT) - yPost.GetAverageLog(yCondF));
                        if (serial)
                        {
                            xPost = Distribution.SetToProductWithAll(xPrior, xLike);
                        }
                    }
                }
                Console.WriteLine(xPost);
            }
            Console.WriteLine("serial/parallel: {0}", ((double)numIters[1] / numIters[0]).ToString("g2"));
            // n=2: 0.83
            // n=10: 0.67
            // n=20: 0.67
            // n=50: 0.67
            // n=100: 0.62
            // n=1000: 0.67
        }

        // This test illustrates an issue with VMP, vague priors, and evidence.
        // EP gives the exact answer.
        // However, VMP gives a badly wrong answer unless:
        // 1. You initialise mean to a point mass, or
        // 2. You move the definition of mean to inside the gate.
        // When using these workarounds, VMP becomes insensitive to the variance of the prior
        // (which can be viewed as either a good thing or bad thing)
        [Fact]
        public void VaguePriorTest()
        {
            // fails with priorVariance=1e16 and EP and DivideMessages(true)
            double priorVariance = 1e6;
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, priorVariance).Named("mean");
            Variable<int> nItems = Variable.New<int>();
            Range item = new Range(nItems);
            VariableArray<double> x = Variable.Array<double>(item).Named("x");
            Variable<bool> c = Variable.Bernoulli(0.5).Named("c");
            using (Variable.If(c))
            {
                x[item] = Variable.GaussianFromMeanAndVariance(mean, 1).ForEach(item);
            }
            using (Variable.IfNot(c))
            {
                x[item] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(item);
            }
            if (false)
            {
                Variable<Gaussian> meanInit = Variable.New<Gaussian>();
                mean.InitialiseTo(meanInit);
                meanInit.ObservedValue = Gaussian.PointMass(0);
                //meanInit.ObservedValue = Gaussian.FromMeanAndVariance(0, priorVariance);
            }

            InferenceEngine engine = new InferenceEngine();
            //engine.Algorithm = new VariationalMessagePassing();
            engine.ShowProgress = false;
            nItems.ObservedValue = 1;
            for (int i = 0; i < 20; i++)
            {
                x.ObservedValue = Util.ArrayInit(nItems.ObservedValue, j => (double)i);
                //meanInit.ObservedValue = Gaussian.PointMass(x.ObservedValue[0]/10*0);
                Console.WriteLine("x={0}: c={1}", x.ObservedValue[0], engine.Infer(c));
            }
            Assert.True(engine.Infer<Bernoulli>(c).LogOdds > 10);
        }

        // This test demonstrates the impact that parameterization can have on inference.
        // Increasing the number of items per group makes the differences more obvious.
        // Increasing the number of groups (relative to the number of items) makes the two approaches more similar.
        // Adding the appropriate constraint makes the results better but not perfect.
        // Adding the wrong constraint makes the results worse.
        // The problem is less dramatic with VMP since it is overconfident.
        internal void HierarchicalNormalExample()
        {
            Rand.Restart(0);
            int nGroups = 3;
            int nItems = 10000;
            double meanVariance = 100;
            double meanMeanVariance = 100;
            double[][] data = Util.ArrayInit(nGroups, g => Util.ArrayInit(nItems, i => Rand.Normal() - 1 + 2.0 * g / (nGroups - 1)));
            IAlgorithm algorithm = new ExpectationPropagation();
            //algorithm = new GibbsSampling();
            algorithm = new VariationalMessagePassing();
            Console.WriteLine("good way:");
            IList<Gaussian> meanPost1 = HierarchicalNormalModel(data, meanVariance, meanMeanVariance, false, false, algorithm);
            for (int i = 0; i < System.Math.Min(4, meanPost1.Count); i++)
            {
                Console.WriteLine("[{0}] {1}", i, meanPost1[i]);
            }
            Console.WriteLine("bad way:");
            IList<Gaussian> meanPost2 = HierarchicalNormalModel(data, meanVariance, meanMeanVariance, true, false, algorithm);
            for (int i = 0; i < System.Math.Min(4, meanPost2.Count); i++)
            {
                Console.WriteLine("[{0}] {1}", i, meanPost2[i]);
            }
            if (algorithm is ExpectationPropagation)
            {
                Console.WriteLine("bad way + constraint:");
                IList<Gaussian> meanPost3 = HierarchicalNormalModel(data, meanVariance, meanMeanVariance, true, true, algorithm);
                for (int i = 0; i < System.Math.Min(4, meanPost3.Count); i++)
                {
                    Console.WriteLine("[{0}] {1}", i, meanPost3[i]);
                }
            }
        }

        public IList<Gaussian> HierarchicalNormalModel(double[][] data, double meanVariance, double meanMeanVariance, bool useDeltas, bool useConstraint, IAlgorithm algorithm)
        {
            var meanMean = Variable.GaussianFromMeanAndVariance(0, meanMeanVariance).Named("meanMean");
            Range group = new Range(data.Length).Named("group");
            var sizes = Variable.Observed(Util.ArrayInit(data.Length, i => data[i].Length), group);
            Range item = new Range(sizes[group]).Named("item");
            var means = Variable.Array<double>(group).Named("means");
            var deltas = Variable.Array<double>(group).Named("deltas");
            var x = Variable.Array(Variable.Array<double>(item), group).Named("x");
            x.ObservedValue = data;
            using (Variable.ForEach(group))
            {
                if (useDeltas)
                {
                    deltas[group] = Variable.GaussianFromMeanAndVariance(0, meanVariance);
                    means[group] = meanMean + deltas[group];
                }
                else
                {
                    means[group] = Variable.GaussianFromMeanAndVariance(meanMean, meanVariance);
                }
                using (Variable.ForEach(item))
                {
                    x[group][item] = Variable.GaussianFromMeanAndVariance(means[group], 1);
                }
            }
            if (useDeltas && useConstraint)
            {
                double step = System.Math.Min(1.0, 2.0 / data.Length);
                Console.WriteLine(step);
                var meanMeanDamped = Variable<double>.Factor<double, double>(Damp.Backward, meanMean, step);
                var deltasDamped = Variable.Array<double>(group);
                deltasDamped[group] = Variable<double>.Factor<double, double>(Damp.Backward, deltas[group], step);
                Variable.ConstrainEqual(meanMeanDamped, Variable.Sum(deltasDamped));
                // wrong constraint
                //Variable.ConstrainEqual(0, Variable.Sum(deltasDamped));
            }

            InferenceEngine engine = new InferenceEngine(algorithm);
            engine.ShowProgress = false;
            if (useDeltas)
            {
                if (false)
                {
                    // monitor convergence
                    for (int iter = 0; iter < 100; iter++)
                    {
                        engine.NumberOfIterations = iter;
                        Console.WriteLine("{0}: {1}", iter, engine.Infer(meanMean));
                    }
                }
                // Compute the posterior on meanMean and deltas, and derive the predicted means from these.
                Gaussian meanMeanPost = engine.Infer<Gaussian>(meanMean);
                Console.WriteLine("meanMeanPost = {0}", meanMeanPost);
                IList<Gaussian> deltasPost = engine.Infer<IList<Gaussian>>(deltas);
                GaussianArray meansPost = new GaussianArray(data.Length, i => DoublePlusOp.SumAverageConditional(meanMeanPost, deltasPost[i]));
                return meansPost;
            }
            else
            {
                Gaussian meanMeanPost = engine.Infer<Gaussian>(meanMean);
                Console.WriteLine("meanMeanPost = {0}", meanMeanPost);
                return engine.Infer<IList<Gaussian>>(means);
            }
        }

        internal void AdditiveSymmetryExample()
        {
            int n = 2;
            int m = 2;
            var b = new Variable<double>[n];
            double priorVariance = 100;
            for (int i = 0; i < n; i++)
            {
                b[i] = Variable.GaussianFromMeanAndVariance(0, priorVariance).Named("b" + i);
            }
            var c = new Variable<double>[m];
            for (int j = 0; j < m; j++)
            {
                c[j] = Variable.GaussianFromMeanAndVariance(0, priorVariance).Named("c" + j);
            }
            double noiseVariance = 1;
            var y = new Variable<double>[n, m];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    y[i, j] = Variable.GaussianFromMeanAndVariance(b[i] + c[j], noiseVariance).Named("y" + i + j);
                    //if (i != 0) continue;
                    //if (j != 0) continue;
                    //if (i == 0 && j == 1) continue;
                    if (i == 1 && j == 1)
                        continue;
                    y[i, j].ObservedValue = 2 * i + j + 1;
                }
            }
            var yOut = Variable.GaussianFromMeanAndVariance(b[1] + c[1], noiseVariance).Named("yOut");
            //var yOut = Variable.GaussianFromMeanAndVariance(b[0]+c[1], noiseVariance).Named("yOut");

            if (true)
            {
                var engine2 = new InferenceEngine();
                engine2.Algorithm = new GibbsSampling();
                engine2.NumberOfIterations = 100000;
                engine2.ShowProgress = false;
                Console.WriteLine("exact yOut = {0}", engine2.Infer(yOut));
            }
            var engine = new InferenceEngine();
            engine.ShowProgress = false;
            Console.WriteLine("unconstrained yOut = {0}", engine.Infer(yOut));
            for (int i = 0; i < n; i++)
            {
                Console.WriteLine("b[{0}] = {1}", i, engine.Infer(b[i]));
            }
            for (int j = 0; j < m; j++)
            {
                Console.WriteLine("c[{0}] = {1}", j, engine.Infer(c[j]));
            }
            b[0].ObservedValue = 0;
            Console.WriteLine("anchored yOut = {0}", engine.Infer(yOut));
            b[0].ClearObservedValue();
            if (true)
            {
                double stepsize = 0.25;
                var bsum = b[0];
                for (int i = 1; i < n; i++)
                {
                    if (stepsize != 1)
                    {
                        var bDamped = Variable<double>.Factor<double, double>(Damp.Backward, b[i], stepsize).Named("bDamped" + i);
                        bsum += bDamped;
                    }
                    else
                    {
                        bsum += b[i];
                    }
                }
                double t = 10;
                var csum = t + c[0];
                for (int j = 1; j < m; j++)
                {
                    if (stepsize != 1)
                    {
                        var cDamped = Variable<double>.Factor<double, double>(Damp.Backward, c[j], stepsize).Named("cDamped" + j);
                        csum += cDamped;
                    }
                    else
                    {
                        csum += c[j];
                    }
                }
                Variable.ConstrainEqual(bsum, csum);
            }
            for (int iter = 1; iter < 100; iter++)
            {
                engine.NumberOfIterations = iter;
                if (false)
                    Console.WriteLine("{0} yOut = {1}", iter, engine.Infer(yOut));
            }
            Console.WriteLine("constrained yOut = {0}", engine.Infer(yOut));
        }

        internal void AdditiveSymmetryExample2()
        {
            int n = 2;
            int m = 2;
            var b = new Variable<double>[n];
            double priorVariance = 100;
            var a = Variable.GaussianFromMeanAndVariance(0, priorVariance).Named("a");
            for (int i = 0; i < n; i++)
            {
                if (i == 0)
                    b[i] = Variable.GaussianFromMeanAndVariance(0, priorVariance * 0).Named("b" + i);
                else
                    b[i] = Variable.GaussianFromMeanAndVariance(-a, priorVariance).Named("b" + i);
            }
            var c = new Variable<double>[m];
            for (int j = 0; j < m; j++)
            {
                c[j] = Variable.GaussianFromMeanAndVariance(a, priorVariance).Named("c" + j);
            }
            double noiseVariance = 1;
            var y = new Variable<double>[n, m];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    y[i, j] = Variable.GaussianFromMeanAndVariance(b[i] + c[j], noiseVariance).Named("y" + i + j);
                    if (i != 0)
                        continue;
                    //if (j != 0) continue;
                    if (i == 0 && j == 1)
                        continue;
                    //if (i == 2 && j == 2) continue;
                    y[i, j].ObservedValue = 2 * i + j + 1;
                }
            }
            //var yOut = Variable.GaussianFromMeanAndVariance(b[2]+c[2], noiseVariance).Named("yOut");
            var yOut = Variable.GaussianFromMeanAndVariance(b[0] + c[1], noiseVariance).Named("yOut");

            if (true)
            {
                var engine2 = new InferenceEngine();
                engine2.Algorithm = new GibbsSampling();
                engine2.NumberOfIterations = 1000000;
                engine2.ShowProgress = false;
                Console.WriteLine("exact yOut = {0}", engine2.Infer(yOut));
            }
            var engine = new InferenceEngine();
            engine.ShowProgress = false;
            Console.WriteLine("unconstrained yOut = {0}", engine.Infer(yOut));
            b[0].ObservedValue = 0;
            Console.WriteLine("anchored yOut = {0}", engine.Infer(yOut));
            for (int i = 0; i < n; i++)
            {
                Console.WriteLine("b[{0}] = {1}", i, engine.Infer(b[i]));
            }
            for (int j = 0; j < m; j++)
            {
                Console.WriteLine("c[{0}] = {1}", j, engine.Infer(c[j]));
            }
        }

        [Fact]
        public void InitialiseToWeakTypeTest()
        {
            Variable<IDistribution<bool[]>> init = Variable.New<IDistribution<bool[]>>().Named("init");
            init.ObservedValue = Distribution<bool>.Array(new Bernoulli[] { new Bernoulli(0.3) });
            Range item = new Range(1);
            var bools = Variable.Array<bool>(item).Named("bools");
            bools.InitialiseTo(init);
            bools[item] = Variable.Bernoulli(0.1).ForEach(item);
            Variable.ConstrainEqualRandom(bools[item], new Bernoulli(0.2));
            Variable.ConstrainEqualRandom(bools[item], new Bernoulli(0.2));

            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(bools));
        }

        [Fact]
        public void InitialiseToWrongTypeError()
        {
            Assert.Throws<InferCompilerException>(() =>
            {
                var init = Distribution<double>.Array(new Gamma[1]);
                Range item = new Range(1);
                var array = Variable.Array<double>(item).Named("array");
                array.InitialiseTo(init);
                array[item] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(item);
                Variable.ConstrainEqualRandom(array[item], new Gaussian(0.2, 1));
                Variable.ConstrainEqualRandom(array[item], new Gaussian(0.2, 1));

                InferenceEngine engine = new InferenceEngine();
                Console.WriteLine(engine.Infer(array));

            });
        }

        [Fact]
        public void InitializeBarrenOutputTest()
        {
            Range item = new Range(10).Named("item");
            IDistribution<double[]> xPrior = Distribution<double>.Array(Util.ArrayInit(item.SizeAsInt, i => new Gaussian(i, 100)));
            VariableArray<double> x = Variable.Array<double>(item).Named("x");
            x.SetTo(Variable.Random(xPrior));
            x.AddAttribute(QueryTypes.Marginal);
            x.AddAttribute(QueryTypes.MarginalDividedByPrior);
            IDistribution<double[]> xInit = Distribution<double>.Array(Util.ArrayInit(item.SizeAsInt, i => new Gaussian(i, 1)));
            x.InitialiseTo(xInit);
            InferenceEngine engine = new InferenceEngine();
            var gen = engine.GetCompiledInferenceAlgorithm(x);
            gen.Execute(10);
            var xActual = (IDistribution<double[]>)gen.Marginal("x");
            Console.WriteLine(xActual);
            Assert.True(xActual.MaxDiff(xPrior) < 1e-10);
            var xOutput = (IDistribution<double[]>)gen.Marginal("x", QueryTypes.MarginalDividedByPrior.Name);
            Console.WriteLine(xOutput);
            Assert.True(xOutput.IsUniform());
        }

        [Fact]
        public void OutputTest()
        {
            Range item = new Range(10).Named("item");
            IDistribution<double[]> xPrior = Distribution<double>.Array(Util.ArrayInit(item.SizeAsInt, i => new Gaussian(i, 100)));
            VariableArray<double> x = Variable.Array<double>(item).Named("x");
            x.SetTo(Variable.Random(xPrior));
            x.AddAttribute(QueryTypes.Marginal);
            x.AddAttribute(QueryTypes.MarginalDividedByPrior);
            Gaussian[] xLikeArray = Util.ArrayInit(item.SizeAsInt, i => new Gaussian(i, 10));
            IDistribution<double[]> xLike = Distribution<double>.Array(xLikeArray);
            var xLikeVar = Variable.Constant(xLikeArray, item);
            Variable.ConstrainEqualRandom(x[item], xLikeVar[item]);
            IDistribution<double[]> xInit = Distribution<double>.Array(Util.ArrayInit(item.SizeAsInt, i => new Gaussian(i, 1)));
            x.InitialiseTo(xInit);
            InferenceEngine engine = new InferenceEngine();
            // This namespace is deliberately chosen to test an empty namespace.
            engine.ModelNamespace = "";
            var gen = engine.GetCompiledInferenceAlgorithm(x);
            gen.Execute(10);
            var xActual = (IDistribution<double[]>)gen.Marginal("x");
            Console.WriteLine(xActual);
            //Assert.True(xActual.MaxDiff(xPrior*xLike) < 1e-10);
            var xOutput = (IDistribution<double[]>)gen.Marginal("x", QueryTypes.MarginalDividedByPrior.Name);
            Console.WriteLine(xOutput);
            Assert.True(xOutput.MaxDiff(xLike) < 1e-10);
        }

        [Fact]
        public void MarginalDividedByPriorPointMassTest()
        {
            Variable<double> x = Variable.Random(Gaussian.PointMass(0.5));
            //Variable<double> x = Variable.GaussianFromMeanAndVariance(0.5, 0.0);
            x.Name = nameof(x);
            x.AddAttribute(QueryTypes.MarginalDividedByPrior);
            Variable.ConstrainEqualRandom(x, Gaussian.FromNatural(2, 0));

            InferenceEngine engine = new InferenceEngine();
            var msg = engine.Infer<Gaussian>(x, QueryTypes.MarginalDividedByPrior);
            Assert.Equal(2, msg.MeanTimesPrecision);
        }

        // model from Mark Vanderwel 
        // VMP performs badly on this model.  EP does better.
        [Fact]
        public void NegativeBinomialTest()
        {
            double gammaLnShape = 10;
            Variable<double> x = Variable.GaussianFromMeanAndPrecision(0, 1).Named("x");
            Variable<double> gammaLnRate = gammaLnShape - x;
            gammaLnRate.Named("GammaLnRate");

            int[] data = new int[] { 3, 4, 5, 6 };
            Range item = new Range(data.Length);
            VariableArray<int> y = Variable.Array<int>(item).Named("y");
            VariableArray<double> poissonRate = Variable.Array<double>(item).Named("poissonRate");
            using (Variable.ForEach(item))
            {
                if (true)
                {
                    //obs have negative-binomial distribution
                    //(represented as mixture of poisson and gamma(shape,rate)
                    poissonRate[item] = Variable.GammaFromShapeAndRate(
                        Variable.Exp(gammaLnShape).Named("GammaShape"),
                        Variable.Exp(gammaLnRate).Named("GammaRate")
                        );
                    y[item] = Variable.Poisson(poissonRate[item]).Named("Poisson");
                }
                else
                {
                    poissonRate[item] = Variable.GammaFromShapeAndRate(1, 1);
                    y[item] = Variable.Poisson(Variable.Exp(x));
                }
            }
            y.ObservedValue = data;

            InferenceEngine engine = new InferenceEngine();
            // This namespace is deliberately chosen to cause a potential name conflict when referring to Math.Exp.
            engine.ModelNamespace = "Microsoft.ML.Probabilistic";
            Gaussian xExpected = new Gaussian(1.395, 0.0568); // VMP on simple Poisson
            xExpected = new Gaussian(1.395, 0.05716); // EP on simple Poisson
            bool useSampling = false;
            if(useSampling)
            {
                GaussianEstimator estimator = new GaussianEstimator();
                for (int iter = 0; iter < 100_000; iter++)
                {
                    double xSample = Gaussian.FromMeanAndPrecision(0, 1).Sample();
                    double logWeight = data.Sum(yi => {
                        double rateSample = Gamma.FromShapeAndRate(System.Math.Exp(gammaLnShape), System.Math.Exp(gammaLnShape - xSample)).Sample();
                        Poisson yDist = new Poisson(rateSample);
                        return yDist.GetLogProb(yi);
                    });
                    double weight = System.Math.Exp(logWeight);
                    estimator.Add(xSample, weight);
                }
                xExpected = estimator.GetDistribution(new Gaussian());
            }
            for (int trial = 0; trial < 2; trial++)
            {
                if(trial == 1) engine.Compiler.GivePriorityTo(typeof(GammaFromShapeAndRateOp_Laplace));
                else if(trial == 2) engine.Algorithm = new VariationalMessagePassing();
                Gaussian xActual = engine.Infer<Gaussian>(x);
                Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
                Assert.True(xExpected.MaxDiff(xActual) < 1);
                IList<Gamma> p = engine.Infer<IList<Gamma>>(poissonRate);
                //Console.WriteLine(p);
                //Console.WriteLine(StringUtil.VerboseToString(p.Select(g => g.Shape)));
            }
        }

        // model from Thore Graepel
        [Fact]
        public void VectorGetItemTest()
        {
            VectorGetItem(new VariationalMessagePassing());
            VectorGetItem(new ExpectationPropagation());
            VectorGetItem(new GibbsSampling());
        }

        private void VectorGetItem(IAlgorithm algorithm)
        {
            int NRows = 2; // number of data points
            int NCols = 3; // number of dimensions

            // Gaussian mean vector
            Variable<Vector> meanVector;
            double[] meanArray = new double[NCols];
            for (int col = 0; col < NCols; col++)
                meanArray[col] = 0;
            meanVector = Variable.VectorGaussianFromMeanAndPrecision(
                Vector.FromArray(meanArray),
                PositiveDefiniteMatrix.IdentityScaledBy(NCols, 0.01));
            meanVector.Named("meanVector");

            // Gaussian precision matrix
            Variable<PositiveDefiniteMatrix> precisionMatrix;
            precisionMatrix = Variable.WishartFromShapeAndScale(1, PositiveDefiniteMatrix.IdentityScaledBy(NCols, 1));
            precisionMatrix.Named("precisionMatrix");
            precisionMatrix.ObservedValue = PositiveDefiniteMatrix.Identity(NCols);

            // Create a variable array which will hold the data
            Range rowRange = new Range(NRows).Named("rowRange");

            // this is the latent full data as vectors
            VariableArray<Vector> completeData = Variable.Array<Vector>(rowRange).Named("completeData");

            // this is the latent full data as arrays in case we want to use subarray or GetItems
            // VariableArray<VariableArray<double>> completeDataArray = Variable.Array<VariableArray<double>>(n).Named("completeDataArray");

            // Read out non-missing data and locations
            int[] NObserved = new int[NRows]; // number of observed data in each row
            double[][] inputValues = new double[NRows][]; // Jagged array holding observed data for each row
            int[][] inputIndices = new int[NRows][]; // Jagged array holding observed indices for each row

            for (int row = 0; row < NRows; row++)
            {
                NObserved[row] = 2;
                inputValues[row] = new double[] { 3, 4 };
                inputIndices[row] = new int[] { 0, 1 };
            }

            // set up range for observed values which are different in each row
            VariableArray<int> NObservedVar = Variable.Constant(NObserved, rowRange).Named("NObservedVar");
            Range observedRange = new Range(NObservedVar[rowRange]).Named("observedRange");

            // set up jagged arrays for observed values and observed indices
            VariableArray<VariableArray<double>, double[][]> observedValuesVar = Variable.Array(Variable.Array<double>(observedRange), rowRange).Named("observedValuesVar");
            VariableArray<VariableArray<int>, int[][]> observedIndicesVar = Variable.Array(Variable.Array<int>(observedRange), rowRange).Named("observedIndicesVar");

            // The Gaussians model (with missing data)
            using (Variable.ForEach(rowRange))
            {
                completeData[rowRange] = Variable.VectorGaussianFromMeanAndPrecision(meanVector, precisionMatrix);
                using (Variable.ForEach(observedRange))
                {
                    var item = Variable.GetItem(completeData[rowRange], observedIndicesVar[rowRange][observedRange]);
                    observedValuesVar[rowRange][observedRange] = item;
                }
                // it would seem more efficient to first create a subarray as follows, but this does not seem to work:
                // completeDataArray[n] = Variable.ArrayFromVector(completeData[n],n);
                // (completeDataArray[n], observedIndicesVar[n][observed]).ForEach(observed);
            }

            // Attach data to model
            observedIndicesVar.ObservedValue = inputIndices; // attach observation indices to model
            observedValuesVar.ObservedValue = inputValues; // attach observed values to model

            // inference
            InferenceEngine ie = new InferenceEngine();
            ie.Algorithm = algorithm;
            //if(algorithm is GibbsSampling) ie.NumberOfIterations = 10000;
            Console.WriteLine("Distribution over meanVector=\n" + ie.Infer(meanVector));
            Console.WriteLine("Distribution over precisionMatrix=\n" + ie.Infer(precisionMatrix));
            VectorGaussian[] inferredDataMarginals = ie.Infer<VectorGaussian[]>(completeData);
            double[,] outputData = new double[NRows, NCols];
            for (int i = 0; i < NRows; i++)
                for (int j = 0; j < NCols; j++)
                    outputData[i, j] = inferredDataMarginals[i].GetMean()[j];
        }

        [Fact]
        public void SoftmaxVectorMarginalPrototype()
        {
            // The number of users.
            var U = new Range(4).Named("U");
            // The number of items.
            var I = new Range(3).Named("I");
            // The number of ratings.
            var R = new Range(1).Named("R");


            var ratingUsers = Variable.Array<int>(R).Named("ratingUsers");
            ratingUsers.ObservedValue = new int[] { 0 };
            var ratingItems = Variable.Array<int>(R).Named("ratingItems");
            ratingItems.ObservedValue = new int[] { 0 };
            var ratingValue = Variable.Array<bool>(R).Named("ratingValue");
            ratingValue.ObservedValue = new bool[] { true };


            // The number of user prototypes.
            var UP = new Range(2).Named("UP");
            // The number of item prototypes.
            var IP = new Range(2).Named("IP");

            // Create a mapping for the user id.
            /*var userIdMap = Variable.Array<Vector>(U).Named("userIdMap");
      using (Variable.ForEach(U))
      {
              userIdMap[U] = Variable.VectorGaussianFromMeanAndPrecision(
                      Vector.Zero(UP.SizeAsInt), PositiveDefiniteMatrix.Identity(UP.SizeAsInt));
      }

      // Create a mapping for the item id.
      var itemIdMap = Variable.Array<Vector>(I).Named("itemIdMap");
      using (Variable.ForEach(I))
      {
              itemIdMap[I] = Variable.VectorGaussianFromMeanAndPrecision(
                      Vector.Zero(IP.SizeAsInt), PositiveDefiniteMatrix.Identity(IP.SizeAsInt));
      }*/

            // Create a prior for the rating matrix.
            var ratingMatrix = Variable.Array<double>(UP, IP).Named("ratingMatrix");
            using (Variable.ForEach(UP))
            {
                using (Variable.ForEach(IP))
                {
                    ratingMatrix[UP, IP] = Variable.Beta(1.0, 1.0);
                }
            }


            var userAllocation = Variable.Array<Vector>(U).Named("userAllocation");
            using (Variable.ForEach(U))
            {
                var v = Variable.VectorGaussianFromMeanAndPrecision(
                    Vector.Zero(UP.SizeAsInt), PositiveDefiniteMatrix.Identity(UP.SizeAsInt));
                userAllocation[U] = Variable.Softmax(v); //Variable.Softmax(userIdMap[U]);
            }
            userAllocation.SetValueRange(UP);

            var itemAllocation = Variable.Array<Vector>(I).Named("itemAllocation");
            using (Variable.ForEach(I))
            {
                var v = Variable.VectorGaussianFromMeanAndPrecision(
                    Vector.Zero(IP.SizeAsInt), PositiveDefiniteMatrix.Identity(IP.SizeAsInt));
                itemAllocation[I] = Variable.Softmax(v); //Variable.Softmax(itemIdMap[I]);
            }
            itemAllocation.SetValueRange(IP);

            using (Variable.ForEach(R))
            {
                var uProto = Variable.Discrete(userAllocation[ratingUsers[R]]).Named("uProto");
                var iProto = Variable.Discrete(itemAllocation[ratingItems[R]]).Named("iProto");
                using (Variable.Switch(uProto))
                {
                    using (Variable.Switch(iProto))
                    {
                        ratingValue[R] = Variable.Bernoulli(ratingMatrix[uProto, iProto]);
                    }
                }
            }

            var ie = new InferenceEngine(new VariationalMessagePassing());
            //ie.BrowserMode = BrowserMode.Always;
            Console.WriteLine("Rating matrix:\n" + ie.Infer(ratingMatrix));
        }

        [Fact]
        public void ReturnCopiesTest()
        {
            Range item = new Range(4);
            VariableArray<double> bPrior = Variable.Constant(Util.ArrayInit(item.SizeAsInt, i => (i + 1) / 10.0), item).Named("bPrior");
            VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
            bools[item] = Variable.Bernoulli(bPrior[item]);
            Variable<int> index = Variable.Discrete(0.1, 0.2, 0.3, 0.4).Named("index");
            index.SetValueRange(item);
            Variable<bool> y = Variable.New<bool>().Named("y");
            using (Variable.Switch(index))
            {
                y.SetTo(!bools[index]);
            }
            InferenceEngine engine = new InferenceEngine();
            // FreeMemory=false ensures that marginals are re-used when ReturnCopies=false
            engine.Compiler.FreeMemory = false;
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial == 0)
                    engine.Compiler.ReturnCopies = true;
                else
                    engine.Compiler.ReturnCopies = false;
                y.ObservedValue = true;
                object indexPost1 = engine.Infer(index);
                Console.WriteLine("index = {0}", indexPost1);
                object boolsPost1 = engine.Infer(bools);
                Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsPost1));

                Console.WriteLine();
                y.ObservedValue = false;
                object indexPost2 = engine.Infer(index);
                Console.WriteLine("index = {0}", indexPost2);
                object boolsPost2 = engine.Infer(bools);
                Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsPost2));
                if (trial == 0)
                {
                    Assert.True(!indexPost1.Equals(indexPost2));
                    Assert.True(!boolsPost1.Equals(boolsPost2));
                }
                else
                {
                    Assert.True(indexPost1.Equals(indexPost2));
                    Assert.True(boolsPost1.Equals(boolsPost2));
                }
            }
        }

        [Fact]
        public void ReturnCopiesTest2()
        {
            Range item = new Range(4);
            VariableArray<double> bPrior = Variable.Constant(Util.ArrayInit(item.SizeAsInt, i => (i + 1) / 10.0), item).Named("bPrior");
            VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
            bools[item] = Variable.Bernoulli(bPrior[item]);
            for (int i = 0; i < item.SizeAsInt - 1; i++)
            {
                Variable.ConstrainEqualRandom(bools[i] | bools[i + 1], new Bernoulli(0.2));
            }
            Variable.ConstrainEqualRandom(bools[0] | bools[item.SizeAsInt - 1], new Bernoulli(0.2));
            InferenceEngine engine = new InferenceEngine();
            // FreeMemory=false ensures that marginals are re-used when ReturnCopies=false
            engine.Compiler.FreeMemory = false;
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial == 0)
                    engine.Compiler.ReturnCopies = true;
                else
                    engine.Compiler.ReturnCopies = false;
                engine.NumberOfIterations = 1;
                object boolsPost1 = engine.Infer(bools);
                Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsPost1));

                Console.WriteLine();
                engine.NumberOfIterations = 2;
                object boolsPost2 = engine.Infer(bools);
                Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsPost2));
                if (trial == 0)
                {
                    Assert.True(!boolsPost1.Equals(boolsPost2));

                    // test resetting inference
                    engine.NumberOfIterations = 1;
                    var boolsPost3 = engine.Infer<Diffable>(bools);
                    Assert.True(boolsPost3.MaxDiff(boolsPost1) < 1e-10);
                }
                else
                {
                    Assert.True(boolsPost1.Equals(boolsPost2));
                }
            }
        }

        [Fact]
        public void ReturnCopiesTest3()
        {
            Range item = new Range(4);
            VariableArray<double> bPrior = Variable.Constant(Util.ArrayInit(item.SizeAsInt, i => (i + 1) / 10.0), item).Named("bPrior");
            VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
            bools[item] = Variable.Bernoulli(bPrior[item]);
            Variable<int> index = Variable.Discrete(0.1, 0.2, 0.3, 0.4).Named("index");
            index.SetValueRange(item);
            Variable<bool> y = Variable.New<bool>().Named("y");
            using (Variable.Switch(index))
            {
                y.SetTo(!bools[index]);
            }
            bools.AddAttribute(QueryTypes.MarginalDividedByPrior);
            index.AddAttribute(QueryTypes.MarginalDividedByPrior);
            InferenceEngine engine = new InferenceEngine();
            // FreeMemory=false ensures that marginals are re-used when ReturnCopies=false
            engine.Compiler.FreeMemory = false;
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial == 0)
                    engine.Compiler.ReturnCopies = true;
                else
                    engine.Compiler.ReturnCopies = false;
                y.ObservedValue = true;
                object indexPost1 = engine.Infer(index, QueryTypes.MarginalDividedByPrior);
                Console.WriteLine("index = {0}", indexPost1);
                object boolsPost1 = engine.Infer(bools, QueryTypes.MarginalDividedByPrior);
                Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsPost1));

                Console.WriteLine();
                y.ObservedValue = false;
                object indexPost2 = engine.Infer(index, QueryTypes.MarginalDividedByPrior);
                Console.WriteLine("index = {0}", indexPost2);
                object boolsPost2 = engine.Infer(bools, QueryTypes.MarginalDividedByPrior);
                Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsPost2));
                if (trial == 0)
                {
                    Assert.True(!indexPost1.Equals(indexPost2));
                    Assert.True(!boolsPost1.Equals(boolsPost2));
                }
                else
                {
                    Assert.True(indexPost1.Equals(indexPost2));
                    Assert.True(boolsPost1.Equals(boolsPost2));
                }
            }
        }

        [Fact]
        public void PartialConstrainEqualTest()
        {
            Discrete xPrior = new Discrete(0.1, 0.2, 0.3, 0.4);
            Range item = new Range(4);
            VariableArray<int> x = Variable.Array<int>(item).Named("x");
            x[item] = Variable.Random(xPrior).ForEach(item);
            VariableArray<bool> b = Variable.Array<bool>(item).Named("b");
            using (Variable.ForEach(item))
            {
                using (Variable.If(b[item]))
                {
                    Variable.ConstrainEqual(x[item], 2);
                }
            }
            InferenceEngine engine = new InferenceEngine();
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial == 0)
                {
                    b.ObservedValue = new bool[] { false, false, true, true };
                }
                else
                {
                    b.ObservedValue = new bool[] { false, false, false, false };
                }
                object xActual = engine.Infer(x);
                IDistribution<int[]> xExpected = Distribution<int>.Array(Util.ArrayInit(item.SizeAsInt, i => (b.ObservedValue[i] ? Discrete.PointMass(2, 4) : xPrior)));
                Console.WriteLine(StringUtil.JoinColumns("x = ", xActual, " should be ", xExpected));
                Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
            }
        }

        [Fact]
        public void PartialConstrainEqualTest3()
        {
            Discrete xPrior = new Discrete(0.1, 0.2, 0.3, 0.4);
            Range item = new Range(4);
            VariableArray<int> x = Variable.Array<int>(item).Named("x");
            x[item] = Variable.Random(xPrior).ForEach(item);
            VariableArray<bool> b = Variable.Array<bool>(item).Named("b");
            using (Variable.ForEach(item))
            {
                b[item] = Variable.Bernoulli(0.1);
                using (Variable.If(b[item]))
                {
                    Variable.ConstrainEqual(x[item], 2);
                    Variable.ConstrainEqual(x[0], 2);
                }
            }
            InferenceEngine engine = new InferenceEngine();
            object xActual = engine.Infer(x);
            IDistribution<int[]> xExpected = Distribution<int>.Array(Util.ArrayInit(item.SizeAsInt, i => xPrior));
            Console.WriteLine(StringUtil.JoinColumns("x = ", xActual, " should be ", xExpected));
            //Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
        }

        [Fact]
        public void PartialConstrainEqualTest2()
        {
            Discrete xPrior = new Discrete(0.1, 0.2, 0.3, 0.4);
            Range item = new Range(4);
            Range inner = new Range(3).Named("inner");
            var y = Variable.Array<int>(inner).Named("y");
            y.ObservedValue = Util.ArrayInit(inner.SizeAsInt, j => 2);
            var x = Variable.Array(Variable.Array<int>(inner), item).Named("x");
            x[item][inner] = Variable.Random(xPrior).ForEach(item, inner);
            VariableArray<bool> b = Variable.Array<bool>(item).Named("b");
            b.ObservedValue = new bool[] { false, false, true, true };
            using (Variable.ForEach(item))
            {
                using (Variable.If(b[item]))
                {
                    Variable.ConstrainEqual(x[item], y);
                }
            }
            InferenceEngine engine = new InferenceEngine();
            object xActual = engine.Infer(x);
            IDistribution<int[][]> xExpected = Distribution<int>.Array(Util.ArrayInit(item.SizeAsInt, i =>
                                                                                                      Util.ArrayInit(inner.SizeAsInt,
                                                                                                                     j =>
                                                                                                                     (b.ObservedValue[i]
                                                                                                                          ? Discrete.PointMass(y.ObservedValue[j], 4)
                                                                                                                          : xPrior))));
            Console.WriteLine(StringUtil.JoinColumns("x = ", xActual, " should be ", xExpected));
            Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
        }

        internal void BufferTest()
        {
            Variable<bool> x = Variable.Bernoulli(0.1).Named("x");
            Variable<bool> y = Variable<bool>.Factor<bool>(BufferTester.Copy, x).Named("y");
            InferenceEngine engine = new InferenceEngine();
            Bernoulli yActual = engine.Infer<Bernoulli>(y);
            Console.WriteLine(yActual);
        }

        [Fact]
        public void SetObservedValueToSameReference()
        {
            Range item = new Range(2).Named("item");
            VariableArray<double> m = Variable.Array<double>(item).Named("m");
            VariableArray<double> x = Variable.Array<double>(item).Named("x");
            x[item] = Variable.GaussianFromMeanAndVariance(m[item], 100);
            VariableArray<double> y = Variable.Array<double>(item).Named("y");
            y[item] = Variable.GaussianFromMeanAndVariance(x[item], 1);
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.Compiler.UnrollLoops = false;
            engine.NumberOfIterations = 10;
            int iterationExpected = 0;
            engine.ProgressChanged += delegate (InferenceEngine sender, InferenceProgressEventArgs e)
                {
                    if (e.Iteration != iterationExpected)
                        Assert.True(false, String.Format("Wrong iteration number: {0} should be {1}", e.Iteration, iterationExpected));
                    iterationExpected++;
                };
            double[] data = new double[] { 1, 2 };
            m.ObservedValue = data;
            engine.Infer(x);
            Assert.Equal(iterationExpected, engine.NumberOfIterations);
            // check that inference does not run again
            engine.Infer(x);
            Assert.Equal(iterationExpected, engine.NumberOfIterations);
            // set the ObservedValue to the same reference, and check that inference is run again
            m.ObservedValue = data;
            iterationExpected = 0;
            engine.Infer(x);
            Assert.Equal(iterationExpected, engine.NumberOfIterations);
            // check that inference does not run again
            engine.Infer(x);
            Assert.Equal(iterationExpected, engine.NumberOfIterations);
        }

        [Fact]
        public void ProgressChangedTest()
        {
            Variable<double> m = Variable.New<double>().Named("m");
            Variable<double> x = Variable.GaussianFromMeanAndVariance(m, 100).Named("x");
            Variable<double> y = Variable.GaussianFromMeanAndVariance(x, 1).Named("y");
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.NumberOfIterations = 10;
            int iterationExpected = 0;
            engine.ProgressChanged += delegate (InferenceEngine sender, InferenceProgressEventArgs e)
                {
                    if (e.Iteration != iterationExpected)
                        Assert.True(false, String.Format("Wrong iteration number: {0} should be {1}", e.Iteration, iterationExpected));
                    iterationExpected++;
                };
            m.ObservedValue = 10;
            engine.Infer(x);
            Assert.Equal(iterationExpected, engine.NumberOfIterations);
            m.ObservedValue = -10;
            engine.ResetOnObservedValueChanged = false;
            iterationExpected = 0;
            engine.Infer(x);
            Assert.Equal(iterationExpected, engine.NumberOfIterations);
        }

        [Fact]
        public void ProgressChangedTest2()
        {
            Variable<double> m = Variable.New<double>().Named("m");
            Variable<double> x = Variable.GaussianFromMeanAndVariance(m, 100).Named("x");
            Variable<double> y = Variable.GaussianFromMeanAndVariance(x, 1).Named("y");
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            m.ObservedValue = 10;
            IGeneratedAlgorithm gen = engine.GetCompiledInferenceAlgorithm(x);
            int iterationExpected = 0;
            gen.ProgressChanged += delegate (object sender, ProgressChangedEventArgs e)
                {
                    if (e.Iteration != iterationExpected)
                        Assert.True(false, String.Format("Wrong iteration number: {0} should be {1}", e.Iteration, iterationExpected));
                    iterationExpected++;
                };
            gen.SetObservedValue(m.NameInGeneratedCode, 10.0);
            int numberOfIterations = 10;
            gen.Execute(numberOfIterations);
            Assert.Equal(iterationExpected, numberOfIterations);
            gen.Update(numberOfIterations);
            Assert.Equal(iterationExpected, 2 * numberOfIterations);
            gen.SetObservedValue(m.NameInGeneratedCode, -10.0);
            iterationExpected = 0;
            gen.Update(numberOfIterations);
            Assert.Equal(iterationExpected, numberOfIterations);
        }

        [Fact]
        public void ResetTest()
        {
            Variable<double> m = Variable.New<double>().Named("m");
            Variable<double> x = Variable.GaussianFromMeanAndVariance(m, 100).Named("x");
            Variable<double> y = Variable.GaussianFromMeanAndVariance(x, 1).Named("y");
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            double[] means = { 10, -10, 10 };
            for (int i = 0; i < means.Length; i++)
            {
                m.ObservedValue = means[i];
                Gaussian xActual;
                Gaussian xExpected;
                // if we do not Reset here, the results will change depending on the order of Infer statements.
                // since we did not set ResumeLastRun, Reset is done automatically.
                //engine.Reset();
                xActual = engine.Infer<Gaussian>(x);
                xExpected = ResetTest_OneShot(m.ObservedValue);
                Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
                Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
            }
        }

        [Fact]
        public void ResetTest2()
        {
            Variable<double> m = Variable.New<double>().Named("m");
            Variable<double> x = Variable.GaussianFromMeanAndVariance(m, 100).Named("x");
            Variable<double> y = Variable.GaussianFromMeanAndVariance(x, 1).Named("y");
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.ResetOnObservedValueChanged = false;
            double[] means = { 10, -10, 10 };
            for (int i = 0; i < means.Length; i++)
            {
                m.ObservedValue = means[i];
                Gaussian xActual;
                Gaussian xExpected;
                if (i > 1)
                {
                    // if we do not Reset here, the results will change depending on the order of Infer statements.
                    //engine.Reset();
                    engine.ResetOnObservedValueChanged = true;
                }
                xActual = engine.Infer<Gaussian>(x);
                xExpected = ResetTest_OneShot(m.ObservedValue);
                Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
                if (i == 1)
                {
                    // here the results should be different
                    Assert.True(xExpected.MaxDiff(xActual) > 1);
                }
                else
                {
                    Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
                }
            }
        }

        public Gaussian ResetTest_OneShot(double mean)
        {
            Variable<double> m = Variable.New<double>().Named("m");
            Variable<double> x = Variable.GaussianFromMeanAndVariance(m, 100).Named("x");
            Variable<double> y = Variable.GaussianFromMeanAndVariance(x, 1).Named("y");
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            m.ObservedValue = mean;
            return engine.Infer<Gaussian>(x);
        }

        [Fact]
        public void ResetTest3()
        {
            Variable<Gaussian> xPrior = Variable.Observed<Gaussian>(Gaussian.FromMeanAndVariance(10, 100)).Named("xPrior");
            Variable<double> x = Variable<double>.Random(xPrior).Named("x");
            Variable<double> y = Variable.GaussianFromMeanAndVariance(x, 1).Named("y");
            Variable<Gaussian> yLike = Variable.Observed<Gaussian>(Gaussian.FromMeanAndVariance(0, 1)).Named("yLike");
            Variable.ConstrainEqualRandom(y, yLike);
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.ModelName = "ResetTest3";
            int[] numIters = new int[] { 1, 2, 3 };
            for (int i = 0; i < numIters.Length; i++)
            {
                Gaussian xActual;
                Gaussian xExpected;
                engine.NumberOfIterations = numIters[i];
                xActual = engine.Infer<Gaussian>(x);
                xExpected = ResetTest3_OneShot(numIters[i]);
                Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
                Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
            }
        }

        public Gaussian ResetTest3_OneShot(int numberOfIterations)
        {
            Variable<Gaussian> xPrior = Variable.Observed<Gaussian>(Gaussian.FromMeanAndVariance(10, 100)).Named("xPrior");
            Variable<double> x = Variable<double>.Random(xPrior).Named("x");
            Variable<double> y = Variable.GaussianFromMeanAndVariance(x, 1).Named("y");
            Variable<Gaussian> yLike = Variable.Observed<Gaussian>(Gaussian.FromMeanAndVariance(0, 1)).Named("yLike");
            Variable.ConstrainEqualRandom(y, yLike);
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.NumberOfIterations = numberOfIterations;
            return engine.Infer<Gaussian>(x);
        }

        // This model has two iteration loops with no statements between them.
        [Fact]
        public void TwoLoopNoDivideTest()
        {
            var csharp = Variable.Bernoulli(0.5).Named("csharp");
            csharp.AddAttribute(new DivideMessages(false));
            var sql = Variable.Bernoulli(0.5).Named("sql");
            sql.AddAttribute(new DivideMessages(false));

            var q1 = (csharp & sql);
            var q2 = (csharp & sql);
            q1.ObservedValue = false;
            q2.ObservedValue = false;

            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            Bernoulli cActual = engine.Infer<Bernoulli>(csharp);
            Bernoulli cExpected = new Bernoulli(0.2764);
            Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
            Assert.True(cExpected.MaxDiff(cActual) < 1e-4);
        }

        [Fact]
        public void TwoLoopScheduleTest()
        {
            Variable<bool> c = TwoLoopScheduleModel();
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.NumberOfIterations = 1;
            Bernoulli cPost1 = engine.Infer<Bernoulli>(c);
            engine.NumberOfIterations = engine.Algorithm.DefaultNumberOfIterations;
            Bernoulli cActual = engine.Infer<Bernoulli>(c);
            Bernoulli cExpected = new Bernoulli(0.2136);
            Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
            Assert.True(cExpected.MaxDiff(cActual) < 1e-4);
            // test resetting inference
            engine.NumberOfIterations = 1;
            var cPost2 = engine.Infer<Diffable>(c);
            Assert.True(cPost2.MaxDiff(cPost1) < 1e-10);
        }

        [Fact]
        public void TwoLoopScheduleTest2()
        {
            Variable<bool> c = TwoLoopScheduleModel();
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.ModelName = "TwoLoopScheduleTest";
            engine.OptimiseForVariables = new Variable[] { c };
            int[] numIters = new int[] { 1, 2, 3 };
            for (int i = 0; i < numIters.Length; i++)
            {
                Bernoulli cActual;
                Bernoulli cExpected;
                engine.NumberOfIterations = numIters[i];
                cActual = engine.Infer<Bernoulli>(c);
                cExpected = TwoLoopScheduleTest2_OneShot(numIters[i]);
                Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
                Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
            }
        }

        public Variable<bool> TwoLoopScheduleModel()
        {
            Variable<Beta> pcPrior = Variable.Constant(new Beta(1, 1)).Named("pcPrior");
            Variable<double> pc = Variable<double>.Random(pcPrior).Named("pc");
            Variable<bool> c = Variable.Bernoulli(pc).Named("c");
            using (Variable.If(c))
            {
                Variable<double> px = Variable.Beta(1, 1).Named("px");
                Variable<bool> x = Variable.Bernoulli(px).Named("x");
                Variable.ConstrainEqualRandom(x, new Bernoulli(0.1));
            }
            return c;
        }

        public Bernoulli TwoLoopScheduleTest2_OneShot(int numberOfIterations)
        {
            Variable<bool> c = TwoLoopScheduleModel();
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.NumberOfIterations = numberOfIterations;
            engine.OptimiseForVariables = new Variable[] { c };
            return engine.Infer<Bernoulli>(c);
        }

        // Gives the wrong answer due to an undetected deterministic loop.
        // Chooses one of two possible solutions instead of averaging them.
        [Fact]
        [Trait("Category", "OpenBug")]
        public void AndOrXorTest()
        {
            Variable<bool> x = Variable.Bernoulli(0.5).Named("x");
            Variable<bool> y = Variable.Bernoulli(0.5).Named("y");
            var xt = x & (~y);
            var yt = (~x) & y;
            var z = yt | xt;
            // This is an equivalent model that gets the right answer.
            //z = (x != y);
            z.ObservedValue = true;
            z.Name = "z";
            InferenceEngine ie = new InferenceEngine();
            Bernoulli xExpected = new Bernoulli(0.5);
            Bernoulli xActual = ie.Infer<Bernoulli>(x);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Bernoulli yExpected = new Bernoulli(0.5);
            Bernoulli yActual = ie.Infer<Bernoulli>(y);
            Console.WriteLine("y = {0} should be {1}", yActual, yExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
        }

        // An interesting case for belief propagation.
        // Constant propagation would help to remove the final "and" factor and make the model simpler.
        [Fact]
        public void AndOrXorTest2()
        {
            Variable<bool> x = Variable.Bernoulli(0.5).Named("x");
            Variable<bool> y = Variable.Bernoulli(0.5).Named("y");
            x.AddAttribute(new DivideMessages(false));
            y.AddAttribute(new DivideMessages(false));
            var or = x | y;
            or.Name = "or";
            var and = x & y;
            and.Name = "and";
            if (true)
            {
                var z = or & !and;
                //z = (x != y);
                z.ObservedValue = true;
                z.Name = "z";
            }
            else
            {
                // removing the final "and" factor
                Variable.ConstrainTrue(or);
                Variable.ConstrainFalse(and);
            }
            InferenceEngine ie = new InferenceEngine();
            Bernoulli xExpected = new Bernoulli(0.5);
            Bernoulli xActual = ie.Infer<Bernoulli>(x);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
        }

        [Fact]
        public void AddObservedVarAfterInfer()
        {
            Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 1).Named("x");

            InferenceEngine engine = new InferenceEngine();
            Gaussian priorExpected = new Gaussian(0, 1);
            Gaussian priorActual = engine.Infer<Gaussian>(x);
            Assert.True(priorExpected.MaxDiff(priorActual) < 1e-10);

            Variable<bool> greaterThanHalf = (x > 0.5);
            greaterThanHalf.ObservedValue = true;

            Gaussian expected = new Gaussian(1.1410777703680648, 0.26848040715587884);
            Gaussian actual = engine.Infer<Gaussian>(x);
            Console.WriteLine("x = {0} should be {1}", actual, expected);
            Assert.True(expected.MaxDiff(actual) < 1e-4);
        }

        [Fact]
        public void KeepOldMarginals()
        {
            int n = 2;
            Range item = new Range(n).Named("item");
            Variable<double> bPrior = Variable.New<double>().Named("bPrior");
            VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
            bools[item] = Variable.Bernoulli(bPrior).ForEach(item);
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.ReturnCopies = true;
            bPrior.ObservedValue = 0.3;
            object marginal1 = engine.Infer(bools);
            bPrior.ObservedValue = 0.4;
            object marginal2 = engine.Infer(bools);
            Bernoulli[] expectedArray1 = new Bernoulli[n];
            for (int i = 0; i < n; i++)
            {
                expectedArray1[i] = new Bernoulli(0.3);
            }
            IDistribution<bool[]> expected1 = Distribution<bool>.Array(expectedArray1);
            Bernoulli[] expectedArray2 = new Bernoulli[n];
            for (int i = 0; i < n; i++)
            {
                expectedArray2[i] = new Bernoulli(0.4);
            }
            IDistribution<bool[]> expected2 = Distribution<bool>.Array(expectedArray2);
            Console.WriteLine(StringUtil.JoinColumns("marginal1 = ", marginal1, " should be ", expected1));
            Console.WriteLine(StringUtil.JoinColumns("marginal2 = ", marginal2, " should be ", expected2));
            Assert.True(expected1.MaxDiff(marginal1) < 1e-10);
            Assert.True(expected2.MaxDiff(marginal2) < 1e-10);
        }

        [Fact]
        public void ConstrainInRepeatTest()
        {
            var p = Variable.Beta(1, 1).Named("p");
            using (Variable.Repeat(10))
            {
                Variable.ConstrainTrue(Variable.Bernoulli(p));
            }
            InferenceEngine engine = new InferenceEngine();
            int iterationsDone = 0;
            engine.ProgressChanged += delegate (InferenceEngine sender, InferenceProgressEventArgs e)
            {
                iterationsDone++;
            };
            Console.WriteLine(engine.Infer(p));
            // Ensure that this is not an iterative algorithm.
            Assert.Equal(0, iterationsDone);
        }

        [Fact]
        public void PowerPlateEnterTest()
        {
            double xPrior = 0.1;
            Variable<bool> x = Variable.Bernoulli(xPrior).Named("x");
            double n = 4.5;
            double data = 0.2;
            using (Variable.Repeat(n))
            {
                Variable.ConstrainEqualRandom(x, new Bernoulli(data));
            }

            InferenceEngine engine = new InferenceEngine();
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial == 1)
                    engine.Algorithm = new VariationalMessagePassing();
                Bernoulli xActual = engine.Infer<Bernoulli>(x);
                double sumxT = xPrior * System.Math.Pow(data, n);
                double z = sumxT + (1 - xPrior) * System.Math.Pow(1 - data, n);
                Bernoulli xExpected = new Bernoulli(sumxT / z);
                Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
                Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
            }
        }

        [Fact]
        public void PowerPlateEnterTest2()
        {
            Variable<double> p = Variable.Beta(1, 1).Named("p");
            double n = 4;
            using (Variable.Repeat(n))
            {
                Variable<bool> x = Variable.Bernoulli(p).Named("x");
                double data = 0.2;
                Variable.ConstrainEqualRandom(x, new Bernoulli(data));
            }

            InferenceEngine engine = new InferenceEngine();
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial == 1)
                    engine.Algorithm = new VariationalMessagePassing();
                Beta pActual = engine.Infer<Beta>(p);
                Beta pExpected = PowerPlateEnterTest2_Unrolled(engine.Algorithm);
                Console.WriteLine("p = {0} should be {1}", pActual, pExpected);
                Assert.True(pExpected.MaxDiff(pActual) < 1e-10);
            }
        }

        public Beta PowerPlateEnterTest2_Unrolled(IAlgorithm alg)
        {
            Variable<double> p = Variable.Beta(1, 1).Named("p");
            double n = 4;
            double data = 0.2;
            for (int i = 0; i < n; i++)
            {
                Variable<bool> x = Variable.Bernoulli(p).Named("x" + i);
                Variable.ConstrainEqualRandom(x, new Bernoulli(data));
            }
            InferenceEngine engine = new InferenceEngine(alg);
            return engine.Infer<Beta>(p);
        }

        [Fact]
        public void GetCodeToInferTest()
        {
            Variable<bool> firstCoin = Variable.Bernoulli(0.5).Named("firstCoin");
            Variable<bool> secondCoin = Variable.Bernoulli(0.5).Named("secondCoin");
            Variable<bool> bothHeads = (firstCoin & secondCoin).Named("bothHeads");
            InferenceEngine engine = new InferenceEngine();
            //engine.RestrictInferenceTo(bothHeads); // optional
            engine.OptimiseForVariables = new[] { bothHeads };
            Console.WriteLine(engine.GetCodeToInfer(bothHeads)[0]);
        }

        [Fact]
        public void OptimiseForVariablesTest()
        {
            Variable<bool> firstCoin = Variable.Bernoulli(0.5).Named("firstCoin");
            Variable<bool> secondCoin = Variable.Bernoulli(0.5).Named("secondCoin");
            Variable<bool> bothHeads = (firstCoin & secondCoin).Named("bothHeads");
            InferenceEngine engine = new InferenceEngine();
            engine.OptimiseForVariables = new[] { bothHeads };
            Console.WriteLine(engine.Infer(bothHeads));
            try
            {
                engine.Infer(firstCoin); // should throw an exception
            }
            catch (Exception ex)
            {
                Console.WriteLine("Correctly threw exception: " + ex);
            }
            engine.OptimiseForVariables = new[] { firstCoin };
            Console.WriteLine(engine.Infer(firstCoin));
            try
            {
                engine.Infer(bothHeads); // should throw an exception
            }
            catch (Exception ex)
            {
                Console.WriteLine("Correctly threw exception: " + ex);
            }
            engine.OptimiseForVariables = null;
            Console.WriteLine(engine.Infer(bothHeads));
            Console.WriteLine(engine.Infer(firstCoin));
        }

        [Fact]
        public void InferDisjointWithoutRecompilationTest()
        {
            Variable<double> xPrior = Variable.Observed(0.1).Named("xPrior");
            Variable<bool> x = Variable.Bernoulli(xPrior).Named("x");
            Variable<double> yPrior = Variable.Observed(0.2).Named("yPrior");
            Variable<bool> y = Variable.Bernoulli(yPrior).Named("y");
            InferenceEngine engine = new InferenceEngine();
            int compileCount = 0;
            engine.Compiler.Compiling += delegate (ModelCompiler sender, ModelCompiler.CompileEventArgs e) { compileCount++; };
            Bernoulli xExpected = new Bernoulli(0.1);
            Bernoulli yExpected = new Bernoulli(0.2);
            for (int trial = 0; trial < 3; trial++)
            {
                Bernoulli xActual = engine.Infer<Bernoulli>(x);
                Bernoulli yActual = engine.Infer<Bernoulli>(y);
                Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
                Assert.True(yExpected.MaxDiff(yActual) < 1e-10);
                Assert.True(compileCount <= 2);
            }
        }

        [Fact]
        public void InferQueryTypeWithoutRecompilationTest()
        {
            Variable<double> xPrior = Variable.Observed(0.1).Named("xPrior");
            Variable<bool> x = Variable.Bernoulli(xPrior).Named("x");
            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = new GibbsSampling();
            int compileCount = 0;
            engine.Compiler.Compiling += delegate (ModelCompiler sender, ModelCompiler.CompileEventArgs e) { compileCount++; };
            Bernoulli xExpected = new Bernoulli(0.1);
            Bernoulli xActual = engine.Infer<Bernoulli>(x);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
            IList<bool> xSamples = engine.Infer<IList<bool>>(x, QueryTypes.Samples);
            Assert.True(compileCount == 1);
        }

        [Fact]
        public void InferDeterministicTest()
        {
            Variable<bool> firstCoin = Variable.Bernoulli(0.5).Named("firstCoin");
            Variable<bool> secondCoin = Variable.Bernoulli(0.5).Named("secondCoin");
            Variable<bool> bothHeads = (firstCoin & secondCoin).Named("bothHeads");
            bothHeads.ObservedValue = true;
            InferenceEngine ie = new InferenceEngine();
            IDistribution<bool> pm = ie.Infer<IDistribution<bool>>(bothHeads);
            Console.WriteLine("Both heads posterior: " + pm);
            Assert.True(pm.IsPointMass);
            Assert.True(pm.Point);
        }

        [Fact]
        public void InferDeterministicTest2()
        {
            Variable<bool> firstCoin = Variable.Bernoulli(0.5).Named("firstCoin");
            Variable<bool> secondCoin = Variable.Bernoulli(0.5).Named("secondCoin");
            Variable<bool> bothHeads = (firstCoin & secondCoin).Named("bothHeads");
            bothHeads.ObservedValue = true;
            InferenceEngine ie = new InferenceEngine();
            Bernoulli pm = ie.Infer<Bernoulli>(bothHeads);
            Console.WriteLine("Both heads posterior: " + pm);
            Assert.True(pm.IsPointMass);
            Assert.True(pm.Point);
        }

        [Fact]
        public void InferDeterministicTest3()
        {
            Variable<bool> x = Variable.Bernoulli(0.1).Named("x");
            x.ObservedValue = true;
            Variable<bool> y = !x;
            InferenceEngine ie = new InferenceEngine();
            Bernoulli yExpected = Bernoulli.PointMass(false);
            Bernoulli yActual = ie.Infer<Bernoulli>(y);
            Console.WriteLine("y = {0} should be {1}", yActual, yExpected);
            Assert.True(yExpected.MaxDiff(yActual) < 1e-10);
            if (false)
            {
                PointMass<bool> yPointExpected = new PointMass<bool>(false);
                PointMass<bool> yPointActual = ie.Infer<PointMass<bool>>(y);
                Console.WriteLine("y = {0} should be {1}", yPointActual, yPointExpected);
                Assert.True(yPointExpected.MaxDiff(yPointActual) < 1e-10);
            }
            if (true)
            {
                Sampleable<bool> ySampleable = ie.Infer<Sampleable<bool>>(y);
                Console.WriteLine("y = {0} should be {1}", ySampleable, yExpected);
                Assert.True(yExpected.MaxDiff(ySampleable) < 1e-10);
            }
            if (false)
            {
                bool yValue = ie.Infer<bool>(y);
                Console.WriteLine("y = {0} should be {1}", yValue, false);
                Assert.True(yValue == false);
            }
        }

        [Fact]
        public void InferDeterministicTest4()
        {
            Range unwrappedValue = new Range(4).Named("unwrappedValue");
            Variable<int> unwrapped = Variable.DiscreteUniform(unwrappedValue).Named("unwrapped");
            Range wrappedValue = new Range(2).Named("wrappedValue");
            Variable<int> wrapped = Variable.New<int>().Named("wrapped");
            VariableArray<int> modulo2 = Variable.Observed(new int[] { 0, 1, 0, 1 }, unwrappedValue).Named("modulo2");
            modulo2.SetValueRange(wrappedValue);
            using (Variable.Switch(unwrapped))
            {
                wrapped.SetTo(modulo2[unwrapped] + 0);
            }
            wrapped.ObservedValue = 1;
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(unwrapped));
            var wrappedActual = engine.Infer<IDistribution<int>>(wrapped);
            Console.WriteLine("wrapped = {0} should be {1}", wrappedActual, Discrete.PointMass(wrapped.ObservedValue, unwrappedValue.SizeAsInt));
            Assert.True(wrappedActual.IsPointMass);
            Assert.Equal(wrapped.ObservedValue, wrappedActual.Point);
        }

        [Fact]
        public void InferDeterministicTest5()
        {
            Range item = new Range(1);
            item.Name = nameof(item);
            Range unwrappedValue = new Range(4).Named("unwrappedValue");
            Variable<int> unwrapped = Variable.DiscreteUniform(unwrappedValue).Named("unwrapped");
            Range wrappedValue = new Range(2).Named("wrappedValue");
            VariableArray<int> wrapped = Variable.Array<int>(item).Named("wrapped");
            VariableArray<int> modulo2 = Variable.Observed(new int[] { 0, 1, 0, 1 }, unwrappedValue).Named("modulo2");
            modulo2.SetValueRange(wrappedValue);
            using (Variable.Switch(unwrapped))
            {
                using (var block = Variable.ForEach(item))
                {
                    wrapped[item].SetTo(modulo2[unwrapped] + block.Index);
                }
            }
            wrapped.ObservedValue = new[] { 1 };
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(unwrapped));
            var wrappedActual = engine.Infer<IDistribution<int[]>>(wrapped);
            //Console.WriteLine("wrapped = {0} should be {1}", wrappedActual, Discrete.PointMass(wrapped.ObservedValue, unwrappedValue.SizeAsInt));
            Assert.True(wrappedActual.IsPointMass);
            Assert.Equal(wrapped.ObservedValue, wrappedActual.Point);
        }

        [Fact]
        public void AllTrueTest()
        {
            AllTrue(new ExpectationPropagation());
        }

        [Fact]
        public void VmpAllTrue()
        {
            AllTrue(new VariationalMessagePassing());
        }

        [Fact]
        public void GibbsAllTrue()
        {
            GibbsSampling gs = new GibbsSampling();
            gs.DefaultNumberOfIterations = 50000;
            AllTrue(gs);
        }

        private void AllTrue(IAlgorithm algorithm)
        {
            for (int i = 0; i < 4; i++)
            {
                AllTrue(i, algorithm);
            }
        }

        private void AllTrue(int n, IAlgorithm algorithm)
        {
            double[] priors = new double[n];
            for (int i = 0; i < n; i++)
            {
                priors[i] = 0.2 + 0.8 * i / n;
            }
            Bernoulli likelihood = new Bernoulli(0.9);

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            if (algorithm is GibbsSampling)
                block.CloseBlock();
            Range item = new Range(priors.Length).Named("item");
            VariableArray<double> priorsVar = Variable.Constant(priors, item).Named("priors");
            VariableArray<bool> inputs = Variable.Array<bool>(item).Named("inputs");
            inputs[item] = Variable.Bernoulli(priorsVar[item]);
            Variable<bool> output = Variable.AllTrue(inputs).Named("output");
            Variable.ConstrainEqualRandom(output, likelihood);
            if (!(algorithm is GibbsSampling))
                block.CloseBlock();
            InferenceEngine engine = new InferenceEngine();
            if (n == 0)
                engine.Compiler.UnrollLoops = false; // otherwise inputs will have no definition
            if (!(algorithm is GibbsSampling))
                engine.Algorithm = algorithm;
            double tolerance = 1e-10;
            if (algorithm is GibbsSampling)
                tolerance = 1e-1;
            IDistribution<bool[]> inputsActual = engine.Infer<IDistribution<bool[]>>(inputs);
            double evExpected;
            IDistribution<bool[]> inputsExpected = Distribution<bool>.Array(AllTrueUnrolled(likelihood, priors, algorithm, out evExpected));
            Console.WriteLine(StringUtil.JoinColumns("inputs = ", inputsActual, " should be ", inputsExpected));
            if (n > 0)
                Assert.True(inputsExpected.MaxDiff(inputsActual) < tolerance);
            if (!(algorithm is GibbsSampling))
            {
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
                Assert.True(MMath.AbsDiff(evExpected, evActual) < tolerance);
            }
        }

        private Bernoulli[] AllTrueUnrolled(Bernoulli likelihood, double[] priors, IAlgorithm algorithm, out double logEvidence)
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            if (algorithm is GibbsSampling)
                block.CloseBlock();
            int n = priors.Length;
            Variable<bool>[] inputs = new Variable<bool>[n];
            for (int i = 0; i < inputs.Length; i++)
            {
                inputs[i] = Variable.Bernoulli(priors[i]).Named("inputs" + i);
            }
            Variable<bool> output;
            if (n == 0)
            {
                output = Variable.Constant(true);
            }
            else
            {
                output = inputs[0];
                for (int i = 1; i < n; i++)
                {
                    output = inputs[i] & output;
                }
            }
            output.Name = "output";
            Variable.ConstrainEqualRandom(output, likelihood);
            if (!(algorithm is GibbsSampling))
                block.CloseBlock();
            InferenceEngine engine = new InferenceEngine(algorithm);
            if (algorithm is GibbsSampling)
            {
                Rand.Restart(12347);
                engine.NumberOfIterations = 50000;
                engine.ShowProgress = false;
            }
            Bernoulli[] posts = new Bernoulli[n];
            for (int i = 0; i < posts.Length; i++)
            {
                posts[i] = engine.Infer<Bernoulli>(inputs[i]);
            }
            if (!(algorithm is GibbsSampling))
                logEvidence = engine.Infer<Bernoulli>(evidence).LogOdds;
            else
                logEvidence = 0.0;
            return posts;
        }

        [Fact]
        public void GammaProductTest()
        {
            GammaProduct(new ExpectationPropagation());
            GammaProduct(new VariationalMessagePassing());
        }

        [Fact]
        public void GibbsGammaProduct()
        {
            GammaProduct(new GibbsSampling());
        }

        private void GammaProduct(IAlgorithm algorithm)
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = null;
            if (!(algorithm is GibbsSampling))
                block = Variable.If(evidence);
            Variable<double> theta = Variable.GammaFromShapeAndScale(3, 1.0 / 5).Named("theta");
            Variable<int> x = Variable.Poisson(2.0 * theta).Named("x");
            x.ObservedValue = 3;
            if (!(algorithm is GibbsSampling))
                block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = algorithm;
            Gamma thetaActual = engine.Infer<Gamma>(theta);
            Gamma thetaExpected = new Gamma(6, 1.0 / 7);
            Console.WriteLine("theta = {0} should be {1}", thetaActual, thetaExpected);
            Assert.True(thetaExpected.MaxDiff(thetaActual) < 1e-10);

            if (!(algorithm is GibbsSampling))
            {
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                // exact evidence:
                // int_theta (2*theta)^3/3!*exp(-2*theta) theta^(3-1)*exp(-5*theta)*5^3/Gamma(3)
                // = (2*5)^3/3!/Gamma(3) int_theta theta^(6-1)*exp(-7*theta)
                // = (2*5)^3/3!/Gamma(3) Gamma(6)/7^6
                // or:
                // int_theta theta^3/3!*exp(-theta)*theta^(3-1)*exp(-5/2*theta)*(5/2)^3/Gamma(3)
                // = (5/2)^3/Gamma(3)/3! int_theta theta^(6-1)*exp(-7/2*theta)
                // = (5/2)^3/Gamma(3)/3! Gamma(6)/(7/2)^6
                double evExpected = 3 * System.Math.Log(2 * 5) - MMath.GammaLn(4) - MMath.GammaLn(3) + MMath.GammaLn(6) - 6 * System.Math.Log(7);
                Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-8) < 1e-10);
            }
        }

        [Fact]
        public void VaryingSizeVectorArrayCopyTest()
        {
            Range item = new Range(3).Named("item");
            VariableArray<Dirichlet> priors = Variable.Array<Dirichlet>(item).Named("priors");
            VariableArray<Vector> probs = Variable.Array<Vector>(item).Named("probs");
            probs[item] = Variable<Vector>.Random(priors[item]);
            VariableArray<Vector> probsCopy = Variable.Array<Vector>(item).Named("probsCopy");
            probsCopy.SetTo(Variable.Copy(probs));
            VariableArray<int> x = Variable.Array<int>(item).Named("x");
            x[item] = Variable.Discrete(probsCopy[item]);
            InferenceEngine engine = new InferenceEngine();
            Dirichlet[] ppriors = new Dirichlet[]
                {
                    new Dirichlet(0.4, 0.6),
                    new Dirichlet(0.1, 0.2, 0.7),
                    new Dirichlet(0.1, 0.3, 0.4, 0.2)
                };
            priors.ObservedValue = ppriors;
            x.ObservedValue = new int[] { 0, 0, 0 };
            Dirichlet[] probsExpected = new Dirichlet[3];
            for (int i = 0; i < probsExpected.Length; i++)
            {
                Dirichlet d = new Dirichlet(ppriors[i]);
                d.PseudoCount[x.ObservedValue[i]]++;
                d.TotalCount++;
                probsExpected[i] = d;
            }
            DistributionArray<Dirichlet> probsActual = engine.Infer<DistributionArray<Dirichlet>>(probs);
            Console.WriteLine(StringUtil.JoinColumns("probs = ", probsActual, " should be ",
                                                     Distribution<Vector>.Array(probsExpected)));
            for (int i = 0; i < probsExpected.Length; i++)
            {
                Assert.True(probsExpected[i].MaxDiff(probsActual[i]) < 1e-10);
            }
        }

        [Fact]
        public void VaryingSizeVectorArrayTest()
        {
            Range item = new Range(3).Named("item");
            VariableArray<Dirichlet> priors = Variable.Array<Dirichlet>(item).Named("priors");
            VariableArray<Vector> probs = Variable.Array<Vector>(item).Named("probs");
            probs[item] = Variable<Vector>.Random(priors[item]);
            VariableArray<int> x = Variable.Array<int>(item).Named("x");
            x[item] = Variable.Discrete(probs[item]);
            InferenceEngine engine = new InferenceEngine();
            Dirichlet[] ppriors = new Dirichlet[]
                {
                    new Dirichlet(0.4, 0.6),
                    new Dirichlet(0.1, 0.2, 0.7),
                    new Dirichlet(0.1, 0.3, 0.4, 0.2)
                };
            priors.ObservedValue = ppriors;
            DistributionArray<Discrete> xActual = engine.Infer<DistributionArray<Discrete>>(x);
            for (int i = 0; i < xActual.Count; i++)
            {
                Discrete xExpected = new Discrete(ppriors[i].GetMean());
                Assert.True(xExpected.MaxDiff(xActual[i]) < 1e-10);
            }
        }

        [Fact]
        public void VaryingSizeVectorArray2Test()
        {
            Range item = new Range(3).Named("item");
            VariableArray<Dirichlet> priors = Variable.Array<Dirichlet>(item).Named("priors");
            VariableArray<Vector> probs = Variable.Array<Vector>(item).Named("probs");
            probs[item] = Variable<Vector>.Random(priors[item]);
            VariableArray<int> x = Variable.Array<int>(item).Named("x");
            x[item] = Variable.Discrete(probs[item]);
            InferenceEngine engine = new InferenceEngine();
            Dirichlet[] ppriors = new Dirichlet[]
                {
                    new Dirichlet(0.4, 0.6),
                    new Dirichlet(0.1, 0.2, 0.7),
                    new Dirichlet(0.1, 0.3, 0.4, 0.2)
                };
            priors.ObservedValue = ppriors;
            x.ObservedValue = new int[] { 0, 0, 0 };
            Dirichlet[] probsExpected = new Dirichlet[3];
            for (int i = 0; i < probsExpected.Length; i++)
            {
                Dirichlet d = new Dirichlet(ppriors[i]);
                d.PseudoCount[x.ObservedValue[i]]++;
                d.TotalCount++;
                probsExpected[i] = d;
            }
            DistributionArray<Dirichlet> probsActual = engine.Infer<DistributionArray<Dirichlet>>(probs);
            Console.WriteLine(StringUtil.JoinColumns("probs = ", probsActual, " should be ",
                                                     Distribution<Vector>.Array(probsExpected)));
            for (int i = 0; i < probsExpected.Length; i++)
            {
                Assert.True(probsExpected[i].MaxDiff(probsActual[i]) < 1e-10);
            }
        }


        [Fact]
        public void VectorJaggedArrayTest()
        {
            VectorJaggedArray(new VariationalMessagePassing());
        }

        [Fact]
        public void GibbsVectorJaggedArrayTest()
        {
            VectorJaggedArray(new GibbsSampling());
        }

        private void VectorJaggedArray(IAlgorithm algorithm)
        {
            int nFeature = 2;
            int nUsers = 2;
            int[] items = new int[] { 20, 10 };

            // data creation
            Rand.Restart(12347);
            Vector[] wtrue = new Vector[nUsers];
            Vector[][] dataX = new Vector[nUsers][];
            double[][] dataY = new double[nUsers][];
            for (int u = 0; u < nUsers; u++)
            {
                wtrue[u] = Vector.Constant(nFeature, .5 * (u + 1));
                dataX[u] = new Vector[items[u]];
                dataY[u] = new double[items[u]];
                for (int itm = 0; itm < items[u]; itm++)
                {
                    dataX[u][itm] = Vector.Constant(nFeature, u + 1);
                    dataY[u][itm] = Vector.InnerProduct(wtrue[u], dataX[u][itm]);
                }
            }

            //model
            Range user = new Range(nUsers).Named("user");
            Variable<Vector> mu = Variable.VectorGaussianFromMeanAndPrecision(
                Vector.Zero(nFeature),
                PositiveDefiniteMatrix.Identity(nFeature)).Named("mu");
            //mu.ObservedValue = Vector.Zero(nFeature);
            Variable<PositiveDefiniteMatrix> prec = Variable.WishartFromShapeAndScale(10.0, PositiveDefiniteMatrix.Identity(nFeature)).Named("prec");
            //prec.ObservedValue = PositiveDefiniteMatrix.Identity(nFeature);
            VariableArray<Vector> w = Variable.Array<Vector>(user);
            w[user] = Variable.VectorGaussianFromMeanAndPrecision(mu, prec).ForEach(user);
            w.Named("w");
            VariableArray<int> itemVar = Variable.Constant(items, user).Named("items");
            Range item = new Range(itemVar[user]).Named("item");
            VectorGaussian xPrior = new VectorGaussian(Vector.Zero(nFeature), PositiveDefiniteMatrix.Identity(nFeature));
            var x = Variable.Array(Variable.Array<Vector>(item), user).Named("x");
            var y = Variable.Array(Variable.Array<double>(item), user).Named("y");
            using (Variable.ForEach(user))
            {
                x[user][item] = Variable.Random(xPrior).ForEach(item);
                y[user][item] = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(x[user][item], w[user]).Named("h"), 1);
            }
            if (false)
            {
                var xdata = Variable.Constant<Vector>(dataX, user, item).Named("xdata");
                Variable.ConstrainEqual(x[user][item], xdata[user][item]);
                var ydata = Variable.Constant<double>(dataY, user, item).Named("ydata");
                Variable.ConstrainEqual(y[user][item], ydata[user][item]);
            }
            else
            {
                x.ObservedValue = dataX;
                y.ObservedValue = dataY;
            }

            InferenceEngine engine = new InferenceEngine(algorithm);
            if (algorithm is GibbsSampling)
            {
                Rand.Restart(12347);
                engine.ShowProgress = false;
                engine.NumberOfIterations = 20000;
            }
            DistributionArray<VectorGaussian> wPost = engine.Infer<DistributionArray<VectorGaussian>>(w);
            VectorGaussian[] wPost_unrolled = VectorJaggedArrayTest_Unrolled(algorithm, dataX, dataY);
            double tolerance = engine.Compiler.UnrollLoops ? 0.8 : 0.1;
            for (int i = 0; i < wPost_unrolled.Length; i++)
            {
                Console.WriteLine(StringUtil.JoinColumns("w[", i, "] = ", wPost[i], " should be ", wPost_unrolled[i]));
                Assert.True(wPost_unrolled[i].MaxDiff(wPost[i]) < tolerance);
            }
        }

        public VectorGaussian[] VectorJaggedArrayTest_Unrolled(IAlgorithm algorithm, Vector[][] dataX, double[][] dataY)
        {
            int nUsers = dataX.Length;
            int nFeature = dataX[0][0].Count;
            Variable<Vector> mu = Variable.VectorGaussianFromMeanAndPrecision(
                Vector.Zero(nFeature),
                PositiveDefiniteMatrix.Identity(nFeature)).Named("mu");
            //mu.ObservedValue = Vector.Zero(nFeature);
            Variable<PositiveDefiniteMatrix> prec = Variable.WishartFromShapeAndScale(10.0, PositiveDefiniteMatrix.Identity(nFeature)).Named("prec");
            //prec.ObservedValue = PositiveDefiniteMatrix.Identity(nFeature);
            VectorGaussian xPrior = new VectorGaussian(Vector.Zero(nFeature), PositiveDefiniteMatrix.Identity(nFeature));
            Variable<Vector>[][] x = new Variable<Vector>[nUsers][];
            Variable<double>[][] y = new Variable<double>[nUsers][];
            Variable<Vector>[] w = new Variable<Vector>[nUsers];
            for (int u = 0; u < nUsers; u++)
            {
                w[u] = Variable.VectorGaussianFromMeanAndPrecision(mu, prec);
                int count = dataX[u].Length;
                x[u] = new Variable<Vector>[count];
                y[u] = new Variable<double>[count];
                for (int i = 0; i < count; i++)
                {
                    x[u][i] = Variable.Random(xPrior);
                    y[u][i] = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(x[u][i], w[u]), 1);
                    if (false)
                    {
                        Variable.ConstrainEqual(x[u][i], dataX[u][i]);
                        Variable.ConstrainEqual(y[u][i], dataY[u][i]);
                    }
                    else
                    {
                        x[u][i].ObservedValue = dataX[u][i];
                        y[u][i].ObservedValue = dataY[u][i];
                    }
                }
            }

            InferenceEngine engine = new InferenceEngine(algorithm);
            if (algorithm is GibbsSampling)
            {
                Rand.Restart(12347);
                engine.ShowProgress = false;
                engine.NumberOfIterations = 20000;
            }
            VectorGaussian[] wInferred = new VectorGaussian[nUsers];
            for (int u = 0; u < nUsers; u++)
            {
                wInferred[u] = engine.Infer<VectorGaussian>(w[u]);
            }
            return wInferred;
        }

        [Fact]
        public void JaggedArray2Test()
        {
            int[] sizes = new int[] { 2, 3 };
            Range user = new Range(sizes.Length).Named("user");
            VariableArray<int> sizesVar = Variable.Constant(sizes, user).Named("sizes");
            Range feature = new Range(sizesVar[user]).Named("feature");

            // make a jagged array
            var xPrior = Variable.Array(Variable.Array<Gaussian>(feature), user).Named("xPrior");
            xPrior.ObservedValue = Util.ArrayInit(sizes.Length, u => Util.ArrayInit(sizes[u], f => new Gaussian(1.2, 3.4)));
            var x = Variable.Array(Variable.Array<double>(feature), user).Named("x");
            var y = Variable.Array(Variable.Array<double>(feature), user).Named("y");
            using (Variable.ForEach(user))
            {
                // this is required for ForEach(feature) to work
                x[user][feature] = Variable<double>.Random(xPrior[user][feature]);
                y[user][feature] = Variable.GaussianFromMeanAndPrecision(x[user][feature], 1.0);
                Variable.ConstrainPositive(x[user][feature]);
            }
            var ydata = Variable.Observed(new double[][] { new double[] { 1, 2 }, new double[] { 1, 2, 3 } }, user, feature);
            Variable.ConstrainEqual(y[user][feature], ydata[user][feature]);

            InferenceEngine engine = new InferenceEngine();
            //engine.BrowserMode = BrowserMode.Always;

            DistributionArray<GaussianArray> dist = engine.Infer<DistributionArray<GaussianArray>>(x);
            Console.WriteLine("x = ");
            Console.WriteLine(dist);
            //Gaussian xPost = xPrior * PositiveOp.XAverageConditional(xPrior);
            for (int i = 0; i < sizes.Length; i++)
            {
                for (int j = 0; j < sizes[i]; j++)
                {
                    //Assert.True(dist[i][j].MaxDiff(xPost) < 1e-10);
                }
            }
        }

        [Fact]
        public void JaggedArrayTest()
        {
            Gaussian xPrior = new Gaussian(1.2, 3.4);
            int[] sizes = new int[] { 2, 3 };
            Range user = new Range(sizes.Length).Named("user");
            VariableArray<int> sizesVar = Variable.Constant(sizes, user).Named("sizes");
            Range feature = new Range(sizesVar[user]).Named("feature");

            // make a jagged array
            var x = Variable.Array(Variable.Array<double>(feature), user).Named("x");
            using (Variable.ForEach(user))
            {
                // this is required for ForEach(feature) to work
                x[user][feature] = Variable.Random(xPrior).ForEach(feature);
                Variable.ConstrainPositive(x[user][feature]);
            }

            InferenceEngine engine = new InferenceEngine();
            DistributionArray<GaussianArray> dist = engine.Infer<DistributionArray<GaussianArray>>(x);
            Console.WriteLine("x = ");
            Console.WriteLine(dist);
            Gaussian xPost = xPrior * IsPositiveOp.XAverageConditional(true, xPrior);
            for (int i = 0; i < sizes.Length; i++)
            {
                for (int j = 0; j < sizes[i]; j++)
                {
                    Assert.True(dist[i][j].MaxDiff(xPost) < 1e-10);
                }
            }
        }

        [Fact]
        public void JaggedArrayInlineTest()
        {
            Gaussian xPrior = new Gaussian(1.2, 3.4);
            int[] sizes = new int[] { 2, 3 };
            Range user = new Range(sizes.Length).Named("user");
            VariableArray<int> sizesVar = Variable.Constant(sizes, user).Named("sizes");
            Range feature = new Range(sizesVar[user]).Named("feature");

            // make a jagged array
            var x = Variable.Array(Variable.Array<double>(feature), user).Named("x");
            x[user][feature] = Variable.Random(xPrior).ForEach(user, feature);
            Variable.ConstrainPositive(x[user][feature]);

            InferenceEngine engine = new InferenceEngine();
            DistributionArray<GaussianArray> dist = engine.Infer<DistributionArray<GaussianArray>>(x);
            Console.WriteLine("x = ");
            Console.WriteLine(dist);
            Gaussian xPost = xPrior * IsPositiveOp.XAverageConditional(true, xPrior);
            for (int i = 0; i < sizes.Length; i++)
            {
                for (int j = 0; j < sizes[i]; j++)
                {
                    Assert.True(dist[i][j].MaxDiff(xPost) < 1e-10);
                }
            }
        }

        [Fact]
        [Trait("Category", "BadTest")]
        public void ImplicitJaggedArrayTest()
        {
            Gaussian xPrior = new Gaussian(1.2, 3.4);
            int[] sizes = new int[] { 2, 3 };
            Range user = new Range(sizes.Length).Named("user");
            VariableArray<int> sizesVar = Variable.Constant(sizes, user).Named("sizes");
            Variable<double> x;
            using (Variable.ForEach(user))
            {
                Range feature = new Range(sizesVar[user]).Named("feature");
                using (Variable.ForEach(feature))
                {
                    x = Variable.Random(xPrior).Named("x");
                    Variable.ConstrainPositive(x);
                }
            }

            InferenceEngine engine = new InferenceEngine();
            DistributionArray<GaussianArray> dist = engine.Infer<DistributionArray<GaussianArray>>(x);
            Console.WriteLine("x = ");
            Console.WriteLine(dist);
            Gaussian xPost = xPrior * IsPositiveOp.XAverageConditional(true, xPrior);
            for (int i = 0; i < sizes.Length; i++)
            {
                for (int j = 0; j < sizes[i]; j++)
                {
                    Assert.True(dist[i][j].MaxDiff(xPost) < 1e-10);
                }
            }
        }

        [Fact]
        [Trait("Category", "BadTest")]
        public void ImplicitJaggedArray2Test()
        {
            Gaussian xPrior = new Gaussian(1.2, 3.4);
            int[] sizes = new int[] { 2, 3 };
            Range user = new Range(sizes.Length).Named("user");
            VariableArray<int> sizesVar = Variable.Constant(sizes, user).Named("sizes");
            VariableArray<double> x;
            using (Variable.ForEach(user))
            {
                Range feature = new Range(sizesVar[user]).Named("feature");
                x = Variable.Array<double>(feature).Named("x");
                x[feature] = Variable<double>.Random(xPrior).ForEach(feature);
                Variable.ConstrainPositive(x[feature]);
            }

            InferenceEngine engine = new InferenceEngine();
            DistributionArray<GaussianArray> dist = engine.Infer<DistributionArray<GaussianArray>>(x);
            Console.WriteLine("x = ");
            Console.WriteLine(dist);
            Gaussian xPost = xPrior * IsPositiveOp.XAverageConditional(true, xPrior);
            for (int i = 0; i < sizes.Length; i++)
            {
                for (int j = 0; j < sizes[i]; j++)
                {
                    Assert.True(dist[i][j].MaxDiff(xPost) < 1e-10);
                }
            }
        }

        // incorrect inference for Max using Gates
        internal void MaxGateTest()
        {
            Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 1);
            Variable<double> y = Variable.GaussianFromMeanAndVariance(0, 1);
            Variable<double> max = Variable.New<double>();
            Variable<bool> xGy = x > y;


            using (Variable.If(xGy))
            {
                max.SetTo(Variable.Copy(x));
            }
            using (Variable.IfNot(xGy))
            {
                max.SetTo(Variable.Copy(y));
            }
            InferenceEngine ie = new InferenceEngine();
            Gaussian maxPost = ie.Infer<Gaussian>(max);
            Bernoulli xGyPost = ie.Infer<Bernoulli>(xGy);
            Console.WriteLine("maxPost = {0}", maxPost.ToString());
            Console.WriteLine("xGyPost = {0}", xGyPost.ToString());
        }

        [Fact]
        public void MaxNoClickTest()
        {
            Variable<int> nUsers = Variable.New<int>().Named("nUsers");
            Range user = new Range(nUsers).Named("user");

            Variable<double> noClick = Variable.GaussianFromMeanAndVariance(0, 100).Named("noClick");
            VariableArray<bool> clicks = Variable.Array<bool>(user).Named("clicks");

            // Sequential gives faster convergence than damping
            bool useSequential = true;
            int modelType = 0;
            double stepsize = 0.5;
            if (modelType == 0)
            {
                // undamped oscillation
                using (Variable.ForEach(user))
                {
                    Variable<double> x = Variable.GaussianFromMeanAndVariance(noClick, 1);
                    Variable<double> y = Variable.GaussianFromMeanAndVariance(.33, 1);
                    Variable<double> z = Variable.GaussianFromMeanAndVariance(.33, 1);
                    clicks[user] = (y > Variable.Max(z, x));
                }
            }
            else if (modelType == 1)
            {
                // incorrect damping location
                var noClickDamped = Variable<double>.Factor<double, double>(Damp.Forward, noClick, stepsize).Named("noClickDamped");
                using (Variable.ForEach(user))
                {
                    Variable<double> x = Variable.GaussianFromMeanAndVariance(noClickDamped, 1);
                    Variable<double> y = Variable.GaussianFromMeanAndVariance(.33, 1);
                    Variable<double> z = Variable.GaussianFromMeanAndVariance(.33, 1);
                    clicks[user] = (y > Variable.Max(z, x));
                }
            }
            else if(modelType == 2)
            {
                // damping the wrong variable (still works in this case)
                using (Variable.ForEach(user))
                {
                    Variable<double> x = Variable.GaussianFromMeanAndVariance(noClick, 1);
                    Variable<double> y = Variable.GaussianFromMeanAndVariance(.33, 1);
                    Variable<double> z = Variable.GaussianFromMeanAndVariance(.33, 1);
                    var xDamped = Variable<double>.Factor<double, double>(Damp.Forward, x, stepsize).Named("xDamped");
                    clicks[user] = (y > Variable.Max(z, xDamped));
                }
            }
            else
            {
                // correct damping
                using (Variable.ForEach(user))
                {
                    // Forward since noClick is the root of the fan
                    var noClickDamped = Variable<double>.Factor<double, double>(Damp.Forward, noClick, stepsize).Named("noClickDamped");
                    Variable<double> x = Variable.GaussianFromMeanAndVariance(noClickDamped, 1);
                    Variable<double> y = Variable.GaussianFromMeanAndVariance(.33, 1);
                    Variable<double> z = Variable.GaussianFromMeanAndVariance(.33, 1);
                    clicks[user] = (y > Variable.Max(z, x));
                }
            }

            nUsers.ObservedValue = 20;
            bool[] clickArray = new bool[nUsers.ObservedValue];
            for (int i = 0; i < nUsers.ObservedValue; i++)
            {
                clickArray[i] = (i % 2 == 0);
            }
            clicks.ObservedValue = clickArray;
            //IsPositiveOp.ForceProper = true;
            //GateEnterPartialOp.ForceProper = true;
            MaxGaussianOp.ForceProper = false; // must be false or else wrong answer
            InferenceEngine engine = new InferenceEngine();
            if (useSequential)
            {
                // don't need damping if use sequential
                user.AddAttribute(new Sequential());
                engine.Compiler.UseSerialSchedules = true;
            }
            for (int iter = 1; iter < 50; iter++)
            {
                engine.NumberOfIterations = iter;
                Console.WriteLine("{0}: {1}", iter, engine.Infer(noClick));
            }
            Gaussian noClickExpected = new Gaussian(-8.366, 18.25);
            Gaussian noClickActual = engine.Infer<Gaussian>(noClick);
            Console.WriteLine("noClick = {0} should be {1}", noClickActual, noClickExpected);
            Assert.True(noClickExpected.MaxDiff(noClickActual) < 1e-4);
        }

        [Fact]
        public void DampBackwardTest()
        {
            Bernoulli xPrior = new Bernoulli(0.1);
            Bernoulli xLike = new Bernoulli(0.6);
            Variable<bool> x = Variable<bool>.Random(xPrior).Named("x");
            var xDamped = Variable<bool>.Factor(Damp.Backward<bool>, x, 0.5).Named("xDamped");
            Variable.ConstrainEqualRandom(xDamped, xLike);

            InferenceEngine engine = new InferenceEngine();
            Bernoulli xExpected = xPrior * xLike;
            Bernoulli xActual = engine.Infer<Bernoulli>(x);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
        }

        [Fact]
        public void DampForwardTest()
        {
            Bernoulli xPrior = new Bernoulli(0.1);
            Bernoulli xLike = new Bernoulli(0.6);
            Variable<bool> x = Variable<bool>.Random(xPrior).Named("x");
            var xDamped = Variable<bool>.Factor(Damp.Forward<bool>, x, 0.5).Named("xDamped");
            var y = Variable.Copy(xDamped).Named("y");
            Variable.ConstrainEqualRandom(y, xLike);

            InferenceEngine engine = new InferenceEngine();
            Bernoulli xExpected = xPrior * xLike;
            Bernoulli xActual = engine.Infer<Bernoulli>(y);
            Console.WriteLine("y = {0} should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
        }

        [Fact]
        public void PriorArrayTest4()
        {
            int n = 1;
            Range item = new Range(n).Named("item");

            Discrete[] init = new Discrete[n];
            for (int i = 0; i < n; i++)
            {
                init[i] = new Discrete(0.1, 0.2);
            }
            var initDist = Distribution<int>.Array(init);
            var bInit = Variable.Observed(initDist).Named("bInit");
            //Variable<bool[]> b = Variable.Random<bool[],DistributionStructArray<Bernoulli,bool>>(bInit).Named("b");
            //Variable<bool[]> b = Variable.Random<bool[],IDistribution<bool[]>>(bInit).Named("b");
            VariableArray<int> b = Variable.Array<int>(item).Named("b");
            b.SetTo(Variable.Random<int[]>(initDist));

            InferenceEngine engine = new InferenceEngine();
            var ca = engine.GetCompiledInferenceAlgorithm(b);
            ca.Execute(1);
            Discrete[] bActual = Distribution.ToArray<Discrete[]>(ca.Marginal("b"));
            Assert.True(System.Math.Abs(bActual[0][0] - 1.0 / 3) < 1e-10);
        }

        [Fact]
        public void InitialiseInCycleTest()
        {
            Gaussian xInit = new Gaussian(3, 4);
            Variable<double> m = Variable.GaussianFromMeanAndVariance(0, 100).Named("m");
            Variable<double> x = Variable.GaussianFromMeanAndVariance(m, 1).Named("x");
            Variable.ConstrainPositive(x);
            Variable.ConstrainEqualRandom(x, Gaussian.Uniform());
            Variable.ConstrainPositive(m);
            x.InitialiseTo(xInit);
            m.AddAttribute(new DivideMessages(false));
            InferenceEngine engine = new InferenceEngine();
            engine.NumberOfIterations = 2;
            Gaussian mPrior = new Gaussian(0, 100);
            Gaussian x_F = xInit;
            Gaussian xExpected = new Gaussian();
            for (int iter = 0; iter < 2; iter++)
            {
                Gaussian x_use_B = IsPositiveOp.XAverageConditional(true, x_F);
                Gaussian m_use_B1 = GaussianOp.MeanAverageConditional(x_use_B, 1);
                Gaussian m_use_B2 = IsPositiveOp.XAverageConditional(true, m_use_B1 * mPrior);
                x_F = GaussianOp.SampleAverageConditional(mPrior * m_use_B2, 1);
                xExpected = x_F * x_use_B;
            }
            Gaussian xActual = engine.Infer<Gaussian>(x);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
        }

        [Fact]
        public void InitialiseToConstantTest()
        {
            InitialiseToConstant(createGeneratedAlgorithm: false);
        }

        [Fact]
        public void CreateGeneratedAlgorithmTest()
        {
            InitialiseToConstant(createGeneratedAlgorithm: true);
        }

        private void InitialiseToConstant(bool createGeneratedAlgorithm)
        {
            Gaussian xInit = Gaussian.PointMass(3);
            Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 100).Named("x");
            Variable<double> y = Variable.GaussianFromMeanAndVariance(x, 1).Named("y");
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.NumberOfIterations = 1;
            x.InitialiseTo(xInit);
            Gaussian xActual;
            if (createGeneratedAlgorithm)
            {
                IGeneratedAlgorithm gen = engine.GetCompiledInferenceAlgorithm(x);
                IGeneratedAlgorithm gen2 = (IGeneratedAlgorithm)Activator.CreateInstance(gen.GetType());
                gen2.Execute(1);
                xActual = gen2.Marginal<Gaussian>("x");
            }
            else
            {
                xActual = engine.Infer<Gaussian>(x);
            }
            Gaussian xExpected = new Gaussian(0, 1);
            if (true)
            {
                // compute the desired answer
                Gaussian xPrior = new Gaussian(0, 100);
                Gaussian xMarginal = xInit;
                Gaussian yMarginal;
                for (int iter = 0; iter < 1; iter++)
                {
                    yMarginal = Gaussian.FromMeanAndVariance(xMarginal.GetMean(), 1);
                    xMarginal = Gaussian.FromMeanAndVariance(yMarginal.GetMean(), 1) * xPrior;
                }
                xExpected = xMarginal;
            }
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-2);
        }

        // This test requires CopyPropagation to skip initialised variables.
        [Fact]
        public void InitialiseToCopyTest()
        {
            Gaussian yInit = new Gaussian(0, 1);
            Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 100).Named("x");
            Variable<double> y = Variable.Copy(x).Named("y");
            y.InitialiseTo(yInit);
            Variable.ConstrainPositive(x);
            Variable.ConstrainPositive(y);
            InferenceEngine engine = new InferenceEngine();
            engine.NumberOfIterations = 1;
            x.AddAttribute(new DivideMessages(false));
            Gaussian xActual = engine.Infer<Gaussian>(x);
            Gaussian xExpected = new Gaussian(0, 1);
            if (true)
            {
                // compute the desired answer
                Gaussian xPrior = new Gaussian(0, 100);
                Gaussian xLike1 = IsPositiveOp.XAverageConditional(true, yInit);
                Gaussian xLike2 = IsPositiveOp.XAverageConditional(true, xPrior * xLike1);
                xExpected = xPrior * xLike1 * xLike2;
            }
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-2);
        }

        /// <summary>
        /// Test initialization of variables outside the iteration loop.
        /// </summary>
        [Fact]
        public void InitialiseToReplicateTest()
        {
            Gaussian xInit = new Gaussian(0, 1);
            Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 100).Named("x");
            x.InitialiseTo(Variable.Observed(xInit).Named("xInit"));
            Range item = new Range(1).Named("i");
            VariableArray<bool> y = Variable.Array<bool>(item).Named("y");
            using (Variable.ForEach(item))
            {
                y[item] = (x > 0);
            }
            y.ObservedValue = new bool[] { true };
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.UnrollLoops = false; // ensure that x is replicated
            engine.NumberOfIterations = 1;
            Gaussian xActual = engine.Infer<Gaussian>(x);
            // compute the desired answer
            Gaussian xPrior = new Gaussian(0, 100);
            Gaussian xExpected = xPrior * IsPositiveOp.XAverageConditional(true, xInit);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-2);

            engine.NumberOfIterations = 10;
            engine.Infer(x);

            // test resetting inference
            engine.NumberOfIterations = 1;
            xActual = engine.Infer<Gaussian>(x);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-2);
        }

        /// <summary>
        /// checks that Variable.InitialiseTo invalidates previous inference results
        /// </summary>
        [Fact]
        public void InitialiseToReplicateTest2()
        {
            Gaussian xInit = new Gaussian(0, 1);
            Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 100).Named("x");
            x.InitialiseTo(Variable.Observed(xInit).Named("xInit"));
            Range item = new Range(1).Named("i");
            VariableArray<bool> y = Variable.Array<bool>(item).Named("y");
            using (Variable.ForEach(item))
            {
                y[item] = (x > 0);
            }
            y.ObservedValue = new bool[] { true };
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.UnrollLoops = false; // ensure that x is replicated
            engine.NumberOfIterations = 1;
            Gaussian xActual = engine.Infer<Gaussian>(x);
            // compute the desired answer
            Gaussian xPrior = new Gaussian(0, 100);
            Gaussian xExpected = xPrior * IsPositiveOp.XAverageConditional(true, xInit);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-2);

            Gaussian xInit2 = new Gaussian(0, 2);
            x.InitialiseTo(Variable.Observed(xInit2).Named("xInit2"));
            xActual = engine.Infer<Gaussian>(x);
            xExpected = xPrior * IsPositiveOp.XAverageConditional(true, xInit2);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-2);
        }

        [Fact]
        public void InitialiseArrayItemsTest()
        {
            Range item = new Range(2);
            var xInit = Gaussian.PointMass(3);
            var xInitArray = Variable.Observed(Util.ArrayInit(item.SizeAsInt, i => xInit), item).Named("xInit");
            VariableArray<double> x = Variable.Array<double>(item).Named("x");
            x[item] = Variable.GaussianFromMeanAndVariance(0, 100).ForEach(item);
            VariableArray<double> y = Variable.Array<double>(item).Named("y");
            y[item] = Variable.GaussianFromMeanAndVariance(x[item], 1);
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.Compiler.UnrollLoops = true;
            engine.NumberOfIterations = 1;
            x[item].InitialiseTo(xInitArray[item]);
            object xActual = engine.Infer(x);
            Gaussian xExpected = new Gaussian(0, 1);
            if (true)
            {
                // compute the desired answer
                Gaussian xPrior = new Gaussian(0, 100);
                Gaussian xMarginal = xInit;
                Gaussian yMarginal;
                for (int iter = 0; iter < 1; iter++)
                {
                    yMarginal = Gaussian.FromMeanAndVariance(xMarginal.GetMean(), 1);
                    xMarginal = Gaussian.FromMeanAndVariance(yMarginal.GetMean(), 1) * xPrior;
                }
                xExpected = xMarginal;
            }
            var xExpectedArray = Distribution<double>.Array(Util.ArrayInit(item.SizeAsInt, i => xExpected));
            Console.WriteLine("x = {0} should be {1}", xActual, xExpectedArray);
            Assert.True(xExpectedArray.MaxDiff(xActual) < 1e-2);
        }

        [Fact]
        public void InitialiseArrayItemsFromIReadOnlyList()
        {
            Range item = new Range(2);
            var xInit = Gaussian.PointMass(3);
            var xInitList = new List<Gaussian>();
            for (int i = 0; i < item.SizeAsInt; i++)
            {
                xInitList.Add(xInit);
            }
            var xInitArray = Variable.Observed((IReadOnlyList<Gaussian>)xInitList, item).Named("xInit");
            VariableArray<double> x = Variable.Array<double>(item).Named("x");
            x[item] = Variable.GaussianFromMeanAndVariance(0, 100).ForEach(item);
            VariableArray<double> y = Variable.Array<double>(item).Named("y");
            y[item] = Variable.GaussianFromMeanAndVariance(x[item], 1);
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.Compiler.UnrollLoops = true;
            engine.NumberOfIterations = 1;
            x[item].InitialiseTo(xInitArray[item]);
            object xActual = engine.Infer(x);
            Gaussian xExpected = new Gaussian(0, 1);
            if (true)
            {
                // compute the desired answer
                Gaussian xPrior = new Gaussian(0, 100);
                Gaussian xMarginal = xInit;
                Gaussian yMarginal;
                for (int iter = 0; iter < 1; iter++)
                {
                    yMarginal = Gaussian.FromMeanAndVariance(xMarginal.GetMean(), 1);
                    xMarginal = Gaussian.FromMeanAndVariance(yMarginal.GetMean(), 1) * xPrior;
                }
                xExpected = xMarginal;
            }
            var xExpectedArray = Distribution<double>.Array(Util.ArrayInit(item.SizeAsInt, i => xExpected));
            Console.WriteLine("x = {0} should be {1}", xActual, xExpectedArray);
            Assert.True(xExpectedArray.MaxDiff(xActual) < 1e-2);
        }

        [Fact]
        public void InitialiseArrayItemsError()
        {
            Assert.Throws<InvalidOperationException>(() =>
            {
                Range item = new Range(2);
                var xInit = Gaussian.PointMass(3);
                var xInitArray = Variable.Observed(Util.ArrayInit(item.SizeAsInt, i => xInit), item).Named("xInit");
                VariableArray<double> x = Variable.Array<double>(item).Named("x");
                x[item] = Variable.GaussianFromMeanAndVariance(0, 100).ForEach(item);
                VariableArray<double> y = Variable.Array<double>(item).Named("y");
                y[item] = Variable.GaussianFromMeanAndVariance(x[item], 1);
                InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
                engine.Compiler.UnrollLoops = true;
                engine.NumberOfIterations = 1;
                x[0].InitialiseTo(xInitArray[0]);
                object xActual = engine.Infer(x);
                Console.WriteLine(xActual);

            });
        }

        [Fact]
        public void InitialiseToObservedDiscreteTest()
        {
            double[] xProbs = new double[] { 0.2, 0.8 };
            double[] yProbs = new double[] { 0.3, 0.7 };
            Variable<int> x = Variable.Discrete(xProbs).Named("x");
            Variable<int> y = Variable.Discrete(yProbs).Named("y");
            Variable.ConstrainEqualRandom(x == y, new Bernoulli(0.6));
            Variable<Discrete> xInit = Variable.New<Discrete>().Named("xInit");
            x.InitialiseTo(xInit);
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());

            Discrete init = new Discrete(0.4, 0.6);
            xInit.ObservedValue = (Discrete)init.Clone();
            Discrete xActual = engine.Infer<Discrete>(x);
            Console.WriteLine("x = {0}", xActual);
            // check that the initializer object has not been modified
            Assert.Equal(init, xInit.ObservedValue);
        }

        [Fact]
        public void InitialiseToObservedArrayTest()
        {
            int n = 1;
            Range item = new Range(n).Named("item");

            VariableArray<double> p = Variable.Array<double>(item).Named("p");
            Beta pPrior = new Beta(2, 3);
            //p[item] = Variable.Beta(2, 3).ForEach(item);
            // this way of defining p triggers a different path in MessageTransform
            p.SetTo(Variable.Random(Distribution<double>.Array(new Beta[] { pPrior })));
            VariableArray<bool> c = Variable.Array<bool>(item).Named("c");
            c[item] = Variable.Bernoulli(p[item]);
            Bernoulli cLike = new Bernoulli(0.2);
            Variable.ConstrainEqualRandom(c[item], cLike);

            Beta[] init = new Beta[n];
            for (int i = 0; i < n; i++)
            {
                init[i] = new Beta(3, 2);
            }
            var pInit = Variable.Observed(Distribution<double>.Array(init)).Named("pInit");
            p.InitialiseTo(pInit);

            Bernoulli cExpected1 = BernoulliFromBetaOp.SampleAverageLogarithm(init[0]) * cLike;
            Beta pMarginal1 = pPrior * BernoulliFromBetaOp.ProbTrueAverageLogarithm(cExpected1);
            Bernoulli cExpected2 = BernoulliFromBetaOp.SampleAverageLogarithm(pMarginal1) * cLike;
            Beta pMarginal2 = pPrior * BernoulliFromBetaOp.ProbTrueAverageLogarithm(cExpected2);
            Bernoulli cExpected3 = BernoulliFromBetaOp.SampleAverageLogarithm(pMarginal2) * cLike;

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            for (int trial = 0; trial < 2; trial++)
            {
                // because of the finalizer code added by SchedulingTransform, we end up doing 1 more iteration than specified
                engine.NumberOfIterations = 1;
                pInit.ObservedValue = Distribution<double>.Array(init);
                Bernoulli[] cActual = engine.Infer<Bernoulli[]>(c);
                Console.WriteLine("c = {0} should be {1}", cActual[0], cExpected2);
                Assert.True(cExpected2.MaxDiff(cActual[0]) < 1e-10);
                engine.NumberOfIterations = 2;
                cActual = engine.Infer<Bernoulli[]>(c);
                Console.WriteLine("c = {0} should be {1}", cActual[0], cExpected3);
                Assert.True(cExpected3.MaxDiff(cActual[0]) < 1e-10);
            }
            // check that the initializer object has not been modified
            Assert.Equal(Distribution<double>.Array(init), pInit.ObservedValue);
        }

        [Fact]
        public void InitialiseToJaggedObservedTest()
        {
            Range outer = new Range(2).Named("outer");
            VariableArray<int> innerSizes = Variable.Constant(new int[] { 3, 4 }, outer).Named("innerSizes");
            Range inner = new Range(innerSizes[outer]).Named("inner");

            Beta pPrior = new Beta(2, 3);
            var p = Variable.Array(Variable.Array<double>(inner), outer).Named("p");
            p[outer][inner] = Variable.Beta(2, 3).ForEach(outer, inner);
            var c = Variable.Array(Variable.Array<bool>(inner), outer).Named("c");
            c[outer][inner] = Variable.Bernoulli(p[outer][inner]);
            Bernoulli cLike = new Bernoulli(0.2);
            Variable.ConstrainEqualRandom(c[outer][inner], cLike);

            Beta[][] init = new Beta[outer.SizeAsInt][];
            for (int i = 0; i < init.Length; i++)
            {
                init[i] = new Beta[innerSizes.ObservedValue[i]];
                for (int j = 0; j < init[i].Length; j++)
                {
                    init[i][j] = new Beta(3, 2);
                }
            }
            var pInit = Variable.Observed(Distribution<double>.Array(init)).Named("pInit");
            p.InitialiseTo(pInit);

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            BernoulliArrayArray cExpected1 = new BernoulliArrayArray(init.Length, i =>
              new BernoulliArray(init[i].Length, j =>
                  BernoulliFromBetaOp.SampleAverageLogarithm(init[i][j]) * cLike
                  ));
            BernoulliArrayArray cExpected2 = new BernoulliArrayArray(init.Length, i =>
              new BernoulliArray(init[i].Length, j =>
                  BernoulliFromBetaOp.SampleAverageLogarithm(pPrior * BernoulliFromBetaOp.ProbTrueAverageLogarithm(cExpected1[i][j])) * cLike
                  ));
            BernoulliArrayArray cExpected3 = new BernoulliArrayArray(init.Length, i =>
              new BernoulliArray(init[i].Length, j =>
                  BernoulliFromBetaOp.SampleAverageLogarithm(pPrior * BernoulliFromBetaOp.ProbTrueAverageLogarithm(cExpected2[i][j])) * cLike
                  ));
            for (int trial = 0; trial < 2; trial++)
            {
                engine.NumberOfIterations = 1;
                object cActual = engine.Infer(c);
                Console.WriteLine(StringUtil.JoinColumns("c = ", cActual, " should be ", cExpected2));
                Assert.True(cExpected2.MaxDiff(cActual) < 1e-10);
                engine.NumberOfIterations = 2;
                cActual = engine.Infer(c);
                Console.WriteLine(StringUtil.JoinColumns("c = ", cActual, " should be ", cExpected3));
                Assert.True(cExpected3.MaxDiff(cActual) < 1e-10);
            }
        }

        [Fact]
        public void InitialiseToJaggedConstantTest()
        {
            InitialiseToJaggedConstant(initialiseByElement: false);
        }

        [Fact]
        public void InitialiseToJaggedConstantTest2()
        {
            InitialiseToJaggedConstant(initialiseByElement: true);
        }

        private void InitialiseToJaggedConstant(bool initialiseByElement)
        {
            Range outer = new Range(2).Named("outer");
            VariableArray<int> innerSizes = Variable.Constant(new int[] { 3, 4 }, outer).Named("innerSizes");
            Range inner = new Range(innerSizes[outer]).Named("inner");

            Beta pPrior = new Beta(2, 3);
            var p = Variable.Array(Variable.Array<double>(inner), outer).Named("p");
            p[outer][inner] = Variable.Beta(2, 3).ForEach(outer, inner);
            var c = Variable.Array(Variable.Array<bool>(inner), outer).Named("c");
            c[outer][inner] = Variable.Bernoulli(p[outer][inner]);
            Bernoulli cLike = new Bernoulli(0.2);
            Variable.ConstrainEqualRandom(c[outer][inner], cLike);

            Beta[][] init = new Beta[outer.SizeAsInt][];
            for (int i = 0; i < init.Length; i++)
            {
                init[i] = new Beta[innerSizes.ObservedValue[i]];
                for (int j = 0; j < init[i].Length; j++)
                {
                    init[i][j] = new Beta(3, 2);
                }
            }

            if (initialiseByElement)
            {
                var pInit = Variable.Constant(init, outer, inner).Named("pInit");
                p[outer][inner].InitialiseTo(pInit[outer][inner]);
            }
            else
            {
                var pInit = Variable.Constant(Distribution<double>.Array(init)).Named("pInit");
                p.InitialiseTo(pInit);
            }

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            BernoulliArrayArray cExpected1 = new BernoulliArrayArray(init.Length, i =>
              new BernoulliArray(init[i].Length, j =>
                  BernoulliFromBetaOp.SampleAverageLogarithm(init[i][j]) * cLike
              ));
            BernoulliArrayArray cExpected2 = new BernoulliArrayArray(init.Length, i =>
              new BernoulliArray(init[i].Length, j =>
                  BernoulliFromBetaOp.SampleAverageLogarithm(pPrior * BernoulliFromBetaOp.ProbTrueAverageLogarithm(cExpected1[i][j])) * cLike
              ));
            BernoulliArrayArray cExpected3 = new BernoulliArrayArray(init.Length, i =>
              new BernoulliArray(init[i].Length, j =>
                  BernoulliFromBetaOp.SampleAverageLogarithm(pPrior * BernoulliFromBetaOp.ProbTrueAverageLogarithm(cExpected2[i][j])) * cLike
              ));
            for (int trial = 0; trial < 2; trial++)
            {
                engine.NumberOfIterations = 1;
                object cActual = engine.Infer(c);
                Console.WriteLine(StringUtil.JoinColumns("c = ", cActual, " should be ", cExpected2));
                Assert.True(cExpected2.MaxDiff(cActual) < 1e-10);
                engine.NumberOfIterations = 2;
                cActual = engine.Infer(c);
                Console.WriteLine(StringUtil.JoinColumns("c = ", cActual, " should be ", cExpected3));
                Assert.True(cExpected3.MaxDiff(cActual) < 1e-10);
            }
        }

        [Fact]
        public void InitialiseToJaggedEnterTest()
        {
            int dimU = 2;
            int numD = 2;
            int numN = 2;
            Range D = new Range(numD).Named("D");
            Range N = new Range(numN).Named("N");
            var U = Variable.Array(Variable.Array<int>(N), D).Named("U");
            U[D][N] = Variable.DiscreteUniform(dimU).ForEach(D, N);
            var b = Variable.Bernoulli(0.1).Named("b");
            using (Variable.If(b))
            {
                Variable.ConstrainEqualRandom(U[D][N], Discrete.Uniform(dimU));
            }

            var initU = new Discrete[numD][];
            for (int i = 0; i < numD; i++)
            {
                initU[i] = new Discrete[numN];
                for (int j = 0; j < numN; j++)
                    initU[i][j] = Discrete.PointMass(Rand.Int(dimU), dimU);
            }
            var uInit = Variable.Observed(Distribution<int>.Array(initU)).Named("uInit");
            //var uInit = Variable.Observed((DiscreteArrayArray)Distribution<int>.Array(initU)).Named("uInit");
            U.InitialiseTo(uInit);
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(U));
        }

        [Fact]
        [Trait("Category", "BadTest")]
        public void InitialiseToConstantForEachTest()
        {
            int n = 1;
            Range item = new Range(n).Named("item");

            Beta pPrior = new Beta(2, 3);
            Beta pInit = new Beta(3, 2);
            Variable<double> p;
            Variable<bool> c;
            Bernoulli cLike = new Bernoulli(0.2);
            using (Variable.ForEach(item))
            {
                p = Variable.Beta(2, 3);
                p.Name = "p";
                p.InitialiseTo(pInit);
                c = Variable.Bernoulli(p);
                c.Name = "c";
                Variable.ConstrainEqualRandom(c, cLike);
            }

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            var ca = engine.GetCompiledInferenceAlgorithm(c);
            ca.Execute(1);
            Bernoulli cExpected = BernoulliFromBetaOp.SampleAverageLogarithm(pInit) * cLike;
            DistributionArray<Bernoulli> cActual = ca.Marginal<DistributionArray<Bernoulli>>("c");
            Assert.True(cExpected.MaxDiff(cActual[0]) < 1e-10);
        }

        [Fact]
        public void InitialiseToIgnoredTest()
        {
            int n = 1;
            Range item = new Range(n).Named("item");

            VariableArray<bool> b = Variable.Array<bool>(item).Named("b");
            b[item] = Variable.Bernoulli(0.2).ForEach(item);
            VariableArray<bool> c = Variable.Array<bool>(item).Named("c");
            c[item] = !b[item];

            Bernoulli[] init = new Bernoulli[n];
            for (int i = 0; i < n; i++)
            {
                init[i] = new Bernoulli(0.1);
            }
            var bInit = Variable.Observed(Distribution<bool>.Array(init)).Named("bInit");
            b.InitialiseTo(bInit);

            InferenceEngine engine = new InferenceEngine();
            var ca = engine.GetCompiledInferenceAlgorithm(c);
            bInit.ObservedValue = Distribution<bool>.Array(init);
            ca.Execute(1);
            Bernoulli[] cActual = Distribution.ToArray<Bernoulli[]>(ca.Marginal("c"));
            Assert.True(System.Math.Abs(cActual[0].GetProbTrue() - 0.8) < 1e-10);
        }

        [Fact]
        public void InitialiseToJaggedIgnoredTest()
        {
            Range outer = new Range(2).Named("outer");
            VariableArray<int> innerSizes = Variable.Constant(new int[] { 3, 4 }, outer).Named("innerSizes");
            Range inner = new Range(innerSizes[outer]).Named("inner");

            var b = Variable.Array(Variable.Array<bool>(inner), outer).Named("b");
            b[outer][inner] = Variable.Bernoulli(0.2).ForEach(outer, inner);
            var c = Variable.Array(Variable.Array<bool>(inner), outer).Named("c");
            c[outer][inner] = !b[outer][inner];

            Bernoulli[][] init = new Bernoulli[outer.SizeAsInt][];
            for (int i = 0; i < init.Length; i++)
            {
                init[i] = new Bernoulli[innerSizes.ObservedValue[i]];
                for (int j = 0; j < init[i].Length; j++)
                {
                    init[i][j] = new Bernoulli(0.1);
                }
            }
            var bInit = Variable.Observed(Distribution<bool>.Array(init)).Named("bInit");
            b.InitialiseTo(bInit);

            InferenceEngine engine = new InferenceEngine();
            var ca = engine.GetCompiledInferenceAlgorithm(c);
            ca.Execute(1);
            Bernoulli[][] cActual = Distribution.ToArray<Bernoulli[][]>(ca.Marginal("c"));
            for (int i = 0; i < cActual.Length; i++)
            {
                for (int j = 0; j < cActual[i].Length; j++)
                {
                    Assert.True(System.Math.Abs(cActual[i][j].GetProbTrue() - 0.8) < 1e-10);
                }
            }
        }

        [Fact]
        [Trait("Category", "BadTest")]
        public void InitialiseToForEachIgnoredTest()
        {
            int n = 1;
            Range item = new Range(n).Named("item");

            Variable<bool> b, c;
            using (Variable.ForEach(item))
            {
                b = Variable.Bernoulli(0.2);
                b.Name = "b";
                b.InitialiseTo(new Bernoulli(0.1));
                c = !b;
                c.Name = "c";
            }

            InferenceEngine engine = new InferenceEngine();
            var ca = engine.GetCompiledInferenceAlgorithm(c);
            ca.Execute(1);
            DistributionArray<Bernoulli> cActual = ca.Marginal<DistributionArray<Bernoulli>>("c");
            Assert.True(System.Math.Abs(cActual[0].GetProbTrue() - 0.8) < 1e-10);
        }

        [Fact]
        public void InferIfLocalTest()
        {
            Variable<bool> b = Variable.Bernoulli(0.3);
            Variable<bool> local;
            using (Variable.If(b))
            {
                local = Variable.Bernoulli(0.9);
                Variable.ConstrainEqualRandom(local, new Bernoulli(0.4));
            }
            InferenceEngine engine = new InferenceEngine();
            Bernoulli actual = engine.Infer<Bernoulli>(local);
            double z = 0.9 * 0.4 + (1 - 0.9) * (1 - 0.4);
            Bernoulli expected = new Bernoulli(0.9 * 0.4 / z);
            Assert.True(expected.MaxDiff(actual) < 1e-10);
        }

        [Fact]
        [Trait("Category", "BadTest")]
        public void InferForEachLocalTest()
        {
            Variable<bool> local;
            Range item = new Range(1);
            using (Variable.ForEach(item))
            {
                local = Variable.Bernoulli(0.9);
            }
            InferenceEngine engine = new InferenceEngine();
            DistributionArray<Bernoulli> actual = engine.Infer<DistributionArray<Bernoulli>>(local);
            Console.WriteLine(actual);
            IDistribution<bool[]> expected = Distribution<bool>.Array(new Bernoulli[] { new Bernoulli(0.9) });
            Assert.True(expected.MaxDiff(actual) < 1e-10);
        }

        [Fact]
        [Trait("Category", "BadTest")]
        public void InferForEachLocal2Test()
        {
            Variable<bool> local;
            Range item = new Range(1);
            Range item2 = new Range(1);
            using (Variable.ForEach(item))
            {
                using (Variable.ForEach(item2))
                {
                    local = Variable.Bernoulli(0.9);
                }
            }
            InferenceEngine engine = new InferenceEngine();
            object actual = engine.Infer(local);
            Console.WriteLine(actual);
            IDistribution<bool[][]> expected = Distribution<bool[]>.Array(1, delegate (int i) { return new BernoulliArray(new Bernoulli[] { new Bernoulli(0.9) }); });
            Assert.True(expected.MaxDiff(actual) < 1e-10);
        }

        internal void VmpSchedulingTest()
        {
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.NumberOfIterations = 1000;
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(FactorAnalysisAlphaModel);
            ca.Execute(engine.NumberOfIterations);
            Console.WriteLine(ca.Marginal("W"));
        }

        internal void VmpSchedulingTest2()
        {
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.NumberOfIterations = 1000;
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            Bernoulli[] cprior = new Bernoulli[] { new Bernoulli(0.5) };
            var ca = engine.Compiler.Compile(FactorAnalysisModel2, cprior);
            ca.Execute(engine.NumberOfIterations);
            Console.WriteLine(ca.Marginal("W"));
        }

        internal void VmpSchedulingTest3()
        {
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.NumberOfIterations = 1000;
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(FactorAnalysisAlphaModel2);
            ca.Execute(engine.NumberOfIterations);
            Console.WriteLine(ca.Marginal("W"));
        }

        internal void FactorAnalysisModel()
        {
            double x = Factor.Gaussian(0.0, 1.0);
            double W;
            // C controls the prior on W
            bool C;
            C = Factor.Bernoulli(0.5);
            if (C)
                W = Factor.Gaussian(0, 1);
            else
                W = Factor.Gaussian(0, 0.001);

            double xTimesW = Factor.Product(x, W);

            double data = 2.2;
            data = Factor.Gaussian(xTimesW, 1.0);

            Gaussian init = new Gaussian(0.0, 1.0);
            Attrib.InitialiseTo(x, init);
            Attrib.InitialiseTo(W, init);
            InferNet.Infer(x, nameof(x));
            InferNet.Infer(W, nameof(W));
        }

        private void FactorAnalysisModel2(Bernoulli[] cprior)
        {
            int N_n_ = 1;
            int N_k_ = 1;
            int N_d_ = 1;
            double[,] x = new double[N_n_, N_k_];
            Gaussian[,] initx = new Gaussian[N_n_, N_k_];
            for (int N = 0; N < N_n_; N++)
            {
                for (int K = 0; K < N_k_; K++)
                {
                    x[N, K] = Factor.Gaussian(0.0, 1.0);
                    initx[N, K] = new Gaussian(0.0, 1.0);
                }
            }
            Attrib.InitialiseTo(x, Distribution<double>.Array(initx));
            double[,] W = new double[N_k_, N_d_];
            Gaussian[,] initw = new Gaussian[N_k_, N_d_];
            // C controls the prior on W
            bool[,] C = new bool[N_k_, N_d_];
            for (int K2 = 0; K2 < N_k_; K2++)
            {
                for (int D = 0; D < N_d_; D++)
                {
                    initw[K2, D] = new Gaussian(0.0, 1.0);
                    C[K2, D] = Factor.Random<bool>(cprior[K2]);
                    if (C[K2, D])
                        W[K2, D] = Factor.Gaussian(0, 1);
                    else
                        W[K2, D] = Factor.Gaussian(0, 0.001);
                }
            }
            Attrib.InitialiseTo(W, Distribution<double>.Array(initw));

            double[,] xTimesW = new double[N_n_, N_d_];
            xTimesW = Factor.MatrixMultiply(x, W);

            double[,] data = new double[N_n_, N_d_];
            for (int N1 = 0; N1 < N_n_; N1++)
            {
                for (int D2 = 0; D2 < N_d_; D2++)
                {
                    data[N1, D2] = 2.2;
                    data[N1, D2] = Factor.Gaussian(xTimesW[N1, D2], 1.0);
                }
            }

            InferNet.Infer(x, nameof(x));
            InferNet.Infer(W, nameof(W));
        }

        private void FactorAnalysisAlphaModel2()
        {
            int N_n_ = 1;
            int N_k_ = 1;
            int N_d_ = 1;
            double[,] x = new double[N_n_, N_k_];
            Gaussian[,] initx = new Gaussian[N_n_, N_k_];
            for (int N = 0; N < N_n_; N++)
            {
                for (int K = 0; K < N_k_; K++)
                {
                    x[N, K] = Factor.Gaussian(0.0, 1.0);
                    initx[N, K] = new Gaussian(0.0, 1.0);
                }
            }
            Attrib.InitialiseTo(x, Distribution<double>.Array(initx));
            double[,] W = new double[N_k_, N_d_];
            Gaussian[,] initw = new Gaussian[N_k_, N_d_];

            double[] Alpha = new double[N_k_];
            Gamma constGamma0 = new Gamma(0.001, 0.001);
            for (int K1 = 0; K1 < N_k_; K1++)
            {
                Alpha[K1] = Factor.Random<double>(constGamma0);
            }

            // C controls the prior on W
            bool[,] C = new bool[N_k_, N_d_];
            for (int K2 = 0; K2 < N_k_; K2++)
            {
                for (int D = 0; D < N_d_; D++)
                {
                    initw[K2, D] = new Gaussian(0.0, 1.0);
                    C[K2, D] = Factor.Bernoulli(0.5);
                    if (C[K2, D])
                        W[K2, D] = Factor.Gaussian(0, Alpha[K2]);
                    else
                        W[K2, D] = Factor.Gaussian(0, 0.001);
                }
            }
            Attrib.InitialiseTo(W, Distribution<double>.Array(initw));

            double[,] xTimesW = new double[N_n_, N_d_];
            xTimesW = Factor.MatrixMultiply(x, W);

            double[,] data = new double[N_n_, N_d_];
            for (int N1 = 0; N1 < N_n_; N1++)
            {
                for (int D2 = 0; D2 < N_d_; D2++)
                {
                    data[N1, D2] = 2.2;
                    data[N1, D2] = Factor.Gaussian(xTimesW[N1, D2], 1.0);
                }
            }

            InferNet.Infer(x, nameof(x));
            InferNet.Infer(W, nameof(W));
        }

        private void FactorAnalysisAlphaModel()
        {
            double x = Factor.Gaussian(0.0, 1.0);
            double W;

            double alpha0, alpha1;
            alpha0 = Factor.Random(new Gamma(0.1, 0.1));
            alpha1 = Factor.Random(new Gamma(0.1, 0.1));

            // C controls the prior on W
            bool C;
            C = Factor.Bernoulli(0.5);
            if (C)
                W = Factor.Gaussian(0, alpha1);
            else
                W = Factor.Gaussian(0, alpha0);

            double xTimesW = Factor.Product(x, W);

            double data = 2.2;
            data = Factor.Gaussian(xTimesW, 1.0);

            Gaussian init = new Gaussian(0.0, 1.0);
            Attrib.InitialiseTo(x, init);
            Attrib.InitialiseTo(W, init);
            InferNet.Infer(x, nameof(x));
            InferNet.Infer(W, nameof(W));
        }

        [Fact]
        public void ImproperMixtureTest()
        {
            // in this model, the message to x from Gate.Exit is improper.
            Variable<double> x = Variable.New<double>();
            Variable<bool> c = Variable.Bernoulli(0.5);
            using (Variable.If(c))
            {
                x.SetTo(Variable.GaussianFromMeanAndVariance(0, 0));
            }
            using (Variable.IfNot(c))
            {
                x.SetTo(Variable.GaussianFromMeanAndVariance(1, 0));
            }
            Variable.ConstrainEqualRandom(x, new Gaussian(0.6, 0.1));
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(x));
        }


        [Fact]
        public void PriorArrayTest3()
        {
            Range item = new Range(2).Named("item");
            VariableArray<Vector> prior = Variable.Array<Vector>(item).Named("prior");
            VariableArray<int> x = Variable.Array<int>(item);
            x[item] = Variable.Discrete(prior[item]).Named("x");

            Vector[] obsPrior = new Vector[] { Vector.FromArray(0.1, 0.9), Vector.FromArray(0.2, 0.8) };
            prior.ObservedValue = obsPrior;
            InferenceEngine engine = new InferenceEngine();
            DistributionArray<Discrete> xActual = engine.Infer<DistributionArray<Discrete>>(x);
            Console.WriteLine(xActual);
            Discrete[] xExpected = new Discrete[item.SizeAsInt];
            for (int i = 0; i < xExpected.Length; i++)
            {
                xExpected[i] = new Discrete(obsPrior[i]);
            }
            Assert.True(Distribution<int>.Array(xExpected).MaxDiff(xActual) < 1e-10);
        }

        [Fact]
        public void PriorArrayTest2()
        {
            Range item = new Range(2).Named("item");
            VariableArray<Gaussian> prior = Variable.Array<Gaussian>(item).Named("prior");
            VariableArray<double> mean = Variable.Array<double>(item);
            mean[item] = Variable.Random<double, Gaussian>(prior[item]).Named("mean");

            Gaussian[] obsPrior = new Gaussian[item.SizeAsInt];
            for (int i = 0; i < obsPrior.Length; i++)
            {
                obsPrior[i] = new Gaussian(0, 1);
            }
            prior.ObservedValue = obsPrior;
            InferenceEngine engine = new InferenceEngine();
            object meanActual = engine.Infer(mean);
            Console.WriteLine(meanActual);
            Assert.True(Distribution<double>.Array(obsPrior).MaxDiff(meanActual) < 1e-10);
        }

        [Fact]
        public void PriorArrayTest()
        {
            Range item = new Range(2).Named("item");
            VariableArray<Gaussian> prior = Variable.Array<Gaussian>(item).Named("prior");
            VariableArray<double> mean = Variable.Array<double>(item).Named("mean");
            mean[item] = Variable.Random<double, Gaussian>(prior[item]);

            Gaussian[] obsPrior = Util.ArrayInit(item.SizeAsInt, i => new Gaussian(0, 1));
            prior.ObservedValue = obsPrior;
            InferenceEngine engine = new InferenceEngine();
            DistributionArray<Gaussian> meanActual = engine.Infer<DistributionArray<Gaussian>>(mean);
            Console.WriteLine(meanActual);
            IDistribution<double[]> meanExpected = Distribution<double>.Array(obsPrior);
            Assert.True(meanActual.MaxDiff(meanExpected) < 1e-10);
        }

        /// <summary>
        /// Test that VMP fails on a deterministic loop.
        /// </summary>
        // Currently fails because it does not throw an exception as it should.
        [Fact]
        [Trait("Category", "OpenBug")]
        public void CyclicProductTest()
        {
            Variable<double> a = Variable.GaussianFromMeanAndVariance(0.1, 100).Named("a");
            Variable<double> b = Variable.GaussianFromMeanAndVariance(0.2, 100).Named("b");
            Variable<double> ab = (a * b).Named("ab");
            Variable<double> c = Variable.GaussianFromMeanAndVariance(0.3, 100).Named("c");
            Variable<double> abc = (ab * c).Named("abc");
            Variable<double> d = Variable.GaussianFromMeanAndVariance(0.4, 100).Named("d");
            Variable<double> abd = (ab * d).Named("abd");
            Variable<double> abcabd = (abc * abd).Named("abcabd");
            Variable.ConstrainEqualRandom(abcabd, Gaussian.FromMeanAndVariance(4.0, 1.0));
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Assert.Throws<InferRuntimeException>(() =>
            {
                Gaussian abcabdMarginal = engine.Infer<Gaussian>(abcabd);
                Console.WriteLine("abcabd = " + abcabdMarginal);
            });
        }

        [Fact]
        public void ConvertArrayTest()
        {
            Range item = new Range(4).Named("item");
            VariableArray<int> x = Variable.Array<int>(item).Named("x");
            x[item] = Variable.Discrete(0.1, 0.3, 0.6).ForEach(item);
            InferenceEngine engine = new InferenceEngine();
            Discrete[] xActual = engine.Infer<Discrete[]>(x);
            Discrete xPrior = new Discrete(0.1, 0.3, 0.6);
            for (int i = 0; i < xActual.Length; i++)
            {
                Assert.True(xPrior.MaxDiff(xActual[i]) < 1e-10);
            }
        }

        [Fact]
        public void ConvertArrayTest2()
        {
            Range item = new Range(4).Named("item");
            VariableArray<int> x = Variable.Array<int>(item).Named("x");
            x[item] = Variable.Discrete(0.1, 0.3, 0.6).ForEach(item);
            InferenceEngine engine = new InferenceEngine();
            IGeneratedAlgorithm gen = engine.GetCompiledInferenceAlgorithm(x);
            gen.Execute(10);
            Discrete[] xActual = gen.Marginal<Discrete[]>(x.NameInGeneratedCode);
            Discrete xPrior = new Discrete(0.1, 0.3, 0.6);
            for (int i = 0; i < xActual.Length; i++)
            {
                Assert.True(xPrior.MaxDiff(xActual[i]) < 1e-10);
            }
        }

        [Fact]
        public void ObservedEvidenceTest()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<bool> b = Variable.Bernoulli(0.1).Named("b");
            b.ObservedValue = true;
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            double evActual = System.Math.Exp(engine.Infer<Bernoulli>(evidence).LogOdds);
            double evExpected = 0.1;
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.Equal(evExpected, evActual, 1e-10);
        }

        [Fact]
        public void InferObservedGammaArrayTest()
        {
            InferObservedGammaArray(false);
            InferObservedGammaArray(true);
        }

        private void InferObservedGammaArray(bool readOnly)
        {
            Range outer = new Range(1).Named("outer");
            Range inner = new Range(1).Named("inner");
            var array = Variable.Array(Variable.Array<double>(inner), outer).Named("array");
            array[outer][inner] = Variable.GammaFromShapeAndRate(1, 1).ForEach(outer, inner);
            array.ObservedValue = new double[][] { new double[] { 2 } };
            array.IsReadOnly = readOnly;

            InferenceEngine engine = new InferenceEngine();
            var bActual = engine.Infer<IReadOnlyList<IReadOnlyList<Gamma>>>(array);
            var bExpected = Distribution<double>.Array(array.ObservedValue.Select(a => a.Select(x => Gamma.PointMass(x)).ToArray()).ToArray());
            Console.WriteLine("b = {0} should be {1}", bActual, bExpected);
            Assert.Equal(bActual.ToString(), bExpected.ToString());
        }

        [Fact]
        public void InferObservedDiscreteTest()
        {
            foreach (bool readOnly in new[] { true, false, true })
            {
                foreach (bool useCompiledAlgorithm in new[] { false, true })
                {
                    InferObservedDiscrete(readOnly, useCompiledAlgorithm);
                }
            }
        }

        private void InferObservedDiscrete(bool readOnly, bool useCompiledAlgorithm)
        {
            Variable<int> b = Variable.Discrete(0.1, 0.2, 0.3, 0.4).Named("b");
            b.ObservedValue = 2;
            b.IsReadOnly = readOnly;
            b.AddAttribute(QueryTypes.Marginal);
            b.AddAttribute(QueryTypes.MarginalDividedByPrior);

            InferenceEngine engine = new InferenceEngine();
            Discrete bActual;
            Discrete marginalDividedByPrior;
            if (useCompiledAlgorithm)
            {
                IGeneratedAlgorithm gen = engine.GetCompiledInferenceAlgorithm(b);
                gen.Execute(10);
                bActual = gen.Marginal<Discrete>(b.NameInGeneratedCode);
                marginalDividedByPrior = gen.Marginal<Discrete>(b.NameInGeneratedCode, QueryTypes.MarginalDividedByPrior.Name);
            }
            else
            {
                bActual = engine.Infer<Discrete>(b);
                marginalDividedByPrior = engine.Infer<Discrete>(b, QueryTypes.MarginalDividedByPrior);
            }
            Discrete bExpected = Discrete.PointMass(b.ObservedValue, 4);
            Discrete marginalDividedByPriorExpected = Discrete.Uniform(4);
            //Console.WriteLine($"b = {bActual} should be {bExpected}");
            //Console.WriteLine($"marginalDividedByPrior = {marginalDividedByPrior} should be {marginalDividedByPriorExpected}");
            Assert.Equal(bExpected, bActual);
            if (!readOnly)
                Assert.Equal(marginalDividedByPriorExpected, marginalDividedByPrior);
        }

        [Fact]
        public void InferObservedTest4()
        {
            Variable<bool> b = Variable.Observed(false).Named("b");

            InferenceEngine engine = new InferenceEngine();
            Bernoulli bActual = engine.Infer<Bernoulli>(b);
            Bernoulli bExpected = Bernoulli.PointMass(b.ObservedValue);
            //Console.WriteLine("b = {0} should be {1}", bActual, bExpected);
            Assert.Equal(bActual, bExpected);
        }

        readonly IAlgorithm[] algorithms = new IAlgorithm[] {
            new ExpectationPropagation(),
            new VariationalMessagePassing(),
            new GibbsSampling(),
        };

        [Fact]
        public void InferConstantTest2()
        {
            foreach (var algorithm in algorithms)
            {
                Variable<bool> b = Variable.Constant(false).Named("b");

                InferenceEngine engine = new InferenceEngine(algorithm);
                Bernoulli bActual = engine.Infer<Bernoulli>(b);
                Bernoulli bExpected = Bernoulli.PointMass(b.ObservedValue);
                //Console.WriteLine("b = {0} should be {1}", bActual, bExpected);
                Assert.Equal(bExpected, bActual);
            }
        }

        [Fact]
        public void InferConstantTest4()
        {
            foreach (var algorithm in algorithms)
            {
                Variable<int> b = Variable.Constant(4).Named("b");
                b.AddAttribute(new MarginalPrototype(Discrete.Uniform(6)));

                InferenceEngine engine = new InferenceEngine(algorithm);
                Discrete bActual = engine.Infer<Discrete>(b);
                Discrete bExpected = Discrete.PointMass(b.ObservedValue, 6);
                //Console.WriteLine("b = {0} should be {1}", bActual, bExpected);
                Assert.Equal(bExpected, bActual);
            }
        }

        [Fact]
        public void InferObservedTest5()
        {
            Variable<int> b = Variable.Observed(4).Named("b");
            b.AddAttribute(new MarginalPrototype(Discrete.Uniform(6)));

            InferenceEngine engine = new InferenceEngine();
            Discrete bActual = engine.Infer<Discrete>(b);
            Discrete bExpected = Discrete.PointMass(b.ObservedValue, 6);
            //Console.WriteLine("b = {0} should be {1}", bActual, bExpected);
            Assert.Equal(bActual, bExpected);
        }

        [Fact]
        public void InferObservedTest()
        {
            foreach (var algorithm in algorithms)
            {
                Variable<bool> b = Variable.Bernoulli(0.1).Named("b");
                b.ObservedValue = true;

                InferenceEngine engine = new InferenceEngine(algorithm);
                Bernoulli bActual = engine.Infer<Bernoulli>(b);
                Bernoulli bExpected = Bernoulli.PointMass(b.ObservedValue);
                //Console.WriteLine("b = {0} should be {1}", bActual, bExpected);
                Assert.Equal(bActual, bExpected);
            }
        }

        [Fact]
        public void InferConstantTest()
        {
            foreach (var algorithm in algorithms)
            {
                Variable<bool> b = Variable.Bernoulli(0.1).Named("b");
                b.ObservedValue = true;
                b.IsReadOnly = true;

                InferenceEngine engine = new InferenceEngine(algorithm);
                Bernoulli bActual = engine.Infer<Bernoulli>(b);
                Bernoulli bExpected = Bernoulli.PointMass(b.ObservedValue);
                //Console.WriteLine("b = {0} should be {1}", bActual, bExpected);
                Assert.Equal(bActual, bExpected);
            }
        }

        [Fact]
        public void InferObservedTest2()
        {
            foreach (var algorithm in algorithms)
            {
                Variable<bool> b = Variable.Bernoulli(0.1).Named("b");
                b.ObservedValue = true;

                InferenceEngine engine = new InferenceEngine(algorithm);
                IGeneratedAlgorithm gen = engine.GetCompiledInferenceAlgorithm(b);
                gen.Execute(10);
                Bernoulli bActual = gen.Marginal<Bernoulli>(b.NameInGeneratedCode);
                Bernoulli bExpected = Bernoulli.PointMass(b.ObservedValue);
                //Console.WriteLine("b = {0} should be {1}", bActual, bExpected);
                Assert.Equal(bActual, bExpected);
            }
        }

        [Fact]
        public void InferObservedTest3()
        {
            Variable<bool> b = Variable.Bernoulli(0.1).Named("b");
            b.ObservedValue = true;
            Variable<bool> x = b & Variable.Bernoulli(0.2).Named("y");
            x.Name = "x";

            InferenceEngine engine = new InferenceEngine();
            IGeneratedAlgorithm gen = engine.GetCompiledInferenceAlgorithm(b, x);
            gen.Execute(10);
            Bernoulli bActual = gen.Marginal<Bernoulli>(b.NameInGeneratedCode);
            Bernoulli bExpected = Bernoulli.PointMass(b.ObservedValue);
            //Console.WriteLine("b = {0} should be {1}", bActual, bExpected);
            Assert.Equal(bActual, bExpected);
        }

        [Fact]
        public void InferConstantTest3()
        {
            Variable<bool> b = Variable.Bernoulli(0.1).Named("b");
            b.ObservedValue = true;
            b.IsReadOnly = true;
            Variable<bool> x = b & Variable.Bernoulli(0.2).Named("y");
            x.Name = "x";

            InferenceEngine engine = new InferenceEngine();
            IGeneratedAlgorithm gen = engine.GetCompiledInferenceAlgorithm(b, x);
            gen.Execute(10);
            Bernoulli bActual = gen.Marginal<Bernoulli>(b.NameInGeneratedCode);
            Bernoulli bExpected = Bernoulli.PointMass(b.ObservedValue);
            //Console.WriteLine("b = {0} should be {1}", bActual, bExpected);
            Assert.Equal(bActual, bExpected);
        }

        [Fact]
        public void InferObservedArrayTest()
        {
            foreach (var readOnly in new[] { false, true })
            {
                foreach (var algorithm in algorithms)
                {
                    Range item = new Range(2);
                    VariableArray<bool> b = Variable.Array<bool>(item).Named("b");
                    b[item] = Variable.Bernoulli(0.1).ForEach(item);
                    b.ObservedValue = new bool[] { true, false };
                    b.IsReadOnly = readOnly;

                    InferenceEngine engine = new InferenceEngine(algorithm);
                    Bernoulli[] bActual;
                    foreach (var useGeneratedAlgorithm in new[] { false, true })
                    {
                        if (useGeneratedAlgorithm)
                        {
                            IGeneratedAlgorithm gen = engine.GetCompiledInferenceAlgorithm(b);
                            gen.Execute(1);
                            bActual = gen.Marginal<Bernoulli[]>(b.NameInGeneratedCode);
                        }
                        else bActual = engine.Infer<Bernoulli[]>(b);
                        //Console.WriteLine("b = {0}", StringUtil.ToString(bActual));
                        for (int i = 0; i < bActual.Length; i++)
                        {
                            Assert.Equal(b.ObservedValue[i], bActual[i].Point);
                        }
                    }
                }
            }
        }

        [Fact]
        public void InferObservedSubarrayTest()
        {
            foreach (var readOnly in new[] { false, true })
            {
                foreach (var algorithm in algorithms)
                {
                    Range item = new Range(2);
                    VariableArray<bool> array = Variable.Array<bool>(item).Named("array");
                    array[item] = Variable.Bernoulli(0.1).ForEach(item);
                    VariableArray<int> indices = Variable.Observed(new int[] { 1 }).Named("indices");
                    VariableArray<bool> b = Variable.Subarray(array, indices).Named("b");
                    b.ObservedValue = new bool[] { false };
                    b.IsReadOnly = readOnly;

                    InferenceEngine engine = new InferenceEngine(algorithm);
                    Bernoulli[] bActual;
                    foreach (var useGeneratedAlgorithm in new[] { false, true })
                    {
                        if (useGeneratedAlgorithm)
                        {
                            IGeneratedAlgorithm gen = engine.GetCompiledInferenceAlgorithm(b);
                            gen.Execute(1);
                            bActual = gen.Marginal<Bernoulli[]>(b.NameInGeneratedCode);
                        }
                        else bActual = engine.Infer<Bernoulli[]>(b);
                        //Console.WriteLine("b = {0}", StringUtil.ToString(bActual));
                        for (int i = 0; i < bActual.Length; i++)
                        {
                            Assert.Equal(b.ObservedValue[i], bActual[i].Point);
                        }
                    }
                }
            }
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void InferObservedArray3DTest()
        {
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
            double[,,] array3D = new double[,,] { { { 2.5 } } };
            var ca = engine.Compiler.Compile(Array3DModel, array3D);
            ca.Execute(1);
            var actual = ca.Marginal<PointMass<double[,,]>>("array3D");
            Assert.Equal(array3D[0, 0, 0], actual.Point[0, 0, 0]);
        }

        private void Array3DModel(double[,,] array3D)
        {
            InferNet.Infer(array3D, nameof(array3D));
        }

        [Fact]
        public void InferUniformDiscreteFromDirichletTest()
        {
            int numStates = 3;
            Range state = new Range(numStates).Named("state");
            Variable<Vector> Xprior = Variable.DirichletUniform(state).Named("Xprior");
            var X = Variable.Discrete(Xprior).Named("X");
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(engine.Infer(X));
        }

        [Fact]
        public void InferDirichletArrayTest()
        {
            int[] data = new int[] { 1, 5, 5, 8, 3 };
            int maxV = 9;
            Range item = new Range(data.Length).Named("item");

            VariableArray<Vector> phi = Variable.Array<Vector>(item).Named("phi");
            phi[item] = Variable.DirichletUniform(maxV).ForEach(item);
            //phi[item] = Variable.DirichletSymmetric(maxV, 1.0).ForEach(item);
            var x = Variable.Array<int>(item).Named("x");
            using (Variable.ForEach(item))
            {
                x[item] = Variable.Discrete(phi[item]);
            }
            Variable.ConstrainEqual(x, data);

            InferenceEngine engine = new InferenceEngine(); //new VariationalMessagePassing());
            var phiExpected = new DirichletArray(data.Length, i =>
                {
                    Dirichlet d = Dirichlet.Uniform(maxV);
                    d.PseudoCount[data[i]]++;
                    d.TotalCount++;
                    return d;
                });
            var phiActual = engine.Infer<IDistribution<Vector[]>>(phi);
            //Console.WriteLine(StringUtil.JoinColumns("phi = ", phiActual, " should be ", phiExpected));
            Assert.True(phiExpected.MaxDiff(phiActual) < 1e-8);
        }

        [Fact]
        public void InferUniformBernoulliFromBetaTest()
        {
            Variable<Beta> prior = Variable<Beta>.Factor(Beta.Uniform).Named("prior");
            Variable<double> p = Variable<double>.Random(prior).Named("p");
            var b = Variable.Bernoulli(p).Named("b");
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(engine.Infer(p));
            Console.WriteLine(engine.Infer(b));
        }

        [Fact]
        public void InferUniformBernoulliFromBetaTest2()
        {
            Variable<Beta> prior = Variable<Beta>.Factor(Beta.Uniform).Named("prior");
            Variable<double> p = Variable<double>.Random(prior).Named("p");
            var b = Variable.Bernoulli(p).Named("b");
            Variable.ConstrainEqualRandom(b, new Bernoulli(0.1));
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(engine.Infer(p));
            Console.WriteLine(engine.Infer(b));
        }

        [Fact]
        public void InferUniformGateEnterPartialTest()
        {
            // This model is specially constructed so that the message down to Gate.EnterPartial is uniform iff the message from Gate.EnterPartial is uniform.
            Variable<Beta> prior = Variable<Beta>.Factor(Beta.Uniform).Named("prior");
            Variable<double> p = Variable<double>.Random(prior).Named("p");
            Variable<bool> b = Variable.Bernoulli(p).Named("b");
            Variable<bool> x = Variable.New<bool>().Named("x");
            using (Variable.If(b))
            {
                x.SetTo(Variable.Bernoulli(p));
            }
            using (Variable.IfNot(b))
            {
                x.SetTo(Variable.Bernoulli(p));
            }
            Range r = new Range(2).Named("r");
            Variable<int> d = Variable.Discrete(0.5, 0.5).Named("d");
            using (Variable.Case(d, 0))
            {
                var c = !x;
                c.ObservedValue = true;
            }
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(p));
        }

        [Fact]
        public void InferUniformGateEnterOneTest()
        {
            // This model is specially constructed so that the message down to Gate.EnterOne is uniform iff the message from Gate.EnterOne is uniform.
            Range r = new Range(2).Named("r");
            Variable<Beta> prior = Variable<Beta>.Factor(Beta.Uniform).Named("prior");
            Variable<double> p = Variable<double>.Random(prior).Named("p");
            Variable<bool> b = Variable.Bernoulli(p).Named("b");
            VariableArray<bool> x = Variable.Array<bool>(r).Named("x");
            using (Variable.If(b))
            {
                x[r] = Variable.Bernoulli(p).ForEach(r);
            }
            using (Variable.IfNot(b))
            {
                x[r] = Variable.Bernoulli(p).ForEach(r);
            }
            Variable<int> index = Variable.Discrete(r, 0.5, 0.5).Named("index");
            using (Variable.Switch(index))
            {
                var c = !x[index];
                c.ObservedValue = true;
            }
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(p));
        }

        [Fact]
        public void InferUniformConstrainEqualRandomTest()
        {
            // This model is specially constructed so that the message down to CER is uniform iff the message from CER is uniform.
            Variable<Bernoulli> prior = Variable<Bernoulli>.Factor(Bernoulli.Uniform).Named("prior");
            Variable<bool> p = Variable<bool>.Random(prior).Named("p");
            Variable<bool> b = (!p).Named("b");
            using (Variable.If(b))
            {
                Variable.ConstrainEqualRandom(p, prior);
            }
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(engine.Infer(p));
        }

        [Fact]
        public void InferUniformConstrainEqualTest()
        {
            Variable<Beta> prior = Variable<Beta>.Factor(Beta.Uniform).Named("prior");
            Variable<double> p = Variable<double>.Random(prior).Named("p");
            Variable<double> q = Variable<double>.Random(prior).Named("q");
            Variable<bool> b = Variable.Bernoulli(p).Named("b");
            using (Variable.If(b))
            {
                Variable.ConstrainEqual(p, q);
            }
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(b));
        }

        [Fact]
        public void InferUniformConstrainEqualTest2()
        {
            Variable<Beta> prior = Variable<Beta>.Factor(Beta.Uniform).Named("prior");
            Variable<double> p = Variable<double>.Random(prior).Named("p");
            Variable<double> q = Variable<double>.Random(prior).Named("q");
            Variable<bool> b = Variable.Bernoulli(p).Named("b");
            using (Variable.If(b))
            {
                Variable.ConstrainEqual(p, q);
            }
            q.ObservedValue = 0.5;
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(b));
        }

        [Fact]
        public void InferUniformNotTest()
        {
            Variable<Bernoulli> prior = Variable<Bernoulli>.Factor(Bernoulli.Uniform).Named("prior");
            Variable<bool> x = Variable<bool>.Random(prior).Named("x");
            Variable<bool> b = Variable<bool>.Random(prior).Named("b");
            using (Variable.If(b))
            {
                var y = !x;
                y.ObservedValue = true;
            }
            Variable.ConstrainEqualRandom(x, prior);
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(b));
        }

        [Fact]
        public void InferUniformRandomTest()
        {
            Variable<Bernoulli> prior = Variable<Bernoulli>.Factor(Bernoulli.Uniform).Named("prior");
            Variable<bool> b = Variable<bool>.Random(prior).Named("b");
            Variable<bool> x = Variable.New<bool>().Named("x");
            using (Variable.If(b))
            {
                x.SetTo(Variable<bool>.Random(prior));
            }
            using (Variable.IfNot(b))
            {
                x.SetTo(Variable<bool>.Random(prior));
            }
            var y = (x == b);
            Variable.ConstrainEqualRandom(y, new Bernoulli(0.5));
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(engine.Infer(b));
        }


        [Fact]
        public void InferNoUsesTest()
        {
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100).Named("mean");
            Variable<double> x = Variable.GaussianFromMeanAndVariance(mean, 1).Named("x");

            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            Gaussian expected = Gaussian.FromMeanAndVariance(0, 101);
            Gaussian actual = engine.Infer<Gaussian>(x);
            Assert.True(expected.MaxDiff(actual) < 1e-10);
            expected = Gaussian.FromMeanAndVariance(0, 100);
            actual = engine.Infer<Gaussian>(mean);
            Assert.True(expected.MaxDiff(actual) < 1e-10);
        }

        [Fact]
        public void InferArrayNoUsesTest()
        {
            Range item = new Range(1).Named("item");
            VariableArray<double> mean = Variable.Array<double>(item).Named("mean");
            mean[item] = Variable.GaussianFromMeanAndVariance(0, 100).ForEach(item);
            VariableArray<double> x = Variable.Array<double>(item).Named("x");
            x[item] = Variable.GaussianFromMeanAndVariance(mean[item], 1);

            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            Gaussian xExpected = Gaussian.FromMeanAndVariance(0, 101);
            Gaussian[] xActual = engine.Infer<Gaussian[]>(x);
            for (int i = 0; i < xActual.Length; i++)
            {
                Assert.True(xExpected.MaxDiff(xActual[i]) < 1e-10);
            }
            Gaussian meanExpected = Gaussian.FromMeanAndVariance(0, 100);
            Gaussian[] meanActual = engine.Infer<Gaussian[]>(mean);
            for (int i = 0; i < meanActual.Length; i++)
            {
                Assert.True(meanExpected.MaxDiff(meanActual[i]) < 1e-10);
            }
        }

        [Fact]
        public void InferArrayItemError()

        {
            Assert.Throws<ArgumentException>(() =>
            {

                Range item = new Range(1).Named("item");
                VariableArray<double> x = Variable.Array<double>(item).Named("x");
                x[item] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(item);

                InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
                Gaussian xExpected = Gaussian.FromMeanAndVariance(0, 1);
                var xActual = engine.Infer<Gaussian[]>(x[item]);
                for (int i = 0; i < xActual.Length; i++)
                {
                    Assert.True(xExpected.MaxDiff(xActual[i]) < 1e-10);
                }

            });
        }

        [Fact]
        public void ConstrainEqualTest()
        {
            Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 1);
            Variable<double> y = Variable.GaussianFromMeanAndVariance(0, 1);
            Variable.ConstrainEqual(x, y);
            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            Gaussian expected = Gaussian.FromMeanAndPrecision(0, 2);
            Gaussian actual = engine.Infer<Gaussian>(x);
            Assert.True(expected.MaxDiff(actual) < 1e-10);
        }

        [Fact]
        public void UniformPriorTest()
        {
            var prior = Variable<Dirichlet>.Factor(Dirichlet.Uniform, 10);
            var v = Variable<Vector>.Random(prior);
            var dummy = Variable.Discrete(v);
            dummy.ObservedValue = 1;

            var engine = new InferenceEngine();
            var vdist = engine.Infer<Dirichlet>(v);
            Console.WriteLine("vdist=" + vdist);
            Assert.False(vdist.IsUniform(), "Posterior distribution should not be uniform");
        }

        [Fact]
        public void PartiallyUniformArrayTest()
        {
            var item = new Range(2).Named("item");
            var means = Variable.Array<double>(item).Named("means");
            means[0] = Variable.GaussianFromMeanAndPrecision(0, 1);
            means[1] = Variable.Random(Gaussian.Uniform());
            var x = Variable.Array<double>(item).Named("x");
            using (Variable.ForEach(item))
            {
                x[item] = Variable.GaussianFromMeanAndPrecision(means[item], 1);
            }
            InferenceEngine engine = new InferenceEngine();
            GaussianArray expected = new GaussianArray(2);
            expected[0] = Gaussian.FromMeanAndPrecision(0, 0.5);
            expected[1] = Gaussian.Uniform();
            IList<Gaussian> actual = engine.Infer<IList<Gaussian>>(x);
            Assert.Equal(expected, actual);
        }

        [Fact]
        public void PartiallyUniformArrayTest2()
        {
            var item = new Range(2).Named("item");
            var inner = new Range(2).Named("inner");
            var arrays = Variable.Array(Variable.Array<double>(inner), item).Named("arrays");
            arrays[0][0] = Variable.GaussianFromMeanAndPrecision(0, 1);
            arrays[0][1] = Variable.GaussianFromMeanAndPrecision(0, 1);
            arrays[1][0] = Variable.Random(Gaussian.Uniform());
            arrays[1][1] = Variable.Random(Gaussian.Uniform());
            var sums = Variable.Array<double>(item).Named("sums");
            using (Variable.ForEach(item))
            {
                sums[item] = Variable.Sum(arrays[item]);
            }
            InferenceEngine engine = new InferenceEngine();
            GaussianArray expected = new GaussianArray(2);
            expected[0] = Gaussian.FromMeanAndPrecision(0, 0.5);
            expected[1] = Gaussian.Uniform();
            IList<Gaussian> actual = engine.Infer<IList<Gaussian>>(sums);
            Assert.Equal(expected, actual);
        }

        [Fact]
        public void VectorGaussianFactorTest()
        {
            Variable<VectorGaussian> meanPrior = Variable.New<VectorGaussian>().Named("meanPrior");
            Variable<Vector> mean = Variable.Random<Vector, VectorGaussian>(meanPrior).Named("mean");
            Variable<Vector> x = Variable.VectorGaussianFromMeanAndPrecision(mean, PositiveDefiniteMatrix.Identity(2)).Named("x");
            //.Attrib(new MarginalPrototype(new VectorGaussian(2)));
            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            //engine.BrowserMode = BrowserMode.Always;
            meanPrior.ObservedValue = new VectorGaussian(Vector.Zero(2), PositiveDefiniteMatrix.Identity(2));
            VectorGaussian expected = new VectorGaussian(Vector.Zero(2), PositiveDefiniteMatrix.IdentityScaledBy(2, 2.0));
            VectorGaussian actual = engine.Infer<VectorGaussian>(x);
            Console.WriteLine("x = {0} (should be {1})", actual, expected);
            Assert.True(expected.MaxDiff(actual) < 1e-10);
        }

        [Fact]
        public void DiscreteFromDirichletFactorTest()
        {
            Variable<Dirichlet> probsPrior = Variable.New<Dirichlet>().Named("probsPrior");
            Variable<Vector> probs = Variable.Random<Vector, Dirichlet>(probsPrior).Named("probs");
            Variable<int> x = Variable.Discrete(probs);
            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            probsPrior.ObservedValue = new Dirichlet(1.0, 2.0);
            Discrete expected = new Discrete(new double[] { 1.0 / 3, 2.0 / 3 });
            Discrete actual = engine.Infer<Discrete>(x);
            Console.WriteLine("x = {0} (should be {1})", actual, expected);
            Assert.True(expected.MaxDiff(actual) < 1e-10);
        }

        [Fact]
        public void SumFactorTest()
        {
            int nItems = 2;
            Range item = new Range(nItems).Named("item");
            VariableArray<double> a = Variable.Array<double>(item).Named("a");
            VariableArray<double> aMeans = Variable.Constant<double>(new double[] { 1, 2 }, item);
            VariableArray<double> aPrecisions = Variable.Constant<double>(new double[] { 3, 4 }, item);
            a[item] = Variable.GaussianFromMeanAndPrecision(aMeans[item], aPrecisions[item]);
            Variable<double> x = Variable.Sum(a).Named("x");
            Gaussian xPrior = new Gaussian(5, 0.1);
            Variable.ConstrainEqualRandom(x, xPrior);

            InferenceEngine engine = new InferenceEngine();
            Gaussian xExpected = xPrior
                                 * new Gaussian(aMeans.ObservedValue[0] + aMeans.ObservedValue[1], 1.0 / aPrecisions.ObservedValue[0] + 1.0 / aPrecisions.ObservedValue[1]);
            Gaussian xActual = engine.Infer<Gaussian>(x);
            Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);
            Assert.True(xActual.MaxDiff(xExpected) < 1e-7);
            Gaussian a0Expected = new Gaussian(aMeans.ObservedValue[0], 1 / aPrecisions.ObservedValue[0])
                                  * new Gaussian(xPrior.GetMean() - aMeans.ObservedValue[1], xPrior.GetVariance() + 1 / aPrecisions.ObservedValue[1]);
            Gaussian a0Actual = engine.Infer<DistributionArray<Gaussian>>(a)[0];
            Console.WriteLine("a[0] = {0} (should be {1})", a0Actual, a0Expected);
            Assert.True(a0Actual.MaxDiff(a0Expected) < 1e-7);
            Gaussian a1Expected = new Gaussian(aMeans.ObservedValue[1], 1 / aPrecisions.ObservedValue[1])
                                  * new Gaussian(xPrior.GetMean() - aMeans.ObservedValue[0], xPrior.GetVariance() + 1 / aPrecisions.ObservedValue[0]);
            Gaussian a1Actual = engine.Infer<DistributionArray<Gaussian>>(a)[1];
            Console.WriteLine("a[1] = {0} (should be {1})", a1Actual, a1Expected);
            Assert.True(a1Actual.MaxDiff(a1Expected) < 1e-7);
        }

        [Fact]
        public void SumFactorTest3()
        {
            int nItems = 3;
            Range item = new Range(nItems).Named("item");
            VariableArray<double> a = Variable.Array<double>(item).Named("a");
            VariableArray<double> aMeans = Variable.Constant<double>(new double[] { 1, 2, 3 }, item);
            VariableArray<double> aPrecisions = Variable.Constant<double>(new double[] { Double.PositiveInfinity, 0, 2 }, item);
            a[item] = Variable.GaussianFromMeanAndPrecision(aMeans[item], aPrecisions[item]);
            Variable<double> x = Variable.Sum(a).Named("x");
            Gaussian xPrior = new Gaussian(5, 0.1);
            Variable.ConstrainEqualRandom(x, xPrior);

            InferenceEngine engine = new InferenceEngine();
            Gaussian xExpected = xPrior;
            Gaussian xActual = engine.Infer<Gaussian>(x);
            Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);
            Assert.True(xActual.MaxDiff(xExpected) < 1e-7);
            Gaussian a0Expected = Gaussian.PointMass(1);
            Gaussian a0Actual = engine.Infer<DistributionArray<Gaussian>>(a)[0];
            Console.WriteLine("a[0] = {0} (should be {1})", a0Actual, a0Expected);
            Assert.True(a0Actual.MaxDiff(a0Expected) < 1e-7);
            Gaussian a1Expected = new Gaussian(xPrior.GetMean() - aMeans.ObservedValue[0] - aMeans.ObservedValue[2],
                                               xPrior.GetVariance() + 1 / aPrecisions.ObservedValue[0] + 1 / aPrecisions.ObservedValue[2]);
            Gaussian a1Actual = engine.Infer<DistributionArray<Gaussian>>(a)[1];
            Console.WriteLine("a[1] = {0} (should be {1})", a1Actual, a1Expected);
            Assert.True(a1Actual.MaxDiff(a1Expected) < 1e-7);
            Gaussian a2Expected = new Gaussian(3, 1.0 / 2);
            Gaussian a2Actual = engine.Infer<DistributionArray<Gaussian>>(a)[2];
            Console.WriteLine("a[2] = {0} (should be {1})", a2Actual, a2Expected);
            Assert.True(a2Actual.MaxDiff(a2Expected) < 1e-7);
        }

        [Fact]
        public void SumFactorTest1()
        {
            int nItems = 1;
            Range item = new Range(nItems).Named("item");
            VariableArray<double> a = Variable.Array<double>(item).Named("a");
            a[item] = Variable.GaussianFromMeanAndVariance(2, Double.PositiveInfinity).ForEach(item);
            Variable<double> x = Variable.Sum(a).Named("x");
            Gaussian xPrior = new Gaussian(5, 0.1);
            Variable.ConstrainEqualRandom(x, xPrior);

            InferenceEngine engine = new InferenceEngine();
            Gaussian xExpected = xPrior;
            Gaussian xActual = engine.Infer<Gaussian>(x);
            Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);
            Assert.True(xActual.MaxDiff(xExpected) < 1e-7);
            Gaussian a0Expected = xPrior;
            Gaussian a0Actual = engine.Infer<DistributionArray<Gaussian>>(a)[0];
            Console.WriteLine("a[0] = {0} (should be {1})", a0Actual, a0Expected);
            Assert.True(a0Actual.MaxDiff(a0Expected) < 1e-7);
        }

        /// <summary>
        /// Call Factor.Sum within a loop.
        /// </summary>
        [Fact]
        public void SumFactorTest2()
        {
            int nItems = 1, nFeatures = 1;
            Range feature = new Range(nFeatures).Named("feature");
            VariableArray<double> a = Variable.Array<double>(feature);
            a[feature] = Variable.GaussianFromMeanAndVariance(1, 2).ForEach(feature);
            Range item = new Range(nItems).Named("item");
            VariableArray<double> sum = Variable.Array<double>(item);
            sum[item] = Variable.Sum(a).ForEach(item);

            InferenceEngine engine = new InferenceEngine();
            DistributionArray<Gaussian> sumActual = engine.Infer<DistributionArray<Gaussian>>(sum);
            IDistribution<double[]> sumExpected = Distribution<double>.Array(new Gaussian[] { new Gaussian(1, 2) });
            Console.WriteLine("sum = {0} (should be {1})", sumActual, sumExpected);
            Assert.True(sumActual.MaxDiff(sumExpected) < 1e-10);
        }

        /// <summary>
        /// Tests the VectorGaussian sum factor for EP.
        /// </summary>
        [Fact]
        public void EpVectorGaussianSumFactorTest()
        {
            this.VectorGaussianSumFactorTest(new ExpectationPropagation());
        }

        /// <summary>
        /// Tests the VectorGaussian sum factor for VMP.
        /// </summary>
        /// <remark>The VMP approximation to the sum is not very precise here.</remark>
        [Fact]
        [Trait("Category", "OpenBug")]
        public void VmpVectorGaussianSumFactorTest()
        {
            this.VectorGaussianSumFactorTest(new VariationalMessagePassing(), 1000);
        }

        /// <summary>
        /// Tests the VectorGaussian sum factor.
        /// </summary>
        /// <param name="algorithm">The inference algorithm.</param>
        /// <param name="inferenceIterationCount">The number of iterations of the inference algorithm. Defaults to 50.</param>
        /// <param name="tolerance">The tolerance for differences in the results. Defaults to 1e-10.</param>
        private void VectorGaussianSumFactorTest(IAlgorithm algorithm, int inferenceIterationCount = 50, double tolerance = 1e-10)
        {
            // Model
            const int ItemCount = 2;
            Range item = new Range(ItemCount).Named("item");
            VariableArray<Vector> array = Variable.Array<Vector>(item).Named("array");
            VariableArray<Vector> arrayMeans = Variable.Constant(
                new[]
                    {
                        Vector.FromArray(1, 2), Vector.FromArray(3, 4)
                    },
                item);
            VariableArray<PositiveDefiniteMatrix> arrayPrecisions = Variable.Constant(
                new[]
                    {
                        new PositiveDefiniteMatrix(new[,] {{3.0, 1.0}, {1.0, 2.0}}),
                        new PositiveDefiniteMatrix(new[,] {{double.PositiveInfinity, 0}, {0, 1}})
                    },
                item);

            array[item] = Variable.VectorGaussianFromMeanAndPrecision(arrayMeans[item], arrayPrecisions[item]);

            Variable<Vector> sum = Variable.Sum(array).Named("sum");

            var sumPrior = new VectorGaussian(Vector.FromArray(5, 5), PositiveDefiniteMatrix.IdentityScaledBy(2, 0.1));
            Variable.ConstrainEqualRandom(sum, sumPrior);

            // Inference engine
            var engine = new InferenceEngine
            {
                Algorithm = algorithm,
                NumberOfIterations = inferenceIterationCount,
                ShowProgress = false
            };
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            engine.Compiler.RequiredQuality = QualityBand.Experimental;

            // Check sum
            VectorGaussian expectedSum = sumPrior * new VectorGaussian(
                                                      arrayMeans.ObservedValue[0] + arrayMeans.ObservedValue[1],
                                                      arrayPrecisions.ObservedValue[0].Inverse() + arrayPrecisions.ObservedValue[1].Inverse());

            var actualSum = engine.Infer<VectorGaussian>(sum);

            Console.WriteLine("[actual] sum =\n{0}\n[expected] sum =\n{1}\n", actualSum, expectedSum);
            Assert.True(actualSum.MaxDiff(expectedSum) < tolerance);

            // Check array[0]
            VectorGaussian expectedArray0 = new VectorGaussian(arrayMeans.ObservedValue[0], arrayPrecisions.ObservedValue[0].Inverse()) *
                                            new VectorGaussian(sumPrior.GetMean() - arrayMeans.ObservedValue[1],
                                                               sumPrior.GetVariance() + arrayPrecisions.ObservedValue[1].Inverse());

            VectorGaussian actualArray0 = engine.Infer<DistributionArray<VectorGaussian>>(array)[0];

            Console.WriteLine("[actual] array[0] =\n{0}\n[expected] array[0] =\n{1}\n", actualArray0, expectedArray0);
            Assert.True(actualArray0.MaxDiff(expectedArray0) < tolerance);

            // Check array[1]
            VectorGaussian expectedArray1 = new VectorGaussian(arrayMeans.ObservedValue[1], arrayPrecisions.ObservedValue[1].Inverse()) *
                                            new VectorGaussian(sumPrior.GetMean() - arrayMeans.ObservedValue[0],
                                                               sumPrior.GetVariance() + arrayPrecisions.ObservedValue[0].Inverse());

            VectorGaussian actualArray1 = engine.Infer<DistributionArray<VectorGaussian>>(array)[1];

            Console.WriteLine("[actual] array[1] =\n{0}\n[expected] array[1] =\n{1}\n", actualArray1, expectedArray1);
            Assert.True(actualArray1.MaxDiff(expectedArray1) < tolerance);
        }

        /// <summary>
        /// Tests the VectorGaussian sum factor for uniform array elements and EP.
        /// </summary>
        [Fact]
        public void EpUniformArrayVectorGaussianSumFactorTest()
        {
            this.UniformArrayVectorGaussianSumFactorTest(new ExpectationPropagation());
        }

        /// <summary>
        /// Tests the VectorGaussian sum factor for uniform array elements and VMP.
        /// </summary>
        [Fact]
        public void VmpUniformArrayVectorGaussianSumFactorTest()
        {
            this.UniformArrayVectorGaussianSumFactorTest(new VariationalMessagePassing());
        }

        /// <summary>
        /// Tests the VectorGaussian sum factor for uniform array elements.
        /// </summary>
        /// <param name="algorithm">The inference algorithm.</param>
        /// <param name="tolerance">The tolerance for differences in the results. Defaults to 1e-10.</param>
        private void UniformArrayVectorGaussianSumFactorTest(IAlgorithm algorithm, double tolerance = 1e-10)
        {
            // Model
            const int ItemCount = 1;
            Range item = new Range(ItemCount).Named("item");
            VariableArray<Vector> array = Variable.Array<Vector>(item).Named("array");
            array[item] = Variable.VectorGaussianFromMeanAndVariance(
                Vector.FromArray(2.0, 1.0),
                PositiveDefiniteMatrix.IdentityScaledBy(2, double.PositiveInfinity)).ForEach(item);

            Variable<Vector> sum = Variable.Sum(array).Named("sum");

            var sumPrior = new VectorGaussian(Vector.FromArray(5, 5), PositiveDefiniteMatrix.IdentityScaledBy(2, 0.1));
            Variable.ConstrainEqualRandom(sum, sumPrior);

            // Inference engine
            var engine = new InferenceEngine
            {
                Algorithm = algorithm,
                ShowProgress = false
            };
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            engine.Compiler.RequiredQuality = QualityBand.Experimental;

            // Check sum
            VectorGaussian expectedSum = sumPrior;
            var actualSum = engine.Infer<VectorGaussian>(sum);

            Console.WriteLine("[actual] sum =\n{0}\n[expected] sum =\n{1}\n", actualSum, expectedSum);
            Assert.True(actualSum.MaxDiff(expectedSum) < tolerance);

            // Check array
            VectorGaussian expectedArray0 = sumPrior;
            VectorGaussian actualArray0 = engine.Infer<DistributionArray<VectorGaussian>>(array)[0];

            Console.WriteLine("[actual] array[0] =\n{0}\n[expected] array[0] =\n{1}\n", actualArray0, expectedArray0);
            Assert.True(actualArray0.MaxDiff(expectedArray0) < tolerance);
        }

        /// <summary>
        /// Tests the VectorGaussian sum factor for EP when a single element in the array is uniform.
        /// </summary>
        [Fact]
        public void EpUniformArrayElementVectorGaussianSumFactorTest()
        {
            this.UniformArrayElementVectorGaussianSumFactorTest(new ExpectationPropagation());
        }

        /// <summary>
        /// Tests the VectorGaussian sum factor for VMP when a single element in the array is uniform.
        /// </summary>
        /// <remark>The VMP approximation to the sum is not very precise here.</remark>
        [Fact]
        [Trait("Category", "OpenBug")]
        public void VmpUniformArrayElementVectorGaussianSumFactorTest()
        {
            this.UniformArrayElementVectorGaussianSumFactorTest(new VariationalMessagePassing(), 1000);
        }

        /// <summary>
        /// Tests the VectorGaussian sum factor when a single element in the array is uniform.
        /// </summary>
        /// <param name="algorithm">The inference algorithm.</param>
        /// <param name="inferenceIterationCount">The number of iterations of the inference algorithm. Defaults to 50.</param>
        /// <param name="tolerance">The tolerance for differences in the results. Defaults to 1e-10.</param>
        private void UniformArrayElementVectorGaussianSumFactorTest(IAlgorithm algorithm, int inferenceIterationCount = 50, double tolerance = 1e-10)
        {
            // Model
            const int ItemCount = 3;
            Range item = new Range(ItemCount).Named("item");
            VariableArray<Vector> array = Variable.Array<Vector>(item).Named("array");
            VariableArray<Vector> arrayMeans =
                Variable.Constant(
                    new[] { Vector.FromArray(1.0, 2.0), Vector.FromArray(3.0, 4.0), Vector.FromArray(5.0, 6.0) }, item);

            VariableArray<PositiveDefiniteMatrix> arrayPrecisions =
                Variable.Constant(
                    new[]
                        {
                            new PositiveDefiniteMatrix(new[,] {{double.PositiveInfinity, 0}, {0, 1}}),
                            PositiveDefiniteMatrix.IdentityScaledBy(2, 0),
                            PositiveDefiniteMatrix.IdentityScaledBy(2, 2)
                        },
                    item);

            array[item] = Variable.VectorGaussianFromMeanAndPrecision(arrayMeans[item], arrayPrecisions[item]);

            Variable<Vector> sum = Variable.Sum(array).Named("sum");

            var sumPrior = new VectorGaussian(Vector.FromArray(5.0, 5.0), PositiveDefiniteMatrix.IdentityScaledBy(2, 0.1));
            Variable.ConstrainEqualRandom(sum, sumPrior);

            // Inference engine
            var engine = new InferenceEngine
            {
                Algorithm = algorithm,
                NumberOfIterations = inferenceIterationCount,
                ShowProgress = false
            };
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            engine.Compiler.RequiredQuality = QualityBand.Experimental;

            // Check sum
            VectorGaussian expectedSum = sumPrior;
            var actualSum = engine.Infer<VectorGaussian>(sum);
            Console.WriteLine("[actual] sum =\n{0}\n[expected] sum =\n{1}\n", actualSum, expectedSum);
            Assert.True(actualSum.MaxDiff(expectedSum) < tolerance);

            // Check array[0]
            var expectedArray0 = new VectorGaussian(arrayMeans.ObservedValue[0], arrayPrecisions.ObservedValue[0].Inverse());
            VectorGaussian actualArray0 = engine.Infer<DistributionArray<VectorGaussian>>(array)[0];

            Console.WriteLine("[actual] array[0] =\n{0}\n[expected] array[0] =\n{1}\n", actualArray0, expectedArray0);
            Assert.True(actualArray0.MaxDiff(expectedArray0) < tolerance);

            // Check array[1]
            var expectedArray1 =
                new VectorGaussian(
                    sumPrior.GetMean() - arrayMeans.ObservedValue[0] - arrayMeans.ObservedValue[2],
                    sumPrior.GetVariance() + arrayPrecisions.ObservedValue[0].Inverse() + arrayPrecisions.ObservedValue[2].Inverse());

            VectorGaussian actualArray1 = engine.Infer<DistributionArray<VectorGaussian>>(array)[1];

            Console.WriteLine("[actual] array[1] =\n{0}\n[expected] array[1] =\n{1}\n", actualArray1, expectedArray1);
            Assert.True(actualArray1.MaxDiff(expectedArray1) < tolerance);

            // Check array[2]
            var expectedArray2 = new VectorGaussian(
                arrayMeans.ObservedValue[2], arrayPrecisions.ObservedValue[2].Inverse());
            VectorGaussian actualArray2 = engine.Infer<DistributionArray<VectorGaussian>>(array)[2];

            Console.WriteLine("[actual] array[2] =\n{0}\n[expected] array[2] =\n{1}\n", actualArray2, expectedArray2);
            Assert.True(actualArray2.MaxDiff(expectedArray2) < tolerance);
        }

        /// <summary>
        /// Tests the VectorGaussian sum factor in a loop using EP.
        /// </summary>
        [Fact]
        public void EpLoopVectorGaussianSumFactorTest()
        {
            this.UniformArrayVectorGaussianSumFactorTest(new ExpectationPropagation());
        }

        /// <summary>
        /// Tests the VectorGaussian sum factor in a loop using VMP.
        /// </summary>
        [Fact]
        public void VmpLoopVectorGaussianSumFactorTest()
        {
            this.UniformArrayVectorGaussianSumFactorTest(new VariationalMessagePassing());
        }

        /// <summary>
        /// Tests the VectorGaussian sum factor in a loop.
        /// </summary>
        /// <param name="algorithm">The inference algorithm.</param>
        /// <param name="tolerance">The tolerance for differences in the results. Defaults to 1e-10.</param>
        internal void LoopVectorGaussianSumFactorTest(IAlgorithm algorithm, double tolerance = 1e-10)
        {
            // Model
            const int ItemCount = 1;
            const int FeatureCount = 1;
            Range feature = new Range(FeatureCount).Named("feature");
            VariableArray<Vector> array = Variable.Array<Vector>(feature).Named("array");
            array[feature] = Variable.VectorGaussianFromMeanAndVariance(
                Vector.FromArray(1, 2),
                new PositiveDefiniteMatrix(new[,] { { 3.0, 0.4 }, { 0.4, 5.0 } })).ForEach(feature);

            Range item = new Range(ItemCount).Named("item");
            VariableArray<Vector> sum = Variable.Array<Vector>(item);
            sum[item] = Variable.Sum(array).ForEach(item).Named("sum");

            // Inference engine
            var engine = new InferenceEngine
            {
                Algorithm = algorithm,
                ShowProgress = false
            };
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            engine.Compiler.RequiredQuality = QualityBand.Experimental;

            // Check sum
            var sumActual = engine.Infer<DistributionArray<VectorGaussian>>(sum);
            IDistribution<Vector[]> sumExpected = Distribution<Vector>.Array(
                new[] { new VectorGaussian(Vector.FromArray(1, 2), new PositiveDefiniteMatrix(new[,] { { 3.0, 0.4 }, { 0.4, 5.0 } })) });

            Console.WriteLine("[actual] sum =\n{0}\n[expected] sum =\n{1}\n", sumActual, sumExpected);
            Assert.True(sumActual.MaxDiff(sumExpected) < tolerance);
        }

        /// <summary>
        /// Tests the VectorGaussian sum factor for an empty array using EP.
        /// </summary>
        [Fact]
        public void EpEmptyArrayVectorGaussianSumFactorTest()
        {
            this.EmptyArrayVectorGaussianSumFactorTest(new ExpectationPropagation());
        }

        /// <summary>
        /// Tests the VectorGaussian sum factor for an empty array using VMP.
        /// </summary>
        [Fact]
        public void VmpEmptyArrayVectorGaussianSumFactorTest()
        {
            this.EmptyArrayVectorGaussianSumFactorTest(new VariationalMessagePassing());
        }

        /// <summary>
        /// Tests the VectorGaussian sum factor for an empty array.
        /// </summary>
        /// <param name="algorithm">The inference algorithm.</param>
        /// <param name="inferenceIterationCount">The number of iterations of the inference algorithm. Defaults to 50.</param>
        /// <param name="tolerance">The tolerance for differences in the results. Defaults to 1e-10.</param>
        private void EmptyArrayVectorGaussianSumFactorTest(
            IAlgorithm algorithm, int inferenceIterationCount = 50, double tolerance = 1e-10)
        {
            // Model
            Variable<int> itemCount = Variable.New<int>().Named("ItemCount");
            Range item = new Range(itemCount).Named("item");
            VariableArray<Vector> array = Variable.Array<Vector>(item).Named("array");
            VariableArray<Vector> arrayMeans =
                Variable.Constant(new[] { Vector.FromArray(1, 2), Vector.FromArray(3, 4) }, item);
            VariableArray<PositiveDefiniteMatrix> arrayPrecisions =
                Variable.Constant(
                    new[]
                        {
                            new PositiveDefiniteMatrix(new[,] {{3.0, 1.0}, {1.0, 2.0}}),
                            new PositiveDefiniteMatrix(new[,] {{double.PositiveInfinity, 0}, {0, 1}})
                        },
                    item);

            array[item] = Variable.VectorGaussianFromMeanAndPrecision(arrayMeans[item], arrayPrecisions[item]);

            Variable<Vector> sum = Variable.Sum(array).Named("sum");

            var sumPrior = new VectorGaussian(Vector.FromArray(5, 5), PositiveDefiniteMatrix.IdentityScaledBy(2, 0.1));
            Variable.ConstrainEqualRandom(sum, sumPrior);

            // Empty variable array
            itemCount.ObservedValue = 0;

            // Inference engine
            var engine = new InferenceEngine
            {
                Algorithm = algorithm,
                NumberOfIterations = inferenceIterationCount,
                ShowProgress = false
            };

            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            engine.Compiler.RequiredQuality = QualityBand.Experimental;

            // Check sum
            VectorGaussian expectedSum = VectorGaussian.PointMass(Vector.Zero(2));
            var actualSum = engine.Infer<VectorGaussian>(sum);
            Console.WriteLine("[actual] sum =\n{0}\n[expected] sum =\n{1}\n", actualSum, expectedSum);
            Assert.True(actualSum.MaxDiff(expectedSum) < tolerance);
        }

        /// <summary>
        /// Infer posterior over number of heads.
        /// </summary>
        [Fact]
        public void CountTrueFactorTest1()
        {
            var range = new Range(4);
            var coinBiases = Variable.Array<double>(range);
            coinBiases.ObservedValue = new[] { 0.1, 0.3, 0.0, 1.0 };
            var coins = Variable.Array<bool>(range);
            coins[range] = Variable.Bernoulli(coinBiases[range]);
            Variable<int> heads = Variable.CountTrue(coins);
            //heads.AddAttribute(new MarginalPrototype(new Discrete(Util.ArrayInit(range.SizeAsInt + 1, i => 1.0))));
            var headsPosterior = new InferenceEngine().Infer<Discrete>(heads);

            const double Tolerance = 1e-6;
            Assert.Equal(0, headsPosterior[0], Tolerance);
            Assert.Equal(0.63, headsPosterior[1], Tolerance);
            Assert.Equal(0.34, headsPosterior[2], Tolerance);
            Assert.Equal(0.03, headsPosterior[3], Tolerance);
            Assert.Equal(0, headsPosterior[4], Tolerance);
        }

        /// <summary>
        /// Infer posterior over bias of a single coin given biases of other coins and the total number of heads.
        /// </summary>
        [Fact]
        public void CountTrueFactorTest2()
        {
            var trueCoinBiases = new[] { 0.1, 0.3, 0.0, 1.0 };
            var trueHeadsPosterior = new Discrete(0, 0.63, 0.34, 0.03, 0);

            var coinRange = new Range(trueCoinBiases.Length);
            var coinBiases = Variable.Array<double>(coinRange);
            coinBiases[coinRange] = Variable.Beta(1.0, 1.0).ForEach(coinRange);
            VariableArray<int> observedCoinBiasIndices = Variable.Array<int>(new Range(trueCoinBiases.Length - 1));
            observedCoinBiasIndices.ObservedValue = Util.ArrayInit(trueCoinBiases.Length - 1, i => i);
            VariableArray<double> observedCoinBiases = Variable.Subarray(coinBiases, observedCoinBiasIndices);

            Rand.Restart(696);
            var observationRange = new Range(5000);
            VariableArray<int> heads = Variable.Array<int>(observationRange);
            heads.ObservedValue = Util.ArrayInit(observationRange.SizeAsInt, i => trueHeadsPosterior.Sample());
            using (Variable.ForEach(observationRange))
            {
                var coins = Variable.Array<bool>(coinRange);
                coins[coinRange] = Variable.Bernoulli(coinBiases[coinRange]);
                heads[observationRange] = Variable.CountTrue(coins);
            }

            // Try to infer bias of each coin 
            var engine = new InferenceEngine
            {
                NumberOfIterations = 15
            };
            const double Tolerance = 0.01;
            for (int i = 0; i < trueCoinBiases.Length; ++i)
            {
                observedCoinBiases.ObservedValue = Util.ArrayInit(trueCoinBiases.Length - 1, j => j < i ? trueCoinBiases[j] : trueCoinBiases[j + 1]);
                Beta[] inferredProbs = engine.Infer<Beta[]>(coinBiases);
                Assert.Equal(trueCoinBiases[i], inferredProbs[inferredProbs.Length - 1].GetMean(), Tolerance);
            }
        }

        [Fact]
        public void ArrayCopyTest()
        {
            int n = 2;
            Range item = new Range(n).Named("item");
            VariableArray<bool> x = Variable.Array<bool>(item).Named("x");
            double xPrior = 0.3;
            x[item] = Variable.Bernoulli(xPrior).ForEach(item);
            Bernoulli[] data = Util.ArrayInit(n, i => new Bernoulli(0.1 * (i + 1)));
            using (Variable.ForEach(item))
            {
                Range item2 = item.Clone();
                VariableArray<bool> xCopy = Variable.Array<bool>(item2).Named("xCopy");
                xCopy[item2] = Variable.Copy(x[item2]);
                Variable.ConstrainEqualRandom(xCopy, (Sampleable<bool[]>)Distribution<bool>.Array(data));
            }

            InferenceEngine engine = new InferenceEngine();
            Bernoulli[] xExpectedArray = new Bernoulli[n];
            for (int i = 0; i < xExpectedArray.Length; i++)
            {
                double p1 = xPrior * System.Math.Pow(data[i].GetProbTrue(), n);
                double p0 = (1 - xPrior) * System.Math.Pow(data[i].GetProbFalse(), n);
                xExpectedArray[i] = new Bernoulli(p1 / (p1 + p0));
            }
            IDistribution<bool[]> xExpected = Distribution<bool>.Array(xExpectedArray);
            object xActual = engine.Infer(x);
            Console.WriteLine(StringUtil.JoinColumns("x = ", xActual, " should be ", xExpected));
            Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
        }

        [Fact]
        public void ArrayCopyTest2()
        {
            Range r1 = new Range(3).Named("r1");
            Range r2 = new Range(4).Named("r2");
            var va2 = Variable.Array<double>(r2);
            va2[r2] = Variable.GaussianFromMeanAndPrecision(0.0, 1.0).ForEach(r2);
            var va1 = Variable.Array(Variable.Array<double>(r2), r1);
            va1[r1].SetTo(Variable.Copy(va2).ForEach(r1));
            var ie = new InferenceEngine();
            Console.WriteLine(ie.Infer(va1));
        }

        [Fact]
        public void JaggedArrayCopyError()
        {
            try
            {
                int n = 2;
                int m = 3;
                Range item = new Range(n).Named("item");
                Range inner = new Range(m).Named("inner");
                var x = Variable.Array(Variable.Array<bool>(inner), item).Named("x");
                double xPrior = 0.3;
                x[item][inner] = Variable.Bernoulli(xPrior).ForEach(item, inner);
                Bernoulli[][] data = Util.ArrayInit(n, i => Util.ArrayInit(m, j => new Bernoulli(0.1 * (i + j + 1))));
                using (Variable.ForEach(item))
                {
                    var xCopy = Variable.Array(Variable.Array<bool>(inner), item).Named("xCopy");
                    xCopy.SetTo(Variable.Copy(x));
                    Variable.ConstrainEqualRandom(xCopy, (Sampleable<bool[][]>)Distribution<bool>.Array(data));
                }

                InferenceEngine engine = new InferenceEngine();
                Bernoulli[][] xExpectedArray = new Bernoulli[n][];
                for (int i = 0; i < n; i++)
                {
                    xExpectedArray[i] = new Bernoulli[m];
                    for (int j = 0; j < m; j++)
                    {
                        double p1 = xPrior * System.Math.Pow(data[i][j].GetProbTrue(), n);
                        double p0 = (1 - xPrior) * System.Math.Pow(data[i][j].GetProbFalse(), n);
                        xExpectedArray[i][j] = new Bernoulli(p1 / (p1 + p0));
                    }
                }
                IDistribution<bool[][]> xExpected = Distribution<bool>.Array(xExpectedArray);
                object xActual = engine.Infer(x);
                Console.WriteLine(StringUtil.JoinColumns("x = ", xActual, " should be ", xExpected));
                Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
            }
            catch (InvalidOperationException ex)
            {
                Console.WriteLine("Correctly threw " + ex);
            }
        }

        [Fact]
        public void JaggedArrayCopyTest()
        {
            int n = 2;
            int m = 3;
            Range item = new Range(n).Named("item");
            Range inner = new Range(m).Named("inner");
            var x = Variable.Array(Variable.Array<bool>(inner), item).Named("x");
            double xPrior = 0.3;
            x[item][inner] = Variable.Bernoulli(xPrior).ForEach(item, inner);
            Bernoulli[][] data = Util.ArrayInit(n, i => Util.ArrayInit(m, j => new Bernoulli(0.1 * (i + j + 1))));
            using (Variable.ForEach(item))
            {
                Range item2 = item.Clone().Named("item2");
                var xCopy = Variable.Array(Variable.Array<bool>(inner), item2).Named("xCopy");
                xCopy[item2] = Variable.Copy(x[item2]);
                Variable.ConstrainEqualRandom(xCopy, (Sampleable<bool[][]>)Distribution<bool>.Array(data));
            }

            InferenceEngine engine = new InferenceEngine();
            Bernoulli[][] xExpectedArray = new Bernoulli[n][];
            for (int i = 0; i < n; i++)
            {
                xExpectedArray[i] = new Bernoulli[m];
                for (int j = 0; j < m; j++)
                {
                    double p1 = xPrior * System.Math.Pow(data[i][j].GetProbTrue(), n);
                    double p0 = (1 - xPrior) * System.Math.Pow(data[i][j].GetProbFalse(), n);
                    xExpectedArray[i][j] = new Bernoulli(p1 / (p1 + p0));
                }
            }
            IDistribution<bool[][]> xExpected = Distribution<bool>.Array(xExpectedArray);
            object xActual = engine.Infer(x);
            Console.WriteLine(StringUtil.JoinColumns("x = ", xActual, " should be ", xExpected));
            Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
        }

        [Fact]
        public void DeepJaggedArrayCopyTest()
        {
            Range outer = new Range(1);
            Range middle = new Range(1);
            Range inner = new Range(1);
            var array = Variable.Array(Variable.Array(Variable.Array<bool>(inner), middle), outer).Named("array");
            array[outer][middle][inner] = Variable.Bernoulli(0.1).ForEach(outer, middle, inner);
            var arrayCopy = Variable.Copy(array).Named("arrayCopy");

            InferenceEngine engine = new InferenceEngine();
            var actual = engine.Infer<Bernoulli[][][]>(arrayCopy);
            var expected = new Bernoulli(0.1);
            for (int i = 0; i < outer.SizeAsInt; i++)
            {
                for (int j = 0; j < middle.SizeAsInt; j++)
                {
                    for (int k = 0; k < inner.SizeAsInt; k++)
                    {
                        Assert.True(expected.MaxDiff(actual[i][j][k]) < 1e-10);
                    }
                }
            }
        }

        [Fact]
        public void DeepJaggedArrayReplicateTest()
        {
            Range outer = new Range(1);
            Range middle = new Range(1);
            Range inner = new Range(1);
            var array = Variable.Array(Variable.Array(Variable.Array<bool>(inner), middle), outer).Named("array");
            array[outer][middle][inner] = Variable.Bernoulli(0.1).ForEach(outer, middle, inner);
            Range newRange = new Range(1);
            var rep = Variable.Replicate(array, newRange).Named("rep");

            InferenceEngine engine = new InferenceEngine();
            var actual = engine.Infer<Bernoulli[][][][]>(rep);
            var expected = new Bernoulli(0.1);
            for (int u = 0; u < newRange.SizeAsInt; u++)
            {
                for (int i = 0; i < outer.SizeAsInt; i++)
                {
                    for (int j = 0; j < middle.SizeAsInt; j++)
                    {
                        for (int k = 0; k < inner.SizeAsInt; k++)
                        {
                            Assert.True(expected.MaxDiff(actual[u][i][j][k]) < 1e-10);
                        }
                    }
                }
            }
        }

        [Fact]
        public void ArrayGetItemsTest()
        {
            int n = 2;
            Range item = new Range(n).Named("item");
            VariableArray<bool> x = Variable.Array<bool>(item).Named("x");
            double xPrior = 0.3;
            x[item] = Variable.Bernoulli(xPrior).ForEach(item);
            Bernoulli[] data = new Bernoulli[n];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = new Bernoulli(0.1 * (i + 1));
            }
            VariableArray<int> indices = Variable.Constant(Util.ArrayInit(n, i => i), item).Named("indices");
            indices.AddAttribute(new ValueRange(item));
            Range item2 = item.Clone().Named("item2");
            using (Variable.ForEach(item2))
            {
                VariableArray<bool> xCopy = Variable.GetItems(x, indices).Named("xCopy");
                Variable.ConstrainEqualRandom(xCopy, (Sampleable<bool[]>)Distribution<bool>.Array(data));
            }

            InferenceEngine engine = new InferenceEngine();
            Bernoulli[] xExpectedArray = new Bernoulli[n];
            for (int i = 0; i < xExpectedArray.Length; i++)
            {
                double p1 = xPrior * System.Math.Pow(data[i].GetProbTrue(), n);
                double p0 = (1 - xPrior) * System.Math.Pow(data[i].GetProbFalse(), n);
                xExpectedArray[i] = new Bernoulli(p1 / (p1 + p0));
            }
            IDistribution<bool[]> xExpected = Distribution<bool>.Array(xExpectedArray);
            object xActual = engine.Infer(x);
            Console.WriteLine(StringUtil.JoinColumns("x = ", xActual, " should be ", xExpected));
            Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
        }

        [Fact]
        public void ArrayGetItemsError()

        {
            Assert.Throws<InvalidOperationException>(() =>
            {

                int n = 2;
                Range item = new Range(n).Named("item");
                VariableArray<bool> x = Variable.Array<bool>(item).Named("x");
                double xPrior = 0.3;
                x[item] = Variable.Bernoulli(xPrior).ForEach(item);
                Bernoulli[] data = new Bernoulli[n];
                for (int i = 0; i < data.Length; i++)
                {
                    data[i] = new Bernoulli(0.1 * (i + 1));
                }
                VariableArray<int> indices = Variable.Constant(Util.ArrayInit(n, i => i), item).Named("indices");
                indices.AddAttribute(new ValueRange(item));
                using (Variable.ForEach(item))
                {
                    VariableArray<bool> xCopy = Variable.GetItems(x, indices).Named("xCopy");
                    Variable.ConstrainEqualRandom(xCopy, (Sampleable<bool[]>)Distribution<bool>.Array(data));
                }

                InferenceEngine engine = new InferenceEngine();
                Bernoulli[] xExpectedArray = new Bernoulli[n];
                for (int i = 0; i < xExpectedArray.Length; i++)
                {
                    double p1 = xPrior * System.Math.Pow(data[i].GetProbTrue(), n);
                    double p0 = (1 - xPrior) * System.Math.Pow(data[i].GetProbFalse(), n);
                    xExpectedArray[i] = new Bernoulli(p1 / (p1 + p0));
                }
                IDistribution<bool[]> xExpected = Distribution<bool>.Array(xExpectedArray);
                object xActual = engine.Infer(x);
                Console.WriteLine(StringUtil.JoinColumns("x = ", xActual, " should be ", xExpected));
                Assert.True(xExpected.MaxDiff(xActual) < 1e-10);

            });
        }

        [Fact]
        public void ArrayConstrainEqualRandomTest()
        {
            int n = 2;
            Range item = new Range(n).Named("item");
            VariableArray<bool> x = Variable.Array<bool>(item).Named("x");
            double xPrior = 0.3;
            x[item] = Variable.Bernoulli(xPrior).ForEach(item);
            Bernoulli[] data = new Bernoulli[n];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = new Bernoulli(0.1 * (i + 1));
            }
            // This model should work without having to mention the DistributionStructArray type.
            //Variable.ConstrainEqualRandom(x, (DistributionStructArray<Bernoulli,bool>)Distribution<bool>.Array(data));
            Variable.ConstrainEqualRandom(x, (Sampleable<bool[]>)Distribution<bool>.Array(data));

            InferenceEngine engine = new InferenceEngine();
            Bernoulli[] xExpectedArray = new Bernoulli[n];
            for (int i = 0; i < xExpectedArray.Length; i++)
            {
                double p1 = xPrior * data[i].GetProbTrue();
                double p0 = (1 - xPrior) * data[i].GetProbFalse();
                xExpectedArray[i] = new Bernoulli(p1 / (p1 + p0));
            }
            IDistribution<bool[]> xExpected = Distribution<bool>.Array(xExpectedArray);
            object xActual = engine.Infer(x);
            Console.WriteLine(StringUtil.JoinColumns("x = ", xActual, " should be ", xExpected));
            Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
        }

        [Fact]
        public void ArrayConstrainEqualTest()
        {
            double[] data = new double[] { 1, 2, 3 };
            Range item = new Range(data.Length).Named("item");
            VariableArray<double> x = Variable.Array<double>(item);
            x[item] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(item);
            Variable.ConstrainEqual(x, data);

            InferenceEngine engine = new InferenceEngine();
            DistributionArray<Gaussian> xDist = engine.Infer<DistributionArray<Gaussian>>(x);
            for (int i = 0; i < xDist.Count; i++)
            {
                Assert.True(xDist[i].IsPointMass && xDist[i].Point == data[i]);
            }
        }

        [Fact]
        public void Array2DConstrainEqualTest()
        {
            double[,] data = new double[,] { { 1 }, { 2 } };
            Range r1 = new Range(data.GetLength(0)).Named("r1");
            Range r2 = new Range(data.GetLength(1)).Named("r2");
            VariableArray2D<double> x = Variable.Array<double>(r1, r2);
            x[r1, r2] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(r1, r2);
            Variable.ConstrainEqual(x, data);

            InferenceEngine engine = new InferenceEngine();
            DistributionArray2D<Gaussian> xDist = engine.Infer<DistributionArray2D<Gaussian>>(x);
            for (int i = 0; i < xDist.GetLength(0); i++)
            {
                for (int j = 0; j < xDist.GetLength(1); j++)
                {
                    Assert.True(xDist[i, j].IsPointMass && xDist[i, j].Point == data[i, j]);
                }
            }
        }

        /// <summary>
        /// Clipped Gaussian, implemented using the modelling API.
        /// </summary>
        [Fact]
        public void ClippedGaussian()
        {
            Variable<double> threshold = Variable.Constant(0.5).Named("threshold");
            Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 1).Named("x");
            Variable.ConstrainPositive((x - threshold).Named("diff"));

            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            Gaussian expected = new Gaussian(1.1410777703680648, 0.26848040715587884);
            Gaussian actual = engine.Infer<Gaussian>(x);
            Console.WriteLine("x = {0} should be {1}", actual, expected);
            Assert.True(expected.MaxDiff(actual) < 1e-4);
        }

        /// <summary>
        /// Clipped Gaussian with threshold parameter.
        /// </summary>
        [Fact]
        public void ClippedGaussianWithThresholdParameter()
        {
            Variable<double> threshold = Variable.New<double>().Named("threshold");
            Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 1).Named("x");
            Variable.ConstrainPositive(x - threshold);

            Gaussian[] expected = new Gaussian[]
                {
                    new Gaussian(0.7979, 0.3634),
                    new Gaussian(0.8626, 0.3422),
                    new Gaussian(0.9294, 0.3221),
                    new Gaussian(0.9982, 0.3031),
                    new Gaussian(1.069, 0.2853),
                    new Gaussian(1.141, 0.2685),
                    new Gaussian(1.215, 0.2527),
                    new Gaussian(1.29, 0.238),
                    new Gaussian(1.367, 0.2241),
                    new Gaussian(1.446, 0.2112)
                };

            // Set parameter on compiled algorithm
            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            for (int i = 0; i < 10; i++)
            {
                threshold.ObservedValue = ((double)i) / 10;
                Console.WriteLine("Dist over x=" + engine.Infer(x));
                Gaussian actual = engine.Infer<Gaussian>(x);
                Assert.True(expected[i].MaxDiff(actual) < 5e-3);
            }
        }

        // TODO: lacks an assertion
        [Fact]
        public void SimpleGaussianHybrid()
        {
            // Model
            double[] data = { 5, 7 };
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100).Named("mean");
            Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1).Named("precision");
            Variable<double> x1 = Variable.GaussianFromMeanAndPrecision(mean, precision);
            x1.AddDefinitionAttribute(new Algorithm(new VariationalMessagePassing()));
            Variable<double> x2 = Variable.GaussianFromMeanAndPrecision(mean, precision);
            x2.AddDefinitionAttribute(new Algorithm(new VariationalMessagePassing()));
            Variable.ConstrainEqual(x1, data[0]);
            Variable.ConstrainEqual(x2, data[1]);

            // Inference
            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            //engine.ShowSchedule = true;
            //engine.BrowserMode = BrowserMode.Always;
            Console.WriteLine("Marginal for mean = " + engine.Infer(mean));
            Console.WriteLine("Marginal for precision = " + engine.Infer(precision));
        }


        [Fact]
        public void TrueSkill()
        {
            double[] playerMeans = { 15, 25, 35 };
            double[] playerVariances = { 8 * 8, 9 * 9, 10 * 10 };
            double tau = 144, beta = 0.0576;
            Variable<double>[] skill = new Variable<double>[playerMeans.Length];
            Variable<double>[] performance = new Variable<double>[playerMeans.Length];
            for (int i = 0; i < playerMeans.Length; i++)
            {
                Variable<double> oldSkill = Variable.GaussianFromMeanAndVariance(playerMeans[i], playerVariances[i]);
                skill[i] = Variable.GaussianFromMeanAndPrecision(oldSkill, tau).Named("skill[" + i + "]");
                performance[i] = Variable.GaussianFromMeanAndPrecision(skill[i], beta).Named("perf[" + i + "]");
                if (i > 0)
                    Variable.ConstrainTrue(performance[i] < performance[i - 1]);
            }

            // Inference
            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            object[] marg = new object[skill.Length];
            // results are compared to the online rank calculator: (0% draw)
            // http://research.microsoft.com/mlp/trueskill/RankCalculator.aspx
            Gaussian[] skillPosts =
                {
                    Gaussian.FromMeanAndVariance(25.592, 6.255*6.255),
                    Gaussian.FromMeanAndVariance(23.611, 6.322*6.322),
                    Gaussian.FromMeanAndVariance(20.165, 7.039*7.039)
                };
            for (int i = 0; i < marg.Length; i++)
            {
                marg[i] = engine.Infer(skill[i]);
                Console.WriteLine("Skills[" + i + "]=" + marg[i]);
                Assert.True(skillPosts[i].MaxDiff(marg[i]) < 0.01);
            }
        }

        //[Fact]
        internal void TrueSkill2()
        {
            double[] playerMeans = { 15, 25, 35 };
            double[] playerVariances = { 8 * 8, 9 * 9, 10 * 10 };
            double beta = 0.0576;
            Variable<double>[] skill = new Variable<double>[playerMeans.Length];
            Variable<double>[] performance = new Variable<double>[playerMeans.Length];
            for (int i = 0; i < playerMeans.Length; i++)
            {
                skill[i] = Variable.GaussianFromMeanAndVariance(playerMeans[i], playerVariances[i]).Named("skill" + i);
                performance[i] = Variable.GaussianFromMeanAndPrecision(skill[i], beta).Named("perf" + i);
                if (i > 0)
                    Variable.ConstrainTrue(performance[i] < performance[i - 1]);
            }

            // Inference
            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            object[] marg = new object[skill.Length];
            for (int i = 0; i < marg.Length; i++)
            {
                marg[i] = engine.Infer(skill[i]);
                Console.WriteLine("Skills[" + i + "]=" + marg[i]);
            }
        }

        internal void TrueSkill3()
        {
            double skillVar = 1000;
            double perfPrec = 1;
            var skill1 = Variable.GaussianFromMeanAndVariance(0, skillVar);
            var skill2 = Variable.GaussianFromMeanAndVariance(0, skillVar);
            int n = 100;
            Range game = new Range(n);
            game.AddAttribute(new Sequential() { BackwardPass = true });
            using (Variable.ForEach(game))
            {
                var perf1 = Variable.GaussianFromMeanAndPrecision(skill1, perfPrec);
                var perf2 = Variable.GaussianFromMeanAndPrecision(skill2, perfPrec);
                Variable.ConstrainTrue(perf1 > perf2);
            }
            Range game2 = new Range(n);
            game2.AddAttribute(new Sequential());
            using (Variable.ForEach(game2))
            {
                var perf1 = Variable.GaussianFromMeanAndPrecision(skill1, perfPrec);
                var perf2 = Variable.GaussianFromMeanAndPrecision(skill2, perfPrec);
                Variable.ConstrainTrue(perf1 < perf2);
            }
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.EnforceTriggers = false;
            Gaussian post1 = engine.Infer<Gaussian>(skill1);
            Gaussian post2 = engine.Infer<Gaussian>(skill2);
            Console.WriteLine(post1);
            Console.WriteLine(post2);
            Console.WriteLine(engine.Infer(skill1 > skill2));
        }

        internal void SymmetryBreakingTest()
        {
            var x1 = Variable.GaussianFromMeanAndVariance(0, 100);
            var x2 = Variable.GaussianFromMeanAndVariance(0, 100);
            Variable.ConstrainEqual(x1 + x2, 2);
            Variable.ConstrainEqual(x1, x2);
            var engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(x1));
            Console.WriteLine(engine.Infer(x1 + x2));
        }

        [Fact]
        public void BossPredictorTest()
        {
            BossPredictor(new ExpectationPropagation());
            BossPredictor(new VariationalMessagePassing());
        }

        [Fact]
        public void GibbsBossPredictorTest()
        {
            BossPredictor(new GibbsSampling());
        }

        private void BossPredictor(IAlgorithm algorithm)
        {
            Variable<bool> raining = Variable.New<bool>().Named("raining");
            Variable<bool> coffee = Variable.Bernoulli(0.6).Named("coffee");

            Variable<bool> temp = (coffee | !raining).Named("temp");
            Variable<bool> goodMood = Variable.New<bool>().Named("goodMood");
            Variable<bool> approvesTrip = Variable.New<bool>().Named("approvesTrip");
            using (Variable.If(temp))
                goodMood.SetTo(Variable.Bernoulli(0.9));
            using (Variable.IfNot(temp))
                goodMood.SetTo(Variable.Bernoulli(0.2));

            using (Variable.If(goodMood))
                approvesTrip.SetTo(Variable.Bernoulli(0.9));
            using (Variable.IfNot(goodMood))
                approvesTrip.SetTo(Variable.Bernoulli(0.4));

            InferenceEngine engine = new InferenceEngine(algorithm);
            double tolerance = 1e-10;
            if (algorithm is GibbsSampling)
            {
                Rand.Restart(12347);
                tolerance = 1e-1;
                engine.NumberOfIterations = 10000;
                engine.ShowProgress = false;
            }

            raining.ObservedValue = true;
            Bernoulli expected = new Bernoulli(0.71);
            Bernoulli actual = engine.Infer<Bernoulli>(approvesTrip);
            Console.WriteLine("Probability of approving trip (raining) = {0} should be {1}", actual, expected);
            Assert.True(expected.MaxDiff(actual) < tolerance);

            raining.ObservedValue = false;
            expected = new Bernoulli(0.85);
            actual = engine.Infer<Bernoulli>(approvesTrip);
            Console.WriteLine("Probability of approving trip (not raining) = {0} should be {1}", actual, expected);
            Assert.True(expected.MaxDiff(actual) < tolerance);
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void GibbsBossPredictorWithGroups()
        {
            Variable<bool> raining = Variable.New<bool>().Named("raining");
            Variable<bool> coffee = Variable.Bernoulli(0.6).Named("coffee");

            Variable<bool> temp = (coffee | !raining).Named("temp");
            Variable<bool> goodMood = Variable.New<bool>().Named("goodMood");
            Variable<bool> approvesTrip = Variable.New<bool>().Named("approvesTrip");
            using (Variable.If(temp))
                goodMood.SetTo(Variable.Bernoulli(0.9));
            using (Variable.IfNot(temp))
                goodMood.SetTo(Variable.Bernoulli(0.2));

            using (Variable.If(goodMood))
                approvesTrip.SetTo(Variable.Bernoulli(0.9));
            using (Variable.IfNot(goodMood))
                approvesTrip.SetTo(Variable.Bernoulli(0.4));

            InferenceEngine engine = new InferenceEngine(new GibbsSampling());
            engine.Group(coffee, temp, goodMood);
            double tolerance = 1e-1;
            engine.NumberOfIterations = 10000;
            engine.ShowProgress = false;

            raining.ObservedValue = true;
            Console.WriteLine("Probability of approving trip (raining)=" + engine.Infer(approvesTrip));
            Bernoulli expected = new Bernoulli(0.71);
            Bernoulli actual = engine.Infer<Bernoulli>(approvesTrip);
            Assert.True(expected.MaxDiff(actual) < tolerance);

            raining.ObservedValue = false;
            Console.WriteLine("Probability of approving trip (not raining)=" + engine.Infer(approvesTrip));
            expected = new Bernoulli(0.85);
            actual = engine.Infer<Bernoulli>(approvesTrip);
            Assert.True(expected.MaxDiff(actual) < tolerance);
        }

        [Fact]
        public void BooleanArrayTest()
        {
            VariableArray<double> probTrue = Variable.Constant(new double[] { 0.1, 0.8 }).Named("probTrue");
            Range n = probTrue.Range;
            VariableArray<bool> notBools = Variable.Array<bool>(n).Named("notBools");
            //notBools[n] = Variable<bool>.Factor(Factor.Not, Variable.Bernoulli(probTrue[n]).Named("bools"));
            notBools[n] = !(Variable.Bernoulli(probTrue[n]).Named("bools"));

            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            object notBoolsActual = engine.Infer(notBools);
            Console.WriteLine("notBools = ");
            Console.WriteLine(notBoolsActual);
            IDistribution<bool[]> notBoolsExpected = Distribution<bool>.Array(new Bernoulli[] { new Bernoulli(0.9), new Bernoulli(0.2) });
            Assert.True(notBoolsExpected.MaxDiff(notBoolsActual) < 1e-10);
        }

        [Fact]
        public void JaggedReplicationTest()
        {
            JaggedReplication(new ExpectationPropagation());
            JaggedReplication(new VariationalMessagePassing());
        }

        [Fact]
        public void GibbsJaggedReplicationTest()
        {
            JaggedReplication(new GibbsSampling());
        }

        private void JaggedReplication(IAlgorithm algorithm)
        {
            Range i = new Range(2).Named("i");
            var jSizes = Variable.Constant(new int[] { 1, 2 }, i).Named("jSizes");
            Range j = new Range(jSizes[i]).Named("j");
            Range k = new Range(2).Named("k");
            var arrayik = Variable.Array<double>(i, k).Named("arrayik");
            arrayik[i, k] = Variable.GaussianFromMeanAndVariance(1, 2).ForEach(i, k);
            //var arrayij = Variable.Array<double>(i,j).Named("arrayij");
            var arrayij = Variable.Array<double>(Variable.Array<double>(j), i).Named("arrayij");
            arrayij[i][j] = Variable.GaussianFromMeanAndVariance(3, 4).ForEach(i, j);
            var arrayikj = Variable.Array<double>(Variable.Array<double>(j), i, k).Named("arrayikj");
            using (Variable.ForEach(i))
            {
                using (Variable.ForEach(j))
                {
                    using (Variable.ForEach(k))
                    {
                        arrayikj[i, k][j] = arrayik[i, k] + arrayij[i][j];
                    }
                }
            }

            double tolerance = (algorithm is GibbsSampling) ? 1e-1 : 1e-10;
            InferenceEngine engine = new InferenceEngine(algorithm);
            Gaussian[,][] arrayikjActual = engine.Infer<Gaussian[,][]>(arrayikj);
            for (int i1 = 0; i1 < arrayikjActual.GetLength(0); i1++)
            {
                for (int k1 = 0; k1 < arrayikjActual.GetLength(1); k1++)
                {
                    for (int j1 = 0; j1 < arrayikjActual[i1, k1].Length; j1++)
                    {
                        Gaussian expected = new Gaussian(4, 6);
                        Console.WriteLine("arrayikj[{0},{1}][{2}] = {3} should be {4}", i1, k1, j1, arrayikjActual[i1, k1][j1], expected);
                        Assert.True(expected.MaxDiff(arrayikjActual[i1, k1][j1]) < tolerance);
                    }
                }
            }
        }

        [Fact]
        public void JaggedReplication2Test()
        {
            JaggedReplication2(new ExpectationPropagation());
            JaggedReplication2(new VariationalMessagePassing());
        }

        [Fact]
        public void GibbsJaggedReplication2Test()
        {
            JaggedReplication2(new GibbsSampling());
        }

        private void JaggedReplication2(IAlgorithm algorithm)
        {
            Range i = new Range(2).Named("i");
            var jSizes = Variable.Constant(new int[] { 1, 2 }, i).Named("jSizes");
            Range j = new Range(jSizes[i]).Named("j");
            Range k = new Range(2).Named("k");
            var arrayik = Variable.Array<double>(i, k).Named("arrayik");
            arrayik[i, k] = Variable.GaussianFromMeanAndVariance(1, 2).ForEach(i, k);
            //var arrayij = Variable.Array<double>(i,j).Named("arrayij");
            var arrayij = Variable.Array<double>(Variable.Array<double>(j), i).Named("arrayij");
            arrayij[i][j] = Variable.GaussianFromMeanAndVariance(3, 4).ForEach(i, j);
            var arrayi = Variable.Array<double>(i).Named("arrayi");
            arrayi[i] = Variable.GaussianFromMeanAndVariance(5, 6).ForEach(i);
            var arrayikj = Variable.Array<double>(Variable.Array<double>(j), i, k).Named("arrayikj");
            using (Variable.ForEach(i))
            {
                using (Variable.ForEach(k))
                {
                    using (Variable.ForEach(j))
                    {
                        arrayikj[i, k][j] = arrayik[i, k] + arrayij[i][j] + arrayi[i];
                    }
                }
            }

            double tolerance = (algorithm is GibbsSampling) ? 1e-1 : 1e-10;
            InferenceEngine engine = new InferenceEngine(algorithm);
            Gaussian[,][] arrayikjActual = engine.Infer<Gaussian[,][]>(arrayikj);
            for (int i1 = 0; i1 < arrayikjActual.GetLength(0); i1++)
            {
                for (int k1 = 0; k1 < arrayikjActual.GetLength(1); k1++)
                {
                    for (int j1 = 0; j1 < arrayikjActual[i1, k1].Length; j1++)
                    {
                        Gaussian expected = new Gaussian(9, 12);
                        Console.WriteLine("arrayikj[{0},{1}][{2}] = {3} should be {4}", i1, k1, j1, arrayikjActual[i1, k1][j1], expected);
                        Assert.True(expected.MaxDiff(arrayikjActual[i1, k1][j1]) < tolerance);
                    }
                }
            }
        }

        // Tests that userBias_use_F_marginal is considered initialized (due to GetItems.MarginalInit)
        [Fact]
        public void InitialiseGetItemsTest()
        {
            // Define counts
            int numUsers = 1;
            Variable<int> numObservations = Variable.Observed(numUsers).Named("numObservations");

            // Define ranges
            Range user = new Range(numUsers).Named("user");
            Range observation = new Range(numObservations).Named("observation");
            observation.AddAttribute(new Sequential());

            // Define latent variables
            var userBias = Variable.Array<double>(user).Named("userBias");
            var itemBias = Variable.GaussianFromMeanAndVariance(0, 1).Named("itemBias");

            // Define latent variables statistically
            userBias[user] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(user);

            // Declare training data variables
            var userData = Variable.Array<int>(observation).Named("userData");
            var affinityData = Variable.Array<double>(observation).Named("affinityData");
            userData.ObservedValue = new int[] { 0 };
            affinityData.ObservedValue = new double[] { 1.0 };

            // Model
            using (Variable.ForEach(observation))
            {
                Variable<double> bias = (userBias[userData[observation]] + itemBias).Named("bias");
                affinityData[observation] = Variable.GaussianFromMeanAndVariance(bias, 1).Named("noisyAffinity");
            }

            userBias.InitialiseTo(Distribution<double>.Array(
                Util.ArrayInit(numUsers, i => Gaussian.PointMass(1))));

            var engine = new InferenceEngine();

            engine.NumberOfIterations = 1;
            var itemBiasPosterior = engine.Infer<Gaussian>(itemBias);
            Gaussian itemBiasExpected = (new Gaussian(0, 1)) * (new Gaussian(0, 1));
            Console.WriteLine("itemBias = {0} should be {1}", itemBiasPosterior, itemBiasExpected);
            Assert.True(itemBiasExpected.MaxDiff(itemBiasPosterior) < 1e-10);
            var userBiasPosterior = engine.Infer<IList<Gaussian>>(userBias);
            Console.WriteLine("userBias = {0}", userBiasPosterior[0]);

            engine.NumberOfIterations = 10;
            userBiasPosterior = engine.Infer<IList<Gaussian>>(userBias);

            // test restarting inference
            engine.NumberOfIterations = 1;
            itemBiasPosterior = engine.Infer<Gaussian>(itemBias);
            Assert.True(itemBiasExpected.MaxDiff(itemBiasPosterior) < 1e-10);
        }

        // Test that loop is made sequential
        [Fact]
        public void HaloSkill()
        {
            Range n = new Range(2).Named("n");
            n.AddAttribute(new Sequential());
            Range w = new Range(1).Named("w");
            Range p = new Range(1).Named("p");

            var Evidence = Variable.Bernoulli(0.5).Named("Evidence");
            var evidenceBlock = Variable.If(Evidence);
            var SkillOffsetWithWeapon = Variable.Array(Variable.Array<double>(w), p).Named("SkillOffsetWithWeapon");
            var Killer = Variable.Array<int>(n).Named("Killer");
            var KillerWeapon = Variable.Array<int>(n).Named("KillerWeapon");
            SkillOffsetWithWeapon[p][w] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(p, w);

            using (Variable.ForEach(n))
            {
                var killer = Killer[n].Named("killer");
                var killerweapon = KillerWeapon[n].Named("killerweapon");
                var killerTotalSkill = SkillOffsetWithWeapon[killer][killerweapon];
                var killerPerformance = Variable.GaussianFromMeanAndVariance(killerTotalSkill, 1.0).Named("killerPerformance");
                Variable.ConstrainPositive(killerPerformance);
            }

            evidenceBlock.CloseBlock();

            Killer.ObservedValue = new int[] { 0, 0 };
            KillerWeapon.ObservedValue = new int[] { 0, 0 };

            var engine = new InferenceEngine();
            engine.NumberOfIterations = 1;
            Gaussian skillExpected = new Gaussian(0.8497, 0.5349);  // 1 iter with sequential
            engine.Compiler.Compiled += delegate (ModelCompiler sender, ModelCompiler.CompileEventArgs e)
            {
                Assert.True(e.Warnings == null || e.Warnings.Count == 0);
            };
            var skillActual = engine.Infer<Gaussian[][]>(SkillOffsetWithWeapon)[0][0];
            Console.WriteLine("skill = {0} should be {1}", skillActual, skillExpected);
            Assert.True(skillExpected.MaxDiff(skillActual) < 2e-4);
        }

        /// <summary>
        /// Tests if batching works by storing internal messages.
        /// </summary>
        [Fact]
        public void BpmBatchingByMessageStoringTest()
        {
            // Obtain the batchable BPM inference algorithms
            InferenceEngine engine = new InferenceEngine();
            // must return copies since the batch messages are being stored by reference
            engine.Compiler.ReturnCopies = true;
            var nonBatchedAlgorithm = engine.GetCompiledInferenceAlgorithm(BatchableBpm());
            var batchedAlgorithm = engine.GetCompiledInferenceAlgorithm(BatchableBpm());

            // Define constants
            const int instanceCount = 6;
            const int featureCount = 2;
            const int batchCount = 3;
            const int batchedInstanceCount = 2;
            const int iterationCount = 20;

            // Define the observed features and labels
            var features = new double[instanceCount][]
            {
                new[] {1.0, -2.0},
                new[] {3.0, 4.0},
                new[] {5.0, -6.0},
                new[] {7.0, 8.0},
                new[] {9.0, -10.0},
                new[] {11.0, 12.0}
            };

            var labels = new double[instanceCount] { -3.0, 11.0, -7.0, 23.0, -11.0, 35.0 };

            // Define the prior used in both the batched and the non-batched models
            var weightPrior = Gaussian.FromMeanAndVariance(0.0, 100.0);

            // Set the observed data in the non-batched BPM
            nonBatchedAlgorithm.SetObservedValue("instanceCount", instanceCount);
            nonBatchedAlgorithm.SetObservedValue("featureCount", featureCount);
            nonBatchedAlgorithm.SetObservedValue("features", features);
            nonBatchedAlgorithm.SetObservedValue("labels", labels);
            nonBatchedAlgorithm.SetObservedValue("weightsPrior", new GaussianArray(weightPrior, featureCount));

            // The initializer in the non-batched BPM is not used, but still needs to be observed
            nonBatchedAlgorithm.SetObservedValue(
                "weightsForInstanceInitializer",
                new GaussianArrayArray(new GaussianArray(Gaussian.Uniform(), featureCount), instanceCount));

            // Define the weights marginal obtained from the non-batched algorithm
            GaussianArray nonBatchedWeights = null;

            // Define variables used in the batching
            var weightsMarginal = new GaussianArray(weightPrior, featureCount);
            var weightsForInstanceMarginal = new GaussianArrayArray(new GaussianArray(Gaussian.Uniform(), featureCount), batchedInstanceCount);
            var batchWeightsMdp = new GaussianArrayArray(new GaussianArray(Gaussian.Uniform(), featureCount), batchCount);
            var batchWeightsForInstanceMdp = Util.ArrayInit(batchCount, x => new GaussianArrayArray(new GaussianArray(Gaussian.Uniform(), featureCount), batchedInstanceCount));
            var currentWeightsPrior = new GaussianArray(Gaussian.Uniform(), featureCount);
            var currentWeightsForInstanceBackwardMessage = new GaussianArrayArray(new GaussianArray(Gaussian.Uniform(), featureCount), batchedInstanceCount);

            // Iterate explicitly
            for (int iteration = 0; iteration < iterationCount; ++iteration)
            {
                // Run inference in the non-batched algorithm
                if (iteration == 0)
                {
                    nonBatchedAlgorithm.Execute(1);
                }
                else
                {
                    nonBatchedAlgorithm.Update(1);
                }

                // Infer the weights marginal from the non-batched algorithm
                nonBatchedWeights = nonBatchedAlgorithm.Marginal<GaussianArray>("weights");

                // Print out the weights learned from the non-batched algorithm
                Console.Write("Iter {0} expected weights: ", iteration);
                for (int feature = 0; feature < featureCount; ++feature)
                {
                    Console.Write(nonBatchedWeights[feature] + " ");
                }
                Console.WriteLine();

                // Batching
                for (int batch = 0; batch < batchCount; ++batch)
                {
                    // Set the observed data for the current batch
                    batchedAlgorithm.SetObservedValue("instanceCount", batchedInstanceCount);
                    batchedAlgorithm.SetObservedValue("featureCount", featureCount);
                    batchedAlgorithm.SetObservedValue("features", new[] { features[2 * batch], features[2 * batch + 1] });
                    batchedAlgorithm.SetObservedValue("labels", new[] { labels[2 * batch], labels[2 * batch + 1] });

                    // Set the weights prior to the ratio of the marginal and output message
                    currentWeightsPrior.SetToRatio(weightsMarginal, batchWeightsMdp[batch], forceProper: true);
                    currentWeightsForInstanceBackwardMessage.SetTo(batchWeightsForInstanceMdp[batch]);

                    // Set the observed values of the weights prior and the backward initializer
                    batchedAlgorithm.SetObservedValue("weightsPrior", currentWeightsPrior);
                    batchedAlgorithm.SetObservedValue("weightsForInstanceInitializer", currentWeightsForInstanceBackwardMessage);

                    // Run inference in the batched algorithm
                    batchedAlgorithm.Execute(1);

                    // Obtain the marginal of the weights and the replicated weights
                    weightsMarginal = batchedAlgorithm.Marginal<GaussianArray>("weights");
                    weightsForInstanceMarginal = batchedAlgorithm.Marginal<GaussianArrayArray>("weightsForInstance");

                    // Obtain the output message from the current batch to the weights and the replicated weights
                    batchWeightsMdp[batch] = batchedAlgorithm.Marginal<GaussianArray>("weights", QueryTypes.MarginalDividedByPrior.Name);
                    batchWeightsForInstanceMdp[batch] = batchedAlgorithm.Marginal<GaussianArrayArray>("weightsForInstance", QueryTypes.MarginalDividedByPrior.Name);
                }

                // Print out the weights learned from the batched algorithm
                Console.Write("Iter {0}  Batched weights: ", iteration);
                for (int feature = 0; feature < featureCount; ++feature)
                {
                    Console.Write(weightsMarginal[feature] + " ");
                }
                Console.WriteLine();

                double error = weightsMarginal.MaxDiff(nonBatchedWeights);
                Console.WriteLine("error = {0}", error);
                //Assert.True(error < 1e-10);
            }
        }

        /// <summary>
        /// Builds a Bayes Point Machine model which allows batching by storing internal messages.
        /// </summary>
        /// <returns>A collection of variables to infer.</returns>
        private static IVariable[] BatchableBpm()
        {
            // Define counts and ranges
            var featureCount = Variable.Observed(default(int)).Named("featureCount");
            var instanceCount = Variable.Observed(default(int)).Named("instanceCount");
            var feature = new Range(featureCount).Named("feature");
            var instance = new Range(instanceCount).Named("instance");

            // Generate a schedule which is sequential over instances
            instance.AddAttribute(new Sequential());

            // Define the weights
            var weightsPrior = Variable.Observed(default(GaussianArray)).Named("weightsPrior");
            var weights = Variable.Array<double>(feature).Named("weights");
            weights.SetTo(Variable<double[]>.Random(weightsPrior));
            weights.AddAttribute(QueryTypes.Marginal);
            weights.AddAttribute(QueryTypes.MarginalDividedByPrior);

            // Define the weight replication inside the instance plate
            var weightsForInstance = Variable.Replicate(weights, instance).Named("weightsForInstance");
            weightsForInstance.AddAttribute(QueryTypes.Marginal);
            weightsForInstance.AddAttribute(QueryTypes.MarginalDividedByPrior);

            // Define the backward initialization of the replicated weights
            var weightsForInstanceInitializer = Variable.Observed(default(GaussianArrayArray)).Named("weightsForInstanceInitializer");
            weightsForInstance.InitialiseBackwardTo(weightsForInstanceInitializer);

            // Define the observed instance data
            var features = Variable.Observed(default(double[][]), instance, feature).Named("features");
            var labels = Variable.Observed(default(double[]), instance).Named("labels");

            // Define the likelihood
            using (Variable.ForEach(instance))
            {
                var products = Variable.Array<double>(feature).Named("products");

                // Use the replicated weight for this instance instead of referring to the original weight
                products[feature] = weightsForInstance[instance][feature] * features[instance][feature];

                var score = Variable.Sum(products).Named("score");
                labels[instance] = Variable.GaussianFromMeanAndVariance(score, 1.0);
            }

            // Return the model variables to infer
            return new IVariable[] { weights, weightsForInstance };
        }
    }


    public class CDARE
    {
        public static void Test()
        {
            var model = new CDARE();
            int workerCount = 10;
            int taskCount = 10;
            model.CreateModel(taskCount, workerCount, 2);
            // [worker][workerTask]
            int[][] taskIndices = Util.ArrayInit(workerCount, worker => Util.ArrayInit(taskCount, workerTask => workerTask));
            // [worker][workerTask]
            int[][] workerLabels = Util.ArrayInit(workerCount, worker => Util.ArrayInit(taskCount, workerTask => (worker > workerCount / 2) ? 0 : 1));
            model.Infer(taskIndices, workerLabels, taskCount);
            Console.WriteLine(StringUtil.ToString(model.CommunityPosterior));
            Assert.False(model.CommunityPosterior[0].IsUniform());
        }

        #region Fields
        // const
        //public const double ABILITY_PRIOR_MEAN = 0;
        //public const double ABILITY_PRIOR_VARIANCE = 1; //50
        public const double DIFFICULTY_PRIOR_MEAN = 0;
        public const double DIFFICULTY_PRIOR_VARIANCE = 1; //50
        public const double DISCRIM_PRIOR_SHAPE = 1;//5;
        public const double DISCRIM_PRIOR_SCALE = 0.0001;//D - 0.01; R - 0.0001; CF - 0.002
        //const int NUMBER_OF_ITERATIONS = 35;//D - 15; R - 35; CF - 35

        // Ranges - size of the variables
        public static Range worker;
        public static Range task;
        public static Range workerTask;
        public static Range choice;
        public static Range M;

        // Main Variables in the model
        public VariableArray<double> workerAbility;
        public VariableArray<double> taskDifficulty;
        public VariableArray<double> discrimination;
        public VariableArray<int> trueLabel;
        public VariableArray<VariableArray<int>, int[][]> workerResponse;

        //community variables
        public VariableArray<int> Community;
        public VariableArray<Discrete> CommunityInit;
        public Variable<Vector> CommunityProb;
        public VariableArray<double> CommunityScore;

        // Prior distributions
        public VariableArray<Gaussian> taskDifficultyPrior;
        public VariableArray<Gamma> discriminationPrior;
        public Variable<Dirichlet> CommunityProbPrior;
        public VariableArray<Gaussian> CommunityScorePrior;

        // Variables in model
        public Variable<int> WorkerCount;
        public VariableArray<int> WorkerTaskCount;
        public VariableArray<VariableArray<int>, int[][]> WorkerTaskIndex;

        // Posterior distributions
        public Gaussian[] workerAbilityPosterior;
        public Gaussian[] taskDifficultyPosterior;
        public Gamma[] discriminationPosterior;
        public static Discrete[] trueLabelPosterior;
        public Discrete[] CommunityPosterior;
        public Dirichlet CommunityProbPosterior;
        public Gaussian[] CommunityScorePosterior;

        // parameters 
        public Variable<int> CommunityCount;
        public int NumbersOfCommunity;
        public double CommunityPseudoCount;
        public double ReliabilityPrecision;
        public int NumberOfIterations;

        // Inference engine
        public InferenceEngine Engine;

        #endregion

        #region Methods
        /// <summary>
        /// Creates a CDARE model instance.
        /// </summary>
        public CDARE()
        {
            NumberOfIterations = 35;
            ReliabilityPrecision = 1;
            NumbersOfCommunity = 4;
            CommunityPseudoCount = 10.0;
        }

        /// <summary>
        /// Initializes the CDARE model.
        /// </summary>
        /// <param name="taskCount">The number of tasks.</param>
        /// <param name="workerCount">The number of workers.</param>
        /// <param name="labelCount">The number of labels.</param>
        public void CreateModel(int taskCount, int workerCount, int labelCount)
        {
            DefineVariablesAndRanges(taskCount, workerCount, labelCount);
            DefineGenerativeProcess();
            DefineInferenceEngine();
        }

        /// <summary>
        /// Defines the variables and the ranges of CDARE.
        /// </summary>
        /// <param name="taskCount">The number of tasks.</param>
        /// <param name="workerCount">The number of workers.</param>
        /// <param name="labelCount">The number of labels.</param>
        public void DefineVariablesAndRanges(int taskCount, int workerCount, int labelCount)
        {
            CommunityCount = Variable.New<int>().Named("CommunityCount");
            M = new Range(CommunityCount).Named("M");
            task = new Range(taskCount).Named("task");
            choice = new Range(labelCount).Named("choice");
            worker = new Range(workerCount).Named("worker");

            // The tasks for each worker
            WorkerTaskCount = Variable.Array<int>(worker).Named("WorkerTaskCount");
            workerTask = new Range(WorkerTaskCount[worker]).Named("workerTask");
            WorkerTaskIndex = Variable.Array(Variable.Array<int>(workerTask), worker).Named("WorkerTaskIndex");
            WorkerTaskIndex.SetValueRange(task);

            // Community membership
            CommunityProbPrior = Variable.New<Dirichlet>().Named("CommunityProbPrior");
            CommunityProb = Variable<Vector>.Random(CommunityProbPrior).Named("CommunityProb");
            CommunityProb.SetValueRange(M);
            Community = Variable.Array<int>(worker).Named("Community"); //.Attrib(QueryTypes.Marginal).Attrib(QueryTypes.MarginalDividedByPrior)
            Community[worker] = Variable.Discrete(CommunityProb).ForEach(worker);
            // Initialiser to break symmetry for community membership
            CommunityInit = Variable.Array<Discrete>(worker).Named("CommunityInit");
            Community[worker].InitialiseTo(CommunityInit[worker]);

            // Community parameters
            CommunityScorePrior = Variable<Gaussian>.Array(M).Named("CommunityScorePrior");
            CommunityScore = Variable.Array<double>(M).Named("CommunityScore");
            CommunityScore[M] = Variable<double>.Random(CommunityScorePrior[M]);

            //worker ability for each worker
            workerAbility = Variable.Array<double>(worker).Named("workerAbility");

            //task difficulty for each task
            taskDifficultyPrior = Variable<Gaussian>.Array(task).Named("taskDifficultyPrior");
            taskDifficulty = Variable.Array<double>(task).Named("taskDifficulty");
            taskDifficulty[task] = Variable<double>.Random(taskDifficultyPrior[task]);

            // discrimination of each task
            discriminationPrior = Variable<Gamma>.Array(task).Named("discriminationPrior");
            discrimination = Variable.Array<double>(task).Named("discrimination");
            discrimination[task] = Variable<double>.Random(discriminationPrior[task]);

            //unobserved true label for each task
            trueLabel = Variable.Array<int>(task).Named("trueLabel");
            trueLabel[task] = Variable.DiscreteUniform(choice).ForEach(task);

            // The labels given by the workers
            workerResponse = Variable.Array(Variable.Array<int>(workerTask), worker).Named("workerResponse");
        }

        /// <summary>
        /// Defines the generative process of CDARE.
        /// </summary>
        public void DefineGenerativeProcess()
        {
            // The process that generates the worker's label
            using (Variable.ForEach(worker))
            {
                using (Variable.Switch(Community[worker]))
                {
                    workerAbility[worker] = Variable.GaussianFromMeanAndPrecision(CommunityScore[Community[worker]], ReliabilityPrecision);
                }

                var workerTaskDifficulty = Variable.Subarray(taskDifficulty, WorkerTaskIndex[worker]);
                var workerTaskDiscrimination = Variable.Subarray(discrimination, WorkerTaskIndex[worker]);
                var TrueLabel = Variable.Subarray(trueLabel, WorkerTaskIndex[worker]);

                using (Variable.ForEach(workerTask))
                {
                    var advantage = (workerAbility[worker] - workerTaskDifficulty[workerTask]).Named("advantage");
                    var advantageNoisy = Variable.GaussianFromMeanAndPrecision(advantage, workerTaskDiscrimination[workerTask]).Named("advantageNoisy");
                    var correct = (advantageNoisy > 0).Named("correct");
                    using (Variable.If(correct))
                        workerResponse[worker][workerTask] = TrueLabel[workerTask];
                    using (Variable.IfNot(correct))
                        workerResponse[worker][workerTask] = Variable.DiscreteUniform(choice);
                }
            }
        }

        /// <summary>
        /// Initializes the CBCC inference engine.
        /// </summary>
        public void DefineInferenceEngine()
        {
            Engine = new InferenceEngine(new ExpectationPropagation());
            Engine.ShowProgress = false;
        }

        /// <summary>
        /// Attachs the data to the workers labels.
        /// </summary>
        /// <param name="taskIndices">The matrix of the task indices (columns) of each worker (rows).</param>
        /// <param name="workerLabels">The matrix of the labels (columns) of each worker (rows).</param>
        public void AttachData(int[][] taskIndices, int[][] workerLabels)
        {
            CommunityCount.ObservedValue = NumbersOfCommunity;
            WorkerTaskCount.ObservedValue = taskIndices.Select(tasks => tasks.Length).ToArray();
            WorkerTaskIndex.ObservedValue = taskIndices;
            workerResponse.ObservedValue = workerLabels;
        }

        /// <summary>
        /// Sets the priors of CDARE.
        /// </summary>
        /// <param name="taskCount">The number of tasks.</param>
        /// <param name="workerCount">The number of workers.</param>
        public void SetPriors(int taskCount, int workerCount)
        {
            taskDifficultyPrior.ObservedValue = Util.ArrayInit(taskCount, t => Gaussian.FromMeanAndPrecision(DIFFICULTY_PRIOR_MEAN, DIFFICULTY_PRIOR_VARIANCE));
            discriminationPrior.ObservedValue = Util.ArrayInit(taskCount, k => Gamma.FromMeanAndVariance(DISCRIM_PRIOR_SHAPE, DISCRIM_PRIOR_SCALE));
            CommunityProbPrior.ObservedValue = Dirichlet.Symmetric(NumbersOfCommunity, CommunityPseudoCount);
            CommunityInit.ObservedValue = Util.ArrayInit(workerCount, worker => Discrete.PointMass(Rand.Int(NumbersOfCommunity), NumbersOfCommunity));
            CommunityScorePrior.ObservedValue = Util.ArrayInit(NumbersOfCommunity, k => Gaussian.FromMeanAndPrecision(1, 1));
        }

        /// <summary>
        /// Infers the posteriors of CDARE using the attached data.
        /// </summary>
        /// <param name="taskIndices">The matrix of the task indices (columns) of each worker (rows).</param>
        /// <param name="workerLabels">The matrix of the labels (columns) of each worker (rows).</param>
        /// <param name="taskCount">The number of tasks.</param>
        /// <returns></returns>
        public void Infer(int[][] taskIndices, int[][] workerLabels, int taskCount)
        {
            SetPriors(taskCount, workerLabels.Length);
            AttachData(taskIndices, workerLabels);
            Engine.NumberOfIterations = NumberOfIterations;

            trueLabelPosterior = Engine.Infer<Discrete[]>(trueLabel);
            workerAbilityPosterior = Engine.Infer<Gaussian[]>(workerAbility);
            taskDifficultyPosterior = Engine.Infer<Gaussian[]>(taskDifficulty);
            discriminationPosterior = Engine.Infer<Gamma[]>(discrimination);
            CommunityScorePosterior = Engine.Infer<Gaussian[]>(CommunityScore);
            CommunityPosterior = Engine.Infer<Discrete[]>(Community);
            CommunityProbPosterior = Engine.Infer<Dirichlet>(CommunityProb);
        }
        #endregion
    }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif
}
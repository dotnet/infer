// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Xunit;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Serialization;

namespace Microsoft.ML.Probabilistic.Tests
{
    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
    using Range = Microsoft.ML.Probabilistic.Models.Range;


    public class SerialTests
    {
#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        internal static void AllTests()
        {
            // to make these tests work, you must set 
            // newMethod = true  in SchedulingTransform
            InferenceEngine.DefaultEngine.Compiler.UseSerialSchedules = true;
            InferenceEngine.DefaultEngine.Compiler.AllowSerialInitialisers = true;

            // DepthCloning tests
            // array_use has fully indexed definition, but used without an index
            //(new MarginalPrototypeTests()).MP_GetItemGamma();
            // skill_use has fully indexed definition, but used without an index
            //(new DocumentationTests()).GameGraphCompact();
            //(new ModelTests()).ArrayUsedAtManyDepths();
            //(new ModelTests()).ArrayUsedAtManyDepths2();
            //(new ModelTests()).JaviOliver();
            // WSharedFeatures_use_uses_F requires cloning
            //(new ModelTests()).HierarchicalBPM();

            // TODO: run all tests with different choices for DivideMessages

            //
            // Non loopy chains
            //
            //(new SerialTests()).SimplestChainTest();
            //(new SerialTests()).SimplestChainWithObservationsTest();
            //(new SerialTests()).TwoIndependentChainsTest();
            //(new SerialTests()).NonGenerativeChainTest();
            //(new SerialTests()).FirstStateConstrainedPositiveChainTest();
            //(new SerialTests()).LastStateConstrainedPositiveChainTest();
            //(new SerialTests()).LastStateConstrainedPositiveChainTest2();
            //(new SerialTests()).JaggedChainsTest();
            //(new SerialTests()).AlternatingChainsTest();
            //(new SerialTests()).OffsetByTwoChainTest();
            //(new SerialTests()).OffsetByTwoChainTest2();
            //(new SerialTests()).OffsetByTwoChainTest3();
            //(new SerialTests()).ChainsWithSharedParameterTest();
            //(new SerialTests()).VectorGaussianChangePointTest();
            //(new SerialTests()).CoinRunLengths();
            //(new SerialTests()).CoinRunLengths2();
            // Doesn't work (but minor)
            //(new SerialTests()).SimplestChainTest2();
            //(new SerialTests()).ChainMiddleRootTest();

            //
            // Loopy chains
            //
            //(new InferTests()).ConstrainBetweenTest4();
            //(new SerialTests()).FadingChannelTest();
            //(new SerialTests()).SkipChainTest();
            //(new SerialTests()).PositiveOffsetSkipChain();
            //(new SerialTests()).TwoThreeSkipChain();
            //(new SerialTests()).EndCoupledChainsTest();
            //(new SerialTests()).EndCoupledChainsTest2();
            //(new SerialTests()).TwoCoupledChainsTest();
            //(new SerialTests()).TwoCoupledChainsTest2();
            //(new SerialTests()).ChainWithTransitionParameterTest();
            //(new SerialTests()).ChainWithTransitionParameterTest2();
            //(new SerialTests()).TrickyCycleChainsTest();

            //
            // Grid models
            //
            //(new SerialTests()).VerticallyConnectedGridTest();  // non-loopy
            //(new SerialTests()).HorizontallyConnectedGridTest(); // non-loopy
            //(new SerialTests()).GridTest();
            //(new SerialTests()).GridNoInitBlockTest();
            //(new SerialTests()).GridTest2();
            //(new SerialTests()).GridTest3();
            //(new SerialTests()).JaggedGridTest();
            //(new SerialTests()).EightGridTest();
            // SkipGrid has an interesting schedule diagram - can see division betw forward and backward loops clearly
            //(new SerialTests()).SkipGridTest();
            //(new SerialTests()).TrickyCycleGridTest();
            // need a test where F and B loops on the same variable are separated by pure F edges on another variable
            // - perhaps a grid with only one vertical edge between rows?
            //(new SerialTests()).GridWithTransitionParameterTest();
            //(new SerialTests()).FadingGridTest();

            //
            // Cube models
            //
            //(new SerialTests()).JaggedCubeTest();

            // Unrolled version throws improper distribution exception with priorVariance=.3, and gridSize >= 5
            // Unrolled version throws improper distribution exception with priorVariance=.5, and gridSize >= 5
            //int gridSize = 10;
            //double priorVariance = .3;
            //(new SerialTests()).FadingGridTestParameterized(priorVariance, .001, priorVariance, gridSize + 2, gridSize);
            //(new SerialTests()).NotFadingGridTestParameterized(priorVariance, .001, priorVariance, gridSize, gridSize, true);
            //int gridSize = 1;
            //for (double priorVariance = .1; priorVariance <= 1.1; priorVariance += .2) {
            //for (double priorVariance = .3; priorVariance <= .3; priorVariance += .2) {
            //(new SerialTests()).FadingGridExperiments(gridSize, 0, 0, priorVariance);
            //(new SerialTests()).FadingGridExperiments(gridSize, gridSize - 1, 0, priorVariance);
            //(new SerialTests()).FadingGridExperiments(gridSize, 0, gridSize - 1, priorVariance);
            //(new SerialTests()).FadingGridExperiments(gridSize, gridSize - 1, gridSize - 1, priorVariance);
            //}
        }

        /// <summary>
        /// Tests a model that needs a fresh node to appear twice in the iteration schedule.
        /// Otherwise, the schedule cannot be initialised.
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")]
        public void FreshChainTest()
        {
            double[] data = new double[] {
                0.592,
                0.708,
                0.789,
                0.621,
                0.873,
                1.074,
                4.634,
                3.945,
                2.118,
                4.207,
                2.884,
                1.462,
                2.851 };

            Range n = new Range(data.Length);

            VariableArray<double> nodes = Variable.Array<double>(n).Named("nodes");
            VariableArray<double> NoisyNodes = Variable.Array<double>(n).Named("NoisyNodes");
            // workaround
            //nodes[n].InitialiseTo(Variable<Gamma>.Factor(Gamma.PointMass, NoisyNodes[n]));

            using (var mblock = Variable.ForEach(n))
            {
                var i = mblock.Index;
                var mIs0 = (i == 0);
                var mIsGr = (i > 0);

                using (Variable.If(mIs0))
                {
                    nodes[n] = Variable.GammaFromMeanAndVariance(1, 1);
                }

                using (Variable.If(mIsGr))
                {
                    var prevM = i - 1;
                    nodes[n] = nodes[prevM] * Variable.GammaFromMeanAndVariance(1, 1);
                }

                NoisyNodes[n] = nodes[n] * Variable.GammaFromMeanAndVariance(1, 1);
            }

            NoisyNodes.ObservedValue = data;

            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());

            Gamma[] postNodes = engine.Infer<Gamma[]>(nodes);

            for (int i = 0; i < n.SizeAsInt; i++)
            {
                Console.WriteLine("{0}: {1}", data[i], postNodes[i]);
            }
        }

        /// <summary>
        /// Test a model where inference fails due to incorrect initial messages.
        /// Fails with "The distribution is improper" because D_uses_F[4] is uniform.
        /// This happens because D_uses_F[day][0] is not updated in the same loop as vdouble16_F which has an offset dependency on it.
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")]
        public void ChainInitializationTest()
        {
            var nDays = Variable.Observed(6).Named("nDays");
            Range day = new Range(nDays).Named("day");
            Range thresh = new Range(2).Named("thresh");

            var R = Variable.Array<bool>(day).Named("R");
            R[day] = Variable.Bernoulli(0.5).ForEach(day);
            Variable<double> wDD = Variable.Random(Gaussian.FromMeanAndVariance(0.0, 1.0)).Named("wDD");
            Variable<double> noiseDPrecision = Variable.Random(Gamma.FromMeanAndVariance(1.0, 1e-2)).Named("noiseDPrecision");
            var threshold = Variable.Array<double>(thresh).Named("threshold");
            threshold[0] = Variable.GaussianFromMeanAndVariance(Double.NegativeInfinity, 0.0);
            threshold[1] = Variable.GaussianFromMeanAndPrecision(0.0, 1.0);

            var D = Variable.Array<double>(day).Named("D");
            using (var block = Variable.ForEach(day))
            {
                var t = block.Index;

                using (Variable.If(t == 0))
                {
                    D[t] = Variable.GaussianFromMeanAndPrecision(0, noiseDPrecision);
                }
                using (Variable.If(t > 0))
                {
                    using (Variable.If(R[t - 1]))
                    {
                        D[t].SetTo(Variable.GaussianFromMeanAndPrecision(wDD * D[t - 1] + 1, noiseDPrecision));
                    }
                    using (Variable.IfNot(R[t - 1]))
                    {
                        D[t].SetTo(Variable.GaussianFromMeanAndPrecision(wDD * D[t - 1], noiseDPrecision));
                    }
                }

                Variable.ConstrainBetween(D[day],
                    threshold[0],
                    threshold[1]);
            }

            // works if D is initialized
            //D[day].InitialiseTo(new Gaussian(0, 1));

            InferenceEngine ie = new InferenceEngine();
            ie.Algorithm = new ExpectationPropagation();
            ie.Compiler.GivePriorityTo(typeof(GaussianProductOp_SHG09));
            //ie.Compiler.GivePriorityTo(typeof(GaussianProductOp_Slow));

            // Compute posterior
            var RPosterior = ie.Infer(R);
        }

        /// <summary>
        /// Tests a model with complex chain dependencies.
        /// Fails with "enemyStrengthAfter is not defined in all cases"
        /// </summary>
        [Fact]
        public void SumForwardBackwardTest()
        {
            var teamCount = Variable.Observed(2).Named("teamCount");
            Range team = new Range(teamCount).Named("team");
            var teamStrength = Variable.Array<double>(team).Named("teamStrength");
            using (Variable.ForEach(team))
            {
                teamStrength[team] = Variable.GaussianFromMeanAndVariance(1, 1);
            }
            var enemyStrengthBefore = Variable.Array<double>(team);
            enemyStrengthBefore.Name = nameof(enemyStrengthBefore);
            enemyStrengthBefore.AddAttribute(new DivideMessages(false));
            using (var block = Variable.ForEach(team))
            {
                var teamIndex = block.Index;
                using (Variable.If(teamIndex == 0))
                {
                    enemyStrengthBefore[teamIndex] = 0.0;
                }
                using (Variable.If(teamIndex > 0))
                {
                    enemyStrengthBefore[teamIndex] = enemyStrengthBefore[teamIndex - 1] + teamStrength[teamIndex - 1];
                }
            }
            var enemyStrengthAfter = Variable.Array<double>(team);
            enemyStrengthAfter.Name = nameof(enemyStrengthAfter);
            var lastIndex = teamCount - 1;
            using (var block = Variable.ForEach(team))
            {
                var teamIndex = block.Index;
                using (Variable.If(teamIndex == lastIndex))
                {
                    enemyStrengthAfter[teamIndex] = 0.0;
                }
                using (Variable.If(teamIndex < lastIndex))
                {
                    enemyStrengthAfter[teamIndex] = enemyStrengthAfter[teamIndex + 1] + teamStrength[teamIndex + 1];
                }
            }
            var enemyStrength = Variable.Array<double>(team).Named("enemyStrength");
            enemyStrength[team] = enemyStrengthBefore[team] + enemyStrengthAfter[team];
            Variable.ConstrainEqual(Variable.Max(enemyStrength[team], 0.0), 0.0);

            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(enemyStrength));
        }

        /// <summary>
        /// Tests a model with complex chain dependencies.
        /// </summary>
        [Fact]
        public void SumForwardBackwardTest2()
        {
            var teamCount = Variable.Observed(2).Named("teamCount");
            Range team = new Range(teamCount).Named("team");
            var teamStrength = Variable.Array<double>(team).Named("teamStrength");
            using (Variable.ForEach(team))
            {
                teamStrength[team] = Variable.GaussianFromMeanAndVariance(1, 1);
            }
            var enemyStrengthBefore = Variable.Array<double>(team);
            enemyStrengthBefore.Name = nameof(enemyStrengthBefore);
            enemyStrengthBefore.AddAttribute(new DivideMessages(false));
            using (var block = Variable.ForEach(team))
            {
                var teamIndex = block.Index;
                using (Variable.If(teamIndex == 0))
                {
                    enemyStrengthBefore[teamIndex] = 0.0;
                }
                using (Variable.If(teamIndex > 0))
                {
                    enemyStrengthBefore[teamIndex] = enemyStrengthBefore[teamIndex - 1] + teamStrength[teamIndex - 1];
                    //enemyStrengthBefore[teamIndex] = Variable<double>.Factor(NoSkip.Plus, enemyStrengthBefore[teamIndex - 1], teamStrength[teamIndex - 1]);
                }
            }
            var enemyStrengthAfter = Variable.Array<double>(team);
            enemyStrengthAfter.Name = nameof(enemyStrengthAfter);
            var lastIndex = teamCount - 1;
            using (var block = Variable.ForEach(team))
            {
                var teamIndex = block.Index;
                using (Variable.If(teamIndex == lastIndex))
                {
                    enemyStrengthAfter[teamIndex] = 0.0;
                }
                using (Variable.If(teamIndex < lastIndex))
                {
                    enemyStrengthAfter[teamIndex] = enemyStrengthAfter[teamIndex + 1] + teamStrength[teamIndex + 1];
                    //enemyStrengthAfter[teamIndex] = Variable<double>.Factor(NoSkip.Plus, enemyStrengthAfter[teamIndex + 1], teamStrength[teamIndex + 1]);
                }
            }
            var enemyStrength = Variable.Array<double>(team).Named("enemyStrength");
            var enemyStrengthBeforeCopy = Variable<double>.Factor(Diode.Copy<double>, enemyStrengthBefore[team]);
            enemyStrengthBeforeCopy.Name = nameof(enemyStrengthBeforeCopy);
            var enemyStrengthAfterCopy = Variable<double>.Factor(Diode.Copy<double>, enemyStrengthAfter[team]);
            enemyStrengthAfterCopy.Name = nameof(enemyStrengthAfterCopy);
            enemyStrength[team] = enemyStrengthBeforeCopy + enemyStrengthAfterCopy;
            //enemyStrength[team] = Variable<double>.Factor(NoSkip.Plus, enemyStrengthBefore[team], enemyStrengthAfter[team]);
            Variable.ConstrainEqual(Variable.Max(enemyStrength[team], 0.0), 0.0);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.AllowSerialInitialisers = false;
            Console.WriteLine(engine.Infer(enemyStrength));
        }

        // Requires an init schedule due to scheduling neutral statements in the backward loop.
        [Fact]
        public void CumulativeSumTest()
        {
            var count = Variable.Observed(2).Named("count");
            Range item = new Range(count).Named("item");
            var x = Variable.Array<double>(item).Named("x");
            using (Variable.ForEach(item))
            {
                x[item] = Variable.GaussianFromMeanAndVariance(1, 1);
            }
            var sumOfPrevious = Variable.Array<double>(item);
            sumOfPrevious.Name = nameof(sumOfPrevious);
            sumOfPrevious.AddAttribute(new DivideMessages(false));
            using (var block = Variable.ForEach(item))
            {
                var index = block.Index;
                using (Variable.If(index == 0))
                {
                    sumOfPrevious[index] = 0.0;
                }
                using (Variable.If(index > 0))
                {
                    sumOfPrevious[index] = sumOfPrevious[index - 1] + x[index - 1];
                }
            }
            Variable.ConstrainEqual(Variable.Max(sumOfPrevious[item], 0.0), 0.0);

            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(x));
        }

        [Fact]
        public void ChangePointTest()
        {
            //double[] data = { 8, 7, -1, 7, 8, 6, 7, 9, 3, 1, 0, 4, 0, 0, 8, 9, 10, 7, 9, 10, 8, 10, 1, 1, 0, 5, 3, 6, 0, 0, 2 };
            double[] data = { 8, 7, -1, 7, 8 };
            bool[] notMissing = Util.ArrayInit(data.Length, i => (data[i] != -1));
            int timesteps = data.Length;
            int numPhases = 4;
            Range time = new Range(timesteps).Named("time");
            Range phase = new Range(numPhases).Named("phase");
            var isObserved = Variable.Array<bool>(time).Named("isObserved");
            isObserved.ObservedValue = notMissing;
            isObserved.IsReadOnly = true;
            var observation = Variable.Array<double>(time).Named("observation");
            observation.ObservedValue = data;
            observation.IsReadOnly = true;

            var phase1Start = Variable.Observed(0).Named("phase1Start");
            var phase2Start = Variable.Observed(1).Named("phase2Start");
            var phase3Start = Variable.Observed(2).Named("phase3Start");

            var intercept = Variable.Array<double>(phase).Named("intercept");
            intercept[phase] = Variable.GaussianFromMeanAndPrecision(0, 1e-4).ForEach(phase);
            var phaseAtTime = Variable.Array<int>(time).Named("phaseAtTime");
            phaseAtTime.SetValueRange(phase);
            phaseAtTime.AddAttribute(new MarginalPrototype(Discrete.Uniform(numPhases)));
            var noisePrecision = Variable.GammaFromMeanAndVariance(1, 1).Named("noisePrecision");
            var noiseAtTime = Variable.Array<double>(time).Named("noiseAtTime");
            var rho = Variable.GaussianFromMeanAndPrecision(0, 1).Named("rho");
            //rho.ObservedValue = 0;  

            using (var block = Variable.ForEach(time))
            {
                var t = block.Index;
                var phase0 = (t < phase1Start).Named("phase0");
                using (Variable.If(phase0))
                {
                    phaseAtTime[t] = 0;
                }
                using (Variable.IfNot(phase0))
                {
                    var phase1 = (t < phase2Start).Named("phase1");
                    using (Variable.If(phase1))
                    {
                        phaseAtTime[t] = 1;
                    }
                    using (Variable.IfNot(phase1))
                    {
                        var phase2 = (t < phase3Start).Named("phase2");
                        using (Variable.If(phase2))
                        {
                            phaseAtTime[t] = 2;
                        }
                        using (Variable.IfNot(phase2))
                        {
                            phaseAtTime[t] = 3;
                        }
                    }
                }
                using (Variable.If(t == 0))
                {
                    noiseAtTime[t] = Variable.GaussianFromMeanAndPrecision(0, noisePrecision);
                }
                using (Variable.If(t > 0))
                {
                    var prevError = noiseAtTime[t - 1] + intercept[phaseAtTime[t - 1]] - intercept[phaseAtTime[t]];
                    noiseAtTime[t] = Variable.GaussianFromMeanAndPrecision(rho * prevError, noisePrecision);
                }
                using (Variable.If(isObserved[t]))
                {
                    using (Variable.Switch(phaseAtTime[t]))
                        observation[t] = intercept[phaseAtTime[t]] + noiseAtTime[t];
                }
            }

            InferenceEngine engine = new InferenceEngine();
            //engine.Algorithm = new VariationalMessagePassing();
            //engine.Compiler.UnrollLoops = true;
            //engine.Compiler.UseSerialSchedules = false;
            //engine.ResetOnObservedValueChanged = false;
            //engine.Compiler.GivePriorityTo(typeof(GaussianProductOp_Laplace));
            //phase1Start.ObservedValue = 8;
            //phase2Start.ObservedValue = 14;
            //phase3Start.ObservedValue = 22;
            Console.WriteLine("intercept:");
            Console.WriteLine(engine.Infer(intercept));
            var rhoActual = engine.Infer<Gaussian>(rho);
            Console.WriteLine("rho = {0}", rhoActual);
            var noiseActual = engine.Infer<Gamma>(noisePrecision);
            Console.WriteLine("noisePrecision = {0}", noiseActual);
        }

        // Fails with "invalid init schedule"
        [Fact]
        [Trait("Category", "OpenBug")]
        public void ChangePointTest2()
        {
            int numPhases = 4;
            Range phase = new Range(numPhases).Named("phase");

            var intercept = Variable.Array<double>(phase).Named("intercept");
            intercept[phase] = Variable.GaussianFromMeanAndPrecision(0, 1e-4).ForEach(phase);
            var noisePrecision = Variable.GammaFromMeanAndVariance(1, 1).Named("noisePrecision");
            //noisePrecision.ObservedValue = 1;
            var rho = Variable.GaussianFromMeanAndPrecision(0, 1).Named("rho");
            //rho.ObservedValue = 0;  

            var phaseAtTime = Variable.New<int>().Named("phaseAtTime");
            phaseAtTime.SetValueRange(phase);
            phaseAtTime.SetTo(0);
            var noise = Variable.GaussianFromMeanAndPrecision(0, noisePrecision);
            var prevNoise = Variable.GaussianFromMeanAndPrecision(rho * noise, noisePrecision);
            Variable<double> observation;
            using (Variable.Switch(phaseAtTime))
                observation = Variable.GaussianFromMeanAndPrecision(rho * prevNoise + intercept[phaseAtTime], noisePrecision);
            observation.ObservedValue = 8;

            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = new VariationalMessagePassing();
            // this is a workaround for the bug
            //prevNoise.InitialiseTo(Gaussian.PointMass(0));

            Console.WriteLine("intercept:");
            Console.WriteLine(engine.Infer(intercept));
            var rhoActual = engine.Infer<Gaussian>(rho);
            Console.WriteLine("rho = {0}", rhoActual);
            var noiseActual = engine.Infer<Gamma>(noisePrecision);
            Console.WriteLine("noisePrecision = {0}", noiseActual);
        }

        // Fails even with loop unrolling because the MSL doesn't preserve correct order of operations
        [Fact]
        [Trait("Category", "OpenBug")]
        public void ChangePointTest3()
        {
            //double[] data = { 8, 7, -1, 7, 8, 6, 7, 9, 3, 1, 0, 4, 0, 0, 8, 9, 10, 7, 9, 10, 8, 10, 1, 1, 0, 5, 3, 6, 0, 0, 2 };
            double[] data = { 8, 7, -1 };
            bool[] notMissing = Util.ArrayInit(data.Length, i => (data[i] != -1));
            int timesteps = data.Length;
            int numPhases = 2;
            Range time = new Range(timesteps).Named("time");
            Range phase = new Range(numPhases).Named("phase");
            var intercept = Variable.Array<double>(phase).Named("intercept");
            intercept[phase] = Variable.GaussianFromMeanAndPrecision(0, 1e-4).ForEach(phase);
            var phaseAtTime = Variable.Array<int>(time).Named("phaseAtTime");
            phaseAtTime.SetValueRange(phase);
            phaseAtTime.AddAttribute(new MarginalPrototype(Discrete.Uniform(numPhases)));
            var noisePrecision = Variable.GammaFromMeanAndVariance(1, 1e-4).Named("noisePrecision");
            var noiseAtTime = Variable.Array<double>(time).Named("noiseAtTime");
            var rho = Variable.GaussianFromMeanAndPrecision(0.5, 100).Named("rho");
            rho.ObservedValue = 0.5;
            var phaseChangeProb = Variable.Beta(1, 1).Named("phaseChangeProb");
            var isObserved = Variable.Array<bool>(time).Named("isObserved");
            isObserved.ObservedValue = notMissing;
            var observation = Variable.Array<double>(time).Named("observation");
            observation.ObservedValue = data;
            using (var block = Variable.ForEach(time))
            {
                var t = block.Index;
                using (Variable.If(t == 0))
                {
                    phaseAtTime[t] = Variable.Random(Discrete.PointMass(0, numPhases));
                    noiseAtTime[t] = Variable.GaussianFromMeanAndPrecision(0, noisePrecision);
                }
                using (Variable.If(t > 0))
                {
                    var isLastPhase = (phaseAtTime[t - 1] == numPhases - 1).Named("isLastPhase");
                    using (Variable.If(isLastPhase)) phaseAtTime[t] = Variable.Copy(phaseAtTime[t - 1]);
                    using (Variable.IfNot(isLastPhase))
                    {
                        var phaseChange = Variable.Bernoulli(phaseChangeProb);
                        using (Variable.If(phaseChange)) phaseAtTime[t] = phaseAtTime[t - 1] + 1;
                        using (Variable.IfNot(phaseChange)) phaseAtTime[t] = Variable.Copy(phaseAtTime[t - 1]);
                    }
                    noiseAtTime[t] = Variable.GaussianFromMeanAndPrecision(rho * noiseAtTime[t - 1], noisePrecision);
                }
                using (Variable.If(isObserved[t]))
                {
                    using (Variable.Switch(phaseAtTime[t]))
                        observation[t] = intercept[phaseAtTime[t]] + noiseAtTime[t];
                }
            }

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.CatchExceptions = true;
            engine.Compiler.UseSerialSchedules = false;
            engine.Compiler.UnrollLoops = true;
            Console.WriteLine("phaseAtTime:");
            Console.WriteLine(engine.Infer(phaseAtTime));
            Console.WriteLine("intercept:");
            Console.WriteLine(engine.Infer(intercept));
        }

        /// <summary>
        /// A simple chain where the whole array is fed into a Copy factor.
        /// </summary>
        [Fact]
        public void SerialArrayCopyTest()
        {
            Range item = new Range(10).Named("i");
            var Z = Variable.Array<double>(item).Named("Z");

            VariableArray<double> S = Variable.Array<double>(item).Named("S");
            using (var bl = Variable.ForEach(item))
            {
                var i = bl.Index;
                using (Variable.If(i == 0))
                {
                    S[i].SetTo(Variable.GaussianFromMeanAndPrecision(0.0, 1.0));
                }
                //using (Variable.If(i != 0))  // fails to compile
                //using (Variable.IfNot(i == 0))   // works
                using (Variable.If(i > 0))    // works
                {
                    S[i].SetTo(Variable.GaussianFromMeanAndPrecision(S[i - 1], 1.0));
                }
            }

            bool useSetTo = true;
            if (useSetTo)
            {
                Z.SetTo(Variable.Copy(S));
            }
            else
            {
                using (Variable.ForEach(item))
                {
                    Z[item] = S[item];
                }
            }

            InferenceEngine engine = new InferenceEngine();
            engine.NumberOfIterations = 1;
            var Zpost = engine.Infer<IList<Gaussian>>(Z);
            Console.WriteLine("Z=" + Zpost);
            Assert.Equal(Zpost[Zpost.Count - 1].GetVariance(), Zpost.Count, 1e-4);
        }

        [Fact]
        public void SequentialError()
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

            var engine = new InferenceEngine();
            engine.Compiler.UseSerialSchedules = false;
            engine.Compiler.TreatWarningsAsErrors = true;
            Assert.Throws<CompilationFailedException>(() =>
            {
                engine.Infer(latentWords);
            });
        }

        [Fact]
        public void CountTrueTest()
        {
            Discrete countExpected = new Discrete(0, 0.63, 0.34, 0.03, 0);
            for (int trial = 0; trial < 2; trial++)
            {
                bool useSerialCount = (trial == 0);
                CountTrue(useSerialCount, 0, countExpected, 0);
                CountTrue(useSerialCount, 1, Discrete.PointMass(2, 5), System.Math.Log(0.34));
                CountTrue(useSerialCount, 2, new Discrete(0, 0.63 * 0.2, 0.34 * 0.3, 0.03 * 0.4, 0), System.Math.Log(0.63 * 0.2 + 0.34 * 0.3 + 0.03 * 0.4));
            }
        }

        private void CountTrue(bool useSerialCount, int obsType, Discrete countExpected, double evExpected)
        {
            var evidence = Variable.Bernoulli(0.5).Named("evidence");
            var block = Variable.If(evidence);
            Range n = new Range(4);
            var coinBiases = Variable.Array<double>(n).Named("coinBiases");
            coinBiases.ObservedValue = new[] { 0.1, 0.3, 0.0, 1.0 };
            var coins = Variable.Array<bool>(n).Named("coins");
            coins[n] = Variable.Bernoulli(coinBiases[n]);
            Variable<int> count;
            if (useSerialCount)
                count = CountTrue(coins);
            else
                count = Variable.CountTrue(coins);
            count.Name = "count";
            if (obsType == 1)
                count.ObservedValue = 2;
            else if (obsType == 2)
                Variable.ConstrainEqualRandom(count, new Discrete(0.1, 0.2, 0.3, 0.4, 0));
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            Discrete countActual = engine.Infer<Discrete>(count);
            if (countExpected.IsPointMass)
                Assert.True(countExpected.Point == countActual.Point);
            else
                Assert.True(countExpected.MaxDiff(countActual) < 1e-6);

            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-8) < 1e-8);
        }

        public Variable<int> CountTrue(VariableArray<bool> bools)
        {
            Range n = bools.Range;
            var sumUpTo = Variable.Array<int>(n).Named("sumUpTo");
            var sizes = Variable.Array<int>(n).Named("sizes");
            Range count = new Range(sizes[n]).Named("c");
            sumUpTo.SetValueRange(count);
            using (var fb = Variable.ForEach(n))
            {
                var i = fb.Index;
                sizes[i] = 2 + i;
                using (Variable.Case(i, 0))
                {
                    using (Variable.If(bools[n]))
                    {
                        sumUpTo[i] = 1;
                    }
                    using (Variable.IfNot(bools[n]))
                    {
                        sumUpTo[i] = 0;
                    }
                }
                using (Variable.If(i > 0))
                {
                    using (Variable.If(bools[n]))
                    {
                        sumUpTo[i] = sumUpTo[i - 1] + 1;
                    }
                    using (Variable.IfNot(bools[n]))
                    {
                        //sumUpTo[i] = Variable.Copy(sumUpTo[i-1]);
                        sumUpTo[i] = sumUpTo[i - 1] + 0;
                    }
                }
            }
            var sum = Variable.Copy(sumUpTo[((Variable<int>)n.Size) - 1]);
            return sum;
        }

        /// <summary>
        /// Test the Max of array example from the documentation
        /// </summary>
        [Fact]
        public void MaxArrayTest()
        {
            Range item = new Range(3);
            VariableArray<double> means = Variable.Observed(new double[] { 0, 0, 0 }, item);
            VariableArray<double> variances = Variable.Observed(new double[] { 1, 1, 1 }, item);
            VariableArray<double> scores = Variable.Array<double>(item);
            scores[item] = Variable.GaussianFromMeanAndVariance(means[item], variances[item]);
            var max = Max(scores);

            InferenceEngine engine = new InferenceEngine();
            var maxActual = engine.Infer<Gaussian>(max);
            Console.WriteLine(maxActual);
        }

        public static Variable<double> Max(VariableArray<double> array)
        {
            Range n = array.Range;
            var maxUpTo = Variable.Array<double>(n).Named("maxUpTo");
            using (var fb = Variable.ForEach(n))
            {
                var i = fb.Index;
                using (Variable.Case(i, 0))
                {
                    maxUpTo[i] = Variable.Copy(array[i]);
                }
                using (Variable.If(i > 0))
                {
                    maxUpTo[i] = Variable.Max(maxUpTo[i - 1], array[i]);
                }
            }
            var max = Variable.Copy(maxUpTo[((Variable<int>)n.Size) - 1]);
            return max;
        }

        public static Variable<double> Min(VariableArray<double> array)
        {
            Range n = array.Range;
            var minUpTo = Variable.Array<double>(n).Named("minUpTo");
            using (var fb = Variable.ForEach(n))
            {
                var i = fb.Index;
                using (Variable.Case(i, 0))
                {
                    minUpTo[i] = Variable.Copy(array[i]);
                }
                using (Variable.If(i > 0))
                {
                    minUpTo[i] = Variable.Min(minUpTo[i - 1], array[i]);
                }
            }
            var min = Variable.Copy(minUpTo[((Variable<int>)n.Size) - 1]);
            return min;
        }

        // VMP loop detection must consider offsets
        // the trigger invalidates a[~i] instead of a[i]
        [Fact]
        [Trait("Category", "OpenBug")]
        public void Regression2()
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
            Variable<double> y = Sum(ytmp).Named("y");
            Variable<double> yNoisy = Variable.GaussianFromMeanAndPrecision(y, 10).Named("yNoisy");
            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
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

        public Variable<double> Sum(VariableArray<double> array)
        {
            Range n = array.Range;
            var sumUpTo = Variable.Array<double>(n).Named("sumUpTo");
            using (var fb = Variable.ForEach(n))
            {
                var i = fb.Index;
                using (Variable.Case(i, 0))
                {
                    sumUpTo[i] = Variable.Copy(array[i]);
                }
                using (Variable.If(i > 0))
                {
                    sumUpTo[i] = sumUpTo[i - 1] + array[i];
                }
            }
            var sum = Variable.Copy(sumUpTo[((Variable<int>)n.Size) - 1]);
            return sum;
        }

        [Fact]
        public void CoinRunLengths()
        {
            // Exact values available at http://mathdl.maa.org/mathDL/22/?pa=content&sa=viewDocument&nodeId=2681
            // p(max run < 2 in 3 flips) = .625
            // p(max run < 4 in 200 flips) = .001
            // p(max run < 5 in 200 flips) = .034
            // p(max run < 6 in 200 flips) = .199
            int numFlips = 3;
            int threshold = 2;
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Range t = new Range(numFlips).Named("t");
            VariableArray<bool> flip = Variable.Array<bool>(t).Named("flip");
            VariableArray<int> runLength = Variable.Array<int>(t).Named("runLength");
            runLength.SetValueRange(new Range(threshold + 1));
            using (ForEachBlock fb = Variable.ForEach(t))
            {
                flip[t] = Variable.Bernoulli(0.5);
                using (Variable.Case(fb.Index, 0))
                {
                    using (Variable.If(flip[t]))
                    {
                        runLength[t] = 1;
                        if (threshold == 1)
                            Variable.ConstrainTrue(Variable.Bernoulli(1e-100));
                    }
                    using (Variable.IfNot(flip[t]))
                    {
                        runLength[t] = 0;
                    }
                }
                using (Variable.If(fb.Index > 0))
                {
                    Variable<int> tPrev = fb.Index - 1;
                    using (Variable.If(flip[t]))
                    {
                        for (int i = 0; i <= threshold; i++)
                        {
                            using (Variable.Case(runLength[tPrev], i))
                            {
                                runLength[t] = System.Math.Min(i + 1, threshold);
                                if (i >= threshold - 1)
                                    Variable.ConstrainTrue(Variable.Bernoulli(1e-100));
                            }
                        }
                    }
                    using (Variable.IfNot(flip[t]))
                    {
                        runLength[t] = 0;
                    }
                }
            }
            block.CloseBlock();
            InferenceEngine engine = new InferenceEngine();
            // this should only require 1 iteration
            engine.NumberOfIterations = 2;
            double evActual = System.Math.Exp(engine.Infer<Bernoulli>(evidence).LogOdds);
            Console.WriteLine("prob of longest run < {0} in {1} flips is {2}", threshold, numFlips, evActual);
            Assert.True(MMath.AbsDiff(evActual, 0.625, 1e-10) < 1e-3);
            if (false)
            {
                var runLengthActual = engine.Infer(runLength);
                Console.WriteLine("runLength = ");
                Console.WriteLine(runLengthActual);
                Console.WriteLine("flip = ");
                Console.WriteLine(engine.Infer(flip));
            }
        }

        // This is the same model as CoinRunLengths but with the gates nested differently
        [Fact]
        public void CoinRunLengths2()
        {
            // Exact values available at http://mathdl.maa.org/mathDL/22/?pa=content&sa=viewDocument&nodeId=2681
            // p(max run < 2 in 3 flips) = .625
            // p(max run < 4 in 200 flips) = .001
            // p(max run < 5 in 200 flips) = .034
            // p(max run < 6 in 200 flips) = .199
            int numFlips = 3;
            int threshold = 2;
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Range t = new Range(numFlips).Named("t");
            VariableArray<bool> flip = Variable.Array<bool>(t).Named("flip");
            VariableArray<int> runLength = Variable.Array<int>(t).Named("runLength");
            runLength.SetValueRange(new Range(threshold + 1));
            using (ForEachBlock fb = Variable.ForEach(t))
            {
                flip[t] = Variable.Bernoulli(0.5);
                using (Variable.If(flip[t]))
                {
                    using (Variable.Case(fb.Index, 0))
                    {
                        runLength[t] = 1;
                        if (threshold == 1)
                            Variable.ConstrainTrue(Variable.Bernoulli(1e-100));
                    }
                    using (Variable.If(fb.Index > 0))
                    {
                        Variable<int> tPrev = fb.Index - 1;
                        for (int i = 0; i <= threshold; i++)
                        {
                            using (Variable.Case(runLength[tPrev], i))
                            {
                                runLength[t] = System.Math.Min(i + 1, threshold);
                                if (i >= threshold - 1)
                                    Variable.ConstrainTrue(Variable.Bernoulli(1e-100));
                            }
                        }
                    }
                }
                using (Variable.IfNot(flip[t]))
                {
                    runLength[t] = 0;
                }
            }
            block.CloseBlock();
            InferenceEngine engine = new InferenceEngine();
            // this should only require 1 iteration
            engine.NumberOfIterations = 2;
            double evActual = System.Math.Exp(engine.Infer<Bernoulli>(evidence).LogOdds);
            Console.WriteLine("prob of longest run < {0} in {1} flips is {2}", threshold, numFlips, evActual);
            Assert.True(MMath.AbsDiff(evActual, 0.625, 1e-10) < 1e-3);
            if (false)
            {
                var runLengthActual = engine.Infer(runLength);
                Console.WriteLine("runLength = ");
                Console.WriteLine(runLengthActual);
                Console.WriteLine("flip = ");
                Console.WriteLine(engine.Infer(flip));
            }
        }


        internal void VectorGaussianChangePointTest()
        {
            double probChangeCluster = 1.0 / 10;
            double tiny = 1e-200; // to avoid AllZeroException
            int numClusters = 2;
            var meanPrior = VectorGaussian.FromMeanAndPrecision(
                Vector.FromArray(2.0, 3.0),
                PositiveDefiniteMatrix.IdentityScaledBy(2, 0.05)
                );
            var precisionPrior = Wishart.FromShapeAndScale(2, 100.0, 0.1);

            Range cluster = new Range(numClusters).Named("cluster");
            var means = Variable.Array<Vector>(cluster).Named("means");
            means[cluster] = Variable<Vector>.Random(meanPrior).ForEach(cluster);
            var precisions = Variable.Array<PositiveDefiniteMatrix>(cluster).Named("variances");
            precisions[cluster] = Variable<PositiveDefiniteMatrix>.Random(precisionPrior).ForEach(cluster);

            var dataCount = Variable.New<int>().Named("dataCount");
            Range item = new Range(dataCount).Named("item");
            var data = Variable.Array<Vector>(item).Named("data");
            var c = Variable.Array<int>(item).Named("c");
            c.SetValueRange(cluster);
            using (ForEachBlock fb = Variable.ForEach(item))
            {
                using (Variable.Case(fb.Index, 0))
                {
                    c[item] = Variable.Discrete(1 - tiny, tiny);
                }
                using (Variable.If(fb.Index > 0))
                {
                    Variable<int> prev = fb.Index - 1;
                    using (Variable.Case(c[prev], 0))
                    {
                        c[item] = Variable.Discrete(1 - probChangeCluster, probChangeCluster);
                    }
                    using (Variable.Case(c[prev], 1))
                    {
                        c[item] = Variable.Discrete(tiny, 1 - tiny);
                    }
                }
            }
            using (Variable.ForEach(item))
            {
                using (Variable.Switch(c[item]))
                {
                    data[item] = Variable.VectorGaussianFromMeanAndPrecision(means[c[item]], precisions[c[item]]);
                }
            }

            PositiveDefiniteMatrix trueVariance = new PositiveDefiniteMatrix(new double[,] { { 10, 1 }, { 1, 10 } });
            VectorGaussian g1 = new VectorGaussian(Vector.FromArray(2, 3), trueVariance);
            VectorGaussian g2 = new VectorGaussian(Vector.FromArray(7, 8), trueVariance);
            data.ObservedValue = Util.ArrayInit(100, i => (i < 50) ? g1.Sample() : g2.Sample());
            dataCount.ObservedValue = data.ObservedValue.Length;

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(engine.Infer(means));
            Console.WriteLine(engine.Infer(c));
        }

        [Fact]
        public void JaggedChainsTest()
        {
            JaggedChains(new ExpectationPropagation());
            //jaggedChainsTest(new VariationalMessagePassing());
            // Gibbs does not yet support SwapIndices
            //jaggedChainsTest(new GibbsSampling());
        }

        private void JaggedChains(IAlgorithm algorithm)
        {
            // FBT attaches SerialLoopInfo to chain, not t
            Range chain = new Range(2).Named("chain");
            VariableArray<int> lengths = Variable.Constant(new int[] { 3, 4 }, chain).Named("lengths");
            Range t = new Range(lengths[chain]).Named("t");
            var states = Variable.Array(Variable.Array<double>(t), chain).Named("states");
            states.AddAttribute(new DivideMessages(false));
            var observation = Variable.Array(Variable.Array<double>(t), chain).Named("observations");

            using (Variable.ForEach(chain))
            {
                using (ForEachBlock rowBlock = Variable.ForEach(t))
                {
                    using (Variable.If(rowBlock.Index == 0))
                    {
                        states[chain][rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                        observation[chain][rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[chain][rowBlock.Index], 1);
                    }
                    using (Variable.If(rowBlock.Index > 0))
                    {
                        states[chain][rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[chain][rowBlock.Index - 1], 1);
                        observation[chain][rowBlock.Index] = Variable.GaussianFromMeanAndPrecision(states[chain][rowBlock.Index], 1);
                    }
                    using (Variable.If(rowBlock.Index == lengths[chain] - 1))
                    {
                        if (algorithm is GibbsSampling)
                        {
                            Variable.ConstrainEqualRandom(states[chain][rowBlock.Index], new Gaussian(2, 1));
                        }
                        else
                        {
                            Variable.ConstrainPositive(states[chain][rowBlock.Index]);
                        }
                    }
                }
            }

            observation.ObservedValue = Util.ArrayInit(chain.SizeAsInt, c => Util.ArrayInit(lengths.ObservedValue[c], i => (double)(i + c)));

            int numberOfIterations = 2;
            double tolerance = 1e-10;
            if (algorithm is GibbsSampling)
            {
                Rand.Restart(12347);
                numberOfIterations = 2000;
                tolerance = 1e-0;
            }
            InferenceEngine engine2 = new InferenceEngine(algorithm);
            engine2.Compiler.UseSerialSchedules = true;
            engine2.NumberOfIterations = numberOfIterations;
            Console.WriteLine("Rolled Up Serial: After {0} Iterations", engine2.NumberOfIterations);
            IDistribution<double[][]> statesSerial = engine2.Infer<IDistribution<double[][]>>(states);

            InferenceEngine engine1 = new InferenceEngine(algorithm);
            engine1.Compiler.UseSerialSchedules = false;
            Console.WriteLine("Rolled Up Parallel: After {0} Iterations", engine1.NumberOfIterations);
            IDistribution<double[][]> statesParallel = engine1.Infer<IDistribution<double[][]>>(states);

            if (false)
            {
                InferenceEngine engine3 = new InferenceEngine(algorithm);
                engine3.Compiler.UnrollLoops = true;
                engine3.NumberOfIterations = numberOfIterations;
                Console.WriteLine("Unrolled: After {0} Iterations", engine3.NumberOfIterations);
                IDistribution<double[][]> statesUnrolled = engine3.Infer<IDistribution<double[][]>>(states);

                Console.WriteLine("MaxDiff(unrolled, parallel) = {0}", statesUnrolled.MaxDiff(statesParallel));
                Console.WriteLine("MaxDiff(unrolled, serial) = {0}", statesUnrolled.MaxDiff(statesSerial));
                Assert.True(statesUnrolled.MaxDiff(statesParallel) < tolerance);
                Assert.True(statesUnrolled.MaxDiff(statesSerial) < tolerance);
                Console.WriteLine("Results match");
            }
            else
            {
                Console.WriteLine("MaxDiff(serial, parallel) = {0}", statesSerial.MaxDiff(statesParallel));
                Assert.True(statesSerial.MaxDiff(statesParallel) < tolerance);
                Console.WriteLine("Results match");
            }
        }

        [Fact]
        public void FadingGridTest()
        {
            double priorVariance = .3;
            double observationVariance = .001;
            double transitionVariance = priorVariance;
            int lengthI = 2;
            int lengthJ = 2;
            Range rows = new Range(lengthI).Named("i");
            Range cols = new Range(lengthJ).Named("j");

            VariableArray2D<double> states = Variable.Array<double>(rows, cols).Named("states");
            VariableArray2D<double> observations = Variable.Array<double>(rows, cols).Named("observations");
            VariableArray2D<bool> symbols = Variable.Array<bool>(rows, cols).Named("symbols");
            states[rows, cols] = Variable.GaussianFromMeanAndVariance(0, 10000).ForEach(rows, cols);

            if (false)
            {
                for (int i = 0; i < 10; i++)
                {
                    Variable.ConstrainEqualRandom(states[rows, cols], new Gaussian());
                }
            }
            if (false)
            {
                Gaussian[,] init = new Gaussian[lengthI, lengthJ];
                for (int i = 0; i < lengthI; i++)
                {
                    for (int j = 0; j < lengthJ; j++)
                    {
                        init[i, j] = new Gaussian(0, 1e5);
                    }
                }
                init[0, 0] = new Gaussian(1, priorVariance);
                states.InitialiseTo(Distribution<double>.Array(init));
            }

            using (ForEachBlock colBlock = Variable.ForEach(cols))
            {
                using (ForEachBlock rowBlock = Variable.ForEach(rows))
                {
                    using (Variable.If(colBlock.Index == 0))
                    {
                        using (Variable.If(rowBlock.Index == 0))
                        {
                            Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index], new Gaussian(1, priorVariance));
                        }
                    }

                    symbols[rowBlock.Index, colBlock.Index] = Variable.Bernoulli(0.5);

                    using (Variable.If(symbols[rowBlock.Index, colBlock.Index]))
                    {
                        //Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index], new Gaussian(-1, observationVariance));
                        observations[rowBlock.Index, colBlock.Index].SetTo(Variable.GaussianFromMeanAndVariance(states[rowBlock.Index, colBlock.Index], observationVariance));
                    }
                    using (Variable.IfNot(symbols[rowBlock.Index, colBlock.Index]))
                    {
                        //Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index], new Gaussian(1, observationVariance));
                        observations[rowBlock.Index, colBlock.Index].SetTo(Variable.GaussianFromMeanAndVariance(-states[rowBlock.Index, colBlock.Index], observationVariance));
                    }

                    using (Variable.If(rowBlock.Index > 0))
                    {
                        Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index] - states[rowBlock.Index - 1, colBlock.Index], new Gaussian(0, transitionVariance));
                    }
                    using (Variable.If(colBlock.Index > 0))
                    {
                        Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index] - states[rowBlock.Index, colBlock.Index - 1], new Gaussian(0, transitionVariance));
                    }
                }
            }

            double[,] observedValues = new double[lengthI, lengthJ];
            for (int i = 0; i < lengthI; i++)
            {
                for (int j = 0; j < lengthJ; j++)
                {
                    observedValues[i, j] = -1;
                }
            }
            observations.ObservedValue = observedValues;

            InferenceEngine engine = new InferenceEngine();
            engine.ModelName = "FadingGrid";
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var result = engine.Infer<Gaussian[,]>(states);

            for (int i = 0; i < lengthI; i++)
            {
                for (int j = 0; j < lengthJ; j++)
                {
                    Console.WriteLine("state[{0}, {1}] = {2}", i, j, result[i, j]);
                }
            }
        }

        [Fact]
        public void SimplestChainTest()
        {
            int length = 10;

            Range rows = new Range(length).Named("i");
            VariableArray<double> states = Variable.Array<double>(rows).Named("states");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                }
                using (Variable.If(rowBlock.Index > 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index - 1], 1);
                }
            }

            InferenceEngine engine = new InferenceEngine();
            engine.NumberOfIterations = 1;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var result1 = engine.Infer<Gaussian[]>(states);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("state[{0}] = {1}", i, result1[i]);
            }
            for (int i = 0; i < length; i++)
            {
                Assert.Equal(result1[i].GetVariance(), i + 1, 1e-8);
            }
            engine.NumberOfIterations = 10;
            int count = 0;
            engine.ProgressChanged += delegate (InferenceEngine sender, InferenceProgressEventArgs progress)
            {
                count++;
            };
            engine.Infer(states);
            Console.WriteLine("iter count = {0} should be 0", count);
            Assert.Equal(0, count);
        }

        // Fails because initialization block doesn't come first
        [Trait("Category", "OpenBug")]
        [Fact]
        public void SimplestChainTest2()
        {
            int length = 10;

            Range rows = new Range(length).Named("i");
            VariableArray<double> states = Variable.Array<double>(rows).Named("states");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index > 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index - 1], 1);
                }
                using (Variable.If(rowBlock.Index == 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                }
            }

            InferenceEngine engine = new InferenceEngine();
            engine.NumberOfIterations = 1;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var result1 = engine.Infer<Gaussian[]>(states);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("state[{0}] = {1}", i, result1[i]);
            }
            for (int i = 0; i < length; i++)
            {
                Assert.Equal(result1[i].GetVariance(), i + 1, 1e-8);
            }
            engine.NumberOfIterations = 10;
            int count = 0;
            engine.ProgressChanged += delegate (InferenceEngine sender, InferenceProgressEventArgs progress)
            {
                count++;
            };
            engine.Infer(states);
            Console.WriteLine("iter count = {0} should be 0", count);
            Assert.Equal(0, count);
        }

        // Fails because (i >= 1) doesn't get inlined properly
        [Trait("Category", "OpenBug")]
        [Fact]
        public void SimplestChainTest3()
        {
            int length = 10;

            Range rows = new Range(length).Named("i");
            VariableArray<double> states = Variable.Array<double>(rows).Named("states");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                }
                using (Variable.If(rowBlock.Index >= 1))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index - 1], 1);
                }
            }

            InferenceEngine engine = new InferenceEngine();
            engine.NumberOfIterations = 1;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var result1 = engine.Infer<Gaussian[]>(states);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("state[{0}] = {1}", i, result1[i]);
            }
            for (int i = 0; i < length; i++)
            {
                Assert.Equal(result1[i].GetVariance(), i + 1, 1e-8);
            }
            engine.NumberOfIterations = 10;
            int count = 0;
            engine.ProgressChanged += delegate (InferenceEngine sender, InferenceProgressEventArgs progress)
            {
                count++;
            };
            engine.Infer(states);
            Console.WriteLine("iter count = {0} should be 0", count);
            Assert.Equal(0, count);
        }

        [Fact]
        public void SimplestChainTest4()
        {
            int length = 10;

            Range rows = new Range(length).Named("i");
            VariableArray<double> states = Variable.Array<double>(rows).Named("states");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                var isFirst = rowBlock.Index == 0;
                using (Variable.If(isFirst))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                }
                using (Variable.IfNot(isFirst))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index - 1], 1);
                }
            }

            InferenceEngine engine = new InferenceEngine();
            engine.NumberOfIterations = 1;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var result1 = engine.Infer<Gaussian[]>(states);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("state[{0}] = {1}", i, result1[i]);
            }
            for (int i = 0; i < length; i++)
            {
                Assert.Equal(result1[i].GetVariance(), i + 1, 1e-8);
            }
            engine.NumberOfIterations = 10;
            int count = 0;
            engine.ProgressChanged += delegate (InferenceEngine sender, InferenceProgressEventArgs progress)
            {
                count++;
            };
            engine.Infer(states);
            Console.WriteLine("iter count = {0} should be 0", count);
            Assert.Equal(0, count);
        }

        [Fact]
        public void SimplestBackwardChainTest()
        {
            int length = 10;

            Range rows = new Range(length).Named("i");
            VariableArray<double> states = Variable.Array<double>(rows).Named("states");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                var index = rowBlock.Index;
                using (Variable.If(index == length - 1))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                }
                using (Variable.If(index < length - 1))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index + 1], 1);
                }
            }

            InferenceEngine engine = new InferenceEngine();
            engine.NumberOfIterations = 1;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var result1 = engine.Infer<Gaussian[]>(states);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("state[{0}] = {1}", i, result1[i]);
            }
            for (int i = 0; i < length; i++)
            {
                Assert.Equal(result1[i].GetVariance(), length - i, 1e-8);
            }
            engine.NumberOfIterations = 10;
            int count = 0;
            engine.ProgressChanged += delegate (InferenceEngine sender, InferenceProgressEventArgs progress)
            {
                count++;
            };
            engine.Infer(states);
            Console.WriteLine("iter count = {0} should be 0", count);
            Assert.Equal(0, count);
        }

        [Fact]
        public void SimplestBackwardChainTest2()
        {
            int length = 10;
            var lengthVar = Variable.Observed(length).Named("length");

            Range rows = new Range(lengthVar).Named("i");
            VariableArray<double> states = Variable.Array<double>(rows).Named("states");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                var index = rowBlock.Index;
                var isLast = (index == lengthVar - 1);
                using (Variable.If(isLast))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                }
                using (Variable.IfNot(isLast))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index + 1], 1);
                }
            }

            InferenceEngine engine = new InferenceEngine();
            engine.NumberOfIterations = 1;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var result1 = engine.Infer<Gaussian[]>(states);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("state[{0}] = {1}", i, result1[i]);
            }
            for (int i = 0; i < length; i++)
            {
                Assert.Equal(result1[i].GetVariance(), length - i, 1e-8);
            }
            engine.NumberOfIterations = 10;
            int count = 0;
            engine.ProgressChanged += delegate (InferenceEngine sender, InferenceProgressEventArgs progress)
            {
                count++;
            };
            engine.Infer(states);
            Console.WriteLine("iter count = {0} should be 0", count);
            Assert.Equal(0, count);
        }

        [Fact]
        public void SimplestBackwardChainTest3()
        {
            int length = 10;
            var lengthVar = Variable.Observed(length).Named("length");

            Range rows = new Range(lengthVar).Named("i");
            VariableArray<double> states = Variable.Array<double>(rows).Named("states");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                var index = rowBlock.Index;
                //var lengthVarMinus1 = (lengthVar - 1).Named("lengthVarMinus1");
                //using (Variable.If(index == lengthVarMinus1))
                using (Variable.If(index == lengthVar - 1))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                }
                //using (Variable.If(index < lengthVarMinus1))
                using (Variable.If(index < lengthVar - 1))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index + 1], 1);
                }
            }

            InferenceEngine engine = new InferenceEngine();
            engine.NumberOfIterations = 1;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var result1 = engine.Infer<Gaussian[]>(states);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("state[{0}] = {1}", i, result1[i]);
            }
            for (int i = 0; i < length; i++)
            {
                Assert.Equal(result1[i].GetVariance(), length - i, 1e-8);
            }
            engine.NumberOfIterations = 10;
            int count = 0;
            engine.ProgressChanged += delegate (InferenceEngine sender, InferenceProgressEventArgs progress)
            {
                count++;
            };
            engine.Infer(states);
            Console.WriteLine("iter count = {0} should be 0", count);
            Assert.Equal(0, count);
        }

        [Fact]
        public void DeterministicChainTest()
        {
            int length = 10;
            var lengthVar = Variable.Observed(length).Named("length");

            Range rows = new Range(lengthVar).Named("i");
            VariableArray<double> states = Variable.Array<double>(rows).Named("states");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                var index = rowBlock.Index;
                using (Variable.If(index == 0))
                {
                    states[rowBlock.Index] = 1;
                }
                using (Variable.If(index > 0))
                {
                    states[rowBlock.Index] = states[rowBlock.Index - 1] + 1;
                }
            }

            InferenceEngine engine = new InferenceEngine();
            engine.NumberOfIterations = 1;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var result1 = engine.Infer<Gaussian[]>(states);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("state[{0}] = {1}", i, result1[i]);
            }
            for (int i = 0; i < length; i++)
            {
                Assert.True(result1[i].IsPointMass);
                Assert.Equal(i + 1, result1[i].Point);
            }
            engine.NumberOfIterations = 10;
            int count = 0;
            engine.ProgressChanged += delegate (InferenceEngine sender, InferenceProgressEventArgs progress)
            {
                count++;
            };
            engine.Infer(states);
            Console.WriteLine("iter count = {0} should be 0", count);
            Assert.Equal(0, count);
        }

        [Fact]
        public void SimplestChainWithObservationsTest()
        {
            int length = 10;

            Range rows = new Range(length).Named("i");
            VariableArray<double> states = Variable.Array<double>(rows).Named("states");
            VariableArray<double> observation = Variable.Array<double>(rows).Named("observations");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                    observation[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index], 1);
                }
                using (Variable.If(rowBlock.Index > 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index - 1], 1);
                    observation[rowBlock.Index] = Variable.GaussianFromMeanAndPrecision(states[rowBlock.Index], 1);
                }
            }

            double[] observationValues = new double[length];
            for (int i = 0; i < length; i++)
            {
                observationValues[i] = i;
            }
            observation.ObservedValue = observationValues;

            InferenceEngine engine = new InferenceEngine();
            engine.NumberOfIterations = 1;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var result1 = engine.Infer<Gaussian[]>(states);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("state[{0}] = {1}", i, result1[i]);
            }

            engine.NumberOfIterations = 10;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var result2 = engine.Infer<Gaussian[]>(states);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("state[{0}] = {1}", i, result2[i]);
            }

            Gaussian[] unrolledResult = SimplestChainWithObservationsUnrolled(engine.Algorithm);

            for (int i = 0; i < length; i++)
            {
                Assert.True(result1[i].MaxDiff(result2[i]) < 1e-10);
                Assert.True(unrolledResult[i].MaxDiff(result2[i]) < 1e-10);
            }
        }

        public Gaussian[] SimplestChainWithObservationsUnrolled(IAlgorithm algorithm)
        {
            int length = 10;
            Variable<double>[] states = new Variable<double>[length];
            Variable<double>[] observation = new Variable<double>[length];

            for (int i = 0; i < length; i++)
            {
                if (i == 0)
                {
                    states[i] = Variable.GaussianFromMeanAndVariance(1, 1);
                    observation[i] = Variable.GaussianFromMeanAndVariance(states[i], 1);
                }
                if (i > 0)
                {
                    states[i] = Variable.GaussianFromMeanAndVariance(states[i - 1], 1);
                    observation[i] = Variable.GaussianFromMeanAndVariance(states[i], 1);
                }
                observation[i].ObservedValue = i;
            }

            InferenceEngine engine = new InferenceEngine(algorithm);
            engine.NumberOfIterations = 1;
            Console.WriteLine();
            Console.WriteLine("==== UNROLLED ====");
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            Gaussian[] result1 = new Gaussian[length];
            for (int i = 0; i < length; i++)
            {
                result1[i] = engine.Infer<Gaussian>(states[i]);
                Console.WriteLine("state[{0}] = {1}", i, result1[i]);
            }

            engine.NumberOfIterations = 10;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            Gaussian[] result2 = new Gaussian[length];
            for (int i = 0; i < length; i++)
            {
                result2[i] = engine.Infer<Gaussian>(states[i]);
                Console.WriteLine("state[{0}] = {1}", i, result2[i]);
            }

            return result2;
        }

        [Fact]
        public void OffsetByTwoChainTest()
        {
            int length = 10;

            Range rows = new Range(length).Named("i");
            VariableArray<double> states = Variable.Array<double>(rows).Named("states");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                }
                using (Variable.If(rowBlock.Index == 1))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 2);
                }
                using (Variable.If(rowBlock.Index > 1))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index - 2], 2);
                }
            }

            InferenceEngine engine = new InferenceEngine();
            engine.NumberOfIterations = 1;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var result1 = engine.Infer<Gaussian[]>(states);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("state[{0}] = {1}", i, result1[i]);
            }
            for (int i = 0; i < length; i++)
            {
                Assert.Equal(result1[i].GetVariance(), i + 1, 1e-8);
            }
            engine.NumberOfIterations = 10;
            int count = 0;
            engine.ProgressChanged += delegate (InferenceEngine sender, InferenceProgressEventArgs progress)
            {
                count++;
            };
            engine.Infer(states);
            Console.WriteLine("iter count = {0} should be 0", count);
            Assert.Equal(0, count);
        }

        [Fact]
        public void OffsetByTwoChainTest2()
        {
            int length = 10;

            Range rows = new Range(length).Named("i");
            VariableArray<double> states = Variable.Array<double>(rows).Named("states");
            // the schedule is non-iterative only if division=false
            states.AddAttribute(new DivideMessages(false));
            Gaussian like = new Gaussian(0, 2);

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                }
                using (Variable.If(rowBlock.Index == 1))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(2, 1);
                }
                using (Variable.If(rowBlock.Index > 1))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(0, 1000);
                    Variable.ConstrainEqualRandom(states[rowBlock.Index - 2] - states[rowBlock.Index], like);
                }
            }

            InferenceEngine engine = new InferenceEngine();
            engine.NumberOfIterations = 1;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var result1 = engine.Infer<IList<Gaussian>>(states);
            engine.NumberOfIterations = 10;
            int count = 0;
            engine.ProgressChanged += delegate (InferenceEngine sender, InferenceProgressEventArgs progress)
            {
                count++;
            };
            var result2 = engine.Infer<IList<Gaussian>>(states);
            Console.WriteLine(StringUtil.JoinColumns("states = ", result1, " should be ", result2));
            Assert.True(((Diffable)result2).MaxDiff(result1) < 1e-10);
            Console.WriteLine("iter count = {0} should be 0", count);
            Assert.Equal(0, count);
        }

        // Same as OffsetByTwoChainTest2 but with mixed positive/negative offsets
        [Fact]
        public void OffsetByTwoChainTest3()
        {
            int length = 10;

            Range rows = new Range(length).Named("i");
            VariableArray<double> states = Variable.Array<double>(rows).Named("states");
            // the schedule is non-iterative only if division=false
            states.AddAttribute(new DivideMessages(false));
            Gaussian like = new Gaussian(0, 2);

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                }
                using (Variable.If(rowBlock.Index == 1))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(2, 1);
                }
                using (Variable.If(rowBlock.Index > 1))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(0, 1000);
                }
                using (Variable.If(rowBlock.Index > 0))
                {
                    using (Variable.If(rowBlock.Index < length - 1))
                    {
                        Variable.ConstrainEqualRandom(states[rowBlock.Index - 1] - states[rowBlock.Index + 1], like);
                    }
                }
            }

            InferenceEngine engine = new InferenceEngine();
            engine.NumberOfIterations = 1;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var result1 = engine.Infer<IList<Gaussian>>(states);
            engine.NumberOfIterations = 10;
            int count = 0;
            engine.ProgressChanged += delegate (InferenceEngine sender, InferenceProgressEventArgs progress)
            {
                count++;
            };
            var result2 = engine.Infer<IList<Gaussian>>(states);
            Console.WriteLine(StringUtil.JoinColumns("states = ", result1, " should be ", result2));
            Assert.True(((Diffable)result2).MaxDiff(result1) < 1e-10);
            Console.WriteLine("iter count = {0} should be 0", count);
            Assert.Equal(0, count);
        }

        [Fact]
        public void TwoIndependentChainsTest()
        {
            int length = 10;

            Range rows = new Range(length).Named("i");
            VariableArray<double> statesA = Variable.Array<double>(rows).Named("statesA");
            VariableArray<double> statesB = Variable.Array<double>(rows).Named("statesB");
            VariableArray<double> observationA = Variable.Array<double>(rows).Named("observationsA");
            VariableArray<double> observationB = Variable.Array<double>(rows).Named("observationsB");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    statesA[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                    observationA[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesA[rowBlock.Index], 1);

                    statesB[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                    observationB[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesB[rowBlock.Index], 1);
                }
                using (Variable.If(rowBlock.Index > 0))
                {
                    statesA[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesA[rowBlock.Index - 1], 1);
                    observationA[rowBlock.Index] = Variable.GaussianFromMeanAndPrecision(statesA[rowBlock.Index], 1);

                    statesB[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesB[rowBlock.Index - 1], 1);
                    observationB[rowBlock.Index] = Variable.GaussianFromMeanAndPrecision(statesB[rowBlock.Index], 1);
                }
            }
            //statesA.AddAttribute(new DivideMessages(false));
            //statesB.AddAttribute(new DivideMessages(false));

            double[] observationValues = new double[length];
            for (int i = 0; i < length; i++)
            {
                observationValues[i] = 1;
            }
            observationA.ObservedValue = observationValues;
            observationB.ObservedValue = observationValues;

            InferenceEngine engine = new InferenceEngine();
            engine.OptimiseForVariables = new List<IVariable>() { statesA, statesB };
            engine.NumberOfIterations = 1;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var resultA1 = engine.Infer<Gaussian[]>(statesA);
            var resultB1 = engine.Infer<Gaussian[]>(statesB);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("stateA[{0}] = {1}", i, resultA1[i]);
                Console.WriteLine("stateB[{0}] = {1}", i, resultB1[i]);
                Assert.True(resultA1[i].MaxDiff(resultB1[i]) < 1e-10);
            }

            engine.NumberOfIterations = 10;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var resultA2 = engine.Infer<Gaussian[]>(statesA);
            var resultB2 = engine.Infer<Gaussian[]>(statesB);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("stateA[{0}] = {1}", i, resultA2[i]);
                Console.WriteLine("stateB[{0}] = {1}", i, resultB2[i]);
            }

            for (int i = 0; i < length; i++)
            {
                Assert.True(resultA1[i].MaxDiff(resultA2[i]) < 1e-10);
                Assert.True(resultB1[i].MaxDiff(resultB2[i]) < 1e-10);
            }
        }

        [Fact]
        public void ChainsWithSharedParameterTest()
        {
            int length = 10;

            Range rows = new Range(length).Named("i");
            Range rows2 = new Range(length).Named("i2");
            VariableArray<double> statesA = Variable.Array<double>(rows).Named("statesA");
            VariableArray<double> statesB = Variable.Array<double>(rows2).Named("statesB");
            VariableArray<double> observationA = Variable.Array<double>(rows).Named("observationsA");
            VariableArray<double> observationB = Variable.Array<double>(rows2).Named("observationsB");
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 1).Named("mean");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    statesA[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(mean, 1);
                    observationA[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesA[rowBlock.Index], 1);
                }
                using (Variable.If(rowBlock.Index > 0))
                {
                    statesA[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesA[rowBlock.Index - 1], 1);
                    observationA[rowBlock.Index] = Variable.GaussianFromMeanAndPrecision(statesA[rowBlock.Index], 1);
                }
            }
            using (ForEachBlock rowBlock = Variable.ForEach(rows2))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    statesB[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(mean, 1);
                    observationB[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesB[rowBlock.Index], 1);
                }
                using (Variable.If(rowBlock.Index > 0))
                {
                    statesB[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesB[rowBlock.Index - 1], 1);
                    observationB[rowBlock.Index] = Variable.GaussianFromMeanAndPrecision(statesB[rowBlock.Index], 1);
                }
            }
            statesA.AddAttribute(new DivideMessages(false));
            statesB.AddAttribute(new DivideMessages(false));

            double[] observationValues = new double[length];
            for (int i = 0; i < length; i++)
            {
                observationValues[i] = 1;
            }
            observationA.ObservedValue = observationValues;
            observationB.ObservedValue = observationValues;

            InferenceEngine engine = new InferenceEngine();
            //engine.Compiler.UnrollLoops = true;
            engine.OptimiseForVariables = new List<IVariable>() { statesA, statesB };
            engine.NumberOfIterations = 1;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var resultA1 = engine.Infer<Gaussian[]>(statesA);
            var resultB1 = engine.Infer<Gaussian[]>(statesB);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("stateA[{0}] = {1}", i, resultA1[i]);
                Console.WriteLine("stateB[{0}] = {1}", i, resultB1[i]);
            }

            engine.NumberOfIterations = 10;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var resultA2 = engine.Infer<Gaussian[]>(statesA);
            var resultB2 = engine.Infer<Gaussian[]>(statesB);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("stateA[{0}] = {1}", i, resultA2[i]);
                Console.WriteLine("stateB[{0}] = {1}", i, resultB2[i]);
            }
            for (int i = 0; i < length; i++)
            {
                Assert.True(System.Math.Abs(resultA1[i].GetVariance() - resultA2[i].GetVariance()) < 1e-10);
                Assert.True(System.Math.Abs(resultB1[i].GetVariance() - resultB2[i].GetVariance()) < 1e-10);
            }
        }

        [Fact]
        public void TwoCoupledChainsTest()
        {
            // removing the spurious init schedule requires either Required attribute on Replicate or making scheduler smarter
            TwoCoupledChains(false);
            TwoCoupledChains(true);
        }
        private void TwoCoupledChains(bool splitObservation)
        {
            int length = 10;

            Range rows = new Range(length).Named("i");
            VariableArray<double> statesA = Variable.Array<double>(rows).Named("statesA");
            VariableArray<double> statesB = Variable.Array<double>(rows).Named("statesB");
            VariableArray<double> observation = Variable.Array<double>(rows).Named("observations");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    statesA[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                    statesB[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                    if (splitObservation)
                    {
                        Variable.ConstrainEqualRandom(statesA[rowBlock.Index] - statesB[rowBlock.Index], new Gaussian(0, 1));
                    }
                }
                using (Variable.If(rowBlock.Index > 0))
                {
                    statesA[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesA[rowBlock.Index - 1], 1);
                    statesB[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesB[rowBlock.Index - 1], 1);
                    if (splitObservation)
                    {
                        Variable.ConstrainEqualRandom(statesA[rowBlock.Index] - statesB[rowBlock.Index], new Gaussian(0, 1));
                    }
                }
                if (!splitObservation)
                {
                    Variable.ConstrainEqualRandom(statesA[rowBlock.Index] - statesB[rowBlock.Index], new Gaussian(0, 1));
                }
                observation[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesA[rowBlock.Index], 1);
            }
            statesA.AddAttribute(new DivideMessages(false));
            statesB.AddAttribute(new DivideMessages(false));

            double[] observationValues = new double[length];
            for (int i = 0; i < length; i++)
            {
                observationValues[i] = 1;
            }
            observation.ObservedValue = observationValues;

            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;
            engine.OptimiseForVariables = new List<IVariable>() { statesA, statesB };
            engine.NumberOfIterations = 2;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var resultA1 = engine.Infer<Gaussian[]>(statesA);
            var resultB1 = engine.Infer<Gaussian[]>(statesB);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("stateA[{0}] = {1}", i, resultA1[i]);
                Console.WriteLine("stateB[{0}] = {1}", i, resultB1[i]);
            }

            engine.NumberOfIterations = 100;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var resultA2 = engine.Infer<Gaussian[]>(statesA);
            var resultB2 = engine.Infer<Gaussian[]>(statesB);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("stateA[{0}] = {1}", i, resultA2[i]);
                Console.WriteLine("stateB[{0}] = {1}", i, resultB2[i]);
            }

            double maxError = Double.NegativeInfinity;
            for (int i = 0; i < length; i++)
            {
                double aError = resultA2[i].MaxDiff(resultA1[i]);
                double bError = resultB2[i].MaxDiff(resultB1[i]);
                maxError = System.Math.Max(maxError, System.Math.Max(aError, bError));
            }
            Console.WriteLine("maxError = {0}", maxError);
            Assert.True(maxError < 0.002);
        }

        [Fact]
        public void TwoCoupledChainsTest2()
        {
            int length = 10;

            Range rows = new Range(length).Named("i");
            VariableArray<double> statesA = Variable.Array<double>(rows).Named("statesA");
            VariableArray<double> statesB = Variable.Array<double>(rows).Named("statesB");
            VariableArray<double> observation = Variable.Array<double>(rows).Named("observations");

            //Variable.ConstrainEqualRandom(statesA[0] - statesB[0], new Gaussian(0, 1));
            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    statesA[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                    statesB[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                    Variable.ConstrainEqualRandom(statesA[rowBlock.Index] - statesB[rowBlock.Index], new Gaussian(0, 1));
                    observation[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesA[rowBlock.Index], 1);
                }
                using (Variable.If(rowBlock.Index > 0))
                {
                    statesA[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesA[rowBlock.Index - 1], 1);
                    statesB[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesB[rowBlock.Index - 1], 1);
                    Variable.ConstrainEqualRandom(statesA[rowBlock.Index] - statesB[rowBlock.Index], new Gaussian(0, 1));
                    observation[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesA[rowBlock.Index], 1);
                }
            }
            statesA.AddAttribute(new DivideMessages(false));
            statesB.AddAttribute(new DivideMessages(false));

            double[] observationValues = new double[length];
            for (int i = 0; i < length; i++)
            {
                observationValues[i] = 1;
            }
            observation.ObservedValue = observationValues;

            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;
            engine.OptimiseForVariables = new List<IVariable>() { statesA, statesB };
            engine.NumberOfIterations = 2;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var resultA1 = engine.Infer<Gaussian[]>(statesA);
            var resultB1 = engine.Infer<Gaussian[]>(statesB);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("stateA[{0}] = {1}", i, resultA1[i]);
                Console.WriteLine("stateB[{0}] = {1}", i, resultB1[i]);
            }

            engine.NumberOfIterations = 1000;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var resultA2 = engine.Infer<Gaussian[]>(statesA);
            var resultB2 = engine.Infer<Gaussian[]>(statesB);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("stateA[{0}] = {1}", i, resultA2[i]);
                Console.WriteLine("stateB[{0}] = {1}", i, resultB2[i]);
            }

            double maxError = Double.NegativeInfinity;
            for (int i = 0; i < length; i++)
            {
                double aError = resultA2[i].MaxDiff(resultA1[i]);
                double bError = resultB2[i].MaxDiff(resultB1[i]);
                maxError = System.Math.Max(maxError, System.Math.Max(aError, bError));
            }
            Console.WriteLine("maxError = {0}", maxError);
            Assert.True(maxError < 0.002);
        }

        [Fact]
        public void EndCoupledChainsTest()
        {
            EndCoupledChains(false);
            EndCoupledChains(true);
        }
        private void EndCoupledChains(bool splitObservation)
        {
            int length = 10;

            Range rows = new Range(length).Named("i");
            VariableArray<double> statesA = Variable.Array<double>(rows).Named("statesA");
            VariableArray<double> statesB = Variable.Array<double>(rows).Named("statesB");
            VariableArray<double> observationA = Variable.Array<double>(rows).Named("observationA");
            VariableArray<double> observationB = Variable.Array<double>(rows).Named("observationB");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    statesA[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                    statesB[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                    if (splitObservation)
                    {
                        observationA[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesA[rowBlock.Index], 1);
                        observationB[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesB[rowBlock.Index], 1);
                    }
                }
                using (Variable.If(rowBlock.Index > 0))
                {
                    statesA[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesA[rowBlock.Index - 1], 1);
                    statesB[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesB[rowBlock.Index - 1], 1);
                    if (splitObservation)
                    {
                        observationA[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesA[rowBlock.Index], 1);
                        observationB[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesB[rowBlock.Index], 1);
                    }
                }
                if (!splitObservation)
                {
                    observationA[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesA[rowBlock.Index], 1);
                    observationB[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesB[rowBlock.Index], 1);
                }
            }
            if (true)
            {
                Variable.ConstrainEqualRandom(statesA[0] - statesB[length - 1], new Gaussian(0, 1));
                Variable.ConstrainEqualRandom(statesA[length - 1] - statesB[0], new Gaussian(0, 1));
            }
            else
            {
                Variable.ConstrainEqualRandom(statesA[0] - statesB[0], new Gaussian(0, 1));
                Variable.ConstrainEqualRandom(statesA[length - 1] - statesB[length - 1], new Gaussian(0, 1));
            }
            statesA.AddAttribute(new DivideMessages(false));
            statesB.AddAttribute(new DivideMessages(false));

            observationA.ObservedValue = Util.ArrayInit(length, i => 1.0);
            observationB.ObservedValue = Util.ArrayInit(length, i => 1.0);

            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;
            engine.OptimiseForVariables = new List<IVariable>() { statesA, statesB };
            engine.NumberOfIterations = 2;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var resultA1 = engine.Infer<Gaussian[]>(statesA);
            var resultB1 = engine.Infer<Gaussian[]>(statesB);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("stateA[{0}] = {1}", i, resultA1[i]);
                Console.WriteLine("stateB[{0}] = {1}", i, resultB1[i]);
            }

            engine.NumberOfIterations = 100;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var resultA2 = engine.Infer<Gaussian[]>(statesA);
            var resultB2 = engine.Infer<Gaussian[]>(statesB);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("stateA[{0}] = {1}", i, resultA2[i]);
                Console.WriteLine("stateB[{0}] = {1}", i, resultB2[i]);
            }

            double maxError = Double.NegativeInfinity;
            for (int i = 0; i < length; i++)
            {
                double aError = resultA2[i].MaxDiff(resultA1[i]);
                double bError = resultB2[i].MaxDiff(resultB1[i]);
                maxError = System.Math.Max(maxError, System.Math.Max(aError, bError));
            }
            Console.WriteLine("maxError = {0}", maxError);
            Assert.True(maxError < 1e-10);
        }

        [Fact]
        public void EndCoupledChainsTest2()
        {
            EndCoupledChains2(true);
            EndCoupledChains2(false);
        }
        private void EndCoupledChains2(bool splitObservation)
        {
            int length = 10;

            Range rows = new Range(length).Named("i");
            Range rowB = new Range(length).Named("j");
            VariableArray<double> statesA = Variable.Array<double>(rows).Named("statesA");
            VariableArray<double> statesB = Variable.Array<double>(rowB).Named("statesB");
            VariableArray<double> observationA = Variable.Array<double>(rows).Named("observationA");
            VariableArray<double> observationB = Variable.Array<double>(rowB).Named("observationB");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    statesA[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                    if (splitObservation)
                    {
                        observationA[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesA[rowBlock.Index], 1);
                    }
                }
                using (Variable.If(rowBlock.Index > 0))
                {
                    statesA[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesA[rowBlock.Index - 1], 1);
                    if (splitObservation)
                    {
                        observationA[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesA[rowBlock.Index], 1);
                    }
                }
                if (!splitObservation)
                {
                    observationA[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesA[rowBlock.Index], 1);
                }
            }
            using (ForEachBlock rowBlock = Variable.ForEach(rowB))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    statesB[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                    if (splitObservation)
                    {
                        observationB[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesB[rowBlock.Index], 1);
                    }
                }
                using (Variable.If(rowBlock.Index > 0))
                {
                    statesB[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesB[rowBlock.Index - 1], 1);
                    if (splitObservation)
                    {
                        observationB[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesB[rowBlock.Index], 1);
                    }
                }
                if (!splitObservation)
                {
                    observationB[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesB[rowBlock.Index], 1);
                }
            }
            Variable.ConstrainEqualRandom(statesA[0] - statesB[length - 1], new Gaussian(0, 1));
            Variable.ConstrainEqualRandom(statesA[length - 1] - statesB[0], new Gaussian(0, 1));
            statesA.AddAttribute(new DivideMessages(false));
            statesB.AddAttribute(new DivideMessages(false));

            observationA.ObservedValue = Util.ArrayInit(length, i => 1.0);
            observationB.ObservedValue = Util.ArrayInit(length, i => 1.0);

            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;
            engine.OptimiseForVariables = new List<IVariable>() { statesA, statesB };
            engine.NumberOfIterations = 2;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var resultA1 = engine.Infer<Gaussian[]>(statesA);
            var resultB1 = engine.Infer<Gaussian[]>(statesB);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("stateA[{0}] = {1}", i, resultA1[i]);
                Console.WriteLine("stateB[{0}] = {1}", i, resultB1[i]);
            }

            engine.NumberOfIterations = 100;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var resultA2 = engine.Infer<Gaussian[]>(statesA);
            var resultB2 = engine.Infer<Gaussian[]>(statesB);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("stateA[{0}] = {1}", i, resultA2[i]);
                Console.WriteLine("stateB[{0}] = {1}", i, resultB2[i]);
            }

            double maxError = Double.NegativeInfinity;
            for (int i = 0; i < length; i++)
            {
                double aError = resultA2[i].MaxDiff(resultA1[i]);
                double bError = resultB2[i].MaxDiff(resultB1[i]);
                maxError = System.Math.Max(maxError, System.Math.Max(aError, bError));
            }
            Console.WriteLine("maxError = {0}", maxError);
            Assert.True(maxError < 1e-10, $"Error was {maxError}");
        }

        [Fact]
        public void AlternatingChainsTest()
        {
            int length = 10;

            Range rows = new Range(length).Named("i");
            VariableArray<double> statesA = Variable.Array<double>(rows).Named("statesA");
            VariableArray<double> statesB = Variable.Array<double>(rows).Named("statesB");
            VariableArray<double> observation = Variable.Array<double>(rows).Named("observations");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    statesA[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                    statesB[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesA[rowBlock.Index], 1);
                    observation[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesB[rowBlock.Index], 1);
                }
                using (Variable.If(rowBlock.Index > 0))
                {
                    statesA[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesB[rowBlock.Index - 1], 1);
                    statesB[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesA[rowBlock.Index], 1);
                    observation[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesB[rowBlock.Index], 1);
                }
            }
            statesA.AddAttribute(new DivideMessages(false));
            statesB.AddAttribute(new DivideMessages(false));

            double[] observationValues = new double[length];
            for (int i = 0; i < length; i++)
            {
                observationValues[i] = 1;
            }
            observation.ObservedValue = observationValues;

            InferenceEngine engine = new InferenceEngine();
            engine.OptimiseForVariables = new List<IVariable>() { statesA, statesB };
            engine.NumberOfIterations = 1;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var resultA1 = engine.Infer<Gaussian[]>(statesA);
            var resultB1 = engine.Infer<Gaussian[]>(statesB);

            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("stateA[{0}] = {1}", i, resultA1[i]);
                Console.WriteLine("stateB[{0}] = {1}", i, resultB1[i]);
            }

            // Unrolling doesn't work properly on this model
            //engine.Compiler.UnrollLoops = true;
            engine.Compiler.UseSerialSchedules = false;
            engine.NumberOfIterations = length;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var resultA2 = engine.Infer<Gaussian[]>(statesA);
            var resultB2 = engine.Infer<Gaussian[]>(statesB);

            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("stateA[{0}] = {1}", i, resultA2[i]);
                Console.WriteLine("stateB[{0}] = {1}", i, resultB2[i]);
            }

            for (int i = 0; i < length; i++)
            {
                Assert.True(resultA1[i].MaxDiff(resultA2[i]) < 1e-10);
                Assert.True(resultB1[i].MaxDiff(resultB2[i]) < 1e-10);
            }
        }

        // This test is tricky because it has a cycle involving offset edges (with zero total offset) but no pure forward or backward loop.
        [Fact]
        public void TrickyCycleChainsTest()
        {
            int length = 10;

            Range rows = new Range(length).Named("i");
            VariableArray<double> statesA = Variable.Array<double>(rows).Named("statesA");
            VariableArray<double> statesB = Variable.Array<double>(rows).Named("statesB");
            VariableArray<double> statesC = Variable.Array<double>(rows).Named("statesC");
            VariableArray<double> observation = Variable.Array<double>(rows).Named("observations");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    statesA[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                    statesB[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                    statesC[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                    observation[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesB[rowBlock.Index], 1);
                }
                using (Variable.If(rowBlock.Index > 0))
                {
                    statesA[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                    statesB[rowBlock.Index] = Variable.GaussianFromMeanAndVariance((statesA[rowBlock.Index - 1] + statesC[rowBlock.Index - 1]).Named("sum"), 1);
                    statesC[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesA[rowBlock.Index], 1);
                    observation[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(statesB[rowBlock.Index], 1);
                }
            }
            statesA.AddAttribute(new DivideMessages(false));
            statesB.AddAttribute(new DivideMessages(false));
            statesC.AddAttribute(new DivideMessages(false));

            double[] observationValues = new double[length];
            for (int i = 0; i < length; i++)
            {
                observationValues[i] = 1;
            }
            observation.ObservedValue = observationValues;

            InferenceEngine engine = new InferenceEngine();
            engine.OptimiseForVariables = new List<IVariable>() { statesA, statesB };
            //engine.NumberOfIterations = 1;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var resultA1 = engine.Infer<Gaussian[]>(statesA);
            var resultB1 = engine.Infer<Gaussian[]>(statesB);

            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("stateA[{0}] = {1}", i, resultA1[i]);
                Console.WriteLine("stateB[{0}] = {1}", i, resultB1[i]);
            }

            // Unrolling doesn't work properly on this model
            //engine.Compiler.UnrollLoops = true;
            engine.Compiler.UseSerialSchedules = false;
            //engine.NumberOfIterations = length;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var resultA2 = engine.Infer<Gaussian[]>(statesA);
            var resultB2 = engine.Infer<Gaussian[]>(statesB);

            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("stateA[{0}] = {1}", i, resultA2[i]);
                Console.WriteLine("stateB[{0}] = {1}", i, resultB2[i]);
            }

            for (int i = 0; i < length; i++)
            {
                Assert.True(resultA1[i].MaxDiff(resultA2[i]) < 1e-10);
                Assert.True(resultB1[i].MaxDiff(resultB2[i]) < 1e-10);
            }
        }

        [Fact]
        public void FirstStateConstrainedPositiveChainTest()
        {
            FirstStateConstrainedPositiveChain(false);
            FirstStateConstrainedPositiveChain(true);
        }
        private void FirstStateConstrainedPositiveChain(bool useDivision)
        {
            int length = 10;

            Range rows = new Range(length).Named("i");
            VariableArray<double> states = Variable.Array<double>(rows).Named("states");
            VariableArray<double> observation = Variable.Array<double>(rows).Named("observations");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                    observation[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index], 1);
                    Variable.ConstrainPositive(states[rowBlock.Index]);
                }
                using (Variable.If(rowBlock.Index > 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index - 1], 1);
                    observation[rowBlock.Index] = Variable.GaussianFromMeanAndPrecision(states[rowBlock.Index], 1);
                }
            }
            states.AddAttribute(new DivideMessages(useDivision));

            double[] observationValues = new double[length];
            for (int i = 0; i < length; i++)
            {
                observationValues[i] = 1;
            }
            observation.ObservedValue = observationValues;

            InferenceEngine engine = new InferenceEngine();
            // Check that convergence is achieved in 2 iterations (if using divide) or 1 (if no divide)
            engine.NumberOfIterations = useDivision ? 2 : 1;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var result = engine.Infer<IList<Gaussian>>(states);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("state[{0}] = {1}", i, result[i]);
            }

            engine.NumberOfIterations = 10;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var result2 = engine.Infer<IList<Gaussian>>(states);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("state[{0}] = {1}", i, result2[i]);
            }
            var error = ((Diffable)result2).MaxDiff(result);
            Console.WriteLine("error = {0}", error);
            Assert.True(error < 1e-4);
        }


        [Fact]
        public void LastStateConstrainedPositiveChainTest()
        {
            // requires updating InitSchedule to use groups
            // states_uses_B[i-1] is updated in the forward loop (where it cannot be loop merged)
            // this is due to DivideMessages
            // _hoist2 is missing an offset dep
            LastStateConstrainedPositiveChain(new ExpectationPropagation());
            //lastStateConstrainedPositiveChainTest(new VariationalMessagePassing());
            // Gibbs does not yet support SwapIndices
            //lastStateConstrainedPositiveChainTest(new GibbsSampling());
        }

        private void LastStateConstrainedPositiveChain(IAlgorithm algorithm)
        {
            int length = 4;

            Range rows = new Range(length).Named("i");
            VariableArray<double> states = Variable.Array<double>(rows).Named("states");
            VariableArray<double> observation = Variable.Array<double>(rows).Named("observations");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                    observation[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index], 1);
                }
                using (Variable.If(rowBlock.Index > 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index - 1], 1);
                    observation[rowBlock.Index] = Variable.GaussianFromMeanAndPrecision(states[rowBlock.Index], 1);
                }
                using (Variable.If(rowBlock.Index == length - 1))
                {
                    if (algorithm is GibbsSampling)
                    {
                        Variable.ConstrainEqualRandom(states[rowBlock.Index], new Gaussian(2, 1));
                    }
                    else
                    {
                        Variable.ConstrainPositive(states[rowBlock.Index]);
                    }
                }
            }

            double[] observationValues = new double[length];
            for (int i = 0; i < length; i++)
            {
                observationValues[i] = 1;
            }
            observation.ObservedValue = observationValues;

            int numberOfIterations = 50;
            double tolerance = 1e-10;
            if (algorithm is GibbsSampling)
            {
                Rand.Restart(12347);
                numberOfIterations = 2000;
                tolerance = 1e-0;
            }
            //states.AddAttribute(new DivideMessages(false));

            InferenceEngine engine2 = new InferenceEngine(algorithm);
            engine2.Compiler.UseSerialSchedules = true;
            engine2.NumberOfIterations = numberOfIterations;
            Console.WriteLine("Rolled Up Serial: After {0} Iterations", engine2.NumberOfIterations);
            IDistribution<double[]> statesSerial = engine2.Infer<IDistribution<double[]>>(states);

            InferenceEngine engine1 = new InferenceEngine(algorithm);
            engine1.Compiler.UseSerialSchedules = false;
            engine1.NumberOfIterations = numberOfIterations;
            Console.WriteLine("Rolled Up Parallel: After {0} Iterations", engine1.NumberOfIterations);
            IDistribution<double[]> statesParallel = engine1.Infer<IDistribution<double[]>>(states);

            InferenceEngine engine3 = new InferenceEngine(algorithm);
            engine3.Compiler.UseSerialSchedules = false;
            engine3.Compiler.UnrollLoops = true;
            engine3.NumberOfIterations = numberOfIterations;
            Console.WriteLine("Unrolled: After {0} Iterations", engine3.NumberOfIterations);
            IDistribution<double[]> statesUnrolled = engine3.Infer<IDistribution<double[]>>(states);

            Console.WriteLine("MaxDiff(unrolled, parallel) = {0}", statesUnrolled.MaxDiff(statesParallel));
            Console.WriteLine("MaxDiff(unrolled, serial) = {0}", statesUnrolled.MaxDiff(statesSerial));
            Assert.True(statesUnrolled.MaxDiff(statesParallel) < tolerance);
            Assert.True(statesUnrolled.MaxDiff(statesSerial) < tolerance);
            Console.WriteLine("Results match");
        }

        /// <summary>
        /// Fails with invalid init schedule.  Same problem as ChangePointTest2
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")]
        public void LastStateConstrainedPositiveChainTest2()
        {
            LastStateConstrainedPositiveChain2(true, true);
            LastStateConstrainedPositiveChain2(false, true);
            LastStateConstrainedPositiveChain2(true, false);
            LastStateConstrainedPositiveChain2(false, false);
        }
        private void LastStateConstrainedPositiveChain2(bool splitObservation, bool useDivision)
        {
            int length = 10;

            Range rows = new Range(length).Named("i");
            VariableArray<double> states = Variable.Array<double>(rows).Named("states");
            VariableArray<double> observation = Variable.Array<double>(rows).Named("observations");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                    if (splitObservation)
                        observation[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index], 1);
                }
                using (Variable.If(rowBlock.Index > 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index - 1], 1);
                    if (splitObservation)
                        observation[rowBlock.Index] = Variable.GaussianFromMeanAndPrecision(states[rowBlock.Index], 1);
                }
                if (!splitObservation)
                    observation[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index], 1);
                if (splitObservation)
                {
                    using (Variable.If(rowBlock.Index == length - 1))
                    {
                        Variable.ConstrainPositive(states[rowBlock.Index]);
                    }
                }
            }
            if (!splitObservation)
                Variable.ConstrainPositive(states[length - 1]);
            states.AddAttribute(new DivideMessages(useDivision));
            double[] observationValues = new double[length];
            for (int i = 0; i < length; i++)
            {
                observationValues[i] = 1;
            }
            observation.ObservedValue = observationValues;

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.UseSerialSchedules = true;
            //engine.NumberOfIterations = 1;
            Console.WriteLine("Rolled Up Serial: After {0} Iterations", engine.NumberOfIterations);
            var result = engine.Infer<Gaussian[]>(states);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("state[{0}] = {1}", i, result[i]);
            }

            InferenceEngine engine3 = new InferenceEngine();
            engine3.Compiler.UseSerialSchedules = false;
            engine3.Compiler.UnrollLoops = true;
            //engine3.NumberOfIterations = 1;
            Console.WriteLine("Unrolled: After {0} Iterations", engine3.NumberOfIterations);
            var result3 = engine3.Infer<Gaussian[]>(states);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("state[{0}] = {1}", i, result3[i]);
            }
            Assert.Equal(0.3819660110574894, result[0].GetVariance(), 1e-2);
            Assert.Equal(0.43769409951740429, result[1].GetVariance(), 1e-2);
            for (int i = 0; i < length; i++)
            {
                Assert.True(result[i].MaxDiff(result3[i]) < 1e-10);
            }

            if (false)
            {
                InferenceEngine engine2 = new InferenceEngine();
                engine2.Compiler.UseSerialSchedules = false;
                engine2.Compiler.UnrollLoops = false;
                engine2.NumberOfIterations = 1;
                Console.WriteLine("Rolled Up Parallel: After {0} Iterations", engine2.NumberOfIterations);
                var result2 = engine2.Infer<Gaussian[]>(states);
                for (int i = 0; i < length; i++)
                {
                    Console.WriteLine("state[{0}] = {1}", i, result2[i]);
                }
            }
        }


        [Fact]
        public void NonGenerativeChainTest()
        {
            int length = 10;
            Range rows = new Range(length).Named("i");
            VariableArray<double> states = Variable.Array<double>(rows).Named("states");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(0, 1000);
                }
                using (Variable.If(rowBlock.Index > 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(0, 1000);
                    Variable.ConstrainEqualRandom(states[rowBlock.Index] - states[rowBlock.Index - 1], new Gaussian(0, 1));
                }
            }

            states.AddAttribute(new DivideMessages(false));
            InferenceEngine engine = new InferenceEngine();
            engine.NumberOfIterations = 1;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var result = engine.Infer<Gaussian[]>(states);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("state[{0}] = {1}", i, result[i]);
            }

            engine.NumberOfIterations = 10;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var result2 = engine.Infer<Gaussian[]>(states);
            for (int i = 0; i < length; i++)
            {
                Console.WriteLine("state[{0}] = {1}", i, result2[i]);
            }
            Assert.Equal(102.82826450007015, result[0].GetVariance(), 1e-2);
            Assert.Equal(102.0330238573348, result[1].GetVariance(), 1e-2);
            for (int i = 0; i < length; i++)
            {
                Assert.True(result[i].MaxDiff(result2[i]) < 1e-10);
            }
        }

        [Fact]
        public void PointEstimateChainTest()
        {
            int length = 2;

            Range rows = new Range(length).Named("i");
            VariableArray<double> states = Variable.Array<double>(rows).Named("states");
            states.AddAttribute(new PointEstimate());
            states[rows].InitialiseTo(Gaussian.PointMass(1));
            VariableArray<Gaussian> observation = Variable.Array<Gaussian>(rows).Named("observations");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 2);
                }
                using (Variable.If(rowBlock.Index > 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index - 1], 1);
                }
            }
            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                Variable.ConstrainEqualRandom(states[rowBlock.Index], observation[rowBlock.Index]);
            }
            observation.ObservedValue = new Gaussian[] { new Gaussian(5, double.PositiveInfinity), new Gaussian(6, 4) };

            Gaussian states0Obs = new Gaussian(1, 2);
            Gaussian[] statesExpected = new Gaussian[] {
                states0Obs * (new Gaussian(6, 5)),
                GaussianFromMeanAndVarianceOp.SampleAverageConditional(states0Obs, 1) * (new Gaussian(6, 4))
            };

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.InitialisationAffectsSchedule = true;
            engine.NumberOfIterations = 150;
            if (false)
            {
                for (int iter = 1; iter < 50; iter++)
                {
                    engine.NumberOfIterations = iter;
                    Console.WriteLine("{0} {1}", iter, engine.Infer(states));
                }
            }
            var statesActual = engine.Infer<IList<Gaussian>>(states);
            //Debug.WriteLine(StringUtil.JoinColumns(statesActual, " ", StringUtil.ArrayToString(statesExpected)));
            for (int i = 0; i < length; i++)
            {
                // The distributions won't match but the means should match.
                double error = MMath.AbsDiff(statesActual[i].GetMean(), statesExpected[i].GetMean(), 1e-10);
                Debug.WriteLine($"statesActual[{i}] = {statesActual[i].GetMean()}, statesExpected[{i}] = {statesExpected[i].GetMean()}, error = {error}");
                Assert.True(error < 1e-8);
            }
        }

        /// <summary>
        /// Vector version of ChainWithTransitionParameterTest3
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")]
        public void ChainWithTransitionParameterVectorTest()
        {
            // without PointEstimator, EP doesn't converge unless length <= 700,
            // where it gets correct mean but variance = 1 (prior variance)
            int length = 1000;
            int dim = 2;
            Vector zero = Vector.Zero(dim);
            Vector one = Vector.Constant(dim, 1);
            PositiveDefiniteMatrix identity = PositiveDefiniteMatrix.Identity(dim);
            PositiveDefiniteMatrix obsPrecision = PositiveDefiniteMatrix.IdentityScaledBy(dim, 1e-8);
            var precision = Variable.WishartFromShapeAndRate(dim, identity).Named("precision");
            //precision.ObservedValue = PositiveDefiniteMatrix.IdentityScaledBy(dim, 0.5);
            precision.AddAttribute(new PointEstimate());
            precision.InitialiseTo(Wishart.PointMass(PositiveDefiniteMatrix.IdentityScaledBy(dim, 1e-8)));
            var precisionDamped = Variable<PositiveDefiniteMatrix>.Factor(Damp.Backward<PositiveDefiniteMatrix>, precision, 1.0);

            Range rows = new Range(length).Named("i");
            // test does not pass with Sequential
            //rows.AddAttribute(new Sequential());
            VariableArray<Vector> states = Variable.Array<Vector>(rows).Named("states");
            VariableArray<Vector> observation = Variable.Array<Vector>(rows).Named("observations");
            Variable<Vector> shift = Variable.VectorGaussianFromMeanAndPrecision(zero, identity).Named("shift");
            //Variable<Vector> shiftPoint = Variable<Vector>.Factor(PointEstimator.Forward<Vector>, shift).Named("shiftPoint");
            //Variable<double> shiftPoint = Variable<double>.Factor(PointEstimator.Forward2<double>, shift, (double)length).Named("shiftPoint");
            //Variable<double> shiftPoint = Variable<double>.Factor(Damp.Forward<double>, shift, 0.1).Named("shiftPoint");
            // when using PointEstimator.Forward with Secant, should always initialize
            //shiftPoint.InitialiseTo(VectorGaussian.PointMass(zero));

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    states[rowBlock.Index] = Variable.VectorGaussianFromMeanAndPrecision(one, identity);
                }
                using (Variable.If(rowBlock.Index > 0))
                {
                    states[rowBlock.Index] = Variable.VectorGaussianFromMeanAndPrecision(states[rowBlock.Index - 1] + shift, precisionDamped);
                }
                observation[rowBlock.Index] = Variable.VectorGaussianFromMeanAndPrecision(states[rowBlock.Index], obsPrecision);
            }
            states.AddAttribute(new DivideMessages(false));
            // Doesn't work without divide because shift_rep_F requires entire shift_rep_B array
            //shift.AddAttribute(new DivideMessages(false));
            //shift.AddAttribute(new PointEstimate());

            //observation.ObservedValue = Util.ArrayInit(length, i => Vector.Constant(dim, i));
            observation.ObservedValue = Util.ArrayInit(length, i => Vector.FromArray(Util.ArrayInit(dim, j => (double)i + j)));

            InferenceEngine engine = new InferenceEngine();
            //engine.Algorithm = new VariationalMessagePassing();
            engine.Compiler.FreeMemory = false;
            engine.Compiler.ReturnCopies = true;
            //engine.Compiler.GivePriorityTo(typeof(VariablePointOp_Secant));
            engine.Compiler.GivePriorityTo(typeof(VariablePointOp_Mean<>));
            engine.OptimiseForVariables = new List<IVariable>() { states, shift, precision };
            engine.NumberOfIterations = 10;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var statesActual1 = engine.Infer<VectorGaussian[]>(states);
            bool showStates = false;
            if (showStates)
            {
                for (int i = 0; i < System.Math.Min(5, length); i++)
                {
                    Console.WriteLine("state[{0}] = {1}", i, statesActual1[i]);
                }
            }
            var shiftActual1 = engine.Infer<VectorGaussian>(shift).Clone();
            Console.WriteLine("shift = {0}", shiftActual1);

            int maxIter = 200;
            if (!engine.ShowProgress)
            {
                for (int iter = 1; iter <= maxIter; iter++)
                {
                    engine.NumberOfIterations = iter;
                    Debug.WriteLine(StringUtil.JoinColumns(iter, " shift=", engine.Infer(shift), " precision=", engine.Infer(precision)));
                    var shiftTemp = engine.Infer<VectorGaussian>(shift);
                    Debug.WriteLine(shiftTemp.GetMean().ToString("f16"));
                }
            }

            engine.NumberOfIterations = maxIter;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var statesActual2 = engine.Infer<VectorGaussian[]>(states);
            if (showStates)
            {
                for (int i = 0; i < System.Math.Min(5, length); i++)
                {
                    Console.WriteLine("state[{0}] = {1}", i, statesActual2[i]);
                }
            }
            var shiftExpected = new VectorGaussian(Vector.FromArray(0.7674549267784490, 0.7686083171516320),
                new PositiveDefiniteMatrix(new double[,] { { 0.2314, -1.459e-06 }, { -1.459e-06, 0.2314 } }));
            var shiftActual2 = engine.Infer<VectorGaussian>(shift);
            Console.WriteLine(StringUtil.JoinColumns("shift = ", shiftActual2, " should be ", shiftExpected));
            Console.WriteLine("maxDiff = {0}", shiftExpected.MaxDiff(shiftActual2));
            //Assert.True(shiftExpected.MaxDiff(shiftActual2) < 1e-3);
            // TODO: make this tighter
            Assert.True(shiftExpected.GetMean().MaxDiff(shiftActual2.GetMean()) < 1e-3);
            var precisionExpected = new PositiveDefiniteMatrix(new double[,] { { 2.001, -0.0001768 }, { -0.0001768, 2.001 } });
            var precisionActual = engine.Infer<Wishart>(precision);
            Console.WriteLine(StringUtil.JoinColumns("precision = ", precisionActual, " should be ", precisionExpected));
            Console.WriteLine("maxDiff = {0}", precisionExpected.MaxDiff(precisionActual.Point));
            Assert.True(precisionExpected.MaxDiff(precisionActual.Point) < 1e-4);

            double maxError = shiftActual2.MaxDiff(shiftActual1);
            for (int i = 0; i < length; i++)
            {
                maxError = System.Math.Max(maxError, statesActual1[i].MaxDiff(statesActual2[i]));
            }
            Console.WriteLine("maxError = {0}", maxError);
            //Assert.True(maxError < 1e-4);
        }

        [Fact]
        public void ChainWithTransitionParameterTest()
        {
            // without PointEstimate, EP doesn't converge unless length <= 700,
            // where it gets correct mean but variance = 1 (prior variance)
            int length = 1000;

            Range rows = new Range(length).Named("i");
            // test does not pass with Sequential
            //rows.AddAttribute(new Sequential());
            VariableArray<double> states = Variable.Array<double>(rows).Named("states");
            VariableArray<double> observation = Variable.Array<double>(rows).Named("observations");
            Variable<double> shift = Variable.GaussianFromMeanAndVariance(0, 1).Named("shift");
            shift.AddAttribute(new PointEstimate());
            shift.InitialiseTo(Gaussian.PointMass(0));

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                }
                using (Variable.If(rowBlock.Index > 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index - 1] + shift, 1);
                }
                observation[rowBlock.Index] = Variable.GaussianFromMeanAndPrecision(states[rowBlock.Index], 1e-8);
            }
            states.AddAttribute(new DivideMessages(false));
            // Doesn't work without divide because shift_rep_F requires entire shift_rep_B array
            //shift.AddAttribute(new DivideMessages(false));

            observation.ObservedValue = Util.ArrayInit(length, i => (double)i);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.InitialisationAffectsSchedule = true;
            //engine.Algorithm = new VariationalMessagePassing();
            engine.Compiler.FreeMemory = false;
            //engine.Compiler.GivePriorityTo(typeof(VariablePointOp_Secant));
            engine.OptimiseForVariables = new List<IVariable>() { states, shift };
            engine.NumberOfIterations = 3;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var statesActual1 = engine.Infer<Gaussian[]>(states);
            bool showStates = false;
            if (showStates)
            {
                for (int i = 0; i < System.Math.Min(5, length); i++)
                {
                    Console.WriteLine("state[{0}] = {1}", i, statesActual1[i]);
                }
            }
            Gaussian shiftActual1 = engine.Infer<Gaussian>(shift);
            Console.WriteLine("shift = {0}", shiftActual1);

            int maxIter = 100;
            if (!engine.ShowProgress)
            {
                for (int iter = 1; iter < maxIter; iter++)
                {
                    engine.NumberOfIterations = iter;
                    Console.WriteLine("{0} shift={1}", iter, engine.Infer<Gaussian>(shift).Point);
                }
            }

            engine.NumberOfIterations = maxIter;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var statesActual2 = engine.Infer<Gaussian[]>(states);
            if (showStates)
            {
                for (int i = 0; i < System.Math.Min(5, length); i++)
                {
                    Console.WriteLine("state[{0}] = {1}", i, statesActual2[i]);
                }
            }
            double shiftMeanExpected = 0.76710095315961;
            Gaussian shiftExpected = new Gaussian(shiftMeanExpected, 0.2317);
            Gaussian shiftActual2 = engine.Infer<Gaussian>(shift);
            Console.WriteLine("shift = {0} should be {1}", shiftActual2, shiftExpected);
            //Gaussian shiftExpected = new Gaussian(0.8465, 0.2051);
            //Assert.True(shiftExpected.MaxDiff(shiftActual2) < 1e-3);
            Assert.True(MMath.AbsDiff(shiftMeanExpected, shiftActual2.GetMean(), 1e-10) < 1e-10);
            //Gaussian statesExpected9 = new Gaussian(8.905, 0.6521);
            //Assert.True(statesExpected9.MaxDiff(statesActual2[9]) < 1e-3);

            double maxError = shiftActual2.MaxDiff(shiftActual1);
            for (int i = 0; i < length; i++)
            {
                maxError = System.Math.Max(maxError, statesActual1[i].MaxDiff(statesActual2[i]));
            }
            Console.WriteLine("maxError = {0}", maxError);
            //Assert.True(maxError < 1e-4);
        }

        [Fact]
        public void ChainWithTwoTransitionParameters()
        {
            int length = 1000;

            Range rows = new Range(length).Named("i");
            // test does not pass with Sequential
            //rows.AddAttribute(new Sequential());
            VariableArray<double> states = Variable.Array<double>(rows).Named("states");
            VariableArray<double> observation = Variable.Array<double>(rows).Named("observations");
            Variable<double> shift = Variable.GaussianFromMeanAndVariance(0, 1).Named("shift");
            Variable<double> shift2 = Variable.GaussianFromMeanAndVariance(0, 1).Named("shift2");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                    observation[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index], 1);
                }
                using (Variable.If(rowBlock.Index > 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index - 1] + shift + shift2, 1);
                    observation[rowBlock.Index] = Variable.GaussianFromMeanAndPrecision(states[rowBlock.Index], 1e-8);
                }
            }
            states.AddAttribute(new DivideMessages(false));
            //shift.AddAttribute(new DivideMessages(false));
            //shift2.AddAttribute(new DivideMessages(false));
            shift.AddAttribute(new PointEstimate());
            shift.InitialiseTo(Gaussian.PointMass(0));
            shift2.AddAttribute(new PointEstimate());
            shift2.InitialiseTo(Gaussian.PointMass(0));

            double[] observationValues = new double[length];
            for (int i = 0; i < length; i++)
            {
                observationValues[i] = i;
            }
            observation.ObservedValue = observationValues;

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.InitialisationAffectsSchedule = true;
            engine.ShowProgress = false;
            //engine.Compiler.GivePriorityTo(typeof(VariablePointOp_Secant));
            engine.OptimiseForVariables = new List<IVariable>() { states, shift, shift2 };
            engine.NumberOfIterations = 3;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var statesActual1 = engine.Infer<Gaussian[]>(states);
            bool showStates = false;
            if (showStates)
            {
                for (int i = 0; i < length; i++)
                {
                    Console.WriteLine("state[{0}] = {1}", i, statesActual1[i]);
                }
            }
            Gaussian shiftActual1 = engine.Infer<Gaussian>(shift);
            Gaussian shift2Actual1 = engine.Infer<Gaussian>(shift2);
            Console.WriteLine("shift = {0}, shift2 = {1}", shiftActual1, shift2Actual1);

            int maxIter = engine.Compiler.OptimiseInferenceCode ? 100 : 200;
            if (!engine.ShowProgress)
            {
                for (int iter = 1; iter < maxIter; iter++)
                {
                    engine.NumberOfIterations = iter;
                    Console.WriteLine("{0} shift={1} shift2={2}", iter, engine.Infer(shift), engine.Infer(shift2));
                }
            }

            engine.NumberOfIterations = maxIter;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var statesActual2 = engine.Infer<Gaussian[]>(states);
            if (showStates)
            {
                for (int i = 0; i < length; i++)
                {
                    Console.WriteLine("state[{0}] = {1}", i, statesActual2[i]);
                }
            }
            Gaussian shiftActual2 = engine.Infer<Gaussian>(shift);
            Gaussian shift2Actual2 = engine.Infer<Gaussian>(shift2);
            Console.WriteLine("shift = {0}, shift2 = {1}", shiftActual2, shift2Actual2);
            Gaussian shiftExpected2 = new Gaussian(0.43414454744112807, 0.13105898265773996);
            //Assert.True(shiftExpected2.MaxDiff(shiftActual2) < 1e-3);
            Assert.True(MMath.AbsDiff(shiftExpected2.GetMean(), shiftActual2.GetMean(), 1e-10) < 1e-10);
            //Gaussian statesExpected9 = new Gaussian(8.936, 0.6851);
            //Assert.True(statesExpected9.MaxDiff(statesActual2[9]) < 1e-3);

            double maxError = shiftActual2.MaxDiff(shiftActual1);
            for (int i = 0; i < length; i++)
            {
                maxError = System.Math.Max(maxError, statesActual1[i].MaxDiff(statesActual2[i]));
            }
            Console.WriteLine("maxError = {0}", maxError);
            //Assert.True(maxError < 1e-4);
        }

        [Fact]
        public void ChainWithTransitionAndPrecisionParameter()
        {
            int length = 1000;

            var evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);

            Range rows = new Range(length).Named("i");
            // test does not pass with Sequential
            //rows.AddAttribute(new Sequential());
            VariableArray<double> states = Variable.Array<double>(rows).Named("states");
            VariableArray<double> observation = Variable.Array<double>(rows).Named("observations");
            Variable<double> shift = Variable.GaussianFromMeanAndVariance(0, 1).Named("shift");
            //Variable<double> shiftPoint = Variable<double>.Factor(PointEstimator.Forward<double>, shift).Named("shiftPoint");
            //Variable<double> shiftPoint = Variable<double>.Factor(PointEstimator.Forward2<double>, shift, (double)length).Named("shiftPoint");
            // when using PointEstimator.Forward with Secant, should always initialize
            shift.InitialiseTo(Gaussian.PointMass(0));
            shift.AddAttribute(new PointEstimate());
            var precisionRate = Variable.GammaFromShapeAndRate(1, 1);
            Variable<double> precision = Variable.GammaFromShapeAndRate(1, precisionRate).Named("precision");
            precision.AddAttribute(new PointEstimate());
            precision.InitialiseTo(Gamma.PointMass(1));
            //precision.ObservedValue = 1.004;

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, 1);
                    observation[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index], 1);
                }
                using (Variable.If(rowBlock.Index > 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndPrecision(states[rowBlock.Index - 1] + shift, precision);
                    observation[rowBlock.Index] = Variable.GaussianFromMeanAndPrecision(states[rowBlock.Index], 1e-8);
                }
            }
            states.AddAttribute(new DivideMessages(false));
            // Doesn't work without divide because shift_rep_F requires entire shift_rep_B array
            //shift.AddAttribute(new DivideMessages(false));
            //shift.AddAttribute(new PointEstimate());

            block.CloseBlock();

            double[] observationValues = new double[length];
            for (int i = 0; i < length; i++)
            {
                observationValues[i] = i;
            }
            observation.ObservedValue = observationValues;

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.InitialisationAffectsSchedule = true;
            //engine.Compiler.GivePriorityTo(typeof(VariablePointOp_Mean<>));
            engine.Compiler.GivePriorityTo(typeof(VariablePointOp_Rprop));
            //engine.OptimiseForVariables = new List<IVariable>() { states, shift, precision };
            engine.NumberOfIterations = 10;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var statesActual1 = engine.Infer<Gaussian[]>(states);
            bool showStates = false;
            if (showStates)
            {
                for (int i = 0; i < length; i++)
                {
                    Console.WriteLine("state[{0}] = {1}", i, statesActual1[i]);
                }
            }
            Gaussian shiftActual1 = engine.Infer<Gaussian>(shift);
            Console.WriteLine("shift = {0}", shiftActual1);

            //const double shiftMeanExpected = 0.767681001959026;
            const double shiftMeanExpected = 0.746576051928051;
            int maxIter = 100;
            if (!engine.ShowProgress)
            {
                for (int iter = 1; iter < maxIter; iter++)
                {
                    engine.NumberOfIterations = iter;
                    var shiftTemp = engine.Infer<Gaussian>(shift);
                    var precisionTemp = engine.Infer<Gamma>(precision);
                    Console.WriteLine("{0} shift={1} prec={2}", iter, shiftTemp.GetMean(), precisionTemp.GetMean());
                    //if (shiftTemp.GetMean() == shiftMeanExpected)
                    //    throw new Exception("converged at iter " + iter);
                }
            }

            engine.NumberOfIterations = maxIter;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var statesActual2 = engine.Infer<Gaussian[]>(states);
            if (showStates)
            {
                for (int i = 0; i < length; i++)
                {
                    Console.WriteLine("state[{0}] = {1} should be {2}", i, statesActual2[i], statesActual1[i]);
                }
            }
            var precisionActual = engine.Infer<Gamma>(precision);
            const double precisionExpectedMode = 0.0315504742766535;
            //const var precisionExpectedMean = 1.0042621263585;
            Console.WriteLine("precision = {0} should be {1}", precisionActual.Point, precisionExpectedMode);
            Gaussian shiftActual2 = engine.Infer<Gaussian>(shift);
            Console.WriteLine("shift = {0} should be {1}", shiftActual2.Point, shiftActual1.Point);
            Gaussian shiftExpected = new Gaussian(shiftMeanExpected, 0.2318);
            Console.WriteLine("shift = {0} should be {1}", shiftActual2.Point, shiftMeanExpected);
            //Assert.True(shiftExpected.MaxDiff(shiftActual2) < 1);
            Assert.True(MMath.AbsDiff(shiftMeanExpected, shiftActual2.GetMean(), 1e-10) < 1e-10);
            Assert.True(MMath.AbsDiff(precisionExpectedMode, precisionActual.Point, 1e-10) < 1e-10);
            //Gaussian statesExpected9 = new Gaussian(8.905, 0.6521);
            //Assert.True(statesExpected9.MaxDiff(statesActual2[9]) < 1e-3);

            // check that result is reached quickly
            double maxError = shiftActual2.MaxDiff(shiftActual1);
            for (int i = 0; i < length; i++)
            {
                maxError = System.Math.Max(maxError, statesActual1[i].MaxDiff(statesActual2[i]));
            }
            Console.WriteLine("maxError = {0}", maxError);
            //Assert.True(maxError < 3);

            // check that answer is same for observed precision
            precision.ObservedValue = precisionActual.Point;
            Gaussian shiftActual3 = engine.Infer<Gaussian>(shift);
            Console.WriteLine("shift = {0} should be {1}", shiftActual3.Point, shiftMeanExpected);
            //Assert.True(shiftExpected.MaxDiff(shiftActual3) < 1e-2);
            Assert.True(MMath.AbsDiff(shiftMeanExpected, shiftActual3.GetMean(), 1e-10) < 1e-10);

            if (false)
            {
                var shifts = EpTests.linspace(0.7, 0.8, 100);
                var logProbs = new double[shifts.Length];
                for (int i = 0; i < shifts.Length; i++)
                {
                    shift.ObservedValue = shifts[i];
                    logProbs[i] = engine.Infer<Bernoulli>(evidence).LogOdds;
                }
                //TODO: change path for cross platform using
                using (var writer = new MatlabWriter(@"..\..\..\..\Prototypes\TrueSkillSparta\TestApp\bin\Debug\logProbs.mat"))
                {
                    writer.Write("shifts", shifts);
                    writer.Write("logProbs", logProbs);
                }
            }
        }

        // A test of different schedules for a difficult time-series model.
        private void FadingChannelTest()
        {
            // longer chains make the effects worse
            // timesteps=10, priorVariance = 0.3 causes TwoPhase to crash
            int length = 10;
            double priorVariance = 0.01;
            double observationVariance = 0.001;
            double transitionVariance = priorVariance * priorVariance / (priorVariance + observationVariance);
            transitionVariance = .1;
            int numIters = 10;

            Gaussian[] statesActual = FadingChannelAutomatic(priorVariance, observationVariance, transitionVariance, length, numIters);

            Gaussian[] statesExpected = FadingChannelOptimal(priorVariance, observationVariance, transitionVariance, length, numIters);
            Console.WriteLine("====OPTIMAL====");
            double maxError = double.NegativeInfinity;
            for (int i = 0; i < length; i++)
            {
                Gaussian result = statesExpected[i];
                Console.WriteLine("state[{0}] = {1}", i, result);
                maxError = System.Math.Max(maxError, statesExpected[i].MaxDiff(statesActual[i]));
            }
            Console.WriteLine("maxError = {0}", maxError);
            Assert.True(maxError < 1e-10);

            if (false)
            {
                try
                {
                    Console.WriteLine("====UNROLLED====");
                    FadingChannelUnrolled(priorVariance, observationVariance, transitionVariance, length);
                }
                catch (ImproperMessageException)
                {
                    Console.WriteLine("Unrolled Schedule threw ImproperDistributionException");
                }

                try
                {
                    Console.WriteLine("====TWO PHASE====");
                    FadingChannelTwoPhase(priorVariance, observationVariance, transitionVariance, length);
                }
                catch (ImproperMessageException)
                {
                    Console.WriteLine("Two Phase Schedule threw ImproperDistributionException");
                }
            }
        }

        public Gaussian[] FadingChannelAutomatic(double priorVariance, double observationVariance, double transitionVariance, int length, int numIters)
        {
            // this is a toy version of the model used by:
            // "Window-Based Expectation Propagation for Adaptive Signal Detection in Flat-Fading Channels"
            // Yuan Qi and Thomas P. Minka
            // IEEE TRANSACTIONS ON WIRELESS COMMUNICATIONS, VOL. 6, NO. 1, JANUARY 2007
            Range rows = new Range(length).Named("i");
            rows.AddAttribute(new Sequential());
            VariableArray<double> states = Variable.Array<double>(rows).Named("states");
            VariableArray<double> observations = Variable.Array<double>(rows).Named("observations");
            VariableArray<bool> symbols = Variable.Array<bool>(rows).Named("symbols");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(1, priorVariance);
                    if (true)
                    {
                        symbols[rowBlock.Index] = Variable.Bernoulli(0.5);
                        using (Variable.If(symbols[rowBlock.Index]))
                        {
                            observations[rowBlock.Index].SetTo(Variable.GaussianFromMeanAndVariance(states[rowBlock.Index], observationVariance));
                        }
                        using (Variable.IfNot(symbols[rowBlock.Index]))
                        {
                            observations[rowBlock.Index].SetTo(Variable.GaussianFromMeanAndVariance(-states[rowBlock.Index], observationVariance));
                        }
                    }
                }

                using (Variable.If(rowBlock.Index > 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index - 1], transitionVariance);
                    symbols[rowBlock.Index] = Variable.Bernoulli(0.5);

                    using (Variable.If(symbols[rowBlock.Index]))
                    {
                        observations[rowBlock.Index].SetTo(Variable.GaussianFromMeanAndVariance(states[rowBlock.Index], observationVariance));
                    }
                    using (Variable.IfNot(symbols[rowBlock.Index]))
                    {
                        observations[rowBlock.Index].SetTo(Variable.GaussianFromMeanAndVariance(-states[rowBlock.Index], observationVariance));
                    }
                }
            }
            states.AddAttribute(new DivideMessages(false));
            symbols.AddAttribute(new DivideMessages(false));

            double[] observationValues = new double[length];
            for (int i = 0; i < length; i++)
            {
                observationValues[i] = 1;
            }
            observations.ObservedValue = observationValues;

            InferenceEngine engine = new InferenceEngine();
            engine.OptimiseForVariables = new List<IVariable>() { states };
            engine.NumberOfIterations = numIters;
            Gaussian[] statesActual = engine.Infer<Gaussian[]>(states);
            if (true)
            {
                Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
                int startIt = 0;
                if (length > 100)
                {
                    startIt = length - 10;
                }
                for (int i = startIt; i < length; i++)
                {
                    Console.WriteLine("state[{0}] = {1}", i, statesActual[i]);
                }
            }
            return statesActual;
        }

        // unrolled model, automatic schedule
        private static void FadingChannelUnrolled(double priorVariance, double observationVariance, double transitionVariance, int timesteps)
        {
            // this is a toy version of the model used by:
            // "Window-Based Expectation Propagation for Adaptive Signal Detection in Flat-Fading Channels"
            // Yuan Qi and Thomas P. Minka
            // IEEE TRANSACTIONS ON WIRELESS COMMUNICATIONS, VOL. 6, NO. 1, JANUARY 2007
            Variable<double>[] state = new Variable<double>[timesteps];
            Variable<double>[] observation = new Variable<double>[timesteps];
            for (int time = 0; time < timesteps; time++)
            {
                if (time == 0)
                    state[time] = Variable.GaussianFromMeanAndVariance(1, priorVariance);
                else
                    state[time] = Variable.GaussianFromMeanAndVariance(state[time - 1], transitionVariance);
                state[time].Name = "state" + time;
                observation[time] = Variable.New<double>().Named("observation" + time);
                Variable<bool> symbol = Variable.Bernoulli(0.5).Named("symbol" + time);
                using (Variable.If(symbol))
                {
                    observation[time].SetTo(Variable.GaussianFromMeanAndVariance(state[time], observationVariance));
                }
                using (Variable.IfNot(symbol))
                {
                    observation[time].SetTo(Variable.GaussianFromMeanAndVariance(-state[time], observationVariance));
                }
                observation[time].ObservedValue = 1;
            }
            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;
            for (int numIters = 1; numIters <= 10; numIters++)
            {
                engine.NumberOfIterations = numIters;
                Gaussian firstState = engine.Infer<Gaussian>(state[0]);
                Gaussian middleState = engine.Infer<Gaussian>(state[timesteps / 2]);
                Gaussian lastState = engine.Infer<Gaussian>(state[timesteps - 1]);
                Console.WriteLine("{0}: first = {1}, last = {2}", numIters, firstState, lastState);
            }
        }

        // same model but divided into a two-phase schedule
        private static void FadingChannelTwoPhase(double priorVariance, double observationVariance, double transitionVariance, int timesteps)
        {
            Variable<double>[] state = new Variable<double>[timesteps];
            Variable<double>[] stateCopy = new Variable<double>[timesteps];
            Variable<Gaussian>[] upwardMessage = new Variable<Gaussian>[timesteps];
            Variable<Gaussian>[] downwardMessage = new Variable<Gaussian>[timesteps];
            Variable<double>[] observation = new Variable<double>[timesteps];
            for (int time = 0; time < timesteps; time++)
            {
                if (time == 0)
                    state[time] = Variable.GaussianFromMeanAndVariance(1, priorVariance);
                else
                    state[time] = Variable.GaussianFromMeanAndVariance(state[time - 1], transitionVariance);
                state[time].Name = "state" + time;
                upwardMessage[time] = Variable.New<Gaussian>().Named("upward" + time);
                Variable.ConstrainEqualRandom(state[time], upwardMessage[time]);
                downwardMessage[time] = Variable.New<Gaussian>().Named("downward" + time);
                stateCopy[time] = Variable.Random<double, Gaussian>(downwardMessage[time]).Named("stateCopy" + time);
                observation[time] = Variable.New<double>().Named("observation" + time);
                Variable<bool> symbol = Variable.Bernoulli(0.5).Named("symbol" + time);
                using (Variable.If(symbol))
                {
                    observation[time].SetTo(Variable.GaussianFromMeanAndVariance(stateCopy[time], observationVariance));
                }
                using (Variable.IfNot(symbol))
                {
                    observation[time].SetTo(Variable.GaussianFromMeanAndVariance(-stateCopy[time], observationVariance));
                }
                observation[time].ObservedValue = 1;
            }
            InferenceEngine engine = new InferenceEngine();
            InferenceEngine engine2 = new InferenceEngine();
            for (int numIters = 1; numIters <= 30; numIters++)
            {
                for (int t = 0; t < timesteps; t++)
                {
                    upwardMessage[t].ObservedValue = Gaussian.Uniform();
                    downwardMessage[t].ObservedValue = Gaussian.Uniform();
                }
                for (int iter = 0; iter < numIters; iter++)
                {
                    // phase 1
                    for (int t = 0; t < timesteps; t++)
                    {
                        downwardMessage[t].ObservedValue = engine.Infer<Gaussian>(state[t]) / upwardMessage[t].ObservedValue;
                    }
                    // phase 2
                    for (int t = 0; t < timesteps; t++)
                    {
                        upwardMessage[t].ObservedValue = engine2.Infer<Gaussian>(stateCopy[t]) / downwardMessage[t].ObservedValue;
                    }
                }
                Gaussian firstState = engine.Infer<Gaussian>(state[0]);
                Gaussian middleState = engine.Infer<Gaussian>(state[timesteps / 2]);
                Gaussian lastState = engine.Infer<Gaussian>(state[timesteps - 1]);
                Console.WriteLine("{0}: first = {1}, last = {2}", numIters, firstState, lastState);
            }
        }

        // same model but using the recommended schedule from the paper.
        public static Gaussian[] FadingChannelOptimal(double priorVariance, double observationVariance, double transitionVariance, int timesteps, int maxIters)
        {
            Variable<double>[] stateCopy = new Variable<double>[timesteps];
            Variable<Gaussian>[] upwardMessage = new Variable<Gaussian>[timesteps];
            Variable<Gaussian>[] downwardMessage = new Variable<Gaussian>[timesteps];
            Variable<double>[] observation = new Variable<double>[timesteps];
            for (int time = 0; time < timesteps; time++)
            {
                upwardMessage[time] = Variable.New<Gaussian>().Named("upward" + time);
                downwardMessage[time] = Variable.New<Gaussian>().Named("downward" + time);
                stateCopy[time] = Variable.Random<double, Gaussian>(downwardMessage[time]).Named("stateCopy" + time);
                observation[time] = Variable.New<double>().Named("observation" + time);
                Variable<bool> symbol = Variable.Bernoulli(0.5).Named("symbol" + time);
                using (Variable.If(symbol))
                {
                    observation[time].SetTo(Variable.GaussianFromMeanAndVariance(stateCopy[time], observationVariance));
                }
                using (Variable.IfNot(symbol))
                {
                    observation[time].SetTo(Variable.GaussianFromMeanAndVariance(-stateCopy[time], observationVariance));
                }
                observation[time].ObservedValue = 1;
            }
            InferenceEngine engine2 = new InferenceEngine();
            Gaussian prior = new Gaussian(1, priorVariance);
            Gaussian[] stateMarginal = new Gaussian[timesteps];
            for (int numIters = maxIters; numIters <= maxIters; numIters++)
            {
                Gaussian[] forwardMessage = new Gaussian[timesteps];
                Gaussian[] backwardMessage = new Gaussian[timesteps];
                for (int t = 0; t < timesteps; t++)
                {
                    upwardMessage[t].ObservedValue = Gaussian.Uniform();
                    downwardMessage[t].ObservedValue = Gaussian.Uniform();
                    forwardMessage[t] = Gaussian.Uniform();
                    backwardMessage[t] = Gaussian.Uniform();
                }
                forwardMessage[0] = prior;
                for (int iter = 0; iter < numIters; iter++)
                {
                    // forward pass
                    for (int t = 0; t < timesteps; t++)
                    {
                        downwardMessage[t].ObservedValue = forwardMessage[t] * backwardMessage[t];
                        upwardMessage[t].ObservedValue = engine2.Infer<Gaussian>(stateCopy[t]) / downwardMessage[t].ObservedValue;
                        stateMarginal[t] = downwardMessage[t].ObservedValue * upwardMessage[t].ObservedValue;
                        if (t < timesteps - 1)
                            forwardMessage[t + 1] = GaussianOp.SampleAverageConditional(stateMarginal[t] / backwardMessage[t], 1 / transitionVariance);
                    }
                    // backward pass
                    for (int t = timesteps - 1; t >= 0; t--)
                    {
                        downwardMessage[t].ObservedValue = forwardMessage[t] * backwardMessage[t];
                        // Recomputing the upwardMessage is optional here, but helps
                        //upwardMessage[t].ObservedValue = engine2.Infer<Gaussian>(stateCopy[t]) / downwardMessage[t].ObservedValue;
                        stateMarginal[t] = downwardMessage[t].ObservedValue * upwardMessage[t].ObservedValue;
                        if (t > 0)
                            backwardMessage[t - 1] = GaussianOp.SampleAverageConditional(stateMarginal[t] / forwardMessage[t], 1 / transitionVariance);
                    }
                }
                Gaussian firstState = stateMarginal[0];
                Gaussian lastState = stateMarginal[timesteps - 1];
                Console.WriteLine("{0}: first = {1}, last = {2}", numIters, firstState, lastState);
            }
            return stateMarginal;
        }

        internal void FadingGridExperiments(int gridSize, int priorIPosition, int priorJPosition, double priorVariance)
        {
            //int gridSize = 5;
            //double priorVariance = .1;
            //int priorIPosition = 0; // gridSize - 1;
            //int priorJPosition = 0; //  gridSize - 1;

            InferenceEngine.DefaultEngine.ShowSchedule = false;
            InferenceEngine.DefaultEngine.ShowProgress = false;

            Console.WriteLine("FADING GRID EXPERIMENTS");
            Console.WriteLine("Grid size:      {0} x {1}", gridSize, gridSize);
            Console.WriteLine("Prior Variance: {0}", priorVariance);
            Console.WriteLine("Prior location: ({0}, {1})", priorIPosition, priorJPosition);
            Console.WriteLine();

            Console.WriteLine("ROLLED UP SERIAL");
            InferenceEngine.DefaultEngine.Compiler.UnrollLoops = false;
            InferenceEngine.DefaultEngine.Compiler.UseSerialSchedules = true;
            (new SerialTests()).FadingGridTestParameterized(priorVariance, .001, priorVariance, gridSize, gridSize, priorIPosition, priorJPosition);
            Console.WriteLine();

            if (gridSize >= 100)
                return;

            Console.WriteLine("ROLLED UP PARALLEL");
            InferenceEngine.DefaultEngine.Compiler.UnrollLoops = false;
            InferenceEngine.DefaultEngine.Compiler.UseSerialSchedules = false;
            try
            {
                (new SerialTests()).FadingGridTestParameterized(priorVariance, .001, priorVariance, gridSize, gridSize, priorIPosition, priorJPosition);
            }
            catch (Exception)
            {
                Console.WriteLine("FAILED");
            }
            Console.WriteLine();

            Console.WriteLine("UNROLLED (NON-WALK)");
            InferenceEngine.DefaultEngine.Compiler.UnrollLoops = true;
            try
            {
                (new SerialTests()).FadingGridTestParameterized(priorVariance, .001, priorVariance, gridSize, gridSize, priorIPosition, priorJPosition, false);
            }
            catch (Exception)
            {
                Console.WriteLine("FAILED");
            }
            Console.WriteLine();

            Console.WriteLine("UNROLLED (WALK)");
            InferenceEngine.DefaultEngine.Compiler.UnrollLoops = true;
            try
            {
                (new SerialTests()).FadingGridTestParameterized(priorVariance, .001, priorVariance, gridSize, gridSize, priorIPosition, priorJPosition);
            }
            catch (Exception)
            {
                Console.WriteLine("FAILED");
            }
            Console.WriteLine();
        }

        private void FadingGridTestParameterized(double priorVariance, double observationVariance, double transitionVariance, int lengthI, int lengthJ,
                                                 int priorIPosition = 0, int priorJPosition = 0, bool smartInit = true, bool print_results = true)
        {
            Range rows = new Range(lengthI).Named("i");
            Range cols = new Range(lengthJ).Named("j");
            rows.AddAttribute(new Sequential());
            cols.AddAttribute(new Sequential());

            VariableArray2D<double> states = Variable.Array<double>(rows, cols).Named("states");
            //VariableArray2D<double> observations = Variable.Array<double>(rows, cols).Named("observations");
            VariableArray2D<bool> symbols = Variable.Array<bool>(rows, cols).Named("symbols");
            states[rows, cols] = Variable.GaussianFromMeanAndVariance(0, 10000).ForEach(rows, cols);

            if (smartInit)
            {
                // || !InferenceEngine.DefaultEngine.Compiler.UnrollLoops) {
                for (int i = 0; i < 10; i++)
                {
                    Variable.ConstrainEqualRandom(states[rows, cols], new Gaussian());
                }
            }

            using (ForEachBlock colBlock = Variable.ForEach(cols))
            {
                using (ForEachBlock rowBlock = Variable.ForEach(rows))
                {
                    using (Variable.If(colBlock.Index == priorJPosition))
                    {
                        using (Variable.If(rowBlock.Index == priorIPosition))
                        {
                            Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index], new Gaussian(1, priorVariance));
                        }
                    }

                    symbols[rowBlock.Index, colBlock.Index] = Variable.Bernoulli(0.5);
                    using (Variable.If(symbols[rowBlock.Index, colBlock.Index]))
                    {
                        Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index], new Gaussian(-1, observationVariance));
                        //observations[rowBlock.Index, colBlock.Index].SetTo(Variable.GaussianFromMeanAndVariance(states[rowBlock.Index, colBlock.Index], observationVariance));
                    }
                    using (Variable.IfNot(symbols[rowBlock.Index, colBlock.Index]))
                    {
                        Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index], new Gaussian(1, observationVariance));
                        //observations[rowBlock.Index, colBlock.Index].SetTo(Variable.GaussianFromMeanAndVariance(-states[rowBlock.Index, colBlock.Index], observationVariance));
                    }

                    using (Variable.If(rowBlock.Index > 0))
                    {
                        Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index] - states[rowBlock.Index - 1, colBlock.Index], new Gaussian(0, transitionVariance));
                    }
                    using (Variable.If(colBlock.Index > 0))
                    {
                        Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index] - states[rowBlock.Index, colBlock.Index - 1], new Gaussian(0, transitionVariance));
                    }
                }
            }
            states.AddAttribute(new DivideMessages(false));

            /*
      double[,] observedValues = new double[lengthI, lengthJ];
      for (int i = 0; i < lengthI; i++) {
        for (int j = 0; j < lengthJ; j++) {
          observedValues[i, j] = -1;
        }
      }
      observations.ObservedValue = observedValues;
      */

            bool convergenceExperiment = false;
            bool converged = false;

            InferenceEngine engine = new InferenceEngine();
            if (convergenceExperiment)
            {
                Gaussian[,] prevResult = null;
                Gaussian[,] result;
                Stopwatch watch = new Stopwatch();
                watch.Start();
                for (int numIts = 1; numIts < 50; numIts++)
                {
                    engine.NumberOfIterations = numIts;

                    result = engine.Infer<Gaussian[,]>(states);

                    if (HasConverged(result, prevResult, 1e-4))
                    {
                        watch.Stop();
                        for (int i = 0; i < lengthI; i++)
                        {
                            for (int j = 0; j < lengthJ; j++)
                            {
                                if (lengthI < 0)
                                {
                                    bool isImproper;
                                    double mean = GetMeanSafe(result[i, j], out isImproper);
                                    if (mean < .5)
                                        throw new Exception();

                                    Console.WriteLine("state[{0}, {1}] = {2}", i, j, result[i, j]);
                                }
                            }
                        }
                        Console.WriteLine("Converged after {0} Iterations", engine.NumberOfIterations);
                        converged = true;
                        break;
                    }
                    prevResult = result;
                }
                if (!converged)
                {
                    watch.Stop();
                    Console.WriteLine("Did not converge after {0} Iterations", engine.NumberOfIterations);
                }
                Console.WriteLine("elapsed time = {0}ms", watch.ElapsedMilliseconds);
            }
            else if (print_results)
            {
                engine.NumberOfIterations = 10;

                Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
                var result = engine.Infer<Gaussian[,]>(states);

                for (int i = 0; i < lengthI; i++)
                {
                    for (int j = 0; j < lengthJ; j++)
                    {
                        Console.WriteLine("state[{0}, {1}] = {2}", i, j, result[i, j]);
                    }
                }
            }
            else
            {
                engine.NumberOfIterations = 10;
            }

            // TODO: how do we test loopy graphs?
        }

        /// <summary>
        /// Get the mean of the Gaussian, but don't crash if the variable has negative variance.  Instead, set isImproper to true.
        /// </summary>
        /// <param name="var"></param>
        /// <returns></returns>
        public double GetMeanSafe(Gaussian var, out bool isImproper)
        {
            try
            {
                double mean = var.GetMean();
                isImproper = false;
                return mean;
            }
            catch (Exception)
            {
                isImproper = true;
                return 0;
            }
        }

        /// <summary>
        /// Just check convergence in the mean for now.
        /// </summary>
        /// <param name="result"></param>
        /// <param name="prevResult"></param>
        /// <returns></returns>
        public bool HasConverged(Gaussian[,] result, Gaussian[,] prevResult, double threshold)
        {
            if (result == null || prevResult == null)
                return false;

            for (int i = 0; i < result.GetUpperBound(0) + 1; i++)
            {
                for (int j = 0; j < result.GetUpperBound(0) + 1; j++)
                {
                    bool meanIsImproper;
                    double mean = GetMeanSafe(result[i, j], out meanIsImproper);
                    bool prevMeanIsImproper;
                    double prevMean = GetMeanSafe(prevResult[i, j], out prevMeanIsImproper);
                    if (meanIsImproper || prevMeanIsImproper || MMath.AbsDiff(mean, prevMean) > threshold)
                        return false;
                }
            }
            return true;
        }


        internal void NotFadingGridTestParameterized(double priorVariance, double observationVariance, double transitionVariance, int lengthI, int lengthJ, bool print_results)
        {
            Range rows = new Range(lengthI).Named("i");
            Range cols = new Range(lengthJ).Named("j");

            int priorIPosition = lengthI;
            int priorJPosition = lengthJ;

            VariableArray2D<double> states = Variable.Array<double>(rows, cols).Named("states");
            VariableArray2D<double> observations = Variable.Array<double>(rows, cols).Named("observations");
            VariableArray2D<bool> symbols = Variable.Array<bool>(rows, cols).Named("symbols");
            states[rows, cols] = Variable.GaussianFromMeanAndVariance(0, 10000).ForEach(rows, cols);


            using (ForEachBlock colBlock = Variable.ForEach(cols))
            {
                using (ForEachBlock rowBlock = Variable.ForEach(rows))
                {
                    using (Variable.If(colBlock.Index == priorJPosition))
                    {
                        using (Variable.If(rowBlock.Index == priorIPosition))
                        {
                            Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index], new Gaussian(1, priorVariance));
                        }
                    }

                    observations[rowBlock.Index, colBlock.Index].SetTo(Variable.GaussianFromMeanAndVariance(states[rowBlock.Index, colBlock.Index], observationVariance));


                    using (Variable.If(rowBlock.Index > 0))
                    {
                        Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index] - states[rowBlock.Index - 1, colBlock.Index], new Gaussian(0, transitionVariance));
                    }
                    using (Variable.If(colBlock.Index > 0))
                    {
                        Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index] - states[rowBlock.Index, colBlock.Index - 1], new Gaussian(0, transitionVariance));
                    }
                }
            }

            double[,] observedValues = new double[lengthI, lengthJ];
            for (int i = 0; i < lengthI; i++)
            {
                for (int j = 0; j < lengthJ; j++)
                {
                    observedValues[i, j] = -1;
                }
            }
            observations.ObservedValue = observedValues;

            InferenceEngine engine = new InferenceEngine();
            engine.NumberOfIterations = 3;

            if (print_results)
            {
                Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
                var result = engine.Infer<Gaussian[,]>(states);

                for (int i = 0; i < lengthI; i++)
                {
                    for (int j = 0; j < lengthJ; j++)
                    {
                        Console.WriteLine("state[{0}, {1}] = {2}", i, j, result[i, j]);
                    }
                }
            }
            // TODO: how do we test loopy graphs?
        }


        [Fact]
        public void SkipChainTest()
        {
            int length = 10;

            Range rows = new Range(length).Named("i");
            VariableArray<double> states = Variable.Array<double>(rows).Named("states");
            VariableArray<double> observation = Variable.Array<double>(rows).Named("observation");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(0, 1000);
                    observation[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index], 1);
                }

                using (Variable.If(rowBlock.Index == 1))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(0, 1000);
                    observation[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index], 1);
                }

                using (Variable.If(rowBlock.Index > 1))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index - 1], 1);
                    observation[rowBlock.Index] = states[rowBlock.Index] - states[rowBlock.Index - 2];
                }
            }
            states.AddAttribute(new DivideMessages(false));

            double[] observedValues = new double[length];
            for (int i = 0; i < length; i++)
            {
                observedValues[i] = 1;
            }
            observation.ObservedValue = observedValues;

            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;
            IList<Gaussian> statesActual = null;
            if (false)
            {
                double[] history = new double[99];
                for (int iter = 1; iter <= history.Length; iter++)
                {
                    engine.NumberOfIterations = iter;
                    statesActual = engine.Infer<IList<Gaussian>>(states);
                    Console.WriteLine("{0}: {1}", iter, statesActual[0]);
                    history[iter - 1] = statesActual[0].GetMean();
                }
                if (false)
                {
                    using (MatlabWriter writer = new MatlabWriter("SkipChain_Unrolled.mat"))
                    {
                        writer.Write("mean0", history);
                    }
                }
            }
            else
            {
                engine.NumberOfIterations = 100;
                statesActual = engine.Infer<IList<Gaussian>>(states);
                Console.WriteLine(statesActual);
            }
            Gaussian state0Expected = new Gaussian(0.7637, 0.1247);
            Assert.True(state0Expected.MaxDiff(statesActual[0]) < 1e-3);

            engine.Compiler.UnrollLoops = true;
            engine.NumberOfIterations = 1000;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var statesExpected = engine.Infer<IList<Gaussian>>(states);
            Console.WriteLine(statesExpected);
            double maxError = double.NegativeInfinity;
            for (int i = 0; i < length; i++)
            {
                double error = statesExpected[i].MaxDiff(statesActual[i]);
                maxError = System.Math.Max(maxError, error);
            }
            Console.WriteLine("maxError = {0}", maxError);
            Assert.True(maxError < 1e-4);
        }

        [Fact]
        public void PositiveOffsetSkipChain()
        {
            int length = 10;

            Range rows = new Range(length).Named("i");
            VariableArray<double> states = Variable.Array<double>(rows).Named("states");
            VariableArray<double> observation = Variable.Array<double>(rows).Named("observation");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(0, 1000);
                    observation[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index], 1);
                }

                using (Variable.If(rowBlock.Index == 1))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(0, 1000);
                    observation[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index], 1);
                }

                using (Variable.If(rowBlock.Index > 1))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index - 1], 1);
                }

                using (Variable.If(rowBlock.Index < length - 2))
                {
                    observation[rowBlock.Index] = states[rowBlock.Index + 2] - states[rowBlock.Index];
                }
            }
            states.AddAttribute(new DivideMessages(false));

            double[] observedValues = new double[length];
            for (int i = 0; i < length; i++)
            {
                observedValues[i] = 1;
            }
            observation.ObservedValue = observedValues;

            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;
            engine.OptimiseForVariables = new List<IVariable>() { states };
            IList<Gaussian> statesActual = null;
            if (false)
            {
                for (int its = 1; its < 10; its++)
                {
                    engine.NumberOfIterations = its;
                    Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
                    statesActual = engine.Infer<IList<Gaussian>>(states);
                    for (int i = 0; i < length; i++)
                    {
                        Console.WriteLine("state[{0}] = {1}", i, statesActual[i]);
                    }
                }
            }
            else
            {
                engine.NumberOfIterations = 100;
                statesActual = engine.Infer<IList<Gaussian>>(states);
                Console.WriteLine(statesActual);
            }

            engine.Compiler.UnrollLoops = true;
            engine.NumberOfIterations = 1000;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var statesExpected = engine.Infer<IList<Gaussian>>(states);
            Console.WriteLine(statesExpected);
            double maxError = double.NegativeInfinity;
            for (int i = 0; i < length; i++)
            {
                double error = statesExpected[i].MaxDiff(statesActual[i]);
                maxError = System.Math.Max(maxError, error);
            }
            Console.WriteLine("maxError = {0}", maxError);
            Assert.True(maxError < 1e-4);
        }


        [Fact]
        public void TwoThreeSkipChain()
        {
            int length = 10;

            Range rows = new Range(length).Named("i");
            VariableArray<double> states = Variable.Array<double>(rows).Named("states");
            states.AddAttribute(new DivideMessages(false));
            VariableArray<double> observation = Variable.Array<double>(rows).Named("observation");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(0, 1000);
                    observation[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index], 1);
                }
                using (Variable.If(rowBlock.Index == 1))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(0, 1000);
                    observation[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index], 1);
                }

                using (Variable.If(rowBlock.Index > 1))
                {
                    states[rowBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index - 2], 1);
                }

                using (Variable.If(rowBlock.Index > 2))
                {
                    observation[rowBlock.Index] = states[rowBlock.Index] - states[rowBlock.Index - 3];
                }
            }

            double[] observedValues = new double[length];
            for (int i = 0; i < length; i++)
            {
                observedValues[i] = 1;
            }
            observation.ObservedValue = observedValues;

            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;
            engine.OptimiseForVariables = new List<IVariable>() { states };
            IList<Gaussian> statesActual = null;
            if (false)
            {
                for (int its = 1; its <= 5; its++)
                {
                    engine.NumberOfIterations = its;
                    Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
                    statesActual = engine.Infer<Gaussian[]>(states);
                    for (int i = 0; i < length; i++)
                    {
                        Console.WriteLine("state[{0}] = {1}", i, statesActual[i]);
                    }
                }
            }
            else
            {
                engine.NumberOfIterations = 100;
                statesActual = engine.Infer<IList<Gaussian>>(states);
                Console.WriteLine(statesActual);
            }

            engine.Compiler.UnrollLoops = true;
            engine.NumberOfIterations = 1000;
            Console.WriteLine("Unrolled: {0}", engine.NumberOfIterations);
            var statesExpected = engine.Infer<IList<Gaussian>>(states);
            Console.WriteLine(statesExpected);
            double maxError = double.NegativeInfinity;
            for (int i = 0; i < length; i++)
            {
                double error = statesExpected[i].MaxDiff(statesActual[i]);
                maxError = System.Math.Max(maxError, error);
            }
            Console.WriteLine("maxError = {0}", maxError);
            Assert.True(maxError < 1e-4);
        }

        [Fact]
        public void HorizontallyConnectedGridTest()
        {
            int length = 10;
            Range rows = new Range(length).Named("i");
            Range cols = new Range(length).Named("j");

            VariableArray2D<double> states = Variable.Array<double>(rows, cols).Named("states");
            VariableArray2D<double> observations = Variable.Array<double>(rows, cols).Named("observations");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (ForEachBlock colBlock = Variable.ForEach(cols))
                {
                    using (Variable.If(colBlock.Index == 0))
                    {
                        states[rowBlock.Index, colBlock.Index] = Variable.GaussianFromMeanAndVariance(0, 1);
                        observations[rowBlock.Index, colBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index, colBlock.Index], 1);
                    }
                    using (Variable.If(colBlock.Index > 0))
                    {
                        states[rowBlock.Index, colBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index, colBlock.Index - 1], 1);
                        observations[rowBlock.Index, colBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index, colBlock.Index], 1);
                    }
                }
            }

            double[,] observedValues = new double[length, length];
            for (int i = 0; i < length; i++)
            {
                for (int j = 0; j < length; j++)
                {
                    observedValues[i, j] = 1;
                }
            }
            observations.ObservedValue = observedValues;

            InferenceEngine engine = new InferenceEngine();
            engine.OptimiseForVariables = new List<IVariable>() { states };
            engine.NumberOfIterations = 1;
            var statesActual = engine.Infer<Gaussian[,]>(states);
            for (int i = 0; i < length; i++)
            {
                for (int j = 0; j < length; j++)
                {
                    Console.WriteLine("state[{0}, {1}] = {2}", i, j, statesActual[i, j]);
                }
            }
            Gaussian state00Converged = new Gaussian(0.618, 0.382);
            Gaussian state10Converged = new Gaussian(0.618, 0.382);
            Gaussian state99Converged = new Gaussian(0.9999, 0.618);
            Console.WriteLine("state[0,0] = {0} converged = {1}", statesActual[0, 0], state00Converged);
            Console.WriteLine("state[1,0] = {0} converged = {1}", statesActual[1, 0], state10Converged);
            Console.WriteLine("state[9,9] = {0} converged = {1}", statesActual[9, 9], state99Converged);
            Assert.True(state00Converged.MaxDiff(statesActual[0, 0]) < 0.01);
            Assert.True(state10Converged.MaxDiff(statesActual[1, 0]) < 0.01);
            Assert.True(state99Converged.MaxDiff(statesActual[9, 9]) < 0.01);
        }

        [Fact]
        public void VerticallyConnectedGridTest()
        {
            int length = 10;
            Range rows = new Range(length).Named("i");
            Range cols = new Range(length).Named("j");

            VariableArray2D<double> states = Variable.Array<double>(rows, cols).Named("states");
            VariableArray2D<double> observations = Variable.Array<double>(rows, cols).Named("observations");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (ForEachBlock colBlock = Variable.ForEach(cols))
                {
                    using (Variable.If(rowBlock.Index == 0))
                    {
                        states[rowBlock.Index, colBlock.Index] = Variable.GaussianFromMeanAndVariance(0, 1);
                        observations[rowBlock.Index, colBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index, colBlock.Index], 1);
                    }
                    using (Variable.If(rowBlock.Index > 0))
                    {
                        states[rowBlock.Index, colBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index - 1, colBlock.Index], 1);
                        observations[rowBlock.Index, colBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index, colBlock.Index], 1);
                    }
                }
            }

            double[,] observedValues = new double[length, length];
            for (int i = 0; i < length; i++)
            {
                for (int j = 0; j < length; j++)
                {
                    observedValues[i, j] = 1;
                }
            }
            observations.ObservedValue = observedValues;

            InferenceEngine engine = new InferenceEngine();
            engine.OptimiseForVariables = new List<IVariable>() { states };
            engine.NumberOfIterations = 1;
            var statesActual = engine.Infer<Gaussian[,]>(states);
            for (int i = 0; i < length; i++)
            {
                for (int j = 0; j < length; j++)
                {
                    Console.WriteLine("state[{0}, {1}] = {2}", i, j, statesActual[i, j]);
                }
            }
            Gaussian state00Converged = new Gaussian(0.618, 0.382);
            Gaussian state10Converged = new Gaussian(0.8541, 0.4377);
            Gaussian state99Converged = new Gaussian(0.9999, 0.618);
            Console.WriteLine("state[0,0] = {0} converged = {1}", statesActual[0, 0], state00Converged);
            Console.WriteLine("state[1,0] = {0} converged = {1}", statesActual[1, 0], state10Converged);
            Console.WriteLine("state[9,9] = {0} converged = {1}", statesActual[9, 9], state99Converged);
            Assert.True(state00Converged.MaxDiff(statesActual[0, 0]) < 0.01);
            Assert.True(state10Converged.MaxDiff(statesActual[1, 0]) < 0.01);
            Assert.True(state99Converged.MaxDiff(statesActual[9, 9]) < 0.01);
        }

        [Fact]
        public void GridNoInitBlockTest()
        {
            GridNoInitBlock(false);
            GridNoInitBlock(true);
        }
        private void GridNoInitBlock(bool splitObservations)
        {
            int length = 10;
            Range rows = new Range(length).Named("i");
            Range cols = new Range(length).Named("j");

            VariableArray2D<double> states = Variable.Array<double>(rows, cols).Named("states");
            VariableArray2D<double> observations = Variable.Array<double>(rows, cols).Named("observations");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                var row = rowBlock.Index;
                using (ForEachBlock colBlock = Variable.ForEach(cols))
                {
                    var col = colBlock.Index;
                    using (Variable.If(row == 0))
                    {
                        states[row, col] = Variable.GaussianFromMeanAndVariance(0, 1);
                        if (splitObservations)
                            observations[row, col] = Variable.GaussianFromMeanAndVariance(states[row, col], 1);
                    }
                    using (Variable.If(row > 0))
                    {
                        states[row, col] = Variable.GaussianFromMeanAndVariance(states[row - 1, col], 1);
                        if (splitObservations)
                            observations[row, col] = Variable.GaussianFromMeanAndVariance(states[row, col], 1);
                    }
                    if (!splitObservations)
                        observations[row, col] = Variable.GaussianFromMeanAndVariance(states[row, col], 1);
                    using (Variable.If(col > 0))
                    {
                        Variable.ConstrainEqualRandom(states[row, col] - states[row, col - 1], new Gaussian(0, 1));
                    }
                }
            }
            states.AddAttribute(new DivideMessages(false));

            double[,] observedValues = new double[length, length];
            for (int i = 0; i < length; i++)
            {
                for (int j = 0; j < length; j++)
                {
                    observedValues[i, j] = 1;
                }
            }
            observations.ObservedValue = observedValues;

            InferenceEngine engine = new InferenceEngine();
            engine.OptimiseForVariables = new List<IVariable>() { states };
            engine.NumberOfIterations = 1;
            var statesActual = engine.Infer<Gaussian[,]>(states);
            for (int i = 0; i < length; i++)
            {
                for (int j = 0; j < length; j++)
                {
                    Console.WriteLine("state[{0}, {1}] = {2}", i, j, statesActual[i, j]);
                }
            }
            Gaussian state00Converged = new Gaussian(0.618, 0.2863);
            Gaussian state10Converged = new Gaussian(0.8541, 0.3109);
            Gaussian state99Converged = new Gaussian(0.9999, 0.4124);
            Console.WriteLine("state[0,0] = {0} converged = {1}", statesActual[0, 0], state00Converged);
            Console.WriteLine("state[1,0] = {0} converged = {1}", statesActual[1, 0], state10Converged);
            Console.WriteLine("state[9,9] = {0} converged = {1}", statesActual[9, 9], state99Converged);
            if (true)
            {
                // results with (backward i)(forward i)
                Assert.True(state00Converged.MaxDiff(statesActual[0, 0]) < 0.05);
                Assert.True(state10Converged.MaxDiff(statesActual[1, 0]) < 0.3);
                Assert.True(state99Converged.MaxDiff(statesActual[9, 9]) < 0.8);
            }
            else if (true)
            {
                // results with (backward j, backward i)(backward j, forward i)(forward j, backward i)(forward j, forward i)
                Assert.True(state00Converged.MaxDiff(statesActual[0, 0]) < 0.004);
                Assert.True(state10Converged.MaxDiff(statesActual[1, 0]) < 0.007);
                Assert.True(state99Converged.MaxDiff(statesActual[9, 9]) < 0.0004);
            }
            else if (true)
            {
                // results with (forward i, forward j)(forward i, backward j)(backward i, backward j)
                Assert.True(state00Converged.MaxDiff(statesActual[0, 0]) < 0.14);
                Assert.True(state10Converged.MaxDiff(statesActual[1, 0]) < 0.17);
                Assert.True(state99Converged.MaxDiff(statesActual[9, 9]) < 0.015);
            }
            else
            {
                // results with (forward i, forward j)(backward i, backward j)
                Assert.True(state00Converged.MaxDiff(statesActual[0, 0]) < 0.01);
                Assert.True(state10Converged.MaxDiff(statesActual[1, 0]) < 0.08);
                Assert.True(state99Converged.MaxDiff(statesActual[9, 9]) < 0.015);
            }
        }


        [Fact]
        public void GridTest3()
        {
            int length = 6;
            Range rows = new Range(length).Named("i");
            Range cols = new Range(length).Named("j");

            VariableArray2D<double> states = Variable.Array<double>(rows, cols).Named("states");
            //VariableArray2D<double> observations = Variable.Array<double>(rows, cols).Named("observations");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (ForEachBlock colBlock = Variable.ForEach(cols))
                {
                    states[rowBlock.Index, colBlock.Index] = Variable.GaussianFromMeanAndVariance(0, 1000);
                    //observations[rowBlock.Index, colBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index, colBlock.Index], 1);

                    using (Variable.If(rowBlock.Index == 0))
                    {
                        using (Variable.If(colBlock.Index == 0))
                        {
                            Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index], new Gaussian(1, .1));
                        }
                    }

                    using (Variable.If(rowBlock.Index > 0))
                    {
                        Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index] - states[rowBlock.Index - 1, colBlock.Index], new Gaussian(0, 1));
                    }
                    using (Variable.If(colBlock.Index > 0))
                    {
                        Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index] - states[rowBlock.Index, colBlock.Index - 1], new Gaussian(0, 1));
                    }
                }
            }
            states.AddAttribute(new DivideMessages(false));

            /*
      double[,] observedValues = new double[length, length];
      for (int i = 0; i < length; i++) {
        for (int j = 0; j < length; j++) {
          observedValues[i, j] = 1;
        }
      }
      observations.ObservedValue = observedValues;
      */

            InferenceEngine engine = new InferenceEngine();
            engine.OptimiseForVariables = new List<IVariable>() { states };
            //engine.Compiler.UnrollLoops = true;
            // if forward loop is first, only need 2 iterations
            // if backward loop is first, need 20 iterations
            engine.NumberOfIterations = 20;
            var statesActual = engine.Infer<Gaussian[,]>(states);
            if (false)
            {
                for (int iter = 1; iter < 15; iter++)
                {
                    engine.NumberOfIterations = iter;
                    statesActual = engine.Infer<Gaussian[,]>(states);
                    Console.WriteLine("{0} {1}", iter, statesActual[5, 5]);
                }
            }
            if (false)
            {
                for (int i = 0; i < length; i++)
                {
                    for (int j = 0; j < length; j++)
                    {
                        Console.WriteLine("state[{0}, {1}] = {2}", i, j, statesActual[i, j]);
                    }
                }
            }
            Gaussian state00Converged = new Gaussian(0.9965, 0.09016);
            Gaussian state55Converged = new Gaussian(0.9556, 0.9217);
            // typical result:
            // state[0,0] = Gaussian(0.9989, 0.09544) converged = Gaussian(0.9965, 0.09016)
            // state[5,5] = Gaussian(0.9464, 9.721) converged = Gaussian(0.9556, 0.9217)
            Console.WriteLine("state[0,0] = {0} converged = {1}", statesActual[0, 0], state00Converged);
            Console.WriteLine("state[5,5] = {0} converged = {1}", statesActual[5, 5], state55Converged);
            Assert.True(state00Converged.MaxDiff(statesActual[0, 0]) < 0.05);
            Assert.True(state55Converged.MaxDiff(statesActual[5, 5]) < 0.15);
        }

        [Fact]
        public void SkipGridTest()
        {
            int length = 10;
            Range rows = new Range(length).Named("i");
            Range cols = new Range(length).Named("j");

            VariableArray2D<double> states = Variable.Array<double>(rows, cols).Named("states");
            VariableArray2D<double> observations = Variable.Array<double>(rows, cols).Named("observations");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (ForEachBlock colBlock = Variable.ForEach(cols))
                {
                    states[rowBlock.Index, colBlock.Index] = Variable.GaussianFromMeanAndVariance(0, 100000);
                    //observations[rowBlock.Index, colBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index, colBlock.Index], 1);

                    // Prior
                    using (Variable.If(rowBlock.Index == 0))
                    {
                        using (Variable.If(colBlock.Index == 0))
                        {
                            Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index], new Gaussian(1, .1));
                        }
                    }

                    // Vertical 1-offset connections
                    using (Variable.If(rowBlock.Index > 0))
                    {
                        Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index] - states[rowBlock.Index - 1, colBlock.Index], new Gaussian(0, 1));
                    }
                    // Vertical 2-offset connections
                    using (Variable.If(rowBlock.Index > 1))
                    {
                        Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index] - states[rowBlock.Index - 2, colBlock.Index], new Gaussian(0, 1));
                    }

                    // Horizontal 1-offset connection
                    using (Variable.If(colBlock.Index > 0))
                    {
                        Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index] - states[rowBlock.Index, colBlock.Index - 1], new Gaussian(0, 1));
                    }
                    // Horizontal 2-offset connection
                    using (Variable.If(colBlock.Index > 1))
                    {
                        Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index] - states[rowBlock.Index, colBlock.Index - 2], new Gaussian(0, 1));
                    }
                }
            }
            states.AddAttribute(new DivideMessages(false));

            /*
      double[,] observedValues = new double[length, length];
      for (int i = 0; i < length; i++) {
        for (int j = 0; j < length; j++) {
          observedValues[i, j] = 1;
        }
      }
      observations.ObservedValue = observedValues;
      */

            InferenceEngine engine = new InferenceEngine();
            engine.NumberOfIterations = 1;
            engine.OptimiseForVariables = new List<IVariable>() { states };
            var statesActual = engine.Infer<Gaussian[,]>(states);
            for (int i = 0; i < length; i++)
            {
                for (int j = 0; j < length; j++)
                {
                    Console.WriteLine("state[{0}, {1}] = {2}", i, j, statesActual[i, j]);
                }
            }
            Gaussian state00Converged = new Gaussian(0.9998, 0.07614);
            Gaussian state10Converged = new Gaussian(0.9993, 0.2395);
            Gaussian state99Converged = new Gaussian(0.999, 0.3193);
            Console.WriteLine("state[0,0] = {0} converged = {1}", statesActual[0, 0], state00Converged);
            Console.WriteLine("state[1,0] = {0} converged = {1}", statesActual[1, 0], state10Converged);
            Console.WriteLine("state[9,9] = {0} converged = {1}", statesActual[9, 9], state99Converged);
            Assert.True(state00Converged.MaxDiff(statesActual[0, 0]) < 0.04);
            Assert.True(state10Converged.MaxDiff(statesActual[1, 0]) < 0.07);
            Assert.True(state99Converged.MaxDiff(statesActual[9, 9]) < 0.15);
        }

        // Fails due to excessive initialization schedule.
        // This test has offsets with mixed signs in the same expression.
        // This leads to cases where the generated code has [i-1,j+1] in a forward loop over i and j.
        // This is correct since the whole j array was computed in the previous iteration of i.
        [Fact]
        [Trait("Category", "OpenBug")]
        public void EightGridTest()
        {
            int lengthI = 10;
            int lengthJ = 10;
            Range rows = new Range(lengthI).Named("i");
            Range cols = new Range(lengthJ).Named("j");

            VariableArray2D<double> states = Variable.Array<double>(rows, cols).Named("states");
            VariableArray2D<double> observations = Variable.Array<double>(rows, cols).Named("observations");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (ForEachBlock colBlock = Variable.ForEach(cols))
                {
                    states[rowBlock.Index, colBlock.Index] = Variable.GaussianFromMeanAndVariance(0, 100000);
                    //observations[rowBlock.Index, colBlock.Index] = Variable.GaussianFromMeanAndVariance(states[rowBlock.Index, colBlock.Index], 1);

                    // Prior
                    using (Variable.If(rowBlock.Index == 0))
                    {
                        using (Variable.If(colBlock.Index == 0))
                        {
                            Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index], new Gaussian(1, .1));
                        }
                    }

                    // Vertical connections
                    using (Variable.If(rowBlock.Index > 0))
                    {
                        Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index] - states[rowBlock.Index - 1, colBlock.Index], new Gaussian(0, 1));
                    }

                    // Up-left, down-right connection
                    using (Variable.If(colBlock.Index > 0))
                    {
                        using (Variable.If(rowBlock.Index > 0))
                        {
                            Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index] - states[rowBlock.Index - 1, colBlock.Index - 1], new Gaussian(0, 1));
                        }
                    }

                    // Horizontal connection
                    using (Variable.If(colBlock.Index > 0))
                    {
                        Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index] - states[rowBlock.Index, colBlock.Index - 1], new Gaussian(0, 1));
                    }

                    // Up-right, down-left connections
                    using (Variable.If(rowBlock.Index > 0))
                    {
                        using (Variable.If(colBlock.Index < lengthJ - 1))
                        {
                            Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index] - states[rowBlock.Index - 1, colBlock.Index + 1], new Gaussian(0, 1));
                        }
                    }
                }
            }
            states.AddAttribute(new DivideMessages(false));

            /*
      double[,] observedValues = new double[length, length];
      for (int i = 0; i < length; i++) {
        for (int j = 0; j < length; j++) {
          observedValues[i, j] = 1;
        }
      }
      observations.ObservedValue = observedValues;
      */

            InferenceEngine engine = new InferenceEngine();
            engine.NumberOfIterations = 1;
            engine.OptimiseForVariables = new List<IVariable>() { states };
            var statesActual = engine.Infer<Gaussian[,]>(states);
            for (int i = 0; i < lengthI; i++)
            {
                for (int j = 0; j < lengthJ; j++)
                {
                    Console.WriteLine("state[{0}, {1}] = {2}", i, j, statesActual[i, j]);
                }
            }
            Gaussian state00Converged = new Gaussian(0.9993, 0.08079);
            Gaussian state10Converged = new Gaussian(0.9972, 0.24);
            Gaussian state99Converged = new Gaussian(0.9946, 0.4212);
            Console.WriteLine("state[0,0] = {0} converged = {1}", statesActual[0, 0], state00Converged);
            Console.WriteLine("state[1,0] = {0} converged = {1}", statesActual[1, 0], state10Converged);
            Console.WriteLine("state[9,9] = {0} converged = {1}", statesActual[9, 9], state99Converged);
            Assert.True(state00Converged.MaxDiff(statesActual[0, 0]) < 1.6);
            Assert.True(state10Converged.MaxDiff(statesActual[1, 0]) < 1.8);
            Assert.True(state99Converged.MaxDiff(statesActual[9, 9]) < 0.4);
        }

        /// <summary>
        /// Tests that the scheduler correctly detects cycles in a 2D array with offset indexing
        /// </summary>
        [Fact]
        public void TrickyCycleGridTest()
        {
            int lengthI = 3;
            int lengthJ = 3;
            Range rows = new Range(lengthI).Named("i");
            Range cols = new Range(lengthJ).Named("j");

            VariableArray2D<double> states = Variable.Array<double>(rows, cols).Named("states");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (ForEachBlock colBlock = Variable.ForEach(cols))
                {
                    states[rowBlock.Index, colBlock.Index] = Variable.GaussianFromMeanAndVariance(0, 100000);

                    // Prior
                    using (Variable.If(rowBlock.Index == 0))
                    {
                        using (Variable.If(colBlock.Index == 0))
                        {
                            Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index], new Gaussian(1, .1));
                        }
                    }

                    // Up-left, down-right connection
                    using (Variable.If(colBlock.Index > 0))
                    {
                        using (Variable.If(rowBlock.Index > 0))
                        {
                            Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index] - states[rowBlock.Index - 1, colBlock.Index - 1], new Gaussian(0, 1));
                        }
                    }

                    // there should be no cycle unless we add these connections
                    if (true)
                    {
                        // Up-right, down-left connections
                        using (Variable.If(rowBlock.Index > 0))
                        {
                            using (Variable.If(colBlock.Index < lengthJ - 1))
                            {
                                Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index] - states[rowBlock.Index - 1, colBlock.Index + 1], new Gaussian(0, 1));
                            }
                        }
                    }
                }
            }
            states.AddAttribute(new DivideMessages(false));

            InferenceEngine engine = new InferenceEngine();
            engine.OptimiseForVariables = new List<IVariable>() { states };
            var statesActual = engine.Infer<Gaussian[,]>(states);
            for (int i = 0; i < lengthI; i++)
            {
                for (int j = 0; j < lengthJ; j++)
                {
                    Console.WriteLine("state[{0}, {1}] = {2}", i, j, statesActual[i, j]);
                }
            }
            Gaussian state00Converged = new Gaussian(1, 0.1);
            Gaussian state10Converged = new Gaussian(0, 284.1);
            Gaussian state22Converged = new Gaussian(0.9999, 2.1);
            Console.WriteLine("state[0,0] = {0} converged = {1}", statesActual[0, 0], state00Converged);
            Console.WriteLine("state[1,0] = {0} converged = {1}", statesActual[1, 0], state10Converged);
            Console.WriteLine("state[2,2] = {0} converged = {1}", statesActual[2, 2], state22Converged);
            Assert.True(state00Converged.MaxDiff(statesActual[0, 0]) < 1e-3);
            Assert.True(state10Converged.MaxDiff(statesActual[1, 0]) < 1e-3);
            Assert.True(state22Converged.MaxDiff(statesActual[2, 2]) < 1e-3);
        }

        [Fact]
        public void GridTest()
        {
            int length = 10;

            Range rows = new Range(length).Named("i");
            Range cols = new Range(length).Named("j");

            VariableArray2D<double> states = Variable.Array<double>(rows, cols).Named("states");
            VariableArray2D<double> observation = Variable.Array<double>(rows, cols).Named("observation");
            //observation[rows, cols] = Variable.GaussianFromMeanAndVariance(states[rows, cols], 1).ForEach(rows, cols);
            states[rows, cols] = Variable.GaussianFromMeanAndVariance(1, 1).ForEach(rows, cols);

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index == 0))
                {
                    using (ForEachBlock colBlock = Variable.ForEach(cols))
                    {
                        using (Variable.If(colBlock.Index == 0))
                        {
                        }
                        using (Variable.If(colBlock.Index > 0))
                        {
                            // Horizontal connection: x[row, col] to x[row, col - 1]                            
                            Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index] - states[rowBlock.Index, colBlock.Index - 1], new Gaussian(0, 1));
                        }
                    }
                }
                using (Variable.If(rowBlock.Index > 0))
                {
                    using (ForEachBlock colBlock = Variable.ForEach(cols))
                    {
                        using (Variable.If(colBlock.Index == 0))
                        {
                            // no vertical connections in first column
                        }
                        using (Variable.If(colBlock.Index > 0))
                        {
                            // Vertical connection: x[row, col] to x[row - 1, col]
                            Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index] - states[rowBlock.Index - 1, colBlock.Index], new Gaussian(0, 1));
                            // Horizontal connection: x[row, col] to x[row, col - 1]                            
                            Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index] - states[rowBlock.Index, colBlock.Index - 1], new Gaussian(0, 1));
                        }
                    }
                }
            }
            states.AddAttribute(new DivideMessages(false));

            /*
      double[,] observedValues = new double[length, length];
      for (int i = 0; i < length; i++)
      {
          for (int j = 0; j < length; j++)
          {
              observedValues[i, j] = 1;
          }
      }
      observation.ObservedValue = observedValues;
      */

            InferenceEngine engine = new InferenceEngine();
            //engine.Compiler.UnrollLoops = true;
            engine.OptimiseForVariables = new List<IVariable>() { states };
            engine.NumberOfIterations = 1;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var statesActual = engine.Infer<Gaussian[,]>(states);
            for (int i = 0; i < length; i++)
            {
                for (int j = 0; j < length; j++)
                {
                    Console.WriteLine("state[{0}, {1}] = {2}", i, j, statesActual[i, j]);
                }
            }
            Gaussian state00Converged = new Gaussian(1, 0.5844);
            Gaussian state10Converged = new Gaussian(1, 0.5675);
            Gaussian state99Converged = new Gaussian(1, 0.4124);
            Console.WriteLine("state[0,0] = {0} converged = {1}", statesActual[0, 0], state00Converged);
            Console.WriteLine("state[1,0] = {0} converged = {1}", statesActual[1, 0], state10Converged);
            Console.WriteLine("state[9,9] = {0} converged = {1}", statesActual[9, 9], state99Converged);
            Assert.True(state00Converged.MaxDiff(statesActual[0, 0]) < 0.01);
            Assert.True(state10Converged.MaxDiff(statesActual[1, 0]) < 0.01);
            Assert.True(state99Converged.MaxDiff(statesActual[9, 9]) < 0.01);
        }

        [Fact]
        public void GridTest2()
        {
            int length = 10;

            Range rows = new Range(length).Named("i");
            Range cols = new Range(length).Named("j");

            VariableArray2D<double> states = Variable.Array<double>(rows, cols).Named("states");
            VariableArray2D<double> observation = Variable.Array<double>(rows, cols).Named("observation");
            //observation[rows, cols] = Variable.GaussianFromMeanAndVariance(states[rows, cols], 1).ForEach(rows, cols);
            states[rows, cols] = Variable.GaussianFromMeanAndVariance(1, 1).ForEach(rows, cols);

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (ForEachBlock colBlock = Variable.ForEach(cols))
                {
                    using (Variable.If(colBlock.Index > 0))
                    {
                        // Horizontal connection: x[row, col] to x[row, col - 1]                            
                        Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index] - states[rowBlock.Index, colBlock.Index - 1], new Gaussian(0, 1));
                    }
                }
                using (Variable.If(rowBlock.Index > 0))
                {
                    using (ForEachBlock colBlock = Variable.ForEach(cols))
                    {
                        // Vertical connection: x[row, col] to x[row - 1, col]
                        Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index] - states[rowBlock.Index - 1, colBlock.Index], new Gaussian(0, 1));
                    }
                }
            }
            states.AddAttribute(new DivideMessages(false));

            /*
      double[,] observedValues = new double[length, length];
      for (int i = 0; i < length; i++)
      {
              for (int j = 0; j < length; j++)
              {
                      observedValues[i, j] = 1;
              }
      }
      observation.ObservedValue = observedValues;
      */

            InferenceEngine engine = new InferenceEngine();
            //engine.Compiler.UnrollLoops = true;
            engine.OptimiseForVariables = new List<IVariable>() { states };
            engine.NumberOfIterations = 1;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var statesActual = engine.Infer<Gaussian[,]>(states);
            for (int i = 0; i < length; i++)
            {
                for (int j = 0; j < length; j++)
                {
                    Console.WriteLine("state[{0}, {1}] = {2}", i, j, statesActual[i, j]);
                }
            }
            Gaussian state00Converged = new Gaussian(1, 0.4124);
            Gaussian state10Converged = new Gaussian(1, 0.3218);
            Gaussian state99Converged = new Gaussian(1, 0.4124);
            Console.WriteLine("state[0,0] = {0} converged = {1}", statesActual[0, 0], state00Converged);
            Console.WriteLine("state[1,0] = {0} converged = {1}", statesActual[1, 0], state10Converged);
            Console.WriteLine("state[9,9] = {0} converged = {1}", statesActual[9, 9], state99Converged);
            Assert.True(state00Converged.MaxDiff(statesActual[0, 0]) < 0.01);
            Assert.True(state10Converged.MaxDiff(statesActual[1, 0]) < 0.015);
            Assert.True(state99Converged.MaxDiff(statesActual[9, 9]) < 0.01);
        }

        // Fails with:
        // schedule splits group 68 at node 15 [68 66] shift_0_2_rep_F[i][j]
        [Trait("Category", "OpenBug")]
        [Fact]
        public void GridWithTransitionParameterTest()
        {
            int length = 10;

            Range rows = new Range(length).Named("i");
            Range cols = new Range(length).Named("j");
            rows.AddAttribute(new Sequential());
            cols.AddAttribute(new Sequential());

            VariableArray2D<double> states = Variable.Array<double>(rows, cols).Named("states");
            states[rows, cols] = Variable.GaussianFromMeanAndVariance(1, 1000).ForEach(rows, cols);
            VariableArray2D<double> observation = Variable.Array<double>(rows, cols).Named("observation");
            observation[rows, cols] = Variable.GaussianFromMeanAndVariance(states[rows, cols], 1);
            Variable<double> shift = Variable.GaussianFromMeanAndVariance(0, 1).Named("shift");

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (ForEachBlock colBlock = Variable.ForEach(cols))
                {
                    using (Variable.If(colBlock.Index > 0))
                    {
                        // Horizontal connection: x[row, col] to x[row, col - 1]                            
                        Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index] - states[rowBlock.Index, colBlock.Index - 1] - shift, new Gaussian(0, 1));
                    }
                }
                using (Variable.If(rowBlock.Index > 0))
                {
                    using (ForEachBlock colBlock = Variable.ForEach(cols))
                    {
                        // Vertical connection: x[row, col] to x[row - 1, col]
                        Variable.ConstrainEqualRandom(states[rowBlock.Index, colBlock.Index] - states[rowBlock.Index - 1, colBlock.Index] - shift, new Gaussian(0, 1));
                    }
                }
            }
            states.AddAttribute(new DivideMessages(false));

            double[,] observedValues = new double[length, length];
            for (int i = 0; i < length; i++)
            {
                for (int j = 0; j < length; j++)
                {
                    observedValues[i, j] = 1 + i + j;
                }
            }
            observation.ObservedValue = observedValues;

            InferenceEngine engine = new InferenceEngine();
            //engine.Compiler.UnrollLoops = true;
            engine.OptimiseForVariables = new List<IVariable>() { states, shift };
            engine.NumberOfIterations = 7;
            var shiftActual = engine.Infer<Gaussian>(shift);
            var shiftExpected = new Gaussian(0.9926, 0.009225);
            if (false)
            {
                for (int iter = 1; iter < 20; iter++)
                {
                    engine.NumberOfIterations = iter;
                    shiftActual = engine.Infer<Gaussian>(shift);
                    Console.WriteLine("{0} {1} error={2}", iter, shiftActual, shiftExpected.MaxDiff(shiftActual));
                }
            }
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            Console.WriteLine("shift = {0} converged = {1}", shiftActual, shiftExpected);
            var statesActual = engine.Infer<Gaussian[,]>(states);
            Gaussian state00Converged = new Gaussian(1.008, 0.4142);
            Gaussian state10Converged = new Gaussian(2.004, 0.3234);
            Gaussian state99Converged = new Gaussian(18.97, 0.4142);
            Console.WriteLine("state[0,0] = {0} converged = {1}", statesActual[0, 0], state00Converged);
            Console.WriteLine("state[1,0] = {0} converged = {1}", statesActual[1, 0], state10Converged);
            Console.WriteLine("state[9,9] = {0} converged = {1}", statesActual[9, 9], state99Converged);
            Assert.True(shiftExpected.MaxDiff(shiftActual) < 0.01);
            Assert.True(state00Converged.MaxDiff(statesActual[0, 0]) < 0.01);
            Assert.True(state10Converged.MaxDiff(statesActual[1, 0]) < 0.01);
            Assert.True(state99Converged.MaxDiff(statesActual[9, 9]) < 0.02);
        }

        [Fact]
        public void JaggedGridTest()
        {
            int length = 10;

            Range rows = new Range(length).Named("i");
            Range cols = new Range(length).Named("j");

            var states = Variable.Array(Variable.Array<double>(cols), rows).Named("states");
            states[rows][cols] = Variable.GaussianFromMeanAndVariance(1, 1).ForEach(rows, cols);

            VariableArray2D<double> observation = Variable.Array<double>(rows, cols).Named("observation");
            bool useObservation = false;
            if (useObservation)
                observation[rows, cols] = Variable.GaussianFromMeanAndVariance(states[rows][cols], 1);

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (ForEachBlock colBlock = Variable.ForEach(cols))
                {
                    using (Variable.If(colBlock.Index > 0))
                    {
                        // Horizontal connection: x[row, col] to x[row, col - 1]                            
                        Variable.ConstrainEqualRandom(states[rowBlock.Index][colBlock.Index] - states[rowBlock.Index][colBlock.Index - 1], new Gaussian(0, 1));
                    }
                }
                using (Variable.If(rowBlock.Index > 0))
                {
                    using (ForEachBlock colBlock = Variable.ForEach(cols))
                    {
                        // Vertical connection: x[row, col] to x[row - 1, col]
                        Variable.ConstrainEqualRandom(states[rowBlock.Index][colBlock.Index] - states[rowBlock.Index - 1][colBlock.Index], new Gaussian(0, 1));
                    }
                }
            }
            states.AddAttribute(new DivideMessages(false));

            double[,] observedValues = new double[length, length];
            for (int i = 0; i < length; i++)
            {
                for (int j = 0; j < length; j++)
                {
                    observedValues[i, j] = 1;
                }
            }
            observation.ObservedValue = observedValues;

            InferenceEngine engine = new InferenceEngine();
            engine.OptimiseForVariables = new List<IVariable>() { states };
            engine.NumberOfIterations = 1;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var statesActual = engine.Infer<Gaussian[][]>(states);
            Gaussian state00Converged = new Gaussian(1, 0.4124);
            Gaussian state10Converged = new Gaussian(1, 0.3218);
            Gaussian state99Converged = new Gaussian(1, 0.4124);
            Console.WriteLine("state[0,0] = {0} converged = {1}", statesActual[0][0], state00Converged);
            Console.WriteLine("state[1,0] = {0} converged = {1}", statesActual[1][0], state10Converged);
            Console.WriteLine("state[9,9] = {0} converged = {1}", statesActual[9][9], state99Converged);
            Assert.True(state00Converged.MaxDiff(statesActual[0][0]) < 0.01);
            Assert.True(state10Converged.MaxDiff(statesActual[1][0]) < 0.015);
            Assert.True(state99Converged.MaxDiff(statesActual[9][9]) < 0.01);
        }

        [Fact]
        public void JaggedCubeTest()
        {
            int length = 10;

            Range rows = new Range(length).Named("i");
            Range cols = new Range(length).Named("j");
            Range height = new Range(length).Named("k");

            var states = Variable.Array(Variable.Array(Variable.Array<double>(cols), rows), height).Named("states");
            states[height][rows][cols] = Variable.GaussianFromMeanAndVariance(1, 1).ForEach(height, rows, cols);

            using (ForEachBlock heightBlock = Variable.ForEach(height))
            {
                using (ForEachBlock rowBlock = Variable.ForEach(rows))
                {
                    using (ForEachBlock colBlock = Variable.ForEach(cols))
                    {
                        using (Variable.If(colBlock.Index > 0))
                        {
                            // Horizontal connection: x[row, col] to x[row, col - 1]                            
                            Variable.ConstrainEqualRandom(states[heightBlock.Index][rowBlock.Index][colBlock.Index] - states[heightBlock.Index][rowBlock.Index][colBlock.Index - 1], new Gaussian(0, 1));
                        }
                    }
                    using (Variable.If(rowBlock.Index > 0))
                    {
                        using (ForEachBlock colBlock = Variable.ForEach(cols))
                        {
                            // Vertical connection: x[row, col] to x[row - 1, col]
                            Variable.ConstrainEqualRandom(states[heightBlock.Index][rowBlock.Index][colBlock.Index] - states[heightBlock.Index][rowBlock.Index - 1][colBlock.Index], new Gaussian(0, 1));
                        }
                    }
                }
                using (Variable.If(heightBlock.Index > 0))
                {
                    using (ForEachBlock rowBlock = Variable.ForEach(rows))
                    {
                        using (ForEachBlock colBlock = Variable.ForEach(cols))
                        {
                            // Height connection
                            Variable.ConstrainEqualRandom(states[heightBlock.Index][rowBlock.Index][colBlock.Index] - states[heightBlock.Index - 1][rowBlock.Index][colBlock.Index], new Gaussian(0, 1));
                        }
                    }
                }
            }
            states.AddAttribute(new DivideMessages(false));

            InferenceEngine engine = new InferenceEngine();
            engine.OptimiseForVariables = new List<IVariable>() { states };
            engine.NumberOfIterations = 1;
            Console.WriteLine("After {0} Iterations", engine.NumberOfIterations);
            var statesActual = engine.Infer<Gaussian[][][]>(states);
            Gaussian state000Converged = new Gaussian(1, 0.3015);
            Gaussian state100Converged = new Gaussian(1, 0.2435);
            Gaussian state999Converged = new Gaussian(1, 0.3015);
            Console.WriteLine("state[0,0,0] = {0} converged = {1}", statesActual[0][0][0], state000Converged);
            Console.WriteLine("state[1,0,0] = {0} converged = {1}", statesActual[1][0][0], state100Converged);
            Console.WriteLine("state[9,9,9] = {0} converged = {1}", statesActual[9][9][9], state999Converged);
            Assert.True(state000Converged.MaxDiff(statesActual[0][0][0]) < 0.01);
            Assert.True(state100Converged.MaxDiff(statesActual[1][0][0]) < 0.011);
            Assert.True(state999Converged.MaxDiff(statesActual[9][9][9]) < 0.01);
        }

        [Fact]
        public void OffsetIndexingTest()
        {
            int length = 10;

            Range rows = new Range(length).Named("i");
            VariableArray<double> x = Variable.Array<double>(rows).Named("x");
            x[rows] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(rows);

            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (Variable.If(rowBlock.Index >= 1))
                {
                    Variable.ConstrainEqualRandom(x[rowBlock.Index - 1], new Gaussian(0, 1));
                }
                using (Variable.If(rowBlock.Index > 1))
                {
                    Variable.ConstrainEqualRandom(x[rowBlock.Index - 1], new Gaussian(0, 1));
                }
                using (Variable.If(rowBlock.Index <= length - 2))
                {
                    Variable.ConstrainEqualRandom(x[rowBlock.Index + 1], new Gaussian(0, 1));
                }
                using (Variable.If(rowBlock.Index < length - 1))
                {
                    Variable.ConstrainEqualRandom(x[rowBlock.Index + 1], new Gaussian(0, 1));
                }
                using (Variable.If(rowBlock.Index == 1))
                {
                    Variable.ConstrainEqualRandom(x[rowBlock.Index - 1], new Gaussian(0, 1));
                }
                using (Variable.If(rowBlock.Index != 1))
                {
                    Variable.ConstrainEqualRandom(x[rowBlock.Index], new Gaussian(0, 1));
                }
            }

            InferenceEngine engine = new InferenceEngine();
            IList<Gaussian> xActual = engine.Infer<IList<Gaussian>>(x);
            engine.Compiler.UnrollLoops = true;
            IList<Gaussian> xExpected = engine.Infer<IList<Gaussian>>(x);
            Console.WriteLine(StringUtil.JoinColumns("x = ", xActual, " should be ", xExpected));
            Assert.True(((Diffable)xExpected).MaxDiff(xActual) < 1e-10);
        }

        // Fails because Channel2 creates multiple state_uses arrays
        // ChannelAnalysis doesn't know that (index>0) and (index<0) are disjoint conditions, while VariableTransform assumes they are disjoint
        // perhaps VariableTransform should post an assertion that they are disjoint
        // these assertions can be in the form of bindings, that get merged in before every reduction
        [Trait("Category", "OpenBug")]
        [Fact]
        public void ChainMiddleRootTest()
        {
            int length = 10;
            int rootIndex = length / 2;
            Range t = new Range(length);
            VariableArray<double> state = Variable.Array<double>(t).Named("state");
            using (var fb = Variable.ForEach(t))
            {
                using (Variable.If(fb.Index == rootIndex))
                {
                    state[fb.Index] = Variable.GaussianFromMeanAndVariance(0, 1);
                }
                using (Variable.If(fb.Index > rootIndex))
                {
                    state[fb.Index] = Variable.GaussianFromMeanAndVariance(state[fb.Index - 1], 1);
                }
                using (Variable.If(fb.Index < rootIndex))
                {
                    state[fb.Index] = Variable.GaussianFromMeanAndVariance(state[fb.Index + 1], 1);
                }
            }
            Variable.ConstrainEqual(state[length - 1], 1);

            InferenceEngine engine = new InferenceEngine();
            var stateActual = engine.Infer<IList<Gaussian>>(state);
            var stateExpected = ChainMiddleRootUnrolled(length);
            Console.WriteLine(StringUtil.JoinColumns("state = ", stateActual, " should be ", stateExpected));
        }

        public IList<Gaussian> ChainMiddleRootUnrolled(int length)
        {
            Variable<double>[] state = new Variable<double>[length];
            int rootIndex = length / 2;

            for (int i = 0; i < length; i++)
            {
                if (i < rootIndex)
                {
                    state[i] = state[i + 1] + Variable.GaussianFromMeanAndVariance(0, 1);
                }
                else if (i == rootIndex)
                {
                    state[i] = Variable.GaussianFromMeanAndVariance(0, 1);
                }
                else if (i > rootIndex)
                {
                    state[i] = state[i - 1] + Variable.GaussianFromMeanAndVariance(0, 1);
                }
            }
            state[length - 1].ObservedValue = 1;

            InferenceEngine engine = new InferenceEngine();
            return Util.ArrayInit(length, i => engine.Infer<Gaussian>(state[i]));
        }

        /// <summary>
        /// This test fails with "The distribution is improper" during initialization due to a poor initialization schedule.
        /// receivingSkill_depth1_F[1] is uniform during the backward loop.  It needs to be initialized by a forward loop.
        /// Previously failed with "Internal: schedule splits group 278"
        /// This happens due to a mistake by the repair algorithm (and possibly also rotation).  Best seen via the TransformBrowser.
        /// Also fails because servingSkill_depth1_F doesn't have its requirements.  It has an Any requirement where the wrong option is chosen.
        /// </summary>
        [Trait("Category", "OpenBug")]
        [Fact]
        public void TennisTest()
        {
            var engine = new InferenceEngine();
            //engine.Compiler.UseParallelForLoops = true;
            int nPlayers = 2;
            int nYears = 2;

            var skillPrecisionPrior = Gamma.FromShapeAndRate(2, 2 * 1 * 1);
            var performancePrecisionPrior = Gamma.FromShapeAndRate(2, 2 * 1 * 1);
            var skillChangePrecisionPrior = Gamma.FromShapeAndRate(2, 2 * 0.5 * 0.5);

            var performancePrecision = Variable.Random(performancePrecisionPrior).Named("performancePrecision");
            var skillChangePrecision = Variable.Random(skillChangePrecisionPrior).Named("skillChangePrecision");
            var skillMean = Variable.Observed<double>(0).Named("skillMean");
            var skillPrecision = Variable.Random(skillPrecisionPrior).Named("skillPrecision");

            var player = new Range(nPlayers).Named("player");
            Range year = new Range(nYears).Named("year");

            var servingSkill = Variable.Array(Variable.Array<double>(player), year).Named("servingSkill");
            var receivingSkill = Variable.Array(Variable.Array<double>(player), year).Named("receivingSkill");
            var firstYear = Variable.Array<int>(player).Named("firstYear");

            using (var yearBlock = Variable.ForEach(year))
            {
                var y = yearBlock.Index;
                using (Variable.If(y == 0))
                {
                    servingSkill[year][player] = Variable.GaussianFromMeanAndPrecision(skillMean, skillPrecision).ForEach(player);
                    receivingSkill[year][player] = Variable.GaussianFromMeanAndPrecision(skillMean, skillPrecision).ForEach(player);
                }
                using (Variable.If(y > 0))
                {
                    using (Variable.ForEach(player))
                    {
                        Variable<bool> isFirstYear = (firstYear[player] >= y).Named("isFirstYear");
                        using (Variable.If(isFirstYear))
                        {
                            servingSkill[year][player] = Variable.GaussianFromMeanAndPrecision(skillMean, skillPrecision);
                            receivingSkill[year][player] = Variable.GaussianFromMeanAndPrecision(skillMean, skillPrecision);
                        }
                        using (Variable.IfNot(isFirstYear))
                        {
                            servingSkill[year][player] = Variable.GaussianFromMeanAndPrecision(servingSkill[y - 1][player], skillChangePrecision);
                            receivingSkill[year][player] = Variable.GaussianFromMeanAndPrecision(receivingSkill[y - 1][player], skillChangePrecision);
                        }
                    }
                }
            }

            // Learn the skills from the data
            firstYear.ObservedValue = Util.ArrayInit(nPlayers, i => 0);
            int[] nGamesData = Util.ArrayInit(nYears, y => 1);
            var nGames = Variable.Observed(nGamesData, year).Named("nGames");
            Range game = new Range(nGames[year]).Named("game");
            var whitePlayer = Variable.Observed(default(int[][]), year, game).Named("whitePlayer");
            var blackPlayer = Variable.Observed(default(int[][]), year, game).Named("blackPlayer");
            var GameP1ServeCount = Variable.Observed(default(int[][]), year, game).Named("GameP1ServeScore");
            var GameP2ServeCount = Variable.Observed(default(int[][]), year, game).Named("GameP2ServeScore");
            var GameP1ServeWinCount = Variable.Observed(default(int[][]), year, game).Named("GameP1ServeWinCount");
            var GameP2ServeWinCount = Variable.Observed(default(int[][]), year, game).Named("GameP2ServeWinCount");

            whitePlayer.ObservedValue = Util.ArrayInit(nYears, y => Util.ArrayInit(nGamesData[y], g => 0));
            blackPlayer.ObservedValue = Util.ArrayInit(nYears, y => Util.ArrayInit(nGamesData[y], g => 1));
            GameP1ServeCount.ObservedValue = Util.ArrayInit(nYears, y => Util.ArrayInit(nGamesData[y], g => 1));
            GameP2ServeCount.ObservedValue = Util.ArrayInit(nYears, y => Util.ArrayInit(nGamesData[y], g => 1));
            GameP1ServeWinCount.ObservedValue = Util.ArrayInit(nYears, y => Util.ArrayInit(nGamesData[y], g => 0));
            GameP2ServeWinCount.ObservedValue = Util.ArrayInit(nYears, y => Util.ArrayInit(nGamesData[y], g => 0));

            using (Variable.ForEach(year))
            {
                using (Variable.ForEach(game))
                {
                    var w = whitePlayer[year][game].Named("w");
                    var b = blackPlayer[year][game].Named("b");
                    Variable<double> white_performance_serving = Variable.GaussianFromMeanAndPrecision(servingSkill[year][w], performancePrecision).Named("white_performance_serving");
                    Variable<double> black_performance_serving = Variable.GaussianFromMeanAndPrecision(servingSkill[year][b], performancePrecision).Named("black_performance_serving");
                    Variable<double> white_performance_receiving = Variable.GaussianFromMeanAndPrecision(receivingSkill[year][w], performancePrecision).Named("white_performance_receiving");
                    Variable<double> black_performance_receiving = Variable.GaussianFromMeanAndPrecision(receivingSkill[year][b], performancePrecision).Named("black_performance_receiving");
                    Variable<double> P1ServeWinProb = Variable.Logistic(white_performance_serving - black_performance_receiving).Named("P1ServeWinProb");
                    Variable<double> P2ServeWinProb = Variable.Logistic(black_performance_serving - white_performance_receiving).Named("P2ServeWinProb");
                    Variable<int> P1ServeWinCountBinomial = Variable.Binomial(GameP1ServeCount[year][game], P1ServeWinProb).Named("P1ServeWinCountBinomial");
                    Variable<int> P2ServeWinCountBinomial = Variable.Binomial(GameP2ServeCount[year][game], P2ServeWinProb).Named("P2ServeWinCountBinomial");
                    Variable.ConstrainEqual(GameP1ServeWinCount[year][game], P1ServeWinCountBinomial);
                    Variable.ConstrainEqual(GameP2ServeWinCount[year][game], P2ServeWinCountBinomial);
                }
            }
            //engine.Compiler.UseSerialSchedules = false;
            var SkillsPosterior = engine.Infer<Gaussian[][]>(receivingSkill);
        }
    }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif
}

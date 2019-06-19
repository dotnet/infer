// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests
{
    using System;
    using Xunit;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors;
    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;

    /// <summary>
    /// Tests for gate operations.
    /// </summary>
    public class GateOpTests
    {
        /// <summary>
        /// Tests EP and BP gate exit ops for Bernoulli random variable for correctness.
        /// </summary>
        [Fact]
        public void BernoulliEnterTest()
        {
            DoBernoulliEnterTest(0.3, 0.8, 0.4);
            DoBernoulliEnterTest(0.7, 0.8, 0.2);
            DoBernoulliEnterTest(0.7, 0.8, 0.0);
            DoBernoulliEnterTest(0.7, 0.8, 1.0);
        }

        /// <summary>
        /// Tests EP and BP gate exit ops for Bernoulli random variable for correctness.
        /// </summary>
        [Fact]
        public void BernoulliExitTest()
        {
            // Test different shifts
            this.DoBernoulliExitTest(2.0/3, 3.0/4, 4.0/5, 0);
            this.DoBernoulliExitTest(2.0/3, 3.0/4, 4.0/5, 1e10);
            this.DoBernoulliExitTest(2.0/3, 3.0/4, 4.0/5, -1e10);

            // Test different message parameters
            this.DoBernoulliExitTest(0.1, 0.2, 0.3, 0);
            this.DoBernoulliExitTest(0.0, 0.2, 0.3, 0);
            this.DoBernoulliExitTest(1.0, 0.2, 0.3, 0);
            this.DoBernoulliExitTest(0.1, 0.0, 0.3, 0);
            this.DoBernoulliExitTest(0.1, 1.0, 0.3, 0);
            this.DoBernoulliExitTest(0.1, 0.2, 0.0, 0);
            this.DoBernoulliExitTest(0.1, 0.2, 1.0, 0);
        }

        /// <summary>
        /// Tests EP gate exit ops for Gaussian random variable for correctness.
        /// </summary>
        [Fact]
        public void GaussianExitTest()
        {
            double pb = 2.0/3;
            var cases = new[] {Bernoulli.FromLogOdds(Math.Log(pb)), Bernoulli.FromLogOdds(Math.Log(1 - pb))};
            var g1 = new Gaussian(1, 2);
            var g2 = new Gaussian(3, 4);
            var values = new[] {g1, g2};
            double meanDiff = g1.GetMean() - g2.GetMean();
            var expected = new Gaussian(
                (pb*g1.GetMean()) + ((1 - pb)*g2.GetMean()),
                (pb*g1.GetVariance()) + ((1 - pb)*g2.GetVariance()) + (pb*(1 - pb)*meanDiff*meanDiff));
            var actual = GateExitOp<double>.ExitAverageConditional(new Gaussian(), cases, values, new Gaussian());
            Assert.True(actual.MaxDiff(expected) < 1e-4);
        }

        /// <summary>
        /// Tests EP and BP gate exit ops for Bernoulli random variable for correctness given message parameters and a shift.
        /// </summary>
        /// <param name="gate1ProbTrue">Probability of being true for the variable when the selector is true.</param>
        /// <param name="gate2ProbTrue">Probability of being true for the variable when the selector is false.</param>
        /// <param name="selectorProbTrue">Probability of being true for the selector variable.</param>
        /// <param name="shift">The value of the shift.</param>
        private void DoBernoulliExitTest(double gate1ProbTrue, double gate2ProbTrue, double selectorProbTrue, double shift)
        {
            const double ExitTwoProbTrue = 0.3; // ExitTwo op depends on the incoming message from outside the gate
            var cases = new[] {Bernoulli.FromLogOdds(Math.Log(selectorProbTrue) - shift), Bernoulli.FromLogOdds(Math.Log(1 - selectorProbTrue) - shift)};
            var values = new[] {new Bernoulli(gate1ProbTrue), new Bernoulli(gate2ProbTrue)};
            var exitTwo = new Bernoulli(ExitTwoProbTrue);
            Bernoulli value1, value2;

            double expectedProbTrueFromExit = (selectorProbTrue*gate1ProbTrue) + ((1 - selectorProbTrue)*gate2ProbTrue);
            value1 = GateExitOp<bool>.ExitAverageConditional(new Bernoulli(), cases, values, new Bernoulli());
            value2 = BeliefPropagationGateExitOp.ExitAverageConditional(cases, values, new Bernoulli());
            Assert.Equal(expectedProbTrueFromExit, value1.GetProbTrue(), 1e-4);
            Assert.Equal(expectedProbTrueFromExit, value2.GetProbTrue(), 1e-4);

            double gate1Scale = (ExitTwoProbTrue*gate1ProbTrue) + ((1 - ExitTwoProbTrue)*(1 - gate1ProbTrue));
            double gate2Scale = (ExitTwoProbTrue*gate2ProbTrue) + ((1 - ExitTwoProbTrue)*(1 - gate2ProbTrue));
            double expectedProbTrueFromExitTwo =
                (selectorProbTrue*gate1ProbTrue/gate1Scale) + ((1 - selectorProbTrue)*gate2ProbTrue/gate2Scale);
            double expectedProbFalseFromExitTwo =
                (selectorProbTrue*(1 - gate1ProbTrue)/gate1Scale) + ((1 - selectorProbTrue)*(1 - gate2ProbTrue)/gate2Scale);
            expectedProbTrueFromExitTwo /= expectedProbTrueFromExitTwo + expectedProbFalseFromExitTwo;
            value1 = GateExitTwoOp.ExitTwoAverageConditional<Bernoulli>(exitTwo, cases[0], cases[1], values, new Bernoulli());
            value2 = BeliefPropagationGateExitTwoOp.ExitTwoAverageConditional(exitTwo, cases[0], cases[1], values, new Bernoulli());
            Assert.Equal(expectedProbTrueFromExitTwo, value1.GetProbTrue(), 1e-4);
            Assert.Equal(expectedProbTrueFromExitTwo, value2.GetProbTrue(), 1e-4);
        }

        /// <summary>
        /// Tests EP and BP gate enter ops for Bernoulli random variable for correctness given message parameters.
        /// </summary>
        /// <param name="valueProbTrue">Probability of being true for the variable entering the gate.</param>
        /// <param name="enterOneProbTrue">Probability of being true for the variable approximation inside the gate when the selector is true.</param>
        /// <param name="selectorProbTrue">Probability of being true for the selector variable.</param>
        private void DoBernoulliEnterTest(double valueProbTrue, double enterOneProbTrue, double selectorProbTrue)
        {
            var value = new Bernoulli(valueProbTrue);
            var enterOne = new Bernoulli(enterOneProbTrue);
            var selector = new Bernoulli(selectorProbTrue);
            var selectorInverse = new Bernoulli(selector.GetProbFalse());
            var discreteSelector = new Discrete(selector.GetProbTrue(), selector.GetProbFalse());
            var discreteSelectorInverse = new Discrete(selector.GetProbFalse(), selector.GetProbTrue());
            var cases = new[] {Bernoulli.FromLogOdds(selector.GetLogProbTrue()), Bernoulli.FromLogOdds(selector.GetLogProbFalse())};

            // Compute expected message
            double logShift = enterOne.GetLogNormalizer() + value.GetLogNormalizer() - (value*enterOne).GetLogNormalizer();
            double expectedProbTrue = selector.GetProbFalse() + (selector.GetProbTrue()*enterOne.GetProbTrue()*Math.Exp(logShift));
            double expectedProbFalse = selector.GetProbFalse() + (selector.GetProbTrue()*enterOne.GetProbFalse()*Math.Exp(logShift));
            double expectedNormalizer = expectedProbTrue + expectedProbFalse;
            expectedProbTrue /= expectedNormalizer;

            Bernoulli value1, value2;

            // Enter partial (bernoulli selector, first case)
            value1 = GateEnterPartialOp<bool>.ValueAverageConditional(
                new[] {enterOne}, selector, value, new[] {0}, new Bernoulli());
            value2 = BeliefPropagationGateEnterPartialOp.ValueAverageConditional(
                new[] {enterOne}, selector, value, new[] {0}, new Bernoulli());
            Assert.Equal(expectedProbTrue, value1.GetProbTrue(), 1e-4);
            Assert.Equal(expectedProbTrue, value2.GetProbTrue(), 1e-4);

            // Enter partial (bernoulli selector, second case)
            value1 = GateEnterPartialOp<bool>.ValueAverageConditional(
                new[] {enterOne}, selectorInverse, value, new[] {1}, new Bernoulli());
            value2 = BeliefPropagationGateEnterPartialOp.ValueAverageConditional(
                new[] {enterOne}, selectorInverse, value, new[] {1}, new Bernoulli());
            Assert.Equal(expectedProbTrue, value1.GetProbTrue(), 1e-4);
            Assert.Equal(expectedProbTrue, value2.GetProbTrue(), 1e-4);

            // Enter partial (bernoulli selector, both cases)
            value1 = GateEnterPartialOp<bool>.ValueAverageConditional(
                new[] {enterOne, Bernoulli.Uniform()}, selector, value, new[] {0, 1}, new Bernoulli());
            value2 = BeliefPropagationGateEnterPartialOp.ValueAverageConditional(
                new[] {enterOne, Bernoulli.Uniform()}, selector, value, new[] {0, 1}, new Bernoulli());
            Assert.Equal(expectedProbTrue, value1.GetProbTrue(), 1e-4);
            Assert.Equal(expectedProbTrue, value2.GetProbTrue(), 1e-4);

            // Enter partial (discrete selector, first case)
            value1 = GateEnterPartialOp<bool>.ValueAverageConditional(
                new[] {enterOne}, discreteSelector, value, new[] {0}, new Bernoulli());
            value2 = BeliefPropagationGateEnterPartialOp.ValueAverageConditional(
                new[] {enterOne}, discreteSelector, value, new[] {0}, new Bernoulli());
            Assert.Equal(expectedProbTrue, value1.GetProbTrue(), 1e-4);
            Assert.Equal(expectedProbTrue, value2.GetProbTrue(), 1e-4);

            // Enter partial (discrete selector, second case)
            value1 = GateEnterPartialOp<bool>.ValueAverageConditional(
                new[] {enterOne}, discreteSelectorInverse, value, new[] {1}, new Bernoulli());
            value2 = BeliefPropagationGateEnterPartialOp.ValueAverageConditional(
                new[] {enterOne}, discreteSelectorInverse, value, new[] {1}, new Bernoulli());
            Assert.Equal(expectedProbTrue, value1.GetProbTrue(), 1e-4);
            Assert.Equal(expectedProbTrue, value2.GetProbTrue(), 1e-4);

            // Enter partial (discrete selector, both cases)
            value1 = GateEnterPartialOp<bool>.ValueAverageConditional(
                new[] {enterOne, Bernoulli.Uniform()}, discreteSelector, value, new[] {0, 1}, new Bernoulli());
            value2 = BeliefPropagationGateEnterPartialOp.ValueAverageConditional(
                new[] {enterOne, Bernoulli.Uniform()}, discreteSelector, value, new[] {0, 1}, new Bernoulli());
            Assert.Equal(expectedProbTrue, value1.GetProbTrue(), 1e-4);
            Assert.Equal(expectedProbTrue, value2.GetProbTrue(), 1e-4);

            // Enter one (discrete selector, first case)
            value1 = GateEnterOneOp<bool>.ValueAverageConditional(
                enterOne, discreteSelector, value, 0, new Bernoulli());
            value2 = BeliefPropagationGateEnterOneOp.ValueAverageConditional(
                enterOne, discreteSelector, value, 0, new Bernoulli());
            Assert.Equal(expectedProbTrue, value1.GetProbTrue(), 1e-4);
            Assert.Equal(expectedProbTrue, value2.GetProbTrue(), 1e-4);

            // Enter one (discrete selector, second case)
            value1 = GateEnterOneOp<bool>.ValueAverageConditional(
                enterOne, discreteSelectorInverse, value, 1, new Bernoulli());
            value2 = BeliefPropagationGateEnterOneOp.ValueAverageConditional(
                enterOne, discreteSelectorInverse, value, 1, new Bernoulli());
            Assert.Equal(expectedProbTrue, value1.GetProbTrue(), 1e-4);
            Assert.Equal(expectedProbTrue, value2.GetProbTrue(), 1e-4);

            // Enter partial two  (first case)
            value1 = GateEnterPartialTwoOp.ValueAverageConditional(
                new[] {enterOne}, cases[0], cases[1], value, new[] {0}, new Bernoulli());
            value2 = BeliefPropagationGateEnterPartialTwoOp.ValueAverageConditional(
                new[] {enterOne}, cases[0], cases[1], value, new[] {0}, new Bernoulli());
            Assert.Equal(expectedProbTrue, value1.GetProbTrue(), 1e-4);
            Assert.Equal(expectedProbTrue, value2.GetProbTrue(), 1e-4);

            // Enter partial two  (second case)
            value1 = GateEnterPartialTwoOp.ValueAverageConditional(
                new[] {enterOne}, cases[1], cases[0], value, new[] {1}, new Bernoulli());
            value2 = BeliefPropagationGateEnterPartialTwoOp.ValueAverageConditional(
                new[] {enterOne}, cases[1], cases[0], value, new[] {1}, new Bernoulli());
            Assert.Equal(expectedProbTrue, value1.GetProbTrue(), 1e-4);
            Assert.Equal(expectedProbTrue, value2.GetProbTrue(), 1e-4);

            // Enter partial two (both cases)
            value1 = GateEnterPartialTwoOp.ValueAverageConditional(
                new[] {enterOne, Bernoulli.Uniform()}, cases[0], cases[1], value, new[] {0, 1}, new Bernoulli());
            value2 = BeliefPropagationGateEnterPartialTwoOp.ValueAverageConditional(
                new[] {enterOne, Bernoulli.Uniform()}, cases[0], cases[1], value, new[] {0, 1}, new Bernoulli());
            Assert.Equal(expectedProbTrue, value1.GetProbTrue(), 1e-4);
            Assert.Equal(expectedProbTrue, value2.GetProbTrue(), 1e-4);

            // Enter (discrete selector)
            value1 = GateEnterOp<bool>.ValueAverageConditional(
                new[] {enterOne, Bernoulli.Uniform()}, discreteSelector, value, new Bernoulli());
            value2 = BeliefPropagationGateEnterOp.ValueAverageConditional(
                new[] {enterOne, Bernoulli.Uniform()}, discreteSelector, value, new Bernoulli());
            Assert.Equal(expectedProbTrue, value1.GetProbTrue(), 1e-4);
            Assert.Equal(expectedProbTrue, value2.GetProbTrue(), 1e-4);
        }
    }
}
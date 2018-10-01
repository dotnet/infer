// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Xunit;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;

namespace Microsoft.ML.Probabilistic.Tests
{
    public enum Coin
    {
        Tails,
        Heads
    };

    
    public class EnumTests
    {
        [Fact]
        public void BabyTestEnum()
        {
            // Observed actions
            BabyAction[] observedActions = new BabyAction[] {BabyAction.Smile, BabyAction.Smile, BabyAction.Cry, BabyAction.Smile, BabyAction.LookSilly};
            var actions = Variable.Observed(observedActions);

            // Conditional probabilities of actions given different attitudes
            var actionProbs = Variable.Observed(new Vector[]
                {
                    Vector.FromArray(0.6, 0.2, 0.2), // Happy
                    Vector.FromArray(0.2, 0.6, 0.2), // Unhappy
                    Vector.FromArray(0.4, 0.3, 0.3), // Quiet
                });

            // Model relating attitude to actions
            var attitude = Variable.EnumUniform<Attitude>(actionProbs.Range);
            var attInt = Variable.EnumToInt(attitude);
            var j = actions.Range;
            using (Variable.Switch(attInt))
            {
                actions[j] = Variable.EnumDiscrete<BabyAction>(actionProbs[attInt]).ForEach(j);
            }

            // Inference of the posterior distribution over attitudes
            InferenceEngine ie = new InferenceEngine();
            Console.WriteLine("Posterior over attitude=" + Environment.NewLine + ie.Infer(attitude));
        }

        [Fact]
        public void TwoCoinsEnum()
        {
            var firstCoin = Variable.EnumUniform<Coin>();
            var secondCoin = Variable.EnumUniform<Coin>();
            Variable<bool> bothHeads = ((firstCoin == Coin.Heads) & (secondCoin == Coin.Heads)).Named("bothHeads");
            InferenceEngine ie = new InferenceEngine();
            Bernoulli bothHeadsActual = ie.Infer<Bernoulli>(bothHeads);
            Bernoulli bothHeadsExpected = new Bernoulli(0.25);
            Console.WriteLine("Probability both coins are heads: {0} (should be {1})", bothHeadsActual, bothHeadsExpected);
            bothHeads.ObservedValue = false;
            DiscreteEnum<Coin> firstCoinActual = ie.Infer<DiscreteEnum<Coin>>(firstCoin);
            DiscreteEnum<Coin> firstCoinExpected = DiscreteEnum<Coin>.FromProbs(2.0/3.0, 1.0/3.0);
            Console.WriteLine("Probability distribution over firstCoin: " + firstCoinActual);
            Console.WriteLine("should be: " + firstCoinExpected);
            Assert.True(bothHeadsExpected.MaxDiff(bothHeadsActual) < 1e-10);
            Assert.True(firstCoinExpected.MaxDiff(firstCoinActual) < 1e-10);
        }


        [Fact]
        public void CaseExampleEnum()
        {
            Variable<Coin> c = Variable.EnumDiscrete<Coin>(new double[] {0.5, 0.5});
            Variable<double> x = Variable.New<double>();
            Variable<int> cint = Variable.EnumToInt(c);
            using (Variable.Case(cint, 0))
            {
                x.SetTo(Variable.GaussianFromMeanAndVariance(1, 1));
            }
            using (Variable.Case(cint, 1))
            {
                x.SetTo(Variable.GaussianFromMeanAndVariance(2, 1));
            }
            InferenceEngine engine = new InferenceEngine();
            Gaussian expected = new Gaussian(1.5, 1.25);
            Gaussian actual = engine.Infer<Gaussian>(x);
            Console.WriteLine("x = {0} (should be {1})", actual, expected);
            Assert.True(expected.MaxDiff(actual) < 1e-10);
        }


        public enum Outcome
        {
            Good,
            Bad
        };

        [Fact]
        public void ClinicalTrialEnum()
        {
            // Data from clinical trial
            VariableArray<Outcome> controlGroup =
                Variable.Observed(new Outcome[] {Outcome.Bad, Outcome.Bad, Outcome.Good, Outcome.Good, Outcome.Bad, Outcome.Bad}).Named("controlGroup");
            VariableArray<Outcome> treatedGroup =
                Variable.Observed(new Outcome[] {Outcome.Good, Outcome.Bad, Outcome.Good, Outcome.Good, Outcome.Good, Outcome.Good}).Named("treatedGroup");
            Range i = controlGroup.Range.Named("i");
            Range j = treatedGroup.Range.Named("j");

            // Prior on being effective treatment
            Variable<bool> isEffective = Variable.Bernoulli(0.5).Named("isEffective");
            Variable<Vector> probIfTreated, probIfControl;
            using (Variable.If(isEffective))
            {
                // Model if treatment is effective
                probIfControl = Variable.Dirichlet(new double[] {1.0, 1.0}).Named("probIfControl");
                controlGroup[i] = Variable.EnumDiscrete<Outcome>(probIfControl).ForEach(i);
                probIfTreated = Variable.Dirichlet(new double[] {1.0, 1.0}).Named("probIfTreated");
                treatedGroup[j] = Variable.EnumDiscrete<Outcome>(probIfTreated).ForEach(j);
            }
            using (Variable.IfNot(isEffective))
            {
                // Model if treatment is not effective
                Variable<Vector> probAll = Variable.Dirichlet(new double[] {1.0, 1.0}).Named("probAll");
                controlGroup[i] = Variable.EnumDiscrete<Outcome>(probAll).ForEach(i);
                treatedGroup[j] = Variable.EnumDiscrete<Outcome>(probAll).ForEach(j);
            }
            InferenceEngine ie = new InferenceEngine();
            Console.WriteLine("Probability treatment has an effect = " + ie.Infer(isEffective));
            Console.WriteLine("Distribution over outcomes if given treatment = "
                              + DiscreteEnum<Outcome>.FromVector(ie.Infer<Dirichlet>(probIfTreated).GetMean()));
            Console.WriteLine("Distribution over outcomes if control = "
                              + DiscreteEnum<Outcome>.FromVector(ie.Infer<Dirichlet>(probIfControl).GetMean()));
        }
    }
}
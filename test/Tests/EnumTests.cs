// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Xunit;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Algorithms;
using Range = Microsoft.ML.Probabilistic.Models.Range;

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

        [Fact]
        public void GatedOutcomeAreEqualTest()
        {
            foreach (var algorithm in new Models.Attributes.IAlgorithm[] { new ExpectationPropagation(), new VariationalMessagePassing() })
            {
                Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
                IfBlock block = Variable.If(evidence);
                Vector priorA = Vector.FromArray(0.1, 0.9);
                Vector priorB = Vector.FromArray(0.2, 0.8);
                Variable<Outcome> a = Variable.EnumDiscrete<Outcome>(priorA).Named("a");
                Variable<Outcome> b = Variable.EnumDiscrete<Outcome>(priorB).Named("b");
                Variable<bool> c = (a == b).Named("c");
                double priorC = 0.3;
                Variable.ConstrainEqualRandom(c, new Bernoulli(priorC));
                block.CloseBlock();

                InferenceEngine engine = new InferenceEngine(algorithm);

                double probEqual = priorA.Inner(priorB);
                double evPrior = 0;
                for (int atrial = 0; atrial < 2; atrial++)
                {
                    if (atrial == 1)
                    {
                        a.ObservedValue = Outcome.Bad;
                        probEqual = priorB[1];
                        c.ClearObservedValue();
                        evPrior = System.Math.Log(priorA[1]);
                        priorA[0] = 0.0;
                        priorA[1] = 1.0;
                    }
                    double evExpected = System.Math.Log(probEqual * priorC + (1 - probEqual) * (1 - priorC)) + evPrior;
                    double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                    Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                    if (algorithm is ExpectationPropagation || atrial == 1)
                        Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-5) < 1e-5);

                    Bernoulli cExpected = new Bernoulli(probEqual * priorC / (probEqual * priorC + (1 - probEqual) * (1 - priorC)));
                    Bernoulli cActual = engine.Infer<Bernoulli>(c);
                    Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
                    if (algorithm is ExpectationPropagation || atrial == 1)
                        Assert.True(cExpected.MaxDiff(cActual) < 1e-10);

                    Vector postB = Vector.Zero(2);
                    postB[0] = priorB[0] * (priorA[0] * priorC + priorA[1] * (1 - priorC));
                    postB[1] = priorB[1] * (priorA[1] * priorC + priorA[0] * (1 - priorC));
                    postB.Scale(1.0 / postB.Sum());
                    DiscreteEnum<Outcome> bExpected = new DiscreteEnum<Outcome>(postB);
                    DiscreteEnum<Outcome> bActual = engine.Infer<DiscreteEnum<Outcome>>(b);
                    Console.WriteLine("b = {0} should be {1}", bActual, bExpected);
                    if (algorithm is ExpectationPropagation || atrial == 1)
                        Assert.True(bExpected.MaxDiff(bActual) < 1e-10);

                    if (atrial == 0 && algorithm is VariationalMessagePassing) continue;

                    for (int trial = 0; trial < 2; trial++)
                    {
                        if (trial == 0)
                        {
                            c.ObservedValue = true;
                            evExpected = System.Math.Log(probEqual * priorC) + evPrior;
                        }
                        else
                        {
                            c.ObservedValue = false;
                            evExpected = System.Math.Log((1 - probEqual) * (1 - priorC)) + evPrior;
                        }
                        evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                        Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
                        Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-5) < 1e-5);

                        if (a.IsObserved)
                        {
                            Outcome flip(Outcome x) => (x == Outcome.Good ? Outcome.Bad : Outcome.Good);
                            bExpected = DiscreteEnum<Outcome>.PointMass(c.ObservedValue ? a.ObservedValue : flip(a.ObservedValue));
                        }
                        else
                        {
                            postB[0] = priorB[0] * (c.ObservedValue ? priorA[0] : priorA[1]);
                            postB[1] = priorB[1] * (c.ObservedValue ? priorA[1] : priorA[0]);
                            postB.Scale(1.0 / postB.Sum());
                            bExpected = new DiscreteEnum<Outcome>(postB);
                        }
                        bActual = engine.Infer<DiscreteEnum<Outcome>>(b);
                        Console.WriteLine("b = {0} should be {1}", bActual, bExpected);
                        Assert.True(bExpected.MaxDiff(bActual) < 1e-10);
                    }
                }
            }
        }
    }
}
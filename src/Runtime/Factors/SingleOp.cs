// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Diagnostics;
    using Distributions;
    using Distributions.Automata;
    using Math;
    using Factors.Attributes;
    using Utilities;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SingleOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Single")]
    [Quality(QualityBand.Experimental)]
    public static class SingleOp
    {
        public static DiscreteChar CharacterAverageConditional(string str)
        {
            Argument.CheckIfNotNull(str, "str");

            if (str.Length == 1)
            {
                return DiscreteChar.PointMass(str[0]);
            }

            throw new AllZeroException("The length of the given string is not 1.");
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SingleOp"]/message_doc[@name="CharacterAverageConditional(StringDistribution)"]/*'/>
        public static DiscreteChar CharacterAverageConditional(StringDistribution str)
        {
            Argument.CheckIfNotNull(str, "str");

            if (str.IsPointMass)
            {
                return CharacterAverageConditional(str.Point);
            }

            Vector resultLogProb = PiecewiseVector.Constant(char.MaxValue + 1, double.NegativeInfinity);
            StringAutomaton probFunc = str.GetWorkspaceOrPoint();
            StringAutomaton.EpsilonClosure startEpsilonClosure = probFunc.Start.GetEpsilonClosure();
            for (int stateIndex = 0; stateIndex < startEpsilonClosure.Size; ++stateIndex)
            {
                StringAutomaton.State state = startEpsilonClosure.GetStateByIndex(stateIndex);
                Weight stateLogWeight = startEpsilonClosure.GetStateWeightByIndex(stateIndex);
                for (int transitionIndex = 0; transitionIndex < state.TransitionCount; ++transitionIndex)
                {
                    StringAutomaton.Transition transition = state.GetTransition(transitionIndex);
                    if (!transition.IsEpsilon)
                    {
                        StringAutomaton.State destState = probFunc.States[transition.DestinationStateIndex];
                        StringAutomaton.EpsilonClosure destStateClosure = destState.GetEpsilonClosure();
                        if (!destStateClosure.EndWeight.IsZero)
                        {
                            Weight weight = Weight.Product(stateLogWeight, transition.Weight, destStateClosure.EndWeight);
                            var logProbs = transition.ElementDistribution.Value.GetProbs();
                            logProbs.SetToFunction(logProbs, Math.Log);
                            resultLogProb = LogSumExp(resultLogProb, logProbs, weight);
                        }
                    }                        
                }
            }

            if (resultLogProb.All(double.IsNegativeInfinity))
            {
                throw new AllZeroException("An input distribution assigns zero probability to all single character strings.");
            }

            Vector resultProb = PiecewiseVector.Zero(char.MaxValue + 1);
            double logNormalizer = resultLogProb.LogSumExp();
            resultProb.SetToFunction(resultLogProb, lp => Math.Exp(lp - logNormalizer));
            return DiscreteChar.FromVector(resultProb);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SingleOp"]/message_doc[@name="StrAverageConditional(DiscreteChar)"]/*'/>
        public static StringDistribution StrAverageConditional(DiscreteChar character)
        {
            return StringDistribution.Char(character);
        }

        private static Vector LogSumExp(Vector logValues1, Vector logValues2, Weight values2Scale)
        {
            Debug.Assert(logValues1.Count == logValues2.Count);
            logValues1.SetToFunction(
                logValues1,
                logValues2,
                (x, y) => Weight.Sum(Weight.FromLogValue(x), Weight.Product(values2Scale, Weight.FromLogValue(y))).LogValue);
            return logValues1;
        }
    }
}

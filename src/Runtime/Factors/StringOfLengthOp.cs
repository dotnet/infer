// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Distributions;
    using Distributions.Automata;
    using Math;
    using Attributes;
    using Utilities;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringOfLengthOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "StringOfLength")]
    [Quality(QualityBand.Experimental)]
    public static class StringOfLengthOp
    {
        #region EP messages

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringOfLengthOp"]/message_doc[@name="StrAverageConditional(DiscreteChar, int)"]/*'/>
        public static StringDistribution StrAverageConditional(DiscreteChar allowedChars, int length)
        {
            Argument.CheckIfValid(allowedChars.IsPartialUniform(), "allowedChars", "The set of allowed characters must be passed as a partial uniform distribution.");
            
            return StringDistribution.Repeat(allowedChars, length, length);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringOfLengthOp"]/message_doc[@name="StrAverageConditional(DiscreteChar, Discrete)"]/*'/>
        public static StringDistribution StrAverageConditional(DiscreteChar allowedChars, Discrete length)
        {
            Argument.CheckIfNotNull(length, "length");
            Argument.CheckIfValid(allowedChars.IsPartialUniform(), "allowedChars", "The set of allowed characters must be passed as a partial uniform distribution.");

            double logNormalizer = allowedChars.GetLogAverageOf(allowedChars);
            var oneCharacter = StringAutomaton.ConstantOnElementLog(logNormalizer, allowedChars);
            var manyCharacters = StringAutomaton.Repeat(oneCharacter, length.GetWorkspace());
            return StringDistribution.FromWorkspace(manyCharacters);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringOfLengthOp"]/message_doc[@name="LengthAverageConditional(StringDistribution, DiscreteChar, Discrete)"]/*'/>
        public static Discrete LengthAverageConditional(StringDistribution str, DiscreteChar allowedChars, Discrete result)
        {
            Vector resultProbabilities = result.GetWorkspace();
            for (int length = 0; length < result.Dimension; ++length)
            {
                StringDistribution factor = StringDistribution.Repeat(allowedChars, length, length);
                resultProbabilities[length] = Math.Exp(factor.GetLogAverageOf(str));
            }
            
            result.SetProbs(resultProbabilities);
            return result;
        }

        #endregion

        #region Evidence messages

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringOfLengthOp"]/message_doc[@name="LogEvidenceRatio(DiscreteChar, Discrete, StringDistribution)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(DiscreteChar allowedChars, Discrete length, StringDistribution str)
        {
            return 0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringOfLengthOp"]/message_doc[@name="LogEvidenceRatio(DiscreteChar, int, String)"]/*'/>
        public static double LogEvidenceRatio(DiscreteChar allowedChars, int length, string str)
        {
            StringDistribution toStr = StrAverageConditional(allowedChars, length);
            return toStr.GetLogProb(str);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringOfLengthOp"]/message_doc[@name="LogEvidenceRatio(DiscreteChar, Discrete, String)"]/*'/>
        public static double LogEvidenceRatio(DiscreteChar allowedChars, Discrete length, string str)
        {
            StringDistribution toStr = StrAverageConditional(allowedChars, length);
            return toStr.GetLogProb(str);
        }

        #endregion
    }
}

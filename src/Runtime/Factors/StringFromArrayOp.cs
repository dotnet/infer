// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringFromArrayOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "StringFromArray")]
    [Quality(QualityBand.Experimental)]
    public static class StringFromArrayOp
    {
        #region EP messages

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringFromArrayOp"]/message_doc[@name="StrAverageConditional(IList{DiscreteChar})"]/*'/>
        public static StringDistribution StrAverageConditional(IList<DiscreteChar> characters)
        {
            StringDistribution result = StringDistribution.Empty();
            for (int i = 0; i < characters.Count; ++i)
            {
                result.AppendInPlace(characters[i]);
            }

            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringFromArrayOp"]/message_doc[@name="CharactersAverageConditional{TDiscreteCharList}(StringDistribution, IList{DiscreteChar}, TDiscreteCharList)"]/*'/>
        /// <typeparam name="TDiscreteCharList">The type of an outgoing message to <c>chars</c>.</typeparam>
        public static TDiscreteCharList CharactersAverageConditional<TDiscreteCharList>(
            StringDistribution str, IList<DiscreteChar> characters, TDiscreteCharList result)
            where TDiscreteCharList : IList<DiscreteChar>
        {
            for (int i = 0; i < characters.Count; ++i)
            {
                // TODO: perhaps there is a faster way to extract the distribution of interest
                var reweightedStr = str.Product(GetCharWeighter(characters, i));
                var outgoingMessageAsStr = SubstringOp.SubAverageConditional(reweightedStr, i, 1);
                if (outgoingMessageAsStr.IsZero())
                {
                    throw new AllZeroException("Impossible model detected in StringFromCharsOp.");
                }
                
                result[i] = SingleOp.CharacterAverageConditional(outgoingMessageAsStr);
            }

            return result;
        }

        #endregion

        #region Evidence messages

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringFromArrayOp"]/message_doc[@name="LogEvidenceRatio(IList{DiscreteChar}, StringDistribution)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(IList<DiscreteChar> characters, StringDistribution str)
        {
            return 0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringFromArrayOp"]/message_doc[@name="LogEvidenceRatio(IList{DiscreteChar}, String)"]/*'/>
        public static double LogEvidenceRatio(IList<DiscreteChar> characters, string str)
        {
            StringDistribution toStr = StrAverageConditional(characters);
            return toStr.GetLogProb(str);
        }

        #endregion

        #region Helpers

        /// <summary>
        /// Creates a string distribution <c>P(s) = \prod_i P_i(s_i)^I[i != j]</c>,
        /// where <c>P_i(c)</c> is a given array of character distributions and <c>j</c> is a given position in the array.
        /// </summary>
        /// <param name="characters">The distributions over individual characters.</param>
        /// <param name="excludedPos">The character to skip.</param>
        /// <returns>The created distribution.</returns>
        private static StringDistribution GetCharWeighter(IList<DiscreteChar> characters, int excludedPos)
        {
            StringDistribution result = StringDistribution.Empty();
            for (int i = 0; i < characters.Count; ++i)
            {
                result.AppendInPlace(i == excludedPos ? DiscreteChar.Uniform() : characters[i]);
            }

            return result;
        }

        #endregion
    }
}

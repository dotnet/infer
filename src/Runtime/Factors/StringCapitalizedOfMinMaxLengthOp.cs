// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringCapitalizedOfMinMaxLengthOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "StringCapitalized", typeof(int), typeof(int))]
    [Quality(QualityBand.Experimental)]
    public static class StringCapitalizedOfMinMaxLengthOp
    {
        #region EP messages

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringCapitalizedOfMinMaxLengthOp"]/message_doc[@name="StrAverageConditional(int, int)"]/*'/>
        public static StringDistribution StrAverageConditional(int minLength, int maxLength)
        {
            return StringDistribution.Capitalized(minLength, maxLength);
        }

        #endregion
    }
}

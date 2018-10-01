// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringCapitalizedOfMinLengthOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "StringCapitalized", typeof(int))]
    [Quality(QualityBand.Experimental)]
    public static class StringCapitalizedOfMinLengthOp
    {
        #region EP messages

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringCapitalizedOfMinLengthOp"]/message_doc[@name="StrAverageConditional(int)"]/*'/>
        public static StringDistribution StrAverageConditional(int minLength)
        {
            return StringDistribution.Capitalized(minLength);
        }

        #endregion
    }
}

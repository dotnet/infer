// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringOfMinLengthOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "String", typeof(int), typeof(DiscreteChar))]
    [Quality(QualityBand.Experimental)]
    public static class StringOfMinLengthOp
    {
        #region EP messages

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringOfMinLengthOp"]/message_doc[@name="StrAverageConditional(DiscreteChar, int)"]/*'/>
        public static StringDistribution StrAverageConditional(DiscreteChar allowedChars, int minLength)
        {
            return StringDistribution.Repeat(allowedChars.WrappedDistribution, minLength);
        }

        #endregion
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringOfMinMaxLengthOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "String", typeof(int), typeof(int), typeof(DiscreteChar))]
    [Quality(QualityBand.Experimental)]
    public static class StringOfMinMaxLengthOp
    {
        #region EP messages

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringOfMinMaxLengthOp"]/message_doc[@name="StrAverageConditional(DiscreteChar, int, int)"]/*'/>
        public static StringDistribution StrAverageConditional(DiscreteChar allowedChars, int minLength, int maxLength)
        {
            return StringDistribution.Repeat(allowedChars.WrappedDistribution, minLength, maxLength);
        }

        #endregion
    }
}

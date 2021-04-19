// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "IsBetween", typeof(double), typeof(double), typeof(double))]
    [Quality(QualityBand.Mature)]
    public static class IsBetweenTruncatedGaussianOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="TruncatedGaussianIsBetweenOp"]/message_doc[@name="AverageLogFactor(TruncatedGaussian)"]/*'/>
        [Skip]
        public static double AverageLogFactor([IgnoreDependency] TruncatedGaussian X)
        {
            return 0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="TruncatedGaussianIsBetweenOp"]/message_doc[@name="LogEvidenceRatio(bool, TruncatedGaussian, double, double)"]/*'/>
        public static double LogEvidenceRatio(bool isBetween, [SkipIfUniform] TruncatedGaussian x, double lowerBound, double upperBound)
        {
            return x.GetLogAverageOf(XAverageConditional(isBetween, lowerBound, upperBound));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="TruncatedGaussianIsBetweenOp"]/message_doc[@name="XAverageConditional(bool, double, double)"]/*'/>
        public static TruncatedGaussian XAverageConditional(bool isBetween, double lowerBound, double upperBound)
        {
            if (!isBetween)
                throw new ArgumentException($"{nameof(TruncatedGaussian)} requires {nameof(isBetween)}=true", nameof(isBetween));
            return new TruncatedGaussian(0, Double.PositiveInfinity, lowerBound, upperBound);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="TruncatedGaussianIsBetweenOp"]/message_doc[@name="XAverageLogarithm(bool, double, double)"]/*'/>
        public static TruncatedGaussian XAverageLogarithm(bool isBetween, double lowerBound, double upperBound)
        {
            return XAverageConditional(isBetween, lowerBound, upperBound);
        }
    }
}

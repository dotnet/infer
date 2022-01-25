// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;
    using System.Text;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="QuantileOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Quantile")]
    [Quality(QualityBand.Preview)]
    public static class QuantileOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianOp"]/message_doc[@name="QuantileAverageConditional(CanGetQuantile{double}, Gaussian)"]/*'/>
        public static Gaussian QuantileAverageConditional(CanGetQuantile<double> canGetQuantile, Gaussian probability)
        {
            if (probability.IsPointMass) return Gaussian.PointMass(canGetQuantile.GetQuantile(probability.Point));
            else throw new NotSupportedException();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianOp"]/message_doc[@name="ProbabilityAverageConditional(Gaussian, CanGetQuantile{double}, Gaussian)"]/*'/>
        public static Gaussian ProbabilityAverageConditional(double quantile, CanGetQuantile<double> canGetQuantile)
        {
            CanGetProbLessThan<double> canGetProbLessThan = (CanGetProbLessThan<double>)canGetQuantile;
            return Gaussian.PointMass(canGetProbLessThan.GetProbLessThan(quantile));
        }


        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianOp"]/message_doc[@name="ProbabilityAverageConditional(Gaussian, CanGetQuantile{double}, double)"]/*'/>
        public static Gaussian ProbabilityAverageConditional(Gaussian quantile, CanGetQuantile<double> canGetQuantile, double probability)
        {
            // The quantile function is the inverse function of the cdf.
            // The derivative of the quantile function is the reciprocal of the pdf.
            double x = canGetQuantile.GetQuantile(probability);
            CanGetLogProb<double> canGetLogProb = (CanGetLogProb<double>)canGetQuantile;
            double dlogp = quantile.MeanTimesPrecision / Math.Exp(canGetLogProb.GetLogProb(x));
            double ddlogp = 0; // approximation ignoring quantile.Precision
            return Gaussian.FromDerivatives(probability, dlogp, ddlogp, false);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianOp"]/message_doc[@name="ProbabilityAverageConditional(Gaussian, CanGetQuantile{double}, Gaussian)"]/*'/>
        public static Gaussian ProbabilityAverageConditional(Gaussian quantile, CanGetQuantile<double> canGetQuantile, [RequiredArgument] Gaussian probability)
        {
            if (quantile.IsPointMass)
            {
                return ProbabilityAverageConditional(quantile.Point, canGetQuantile);
            }
            else if (probability.IsPointMass)
            {
                return ProbabilityAverageConditional(quantile, canGetQuantile, probability.Point);
            }
            else throw new NotSupportedException();
        }
    }
}

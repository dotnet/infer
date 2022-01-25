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

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegralOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Integral")]
    [Quality(QualityBand.Experimental)]
    public static class IntegralOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegralOp"]/message_doc[@name="IntegralAverageConditional(Gaussian, Gaussian, Func{double,double}, ITruncatableDistribution{double})"]/*'/>
        public static Gaussian IntegralAverageConditional([RequiredArgument] Gaussian lowerBound, [RequiredArgument] Gaussian upperBound, Func<double,double> func, ITruncatableDistribution<double> distribution)
        {
            if (lowerBound.IsPointMass && upperBound.IsPointMass)
            {
                return Gaussian.PointMass(Factor.Integral(lowerBound.Point, upperBound.Point, func, distribution));
            }
            else throw new NotSupportedException();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegralOp"]/message_doc[@name="LowerBoundAverageConditional(Gaussian, double, Func{double,double}, ITruncatableDistribution{double})"]/*'/>
        public static Gaussian LowerBoundAverageConditional(Gaussian integral, double lowerBound, Func<double, double> func, ITruncatableDistribution<double> distribution)
        {
            CanGetLogProb<double> canGetLogProb = (CanGetLogProb<double>)distribution;
            double dlogf = integral.MeanTimesPrecision * -func(lowerBound) * Math.Exp(canGetLogProb.GetLogProb(lowerBound));
            double ddlogf = 0; // approximation ignoring integral.Precision
            return Gaussian.FromDerivatives(lowerBound, dlogf, ddlogf, false);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegralOp"]/message_doc[@name="LowerBoundAverageConditional(Gaussian, Gaussian, Func{double,double}, ITruncatableDistribution{double})"]/*'/>
        public static Gaussian LowerBoundAverageConditional(Gaussian integral, [RequiredArgument] Gaussian lowerBound, Func<double, double> func, ITruncatableDistribution<double> distribution)
        {
            if (lowerBound.IsPointMass)
            {
                return LowerBoundAverageConditional(integral, lowerBound.Point, func, distribution);
            }
            else throw new NotSupportedException();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegralOp"]/message_doc[@name="UpperBoundAverageConditional(Gaussian, double, Func{double,double}, ITruncatableDistribution{double})"]/*'/>
        public static Gaussian UpperBoundAverageConditional(Gaussian integral, double upperBound, Func<double, double> func, ITruncatableDistribution<double> distribution)
        {
            CanGetLogProb<double> canGetLogProb = (CanGetLogProb<double>)distribution;
            double dlogf = integral.MeanTimesPrecision * func(upperBound) * Math.Exp(canGetLogProb.GetLogProb(upperBound));
            double ddlogf = 0; // approximation ignoring integral.Precision
            return Gaussian.FromDerivatives(upperBound, dlogf, ddlogf, false);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegralOp"]/message_doc[@name="UpperBoundAverageConditional(Gaussian, Gaussian, Func{double,double}, ITruncatableDistribution{double})"]/*'/>
        public static Gaussian UpperBoundAverageConditional(Gaussian integral, [RequiredArgument] Gaussian upperBound, Func<double, double> func, ITruncatableDistribution<double> distribution)
        {
            if (upperBound.IsPointMass)
            {
                return UpperBoundAverageConditional(integral, upperBound.Point, func, distribution);
            }
            else throw new NotSupportedException();
        }
    }
}

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
        public static Gaussian IntegralAverageConditional([RequiredArgument] Gaussian lowerBound, [RequiredArgument] Gaussian upperBound, Func<double,double> func, ITruncatableDistribution<double> distribution)
        {
            if (lowerBound.IsPointMass && upperBound.IsPointMass)
            {
                return Gaussian.PointMass(Factor.Integral(lowerBound.Point, upperBound.Point, func, distribution));
            }
            else throw new NotSupportedException();
        }

        public static Gaussian LowerBoundAverageConditional(Gaussian integral, [RequiredArgument] Gaussian lowerBound, Func<double, double> func, ITruncatableDistribution<double> distribution)
        {
            CanGetLogProb<double> canGetLogProb = (CanGetLogProb<double>)distribution;
            if (lowerBound.IsPointMass && integral.Precision == 0)
            {
                double dlogf = integral.MeanTimesPrecision * -func(lowerBound.Point) * Math.Exp(canGetLogProb.GetLogProb(lowerBound.Point));
                double ddlogf = 0;
                return Gaussian.FromDerivatives(lowerBound.Point, dlogf, ddlogf, false);
            }
            else throw new NotSupportedException();
        }

        public static Gaussian UpperBoundAverageConditional(Gaussian integral, [RequiredArgument] Gaussian upperBound, Func<double, double> func, ITruncatableDistribution<double> distribution)
        {
            CanGetLogProb<double> canGetLogProb = (CanGetLogProb<double>)distribution;
            if (upperBound.IsPointMass && integral.Precision == 0)
            {
                double dlogf = integral.MeanTimesPrecision * func(upperBound.Point) * Math.Exp(canGetLogProb.GetLogProb(upperBound.Point));
                double ddlogf = 0;
                return Gaussian.FromDerivatives(upperBound.Point, dlogf, ddlogf, false);
            }
            else throw new NotSupportedException();
        }
    }
}

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

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProbBetweenOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "ProbBetween<>")]
    [Quality(QualityBand.Preview)]
    public static class ProbBetweenOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProbBetweenOp"]/message_doc[@name="ProbBetweenAverageConditional(CanGetProbLessThan{double}, Gaussian, Gaussian)"]/*'/>
        public static Gaussian ProbBetweenAverageConditional(CanGetProbLessThan<double> canGetProbLessThan, [RequiredArgument] Gaussian lowerBound, [RequiredArgument] Gaussian upperBound)
        {
            if (lowerBound.IsPointMass && upperBound.IsPointMass) return Gaussian.PointMass(canGetProbLessThan.GetProbBetween(lowerBound.Point, upperBound.Point));
            else throw new NotSupportedException();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProbBetweenOp"]/message_doc[@name="LowerBoundAverageConditional(Gaussian, CanGetProbLessThan{double}, double)"]/*'/>
        public static Gaussian LowerBoundAverageConditional(Gaussian probBetween, CanGetProbLessThan<double> canGetProbLessThan, double lowerBound)
        {
            CanGetLogProb<double> canGetLogProb = (CanGetLogProb<double>)canGetProbLessThan;
            // factor is N(ProbBetween(left, right); m, v)
            // logp = -(m - ProbBetween(left, right))^2/(2v)
            // dlogp = (m - ProbBetween(left, right))/v * dProbBetween(left, right)
            // dProbBetween(left, right)/dleft = -p(left)
            // ddlogp = (m - ProbBetween(left, right))/v * ddProbBetween(left, right) - dProbBetween(left, right)/v * dProbBetween(left, right)
            // (m - ProbBetween(left, right))/v -> m/v
            double dlogp = -Math.Exp(canGetLogProb.GetLogProb(lowerBound)) * probBetween.MeanTimesPrecision;
            double ddlogp = 0; // approximation ignoring probBetween.Precision
            return Gaussian.FromDerivatives(lowerBound, dlogp, ddlogp, false);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProbBetweenOp"]/message_doc[@name="LowerBoundAverageConditional(Gaussian, CanGetProbLessThan{double}, Gaussian)"]/*'/>
        public static Gaussian LowerBoundAverageConditional(Gaussian probBetween, CanGetProbLessThan<double> canGetProbLessThan, [RequiredArgument] Gaussian lowerBound)
        {
            // factor is N(ProbBetween(left, right); m, v)
            if (lowerBound.IsPointMass)
            {
                return LowerBoundAverageConditional(probBetween, canGetProbLessThan, lowerBound.Point);
            }
            else throw new NotSupportedException();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProbBetweenOp"]/message_doc[@name="UpperBoundAverageConditional(Gaussian, CanGetProbLessThan{double}, Gaussian)"]/*'/>
        public static Gaussian UpperBoundAverageConditional(Gaussian probBetween, CanGetProbLessThan<double> canGetProbLessThan, [RequiredArgument] Gaussian upperBound)
        {
            CanGetLogProb<double> canGetLogProb = (CanGetLogProb<double>)canGetProbLessThan;
            // factor is N(ProbBetween(left, right); m, v)
            if (upperBound.IsPointMass)
            {
                // logp = -(m - ProbBetween(left, right))^2/(2v)
                // dlogp = (m - ProbBetween(left, right))/v * dProbBetween(left, right)
                // dProbBetween(left, right)/dright = p(right)
                // ddlogp = (m - ProbBetween(left, right))/v * ddProbBetween(left, right) - dProbBetween(left, right)/v * dProbBetween(left, right)
                double dlogp = Math.Exp(canGetLogProb.GetLogProb(upperBound.Point)) * probBetween.MeanTimesPrecision;
                double ddlogp = 0; // approximation ignoring probBetween.Precision
                return Gaussian.FromDerivatives(upperBound.Point, dlogp, ddlogp, false);
            }
            else throw new NotSupportedException();
        }
    }
}

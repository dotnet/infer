// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Factors.Attributes;

namespace Microsoft.ML.Probabilistic.Factors
{
    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UniformPlusMinusOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "UniformPlusMinus", typeof(double))]
    [Quality(QualityBand.Experimental)]
    public static class UniformPlusMinusOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UniformPlusMinusOp"]/message_doc[@name="LogEvidenceRatio(double, Pareto)"]/*'/>
        public static double LogEvidenceRatio(double sample, Pareto upperBound)
        {
            // factor is 1/(2*upperBound)
            // result is int_y^inf 1/(2*x)*s*L^s/x^(s+1) dx = 0.5*s*L^s/y^(s+1)/(s+1)
            // where y = max(sample,L)
            double x = System.Math.Abs(sample);
            double result = -MMath.Ln2 + System.Math.Log(upperBound.Shape / (upperBound.Shape + 1));
            if (upperBound.LowerBound < x)
            {
                result += upperBound.Shape * System.Math.Log(upperBound.LowerBound) - (upperBound.Shape + 1) * System.Math.Log(x);
            }
            else
            {
                result -= System.Math.Log(upperBound.LowerBound);
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UniformPlusMinusOp"]/message_doc[@name="UpperBoundAverageConditional(double)"]/*'/>
        public static Pareto UpperBoundAverageConditional(double sample)
        {
            return new Pareto(0, System.Math.Abs(sample));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UniformPlusMinusOp"]/message_doc[@name="UpperBoundAverageLogarithm(double)"]/*'/>
        public static Pareto UpperBoundAverageLogarithm(double sample)
        {
            return UpperBoundAverageConditional(sample);
        }
    }
}

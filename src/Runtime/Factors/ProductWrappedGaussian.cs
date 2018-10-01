// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Factors.Attributes;

namespace Microsoft.ML.Probabilistic.Factors
{
    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WrappedGaussianProductOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
    [FactorMethod(new string[] { "A", "Product", "B" }, typeof(Factor), "Ratio", typeof(double), typeof(double))]
    [Quality(QualityBand.Experimental)]
    public static class WrappedGaussianProductOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WrappedGaussianProductOp"]/message_doc[@name="AAverageConditional(WrappedGaussian, double, WrappedGaussian)"]/*'/>
        public static WrappedGaussian AAverageConditional([SkipIfUniform] WrappedGaussian Product, double B, WrappedGaussian result)
        {
            result.Period = Product.Period / B;
            result.Gaussian = GaussianProductOp.AAverageConditional(Product.Gaussian, B);
            result.Normalize();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WrappedGaussianProductOp"]/message_doc[@name="BAverageConditional(WrappedGaussian, double, WrappedGaussian)"]/*'/>
        public static WrappedGaussian BAverageConditional([SkipIfUniform] WrappedGaussian Product, double A, WrappedGaussian result)
        {
            return AAverageConditional(Product, A, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WrappedGaussianProductOp"]/message_doc[@name="AAverageConditional(double, double, WrappedGaussian)"]/*'/>
        public static WrappedGaussian AAverageConditional(double Product, double B, WrappedGaussian result)
        {
            if (B == 0)
            {
                if (Product != 0)
                    throw new AllZeroException();
                result.SetToUniform();
            }
            else
                result.Point = Product / B;
            result.Normalize();
            return result;
        }

        // ----------------------------------------------------------------------------------------------------------------------
        // VMP
        // ----------------------------------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WrappedGaussianProductOp"]/message_doc[@name="AverageLogFactor(WrappedGaussian)"]/*'/>
        [Skip]
        public static double AverageLogFactor(WrappedGaussian product)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WrappedGaussianProductOp"]/message_doc[@name="ProductAverageLogarithm(WrappedGaussian, double, WrappedGaussian)"]/*'/>
        public static WrappedGaussian ProductAverageLogarithm([SkipIfUniform] WrappedGaussian A, double B, WrappedGaussian result)
        {
            double m, v;
            A.Gaussian.GetMeanAndVariance(out m, out v);
            result.Gaussian.SetMeanAndVariance(B * m, B * B * v);
            double period = B * A.Period;
            if (period != result.Period)
            {
                double ratio = period / result.Period;
                double intRatio = System.Math.Round(ratio);
                if (System.Math.Abs(ratio - intRatio) > result.Period * 1e-4)
                    throw new ArgumentException("B*A.Period (" + period + ") is not a multiple of result.Period (" + result.Period + ")");
                // if period is a multiple of result.Period, then wrapping to result.Period is equivalent to first wrapping to period, then to result.Period.
            }
            result.Normalize();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WrappedGaussianProductOp"]/message_doc[@name="ProductAverageLogarithm(double, WrappedGaussian, WrappedGaussian)"]/*'/>
        public static WrappedGaussian ProductAverageLogarithm(double A, [SkipIfUniform] WrappedGaussian B, WrappedGaussian result)
        {
            return ProductAverageLogarithm(B, A, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WrappedGaussianProductOp"]/message_doc[@name="AAverageLogarithm(WrappedGaussian, double, WrappedGaussian)"]/*'/>
        public static WrappedGaussian AAverageLogarithm([SkipIfUniform] WrappedGaussian Product, double B, WrappedGaussian result)
        {
            if (Product.IsPointMass)
                return AAverageLogarithm(Product.Point, B, result);
            return AAverageConditional(Product, B, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WrappedGaussianProductOp"]/message_doc[@name="BAverageLogarithm(WrappedGaussian, double, WrappedGaussian)"]/*'/>
        public static WrappedGaussian BAverageLogarithm([SkipIfUniform] WrappedGaussian Product, double A, WrappedGaussian result)
        {
            return AAverageLogarithm(Product, A, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WrappedGaussianProductOp"]/message_doc[@name="AAverageLogarithm(double, double, WrappedGaussian)"]/*'/>
        public static WrappedGaussian AAverageLogarithm(double Product, double B, WrappedGaussian result)
        {
            return AAverageConditional(Product, B, result);
        }
    }
}

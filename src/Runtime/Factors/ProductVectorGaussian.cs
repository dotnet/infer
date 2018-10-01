// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductWishartOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Product", typeof(Vector), typeof(double))]
    //[FactorMethod(new[] { "Product", "b", "a" }, typeof(Factor), "Product", typeof(double), typeof(Vector))]
    [Quality(QualityBand.Experimental)]
    public static class ProductVectorGaussianOp_PointB
    {
        public static double LogAverageFactor([SkipIfUniform] VectorGaussian product, [SkipIfUniform] VectorGaussian a, Gaussian b)
        {
            if (!b.IsPointMass) throw new ArgumentException("b is not a point mass", nameof(b));
            double bPoint = b.Point;
            Vector productMean = Vector.Zero(product.Dimension);
            PositiveDefiniteMatrix productVariance = new PositiveDefiniteMatrix(product.Dimension, product.Dimension);
            product.GetMeanAndVariance(productMean, productVariance);
            Vector aMean = Vector.Zero(product.Dimension);
            PositiveDefiniteMatrix aVariance = new PositiveDefiniteMatrix(product.Dimension, product.Dimension);
            a.GetMeanAndVariance(aMean, aVariance);
            PositiveDefiniteMatrix variance = new PositiveDefiniteMatrix(product.Dimension, product.Dimension);
            variance.SetToSum(1, productVariance, bPoint * bPoint, aVariance);
            // variance = productVariance + aVariance * b^2
            return VectorGaussian.GetLogProb(productMean, aMean * bPoint, variance);
        }

        public static VectorGaussian ProductAverageConditional([SkipIfUniform] VectorGaussian a, Gaussian b, VectorGaussian result)
        {
            if (!b.IsPointMass) throw new ArgumentException("b is not a point mass", nameof(b));
            double bPoint = b.Point;
            if (a.IsPointMass)
            {
                result.Point = result.Point;
                result.Point.SetToProduct(a.Point, bPoint);
                return result;
            }
            return AAverageConditional(a, Gaussian.PointMass(1.0 / bPoint), result);
        }

        public static VectorGaussian AAverageConditional([SkipIfUniform] VectorGaussian product, Gaussian b, VectorGaussian result)
        {
            if (!b.IsPointMass) throw new ArgumentException("b is not a point mass", nameof(b));
            double bPoint = b.Point;
            if (bPoint == 0)
            {
                result.SetToUniform();
            }
            else if (double.IsPositiveInfinity(bPoint))
            {
                result.Point = result.Point;
                result.Point.SetAllElementsTo(0.0);
            }
            else if (product.IsPointMass)
            {
                result.Point = result.Point;
                result.Point.SetToProduct(product.Point, 1.0 / bPoint);
            }
            else
            {
                // mean = xMean/b
                // variance = xVariance/b^2
                // precision = xPrecision*b^2
                // meanTimesPrecision = xMeanTimesPrecision*b
                result.MeanTimesPrecision.SetToProduct(product.MeanTimesPrecision, bPoint);
                result.Precision.SetToProduct(product.Precision, bPoint * bPoint);
            }
            return result;
        }

        public static Gaussian BAverageConditional([SkipIfUniform] VectorGaussian product, [SkipIfUniform] VectorGaussian a, Gaussian b)
        {
            if (!b.IsPointMass) throw new ArgumentException("b is not a point mass", nameof(b));
            double bPoint = b.Point;
            Vector productMean = Vector.Zero(product.Dimension);
            PositiveDefiniteMatrix productVariance = new PositiveDefiniteMatrix(product.Dimension, product.Dimension);
            product.GetMeanAndVariance(productMean, productVariance);
            Vector aMean = Vector.Zero(product.Dimension);
            PositiveDefiniteMatrix aVariance = new PositiveDefiniteMatrix(product.Dimension, product.Dimension);
            a.GetMeanAndVariance(aMean, aVariance);
            PositiveDefiniteMatrix variance = new PositiveDefiniteMatrix(product.Dimension, product.Dimension);
            variance.SetToSum(1, productVariance, bPoint * bPoint, aVariance);
            // variance = productVariance + aVariance * b^2
            PositiveDefiniteMatrix precision = variance.Inverse();
            Vector diff = aMean * bPoint;
            diff.SetToDifference(productMean, diff);
            // diff = productMean - aMean * b
            var precisionDiff = precision * diff;
            double dlogf = aMean.Inner(precisionDiff) + bPoint * (precisionDiff.Inner(aVariance * precisionDiff) - Matrix.TraceOfProduct(precision, aVariance));
            double ddlogf = -1;
            return Gaussian.FromDerivatives(bPoint, dlogf, ddlogf, GaussianProductOp.ForceProper);
        }
    }
}

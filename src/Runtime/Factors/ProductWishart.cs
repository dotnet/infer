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
    [FactorMethod(typeof(Factor), "Product", typeof(PositiveDefiniteMatrix), typeof(double))]
    [FactorMethod(new[] { "Product", "b", "a" }, typeof(Factor), "Product", typeof(double), typeof(PositiveDefiniteMatrix))]
    [Quality(QualityBand.Experimental)]
    public static class ProductWishartOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductWishartOp"]/message_doc[@name="LogEvidenceRatio(Wishart, Wishart, double)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Wishart product, Wishart a, double b)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductWishartOp"]/message_doc[@name="LogEvidenceRatio(Wishart, PositiveDefiniteMatrix, Gamma)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Wishart product, PositiveDefiniteMatrix a, Gamma b)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductWishartOp"]/message_doc[@name="LogEvidenceRatio(PositiveDefiniteMatrix, Wishart, double)"]/*'/>
        public static double LogEvidenceRatio(PositiveDefiniteMatrix product, Wishart a, double b)
        {
            throw new NotImplementedException();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductWishartOp"]/message_doc[@name="ProductAverageConditional(PositiveDefiniteMatrix, double, Wishart)"]/*'/>
        public static Wishart ProductAverageConditional(PositiveDefiniteMatrix A, double B, Wishart result)
        {
            result.Point = A;
            result.Rate.Scale(B);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductWishartOp"]/message_doc[@name="ProductAverageConditional(Wishart, double, Wishart)"]/*'/>
        public static Wishart ProductAverageConditional([SkipIfUniform] Wishart A, double B, Wishart result)
        {
            if (A.IsPointMass) return ProductAverageConditional(A.Point, B, result);
            result.SetTo(A);
            result.Rate.Scale(1/B);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductWishartOp"]/message_doc[@name="ProductAverageConditional(Wishart, PositiveDefiniteMatrix, Gamma, Gamma, Wishart)"]/*'/>
        public static Wishart ProductAverageConditional(Wishart Product, PositiveDefiniteMatrix A, [SkipIfUniform] Gamma B, Gamma to_B, Wishart result)
        {
            if (B.IsPointMass) return ProductAverageConditional(A, B.Point, result);
            Gamma Bpost = B * to_B;
            // E[x] = a*E[b]
            // E[logdet(x)] = logdet(a) + d*E[log(b)]
            PositiveDefiniteMatrix m = new PositiveDefiniteMatrix(A.Rows, A.Cols);
            m.SetTo(A);
            m.Scale(Bpost.GetMean());
            double meanLogDet = A.Rows * Bpost.GetMeanLog() + A.LogDeterminant();
            if (m.LogDeterminant() < meanLogDet) throw new MatrixSingularException(m);
            Wishart.FromMeanAndMeanLogDeterminant(m, meanLogDet, result);
            result.SetToRatio(result, Product);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductWishartOp"]/message_doc[@name="BAverageConditional(Wishart, PositiveDefiniteMatrix)"]/*'/>
        public static Gamma BAverageConditional([SkipIfUniform] Wishart Product, PositiveDefiniteMatrix A)
        {
            if (Product.IsPointMass) return BAverageConditional(Product.Point, A);
            // det(ab)^(shape-(d+1)/2) exp(-tr(rate*(ab))) =propto b^((shape-(d+1)/2)*d) exp(-b*tr(rate*a))
            return Gamma.FromShapeAndRate(Product.Dimension * (Product.Shape - (Product.Dimension + 1) * 0.5) + 1, Matrix.TraceOfProduct(Product.Rate, A));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductWishartOp"]/message_doc[@name="BAverageConditional(PositiveDefiniteMatrix, PositiveDefiniteMatrix)"]/*'/>
        public static Gamma BAverageConditional(PositiveDefiniteMatrix Product, PositiveDefiniteMatrix A)
        {
            if (Product.Count == 0) return Gamma.Uniform();
            bool allZeroA = true;
            double ratio = 0;
            for (int i = 0; i < Product.Count; i++)
            {
                if (A[i] != 0)
                {
                    ratio = Product[i]/A[i];
                    allZeroA = false;
                }
            }
            if (allZeroA) return Gamma.Uniform();
            for (int i = 0; i < Product.Count; i++)
            {
                if (Math.Abs(Product[i] - A[i]*ratio) > 1e-15) throw new ConstraintViolatedException("Product is not a multiple of B");
            }
            return Gamma.PointMass(ratio);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductWishartOp"]/message_doc[@name="AAverageConditional(Wishart, double, Wishart)"]/*'/>
        public static Wishart AAverageConditional([SkipIfUniform] Wishart Product, double B, Wishart result)
        {
            result.SetTo(Product);
            result.Rate.Scale(B);
            return result;
        }

        //- VMP ----------------------------------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductWishartOp"]/message_doc[@name="AverageLogFactor(Wishart)"]/*'/>
        [Skip]
        public static double AverageLogFactor(Wishart product)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductWishartOp"]/message_doc[@name="ProductAverageLogarithm(Wishart, Gamma, Wishart)"]/*'/>
        public static Wishart ProductAverageLogarithm([SkipIfUniform] Wishart A, [SkipIfUniform] Gamma B, Wishart result)
        {
            if (B.IsPointMass) return ProductAverageLogarithm(A, B.Point, result);
            // E[x] = E[a]*E[b]
            // E[log(x)] = E[log(a)]+E[log(b)]
            PositiveDefiniteMatrix m = new PositiveDefiniteMatrix(A.Dimension, A.Dimension);
            A.GetMean(m);
            m.Scale(B.GetMean());
            double meanLogDet = A.Dimension*B.GetMeanLog() + A.GetMeanLogDeterminant();
            if (m.LogDeterminant() < meanLogDet) throw new MatrixSingularException(m);
            return Wishart.FromMeanAndMeanLogDeterminant(m, meanLogDet, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductWishartOp"]/message_doc[@name="ProductAverageLogarithm(Wishart, double, Wishart)"]/*'/>
        public static Wishart ProductAverageLogarithm([SkipIfUniform] Wishart A, double B, Wishart result)
        {
            return ProductAverageConditional(A, B, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductWishartOp"]/message_doc[@name="BAverageLogarithm(Wishart, Wishart)"]/*'/>
        public static Gamma BAverageLogarithm([SkipIfUniform] Wishart Product, [Proper] Wishart A)
        {
            if (A.IsPointMass) return BAverageLogarithm(Product, A.Point);
            if (Product.IsPointMass) return BAverageLogarithm(Product.Point, A);
            return BAverageConditional(Product, A.GetMean());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductWishartOp"]/message_doc[@name="BAverageLogarithm(Wishart, PositiveDefiniteMatrix)"]/*'/>
        public static Gamma BAverageLogarithm([SkipIfUniform] Wishart Product, [Proper] PositiveDefiniteMatrix A)
        {
            return BAverageConditional(Product, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductWishartOp"]/message_doc[@name="BAverageLogarithm(PositiveDefiniteMatrix, PositiveDefiniteMatrix)"]/*'/>
        public static Gamma BAverageLogarithm(PositiveDefiniteMatrix Product, PositiveDefiniteMatrix A)
        {
            return BAverageConditional(Product, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductWishartOp"]/message_doc[@name="AAverageLogarithm(Wishart, Gamma, Wishart)"]/*'/>
        public static Wishart AAverageLogarithm([SkipIfUniform] Wishart Product, [Proper] Gamma B, Wishart result)
        {
            if (B.IsPointMass) return AAverageLogarithm(Product, B.Point, result);
            if (Product.IsPointMass) return AAverageLogarithm(Product.Point, B, result);
            // (ab)^(shape-1) exp(-rate*(ab))
            result.Shape = Product.Shape;
            result.Rate.SetToProduct(Product.Rate, B.GetMean());
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductWishartOp"]/message_doc[@name="BAverageLogarithm(PositiveDefiniteMatrix, Wishart)"]/*'/>
        [NotSupported(GaussianProductVmpOp.NotSupportedMessage)]
        public static Gamma BAverageLogarithm(PositiveDefiniteMatrix Product, [Proper] Wishart A)
        {
            throw new NotSupportedException(GaussianProductVmpOp.NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductWishartOp"]/message_doc[@name="AAverageLogarithm(PositiveDefiniteMatrix, Gamma, Wishart)"]/*'/>
        [NotSupported(GaussianProductVmpOp.NotSupportedMessage)]
        public static Wishart AAverageLogarithm(PositiveDefiniteMatrix Product, [Proper] Gamma B, Wishart result)
        {
            throw new NotSupportedException(GaussianProductVmpOp.NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductWishartOp"]/message_doc[@name="AAverageLogarithm(Wishart, double, Wishart)"]/*'/>
        public static Wishart AAverageLogarithm([SkipIfUniform] Wishart Product, double B, Wishart result)
        {
            if (Product.IsPointMass) return AAverageLogarithm(Product.Point, B, result);
            return AAverageConditional(Product, B, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductWishartOp"]/message_doc[@name="AAverageLogarithm(PositiveDefiniteMatrix, double, Wishart)"]/*'/>
        public static Wishart AAverageLogarithm(PositiveDefiniteMatrix Product, double B, Wishart result)
        {
            result.Point = Product;
            result.Point.Scale(1 / B);
            return result;
        }
    }

#if false
    [FactorMethod(typeof(Factor), "Product", typeof(PositiveDefiniteMatrix), typeof(double))]
    [FactorMethod(new string[] { "Product", "b", "a" }, typeof(Factor), "Product", typeof(double), typeof(PositiveDefiniteMatrix))]
    [Buffers("Q")]
    [Quality(QualityBand.Experimental)]
    public static class ProductWishartOp_Laplace
    {
        // derivatives of the factor marginalized over Product and A, multiplied by powers of b
        private static double[] xdlogfs(double r, double shape, Wishart Product, Wishart A)
        {
            if (Product.IsPointMass || A.IsPointMass)
            {
                throw new NotImplementedException();
            }
            else
            {
                // logf = dc \log(b) - (c+a_a)\log\det{\B_a + b\B_x}
                double d = Product.Dimension;
                double c = Product.Shape - (d + 1) / 2;
                double dc = d * c;
                double dlogf = dc - shape2 * p;
                double ddlogf = -dc + shape2 * p2;
                double dddlogf = 2 * dc - 2 * shape2 * p * p2;
                double d4logf = -6 * dc + 6 * shape2 * p2 * p2;
                return new double[] { dlogf, ddlogf, dddlogf, d4logf };
            }
        }

        public static Wishart ProductAverageConditional(Wishart Product, Wishart A, Gamma B, Wishart result)
        {
        }
    }
#endif
}

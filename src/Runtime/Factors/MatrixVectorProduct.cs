// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Product", typeof(Matrix), typeof(Vector))]
    [FactorMethod(typeof(Factor), "Product", typeof(double[,]), typeof(Vector))]
    [Buffers("BMean", "BVariance")]
    [Quality(QualityBand.Stable)]
    public static class MatrixVectorProductOp
    {
        public static bool UseAccurateMethod = true;

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="LogAverageFactor(Vector, Matrix, Vector)"]/*'/>
        public static double LogAverageFactor(Vector product, Matrix A, Vector B)
        {
            return product.Equals(Factor.Product(A, B)) ? 0.0 : Double.NegativeInfinity;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="LogEvidenceRatio(Vector, Matrix, Vector)"]/*'/>
        public static double LogEvidenceRatio(Vector product, Matrix A, Vector B)
        {
            return LogAverageFactor(product, A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="AverageLogFactor(Vector, Matrix, Vector)"]/*'/>
        public static double AverageLogFactor(Vector product, Matrix A, Vector B)
        {
            return LogAverageFactor(product, A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="BVarianceInit(VectorGaussian)"]/*'/>
        [Skip]
        public static PositiveDefiniteMatrix BVarianceInit([IgnoreDependency] VectorGaussian B)
        {
            return new PositiveDefiniteMatrix(B.Dimension, B.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="BVariance(VectorGaussian, PositiveDefiniteMatrix)"]/*'/>
        [Fresh]
        public static PositiveDefiniteMatrix BVariance([Proper] VectorGaussian B, PositiveDefiniteMatrix result)
        {
            return B.GetVariance(result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="BMeanInit(VectorGaussian)"]/*'/>
        [Skip]
        public static Vector BMeanInit([IgnoreDependency] VectorGaussian B)
        {
            return Vector.Zero(B.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="BMean(VectorGaussian, PositiveDefiniteMatrix, Vector)"]/*'/>
        [Fresh]
        public static Vector BMean([Proper] VectorGaussian B, PositiveDefiniteMatrix BVariance, Vector result)
        {
            return B.GetMean(result, BVariance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="LogAverageFactor(VectorGaussian, VectorGaussian)"]/*'/>
        public static double LogAverageFactor(VectorGaussian product, [Fresh] VectorGaussian to_product)
        {
            return to_product.GetLogAverageOf(product);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="LogAverageFactor(Vector, Matrix, VectorGaussian, Vector, PositiveDefiniteMatrix)"]/*'/>
        public static double LogAverageFactor(Vector product, Matrix A, VectorGaussian B, Vector BMean, PositiveDefiniteMatrix BVariance)
        {
            VectorGaussian toProduct = ProductAverageConditional(A, BMean, BVariance, new VectorGaussian(A.Rows));
            return toProduct.GetLogProb(product);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="LogEvidenceRatio(VectorGaussian, Matrix, VectorGaussian)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(VectorGaussian product, Matrix A, VectorGaussian B)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="AverageLogFactor(VectorGaussian, Matrix, VectorGaussian)"]/*'/>
        [Skip]
        public static double AverageLogFactor(VectorGaussian product, Matrix A, VectorGaussian B)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="LogEvidenceRatio(Vector, Matrix, VectorGaussian, Vector, PositiveDefiniteMatrix)"]/*'/>
        public static double LogEvidenceRatio(Vector product, Matrix A, VectorGaussian B, Vector BMean, PositiveDefiniteMatrix BVariance)
        {
            return LogAverageFactor(product, A, B, BMean, BVariance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="AverageLogFactor(Vector, Matrix, VectorGaussian, Vector, PositiveDefiniteMatrix)"]/*'/>
        public static double AverageLogFactor(Vector product, Matrix A, VectorGaussian B, Vector BMean, PositiveDefiniteMatrix BVariance)
        {
            return LogAverageFactor(product, A, B, BMean, BVariance);
        }

        private static void GetProductMoments(Matrix A, Vector BMean, PositiveDefiniteMatrix BVariance, Vector mean, PositiveDefiniteMatrix variance)
        {
            // P.mean = A*B.mean
            // P.var = A*B.var*A'
            mean.SetToProduct(A, BMean);
            if (UseAccurateMethod)
            {
                int dim = BVariance.Rows;
                LowerTriangularMatrix cholesky = new LowerTriangularMatrix(dim, dim);
                cholesky.SetToCholesky(BVariance);
                Matrix AL = A * cholesky;
                variance.SetToOuter(AL);
            }
            else
            {
                variance.SetToProduct(A, (A * BVariance).Transpose());
                variance.Symmetrize();
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="ProductAverageConditional(Matrix, Vector, PositiveDefiniteMatrix, VectorGaussian)"]/*'/>
        public static VectorGaussian ProductAverageConditional(Matrix A, Vector BMean, PositiveDefiniteMatrix BVariance, VectorGaussian result)
        {
            Vector mean = Vector.Zero(result.Dimension);
            PositiveDefiniteMatrix variance = result.Precision;
            GetProductMoments(A, BMean, BVariance, mean, variance);
            result.SetMeanAndVariance(mean, variance);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="ProductAverageConditional(Matrix, Vector, PositiveDefiniteMatrix, VectorGaussianMoments)"]/*'/>
        public static VectorGaussianMoments ProductAverageConditional(Matrix A, Vector BMean, PositiveDefiniteMatrix BVariance, VectorGaussianMoments result)
        {
            GetProductMoments(A, BMean, BVariance, result.Mean, result.Variance);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="ProductAverageConditional(double[,], Vector, PositiveDefiniteMatrix, VectorGaussian)"]/*'/>
        public static VectorGaussian ProductAverageConditional(double[,] A, Vector BMean, PositiveDefiniteMatrix BVariance, VectorGaussian result)
        {
            return ProductAverageConditional(new Matrix(A), BMean, BVariance, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="ProductAverageConditional(double[,], Vector, PositiveDefiniteMatrix, VectorGaussianMoments)"]/*'/>
        public static VectorGaussianMoments ProductAverageConditional(double[,] A, Vector BMean, PositiveDefiniteMatrix BVariance, VectorGaussianMoments result)
        {
            return ProductAverageConditional(new Matrix(A), BMean, BVariance, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="ProductAverageConditional(double[,], Vector, PositiveDefiniteMatrix, VectorGaussian)"]/*'/>
        public static VectorGaussian ProductAverageConditional(DistributionArray2D<Gaussian, double> A, Vector BMean, PositiveDefiniteMatrix BVariance, VectorGaussian result)
        {
            if (!A.IsPointMass) throw new ArgumentException("A is not a point mass");
            return ProductAverageConditional(new Matrix(A.Point), BMean, BVariance, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="ProductAverageConditional(double[,], Vector, PositiveDefiniteMatrix, VectorGaussianMoments)"]/*'/>
        public static VectorGaussianMoments ProductAverageConditional(DistributionArray2D<Gaussian, double> A, Vector BMean, PositiveDefiniteMatrix BVariance, VectorGaussianMoments result)
        {
            if (!A.IsPointMass) throw new ArgumentException("A is not a point mass");
            return ProductAverageConditional(new Matrix(A.Point), BMean, BVariance, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="ProductAverageConditionalInit(Matrix)"]/*'/>
        [Skip]
        public static VectorGaussian ProductAverageConditionalInit([IgnoreDependency] Matrix A)
        {
            return new VectorGaussian(A.Rows);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="ProductAverageLogarithmInit(Matrix)"]/*'/>
        [Skip]
        public static VectorGaussian ProductAverageLogarithmInit([IgnoreDependency] Matrix A)
        {
            return new VectorGaussian(A.Rows);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="ProductAverageLogarithm(Matrix, Vector, PositiveDefiniteMatrix, VectorGaussian)"]/*'/>
        public static VectorGaussian ProductAverageLogarithm(Matrix A, Vector BMean, PositiveDefiniteMatrix BVariance, VectorGaussian result)
        {
            return ProductAverageConditional(A, BMean, BVariance, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="ProductAverageLogarithm(Matrix, Vector, PositiveDefiniteMatrix, VectorGaussianMoments)"]/*'/>
        public static VectorGaussianMoments ProductAverageLogarithm(Matrix A, Vector BMean, PositiveDefiniteMatrix BVariance, VectorGaussianMoments result)
        {
            return ProductAverageConditional(A, BMean, BVariance, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="ProductAverageLogarithm(double[,], Vector, PositiveDefiniteMatrix, VectorGaussian)"]/*'/>
        public static VectorGaussian ProductAverageLogarithm(double[,] A, Vector BMean, PositiveDefiniteMatrix BVariance, VectorGaussian result)
        {
            return ProductAverageConditional(A, BMean, BVariance, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="ProductAverageLogarithm(double[,], Vector, PositiveDefiniteMatrix, VectorGaussianMoments)"]/*'/>
        public static VectorGaussianMoments ProductAverageLogarithm(double[,] A, Vector BMean, PositiveDefiniteMatrix BVariance, VectorGaussianMoments result)
        {
            return ProductAverageConditional(A, BMean, BVariance, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="BAverageConditional(VectorGaussian, Matrix, VectorGaussian)"]/*'/>
        public static VectorGaussian BAverageConditional([SkipIfUniform] VectorGaussian product, Matrix A, VectorGaussian result)
        {
            if (product.IsPointMass)
                return BAverageConditional(product.Point, A, result);
            //   (p.mean - A*B)'*p.prec*(p.mean - A*B)
            // = B'*(A'*p.prec*A)*B - B'*A'*p.prec*p.mean - ...
            // B.prec = A'*p.prec*A
            // B.precTimesMean = A'*p.precTimesMean
            if (UseAccurateMethod)
            {
                // this method is slower but more numerically accurate
                // L*L' = p.prec
                int dim = product.Precision.Cols;
                LowerTriangularMatrix L = new LowerTriangularMatrix(dim, dim);
                L.SetToCholesky(product.Precision);                
                Matrix At = A.Transpose();
                Matrix temp = At * L;
                result.Precision.SetToOuter(temp);
            }
            else
            {
                Matrix temp = (product.Precision * A).Transpose();
                result.Precision.SetToProduct(temp, A);
                result.Precision.Symmetrize();
            }
            result.MeanTimesPrecision.SetToProduct(product.MeanTimesPrecision, A);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="BAverageConditional(VectorGaussianMoments, Matrix, VectorGaussian)"]/*'/>
        public static VectorGaussian BAverageConditional([SkipIfUniform] VectorGaussianMoments product, Matrix A, VectorGaussian result)
        {
            if (product.IsPointMass)
                return BAverageConditional(product.Point, A, result);
            VectorGaussian productNatural = new VectorGaussian(product.Mean, product.Variance);
            return BAverageConditional(productNatural, A, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="BAverageConditional(VectorGaussian, double[,], VectorGaussian)"]/*'/>
        public static VectorGaussian BAverageConditional([SkipIfUniform] VectorGaussian product, double[,] A, VectorGaussian result)
        {
            return BAverageConditional(product, new Matrix(A), result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="BAverageConditional(VectorGaussianMoments, double[,], VectorGaussian)"]/*'/>
        public static VectorGaussian BAverageConditional([SkipIfUniform] VectorGaussianMoments product, double[,] A, VectorGaussian result)
        {
            return BAverageConditional(product, new Matrix(A), result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="BAverageConditional(VectorGaussian, DistributionArray2D{Gaussian,double}, VectorGaussian)"]/*'/>
        public static VectorGaussian BAverageConditional([SkipIfUniform] VectorGaussian product, DistributionArray2D<Gaussian,double> A, VectorGaussian result)
        {
            if (!A.IsPointMass) throw new ArgumentException("A is not a point mass");
            return BAverageConditional(product, new Matrix(A.Point), result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="BAverageConditional(VectorGaussianMoments, DistributionArray2D{Gaussian,double}, VectorGaussian)"]/*'/>
        public static VectorGaussian BAverageConditional([SkipIfUniform] VectorGaussianMoments product, DistributionArray2D<Gaussian, double> A, VectorGaussian result)
        {
            if (!A.IsPointMass) throw new ArgumentException("A is not a point mass");
            return BAverageConditional(product, new Matrix(A.Point), result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="BAverageLogarithm(VectorGaussian, Matrix, VectorGaussian)"]/*'/>
        public static VectorGaussian BAverageLogarithm([SkipIfUniform] VectorGaussian product, Matrix A, VectorGaussian result)
        {
            return BAverageConditional(product, A, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="BAverageLogarithm(VectorGaussianMoments, Matrix, VectorGaussian)"]/*'/>
        public static VectorGaussian BAverageLogarithm([SkipIfUniform] VectorGaussianMoments product, Matrix A, VectorGaussian result)
        {
            return BAverageConditional(product, A, result);
        }

        private const string LowRankNotSupportedMessage = "A matrix-vector product with fixed output is not yet implemented.";

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="BAverageConditional(Vector, Matrix, VectorGaussian)"]/*'/>
        [NotSupported(MatrixVectorProductOp.LowRankNotSupportedMessage)]
        public static VectorGaussian BAverageConditional(Vector product, Matrix A, VectorGaussian result)
        {
            throw new NotSupportedException(MatrixVectorProductOp.LowRankNotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="BAverageLogarithm(Vector, Matrix, VectorGaussian)"]/*'/>
        [NotSupported(MatrixVectorProductOp.LowRankNotSupportedMessage)]
        public static VectorGaussian BAverageLogarithm(Vector product, Matrix A, VectorGaussian result)
        {
            throw new NotSupportedException(MatrixVectorProductOp.LowRankNotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MatrixVectorProductOp"]/message_doc[@name="AAverageConditional(VectorGaussian, DistributionArray2D{Gaussian, double}, Vector, PositiveDefiniteMatrix, DistributionStructArray2D{Gaussian, double})"]/*'/>
        public static DistributionStructArray2D<Gaussian, double> AAverageConditional([SkipIfUniform] VectorGaussian product, DistributionArray2D<Gaussian, double> A, Vector BMean, PositiveDefiniteMatrix BVariance, DistributionStructArray2D<Gaussian, double> result)
        {
            if (product.IsUniform())
            {
                result.SetToUniform();
                return result;
            }
            if (!A.IsPointMass) throw new ArgumentException("A is not a point mass");
            // logZ = log N(mProduct; A*BMean, vProduct + A*BVariance*A')
            //      = -0.5 (mProduct - A*BMean)' inv(vProduct + A*BVariance*A') (mProduct - A*BMean) - 0.5 logdet(vProduct + A*BVariance*A')
            //      = -0.5 (mProduct - A*BMean)' pPrec inv(pProduct + pProduct*A*BVariance*A'*pProduct) pProduct (mProduct - A*BMean) 
            //        - 0.5 logdet(pProduct + pProduct*A*BVariance*A'*pProduct) + logdet(pProduct)
            // dlogZ   = 0.5 (dA*BMean)' pProduct inv(pProduct + pProduct*A*BVariance*A'*pProduct) pProduct (mProduct - A*BMean) 
            //         +0.5 (mProduct - A*BMean)' pProduct inv(pProduct + pProduct*A*BVariance*A'*pProduct) pProduct (dA*BMean) 
            //         +0.5 (mProduct - A*BMean)' pProduct inv(pProduct + pProduct*A*BVariance*A'*pProduct) (pProduct*dA*BVariance*A'*pProduct + pProduct*A*BVariance*dA'*pProduct) inv(pProduct + pProduct*A*BVariance*A'*pProduct) pProduct (mProduct - A*BMean) 
            //         - 0.5 tr(inv(pProduct + pProduct*A*BVariance*A'*pProduct) (pProduct*dA*BVariance*A'*pProduct + pProduct*A*BVariance*dA'*pProduct))
            // dlogZ/dA = pProduct inv(pProduct + pProduct*A*BVariance*A'*pProduct) pProduct (mProduct - A*BMean) BMean'
            //          + pProduct inv(pProduct + pProduct*A*BVariance*A'*pProduct) pProduct (mProduct - A*BMean) (mProduct - A*BMean)' pProduct inv(pProduct + pProduct*A*BVariance*A'*pProduct) pProduct*A*BVariance
            //          - pProduct inv(pProduct + pProduct*A*BVariance*A'*pProduct) pProduct A*BVariance
            var Amatrix = new Matrix(A.Point);
            var pProductA = product.Precision * Amatrix;
            var pProductABV = pProductA * BVariance;
            PositiveDefiniteMatrix prec = new PositiveDefiniteMatrix(product.Dimension, product.Dimension);
            prec.SetToSum(product.Precision, pProductABV * pProductA.Transpose());
            // pProductA is now free
            for (int i = 0; i < prec.Rows; i++)
            {
                if (prec[i, i] == 0) prec[i, i] = 1;
            }
            var v = prec.Inverse();
            var ABM = Amatrix * BMean;
            var pProductABM = product.Precision * ABM;
            var diff = pProductABM;
            diff.SetToDifference(product.MeanTimesPrecision, pProductABM);
            // ABM is now free
            var pProductV = product.Precision * v;
            var pProductVdiff = ABM;
            pProductVdiff.SetToProduct(pProductV, diff);
            var Vdiff = v * diff;
            pProductV.Scale(-1);
            pProductV.SetToSumWithOuter(pProductV, 1, pProductVdiff, Vdiff);
            Matrix dlogZ = pProductA;
            dlogZ.SetToProduct(pProductV, pProductABV);
            dlogZ.SetToSumWithOuter(dlogZ, 1, pProductVdiff, BMean);
            int rows = A.GetLength(0);
            int cols = A.GetLength(1);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    double dlogp = dlogZ[i,j];
                    // for now, we don't compute the second derivative.
                    double ddlogp = -1;
                    result[i, j] = Gaussian.FromDerivatives(A[i, j].Point, dlogp, ddlogp, false);
                }
            }
            return result;
        }
    }
}

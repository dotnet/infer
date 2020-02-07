// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOp"]/doc/*'/>
    [FactorMethod(typeof(Vector), "InnerProduct", Default = true)]
    [Buffers("AVariance", "BVariance", "AMean", "BMean")]
    [Quality(QualityBand.Mature)]
    public class InnerProductOp : InnerProductOpBase
    {
        private const string BothRandomNotSupportedMessage =
            "An InnerProduct factor between two VectorGaussian variables is not yet implemented for Expectation Propagation.  Try using Variational Message Passing.";

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOp"]/message_doc[@name="InnerProductAverageConditional(VectorGaussian, VectorGaussian)"]/*'/>
        [NotSupported(InnerProductOp.BothRandomNotSupportedMessage)]
        public static Gaussian InnerProductAverageConditional([SkipIfUniform] VectorGaussian A, [SkipIfUniform] VectorGaussian B)
        {
            throw new NotSupportedException(InnerProductOp.BothRandomNotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOp"]/message_doc[@name="AAverageConditional(Gaussian, VectorGaussian, VectorGaussian, VectorGaussian)"]/*'/>
        [NotSupported(InnerProductOp.BothRandomNotSupportedMessage)]
        public static VectorGaussian AAverageConditional(Gaussian innerProduct, VectorGaussian A, [SkipIfUniform] VectorGaussian B, VectorGaussian result)
        {
            throw new NotSupportedException(InnerProductOp.BothRandomNotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOp"]/message_doc[@name="BAverageConditional(Gaussian, VectorGaussian, VectorGaussian, VectorGaussian)"]/*'/>
        [NotSupported(InnerProductOp.BothRandomNotSupportedMessage)]
        public static VectorGaussian BAverageConditional(Gaussian innerProduct, [SkipIfUniform] VectorGaussian A, VectorGaussian B, VectorGaussian result)
        {
            throw new NotSupportedException(InnerProductOp.BothRandomNotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOp"]/message_doc[@name="LogAverageFactor(Gaussian, VectorGaussian, VectorGaussian)"]/*'/>
        [NotSupported(InnerProductOp.BothRandomNotSupportedMessage)]
        public static double LogAverageFactor(Gaussian innerProduct, VectorGaussian A, VectorGaussian B)
        {
            throw new NotSupportedException(InnerProductOp.BothRandomNotSupportedMessage);
            //Gaussian to_innerProduct = InnerProductAverageConditional(A, B);
            //return to_innerProduct.GetLogAverageOf(innerProduct);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOp"]/message_doc[@name="LogAverageFactor(double, VectorGaussian, VectorGaussian)"]/*'/>
        [NotSupported(InnerProductOp.BothRandomNotSupportedMessage)]
        public static double LogAverageFactor(double innerProduct, VectorGaussian A, VectorGaussian B)
        {
            throw new NotSupportedException(InnerProductOp.BothRandomNotSupportedMessage);
            //Gaussian to_innerProduct = InnerProductAverageConditional(A, B);
            //return to_innerProduct.GetLogProb(innerProduct);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOp"]/message_doc[@name="LogEvidenceRatio(Gaussian, VectorGaussian, VectorGaussian)"]/*'/>
        [NotSupported(InnerProductOp.BothRandomNotSupportedMessage)]
        public static double LogEvidenceRatio(Gaussian innerProduct, VectorGaussian A, VectorGaussian B)
        {
            throw new NotSupportedException(InnerProductOp.BothRandomNotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOp"]/message_doc[@name="LogEvidenceRatio(double, VectorGaussian, VectorGaussian)"]/*'/>
        [NotSupported(InnerProductOp.BothRandomNotSupportedMessage)]
        public static double LogEvidenceRatio(double innerProduct, VectorGaussian A, VectorGaussian B)
        {
            throw new NotSupportedException(InnerProductOp.BothRandomNotSupportedMessage);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/doc/*'/>
    public class InnerProductOpBase
    {
        private const string NotSupportedMessage = "Variational Message Passing does not support an InnerProduct factor with fixed output.";

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="AAverageLogarithm(double, VectorGaussian, VectorGaussian)"]/*'/>
        [NotSupported(InnerProductOp.NotSupportedMessage)]
        public static VectorGaussian AAverageLogarithm(double innerProduct, [SkipIfUniform] VectorGaussian B, VectorGaussian result)
        {
            throw new NotSupportedException(InnerProductOp.NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="BAverageLogarithm(double, VectorGaussian, VectorGaussian)"]/*'/>
        [NotSupported(InnerProductOp.NotSupportedMessage)]
        public static VectorGaussian BAverageLogarithm(double innerProduct, [SkipIfUniform] VectorGaussian A, VectorGaussian result)
        {
            return AAverageLogarithm(innerProduct, A, result);
        }

        private const string LowRankNotSupportedMessage = "A InnerProduct factor with fixed output is not yet implemented.";

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="AAverageConditional(double, Vector, VectorGaussian)"]/*'/>
        [NotSupported(InnerProductOp.LowRankNotSupportedMessage)]
        public static VectorGaussian AAverageConditional(double innerProduct, Vector B, VectorGaussian result)
        {
            // a'*b == ip, therefore:
            // E[a]'*b == ip
            // b'*var(a)*b == 0
            // inv(var(a)) = Inf*bb'
            // E[a] = ip*b/(b'b)
            result.SetToUniform();
            bool nonzero(double x) => x != 0;
            int nonZeroCount = B.CountAll(nonzero);
            if (nonZeroCount == 0) return result;
            else if (nonZeroCount == 1)
            {
                int index = B.FindFirstIndex(nonzero);
                result.Precision[index, index] = double.PositiveInfinity;
                result.MeanTimesPrecision[index] = innerProduct / B[index];
                return result;
            }
            else throw new NotImplementedException(LowRankNotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="BAverageConditional(double, Vector, VectorGaussian)"]/*'/>
        [NotSupported(InnerProductOp.LowRankNotSupportedMessage)]
        public static VectorGaussian BAverageConditional(double innerProduct, Vector A, VectorGaussian result)
        {
            return AAverageConditional(innerProduct, A, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="AAverageLogarithm(double, Vector, VectorGaussian)"]/*'/>
        [NotSupported(InnerProductOp.LowRankNotSupportedMessage)]
        public static VectorGaussian AAverageLogarithm(double innerProduct, Vector B, VectorGaussian result)
        {
            return AAverageConditional(innerProduct, B, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="BAverageLogarithm(double, Vector, VectorGaussian)"]/*'/>
        [NotSupported(InnerProductOp.LowRankNotSupportedMessage)]
        public static VectorGaussian BAverageLogarithm(double innerProduct, Vector A, VectorGaussian result)
        {
            return AAverageLogarithm(innerProduct, A, result);
        }

        //-- VMP ---------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="AverageLogFactor()"]/*'/>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="InnerProductAverageLogarithm(DenseVector, PositiveDefiniteMatrix, DenseVector, PositiveDefiniteMatrix)"]/*'/>
        public static Gaussian InnerProductAverageLogarithm(DenseVector AMean, PositiveDefiniteMatrix AVariance, DenseVector BMean, PositiveDefiniteMatrix BVariance)
        {
            Gaussian result = new Gaussian();
            // p(x|a,b) = N(E[a]'*E[b], E[b]'*var(a)*E[b] + E[a]'*var(b)*E[a] + trace(var(a)*var(b)))
            // Uses John Winn's rule for deterministic factors.
            // Strict variational inference would set the variance to 0.
            result.SetMeanAndVariance(AMean.Inner(BMean), AVariance.QuadraticForm(BMean) + BVariance.QuadraticForm(AMean) + AVariance.Inner(BVariance));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="InnerProductAverageLogarithm(Vector, DenseVector, PositiveDefiniteMatrix)"]/*'/>
        public static Gaussian InnerProductAverageLogarithm(Vector A, DenseVector BMean, PositiveDefiniteMatrix BVariance)
        {
            Gaussian result = new Gaussian();
            // Uses John Winn's rule for deterministic factors.
            // Strict variational inference would set the variance to 0.
            // p(x) = N(a' E[b], a' var(b) a)
            result.SetMeanAndVariance(A.Inner(BMean), BVariance.QuadraticForm(A));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="InnerProductAverageLogarithmInit()"]/*'/>
        [Skip]
        public static Gaussian InnerProductAverageLogarithmInit()
        {
            return new Gaussian();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="InnerProductAverageLogarithm(DenseVector, PositiveDefiniteMatrix, Vector)"]/*'/>
        public static Gaussian InnerProductAverageLogarithm(DenseVector AMean, PositiveDefiniteMatrix AVariance, Vector B)
        {
            return InnerProductAverageLogarithm(B, AMean, AVariance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="BVarianceInit(VectorGaussian)"]/*'/>
        [Skip]
        public static PositiveDefiniteMatrix BVarianceInit([IgnoreDependency] VectorGaussian B)
        {
            return new PositiveDefiniteMatrix(B.Dimension, B.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="BVariance(VectorGaussian, PositiveDefiniteMatrix)"]/*'/>
        [Fresh]
        public static PositiveDefiniteMatrix BVariance([Proper] VectorGaussian B, PositiveDefiniteMatrix result)
        {
            return B.GetVariance(result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="BMeanInit(VectorGaussian)"]/*'/>
        [Skip]
        public static DenseVector BMeanInit([IgnoreDependency] VectorGaussian B)
        {
            return DenseVector.Zero(B.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="BMean(VectorGaussian, PositiveDefiniteMatrix, DenseVector)"]/*'/>
        [Fresh]
        public static DenseVector BMean([Proper] VectorGaussian B, PositiveDefiniteMatrix BVariance, DenseVector result)
        {
            return (DenseVector)B.GetMean(result, BVariance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="AVarianceInit(VectorGaussian)"]/*'/>
        [Skip]
        public static PositiveDefiniteMatrix AVarianceInit([IgnoreDependency] VectorGaussian A)
        {
            return new PositiveDefiniteMatrix(A.Dimension, A.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="AVariance(VectorGaussian, PositiveDefiniteMatrix)"]/*'/>
        [Fresh]
        public static PositiveDefiniteMatrix AVariance([Proper] VectorGaussian A, PositiveDefiniteMatrix result)
        {
            return A.GetVariance(result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="AMeanInit(VectorGaussian)"]/*'/>
        [Skip]
        public static DenseVector AMeanInit([IgnoreDependency] VectorGaussian A)
        {
            return DenseVector.Zero(A.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="AMean(VectorGaussian, PositiveDefiniteMatrix, DenseVector)"]/*'/>
        [Fresh]
        public static DenseVector AMean([Proper] VectorGaussian A, PositiveDefiniteMatrix AVariance, DenseVector result)
        {
            return (DenseVector)A.GetMean(result, AVariance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="AAverageLogarithm(Gaussian, VectorGaussian, DenseVector, PositiveDefiniteMatrix, VectorGaussian)"]/*'/>
        public static VectorGaussian AAverageLogarithm(
            [SkipIfUniform] Gaussian innerProduct, [SkipIfUniform] VectorGaussian B, DenseVector BMean, PositiveDefiniteMatrix BVariance, VectorGaussian result)
        {
            if (innerProduct.IsPointMass)
                return AAverageLogarithm(innerProduct.Point, B, result);
            if (result == default(VectorGaussian))
                result = new VectorGaussian(B.Dimension);
            // E[log N(x; ab, 0)] = -0.5 E[(x-ab)^2]/0 = -0.5 (E[x^2] - 2 E[x] a' E[b] + trace(aa' E[bb']))/0
            // message to a = N(a; E[x]*inv(var(b)+E[b]E[b]')*E[b], var(x)*inv(var(b)+E[b]E[b]'))
            // result.Precision = (var(b)+E[b]*E[b]')/var(x)
            // result.MeanTimesPrecision = E[x]/var(x)*E[b] = E[b]*X.MeanTimesPrecision
            // note this is exact if B is a point mass (vb=0).
            result.Precision.SetToSumWithOuter(BVariance, 1, BMean, BMean);
            result.Precision.SetToProduct(result.Precision, innerProduct.Precision);
            result.MeanTimesPrecision.SetToProduct(BMean, innerProduct.MeanTimesPrecision);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="AAverageLogarithm(Gaussian, Vector, VectorGaussian)"]/*'/>
        public static VectorGaussian AAverageLogarithm([SkipIfUniform] Gaussian innerProduct, Vector B, VectorGaussian result)
        {
            if (innerProduct.IsPointMass)
                return AAverageLogarithm(innerProduct.Point, B, result);
            if (result == default(VectorGaussian))
                result = new VectorGaussian(B.Count);
            result.Precision.SetToOuter(B, B);
            result.Precision.Scale(innerProduct.Precision);
            result.MeanTimesPrecision.SetToProduct(B, innerProduct.MeanTimesPrecision);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="BAverageLogarithm(Gaussian, VectorGaussian, DenseVector, PositiveDefiniteMatrix, VectorGaussian)"]/*'/>
        public static VectorGaussian BAverageLogarithm(
            [SkipIfUniform] Gaussian innerProduct, [SkipIfUniform] VectorGaussian A, DenseVector AMean, PositiveDefiniteMatrix AVariance, VectorGaussian result)
        {
            return AAverageLogarithm(innerProduct, A, AMean, AVariance, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="BAverageLogarithm(Gaussian, Vector, VectorGaussian)"]/*'/>
        public static VectorGaussian BAverageLogarithm([SkipIfUniform] Gaussian innerProduct, Vector A, VectorGaussian result)
        {
            return AAverageLogarithm(innerProduct, A, result);
        }

        // ----------------------- AverageConditional ------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="InnerProductAverageConditional(Vector, DenseVector, PositiveDefiniteMatrix)"]/*'/>
        public static Gaussian InnerProductAverageConditional(Vector A, DenseVector BMean, PositiveDefiniteMatrix BVariance)
        {
            return InnerProductAverageLogarithm(A, BMean, BVariance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="InnerProductAverageConditionalInit()"]/*'/>
        [Skip]
        public static Gaussian InnerProductAverageConditionalInit()
        {
            return new Gaussian();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="InnerProductAverageConditional(DenseVector, PositiveDefiniteMatrix, Vector)"]/*'/>
        public static Gaussian InnerProductAverageConditional(DenseVector AMean, PositiveDefiniteMatrix AVariance, Vector B)
        {
            return InnerProductAverageConditional(B, AMean, AVariance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="AAverageConditional(Gaussian, Vector, VectorGaussian)"]/*'/>
        public static VectorGaussian AAverageConditional([SkipIfUniform] Gaussian innerProduct, Vector B, VectorGaussian result)
        {
            if (innerProduct.IsPointMass)
                return AAverageConditional(innerProduct.Point, B, result);
            if (result == default(VectorGaussian))
                result = new VectorGaussian(B.Count);
            // (m - a'b)^2/v = (a'bb'a - 2a'bm + m^2)/v
            result.Precision.SetToOuter(B, B);
            result.Precision.Scale(innerProduct.Precision);
            result.MeanTimesPrecision.SetToProduct(B, innerProduct.MeanTimesPrecision);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="AAverageConditional(Gaussian, Vector, DenseVector, PositiveDefiniteMatrix, VectorGaussian)"]/*'/>
        public static VectorGaussian AAverageConditional([SkipIfUniform] Gaussian innerProduct, Vector A, DenseVector BMean, PositiveDefiniteMatrix BVariance, VectorGaussian result)
        {
            if (innerProduct.IsUniform())
                return VectorGaussian.Uniform(A.Count);
            // logZ = log N(mProduct; A'*BMean, vProduct + A'*BVariance*A) 
            //      = -0.5 (mProduct - A'*BMean)^2 / (vProduct + A'*BVariance*A) -0.5 log(vProduct + A'*BVariance*A)
            // v*innerProduct.Precision
            double v = 1 + BVariance.QuadraticForm(A, A) * innerProduct.Precision;
            double diff = innerProduct.MeanTimesPrecision - A.Inner(BMean) * innerProduct.Precision;
            // dlogZ/dA = BMean * (mProduct - A'*BMean)/v + (BVariance*A) (diff)^2 / v^2 - (BVariance*A)/v
            double diff2 = diff * diff;
            double v2 = v * v;
            var avb = BVariance * A;
            var avbPrec = avb * innerProduct.Precision;
            var dlogZ = (BMean * diff - avbPrec) / v + avb * diff2 / v2;
            // -ddlogZ/dA^2 = (BMean.Outer(BMean) + BVariance) / v + avb.Outer(avb) * (4 * diff2 / (v2 * v))
            //               -(avb.Outer(avb - BMean * (2 * diff)) * 2 + BVariance * diff2) / v2;
            PositiveDefiniteMatrix negativeHessian = BVariance * (innerProduct.Precision / v - diff2 / v2);
            negativeHessian.SetToSumWithOuter(negativeHessian, innerProduct.Precision / v, BMean, BMean);
            negativeHessian.SetToSumWithOuter(negativeHessian, 4 * diff2 / (v2 * v) - 2 * innerProduct.Precision / v2, avb, avbPrec);
            negativeHessian.SetToSumWithOuter(negativeHessian, 2 * diff / v2, avbPrec, BMean);
            negativeHessian.SetToSumWithOuter(negativeHessian, 2 * diff / v2, BMean, avbPrec);
            negativeHessian.Symmetrize();
            return VectorGaussian.FromDerivatives(A, dlogZ, negativeHessian, GaussianProductOp.ForceProper);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="BAverageConditional(Gaussian, Vector, VectorGaussian)"]/*'/>
        public static VectorGaussian BAverageConditional([SkipIfUniform] Gaussian innerProduct, Vector A, VectorGaussian result)
        {
            return AAverageConditional(innerProduct, A, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="LogEvidenceRatio(Gaussian, Vector, VectorGaussian)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Gaussian innerProduct, Vector A, VectorGaussian b)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="LogEvidenceRatio(double, Vector, DenseVector, PositiveDefiniteMatrix)"]/*'/>
        public static double LogEvidenceRatio(double innerProduct, Vector A, DenseVector BMean, PositiveDefiniteMatrix BVariance)
        {
            return LogAverageFactor(innerProduct, A, BMean, BVariance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="LogEvidenceRatio(Gaussian, VectorGaussian, Vector)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Gaussian innerProduct, VectorGaussian a, Vector B)
        {
            return LogEvidenceRatio(innerProduct, B, a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="LogEvidenceRatio(double, DenseVector, PositiveDefiniteMatrix, Vector)"]/*'/>
        public static double LogEvidenceRatio(double innerProduct, DenseVector AMean, PositiveDefiniteMatrix AVariance, Vector B)
        {
            return LogEvidenceRatio(innerProduct, B, AMean, AVariance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="LogAverageFactor(Gaussian, Gaussian)"]/*'/>
        public static double LogAverageFactor(Gaussian innerProduct, [Fresh] Gaussian to_innerProduct)
        {
            return to_innerProduct.GetLogAverageOf(innerProduct);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="LogAverageFactor(double, Vector, DenseVector, PositiveDefiniteMatrix)"]/*'/>
        public static double LogAverageFactor(double innerProduct, Vector A, DenseVector BMean, PositiveDefiniteMatrix BVariance)
        {
            Gaussian to_innerProduct = InnerProductAverageConditional(A, BMean, BVariance);
            return to_innerProduct.GetLogProb(innerProduct);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOpBase"]/message_doc[@name="LogAverageFactor(double, DenseVector, PositiveDefiniteMatrix, Vector)"]/*'/>
        public static double LogAverageFactor(double innerProduct, DenseVector AMean, PositiveDefiniteMatrix AVariance, Vector B)
        {
            return LogAverageFactor(innerProduct, B, AMean, AVariance);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOp_PointB"]/doc/*'/>
    [FactorMethod(typeof(Vector), "InnerProduct", Default = false)]
    [Buffers("AVariance", "AMean")]
    [Quality(QualityBand.Experimental)]
    public class InnerProductOp_PointB : InnerProductOpBase
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOp_PointB"]/message_doc[@name="InnerProductAverageConditional(DenseVector, PositiveDefiniteMatrix, VectorGaussian)"]/*'/>
        public static Gaussian InnerProductAverageConditional(DenseVector AMean, PositiveDefiniteMatrix AVariance, [SkipIfUniform] VectorGaussian B)
        {
            if (!B.IsPointMass)
                throw new ArgumentException("B is not a point mass");
            return InnerProductOp.InnerProductAverageConditional(AMean, AVariance, B.Point);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOp_PointB"]/message_doc[@name="AAverageConditional(Gaussian, VectorGaussian, VectorGaussian)"]/*'/>
        public static VectorGaussian AAverageConditional([SkipIfUniform] Gaussian innerProduct, [SkipIfUniform] VectorGaussian B, VectorGaussian result)
        {
            if (!B.IsPointMass)
                throw new ArgumentException("B is not a point mass");
            return InnerProductOp.AAverageConditional(innerProduct, B.Point, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductOp_PointB"]/message_doc[@name="BAverageConditional(Gaussian, DenseVector, PositiveDefiniteMatrix, VectorGaussian)"]/*'/>
        public static VectorGaussian BAverageConditional([SkipIfUniform] Gaussian innerProduct, DenseVector AMean, PositiveDefiniteMatrix AVariance, VectorGaussian B, VectorGaussian result)
        {
            if (!B.IsPointMass)
                throw new ArgumentException("B is not a point mass");
            return InnerProductOp.AAverageConditional(innerProduct, B.Point, AMean, AVariance, result);
        }
    }
}

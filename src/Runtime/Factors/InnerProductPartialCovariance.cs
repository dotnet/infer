// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;
    using GaussianArray = Distributions.DistributionStructArray<Distributions.Gaussian, double>;

    [FactorMethod(typeof(Factor), "InnerProduct", typeof(double[]), typeof(Vector))]
    [Buffers("MeanOfB", "CovarianceOfB" /*, "Ebbt"*/)]
    [Quality(QualityBand.Preview)]
    public static class InnerProductPartialCovarianceOp
    {
        //-- VMP -------------------------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductPartialCovarianceOp"]/message_doc[@name="AverageLogFactor()"]/*'/>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductPartialCovarianceOp"]/message_doc[@name="EbbtInit(VectorGaussian)"]/*'/>
        [Skip]
        public static PositiveDefiniteMatrix EbbtInit([IgnoreDependency] VectorGaussian B)
        {
            return new PositiveDefiniteMatrix(B.Dimension, B.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductPartialCovarianceOp"]/message_doc[@name="Ebbt(PositiveDefiniteMatrix, Vector, PositiveDefiniteMatrix)"]/*'/>
        public static PositiveDefiniteMatrix Ebbt(PositiveDefiniteMatrix CovarianceOfB, Vector MeanOfB, PositiveDefiniteMatrix result)
        {
            result.SetToSumWithOuter(CovarianceOfB, 1, MeanOfB, MeanOfB);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductPartialCovarianceOp"]/message_doc[@name="EaatInit(DistributionStructArray{Gaussian, double})"]/*'/>
        [Skip]
        public static PositiveDefiniteMatrix EaatInit([IgnoreDependency] GaussianArray A)
        {
            return new PositiveDefiniteMatrix(A.Count, A.Count);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductPartialCovarianceOp"]/message_doc[@name="Eaat(DistributionStructArray{Gaussian, double}, PositiveDefiniteMatrix)"]/*'/>
        public static PositiveDefiniteMatrix Eaat(GaussianArray A, PositiveDefiniteMatrix result)
        {
            int inner = A.Count;
            var VarianceOfA = Vector.Zero(inner);
            var MeanOfA = Vector.Zero(inner);
            for (int k = 0; k < inner; k++)
            {
                MeanOfA[k] = A[k].GetMean();
                VarianceOfA[k] = A[k].GetVariance();
            }
            result.SetToDiagonal(VarianceOfA);
            result.SetToSumWithOuter(result, 1, MeanOfA, MeanOfA);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductPartialCovarianceOp"]/message_doc[@name="CovarianceOfBInit(VectorGaussian)"]/*'/>
        [Skip]
        public static PositiveDefiniteMatrix CovarianceOfBInit([IgnoreDependency] VectorGaussian B)
        {
            return new PositiveDefiniteMatrix(B.Dimension, B.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductPartialCovarianceOp"]/message_doc[@name="CovarianceOfB(VectorGaussian, PositiveDefiniteMatrix)"]/*'/>
        public static PositiveDefiniteMatrix CovarianceOfB([Proper] VectorGaussian B, PositiveDefiniteMatrix result)
        {
            return B.GetVariance(result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductPartialCovarianceOp"]/message_doc[@name="MeanOfBInit(VectorGaussian)"]/*'/>
        [Skip]
        public static Vector MeanOfBInit([IgnoreDependency] VectorGaussian B)
        {
            return Vector.Zero(B.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductPartialCovarianceOp"]/message_doc[@name="MeanOfB(VectorGaussian, PositiveDefiniteMatrix, Vector)"]/*'/>
        public static Vector MeanOfB([Proper] VectorGaussian B, PositiveDefiniteMatrix CovarianceOfB, Vector result)
        {
            return B.GetMean(result, CovarianceOfB);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductPartialCovarianceOp"]/message_doc[@name="AAverageLogarithm(Gaussian, DistributionStructArray{Gaussian, double}, VectorGaussian, Vector, PositiveDefiniteMatrix, DistributionStructArray{Gaussian, double})"]/*'/>
        public static GaussianArray AAverageLogarithm(
            [SkipIfUniform] Gaussian X,
            GaussianArray A,
            [SkipIfUniform] VectorGaussian B,
            Vector MeanOfB,
            PositiveDefiniteMatrix CovarianceOfB,
            /*PositiveDefiniteMatrix Ebbt,*/
            GaussianArray result)
        {
            int inner = MeanOfB.Count;
            if (result == null) result = new GaussianArray(inner);
            // E[log N(x[i,j]; a[i,:]*b[:,j], 0)] = -0.5 E[(x[i,j]- sum_k a[i,k]*b[k,j])^2]/0 
            // = -0.5 (E[x[i,j]^2] - 2 E[x[i,j]] a[i,k] E[b[k,j]] + a[i,k] a[i,k2] E(b[k,j] b[k2,j]))/0
            // a[i,k] * (-2 E[x[i,j]] E[b[k,j]] + sum_{k2 not k} E[a[i,k2]] E(b[k,j] b[k2,j]))
            // a[i,k]^2 * E(b[k,j]^2)
            // message to a[i,k] = N(a; inv(prec[i,k])*(sum_j E[b[k,j]]*res[i,j,k]/var(x[i,j])), inv(prec[i,k]))
            // where res[i,j,k] = E[x[i,j]] - sum_{k2 not k} E[a[i,k2]] E[b[k2,j]]
            // prec[i,k] = sum_j E(b[k,j]^2)/var(x[i,j])
            // result.Precision = prec[i,k]
            // result.MeanTimesPrecision = sum_j E[b[k,j]]*res[i,j,k]/var(x[i,j]) 
            //                           = sum_j E[b[k,j]]*(X.MeanTimesPrecision - X.precision*(sum_{k2 not k}))

            var Ebbt = new PositiveDefiniteMatrix(inner, inner);
            // should we be caching this too? 
            Ebbt.SetToSumWithOuter(CovarianceOfB, 1, MeanOfB, MeanOfB);
            //var ma = A.Select(z => z.GetMean()).ToArray(); 
            var ma = new double[inner];
            for (int k = 0; k < inner; k++)
                ma[k] = A[k].GetMean();
            for (int k = 0; k < inner; k++)
            {
                double prec = 0.0;
                double pm = 0.0;
                prec += Ebbt[k, k]*X.Precision;
                double sum = 0.0;
                for (int r = 0; r < inner; r++)
                    if (r != k)
                        sum += Ebbt[r, k]*ma[r];
                pm += MeanOfB[k]*X.MeanTimesPrecision - X.Precision*sum;

                Gaussian rk = result[k];
                rk.Precision = prec;
                if (prec < 0)
                    throw new InferRuntimeException("improper message");
                rk.MeanTimesPrecision = pm;
                result[k] = rk;
            }

            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductPartialCovarianceOp"]/message_doc[@name="BAverageLogarithm(Gaussian, double[], VectorGaussian)"]/*'/>
        public static VectorGaussian BAverageLogarithm(
            [SkipIfUniform, Proper] Gaussian X, [SkipIfAllUniform] double[] A, VectorGaussian result)
        {
            var ma = Vector.FromArray(A);
            result.Precision.SetToOuter(ma, ma);
            result.Precision.Scale(X.Precision);
            result.MeanTimesPrecision.SetToProduct(ma, X.MeanTimesPrecision);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductPartialCovarianceOp"]/message_doc[@name="BAverageLogarithm(Gaussian, DistributionStructArray{Gaussian, double}, VectorGaussian)"]/*'/>
        public static VectorGaussian BAverageLogarithm(
            [SkipIfUniform, Proper] Gaussian X, [SkipIfAllUniform] GaussianArray A, VectorGaussian result)
        {
            int K = A.Count;
            var va = Vector.Zero(K);
            var ma = Vector.Zero(K);
            for (int k = 0; k < K; k++)
            {
                double m, v;
                A[k].GetMeanAndVariance(out m, out v);
                ma[k] = m;
                va[k] = v;
            }

            result.Precision.SetToDiagonal(X.Precision*va);
            result.Precision.SetToSumWithOuter(result.Precision, X.Precision, ma, ma);
            result.MeanTimesPrecision.SetToProduct(ma, X.MeanTimesPrecision);
            //if (!result.Precision.IsPositiveDefinite())
            //    throw new InferRuntimeException("improper message");

            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductPartialCovarianceOp"]/message_doc[@name="XAverageLogarithm(double[], VectorGaussian, Vector, PositiveDefiniteMatrix)"]/*'/>
        public static Gaussian XAverageLogarithm([SkipIfAllUniform] double[] A, [SkipIfAllUniform] VectorGaussian B, Vector MeanOfB, PositiveDefiniteMatrix CovarianceOfB)
        {
            // p(x|a,b) = N(E[a]'*E[b], E[b]'*var(a)*E[b] + E[a]'*var(b)*E[a] + trace(var(a)*var(b)))
            var ma = Vector.FromArray(A);
            // Uses John Winn's rule for deterministic factors.
            // Strict variational inference would set the variance to 0.
            Gaussian result = new Gaussian();
            result.SetMeanAndVariance(ma.Inner(MeanOfB), CovarianceOfB.QuadraticForm(ma));
            if (result.Precision < 0)
                throw new InferRuntimeException("improper message");

            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductPartialCovarianceOp"]/message_doc[@name="XAverageLogarithm(DistributionStructArray{Gaussian, double}, VectorGaussian, Vector, PositiveDefiniteMatrix)"]/*'/>
        public static Gaussian XAverageLogarithm([SkipIfAllUniform] GaussianArray A, [SkipIfAllUniform] VectorGaussian B, Vector MeanOfB, PositiveDefiniteMatrix CovarianceOfB)
        {
            int K = MeanOfB.Count;
            // p(x|a,b) = N(E[a]'*E[b], E[b]'*var(a)*E[b] + E[a]'*var(b)*E[a] + trace(var(a)*var(b)))
            var ma = Vector.Zero(K);
            var va = Vector.Zero(K);
            for (int k = 0; k < K; k++)
            {
                double m, v;
                A[k].GetMeanAndVariance(out m, out v);
                ma[k] = m;
                va[k] = v;
            }
            // Uses John Winn's rule for deterministic factors.
            // Strict variational inference would set the variance to 0.
            var mbj2 = Vector.Zero(K);
            mbj2.SetToFunction(MeanOfB, x => x * x);
            // slooow
            Gaussian result = new Gaussian();
            result.SetMeanAndVariance(ma.Inner(MeanOfB), va.Inner(mbj2) + CovarianceOfB.QuadraticForm(ma) + va.Inner(CovarianceOfB.Diagonal()));
            if (result.Precision < 0)
                throw new InferRuntimeException("improper message");

            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductPartialCovarianceOp"]/message_doc[@name="XAverageLogarithmInit()"]/*'/>
        [Skip]
        public static Gaussian XAverageLogarithmInit()
        {
            return new Gaussian();
        }
    }
}

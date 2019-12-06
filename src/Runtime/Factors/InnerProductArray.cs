// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "InnerProduct", typeof(double[]), typeof(double[]))]
    [Quality(QualityBand.Experimental)]
    public static class InnerProductArrayOp
    {
        //-- VMP -------------------------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="AverageLogFactor()"]/*'/>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="InnerProductAverageLogarithm(IList{Gaussian}, IList{Gaussian}, Gaussian)"]/*'/>
        public static Gaussian InnerProductAverageLogarithm(
            [SkipIfUniform] IList<Gaussian> A, [SkipIfUniform] IList<Gaussian> B, Gaussian result)
        {
            if (result == null)
                result = new Gaussian();
            // x[i,j] = sum_k a[i,k]*b[k,j]
            // E(x[i,j]) = sum_k E(a[i,k])*E(b[k,j])
            // var(x[i,j]) = sum_k E(a[i,k])^2*var(b[k,j]) + var(a[i,k])*E(b[k,j])^2 + var(a[i,k])*var(b[k,j])
            double mean = 0;
            double var = 0;
            for (int k = 0; k < A.Count; k++)
            {
                double Am, Av, Bm, Bv;
                A[k].GetMeanAndVariance(out Am, out Av);
                B[k].GetMeanAndVariance(out Bm, out Bv);
                mean += Am * Bm;
                if ((Av == 0 && Am == 0) || (Bv == 0 && Bm == 0))
                    var = 0; // avoid infinity * 0
                else if (double.IsPositiveInfinity(Av) || double.IsPositiveInfinity(Bv))
                    var = double.PositiveInfinity; // avoid infinity * 0
                else
                    var += Av * Bm * Bm + Bv * Am * Am + Av * Bv;
            }
            Gaussian r = result;
            r.SetMeanAndVariance(mean, var);
            result = r;
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="InnerProductAverageLogarithmInit(IList{Gaussian}, IList{Gaussian})"]/*'/>
        [Skip]
        public static Gaussian InnerProductAverageLogarithmInit([IgnoreDependency] IList<Gaussian> A, [IgnoreDependency] IList<Gaussian> B)
        {
            return new Gaussian();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="InnerProductAverageLogarithmInit(double[], IList{Gaussian})"]/*'/>
        [Skip]
        public static Gaussian InnerProductAverageLogarithmInit([IgnoreDependency] double[] A, [IgnoreDependency] IList<Gaussian> B)
        {
            return new Gaussian();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="InnerProductAverageLogarithmInit(IList{Gaussian}, double[])"]/*'/>
        [Skip]
        public static Gaussian InnerProductAverageLogarithmInit([IgnoreDependency] IList<Gaussian> A, [IgnoreDependency] double[] B)
        {
            return new Gaussian();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="InnerProductAverageLogarithm(double[], IList{Gaussian})"]/*'/>
        public static Gaussian InnerProductAverageLogarithm(double[] A, [SkipIfUniform] IList<Gaussian> B)
        {
            return InnerProductAverageConditional(A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="InnerProductAverageLogarithm(IList{Gaussian}, double[])"]/*'/>
        public static Gaussian InnerProductAverageLogarithm([SkipIfUniform] IList<Gaussian> A, double[] B)
        {
            return InnerProductAverageConditional(A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="AAverageLogarithm{GaussianList}(Gaussian, IList{Gaussian}, IList{Gaussian}, GaussianList)"]/*'/>
        public static GaussianList AAverageLogarithm<GaussianList>(
            [SkipIfUniform] Gaussian innerProduct, [Stochastic] IList<Gaussian> A, [SkipIfUniform] IList<Gaussian> B, GaussianList to_A)
            where GaussianList : IList<Gaussian>
        {
            GaussianList result = to_A;
            //if (result == null)
            //{
            //    result = new GaussianList(A.Count);
            //    for (int k = 0; k < A.Count; k++)
            //        result[k] = new Gaussian();
            //}

            double ipMean = 0;
            for (int k = 0; k < A.Count; k++)
                ipMean += A[k].GetMean() * B[k].GetMean();

            double aMean,
                   bMean, bVariance;

            Gaussian result_old = new Gaussian();
            Gaussian cc;

            for (int k = 0; k < A.Count; k++)
            {
                aMean = A[k].GetMean();
                B[k].GetMeanAndVariance(out bMean, out bVariance);


                //result_old.SetNatural(result[k].MeanTimesPrecision, result[k].Precision);
                result_old.SetTo(result[k]);
                cc = result[k];

                if (!innerProduct.IsPointMass)
                    cc.SetNatural(bMean * (innerProduct.MeanTimesPrecision - innerProduct.Precision * (ipMean - aMean * bMean)), innerProduct.Precision * (bVariance + bMean * bMean));
                else
                    cc.SetTo(Gaussian.PointMass(bMean * (innerProduct.Point - (ipMean - aMean * bMean)) / (bVariance + bMean * bMean)));

                result[k] = cc;

                // book-keeping for ipMean so that we have a sequential scheduling/coordinate descent for the elements of A 
                Gaussian newPost = A[k] * result[k] / result_old;           // update the global marginal
                ipMean = ipMean - (aMean - newPost.GetMean()) * bMean;     // update the dependent quantities
            }

            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="AAverageLogarithm{GaussianList}(Gaussian, IList{Gaussian}, double[], GaussianList)"]/*'/>
        public static GaussianList AAverageLogarithm<GaussianList>(
            [SkipIfUniform] Gaussian innerProduct, [Proper, Stochastic] IList<Gaussian> A, double[] B, GaussianList to_A)
            where GaussianList : IList<Gaussian>
        {

            GaussianList result = to_A;
            //if (result == null)
            //{
            //    result = new DistributionStructArray<Gaussian, double>(A.Count);
            //    for (int k = 0; k < A.Count; k++)
            //        result[k] = new Gaussian();
            //}

            double ipMean = 0;
            for (int k = 0; k < A.Count; k++)
                ipMean += A[k].GetMean() * B[k];

            double aMean;

            Gaussian result_old = new Gaussian();
            Gaussian cc;

            for (int k = 0; k < A.Count; k++)
            {
                aMean = A[k].GetMean();

                //result_old.SetNatural(result[k].MeanTimesPrecision, result[k].Precision);
                result_old.SetTo(result[k]);
                cc = result[k];

                if (!innerProduct.IsPointMass)
                    cc.SetNatural(B[k] * (innerProduct.MeanTimesPrecision - innerProduct.Precision * (ipMean - aMean * B[k])), innerProduct.Precision * (B[k] * B[k]));
                else
                    cc.SetTo(Gaussian.PointMass(B[k] * (innerProduct.Point - (ipMean - aMean * B[k])) / (B[k] * B[k])));

                result[k] = cc;

                // book-keeping for ipMean so that we have a sequential scheduling/coordinate descent for the elements of A 
                Gaussian newPost = A[k] * result[k] / result_old;          // update the global marginal
                ipMean = ipMean - (aMean - newPost.GetMean()) * B[k];      // update the dependent quantities
            }

            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="BAverageLogarithm{GaussianList}(Gaussian, IList{Gaussian}, IList{Gaussian}, GaussianList)"]/*'/>
        public static GaussianList BAverageLogarithm<GaussianList>(
            [SkipIfUniform] Gaussian innerProduct, [SkipIfUniform] IList<Gaussian> A, [Proper, Stochastic] IList<Gaussian> B, GaussianList to_B)
            where GaussianList : IList<Gaussian>
        {
            return AAverageLogarithm(innerProduct, B, A, to_B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="BAverageLogarithm{GaussianList}(Gaussian, double[], IList{Gaussian}, GaussianList)"]/*'/>
        public static GaussianList BAverageLogarithm<GaussianList>(
            [SkipIfUniform] Gaussian innerProduct, double[] A, [Proper, Stochastic] IList<Gaussian> B, GaussianList to_B)
            where GaussianList : IList<Gaussian>
        {
            return AAverageLogarithm(innerProduct, B, A, to_B);
        }

        private const string NotSupportedMessage = "Variational Message Passing does not support a InnerProduct factor with fixed output.";

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="AAverageLogarithm{GaussianList}(double)"]/*'/>
        [NotSupported(InnerProductArrayOp.NotSupportedMessage)]
        public static GaussianList AAverageLogarithm<GaussianList>(double innerProduct)
               where GaussianList : IList<Gaussian>
        {
            throw new NotSupportedException(InnerProductArrayOp.NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="BAverageLogarithm(double)"]/*'/>
        [NotSupported(InnerProductArrayOp.NotSupportedMessage)]
        public static IList<Gaussian> BAverageLogarithm(double innerProduct)
        {
            throw new NotSupportedException(InnerProductArrayOp.NotSupportedMessage);
        }

        // AverageConditional ------------------------------------------------------------------------------------------------------------------------

        private const string LowRankNotSupportedMessage = "A InnerProduct factor with fixed output is not yet implemented for Expectation Propagation.";

        private const string BothRandomNotSupportedMessage =
            "A InnerProduct factor between two Gaussian arrays is not yet implemented for Expectation Propagation.  Try using Variational Message Passing.";

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="InnerProductAverageConditional(IList{Gaussian}, IList{Gaussian}, IList{Gaussian})"]/*'/>
        [NotSupported(InnerProductArrayOp.BothRandomNotSupportedMessage)]
        public static Gaussian InnerProductAverageConditional([SkipIfUniform] IList<Gaussian> A, [SkipIfUniform] IList<Gaussian> B, IList<Gaussian> result)
        {
            throw new NotSupportedException(InnerProductArrayOp.BothRandomNotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="AAverageConditional{GaussianList}(Gaussian, IList{Gaussian}, GaussianList)"]/*'/>
        [NotSupported(InnerProductArrayOp.BothRandomNotSupportedMessage)]
        public static GaussianList AAverageConditional<GaussianList>(Gaussian innerProduct, [SkipIfUniform] IList<Gaussian> B, GaussianList result)
            where GaussianList : IList<Gaussian>
        {
            throw new NotSupportedException(InnerProductArrayOp.BothRandomNotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="BAverageConditional{GaussianList}(Gaussian, IList{Gaussian}, GaussianList)"]/*'/>
        [NotSupported(InnerProductArrayOp.BothRandomNotSupportedMessage)]
        public static GaussianList BAverageConditional<GaussianList>(Gaussian innerProduct, [SkipIfUniform] IList<Gaussian> A, GaussianList result)
            where GaussianList : IList<Gaussian>
        {
            throw new NotSupportedException(InnerProductArrayOp.BothRandomNotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="LogAverageFactor(double, double[], double[])"]/*'/>
        public static double LogAverageFactor(double innerProduct, double[] A, double[] B)
        {

            double sum = 0;
            for (int k = 0; k < A.GetLength(0); k++)
            {
                sum += A[k] * B[k];
            }
            if (innerProduct != sum)
                return Double.NegativeInfinity;
            // return sum

            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="LogEvidenceRatio(double, double[], double[])"]/*'/>
        public static double LogEvidenceRatio(double innerProduct, double[] A, double[] B)
        {
            return LogAverageFactor(innerProduct, A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="AverageLogFactor(double, double[], double[])"]/*'/>
        public static double AverageLogFactor(double innerProduct, double[] A, double[] B)
        {
            return LogAverageFactor(innerProduct, A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="LogAverageFactor(Gaussian, double[], IList{Gaussian})"]/*'/>
        public static double LogAverageFactor(Gaussian innerProduct, double[] A, IList<Gaussian> B)
        {
            Gaussian to_innerProduct = InnerProductAverageConditional(A, B);
            return to_innerProduct.GetLogAverageOf(innerProduct);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="LogAverageFactor(double, double[], IList{Gaussian})"]/*'/>
        public static double LogAverageFactor(double innerProduct, double[] A, IList<Gaussian> B)
        {
            Gaussian to_innerProduct = InnerProductAverageConditional(A, B);
            return to_innerProduct.GetLogProb(innerProduct);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="LogAverageFactor(Gaussian, IList{Gaussian}, double[])"]/*'/>
        public static double LogAverageFactor(Gaussian innerProduct, IList<Gaussian> A, double[] B)
        {
            Gaussian to_innerProduct = InnerProductAverageConditional(A, B);
            return to_innerProduct.GetLogAverageOf(innerProduct);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="LogAverageFactor(double, IList{Gaussian}, double[])"]/*'/>
        public static double LogAverageFactor(double innerProduct, IList<Gaussian> A, double[] B)
        {
            Gaussian to_innerProduct = InnerProductAverageConditional(A, B);
            return to_innerProduct.GetLogProb(innerProduct);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="LogEvidenceRatio(Gaussian, double[], IList{Gaussian})"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Gaussian innerProduct, double[] A, IList<Gaussian> B)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="LogEvidenceRatio(Gaussian, IList{Gaussian}, double[])"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Gaussian innerProduct, IList<Gaussian> A, double[] B)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="LogEvidenceRatio(double, double[], IList{Gaussian})"]/*'/>
        public static double LogEvidenceRatio(double innerProduct, double[] A, IList<Gaussian> B)
        {
            return LogAverageFactor(innerProduct, A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="LogEvidenceRatio(double, IList{Gaussian}, double[])"]/*'/>
        public static double LogEvidenceRatio(double innerProduct, IList<Gaussian> A, double[] B)
        {
            return LogAverageFactor(innerProduct, A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="InnerProductAverageConditional(double[], IList{Gaussian})"]/*'/>
        public static Gaussian InnerProductAverageConditional(double[] A, [SkipIfUniform] IList<Gaussian> B)
        {
            double xMean = 0, xVariance = 0,
                   bMean, bVariance;

            for (int k = 0; k < A.Length; k++)
            {
                B[k].GetMeanAndVariance(out bMean, out bVariance);
                xMean += A[k] * bMean;
                xVariance += A[k] * A[k] * bVariance;
            }

            return Gaussian.FromMeanAndVariance(xMean, xVariance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="InnerProductAverageConditional(IList{Gaussian}, double[])"]/*'/>
        public static Gaussian InnerProductAverageConditional([SkipIfUniform] IList<Gaussian> A, double[] B)
        {
            return InnerProductAverageConditional(B, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="AAverageConditional{GaussianList}(Gaussian, IList{Gaussian}, double[], Gaussian, GaussianList)"]/*'/>
        public static GaussianList AAverageConditional<GaussianList>(
            [SkipIfUniform] Gaussian innerProduct, IList<Gaussian> A, double[] B, [Fresh] Gaussian to_innerProduct, GaussianList result)
            where GaussianList : IList<Gaussian>
        {
            // It is tempting to put SkipIfAllUniform on A but this isn't correct if the array has one element.
            if (innerProduct.Precision == 0 || A.Count == 1)
            {
                for (int i = 0; i < result.Count; i++)
                {
                    result[i] = GaussianProductOpBase.AAverageConditional(innerProduct, B[i]);
                }
                return result;
            }

            double xMean, xVariance;
            innerProduct.GetMeanAndVariance(out xMean, out xVariance);

            double aMean, aVariance;
            double toXVariance, toXMean;
            to_innerProduct.GetMeanAndVariance(out toXMean, out toXVariance);

            for (int k = 0; k < B.Length; k++)
            {
                A[k].GetMeanAndVariance(out aMean, out aVariance);

                Gaussian msg = new Gaussian(xMean - (toXMean - B[k] * aMean), xVariance + toXVariance - B[k] * B[k] * aVariance);
                msg = GaussianProductOpBase.AAverageConditional(msg, B[k]);
                bool damp = false;
                if (damp)
                {
                    Gaussian msg_old = result[k];
                    double step = 0.62;
                    msg = (msg ^ step) * (msg_old ^ (1 - step));
                }
                result[k] = msg;
            }

            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="BAverageConditional{GaussianList}(Gaussian, double[], IList{Gaussian}, Gaussian, GaussianList)"]/*'/>
        public static GaussianList BAverageConditional<GaussianList>(
            [SkipIfUniform] Gaussian innerProduct, double[] A, [SkipIfUniform] IList<Gaussian> B, Gaussian to_innerProduct, GaussianList result)
            where GaussianList : IList<Gaussian>
        {
            return AAverageConditional(innerProduct, B, A, to_innerProduct, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="AAverageConditional{GaussianList}(double, double[], GaussianList)"]/*'/>
        [NotSupported(InnerProductArrayOp.LowRankNotSupportedMessage)]
        public static GaussianList AAverageConditional<GaussianList>(double innerProduct, double[] B, GaussianList result)
            where GaussianList : IList<Gaussian>
        {
            throw new NotImplementedException(InnerProductArrayOp.LowRankNotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InnerProductArrayOp"]/message_doc[@name="BAverageConditional{GaussianList}(double, double[], GaussianList)"]/*'/>
        [NotSupported(InnerProductArrayOp.LowRankNotSupportedMessage)]
        public static GaussianList BAverageConditional<GaussianList>(double innerProduct, double[] A, GaussianList result)
            where GaussianList : IList<Gaussian>
        {
            throw new NotImplementedException(InnerProductArrayOp.LowRankNotSupportedMessage);
        }
    }
}

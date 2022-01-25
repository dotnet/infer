// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Distributions;
    using Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Ratio", typeof(double), typeof(double))]
    [Quality(QualityBand.Mature)]
    public static class RatioGaussianOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianOp"]/message_doc[@name="RatioAverageConditional(Gaussian, double)"]/*'/>
        public static Gaussian RatioAverageConditional([SkipIfUniform] Gaussian a, double b)
        {
            return RatioGaussianVmpOp.RatioAverageLogarithm(a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianOp"]/message_doc[@name="RatioAverageConditional(Gaussian, Gaussian, Gaussian)"]/*'/>
        public static Gaussian RatioAverageConditional(Gaussian ratio, [SkipIfUniform] Gaussian a, [SkipIfUniform] Gaussian b)
        {
            if (b.IsPointMass) return RatioAverageConditional(a, b.Point);
            else throw new NotSupportedException();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianOp"]/message_doc[@name="AAverageConditional(Gaussian, double)"]/*'/>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian ratio, double b)
        {
            return GaussianProductOp.ProductAverageConditional(ratio, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianOp"]/message_doc[@name="AAverageConditional(Gaussian, Gaussian, Gaussian)"]/*'/>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian ratio, Gaussian a, [SkipIfUniform] Gaussian b)
        {
            if (b.IsPointMass)
            {
                return AAverageConditional(ratio, b.Point);
            }
            else throw new NotSupportedException();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianOp"]/message_doc[@name="AAverageConditional(Gaussian, Gaussian, Gaussian)"]/*'/>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian ratio, [SkipIfUniform] Gaussian a, Gaussian b)
        {
            //if (string.Empty.Length == 0) return GaussianProductOp.BAverageConditional(Gaussian.FromNatural(a.MeanTimesPrecision, Math.Min(a.Precision, 1e16)), Gaussian.FromNatural(ratio.MeanTimesPrecision, ratio.Precision + 1e-16), b);
            if (b.IsPointMass && ratio.Precision == 0)
            {
                // f(b) = int_a N(m; a/b, v) p(a) da
                //      = N(m; ma/b, v + va/b^2)
                // logf = -0.5*log(v + va/b^2) -0.5*(m - ma/b)^2/(v + va/b^2)
                // dlogf = va/b^3/(v + va/b^2) -(m - ma/b)/(v + va/b^2)*(ma/b^2) + (m - ma/b)^2/(v + va/b^2)^2*(va/b^3)
                // -> -(m - ma/b)/v * (ma/b^2)
                double dlogp = -a.GetMean() / (b.Point * b.Point) * ratio.MeanTimesPrecision;
                double ddlogp = 0;
                return Gaussian.FromDerivatives(b.Point, dlogp, ddlogp, false);
            }
            else throw new NotSupportedException();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianOp"]/message_doc[@name="LogAverageFactor(Gaussian, Gaussian, double, Gaussian)"]/*'/>
        public static double LogAverageFactor(Gaussian ratio, Gaussian a, double b, [Fresh] Gaussian to_ratio)
        {
            //Gaussian to_ratio = GaussianProductOp.AAverageConditional(a, b);
            return to_ratio.GetLogAverageOf(ratio);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianOp"]/message_doc[@name="LogAverageFactor(Gaussian, double, Gaussian, Gaussian)"]/*'/>
        public static double LogAverageFactor(Gaussian ratio, double a, Gaussian b, [Fresh] Gaussian to_ratio)
        {
            return LogAverageFactor(ratio, b, a, to_ratio);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianOp"]/message_doc[@name="LogAverageFactor(double, Gaussian, double)"]/*'/>
        public static double LogAverageFactor(double ratio, Gaussian a, double b)
        {
            Gaussian to_ratio = GaussianProductOp.AAverageConditional(a, b);
            return to_ratio.GetLogProb(ratio);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianOp"]/message_doc[@name="LogAverageFactor(double, double, Gaussian)"]/*'/>
        public static double LogAverageFactor(double ratio, double a, Gaussian b)
        {
            return LogAverageFactor(ratio, b, a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianOp"]/message_doc[@name="LogAverageFactor(double, double, double)"]/*'/>
        public static double LogAverageFactor(double ratio, double a, double b)
        {
            return (ratio == Factor.Ratio(a, b)) ? 0.0 : Double.NegativeInfinity;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianOp"]/message_doc[@name="LogEvidenceRatio(double, double, double)"]/*'/>
        public static double LogEvidenceRatio(double ratio, double a, double b)
        {
            return LogAverageFactor(ratio, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianOp"]/message_doc[@name="LogAverageFactor(Gaussian, double, double)"]/*'/>
        public static double LogAverageFactor(Gaussian ratio, double a, double b)
        {
            return ratio.GetLogProb(Factor.Ratio(a, b));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianOp"]/message_doc[@name="LogEvidenceRatio(Gaussian, Gaussian, double)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Gaussian ratio, Gaussian a, double b)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianOp"]/message_doc[@name="LogEvidenceRatio(Gaussian, double, Gaussian)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Gaussian ratio, double a, Gaussian b)
        {
            return LogEvidenceRatio(ratio, b, a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianOp"]/message_doc[@name="LogEvidenceRatio(double, Gaussian, double)"]/*'/>
        public static double LogEvidenceRatio(double ratio, Gaussian a, double b)
        {
            return LogAverageFactor(ratio, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianOp"]/message_doc[@name="LogEvidenceRatio(double, double, Gaussian)"]/*'/>
        public static double LogEvidenceRatio(double ratio, double a, Gaussian b)
        {
            return LogEvidenceRatio(ratio, b, a);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianVmpOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Ratio", typeof(double), typeof(double))]
    [Quality(QualityBand.Mature)]
    public static class RatioGaussianVmpOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianVmpOp"]/message_doc[@name="AverageLogFactor(Gaussian)"]/*'/>
        /// <remarks><para>
        /// Variational Message Passing does not support a Ratio factor with fixed output or random denominator
        /// </para></remarks>
        [Skip]
        public static double AverageLogFactor(Gaussian ratio)
        {
            return 0.0;
        }

        internal const string NotSupportedMessage = "Variational Message Passing does not support a Ratio factor with fixed output or random denominator.";

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianVmpOp"]/message_doc[@name="RatioAverageLogarithm(Gaussian)"]/*'/>
        /// <remarks><para>
        /// Variational Message Passing does not support a Ratio factor with fixed output or random denominator
        /// </para></remarks>
        [NotSupported(RatioGaussianVmpOp.NotSupportedMessage)]
        public static Gaussian RatioAverageLogarithm(Gaussian B)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianVmpOp"]/message_doc[@name="RatioAverageLogarithm(Gaussian, double)"]/*'/>
        public static Gaussian RatioAverageLogarithm([SkipIfUniform] Gaussian A, double B)
        {
            return GaussianProductOp.AAverageConditional(A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianVmpOp"]/message_doc[@name="AAverageLogarithm(Gaussian)"]/*'/>
        /// <remarks><para>
        /// Variational Message Passing does not support a Ratio factor with fixed output or random denominator
        /// </para></remarks>
        [NotSupported(RatioGaussianVmpOp.NotSupportedMessage)]
        public static Gaussian AAverageLogarithm(Gaussian B)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianVmpOp"]/message_doc[@name="AAverageLogarithm(Gaussian, double)"]/*'/>
        public static Gaussian AAverageLogarithm([SkipIfUniform] Gaussian ratio, double B)
        {
            return GaussianProductOp.ProductAverageConditional(ratio, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianVmpOp"]/message_doc[@name="AAverageLogarithm(double, double)"]/*'/>
        public static Gaussian AAverageLogarithm(double ratio, double B)
        {
            return GaussianProductOp.ProductAverageConditional(ratio, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RatioGaussianVmpOp"]/message_doc[@name="BAverageLogarithm()"]/*'/>
        /// <remarks><para>
        /// Variational Message Passing does not support a Ratio factor with fixed output or random denominator
        /// </para></remarks>
        [NotSupported(RatioGaussianVmpOp.NotSupportedMessage)]
        public static Gaussian BAverageLogarithm()
        {
            throw new NotSupportedException(NotSupportedMessage);
        }
    }
}

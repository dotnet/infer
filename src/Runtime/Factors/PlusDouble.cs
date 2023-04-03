// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;
    using Microsoft.ML.Probabilistic.Math;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Plus", typeof(double), typeof(double), Default = true)]
    [FactorMethod(new string[] { "A", "Sum", "B" }, typeof(Factor), "Difference", typeof(double), typeof(double), Default = true)]
    [Quality(QualityBand.Mature)]
    public static class DoublePlusOp
    {
        // ----------------------------------------------------------------------------------------------------------------------
        // Gaussian 
        // ----------------------------------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusOp"]/message_doc[@name="LogAverageFactor(Gaussian, Gaussian)"]/*'/>
        public static double LogAverageFactor([SkipIfUniform] Gaussian Sum, [Fresh] Gaussian to_sum)
        {
            return to_sum.GetLogAverageOf(Sum);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusOp"]/message_doc[@name="LogAverageFactor(double, Gaussian, Gaussian)"]/*'/>
        public static double LogAverageFactor(double Sum, [SkipIfUniform] Gaussian a, [Fresh] Gaussian to_a)
        {
            return to_a.GetLogAverageOf(a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusOp"]/message_doc[@name="LogAverageFactor(double, double, Gaussian, Gaussian)"]/*'/>
        public static double LogAverageFactor(double Sum, double a, Gaussian b, [Fresh] Gaussian to_b)
        {
            return LogAverageFactor(Sum, b, to_b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusOp"]/message_doc[@name="LogAverageFactor(double, double, double)"]/*'/>
        public static double LogAverageFactor(double Sum, double a, double b)
        {
            return (Sum == Factor.Plus(a, b)) ? 0.0 : Double.NegativeInfinity;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusOp"]/message_doc[@name="SumAverageConditional(Gaussian, Gaussian)"]/*'/>
        public static Gaussian SumAverageConditional([SkipIfUniform] Gaussian a, [SkipIfUniform] Gaussian b)
        {
            if (a.IsPointMass)
                return SumAverageConditional(a.Point, b);
            if (b.IsPointMass)
                return SumAverageConditional(a, b.Point);
            if (a.IsUniform() || b.IsUniform()) return Gaussian.Uniform();
            // E[sum] = E[a] + E[b]
            // var(sum) = var(a) + var(b) = 1/aPrec + 1/bPrec
            // E[sum]/var(sum) = E[a]/var(sum) + E[b]/var(sum)
            //                 = E[a]/var(a)*var(a)/(1/aPrec + 1/bPrec) + E[b]/var(sum)
            // var(a)/(1/aPrec + 1/bPrec) = 1/(1 + aPrec/bPrec) = bPrec/(aPrec + bPrec)
            double meanTimesPrec = MMath.WeightedAverage(b.Precision, a.MeanTimesPrecision, a.Precision, b.MeanTimesPrecision);
            return Gaussian.FromNatural(meanTimesPrec, GetPrecisionOfSum(a, b));
        }

        private static double GetPrecisionOfSum(Gaussian a, Gaussian b)
        {
            if (a.Precision >= b.Precision)
            {
                double precOverAPrec = 1 + b.Precision / a.Precision;
                if (precOverAPrec == 0)
                    throw new ImproperDistributionException(a.IsProper() ? b : a);
                return b.Precision / precOverAPrec;
            }
            else
            {
                double precOverBPrec = a.Precision / b.Precision + 1;
                if (precOverBPrec == 0)
                    throw new ImproperDistributionException(a.IsProper() ? b : a);
                return a.Precision / precOverBPrec;
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusOp"]/message_doc[@name="SumAverageConditional(double, Gaussian)"]/*'/>
        public static Gaussian SumAverageConditional(double a, [SkipIfUniform] Gaussian b)
        {
            if (b.IsPointMass)
                return SumAverageConditional(a, b.Point);
            if (b.Precision == 0)
                return b;
            double meanTimesPrecision = b.MeanTimesPrecision + a * b.Precision;
            if (Math.Abs(meanTimesPrecision) > double.MaxValue)
            {
                return Gaussian.FromMeanAndPrecision(b.GetMean() + a, b.Precision);
            }
            else
            {
                return Gaussian.FromNatural(meanTimesPrecision, b.Precision);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusOp"]/message_doc[@name="SumAverageConditional(Gaussian, double)"]/*'/>
        public static Gaussian SumAverageConditional([SkipIfUniform] Gaussian a, double b)
        {
            return SumAverageConditional(b, a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusOp"]/message_doc[@name="SumAverageConditional(double, double)"]/*'/>
        public static Gaussian SumAverageConditional(double a, double b)
        {
            return Gaussian.PointMass(a + b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusOp"]/message_doc[@name="AAverageConditional(Gaussian, Gaussian)"]/*'/>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Sum, [SkipIfUniform] Gaussian b)
        {
            return SumAverageConditional(Sum, Gaussian.FromNatural(-b.MeanTimesPrecision, b.Precision));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusOp"]/message_doc[@name="AAverageConditional(double, Gaussian)"]/*'/>
        public static Gaussian AAverageConditional(double Sum, [SkipIfUniform] Gaussian b)
        {
            return SumAverageConditional(Sum, Gaussian.FromNatural(-b.MeanTimesPrecision, b.Precision));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusOp"]/message_doc[@name="AAverageConditional(Gaussian, double)"]/*'/>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Sum, double b)
        {
            return SumAverageConditional(Sum, -b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusOp"]/message_doc[@name="AAverageConditional(double, double)"]/*'/>
        public static Gaussian AAverageConditional(double Sum, double b)
        {
            return Gaussian.PointMass(Sum - b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusOp"]/message_doc[@name="BAverageConditional(Gaussian, Gaussian)"]/*'/>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Sum, [SkipIfUniform] Gaussian a)
        {
            return AAverageConditional(Sum, a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusOp"]/message_doc[@name="BAverageConditional(double, Gaussian)"]/*'/>
        public static Gaussian BAverageConditional(double Sum, [SkipIfUniform] Gaussian a)
        {
            return AAverageConditional(Sum, a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusOp"]/message_doc[@name="BAverageConditional(Gaussian, double)"]/*'/>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Sum, double a)
        {
            return AAverageConditional(Sum, a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusOp"]/message_doc[@name="BAverageConditional(double, double)"]/*'/>
        public static Gaussian BAverageConditional(double Sum, double a)
        {
            return AAverageConditional(Sum, a);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusEvidenceOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Plus", typeof(double), typeof(double))]
    [Quality(QualityBand.Stable)]
    public static class DoublePlusEvidenceOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusEvidenceOp"]/message_doc[@name="LogEvidenceRatio(Gaussian)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Gaussian Sum)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusEvidenceOp"]/message_doc[@name="LogEvidenceRatio(TruncatedGaussian)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(TruncatedGaussian sum)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusEvidenceOp"]/message_doc[@name="LogEvidenceRatio(double, double, double)"]/*'/>
        public static double LogEvidenceRatio(double Sum, double a, double b)
        {
            return DoublePlusOp.LogAverageFactor(Sum, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusEvidenceOp"]/message_doc[@name="LogEvidenceRatio(double, double, Gaussian, Gaussian)"]/*'/>
        public static double LogEvidenceRatio(double Sum, double a, Gaussian b, [Fresh] Gaussian to_b)
        {
            return DoublePlusOp.LogAverageFactor(b, to_b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusEvidenceOp"]/message_doc[@name="LogEvidenceRatio(double, Gaussian, Gaussian)"]/*'/>
        public static double LogEvidenceRatio(double Sum, Gaussian a, [Fresh] Gaussian to_a)
        {
            return DoublePlusOp.LogAverageFactor(a, to_a);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleMinusEvidenceOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Difference", typeof(double), typeof(double))]
    [Quality(QualityBand.Stable)]
    public static class DoubleMinusEvidenceOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleMinusEvidenceOp"]/message_doc[@name="LogEvidenceRatio(Gaussian)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Gaussian difference)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleMinusEvidenceOp"]/message_doc[@name="LogEvidenceRatio(TruncatedGaussian)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(TruncatedGaussian difference)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleMinusEvidenceOp"]/message_doc[@name="LogEvidenceRatio(double, double, double)"]/*'/>
        public static double LogEvidenceRatio(double difference, double a, double b)
        {
            return DoublePlusOp.LogAverageFactor(a, difference, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleMinusEvidenceOp"]/message_doc[@name="LogEvidenceRatio(double, Gaussian, Gaussian)"]/*'/>
        public static double LogEvidenceRatio(double difference, Gaussian a, [Fresh] Gaussian to_a)
        {
            return DoublePlusOp.LogAverageFactor(a, to_a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleMinusEvidenceOp"]/message_doc[@name="LogEvidenceRatio(double, double, Gaussian, Gaussian)"]/*'/>
        public static double LogEvidenceRatio(double difference, double a, Gaussian b, [Fresh] Gaussian to_b)
        {
            return DoublePlusOp.LogAverageFactor(b, to_b);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusWrappedGaussianOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Plus", typeof(double), typeof(double), Default = true)]
    [FactorMethod(new string[] { "A", "Sum", "B" }, typeof(Factor), "Difference", typeof(double), typeof(double), Default = true)]
    [Quality(QualityBand.Mature)]
    public static class PlusWrappedGaussianOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusWrappedGaussianOp"]/message_doc[@name="SumAverageConditional(WrappedGaussian, WrappedGaussian)"]/*'/>
        public static WrappedGaussian SumAverageConditional([SkipIfUniform] WrappedGaussian a, [SkipIfUniform] WrappedGaussian b)
        {
            if (a.Period != b.Period)
                throw new ArgumentException("a.Period (" + a.Period + ") != b.Period (" + b.Period + ")");
            WrappedGaussian result = WrappedGaussian.Uniform(a.Period);
            result.Gaussian = DoublePlusOp.SumAverageConditional(a.Gaussian, b.Gaussian);
            result.Normalize();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusWrappedGaussianOp"]/message_doc[@name="SumAverageConditional(WrappedGaussian, double)"]/*'/>
        public static WrappedGaussian SumAverageConditional([SkipIfUniform] WrappedGaussian a, double b)
        {
            WrappedGaussian result = WrappedGaussian.Uniform(a.Period);
            result.Gaussian = DoublePlusOp.SumAverageConditional(a.Gaussian, b);
            result.Normalize();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusWrappedGaussianOp"]/message_doc[@name="SumAverageConditional(double, WrappedGaussian)"]/*'/>
        public static WrappedGaussian SumAverageConditional(double a, [SkipIfUniform] WrappedGaussian b)
        {
            return SumAverageConditional(b, a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusWrappedGaussianOp"]/message_doc[@name="AAverageConditional(WrappedGaussian, WrappedGaussian)"]/*'/>
        public static WrappedGaussian AAverageConditional([SkipIfUniform] WrappedGaussian sum, [SkipIfUniform] WrappedGaussian b)
        {
            if (sum.Period != b.Period)
                throw new ArgumentException("sum.Period (" + sum.Period + ") != b.Period (" + b.Period + ")");
            WrappedGaussian result = WrappedGaussian.Uniform(sum.Period);
            result.Gaussian = DoublePlusOp.AAverageConditional(sum.Gaussian, b.Gaussian);
            result.Normalize();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusWrappedGaussianOp"]/message_doc[@name="AAverageConditional(WrappedGaussian, double)"]/*'/>
        public static WrappedGaussian AAverageConditional([SkipIfUniform] WrappedGaussian sum, double b)
        {
            WrappedGaussian result = WrappedGaussian.Uniform(sum.Period);
            result.Gaussian = DoublePlusOp.AAverageConditional(sum.Gaussian, b);
            result.Normalize();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusWrappedGaussianOp"]/message_doc[@name="BAverageConditional(WrappedGaussian, WrappedGaussian)"]/*'/>
        public static WrappedGaussian BAverageConditional([SkipIfUniform] WrappedGaussian sum, [SkipIfUniform] WrappedGaussian a)
        {
            return AAverageConditional(sum, a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusWrappedGaussianOp"]/message_doc[@name="BAverageConditional(WrappedGaussian, double)"]/*'/>
        public static WrappedGaussian BAverageConditional([SkipIfUniform] WrappedGaussian sum, double a)
        {
            return AAverageConditional(sum, a);
        }

        // ----------------------------------------------------------------------------------------------------------------------
        // VMP
        // ----------------------------------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusWrappedGaussianOp"]/message_doc[@name="AverageLogFactor(WrappedGaussian)"]/*'/>
        [Skip]
        public static double AverageLogFactor(WrappedGaussian sum)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusWrappedGaussianOp"]/message_doc[@name="SumAverageLogarithm(WrappedGaussian, WrappedGaussian)"]/*'/>
        public static WrappedGaussian SumAverageLogarithm([SkipIfUniform] WrappedGaussian a, [SkipIfUniform] WrappedGaussian b)
        {
            return PlusWrappedGaussianOp.SumAverageConditional(a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusWrappedGaussianOp"]/message_doc[@name="SumAverageLogarithm(WrappedGaussian, double)"]/*'/>
        public static WrappedGaussian SumAverageLogarithm([SkipIfUniform] WrappedGaussian a, double b)
        {
            return PlusWrappedGaussianOp.SumAverageConditional(a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusWrappedGaussianOp"]/message_doc[@name="SumAverageLogarithm(double, WrappedGaussian)"]/*'/>
        public static WrappedGaussian SumAverageLogarithm(double a, [SkipIfUniform] WrappedGaussian b)
        {
            return PlusWrappedGaussianOp.SumAverageConditional(a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusWrappedGaussianOp"]/message_doc[@name="AAverageLogarithm(WrappedGaussian, WrappedGaussian)"]/*'/>
        public static WrappedGaussian AAverageLogarithm([SkipIfUniform] WrappedGaussian sum, [SkipIfUniform] WrappedGaussian b)
        {
            if (sum.Period != b.Period)
                throw new ArgumentException("sum.Period (" + sum.Period + ") != b.Period (" + b.Period + ")");
            WrappedGaussian result = WrappedGaussian.Uniform(sum.Period);
            result.Gaussian = DoublePlusVmpOp.AAverageLogarithm(sum.Gaussian, b.Gaussian);
            result.Normalize();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusWrappedGaussianOp"]/message_doc[@name="AAverageLogarithm(WrappedGaussian, double)"]/*'/>
        public static WrappedGaussian AAverageLogarithm([SkipIfUniform] WrappedGaussian sum, double b)
        {
            WrappedGaussian result = WrappedGaussian.Uniform(sum.Period);
            result.Gaussian = DoublePlusVmpOp.AAverageLogarithm(sum.Gaussian, b);
            result.Normalize();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusWrappedGaussianOp"]/message_doc[@name="BAverageLogarithm(WrappedGaussian, WrappedGaussian)"]/*'/>
        public static WrappedGaussian BAverageLogarithm([SkipIfUniform] WrappedGaussian sum, [SkipIfUniform] WrappedGaussian a)
        {
            return AAverageLogarithm(sum, a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusWrappedGaussianOp"]/message_doc[@name="BAverageLogarithm(WrappedGaussian, double)"]/*'/>
        public static WrappedGaussian BAverageLogarithm([SkipIfUniform] WrappedGaussian sum, double a)
        {
            return AAverageLogarithm(sum, a);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusTruncatedGaussianOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Plus", typeof(double), typeof(double), Default = true)]
    [FactorMethod(new string[] { "A", "Sum", "B" }, typeof(Factor), "Difference", typeof(double), typeof(double), Default = true)]
    [Quality(QualityBand.Mature)]
    public static class PlusTruncatedGaussianOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusTruncatedGaussianOp"]/message_doc[@name="SumAverageConditional(double, TruncatedGaussian)"]/*'/>
        public static TruncatedGaussian SumAverageConditional(double a, [SkipIfUniform] TruncatedGaussian b)
        {
            Gaussian prior = b.ToGaussian();
            Gaussian post = DoublePlusOp.SumAverageConditional(a, prior);
            TruncatedGaussian result = b;
            result.Gaussian = post;
            result.LowerBound += a;
            result.UpperBound += a;
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusTruncatedGaussianOp"]/message_doc[@name="SumAverageConditional(TruncatedGaussian, double)"]/*'/>
        public static TruncatedGaussian SumAverageConditional([SkipIfUniform] TruncatedGaussian a, double b)
        {
            return SumAverageConditional(b, a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusTruncatedGaussianOp"]/message_doc[@name="SumAverageConditional(TruncatedGaussian, Gaussian)"]/*'/>
        public static Gaussian SumAverageConditional([SkipIfUniform] TruncatedGaussian a, [SkipIfUniform] Gaussian b)
        {
            return SumAverageConditional(b, a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusTruncatedGaussianOp"]/message_doc[@name="SumAverageConditional(Gaussian, TruncatedGaussian)"]/*'/>
        public static Gaussian SumAverageConditional([SkipIfUniform] Gaussian a, [SkipIfUniform] TruncatedGaussian b)
        {
            return DoublePlusOp.SumAverageConditional(a, b.ToGaussian());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusTruncatedGaussianOp"]/message_doc[@name="AAverageConditional(TruncatedGaussian, double)"]/*'/>
        public static TruncatedGaussian AAverageConditional([SkipIfUniform] TruncatedGaussian sum, double b)
        {
            Gaussian prior = sum.ToGaussian();
            Gaussian post = DoublePlusOp.AAverageConditional(prior, b);
            TruncatedGaussian result = sum;
            result.Gaussian = post;
            result.LowerBound -= b;
            result.UpperBound -= b;
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusTruncatedGaussianOp"]/message_doc[@name="AAverageConditional(double, TruncatedGaussian)"]/*'/>
        public static TruncatedGaussian AAverageConditional(double sum, [SkipIfUniform] TruncatedGaussian b)
        {
            Gaussian prior = b.ToGaussian();
            Gaussian post = DoublePlusOp.AAverageConditional(sum, prior);
            return new TruncatedGaussian(post, sum - b.UpperBound, sum - b.LowerBound);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusTruncatedGaussianOp"]/message_doc[@name="AAverageConditional(Gaussian, Gaussian)"]/*'/>
        public static TruncatedGaussian AAverageConditional([SkipIfUniform] Gaussian sum, [SkipIfUniform] Gaussian b)
        {
            return TruncatedGaussian.FromGaussian(DoublePlusOp.AAverageConditional(sum, b));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusTruncatedGaussianOp"]/message_doc[@name="AAverageConditional(Gaussian, Gaussian)"]/*'/>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian sum, [SkipIfUniform] TruncatedGaussian b)
        {
            return DoublePlusOp.AAverageConditional(sum, b.ToGaussian());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusTruncatedGaussianOp"]/message_doc[@name="BAverageConditional(Gaussian, Gaussian)"]/*'/>
        public static TruncatedGaussian BAverageConditional([SkipIfUniform] Gaussian sum, [SkipIfUniform] Gaussian a)
        {
            return AAverageConditional(sum, a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusTruncatedGaussianOp"]/message_doc[@name="BAverageConditional(Gaussian, Gaussian)"]/*'/>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian sum, [SkipIfUniform] TruncatedGaussian a)
        {
            return AAverageConditional(sum, a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusTruncatedGaussianOp"]/message_doc[@name="BAverageConditional(TruncatedGaussian, double)"]/*'/>
        public static TruncatedGaussian BAverageConditional([SkipIfUniform] TruncatedGaussian sum, double a)
        {
            return AAverageConditional(sum, a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusTruncatedGaussianOp"]/message_doc[@name="BAverageConditional(double, TruncatedGaussian)"]/*'/>
        public static TruncatedGaussian BAverageConditional(double sum, [SkipIfUniform] TruncatedGaussian a)
        {
            return AAverageConditional(sum, a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusTruncatedGaussianOp"]/message_doc[@name="LogAverageFactor(double, TruncatedGaussian, double)"]/*'/>
        public static double LogAverageFactor(double sum, [SkipIfUniform] TruncatedGaussian a, double b)
        {
            TruncatedGaussian to_sum = SumAverageConditional(a, b);
            return to_sum.GetLogProb(sum);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusTruncatedGaussianOp"]/message_doc[@name="LogAverageFactor(double, double, TruncatedGaussian)"]/*'/>
        public static double LogAverageFactor(double sum, double a, [SkipIfUniform] TruncatedGaussian b)
        {
            TruncatedGaussian to_sum = SumAverageConditional(a, b);
            return to_sum.GetLogProb(sum);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusTruncatedGaussianOp"]/message_doc[@name="LogAverageFactor(TruncatedGaussian, double, double)"]/*'/>
        public static double LogAverageFactor([SkipIfUniform] TruncatedGaussian sum, double a, double b)
        {
            return sum.GetLogProb(Factor.Plus(a, b));
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusVmpOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Plus", typeof(double), typeof(double))]
    [Quality(QualityBand.Stable)]
    public static class DoublePlusVmpOp
    {
        // ----------------------------------------------------------------------------------------------------------------------
        // Gaussian
        // ----------------------------------------------------------------------------------------------------------------------

        internal const string NotSupportedMessage = "Variational Message Passing does not support a Plus factor with fixed output and two random inputs.";

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusVmpOp"]/message_doc[@name="AverageLogFactor(Gaussian)"]/*'/>
        [Skip]
        public static double AverageLogFactor(Gaussian sum)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusVmpOp"]/message_doc[@name="AverageLogFactor(double, double, double)"]/*'/>
        public static double AverageLogFactor(double sum, double a, double b)
        {
            return DoublePlusOp.LogAverageFactor(sum, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusVmpOp"]/message_doc[@name="AverageLogFactor(double, Gaussian)"]/*'/>
        [Skip]
        public static double AverageLogFactor(double sum, Gaussian a)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusVmpOp"]/message_doc[@name="AverageLogFactor(double, double, Gaussian)"]/*'/>
        [Skip]
        public static double AverageLogFactor(double sum, double a, Gaussian b)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusVmpOp"]/message_doc[@name="SumAverageLogarithm(Gaussian, Gaussian)"]/*'/>
        public static Gaussian SumAverageLogarithm([SkipIfUniform] Gaussian a, [SkipIfUniform] Gaussian b)
        {
            return DoublePlusOp.SumAverageConditional(a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusVmpOp"]/message_doc[@name="SumAverageLogarithm(double, Gaussian)"]/*'/>
        public static Gaussian SumAverageLogarithm(double a, [SkipIfUniform] Gaussian b)
        {
            return DoublePlusOp.SumAverageConditional(a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusVmpOp"]/message_doc[@name="SumAverageLogarithm(Gaussian, double)"]/*'/>
        public static Gaussian SumAverageLogarithm([SkipIfUniform] Gaussian a, double b)
        {
            return DoublePlusOp.SumAverageConditional(a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusVmpOp"]/message_doc[@name="SumAverageLogarithm(double, double)"]/*'/>
        public static Gaussian SumAverageLogarithm(double a, double b)
        {
            return DoublePlusOp.SumAverageConditional(a, b);
        }

        [Skip]
        public static Gaussian SumDeriv(Gaussian a, double b)
        {
            return Gaussian.Uniform();
        }

        [Skip]
        public static Gaussian SumDeriv(double a, Gaussian b)
        {
            return Gaussian.Uniform();
        }

        public static Gaussian SumDeriv([SkipIfUniform, Proper] Gaussian Sum, Gaussian a, Gaussian a_deriv, Gaussian to_a, [Proper] Gaussian b, Gaussian b_deriv, Gaussian to_b)
        {
            double sa = a_deriv.MeanTimesPrecision;
            double ta = a.MeanTimesPrecision - sa * to_a.MeanTimesPrecision;
            double sb = b_deriv.MeanTimesPrecision;
            double tb = b.MeanTimesPrecision - sb * to_b.MeanTimesPrecision;
            double va = 1 / a.Precision;
            double vb = 1 / b.Precision;
            double aa = va * sa * Sum.Precision;
            double ab = vb * sb * Sum.Precision;
            double ba = va * sa;
            double bb = vb * sb;
            double deriv = (ba * (1 - ab) + bb * (1 - aa)) / (1 - aa * ab);
            return Gaussian.FromNatural(deriv - 1, 0);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusVmpOp"]/message_doc[@name="AAverageLogarithm(Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian)"]/*'/>
        public static Gaussian AAverageLogarithm(
            [SkipIfUniform, Proper] Gaussian Sum, Gaussian a, Gaussian a_deriv, Gaussian to_a, [Proper] Gaussian b, Gaussian b_deriv, Gaussian to_b)
        {
            double sa = a_deriv.MeanTimesPrecision + 1;
            double ta = a.MeanTimesPrecision - sa * to_a.MeanTimesPrecision;
            double sb = b_deriv.MeanTimesPrecision + 1;
            double tb = b.MeanTimesPrecision - sb * to_b.MeanTimesPrecision;
            double va = 1 / a.Precision;
            double vb = 1 / b.Precision;
            double aa = va * sa * Sum.Precision;
            double ab = vb * sb * Sum.Precision;
            double ba = va * (ta + sa * Sum.MeanTimesPrecision);
            double bb = vb * (tb + sb * Sum.MeanTimesPrecision);
            double ma = (ba - bb * aa) / (1 - aa * ab);
            double mu = (ma / va - ta) / sa;
            return new Gaussian(mu / Sum.Precision, 1 / Sum.Precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusVmpOp"]/message_doc[@name="AAverageLogarithm(Gaussian, Gaussian)"]/*'/>
        public static Gaussian AAverageLogarithm([SkipIfUniform, Proper] Gaussian Sum, [Proper] Gaussian b)
        {
            if (Sum.Precision == 0)
            {
                return Sum;
            }
            else if (Sum.Precision < 0)
            {
                throw new ImproperMessageException(Sum);
            }
            else if (Sum.IsPointMass)
            {
                return Gaussian.PointMass(Sum.Point - b.GetMean());
            }
            else
            {
                // p(a|sum,b) = N(E[sum] - E[b], var(sum) )
                Gaussian result = new Gaussian();
                result.MeanTimesPrecision = Sum.MeanTimesPrecision - b.GetMean() * Sum.Precision;
                result.Precision = Sum.Precision;
                return result;
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusVmpOp"]/message_doc[@name="AAverageLogarithm(double, Gaussian)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Gaussian AAverageLogarithm(double Sum, [Proper] Gaussian b)
        {
            // Throw an exception rather than return a meaningless point mass.
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusVmpOp"]/message_doc[@name="AAverageLogarithm(Gaussian, double)"]/*'/>
        public static Gaussian AAverageLogarithm([SkipIfUniform, Proper] Gaussian Sum, double b)
        {
            return AAverageLogarithm(Sum, Gaussian.PointMass(b));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusVmpOp"]/message_doc[@name="AAverageLogarithm(double, double)"]/*'/>
        public static Gaussian AAverageLogarithm(double Sum, double b)
        {
            return DoublePlusOp.AAverageConditional(Sum, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusVmpOp"]/message_doc[@name="BAverageLogarithm(Gaussian, Gaussian)"]/*'/>
        public static Gaussian BAverageLogarithm([SkipIfUniform, Proper] Gaussian Sum, [Proper] Gaussian a)
        {
            return AAverageLogarithm(Sum, a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusVmpOp"]/message_doc[@name="BAverageLogarithm(double, Gaussian)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Gaussian BAverageLogarithm(double Sum, [Proper] Gaussian a)
        {
            return AAverageLogarithm(Sum, a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusVmpOp"]/message_doc[@name="BAverageLogarithm(Gaussian, double)"]/*'/>
        public static Gaussian BAverageLogarithm([SkipIfUniform, Proper] Gaussian Sum, double a)
        {
            return AAverageLogarithm(Sum, a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusVmpOp"]/message_doc[@name="BAverageLogarithm(double, double)"]/*'/>
        public static Gaussian BAverageLogarithm(double Sum, double a)
        {
            return AAverageLogarithm(Sum, a);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleMinusVmpOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Difference", typeof(double), typeof(double))]
    [Quality(QualityBand.Stable)]
    public static class DoubleMinusVmpOp
    {
        //-- VMP ----------------------------------------------------------------------------------------------

        internal const string NotSupportedMessage = "Variational Message Passing does not support a Minus factor with fixed output and two random inputs.";

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleMinusVmpOp"]/message_doc[@name="AverageLogFactor()"]/*'/>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleMinusVmpOp"]/message_doc[@name="DifferenceAverageLogarithm(Gaussian, Gaussian)"]/*'/>
        public static Gaussian DifferenceAverageLogarithm([SkipIfUniform] Gaussian a, [SkipIfUniform] Gaussian b)
        {
            return DoublePlusOp.AAverageConditional(a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleMinusVmpOp"]/message_doc[@name="DifferenceAverageLogarithm(double, Gaussian)"]/*'/>
        public static Gaussian DifferenceAverageLogarithm(double a, [SkipIfUniform] Gaussian b)
        {
            return DoublePlusOp.AAverageConditional(a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleMinusVmpOp"]/message_doc[@name="DifferenceAverageLogarithm(Gaussian, double)"]/*'/>
        public static Gaussian DifferenceAverageLogarithm([SkipIfUniform] Gaussian a, double b)
        {
            return DoublePlusOp.AAverageConditional(a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleMinusVmpOp"]/message_doc[@name="DifferenceAverageLogarithm(double, double)"]/*'/>
        public static Gaussian DifferenceAverageLogarithm(double a, double b)
        {
            return DoublePlusOp.AAverageConditional(a, b);
        }
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleMinusVmpOp"]/message_doc[@name="AAverageLogarithm(Gaussian, Gaussian)"]/*'/>
        public static Gaussian AAverageLogarithm([SkipIfUniform, Proper] Gaussian Difference, [Proper] Gaussian b)
        {
            Gaussian result = new Gaussian();
            if (Difference.IsUniform())
            {
                result.SetToUniform();
            }
            else if (Difference.Precision < 0)
            {
                throw new ImproperMessageException(Difference);
            }
            else
            {
                // p(a|diff,b) = N(E[diff] + E[b], var(diff) )
                double ms, vs;
                double mb = b.GetMean();
                Difference.GetMeanAndVariance(out ms, out vs);
                result.SetMeanAndVariance(ms + mb, vs);
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleMinusVmpOp"]/message_doc[@name="AAverageLogarithm(Gaussian, double)"]/*'/>
        public static Gaussian AAverageLogarithm([SkipIfUniform, Proper] Gaussian Difference, double b)
        {
            return AAverageLogarithm(Difference, Gaussian.PointMass(b));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleMinusVmpOp"]/message_doc[@name="AAverageLogarithm(double, Gaussian)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Gaussian AAverageLogarithm(double Difference, [Proper] Gaussian b)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleMinusVmpOp"]/message_doc[@name="AAverageLogarithm(double, double)"]/*'/>
        public static Gaussian AAverageLogarithm(double Difference, double b)
        {
            return Gaussian.PointMass(Difference + b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleMinusVmpOp"]/message_doc[@name="BAverageLogarithm(Gaussian, Gaussian)"]/*'/>
        public static Gaussian BAverageLogarithm([SkipIfUniform, Proper] Gaussian Difference, [Proper] Gaussian a)
        {
            Gaussian result = new Gaussian();
            if (Difference.IsUniform())
            {
                result.SetToUniform();
            }
            else if (Difference.Precision < 0)
            {
                throw new ImproperMessageException(Difference);
            }
            else
            {
                // p(b|diff,a) = N(E[a] - E[diff], var(diff) )
                double ms, vs;
                double ma = a.GetMean();
                Difference.GetMeanAndVariance(out ms, out vs);
                result.SetMeanAndVariance(ma - ms, vs);
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleMinusVmpOp"]/message_doc[@name="BAverageLogarithm(Gaussian, double)"]/*'/>
        public static Gaussian BAverageLogarithm([SkipIfUniform, Proper] Gaussian Difference, double a)
        {
            return BAverageLogarithm(Difference, Gaussian.PointMass(a));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleMinusVmpOp"]/message_doc[@name="BAverageLogarithm(double, Gaussian)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Gaussian BAverageLogarithm(double Difference, [Proper] Gaussian a)
        {
            // Throw an exception rather than return a meaningless point mass.
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleMinusVmpOp"]/message_doc[@name="BAverageLogarithm(double, double)"]/*'/>
        public static Gaussian BAverageLogarithm(double Difference, double a)
        {
            return Gaussian.PointMass(a - Difference);
        }
    }

    /// <summary>
    /// Special factors used to modify the inference schedule.
    /// </summary>
    [Hidden]
    public static class NoSkip
    {
        /// <summary>
        /// Returns the sum of the two arguments: (a + b).
        /// </summary>
        /// <returns>a+b</returns>
        [ParameterNames("Sum", "A", "B")]
        [HasUnitDerivative]
        public static double Plus(double a, double b)
        {
            return Factor.Plus(a, b);
        }

        /// <summary>
        /// Returns the difference of the two arguments: (a - b).
        /// </summary>
        /// <returns>a-b</returns>
        [HasUnitDerivative]
        public static double Difference(double a, double b)
        {
            return a - b;
        }
    }

    [FactorMethod(typeof(NoSkip), "Plus", typeof(double), typeof(double), Default = true)]
    [FactorMethod(new string[] { "A", "Sum", "B" }, typeof(NoSkip), "Difference", typeof(double), typeof(double), Default = true)]
    [Quality(QualityBand.Experimental)]
    public static class DoublePlusOp_NoSkip
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoublePlusOp"]/message_doc[@name="SumAverageConditional(Gaussian, Gaussian)"]/*'/>
        public static Gaussian SumAverageConditional([NoInit] Gaussian a, [NoInit] Gaussian b)
        {
            return DoublePlusOp.SumAverageConditional(a, b);
        }

        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Sum, Gaussian b)
        {
            return DoublePlusOp.AAverageConditional(Sum, b);
        }

        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Sum, Gaussian a)
        {
            return DoublePlusOp.BAverageConditional(Sum, a);
        }
    }

    [Hidden]
    public static class Diode
    {
        /// <summary>
        /// Creates a copy of a value, such that the copy cannot participate in a backward sequential loop.
        /// Used by SumForwardBackwardTest2.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value"></param>
        /// <returns></returns>
        [HasUnitDerivative]
        public static T Copy<T>(T value)
        {
            return value;
        }
    }

    [FactorMethod(typeof(Diode), "Copy<>")]
    [Quality(QualityBand.Experimental)]
    public static class DiodeCopyOp<T>
    {
        public static TDist CopyAverageConditional<TDist>([Diode, IsReturned] TDist value)
            where TDist : IDistribution<T>
        {
            return value;
        }

        public static TDist ValueAverageConditional<TDist>([Diode, IsReturned] TDist copy)
            where TDist : IDistribution<T>
        {
            return copy;
        }
    }
}

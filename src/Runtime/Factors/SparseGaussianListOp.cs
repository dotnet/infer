// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/doc/*'/>
    [FactorMethod(new string[] { "sample", "mean", "precision" }, typeof(SparseGaussianList), "Sample", typeof(ISparseList<double>), typeof(ISparseList<double>))]
    [Quality(QualityBand.Stable)]
    public static class SparseGaussianListOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="SampleAverageConditionalInit(ISparseList{double})"]/*'/>
        [Skip]
        public static SparseGaussianList SampleAverageConditionalInit([IgnoreDependency] ISparseList<double> mean)
        {
            return SparseGaussianList.FromSize(mean.Count);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="SampleAverageConditionalInit(SparseGaussianList)"]/*'/>
        [Skip]
        public static SparseGaussianList SampleAverageConditionalInit([IgnoreDependency] SparseGaussianList mean)
        {
            return (SparseGaussianList)mean.Clone();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="SampleAverageConditional(ISparseList{double}, ISparseList{double}, SparseGaussianList)"]/*'/>
        public static SparseGaussianList SampleAverageConditional(ISparseList<double> mean, ISparseList<double> precision, SparseGaussianList result)
        {
            result.SetToFunction(mean, precision, (m, p) => GaussianOp.SampleAverageConditional(m, p));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="MeanAverageConditional(ISparseList{double}, ISparseList{double}, SparseGaussianList)"]/*'/>
        public static SparseGaussianList MeanAverageConditional(ISparseList<double> sample, ISparseList<double> precision, SparseGaussianList result)
        {
            result.SetToFunction(sample, precision, (s, p) => GaussianOp.MeanAverageConditional(s, p));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="PrecisionAverageConditional(ISparseList{double}, ISparseList{double}, SparseGammaList)"]/*'/>
        public static SparseGammaList PrecisionAverageConditional(ISparseList<double> sample, ISparseList<double> mean, SparseGammaList result)
        {
            result.SetToFunction(sample, mean, (s, m) => GaussianOp.PrecisionAverageConditional(s, m));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="SampleAverageConditional(SparseGaussianList, ISparseList{double}, SparseGaussianList)"]/*'/>
        public static SparseGaussianList SampleAverageConditional([SkipIfUniform] SparseGaussianList mean, ISparseList<double> precision, SparseGaussianList result)
        {
            result.SetToFunction(mean, precision, (m, p) => GaussianOp.SampleAverageConditional(m, p));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="MeanAverageConditional(SparseGaussianList, ISparseList{double}, SparseGaussianList)"]/*'/>
        public static SparseGaussianList MeanAverageConditional([SkipIfUniform] SparseGaussianList sample, ISparseList<double> precision, SparseGaussianList result)
        {
            result.SetToFunction(sample, precision, (s, p) => GaussianOp.MeanAverageConditional(s, p));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="SampleAverageConditional(SparseGaussianList, SparseGaussianList, SparseGammaList, SparseGammaList, SparseGaussianList)"]/*'/>
        public static SparseGaussianList SampleAverageConditional(
            SparseGaussianList sample, [SkipIfUniform] SparseGaussianList mean, [SkipIfUniform] SparseGammaList precision, SparseGammaList to_precision, SparseGaussianList result)
        {
            result.SetToFunction(sample, mean, precision, to_precision, (s, m, p, tp) => GaussianOp.SampleAverageConditional(s, m, p, tp));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="SampleAverageConditional(SparseGaussianList, ISparseList{double}, SparseGammaList, SparseGammaList, SparseGaussianList)"]/*'/>
        public static SparseGaussianList SampleAverageConditional(
            SparseGaussianList sample, ISparseList<double> mean, [SkipIfUniform] SparseGammaList precision, SparseGammaList to_precision, SparseGaussianList result)
        {
            result.SetToFunction(sample, mean, precision, to_precision, (s, m, p, tp) => GaussianOp.SampleAverageConditional(s, m, p, tp));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="MeanAverageConditional(SparseGaussianList, SparseGaussianList, SparseGammaList, SparseGammaList, SparseGaussianList)"]/*'/>
        public static SparseGaussianList MeanAverageConditional(
            [SkipIfUniform] SparseGaussianList sample, [SkipIfUniform] SparseGaussianList mean, [SkipIfUniform] SparseGammaList precision, SparseGammaList to_precision, SparseGaussianList result)
        {
            result.SetToFunction(sample, mean, precision, to_precision, (s, m, p, tp) => GaussianOp.MeanAverageConditional(s, m, p, tp));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="MeanAverageConditional(ISparseList{double}, SparseGaussianList, SparseGammaList, SparseGammaList, SparseGaussianList)"]/*'/>
        public static SparseGaussianList MeanAverageConditional(
            ISparseList<double> sample, [SkipIfUniform] SparseGaussianList mean, [SkipIfUniform] SparseGammaList precision, SparseGammaList to_precision, SparseGaussianList result)
        {
            result.SetToFunction(sample, mean, precision, to_precision, (s, m, p, tp) => GaussianOp.MeanAverageConditional(s, m, p, tp));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="PrecisionAverageConditional(ISparseList{double}, SparseGaussianList, SparseGammaList, SparseGammaList)"]/*'/>
        public static SparseGammaList PrecisionAverageConditional(
            ISparseList<double> sample, [SkipIfUniform] SparseGaussianList mean, [SkipIfUniform] SparseGammaList precision, SparseGammaList result)
        {
            result.SetToFunction(sample, mean, precision, (s, m, p) => GaussianOp.PrecisionAverageConditional_slow(Gaussian.PointMass(s), m, p));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="PrecisionAverageConditional(SparseGaussianList, ISparseList{double}, SparseGammaList, SparseGammaList)"]/*'/>
        public static SparseGammaList PrecisionAverageConditional(
            [SkipIfUniform] SparseGaussianList sample, ISparseList<double> mean, [SkipIfUniform] SparseGammaList precision, SparseGammaList result)
        {
            result.SetToFunction(sample, mean, precision, (s, m, p) => GaussianOp.PrecisionAverageConditional_slow(s, Gaussian.PointMass(m), p));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="PrecisionAverageConditional(SparseGaussianList, SparseGaussianList, SparseGammaList, SparseGammaList)"]/*'/>
        public static SparseGammaList PrecisionAverageConditional(
            [SkipIfUniform] SparseGaussianList sample, [SkipIfUniform] SparseGaussianList mean, [SkipIfUniform] SparseGammaList precision, SparseGammaList result)
        {
            result.SetToFunction(sample, mean, precision, (s, m, p) => GaussianOp.PrecisionAverageConditional_slow(s, m, p));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="LogAverageFactor(ISparseList{double}, ISparseList{double}, ISparseList{double})"]/*'/>
        public static double LogAverageFactor(ISparseList<double> sample, ISparseList<double> mean, ISparseList<double> precision)
        {
            Func<double, double, double, double> f = GaussianOp.LogAverageFactor;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="LogAverageFactor(SparseGaussianList, SparseGaussianList, ISparseList{double})"]/*'/>
        public static double LogAverageFactor([SkipIfUniform] SparseGaussianList sample, [SkipIfUniform] SparseGaussianList mean, ISparseList<double> precision)
        {
            Func<Gaussian, Gaussian, double, double> f = GaussianOp.LogAverageFactor;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="LogAverageFactor(SparseGaussianList, ISparseList{double}, ISparseList{double})"]/*'/>
        public static double LogAverageFactor([SkipIfUniform] SparseGaussianList sample, ISparseList<double> mean, ISparseList<double> precision)
        {
            Func<Gaussian, double, double, double> f = GaussianOp.LogAverageFactor;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="LogAverageFactor(ISparseList{double}, SparseGaussianList, ISparseList{double})"]/*'/>
        public static double LogAverageFactor(ISparseList<double> sample, [SkipIfUniform] SparseGaussianList mean, ISparseList<double> precision)
        {
            Func<double, Gaussian, double, double> f = GaussianOp.LogAverageFactor;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="LogAverageFactor(ISparseList{double}, ISparseList{double}, SparseGammaList)"]/*'/>
        public static double LogAverageFactor(ISparseList<double> sample, ISparseList<double> mean, [SkipIfUniform] SparseGammaList precision)
        {
            Func<double, double, Gamma, double> f = GaussianOp.LogAverageFactor;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="LogAverageFactor(ISparseList{double}, SparseGaussianList, SparseGammaList, SparseGammaList)"]/*'/>
        public static double LogAverageFactor(
            ISparseList<double> sample, [SkipIfUniform] SparseGaussianList mean, [SkipIfUniform] SparseGammaList precision, SparseGammaList to_precision)
        {
            Func<double, Gaussian, Gamma, Gamma, double> f = GaussianOp.LogAverageFactor;
            return f.Map(sample, mean, precision, to_precision).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="LogAverageFactor(SparseGaussianList, ISparseList{double}, SparseGammaList, SparseGammaList)"]/*'/>
        public static double LogAverageFactor(
            [SkipIfUniform] SparseGaussianList sample, ISparseList<double> mean, [SkipIfUniform] SparseGammaList precision, SparseGammaList to_precision)
        {
            Func<Gaussian, double, Gamma, Gamma, double> f = GaussianOp.LogAverageFactor;
            return f.Map(sample, mean, precision, to_precision).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="LogAverageFactor(SparseGaussianList, SparseGaussianList, SparseGammaList, SparseGammaList)"]/*'/>
        public static double LogAverageFactor(
            [SkipIfUniform] SparseGaussianList sample, [SkipIfUniform] SparseGaussianList mean, [SkipIfUniform] SparseGammaList precision, SparseGammaList to_precision)
        {
            Func<Gaussian, Gaussian, Gamma, Gamma, double> f = GaussianOp.LogAverageFactor;
            return f.Map(sample, mean, precision, to_precision).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="LogEvidenceRatio(ISparseList{double}, ISparseList{double}, ISparseList{double})"]/*'/>
        public static double LogEvidenceRatio(ISparseList<double> sample, ISparseList<double> mean, ISparseList<double> precision)
        {
            Func<double, double, double, double> f = GaussianOp.LogEvidenceRatio;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="LogEvidenceRatio(SparseGaussianList, SparseGaussianList, ISparseList{double})"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(SparseGaussianList sample, SparseGaussianList mean, ISparseList<double> precision)
        {
            Func<Gaussian, Gaussian, double, double> f = GaussianOp.LogEvidenceRatio;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="LogEvidenceRatio(SparseGaussianList, ISparseList{double}, ISparseList{double})"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(SparseGaussianList sample, ISparseList<double> mean, ISparseList<double> precision)
        {
            Func<Gaussian, double, double, double> f = GaussianOp.LogEvidenceRatio;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="LogEvidenceRatio(ISparseList{double}, SparseGaussianList, ISparseList{double})"]/*'/>
        public static double LogEvidenceRatio(ISparseList<double> sample, SparseGaussianList mean, ISparseList<double> precision)
        {
            Func<double, Gaussian, double, double> f = GaussianOp.LogEvidenceRatio;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="LogEvidenceRatio(ISparseList{double}, ISparseList{double}, SparseGammaList)"]/*'/>
        public static double LogEvidenceRatio(ISparseList<double> sample, ISparseList<double> mean, [SkipIfUniform] SparseGammaList precision)
        {
            Func<double, double, Gamma, double> f = GaussianOp.LogEvidenceRatio;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="LogEvidenceRatio(ISparseList{double}, SparseGaussianList, SparseGammaList, SparseGammaList)"]/*'/>
        public static double LogEvidenceRatio(
            ISparseList<double> sample, [SkipIfUniform] SparseGaussianList mean, [SkipIfUniform] SparseGammaList precision, SparseGammaList to_precision)
        {
            Func<double, Gaussian, Gamma, Gamma, double> f = GaussianOp.LogEvidenceRatio;
            return f.Map(sample, mean, precision, to_precision).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="LogEvidenceRatio(SparseGaussianList, SparseGaussianList, SparseGammaList, SparseGaussianList, SparseGammaList)"]/*'/>
        public static double LogEvidenceRatio(
            [SkipIfUniform] SparseGaussianList sample, [SkipIfUniform] SparseGaussianList mean, [SkipIfUniform] SparseGammaList precision, [Fresh] SparseGaussianList to_sample, SparseGammaList to_precision)
        {
            Func<Gaussian, Gaussian, Gamma, Gaussian, Gamma, double> f = GaussianOp.LogEvidenceRatio;
            return f.Map(sample, mean, precision, to_sample, to_precision).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="LogEvidenceRatio(SparseGaussianList, ISparseList{double}, SparseGammaList, SparseGaussianList, SparseGammaList)"]/*'/>
        public static double LogEvidenceRatio(
            [SkipIfUniform] SparseGaussianList sample, ISparseList<double> mean, [SkipIfUniform] SparseGammaList precision, [Fresh] SparseGaussianList to_sample, SparseGammaList to_precision)
        {
            Func<Gaussian, double, Gamma, Gaussian, Gamma, double> f = GaussianOp.LogEvidenceRatio;
            return f.Map(sample, mean, precision, to_sample, to_precision).EnumerableSum(x => x);
        }

        //-- VMP ------------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="SampleAverageLogarithmInit(ISparseList{double})"]/*'/>
        [Skip]
        public static SparseGaussianList SampleAverageLogarithmInit([IgnoreDependency] ISparseList<double> mean)
        {
            return SparseGaussianList.FromSize(mean.Count);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="SampleAverageLogarithmInit(SparseGaussianList)"]/*'/>
        [Skip]
        public static SparseGaussianList SampleAverageLogarithmInit([IgnoreDependency] SparseGaussianList mean)
        {
            return (SparseGaussianList)mean.Clone();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="SampleAverageLogarithm(ISparseList{double}, ISparseList{double}, SparseGaussianList)"]/*'/>
        public static SparseGaussianList SampleAverageLogarithm(ISparseList<double> mean, ISparseList<double> precision, SparseGaussianList result)
        {
            result.SetToFunction(mean, precision, (m, p) => GaussianOp.SampleAverageLogarithm(m, p));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="SampleAverageLogarithm(SparseGaussianList, ISparseList{double}, SparseGaussianList)"]/*'/>
        public static SparseGaussianList SampleAverageLogarithm([Proper] SparseGaussianList mean, ISparseList<double> precision, SparseGaussianList result)
        {
            result.SetToFunction(mean, precision, (m, p) => GaussianOp.SampleAverageLogarithm(m, p));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="MeanAverageLogarithm(ISparseList{double}, ISparseList{double}, SparseGaussianList)"]/*'/>
        public static SparseGaussianList MeanAverageLogarithm(ISparseList<double> sample, ISparseList<double> precision, SparseGaussianList result)
        {
            result.SetToFunction(sample, precision, (s, p) => GaussianOp.MeanAverageLogarithm(s, p));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="MeanAverageLogarithm(SparseGaussianList, ISparseList{double}, SparseGaussianList)"]/*'/>
        public static SparseGaussianList MeanAverageLogarithm([Proper] SparseGaussianList sample, ISparseList<double> precision, SparseGaussianList result)
        {
            result.SetToFunction(sample, precision, (s, p) => GaussianOp.MeanAverageLogarithm(s, p));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="SampleAverageLogarithm(SparseGaussianList, SparseGammaList, SparseGaussianList)"]/*'/>
        public static SparseGaussianList SampleAverageLogarithm([Proper] SparseGaussianList mean, [Proper] SparseGammaList precision, SparseGaussianList result)
        {
            result.SetToFunction(mean, precision, (m, p) => GaussianOp.SampleAverageLogarithm(m, p));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="MeanAverageLogarithm(SparseGaussianList, SparseGammaList, SparseGaussianList)"]/*'/>
        public static SparseGaussianList MeanAverageLogarithm([Proper] SparseGaussianList sample, [Proper] SparseGammaList precision, SparseGaussianList result)
        {
            result.SetToFunction(sample, precision, (s, p) => GaussianOp.MeanAverageLogarithm(s, p));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="SampleAverageLogarithm(ISparseList{double}, SparseGammaList, SparseGaussianList)"]/*'/>
        public static SparseGaussianList SampleAverageLogarithm(ISparseList<double> mean, [Proper] SparseGammaList precision, SparseGaussianList result)
        {
            result.SetToFunction(mean, precision, (m, p) => GaussianOp.SampleAverageLogarithm(m, p));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="MeanAverageLogarithm(ISparseList{double}, SparseGammaList, SparseGaussianList)"]/*'/>
        public static SparseGaussianList MeanAverageLogarithm(ISparseList<double> sample, [Proper] SparseGammaList precision, SparseGaussianList result)
        {
            result.SetToFunction(sample, precision, (s, p) => GaussianOp.MeanAverageLogarithm(s, p));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="PrecisionAverageLogarithm(ISparseList{double}, ISparseList{double}, SparseGammaList)"]/*'/>
        public static SparseGammaList PrecisionAverageLogarithm(ISparseList<double> sample, ISparseList<double> mean, SparseGammaList result)
        {
            result.SetToFunction(sample, mean, (s, m) => GaussianOp.PrecisionAverageLogarithm(s, m));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="PrecisionAverageLogarithm(SparseGaussianList, SparseGaussianList, SparseGammaList)"]/*'/>
        public static SparseGammaList PrecisionAverageLogarithm([Proper] SparseGaussianList sample, [Proper] SparseGaussianList mean, SparseGammaList result)
        {
            result.SetToFunction(sample, mean, (s, m) => GaussianOp.PrecisionAverageLogarithm(s, m));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="PrecisionAverageLogarithm(SparseGaussianList, ISparseList{double}, SparseGammaList)"]/*'/>
        public static SparseGammaList PrecisionAverageLogarithm([Proper] SparseGaussianList sample, ISparseList<double> mean, SparseGammaList result)
        {
            result.SetToFunction(sample, mean, (s, m) => GaussianOp.PrecisionAverageLogarithm(s, m));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="PrecisionAverageLogarithm(ISparseList{double}, SparseGaussianList, SparseGammaList)"]/*'/>
        public static SparseGammaList PrecisionAverageLogarithm(ISparseList<double> sample, [Proper] SparseGaussianList mean, SparseGammaList result)
        {
            result.SetToFunction(sample, mean, (s, m) => GaussianOp.PrecisionAverageLogarithm(s, m));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="AverageLogFactor(SparseGaussianList, SparseGaussianList, SparseGammaList)"]/*'/>
        public static double AverageLogFactor([Proper] SparseGaussianList sample, [Proper] SparseGaussianList mean, [Proper] SparseGammaList precision)
        {
            Func<Gaussian, Gaussian, Gamma, double> f = GaussianOp.AverageLogFactor;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="AverageLogFactor(ISparseList{double}, ISparseList{double}, SparseGammaList)"]/*'/>
        public static double AverageLogFactor(ISparseList<double> sample, ISparseList<double> mean, [Proper] SparseGammaList precision)
        {
            Func<double, double, Gamma, double> f = GaussianOp.AverageLogFactor;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="AverageLogFactor(ISparseList{double}, ISparseList{double}, ISparseList{double})"]/*'/>
        public static double AverageLogFactor(ISparseList<double> sample, ISparseList<double> mean, ISparseList<double> precision)
        {
            Func<double, double, double, double> f = GaussianOp.AverageLogFactor;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="AverageLogFactor(SparseGaussianList, ISparseList{double}, ISparseList{double})"]/*'/>
        public static double AverageLogFactor([Proper] SparseGaussianList sample, ISparseList<double> mean, ISparseList<double> precision)
        {
            Func<Gaussian, double, double, double> f = GaussianOp.AverageLogFactor;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="AverageLogFactor(ISparseList{double}, SparseGaussianList, ISparseList{double})"]/*'/>
        public static double AverageLogFactor(ISparseList<double> sample, [Proper] SparseGaussianList mean, ISparseList<double> precision)
        {
            Func<double, Gaussian, double, double> f = GaussianOp.AverageLogFactor;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="AverageLogFactor(ISparseList{double}, SparseGaussianList, SparseGammaList)"]/*'/>
        public static double AverageLogFactor(ISparseList<double> sample, [Proper] SparseGaussianList mean, [Proper] SparseGammaList precision)
        {
            Func<double, Gaussian, Gamma, double> f = GaussianOp.AverageLogFactor;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="AverageLogFactor(SparseGaussianList, ISparseList{double}, SparseGammaList)"]/*'/>
        public static double AverageLogFactor([Proper] SparseGaussianList sample, ISparseList<double> mean, [Proper] SparseGammaList precision)
        {
            Func<Gaussian, double, Gamma, double> f = GaussianOp.AverageLogFactor;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGaussianListOp"]/message_doc[@name="AverageLogFactor(SparseGaussianList, SparseGaussianList, ISparseList{double})"]/*'/>
        public static double AverageLogFactor([Proper] SparseGaussianList sample, [Proper] SparseGaussianList mean, ISparseList<double> precision)
        {
            Func<Gaussian, Gaussian, double, double> f = GaussianOp.AverageLogFactor;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }
    }
}

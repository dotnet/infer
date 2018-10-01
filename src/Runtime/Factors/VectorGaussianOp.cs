// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/doc/*'/>
    [FactorMethod(typeof(VectorGaussian), "Sample", typeof(Vector), typeof(PositiveDefiniteMatrix))]
    [FactorMethod(new string[] { "sample", "mean", "precision" }, typeof(Factor), "VectorGaussian")]
    [Buffers("SampleMean", "SampleVariance", "MeanMean", "MeanVariance", "PrecisionMean", "PrecisionMeanLogDet")]
    [Quality(QualityBand.Stable)]
    public static class VectorGaussianOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="SampleVarianceInit(VectorGaussian)"]/*'/>
        [Skip]
        public static PositiveDefiniteMatrix SampleVarianceInit([IgnoreDependency] VectorGaussian Sample)
        {
            return new PositiveDefiniteMatrix(Sample.Dimension, Sample.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="SampleVariance(VectorGaussian, PositiveDefiniteMatrix)"]/*'/>
        [Fresh]
        public static PositiveDefiniteMatrix SampleVariance([Proper] VectorGaussian Sample, PositiveDefiniteMatrix result)
        {
            return Sample.GetVariance(result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="SampleMeanInit(VectorGaussian)"]/*'/>
        [Skip]
        public static Vector SampleMeanInit([IgnoreDependency] VectorGaussian Sample)
        {
            return Vector.Zero(Sample.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="SampleMean(VectorGaussian, PositiveDefiniteMatrix, Vector)"]/*'/>
        [Fresh]
        public static Vector SampleMean([Proper] VectorGaussian Sample, PositiveDefiniteMatrix SampleVariance, Vector result)
        {
            return Sample.GetMean(result, SampleVariance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="MeanVarianceInit(VectorGaussian)"]/*'/>
        [Skip]
        public static PositiveDefiniteMatrix MeanVarianceInit([IgnoreDependency] VectorGaussian Mean)
        {
            return new PositiveDefiniteMatrix(Mean.Dimension, Mean.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="MeanVariance(VectorGaussian, PositiveDefiniteMatrix)"]/*'/>
        [Fresh]
        public static PositiveDefiniteMatrix MeanVariance([Proper] VectorGaussian Mean, PositiveDefiniteMatrix result)
        {
            return Mean.GetVariance(result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="MeanMeanInit(VectorGaussian)"]/*'/>
        [Skip]
        public static Vector MeanMeanInit([IgnoreDependency] VectorGaussian Mean)
        {
            return Vector.Zero(Mean.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="MeanMean(VectorGaussian, PositiveDefiniteMatrix, Vector)"]/*'/>
        [Fresh]
        public static Vector MeanMean([Proper] VectorGaussian Mean, PositiveDefiniteMatrix MeanVariance, Vector result)
        {
            return Mean.GetMean(result, MeanVariance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="PrecisionMeanInit(Wishart)"]/*'/>
        [Skip]
        public static PositiveDefiniteMatrix PrecisionMeanInit([IgnoreDependency] Wishart Precision)
        {
            return new PositiveDefiniteMatrix(Precision.Dimension, Precision.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="PrecisionMean(Wishart, PositiveDefiniteMatrix)"]/*'/>
        [Fresh]
        public static PositiveDefiniteMatrix PrecisionMean([Proper] Wishart Precision, PositiveDefiniteMatrix result)
        {
            return Precision.GetMean(result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="PrecisionMeanLogDet(Wishart)"]/*'/>
        [Fresh]
        public static double PrecisionMeanLogDet([Proper] Wishart Precision)
        {
            return Precision.GetMeanLogDeterminant();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="LogAverageFactor(Vector, Vector, PositiveDefiniteMatrix)"]/*'/>
        public static double LogAverageFactor(Vector sample, Vector mean, PositiveDefiniteMatrix precision)
        {
            int Dimension = sample.Count;
            LowerTriangularMatrix precL = new LowerTriangularMatrix(Dimension, Dimension);
            Vector iLb = Vector.Zero(Dimension);
            Vector precisionTimesMean = precision * mean;
            return VectorGaussian.GetLogProb(sample, precisionTimesMean, precision, precL, iLb);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="SampleConditional(Vector, PositiveDefiniteMatrix, VectorGaussian)"]/*'/>
        public static VectorGaussian SampleConditional(Vector Mean, PositiveDefiniteMatrix Precision, VectorGaussian result)
        {
            result.SetMeanAndPrecision(Mean, Precision);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="MeanConditional(Vector, PositiveDefiniteMatrix, VectorGaussian)"]/*'/>
        public static VectorGaussian MeanConditional(Vector Sample, PositiveDefiniteMatrix Precision, VectorGaussian result)
        {
            return SampleConditional(Sample, Precision, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="PrecisionConditional(Vector, Vector, Wishart, Vector)"]/*'/>
        public static Wishart PrecisionConditional(Vector Sample, Vector Mean, Wishart result, Vector diff)
        {
            if (result == default(Wishart))
                result = new Wishart(Sample.Count);
            diff.SetToDifference(Sample, Mean);
            const double SQRT_HALF = 0.70710678118654752440084436210485;
            diff.Scale(SQRT_HALF);
            result.Rate.SetToOuter(diff, diff);
            result.Shape = 0.5 * (result.Dimension + 2);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="PrecisionConditional(Vector, Vector, Wishart)"]/*'/>
        public static Wishart PrecisionConditional(Vector Sample, Vector Mean, Wishart result)
        {
            Vector workspace = Vector.Zero(Sample.Count);
            return PrecisionConditional(Sample, Mean, result, workspace);
        }

        //-- EP -----------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="LogAverageFactor(Vector, PositiveDefiniteMatrix, Vector, PositiveDefiniteMatrix, PositiveDefiniteMatrix)"]/*'/>
        public static double LogAverageFactor(
            Vector SampleMean,
            PositiveDefiniteMatrix SampleVariance,
            Vector MeanMean,
            PositiveDefiniteMatrix MeanVariance,
            PositiveDefiniteMatrix Precision)
        {
            return VectorGaussian.GetLogProb(SampleMean, MeanMean, Precision.Inverse() + SampleVariance + MeanVariance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="LogAverageFactor(Vector, Vector, PositiveDefiniteMatrix, PositiveDefiniteMatrix)"]/*'/>
        public static double LogAverageFactor(
            Vector Sample, Vector MeanMean, PositiveDefiniteMatrix MeanVariance, PositiveDefiniteMatrix Precision)
        {
            return VectorGaussian.GetLogProb(Sample, MeanMean, Precision.Inverse() + MeanVariance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="LogEvidenceRatio(Vector, Vector, PositiveDefiniteMatrix)"]/*'/>
        public static double LogEvidenceRatio(Vector sample, Vector mean, PositiveDefiniteMatrix precision)
        {
            return LogAverageFactor(sample, mean, precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="LogEvidenceRatio(Vector, VectorGaussian, Vector, PositiveDefiniteMatrix, PositiveDefiniteMatrix)"]/*'/>
        public static double LogEvidenceRatio(
            Vector sample, VectorGaussian mean, Vector MeanMean, PositiveDefiniteMatrix MeanVariance, PositiveDefiniteMatrix precision)
        {
            return LogAverageFactor(sample, MeanMean, MeanVariance, precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="LogEvidenceRatio(VectorGaussian, Vector, PositiveDefiniteMatrix)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(VectorGaussian sample, Vector mean, PositiveDefiniteMatrix precision)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="LogEvidenceRatio(VectorGaussian, VectorGaussian, PositiveDefiniteMatrix)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio([SkipIfUniform] VectorGaussian sample, [SkipIfUniform] VectorGaussian mean, PositiveDefiniteMatrix precision)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="SampleAverageConditional(Vector, PositiveDefiniteMatrix, VectorGaussian)"]/*'/>
        public static VectorGaussian SampleAverageConditional(Vector Mean, PositiveDefiniteMatrix Precision, VectorGaussian result)
        {
            return SampleConditional(Mean, Precision, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="SampleAverageConditional(VectorGaussian, PositiveDefiniteMatrix, VectorGaussian)"]/*'/>
        public static VectorGaussian SampleAverageConditional([SkipIfUniform] VectorGaussian Mean, PositiveDefiniteMatrix Precision, VectorGaussian result)
        {
            if (Mean.IsPointMass)
                return SampleConditional(Mean.Point, Precision, result);
            if (result == default(VectorGaussian))
                result = new VectorGaussian(Mean.Dimension);
            // R = Prec/(Prec + Mean.Prec)
            PositiveDefiniteMatrix R = Precision + Mean.Precision;
            R.SetToProduct(Precision, R.Inverse());
            for (int i = 0; i < Mean.Dimension; i++)
            {
                if (double.IsPositiveInfinity(Mean.Precision[i, i]))
                    R[i, i] = 1;
            }
            result.Precision.SetToProduct(R, Mean.Precision);
            result.Precision.Symmetrize();
            for (int i = 0; i < Mean.Dimension; i++)
            {
                if (double.IsPositiveInfinity(Mean.Precision[i, i]))
                {
                    for (int j = 0; j < Mean.Dimension; j++)
                    {
                        result.Precision[i, j] = 0;
                        result.Precision[j, i] = 0;
                    }
                    result.Precision[i, i] = 1;
                }
            }
            result.MeanTimesPrecision.SetToProduct(R, Mean.MeanTimesPrecision);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="SampleAverageConditionalInit(Vector)"]/*'/>
        [Skip]
        public static VectorGaussian SampleAverageConditionalInit(Vector Mean)
        {
            return VectorGaussian.Uniform(Mean.Count);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="SampleAverageConditionalInit(VectorGaussian)"]/*'/>
        [Skip]
        public static VectorGaussian SampleAverageConditionalInit([IgnoreDependency] VectorGaussian Mean)
        {
            return new VectorGaussian(Mean.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="MeanAverageConditional(Vector, PositiveDefiniteMatrix, VectorGaussian)"]/*'/>
        public static VectorGaussian MeanAverageConditional(Vector Sample, PositiveDefiniteMatrix Precision, VectorGaussian result)
        {
            return SampleConditional(Sample, Precision, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="MeanAverageConditional(VectorGaussian, PositiveDefiniteMatrix, VectorGaussian)"]/*'/>
        public static VectorGaussian MeanAverageConditional([SkipIfUniform] VectorGaussian Sample, PositiveDefiniteMatrix Precision, VectorGaussian result)
        {
            return SampleAverageConditional(Sample, Precision, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="PrecisionAverageConditional(Vector, Vector, Wishart)"]/*'/>
        public static Wishart PrecisionAverageConditional(Vector Sample, Vector Mean, Wishart result)
        {
            return PrecisionConditional(Sample, Mean, result);
        }

        //-- VMP ----------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="AverageLogFactor(VectorGaussian, Vector, PositiveDefiniteMatrix, VectorGaussian, Vector, PositiveDefiniteMatrix, Wishart, PositiveDefiniteMatrix, double)"]/*'/>
        public static double AverageLogFactor(
            [Proper] VectorGaussian sample,
            Vector SampleMean,
            PositiveDefiniteMatrix SampleVariance,
            [Proper] VectorGaussian mean,
            Vector MeanMean,
            PositiveDefiniteMatrix MeanVariance,
            [Proper] Wishart precision,
            PositiveDefiniteMatrix precisionMean,
            double precisionMeanLogDet)
        {
            if (sample.IsPointMass)
                return AverageLogFactor(sample.Point, mean, MeanMean, MeanVariance, precision, precisionMean, precisionMeanLogDet);
            if (mean.IsPointMass)
                return AverageLogFactor(sample, SampleMean, SampleVariance, mean.Point, precision, precisionMean, precisionMeanLogDet);
            if (precision.IsPointMass)
                return AverageLogFactor(sample, SampleMean, SampleVariance, mean, MeanMean, MeanVariance, precision.Point);

            return ComputeAverageLogFactor(SampleMean, SampleVariance, MeanMean, MeanVariance, precisionMeanLogDet, precisionMean);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="AverageLogFactor(Vector, Vector, Wishart, PositiveDefiniteMatrix, double)"]/*'/>
        public static double AverageLogFactor(
            Vector sample, Vector mean, [Proper] Wishart precision, PositiveDefiniteMatrix precisionMean, double precisionMeanLogDet)
        {
            if (precision.IsPointMass)
                return AverageLogFactor(sample, mean, precision.Point);
            else
                return ComputeAverageLogFactor(sample, mean, precisionMeanLogDet, precisionMean);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="AverageLogFactor(Vector, Vector, PositiveDefiniteMatrix)"]/*'/>
        public static double AverageLogFactor(Vector sample, Vector mean, PositiveDefiniteMatrix precision)
        {
            return ComputeAverageLogFactor(sample, mean, precision.LogDeterminant(ignoreInfinity: true), precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="AverageLogFactor(VectorGaussian, Vector, PositiveDefiniteMatrix, Vector, PositiveDefiniteMatrix)"]/*'/>
        public static double AverageLogFactor(
            [Proper] VectorGaussian sample,
            Vector SampleMean,
            PositiveDefiniteMatrix SampleVariance,
            Vector mean,
            PositiveDefiniteMatrix precision)
        {
            if (sample.IsPointMass)
                return AverageLogFactor(sample.Point, mean, precision);
            else
                return ComputeAverageLogFactor(SampleMean, SampleVariance, mean, precision.LogDeterminant(ignoreInfinity: true), precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="AverageLogFactor(Vector, VectorGaussian, Vector, PositiveDefiniteMatrix, PositiveDefiniteMatrix)"]/*'/>
        public static double AverageLogFactor(
            Vector sample, [Proper] VectorGaussian mean, Vector MeanMean, PositiveDefiniteMatrix MeanVariance, PositiveDefiniteMatrix precision)
        {
            return AverageLogFactor(mean, MeanMean, MeanVariance, sample, precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="AverageLogFactor(Vector, VectorGaussian, Vector, PositiveDefiniteMatrix, Wishart, PositiveDefiniteMatrix, double)"]/*'/>
        public static double AverageLogFactor(
            Vector sample,
            [Proper] VectorGaussian mean,
            Vector MeanMean,
            PositiveDefiniteMatrix MeanVariance,
            [Proper] Wishart precision,
            PositiveDefiniteMatrix precisionMean,
            double precisionMeanLogDet)
        {
            return AverageLogFactor(mean, MeanMean, MeanVariance, sample, precision, precisionMean, precisionMeanLogDet);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="AverageLogFactor(VectorGaussian, Vector, PositiveDefiniteMatrix, Vector, Wishart, PositiveDefiniteMatrix, double)"]/*'/>
        public static double AverageLogFactor(
            [Proper] VectorGaussian sample,
            Vector SampleMean,
            PositiveDefiniteMatrix SampleVariance,
            Vector mean,
            [Proper] Wishart precision,
            PositiveDefiniteMatrix precisionMean,
            double precisionMeanLogDet)
        {
            if (sample.IsPointMass)
                return AverageLogFactor(sample.Point, mean, precision, precisionMean, precisionMeanLogDet);
            if (precision.IsPointMass)
                return AverageLogFactor(sample, SampleMean, SampleVariance, mean, precision.Point);

            return ComputeAverageLogFactor(SampleMean, SampleVariance, mean, precisionMeanLogDet, precisionMean);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="AverageLogFactor(VectorGaussian, Vector, PositiveDefiniteMatrix, VectorGaussian, Vector, PositiveDefiniteMatrix, PositiveDefiniteMatrix)"]/*'/>
        public static double AverageLogFactor(
            [Proper] VectorGaussian sample,
            Vector SampleMean,
            PositiveDefiniteMatrix SampleVariance,
            [Proper] VectorGaussian mean,
            Vector MeanMean,
            PositiveDefiniteMatrix MeanVariance,
            PositiveDefiniteMatrix precision)
        {
            if (sample.IsPointMass)
                return AverageLogFactor(sample.Point, mean, MeanMean, MeanVariance, precision);
            if (mean.IsPointMass)
                return AverageLogFactor(sample, SampleMean, SampleVariance, mean.Point, precision);

            return ComputeAverageLogFactor(SampleMean, SampleVariance, MeanMean, MeanVariance, precision.LogDeterminant(ignoreInfinity: true), precision);
        }

        /// <summary>
        /// Helper method for computing average log factor
        /// </summary>
        /// <param name="SampleMean">Mean of incoming message from 'sample'</param>
        /// <param name="SampleVariance">Variance of incoming message from 'sample'</param>
        /// <param name="MeanMean">Mean of incoming message from 'mean'</param>
        /// <param name="MeanVariance">Variance of incoming message from 'mean'</param>
        /// <param name="precision_Elogx">Expected log value of the incoming message from 'precision'</param>
        /// <param name="precision_Ex">Expected value of incoming message from 'precision'</param>
        /// <returns>Computed average log factor</returns>
        private static double ComputeAverageLogFactor(
            Vector SampleMean,
            PositiveDefiniteMatrix SampleVariance,
            Vector MeanMean,
            PositiveDefiniteMatrix MeanVariance,
            double precision_Elogx,
            PositiveDefiniteMatrix precision_Ex)
        {
            int dim = SampleMean.Count;
            int nonzeroDims = 0;
            double precTimesVariance = 0.0;
            double precTimesDiff = 0.0;
            for (int i = 0; i < dim; i++)
            {
                if (double.IsPositiveInfinity(precision_Ex[i, i]))
                {
                    if (SampleMean[i] != MeanMean[i] || SampleVariance[i, i] + MeanVariance[i, i] > 0)
                        return double.NegativeInfinity;
                }
                else
                {
                    nonzeroDims++;
                    double sum = 0.0;
                    for (int j = 0; j < dim; j++)
                    {
                        sum += precision_Ex[i, j] * (SampleMean[j] - MeanMean[j]);
                        precTimesVariance += precision_Ex[i, j] * (SampleVariance[i, j] + MeanVariance[i, j]);
                    }
                    precTimesDiff += sum * (SampleMean[i] - MeanMean[i]);
                }
            }
            return -nonzeroDims * MMath.LnSqrt2PI + 0.5 * (precision_Elogx - precTimesVariance - precTimesDiff);
        }

        /// <summary>
        /// Helper method for computing average log factor
        /// </summary>
        /// <param name="SampleMean">Mean of incoming sample message</param>
        /// <param name="SampleVariance">Variance of incoming sample message</param>
        /// <param name="mean">Constant value for 'mean'.</param>
        /// <param name="precision_Elogx">Expected log value of the incoming message from 'precision'</param>
        /// <param name="precision_Ex">Expected value of incoming message from 'precision'</param>
        /// <returns>Computed average log factor</returns>
        private static double ComputeAverageLogFactor(
            Vector SampleMean,
            PositiveDefiniteMatrix SampleVariance,
            Vector mean,
            double precision_Elogx,
            PositiveDefiniteMatrix precision_Ex)
        {
            int dim = mean.Count;
            int nonzeroDims = 0;
            double precTimesVariance = 0.0;
            double precTimesDiff = 0.0;
            for (int i = 0; i < dim; i++)
            {
                if (double.IsPositiveInfinity(precision_Ex[i, i]))
                {
                    if (SampleMean[i] != mean[i] || SampleVariance[i, i] > 0)
                        return double.NegativeInfinity;
                }
                else
                {
                    nonzeroDims++;
                    double sum = 0.0;
                    for (int j = 0; j < dim; j++)
                    {
                        sum += precision_Ex[i, j] * (SampleMean[j] - mean[j]);
                        precTimesVariance += precision_Ex[i, j] * SampleVariance[j, i];
                    }
                    precTimesDiff += sum * (SampleMean[i] - mean[i]);
                }
            }
            return -nonzeroDims * MMath.LnSqrt2PI + 0.5 * (precision_Elogx - precTimesVariance - precTimesDiff);
        }

        /// <summary>
        /// Helper method for computing average log factor
        /// </summary>
        /// <param name="sample">Constant value for 'sample'.</param>
        /// <param name="mean">Constant value for 'mean'.</param>
        /// <param name="precision_Elogx">Expected log value of the incoming message from 'precision'</param>
        /// <param name="precision_Ex">Expected value of incoming message from 'precision'</param>
        /// <returns>Computed average log factor</returns>
        private static double ComputeAverageLogFactor(Vector sample, Vector mean, double precision_Elogx, PositiveDefiniteMatrix precision_Ex)
        {
            int dim = mean.Count;
            int nonzeroDims = 0;
            double precTimesDiff = 0.0;
            for (int i = 0; i < dim; i++)
            {
                if (double.IsPositiveInfinity(precision_Ex[i, i]))
                {
                    if (sample[i] != mean[i])
                        return double.NegativeInfinity;
                }
                else
                {
                    nonzeroDims++;
                    double sum = 0.0;
                    for (int j = 0; j < dim; j++)
                    {
                        sum += precision_Ex[i, j] * (sample[j] - mean[j]);
                    }
                    precTimesDiff += sum * (sample[i] - mean[i]);
                }
            }
            return -nonzeroDims * MMath.LnSqrt2PI + 0.5 * (precision_Elogx - precTimesDiff);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="SampleAverageLogarithm(Vector, PositiveDefiniteMatrix, VectorGaussian)"]/*'/>
        public static VectorGaussian SampleAverageLogarithm(Vector Mean, PositiveDefiniteMatrix Precision, VectorGaussian result)
        {
            return SampleConditional(Mean, Precision, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="SampleAverageLogarithm(VectorGaussian, Vector, Wishart, PositiveDefiniteMatrix, VectorGaussian)"]/*'/>
        public static VectorGaussian SampleAverageLogarithm(
            [Proper] VectorGaussian Mean,
            Vector MeanMean,
            [Proper] Wishart Precision,
            PositiveDefiniteMatrix PrecisionMean,
            VectorGaussian result)
        {
            return SampleAverageLogarithm(MeanMean, PrecisionMean, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="SampleAverageLogarithm(VectorGaussian, Vector, PositiveDefiniteMatrix, VectorGaussian)"]/*'/>
        public static VectorGaussian SampleAverageLogarithm([Proper] VectorGaussian mean, Vector MeanMean, PositiveDefiniteMatrix Precision, VectorGaussian result)
        {
            if (result == default(VectorGaussian))
                result = new VectorGaussian(MeanMean.Count);
            result.Precision.SetTo(Precision);
            result.MeanTimesPrecision.SetToProduct(result.Precision, MeanMean);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="SampleAverageLogarithm(Vector, Wishart, PositiveDefiniteMatrix, VectorGaussian)"]/*'/>
        public static VectorGaussian SampleAverageLogarithm(Vector Mean, [Proper] Wishart Precision, PositiveDefiniteMatrix PrecisionMean, VectorGaussian result)
        {
            return SampleAverageLogarithm(Mean, PrecisionMean, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="SampleAverageLogarithmInit(Vector)"]/*'/>
        [Skip]
        public static VectorGaussian SampleAverageLogarithmInit(Vector Mean)
        {
            return VectorGaussian.Uniform(Mean.Count);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="SampleAverageLogarithmInit(VectorGaussian)"]/*'/>
        [Skip]
        public static VectorGaussian SampleAverageLogarithmInit([IgnoreDependency] VectorGaussian Mean)
        {
            return new VectorGaussian(Mean.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="MeanAverageLogarithm(Vector, PositiveDefiniteMatrix, VectorGaussian)"]/*'/>
        public static VectorGaussian MeanAverageLogarithm(Vector Sample, PositiveDefiniteMatrix Precision, VectorGaussian result)
        {
            return SampleConditional(Sample, Precision, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="MeanAverageLogarithm(VectorGaussian, Vector, Wishart, PositiveDefiniteMatrix, VectorGaussian)"]/*'/>
        public static VectorGaussian MeanAverageLogarithm(
            [Proper] VectorGaussian Sample, Vector SampleMean, [Proper] Wishart Precision, PositiveDefiniteMatrix PrecisionMean, VectorGaussian result)
        {
            return SampleAverageLogarithm(Sample, SampleMean, Precision, PrecisionMean, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="MeanAverageLogarithm(VectorGaussian, Vector, PositiveDefiniteMatrix, VectorGaussian)"]/*'/>
        public static VectorGaussian MeanAverageLogarithm(
            [Proper] VectorGaussian Sample, Vector SampleMean, PositiveDefiniteMatrix Precision, VectorGaussian result)
        {
            return SampleAverageLogarithm(Sample, SampleMean, Precision, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="MeanAverageLogarithm(Vector, Wishart, PositiveDefiniteMatrix, VectorGaussian)"]/*'/>
        public static VectorGaussian MeanAverageLogarithm(Vector Sample, [Proper] Wishart Precision, PositiveDefiniteMatrix PrecisionMean, VectorGaussian result)
        {
            return SampleAverageLogarithm(Sample, Precision, PrecisionMean, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="PrecisionAverageLogarithm(Vector, Vector, Wishart)"]/*'/>
        public static Wishart PrecisionAverageLogarithm(Vector Sample, Vector Mean, Wishart result)
        {
            return PrecisionConditional(Sample, Mean, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="PrecisionAverageLogarithm(VectorGaussian, Vector, PositiveDefiniteMatrix, VectorGaussian, Vector, PositiveDefiniteMatrix, Wishart)"]/*'/>
        public static Wishart PrecisionAverageLogarithm(
            [Proper] VectorGaussian Sample,
            Vector SampleMean,
            PositiveDefiniteMatrix SampleVariance,
            [Proper] VectorGaussian Mean,
            Vector MeanMean,
            PositiveDefiniteMatrix MeanVariance,
            Wishart result)
        {
            if (Sample.IsPointMass)
                return PrecisionAverageLogarithm(Sample.Point, Mean, MeanMean, MeanVariance, result);
            if (Mean.IsPointMass)
                return PrecisionAverageLogarithm(Sample, SampleMean, SampleVariance, Mean.Point, result);
            // The formula is exp(int_x int_mean p(x) p(mean) log N(x;mean,1/prec)) =
            // exp(-0.5 prec E[(x-mean)^2] + 0.5 log(prec)) =
            // Gamma(prec; 0.5, 0.5*E[(x-mean)^2])
            // E[(x-mean)^2] = E[x^2] - 2 E[x] E[mean] + E[mean^2] = var(x) + (E[x]-E[mean])^2 + var(mean)
            if (result == default(Wishart))
                result = new Wishart(Sample.Dimension);
            // we want shape - (d+1)/2 = 0.5, therefore shape = (d+2)/2
            result.Shape = 0.5 * (result.Dimension + 2);
            Vector diff = SampleMean - MeanMean;
            result.Rate.SetToOuter(diff, diff);
            result.Rate.SetToSum(result.Rate, SampleVariance);
            result.Rate.SetToSum(result.Rate, MeanVariance);
            result.Rate.Scale(0.5);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="PrecisionAverageLogarithm(Vector, VectorGaussian, Vector, PositiveDefiniteMatrix, Wishart)"]/*'/>
        public static Wishart PrecisionAverageLogarithm(
            Vector Sample, [Proper] VectorGaussian Mean, Vector MeanMean, PositiveDefiniteMatrix MeanVariance, Wishart result)
        {
            if (Mean.IsPointMass)
                return PrecisionAverageLogarithm(Sample, Mean.Point, result);
            // The formula is exp(int_x int_mean p(x) p(mean) log N(x;mean,1/prec)) =
            // exp(-0.5 prec E[(x-mean)^2] + 0.5 log(prec)) =
            // Gamma(prec; 0.5, 0.5*E[(x-mean)^2])
            // E[(x-mean)^2] = E[x^2] - 2 E[x] E[mean] + E[mean^2] = var(x) + (E[x]-E[mean])^2 + var(mean)
            if (result == default(Wishart))
                result = new Wishart(Sample.Count);
            result.Shape = 0.5 * (result.Dimension + 2);
            Vector diff = Sample - MeanMean;
            result.Rate.SetToOuter(diff, diff);
            result.Rate.SetToSum(result.Rate, MeanVariance);
            result.Rate.Scale(0.5);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp"]/message_doc[@name="PrecisionAverageLogarithm(VectorGaussian, Vector, PositiveDefiniteMatrix, Vector, Wishart)"]/*'/>
        public static Wishart PrecisionAverageLogarithm(
            [Proper] VectorGaussian Sample,
            Vector SampleMean,
            PositiveDefiniteMatrix SampleVariance,
            Vector Mean,
            Wishart result)
        {
            return PrecisionAverageLogarithm(Mean, Sample, SampleMean, SampleVariance, result);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianFromMeanAndVarianceOp"]/doc/*'/>
    [FactorMethod(typeof(VectorGaussian), "SampleFromMeanAndVariance")]
    [Quality(QualityBand.Stable)]
    public static class VectorGaussianFromMeanAndVarianceOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianFromMeanAndVarianceOp"]/message_doc[@name="LogAverageFactor(Vector, Vector, PositiveDefiniteMatrix)"]/*'/>
        public static double LogAverageFactor(Vector sample, Vector mean, PositiveDefiniteMatrix variance)
        {
            VectorGaussian to_sample = SampleAverageConditional(mean, variance);
            return to_sample.GetLogProb(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianFromMeanAndVarianceOp"]/message_doc[@name="LogEvidenceRatio(Vector, Vector, PositiveDefiniteMatrix)"]/*'/>
        public static double LogEvidenceRatio(Vector sample, Vector mean, PositiveDefiniteMatrix variance)
        {
            return LogAverageFactor(sample, mean, variance);
        }

        [Skip]
        public static double LogEvidenceRatio(VectorGaussian sample, VectorGaussian mean, PositiveDefiniteMatrix variance)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianFromMeanAndVarianceOp"]/message_doc[@name="AverageLogFactor(Vector, Vector, PositiveDefiniteMatrix)"]/*'/>
        public static double AverageLogFactor(Vector sample, Vector mean, PositiveDefiniteMatrix variance)
        {
            return LogAverageFactor(sample, mean, variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianFromMeanAndVarianceOp"]/message_doc[@name="AverageLogFactor(VectorGaussian, VectorGaussian)"]/*'/>
        public static double AverageLogFactor(VectorGaussian sample, [Fresh] VectorGaussian to_sample)
        {
            return LogAverageFactor(sample, to_sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianFromMeanAndVarianceOp"]/message_doc[@name="AverageLogFactor(Vector, VectorGaussian, VectorGaussian)"]/*'/>
        public static double AverageLogFactor(Vector sample, VectorGaussian mean, [Fresh] VectorGaussian to_mean)
        {
            return AverageLogFactor(mean, to_mean);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianFromMeanAndVarianceOp"]/message_doc[@name="LogAverageFactor(VectorGaussian, VectorGaussian)"]/*'/>
        public static double LogAverageFactor(VectorGaussian sample, [Fresh] VectorGaussian to_sample)
        {
            return to_sample.GetLogAverageOf(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianFromMeanAndVarianceOp"]/message_doc[@name="LogEvidenceRatio(Vector, VectorGaussian, PositiveDefiniteMatrix)"]/*'/>
        public static double LogEvidenceRatio(Vector sample, VectorGaussian mean, PositiveDefiniteMatrix variance)
        {
            return SampleAverageConditional(sample, variance).GetLogAverageOf(mean);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianFromMeanAndVarianceOp"]/message_doc[@name="LogEvidenceRatio(VectorGaussian, Vector, PositiveDefiniteMatrix)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(VectorGaussian sample, Vector mean, PositiveDefiniteMatrix variance)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianFromMeanAndVarianceOp"]/message_doc[@name="SampleAverageLogarithm(Vector, PositiveDefiniteMatrix)"]/*'/>
        public static VectorGaussian SampleAverageLogarithm(Vector mean, PositiveDefiniteMatrix variance)
        {
            return VectorGaussian.FromMeanAndVariance(mean, variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianFromMeanAndVarianceOp"]/message_doc[@name="MeanAverageLogarithm(Vector, PositiveDefiniteMatrix)"]/*'/>
        public static VectorGaussian MeanAverageLogarithm(Vector sample, PositiveDefiniteMatrix variance)
        {
            return SampleAverageLogarithm(sample, variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianFromMeanAndVarianceOp"]/message_doc[@name="SampleAverageConditional(Vector, PositiveDefiniteMatrix)"]/*'/>
        public static VectorGaussian SampleAverageConditional(Vector mean, PositiveDefiniteMatrix variance)
        {
            return VectorGaussian.FromMeanAndVariance(mean, variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianFromMeanAndVarianceOp"]/message_doc[@name="MeanAverageConditional(Vector, PositiveDefiniteMatrix)"]/*'/>
        public static VectorGaussian MeanAverageConditional(Vector sample, PositiveDefiniteMatrix variance)
        {
            return SampleAverageConditional(sample, variance);
        }

        public static VectorGaussian SampleAverageConditional([SkipIfUniform] VectorGaussian mean, PositiveDefiniteMatrix variance, VectorGaussian result)
        {
            if (mean.IsPointMass)
                return SampleAverageConditional(mean.Point, variance);
            int d = mean.Dimension;
            if (result == default(VectorGaussian))
                result = new VectorGaussian(d);
            Vector meanMean = Vector.Zero(d);
            PositiveDefiniteMatrix meanVariance = new PositiveDefiniteMatrix(d, d);
            mean.GetMeanAndVariance(meanMean, meanVariance);
            meanVariance.SetToSum(meanVariance, variance);
            result.SetMeanAndVariance(meanMean, meanVariance);
            return result;
        }

        public static VectorGaussian MeanAverageConditional([SkipIfUniform] VectorGaussian sample, PositiveDefiniteMatrix variance, VectorGaussian result)
        {
            return SampleAverageConditional(sample, variance, result);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp_Laplace2"]/doc/*'/>
    [FactorMethod(typeof(VectorGaussian), "Sample", typeof(Vector), typeof(PositiveDefiniteMatrix))]
    [FactorMethod(new string[] { "sample", "mean", "precision" }, typeof(Factor), "VectorGaussian")]
    [Buffers("SampleMean", "SampleVariance", "MeanMean", "MeanVariance", "PrecisionMean", "PrecisionMeanLogDet")]
    [Quality(QualityBand.Preview)]
    public static class VectorGaussianOp_Laplace2
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp_Laplace2"]/message_doc[@name="LogAverageFactor(Vector, VectorGaussian, Wishart, Wishart)"]/*'/>
        public static double LogAverageFactor(Vector sample, VectorGaussian mean, [Proper] Wishart precision, Wishart to_precision)
        {
            // int_(x,m) f(x,m,r) p(x) p(m) dx dm = N(mx;mm, vx+vm+1/r) = |vx+vm+1/r|^(-1/2) exp(-0.5 tr((vx+vm+1/r)^(-1) (mx-mm)(mx-mm)'))
            int dim = precision.Dimension;
            Wishart rPost = precision * to_precision;
            PositiveDefiniteMatrix r = rPost.GetMean();
            PositiveDefiniteMatrix ir = r.Inverse();
            Vector mm = Vector.Zero(dim);
            PositiveDefiniteMatrix vm = new PositiveDefiniteMatrix(dim, dim);
            mean.GetMeanAndVariance(mm, vm);
            PositiveDefiniteMatrix v = vm;
            vm = null;
            v.SetToSum(v, ir);
            PositiveDefiniteMatrix iv = v.Inverse();
            Matrix ivr = ir * iv;
            Vector m = mm;
            mm = null;
            m.SetToDifference(sample, m);
            double result = -dim * MMath.LnSqrt2PI + 0.5 * iv.LogDeterminant() - 0.5 * iv.QuadraticForm(m);
            result += precision.GetLogProb(r) - rPost.GetLogProb(r);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp_Laplace2"]/message_doc[@name="LogEvidenceRatio(Vector, VectorGaussian, Wishart, Wishart)"]/*'/>
        public static double LogEvidenceRatio(Vector sample, VectorGaussian mean, [Proper] Wishart precision, Wishart to_precision)
        {
            return LogAverageFactor(sample, mean, precision, to_precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp_Laplace2"]/message_doc[@name="LogAverageFactor(VectorGaussian, VectorGaussian, Wishart, Wishart)"]/*'/>
        public static double LogAverageFactor(VectorGaussian sample, VectorGaussian mean, [Proper] Wishart precision, Wishart to_precision)
        {
            // int_(x,m) f(x,m,r) p(x) p(m) dx dm = N(mx;mm, vx+vm+1/r) = |vx+vm+1/r|^(-1/2) exp(-0.5 tr((vx+vm+1/r)^(-1) (mx-mm)(mx-mm)'))
            int dim = precision.Dimension;
            Wishart rPost = precision * to_precision;
            PositiveDefiniteMatrix r = rPost.GetMean();
            PositiveDefiniteMatrix ir = r.Inverse();
            Vector mx = Vector.Zero(dim);
            PositiveDefiniteMatrix vx = new PositiveDefiniteMatrix(dim, dim);
            sample.GetMeanAndVariance(mx, vx);
            Vector mm = Vector.Zero(dim);
            PositiveDefiniteMatrix vm = new PositiveDefiniteMatrix(dim, dim);
            mean.GetMeanAndVariance(mm, vm);
            PositiveDefiniteMatrix v = vx;
            vx = null;
            v.SetToSum(v, vm);
            v.SetToSum(v, ir);
            PositiveDefiniteMatrix iv = v.Inverse();
            Matrix ivr = ir * iv;
            Vector m = mx;
            mx = null;
            m.SetToDifference(m, mm);
            double result = -dim * MMath.LnSqrt2PI + 0.5 * iv.LogDeterminant() - 0.5 * iv.QuadraticForm(m);
            result += precision.GetLogProb(r) - rPost.GetLogProb(r);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp_Laplace2"]/message_doc[@name="LogEvidenceRatio(VectorGaussian, VectorGaussian, Wishart, Wishart, VectorGaussian)"]/*'/>
        public static double LogEvidenceRatio(VectorGaussian sample, VectorGaussian mean, [Proper] Wishart precision, Wishart to_precision, VectorGaussian to_sample)
        {
            return LogAverageFactor(sample, mean, precision, to_precision) - to_sample.GetLogAverageOf(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp_Laplace2"]/message_doc[@name="SampleAverageConditional(VectorGaussian, Wishart, Wishart, VectorGaussian)"]/*'/>
        public static VectorGaussian SampleAverageConditional([SkipIfUniform] VectorGaussian mean, [Proper] Wishart precision, [NoInit] Wishart to_precision, VectorGaussian result)
        {
            var rPost = precision * to_precision;
            PositiveDefiniteMatrix r = rPost.GetMean();
            return VectorGaussianOp.SampleAverageConditional(mean, r, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp_Laplace2"]/message_doc[@name="MeanAverageConditional(VectorGaussian, Wishart, Wishart, VectorGaussian)"]/*'/>
        public static VectorGaussian MeanAverageConditional([SkipIfUniform] VectorGaussian sample, [Proper] Wishart precision, [NoInit] Wishart to_precision, VectorGaussian result)
        {
            return SampleAverageConditional(sample, precision, to_precision, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorGaussianOp_Laplace2"]/message_doc[@name="PrecisionAverageConditional(VectorGaussian, VectorGaussian, Wishart, Wishart, Wishart)"]/*'/>
        public static Wishart PrecisionAverageConditional([SkipIfUniform] VectorGaussian sample, [SkipIfUniform] VectorGaussian mean, [Proper] Wishart precision, Wishart to_precision, Wishart result)
        {
            // int_(x,m) f(x,m,r) p(x) p(m) dx dm = N(mx;mm, vx+vm+1/r) = |vx+vm+1/r|^(-1/2) exp(-0.5 tr((vx+vm+1/r)^(-1) (mx-mm)(mx-mm)'))
            // log f(r) = -0.5 log|v+1/r| -0.5 tr((v+1/r)^(-1) S)
            // dlogf(r) = 0.5 tr((1/r) (v+1/r)^(-1) (1/r) dr) - 0.5 tr((1/r) (v+1/r)^(-1) S (v+1/r)^(-1) (1/r) dr)
            // tr(r dlogf') = 0.5 tr(r (d(1/r) (v+1/r)^(-1) (1/r) + (1/r) d(v+1/r)^(-1) (1/r) + (1/r) (v+1/r)^(-1) d(1/r)))
            //               -0.5 tr(r (d(1/r) (v+1/r)^(-1) S (v+1/r)^(-1) (1/r) ...
            //              = -tr((1/r) (v+1/r)^(-1) (1/r) dr) + 0.5 tr((1/r) (v+1/r)^(-1) (1/r) (v+1/r)^(-1) (1/r) dr)
            //                +tr((1/r) (v+1/r)^(-1) S (v+1/r)^(-1) (1/r) dr) - tr((1/r) (v+1/r)^(-1) S (v+1/r)^(-1) (1/r) (v+1/r)^(-1) (1/r) dr)
            // tr(r tr(r dlogf')/dr) = -tr((v+1/r)^(-1) (1/r)) + 0.5 tr((v+1/r)^(-1) (1/r) (v+1/r)^(-1) (1/r))
            //                         +tr((v+1/r)^(-1) S (v+1/r)^(-1) (1/r)) - tr((v+1/r)^(-1) S (v+1/r)^(-1) (1/r) (v+1/r)^(-1) (1/r))
            int dim = sample.Dimension;
            Wishart rPost = precision * to_precision;
            PositiveDefiniteMatrix r = rPost.GetMean();
            PositiveDefiniteMatrix ir = r.Inverse();
            Vector mx = Vector.Zero(dim);
            PositiveDefiniteMatrix vx = new PositiveDefiniteMatrix(dim, dim);
            sample.GetMeanAndVariance(mx, vx);
            Vector mm = Vector.Zero(dim);
            PositiveDefiniteMatrix vm = new PositiveDefiniteMatrix(dim, dim);
            mean.GetMeanAndVariance(mm, vm);
            PositiveDefiniteMatrix v = vx;
            vx = null;
            v.SetToSum(v, vm);
            v.SetToSum(v, ir);
            PositiveDefiniteMatrix iv = v.Inverse();
            Matrix ivr = ir * iv;
            Vector m = mx;
            mx = null;
            m.SetToDifference(m, mm);
            Vector ivrm = mm;
            mm = null;
            ivrm.SetToProduct(ivr, m);
            PositiveDefiniteMatrix ivrs = vm;
            vm = null;
            ivrs.SetToOuter(ivrm, ivrm);
            var ivrr = new PositiveDefiniteMatrix(dim, dim);
            ivrr.SetToProduct(ivr, ir);
            ivrr.Symmetrize();
            double xxddlogf = -ivr.Trace() + 0.5 * Matrix.TraceOfProduct(ivrr, iv) + Matrix.TraceOfProduct(ivrs, r) - Matrix.TraceOfProduct(ivrs, iv);
            // result.Rate holds dlogf
            result.Rate.SetToSum(0.5, ivrr, -0.5, ivrs);
            LowerTriangularMatrix rChol = new LowerTriangularMatrix(dim, dim);
            rChol.SetToCholesky(r);
            result.SetDerivatives(rChol, ir, result.Rate, xxddlogf, GaussianOp.ForceProper);
            return result;
        }
    }
}

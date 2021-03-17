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

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubvectorOp"]/doc/*'/>
    [FactorMethod(typeof(Vector), "Subvector", typeof(Vector), typeof(int), typeof(int))]
    [Buffers("SourceMean", "SourceVariance")]
    [Quality(QualityBand.Stable)]
    public static class SubvectorOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubvectorOp"]/message_doc[@name="LogAverageFactor(Vector, Vector, int)"]/*'/>
        public static double LogAverageFactor(Vector subvector, Vector source, int startIndex)
        {
            for (int i = 0; i < subvector.Count; i++)
            {
                if (subvector[i] != source[startIndex + i])
                    return Double.NegativeInfinity;
            }
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubvectorOp"]/message_doc[@name="LogEvidenceRatio(Vector, Vector, int)"]/*'/>
        public static double LogEvidenceRatio(Vector subvector, Vector source, int startIndex)
        {
            return LogAverageFactor(subvector, source, startIndex);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubvectorOp"]/message_doc[@name="AverageLogFactor(Vector, Vector, int)"]/*'/>
        public static double AverageLogFactor(Vector subvector, Vector source, int startIndex)
        {
            return LogAverageFactor(subvector, source, startIndex);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubvectorOp"]/message_doc[@name="SourceVarianceInit(VectorGaussian)"]/*'/>
        [Skip]
        public static PositiveDefiniteMatrix SourceVarianceInit([IgnoreDependency] VectorGaussian Source)
        {
            return new PositiveDefiniteMatrix(Source.Dimension, Source.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubvectorOp"]/message_doc[@name="SourceVariance(VectorGaussian, PositiveDefiniteMatrix)"]/*'/>
        [Fresh]
        public static PositiveDefiniteMatrix SourceVariance([Proper] VectorGaussian Source, PositiveDefiniteMatrix result)
        {
            return Source.GetVariance(result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubvectorOp"]/message_doc[@name="SourceMeanInit(VectorGaussian)"]/*'/>
        [Skip]
        public static Vector SourceMeanInit([IgnoreDependency] VectorGaussian Source)
        {
            return Vector.Zero(Source.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubvectorOp"]/message_doc[@name="SourceMean(VectorGaussian, PositiveDefiniteMatrix, Vector)"]/*'/>
        [Fresh]
        public static Vector SourceMean([Proper] VectorGaussian Source, PositiveDefiniteMatrix SourceVariance, Vector result)
        {
            return Source.GetMean(result, SourceVariance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubvectorOp"]/message_doc[@name="LogAverageFactor(Vector, Vector, PositiveDefiniteMatrix, int)"]/*'/>
        public static double LogAverageFactor(Vector subvector, Vector SourceMean, PositiveDefiniteMatrix SourceVariance, int startIndex)
        {
            double sum = 0.0;
            for (int i = startIndex; i < SourceMean.Count; i++)
            {
                sum += Gaussian.GetLogProb(subvector[i], SourceMean[i], SourceVariance[i, i]);
            }
            return sum;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubvectorOp"]/message_doc[@name="LogEvidenceRatio(Vector, Vector, PositiveDefiniteMatrix, int)"]/*'/>
        public static double LogEvidenceRatio(Vector subvector, Vector SourceMean, PositiveDefiniteMatrix SourceVariance, int startIndex)
        {
            return LogAverageFactor(subvector, SourceMean, SourceVariance, startIndex);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubvectorOp"]/message_doc[@name="LogAverageFactor(VectorGaussian, VectorGaussian)"]/*'/>
        public static double LogAverageFactor(VectorGaussian subvector, [Fresh] VectorGaussian to_subvector)
        {
            return to_subvector.GetLogAverageOf(subvector);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubvectorOp"]/message_doc[@name="LogEvidenceRatio(VectorGaussian)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(VectorGaussian subvector)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubvectorOp"]/message_doc[@name="SubvectorAverageConditionalInit(int)"]/*'/>
        [Skip]
        public static VectorGaussian SubvectorAverageConditionalInit(int count)
        {
            return new VectorGaussian(count);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubvectorOp"]/message_doc[@name="SubvectorAverageLogarithmInit(int)"]/*'/>
        [Skip]
        public static VectorGaussian SubvectorAverageLogarithmInit(int count)
        {
            return new VectorGaussian(count);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubvectorOp"]/message_doc[@name="SubvectorAverageConditional(Vector, PositiveDefiniteMatrix, int, VectorGaussian)"]/*'/>
        public static VectorGaussian SubvectorAverageConditional(
            Vector SourceMean, PositiveDefiniteMatrix SourceVariance, int startIndex, VectorGaussian result)
        {
            PositiveDefiniteMatrix subVariance = new PositiveDefiniteMatrix(result.Dimension, result.Dimension);
            subVariance.SetToSubmatrix(SourceVariance, startIndex, startIndex);
            Vector subMean = Vector.Zero(result.Dimension);
            subMean.SetToSubvector(SourceMean, startIndex);
            result.SetMeanAndVariance(subMean, subVariance);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubvectorOp"]/message_doc[@name="SourceAverageConditional(VectorGaussian, int, VectorGaussian)"]/*'/>
        public static VectorGaussian SourceAverageConditional([SkipIfUniform] VectorGaussian subvector, int startIndex, VectorGaussian result)
        {
            result.MeanTimesPrecision.SetAllElementsTo(0.0);
            result.MeanTimesPrecision.SetSubvector(startIndex, subvector.MeanTimesPrecision);
            result.Precision.SetAllElementsTo(0.0);
            result.Precision.SetSubmatrix(startIndex, startIndex, subvector.Precision);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubvectorOp"]/message_doc[@name="SourceAverageConditional(Vector, int, VectorGaussian)"]/*'/>
        public static VectorGaussian SourceAverageConditional(Vector subvector, int startIndex, VectorGaussian result)
        {
            result.MeanTimesPrecision.SetAllElementsTo(0.0);
            result.MeanTimesPrecision.SetSubvector(startIndex, subvector);
            result.Precision.SetAllElementsTo(0.0);
            int dim = result.Dimension;
            for (int i = startIndex; i < dim; i++)
            {
                result.Precision[i, i] = Double.PositiveInfinity;
            }
            return result;
        }

        //-- VMP ---------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubvectorOp"]/message_doc[@name="AverageLogFactor()"]/*'/>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubvectorOp"]/message_doc[@name="SubvectorAverageLogarithm(Vector, PositiveDefiniteMatrix, int, VectorGaussian)"]/*'/>
        public static VectorGaussian SubvectorAverageLogarithm(Vector SourceMean, PositiveDefiniteMatrix SourceVariance, int startIndex, VectorGaussian result)
        {
            return SubvectorAverageConditional(SourceMean, SourceVariance, startIndex, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubvectorOp"]/message_doc[@name="SourceAverageLogarithm(VectorGaussian, int, VectorGaussian)"]/*'/>
        public static VectorGaussian SourceAverageLogarithm([SkipIfUniform] VectorGaussian subvector, int startIndex, VectorGaussian result)
        {
            return SourceAverageConditional(subvector, startIndex, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubvectorOp"]/message_doc[@name="SourceAverageLogarithm(Vector, int, VectorGaussian)"]/*'/>
        public static VectorGaussian SourceAverageLogarithm(Vector subvector, int startIndex, VectorGaussian result)
        {
            return SourceAverageConditional(subvector, startIndex, result);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorElementOp"]/doc/*'/>
    [FactorMethod(typeof(Collection), "GetItem<>", typeof(double), typeof(Vector), typeof(int))]
    [Buffers("ArrayMean", "ArrayVariance")]
    [Quality(QualityBand.Preview)]
    public static class VectorElementOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorElementOp"]/message_doc[@name="ArrayVarianceInit(VectorGaussian)"]/*'/>
        [Skip]
        public static PositiveDefiniteMatrix ArrayVarianceInit([IgnoreDependency] VectorGaussian array)
        {
            return new PositiveDefiniteMatrix(array.Dimension, array.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorElementOp"]/message_doc[@name="ArrayVariance(VectorGaussian, PositiveDefiniteMatrix)"]/*'/>
        [Fresh]
        public static PositiveDefiniteMatrix ArrayVariance([Proper] VectorGaussian array, PositiveDefiniteMatrix result)
        {
            return array.GetVariance(result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorElementOp"]/message_doc[@name="ArrayMeanInit(VectorGaussian)"]/*'/>
        [Skip]
        public static Vector ArrayMeanInit([IgnoreDependency] VectorGaussian array)
        {
            return Vector.Zero(array.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorElementOp"]/message_doc[@name="ArrayMean(VectorGaussian, PositiveDefiniteMatrix, Vector)"]/*'/>
        [Fresh]
        public static Vector ArrayMean([Proper] VectorGaussian array, PositiveDefiniteMatrix ArrayVariance, Vector result)
        {
            return array.GetMean(result, ArrayVariance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorElementOp"]/message_doc[@name="LogAverageFactor(double, VectorGaussian, Vector, PositiveDefiniteMatrix, int)"]/*'/>
        public static double LogAverageFactor(
            double item, [SkipIfUniform] VectorGaussian array, Vector ArrayMean, PositiveDefiniteMatrix ArrayVariance, int index)
        {
            Gaussian to_item = ItemAverageConditional(array, ArrayMean, ArrayVariance, index);
            return to_item.GetLogProb(item);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorElementOp"]/message_doc[@name="LogEvidenceRatio(double, VectorGaussian, Vector, PositiveDefiniteMatrix, int)"]/*'/>
        public static double LogEvidenceRatio(
            double item, [SkipIfUniform] VectorGaussian array, Vector ArrayMean, PositiveDefiniteMatrix ArrayVariance, int index)
        {
            return LogAverageFactor(item, array, ArrayMean, ArrayVariance, index);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorElementOp"]/message_doc[@name="ItemAverageConditional(VectorGaussian, Vector, PositiveDefiniteMatrix, int)"]/*'/>
        public static Gaussian ItemAverageConditional(
            [SkipIfUniform] VectorGaussian array, Vector ArrayMean, PositiveDefiniteMatrix ArrayVariance, int index)
        {
            return new Gaussian(ArrayMean[index], ArrayVariance[index, index]);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorElementOp"]/message_doc[@name="ItemAverageConditionalInit()"]/*'/>
        [Skip]
        public static Gaussian ItemAverageConditionalInit()
        {
            return Gaussian.Uniform();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorElementOp"]/message_doc[@name="ItemAverageLogarithmInit()"]/*'/>
        [Skip]
        public static Gaussian ItemAverageLogarithmInit()
        {
            return Gaussian.Uniform();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorElementOp"]/message_doc[@name="ArrayAverageConditional(Gaussian, int, VectorGaussian)"]/*'/>
        public static VectorGaussian ArrayAverageConditional([SkipIfUniform] Gaussian item, int index, VectorGaussian result)
        {
            result.MeanTimesPrecision.SetAllElementsTo(0.0);
            result.MeanTimesPrecision[index] = item.MeanTimesPrecision;
            result.Precision.SetAllElementsTo(0.0);
            result.Precision[index, index] = item.Precision;
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorElementOp"]/message_doc[@name="ArrayAverageConditional(double, int, VectorGaussian)"]/*'/>
        public static VectorGaussian ArrayAverageConditional(double item, int index, VectorGaussian result)
        {
            result.MeanTimesPrecision.SetAllElementsTo(0.0);
            result.MeanTimesPrecision[index] = item;
            result.Precision.SetAllElementsTo(0.0);
            result.Precision[index, index] = Double.PositiveInfinity;
            return result;
        }

        //-- VMP ------------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorElementOp"]/message_doc[@name="AverageLogFactor(double, VectorGaussian, Vector, PositiveDefiniteMatrix, int)"]/*'/>
        public static double AverageLogFactor(
            double item, [SkipIfUniform] VectorGaussian array, Vector ArrayMean, PositiveDefiniteMatrix ArrayVariance, int index)
        {
            return LogAverageFactor(item, array, ArrayMean, ArrayVariance, index);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorElementOp"]/message_doc[@name="AverageLogFactor(Gaussian, VectorGaussian)"]/*'/>
        [Skip]
        public static double AverageLogFactor(Gaussian item, VectorGaussian array)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorElementOp"]/message_doc[@name="ItemAverageLogarithm(VectorGaussian, Vector, PositiveDefiniteMatrix, int)"]/*'/>
        public static Gaussian ItemAverageLogarithm(
            [SkipIfUniform] VectorGaussian array, Vector ArrayMean, PositiveDefiniteMatrix ArrayVariance, int index)
        {
            return ItemAverageConditional(array, ArrayMean, ArrayVariance, index);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorElementOp"]/message_doc[@name="ArrayAverageLogarithm(Gaussian, int, VectorGaussian)"]/*'/>
        public static VectorGaussian ArrayAverageLogarithm([SkipIfUniform] Gaussian item, int index, VectorGaussian result)
        {
            return ArrayAverageConditional(item, index, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorElementOp"]/message_doc[@name="ArrayAverageLogarithm(double, int, VectorGaussian)"]/*'/>
        public static VectorGaussian ArrayAverageLogarithm(double item, int index, VectorGaussian result)
        {
            return ArrayAverageConditional(item, index, result);
        }
    }
}

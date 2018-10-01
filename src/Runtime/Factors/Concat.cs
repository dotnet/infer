// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/doc/*'/>
    [FactorMethod(typeof(Vector), "Concat", typeof(Vector), typeof(Vector))]
    [Quality(QualityBand.Stable)]
    public static class ConcatOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="LogAverageFactor(Vector, Vector, Vector)"]/*'/>
        public static double LogAverageFactor(Vector concat, Vector first, Vector second)
        {
            for (int i = 0; i < first.Count; i++)
            {
                if (concat[i] != first[i])
                    return Double.NegativeInfinity;
            }
            int dim1 = first.Count;
            for (int i = 0; i < second.Count; i++)
            {
                if (concat[i + dim1] != second[i])
                    return Double.NegativeInfinity;
            }
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="LogAverageFactor(VectorGaussian, Vector, Vector)"]/*'/>
        public static double LogAverageFactor(VectorGaussian concat, Vector first, Vector second)
        {
            return concat.GetLogProb(Vector.Concat(first, second));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="LogAverageFactor(Vector, VectorGaussian, VectorGaussian)"]/*'/>
        public static double LogAverageFactor(Vector concat, VectorGaussian first, VectorGaussian second)
        {
            Vector concat1 = Vector.Subvector(concat, 0, first.Dimension);
            Vector concat2 = Vector.Subvector(concat, first.Dimension, second.Dimension);
            return first.GetLogProb(concat1) + second.GetLogProb(concat2);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="LogAverageFactor(Vector, Vector, VectorGaussian)"]/*'/>
        public static double LogAverageFactor(Vector concat, Vector first, VectorGaussian second)
        {
            for (int i = 0; i < first.Count; i++)
            {
                if (concat[i] != first[i])
                    return Double.NegativeInfinity;
            }
            Vector concat2 = Vector.Subvector(concat, first.Count, second.Dimension);
            return second.GetLogProb(concat2);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="LogAverageFactor(Vector, VectorGaussian, Vector)"]/*'/>
        public static double LogAverageFactor(Vector concat, VectorGaussian first, Vector second)
        {
            int dim1 = first.Dimension;
            for (int i = 0; i < second.Count; i++)
            {
                if (concat[i + dim1] != second[i])
                    return Double.NegativeInfinity;
            }
            Vector concat1 = Vector.Subvector(concat, 0, first.Dimension);
            return first.GetLogProb(concat1);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="LogEvidenceRatio(Vector, VectorGaussian, VectorGaussian)"]/*'/>
        public static double LogEvidenceRatio(Vector concat, VectorGaussian first, VectorGaussian second)
        {
            return LogAverageFactor(concat, first, second);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="LogEvidenceRatio(Vector, Vector, Vector)"]/*'/>
        public static double LogEvidenceRatio(Vector concat, Vector first, Vector second)
        {
            return LogAverageFactor(concat, first, second);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="LogEvidenceRatio(Vector, Vector, VectorGaussian)"]/*'/>
        public static double LogEvidenceRatio(Vector concat, Vector first, VectorGaussian second)
        {
            return LogAverageFactor(concat, first, second);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="LogEvidenceRatio(Vector, VectorGaussian, Vector)"]/*'/>
        public static double LogEvidenceRatio(Vector concat, VectorGaussian first, Vector second)
        {
            return LogAverageFactor(concat, first, second);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="LogAverageFactor(VectorGaussian, VectorGaussian)"]/*'/>
        public static double LogAverageFactor(VectorGaussian concat, [Fresh] VectorGaussian to_concat)
        {
            return to_concat.GetLogAverageOf(concat);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="LogEvidenceRatio(VectorGaussian)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(VectorGaussian concat)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="ConcatAverageConditionalInit(VectorGaussian, VectorGaussian)"]/*'/>
        [Skip]
        public static VectorGaussian ConcatAverageConditionalInit([IgnoreDependency] VectorGaussian first, [IgnoreDependency] VectorGaussian second)
        {
            return new VectorGaussian(first.Dimension + second.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="ConcatAverageConditionalInit(Vector, VectorGaussian)"]/*'/>
        [Skip]
        public static VectorGaussian ConcatAverageConditionalInit([IgnoreDependency] Vector first, [IgnoreDependency] VectorGaussian second)
        {
            return new VectorGaussian(first.Count + second.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="ConcatAverageConditionalInit(VectorGaussian, Vector)"]/*'/>
        [Skip]
        public static VectorGaussian ConcatAverageConditionalInit([IgnoreDependency] VectorGaussian first, [IgnoreDependency] Vector second)
        {
            return new VectorGaussian(first.Dimension + second.Count);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="ConcatAverageConditional(VectorGaussian, VectorGaussian, VectorGaussian)"]/*'/>
        [SkipIfAllUniform]
        public static VectorGaussian ConcatAverageConditional(VectorGaussian first, VectorGaussian second, VectorGaussian result)
        {
            int dim1 = first.Dimension;
            int dim2 = second.Dimension;
            if (result.Dimension != dim1 + dim2)
                throw new ArgumentException("concat.Dimension (" + result.Dimension + ") != first.Dimension (" + first.Dimension + ") + second.Dimension (" + second.Dimension +
                                            ")");
            // assume result.Precision was initialized to 0.0?
            result.Precision.SetAllElementsTo(0.0);
            result.Precision.SetSubmatrix(0, 0, first.Precision);
            result.Precision.SetSubmatrix(dim1, dim1, second.Precision);
            result.MeanTimesPrecision.SetSubvector(0, first.MeanTimesPrecision);
            result.MeanTimesPrecision.SetSubvector(dim1, second.MeanTimesPrecision);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="ConcatAverageConditional(Vector, VectorGaussian, VectorGaussian)"]/*'/>
        public static VectorGaussian ConcatAverageConditional(Vector first, VectorGaussian second, VectorGaussian result)
        {
            int dim1 = first.Count;
            result.Precision.SetAllElementsTo(0.0);
            for (int i = 0; i < dim1; i++)
            {
                result.Precision[i, i] = Double.PositiveInfinity;
                result.MeanTimesPrecision[i] = first[i];
            }
            result.Precision.SetSubmatrix(dim1, dim1, second.Precision);
            result.MeanTimesPrecision.SetSubvector(dim1, second.MeanTimesPrecision);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="ConcatAverageConditional(VectorGaussian, Vector, VectorGaussian)"]/*'/>
        public static VectorGaussian ConcatAverageConditional(VectorGaussian first, Vector second, VectorGaussian result)
        {
            int dim1 = first.Dimension;
            int dim2 = second.Count;
            result.Precision.SetAllElementsTo(0.0);
            result.Precision.SetSubmatrix(0, 0, first.Precision);
            result.MeanTimesPrecision.SetSubvector(0, first.MeanTimesPrecision);
            for (int i = 0; i < dim2; i++)
            {
                int j = i + dim1;
                result.Precision[j, j] = Double.PositiveInfinity;
                result.MeanTimesPrecision[j] = second[i];
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="FirstAverageConditional(VectorGaussian, Vector, VectorGaussian)"]/*'/>
        public static VectorGaussian FirstAverageConditional([SkipIfUniform] VectorGaussian concat, Vector second, VectorGaussian result)
        {
            // joint distribution is proportional to: exp(-0.5 [first-mean1; second-mean2]' [prec11 prec12; prec21 prec22] [first-mean1; second-mean2])
            // posterior for first is proportional to: exp(-0.5 ((first-mean1)' prec11 (first-mean1) + 2 (first-mean1)' prec12 (second-mean2)))
            // = exp(-0.5 (first' prec11 first - 2 first' prec11 mean1 + 2 first' prec12 (second-mean2)))
            // first.precision = prec11
            // first.meanTimesPrecision = prec11 mean1 - prec12 (second-mean2) = [prec11; prec12] mean - prec12 second
            int dim1 = result.Dimension;
            int dim2 = second.Count;
            if (concat.Dimension != dim1 + dim2)
                throw new ArgumentException("concat.Dimension (" + concat.Dimension + ") != first.Dimension (" + dim1 + ") + second.Dimension (" + dim2 + ")");
            result.Precision.SetToSubmatrix(concat.Precision, 0, 0);
            Matrix prec12 = new Matrix(dim1, dim2);
            prec12.SetToSubmatrix(concat.Precision, 0, dim1);
            Vector prec12second = Vector.Zero(dim1);
            prec12second.SetToProduct(prec12, second);
            result.MeanTimesPrecision.SetToSubvector(concat.MeanTimesPrecision, 0);
            result.MeanTimesPrecision.SetToDifference(result.MeanTimesPrecision, prec12second);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="FirstAverageConditional(VectorGaussian, VectorGaussian, VectorGaussian)"]/*'/>
        public static VectorGaussian FirstAverageConditional([SkipIfUniform] VectorGaussian concat, VectorGaussian second, VectorGaussian result)
        {
            if (second.IsPointMass)
                return FirstAverageConditional(concat, second.Point, result);
            int dim1 = result.Dimension;
            VectorGaussian concatTimesSecond = new VectorGaussian(concat.Dimension);
            concatTimesSecond.MeanTimesPrecision.SetSubvector(dim1, second.MeanTimesPrecision);
            concatTimesSecond.Precision.SetSubmatrix(dim1, dim1, second.Precision);
            concatTimesSecond.SetToProduct(concatTimesSecond, concat);
            concatTimesSecond.GetMarginal(0, result);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="SecondAverageConditional(VectorGaussian, Vector, VectorGaussian)"]/*'/>
        public static VectorGaussian SecondAverageConditional([SkipIfUniform] VectorGaussian concat, Vector first, VectorGaussian result)
        {
            // prec = concat.Precision[dim2,dim2]
            // meanTimesPrec = concat.MeanTimesPrecision[dim2] - concat.Precision[dim2,dim1]*first
            int dim1 = first.Count;
            int dim2 = result.Dimension;
            if (concat.Dimension != dim1 + dim2)
                throw new ArgumentException("concat.Dimension (" + concat.Dimension + ") != first.Dimension (" + dim1 + ") + second.Dimension (" + dim2 + ")");
            result.Precision.SetToSubmatrix(concat.Precision, dim1, dim1);
            Matrix prec21 = new Matrix(dim2, dim1);
            prec21.SetToSubmatrix(concat.Precision, dim1, 0);
            Vector prec21first = Vector.Zero(dim2);
            prec21first.SetToProduct(prec21, first);
            result.MeanTimesPrecision.SetToSubvector(concat.MeanTimesPrecision, dim1);
            result.MeanTimesPrecision.SetToDifference(result.MeanTimesPrecision, prec21first);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="SecondAverageConditional(VectorGaussian, VectorGaussian, VectorGaussian)"]/*'/>
        public static VectorGaussian SecondAverageConditional([SkipIfUniform] VectorGaussian concat, VectorGaussian first, VectorGaussian result)
        {
            if (first.IsPointMass)
                return SecondAverageConditional(concat, first.Point, result);
            int dim1 = first.Dimension;
            VectorGaussian concatTimesFirst = new VectorGaussian(concat.Dimension);
            concatTimesFirst.MeanTimesPrecision.SetSubvector(0, first.MeanTimesPrecision);
            concatTimesFirst.Precision.SetSubmatrix(0, 0, first.Precision);
            concatTimesFirst.SetToProduct(concatTimesFirst, concat);
            concatTimesFirst.GetMarginal(dim1, result);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="FirstAverageConditional(Vector, VectorGaussian)"]/*'/>
        public static VectorGaussian FirstAverageConditional(Vector concat, VectorGaussian result)
        {
            result.Precision.SetAllElementsTo(0.0);
            for (int i = 0; i < result.Dimension; i++)
            {
                result.MeanTimesPrecision[i] = concat[i];
                result.Precision[i, i] = Double.PositiveInfinity;
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="SecondAverageConditional(Vector, Vector, VectorGaussian)"]/*'/>
        public static VectorGaussian SecondAverageConditional(Vector concat, Vector first, VectorGaussian result)
        {
            int dim1 = first.Count;
            result.Precision.SetAllElementsTo(0.0);
            for (int i = 0; i < result.Dimension; i++)
            {
                result.MeanTimesPrecision[i] = concat[i + dim1];
                result.Precision[i, i] = Double.PositiveInfinity;
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="SecondAverageConditional(Vector, VectorGaussian, VectorGaussian)"]/*'/>
        public static VectorGaussian SecondAverageConditional(Vector concat, VectorGaussian first, VectorGaussian result)
        {
            int dim1 = first.Dimension;
            result.Precision.SetAllElementsTo(0.0);
            for (int i = 0; i < result.Dimension; i++)
            {
                result.MeanTimesPrecision[i] = concat[i + dim1];
                result.Precision[i, i] = Double.PositiveInfinity;
            }
            return result;
        }

        // VMP //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="AverageLogFactor()"]/*'/>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="ConcatAverageLogarithmInit(VectorGaussian, VectorGaussian)"]/*'/>
        [Skip]
        public static VectorGaussian ConcatAverageLogarithmInit([IgnoreDependency] VectorGaussian first, [IgnoreDependency] VectorGaussian second)
        {
            return new VectorGaussian(first.Dimension + second.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="ConcatAverageLogarithmInit(Vector, VectorGaussian)"]/*'/>
        [Skip]
        public static VectorGaussian ConcatAverageLogarithmInit([IgnoreDependency] Vector first, [IgnoreDependency] VectorGaussian second)
        {
            return new VectorGaussian(first.Count + second.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="ConcatAverageLogarithmInit(VectorGaussian, Vector)"]/*'/>
        [Skip]
        public static VectorGaussian ConcatAverageLogarithmInit([IgnoreDependency] VectorGaussian first, [IgnoreDependency] Vector second)
        {
            return new VectorGaussian(first.Dimension + second.Count);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="ConcatAverageLogarithm(VectorGaussian, VectorGaussian, VectorGaussian)"]/*'/>
        [SkipIfAllUniform]
        public static VectorGaussian ConcatAverageLogarithm(VectorGaussian first, VectorGaussian second, VectorGaussian result)
        {
            return ConcatAverageConditional(first, second, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="ConcatAverageLogarithm(Vector, VectorGaussian, VectorGaussian)"]/*'/>
        public static VectorGaussian ConcatAverageLogarithm(Vector first, VectorGaussian second, VectorGaussian result)
        {
            return ConcatAverageConditional(first, second, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="ConcatAverageLogarithm(VectorGaussian, Vector, VectorGaussian)"]/*'/>
        public static VectorGaussian ConcatAverageLogarithm(VectorGaussian first, Vector second, VectorGaussian result)
        {
            return ConcatAverageConditional(first, second, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="FirstAverageLogarithm(VectorGaussian, Vector, VectorGaussian)"]/*'/>
        public static VectorGaussian FirstAverageLogarithm([SkipIfUniform] VectorGaussian concat, Vector second, VectorGaussian result)
        {
            return FirstAverageConditional(concat, second, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="FirstAverageLogarithm(VectorGaussian, VectorGaussian, VectorGaussian)"]/*'/>
        public static VectorGaussian FirstAverageLogarithm([SkipIfUniform] VectorGaussian concat, VectorGaussian second, VectorGaussian result)
        {
            // prec = concat.Precision[dim1,dim1]
            // meanTimesPrec = concat.MeanTimesPrecision[dim1] - concat.Precision[dim1,dim2]*second.Mean
            Vector mSecond = second.GetMean();
            return FirstAverageConditional(concat, mSecond, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="SecondAverageLogarithm(VectorGaussian, Vector, VectorGaussian)"]/*'/>
        public static VectorGaussian SecondAverageLogarithm([SkipIfUniform] VectorGaussian concat, Vector first, VectorGaussian result)
        {
            return SecondAverageConditional(concat, first, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="SecondAverageLogarithm(VectorGaussian, VectorGaussian, VectorGaussian)"]/*'/>
        public static VectorGaussian SecondAverageLogarithm([SkipIfUniform] VectorGaussian concat, VectorGaussian first, VectorGaussian result)
        {
            Vector mFirst = first.GetMean();
            return SecondAverageConditional(concat, mFirst, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="FirstAverageLogarithm(Vector, VectorGaussian)"]/*'/>
        public static VectorGaussian FirstAverageLogarithm(Vector concat, VectorGaussian result)
        {
            return FirstAverageConditional(concat, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="SecondAverageLogarithm(Vector, Vector, VectorGaussian)"]/*'/>
        public static VectorGaussian SecondAverageLogarithm(Vector concat, Vector first, VectorGaussian result)
        {
            return SecondAverageConditional(concat, first, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConcatOp"]/message_doc[@name="SecondAverageLogarithm(Vector, VectorGaussian, VectorGaussian)"]/*'/>
        public static VectorGaussian SecondAverageLogarithm(Vector concat, VectorGaussian first, VectorGaussian result)
        {
            return SecondAverageConditional(concat, first, result);
        }
    }
}

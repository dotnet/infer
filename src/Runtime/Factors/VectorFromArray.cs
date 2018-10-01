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

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorFromArrayOp"]/doc/*'/>
    [FactorMethod(new string[] { "vector", "array" }, typeof(Vector), "FromArray", typeof(double[]))]
    [Quality(QualityBand.Preview)]
    public static class VectorFromArrayOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorFromArrayOp"]/message_doc[@name="LogAverageFactor(Vector, double[])"]/*'/>
        public static double LogAverageFactor(Vector vector, double[] array)
        {
            for (int i = 0; i < array.Length; i++)
            {
                if (vector[i] != array[i])
                    return Double.NegativeInfinity;
            }
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorFromArrayOp"]/message_doc[@name="LogEvidenceRatio(Vector, double[])"]/*'/>
        public static double LogEvidenceRatio(Vector vector, double[] array)
        {
            return LogAverageFactor(vector, array);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorFromArrayOp"]/message_doc[@name="LogAverageFactor(Vector, IList{Gaussian})"]/*'/>
        public static double LogAverageFactor(Vector vector, IList<Gaussian> array)
        {
            double sum = 0.0;
            for (int i = 0; i < array.Count; i++)
            {
                sum += array[i].GetLogProb(vector[i]);
            }
            return sum;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorFromArrayOp"]/message_doc[@name="LogEvidenceRatio(Vector, IList{Gaussian})"]/*'/>
        public static double LogEvidenceRatio(Vector vector, IList<Gaussian> array)
        {
            return LogAverageFactor(vector, array);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorFromArrayOp"]/message_doc[@name="LogAverageFactor(VectorGaussian, VectorGaussian)"]/*'/>
        public static double LogAverageFactor(VectorGaussian vector, [Fresh] VectorGaussian to_vector)
        {
            return to_vector.GetLogAverageOf(vector);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorFromArrayOp"]/message_doc[@name="LogEvidenceRatio(VectorGaussian)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(VectorGaussian vector)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorFromArrayOp"]/message_doc[@name="AverageLogFactor()"]/*'/>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorFromArrayOp"]/message_doc[@name="VectorAverageConditional(IList{Gaussian}, VectorGaussian)"]/*'/>
        public static VectorGaussian VectorAverageConditional(
            [SkipIfAnyUniform] IList<Gaussian> array, VectorGaussian result) // TM: SkipIfAllUniform would be more accurate but leads to half-uniform distributions
        {
            return ArrayFromVectorOp.VectorAverageConditional(array, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorFromArrayOp"]/message_doc[@name="VectorAverageConditionalInit(IList{Gaussian})"]/*'/>
        [Skip]
        public static VectorGaussian VectorAverageConditionalInit([IgnoreDependency] IList<Gaussian> array)
        {
            return new VectorGaussian(array.Count);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorFromArrayOp"]/message_doc[@name="VectorAverageLogarithmInit(IList{Gaussian})"]/*'/>
        [Skip]
        public static VectorGaussian VectorAverageLogarithmInit([IgnoreDependency] IList<Gaussian> array)
        {
            return new VectorGaussian(array.Count);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorFromArrayOp"]/message_doc[@name="VectorAverageLogarithm(IList{Gaussian}, VectorGaussian)"]/*'/>
        public static VectorGaussian VectorAverageLogarithm(
            [SkipIfAnyUniform] IList<Gaussian> array, VectorGaussian result) // TM: SkipIfAllUniform would be more accurate but leads to half-uniform distributions
        {
            return VectorAverageConditional(array, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorFromArrayOp"]/message_doc[@name="ArrayAverageConditional{GaussianList}(VectorGaussian, IList{Gaussian}, GaussianList)"]/*'/>
        /// <typeparam name="GaussianList">The type of the outgoing message.</typeparam>
        public static GaussianList ArrayAverageConditional<GaussianList>(
            [SkipIfUniform] VectorGaussian vector, [NoInit] IList<Gaussian> array, GaussianList result)
            where GaussianList : IList<Gaussian>
        {
            return ArrayFromVectorOp.ArrayAverageConditional(array, vector, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorFromArrayOp"]/message_doc[@name="ArrayAverageLogarithm{GaussianList}(VectorGaussian, IList{Gaussian}, GaussianList)"]/*'/>
        /// <typeparam name="GaussianList">The type of the outgoing message.</typeparam>
        public static GaussianList ArrayAverageLogarithm<GaussianList>([SkipIfUniform] VectorGaussian vector, [Proper] IList<Gaussian> array, GaussianList result)
            where GaussianList : IList<Gaussian>
        {
            // prec[i] = vector[i].Prec
            // meanTimesPrec = vector.MeanTimesPrec - vector.Prec[:,noti]*array[noti].Mean
            //               = vector.MeanTimesPrec - vector.Prec*array.Mean + diag(invdiag(vector.Prec))*array.Mean
            if (result.Count != vector.Dimension)
                throw new ArgumentException("vector.Dimension (" + vector.Dimension + ") != result.Count (" + result.Count + ")");
            if (result.Count != array.Count)
                throw new ArgumentException("array.Count (" + array.Count + ") != result.Count (" + result.Count + ")");
            if (vector.IsPointMass)
                return ArrayAverageLogarithm(vector.Point, result);
            int length = result.Count;
            Vector mean = Vector.Zero(length);
            for (int i = 0; i < length; i++)
            {
                Gaussian item = array[i];
                mean[i] = item.GetMean();
            }
            Vector meanTimesPrecision = vector.Precision * mean;
            for (int i = 0; i < length; i++)
            {
                double prec = vector.Precision[i, i];
                if (Double.IsPositiveInfinity(prec))
                    throw new NotSupportedException("Singular VectorGaussians not supported");
                double mprec = vector.MeanTimesPrecision[i] - meanTimesPrecision[i] + prec * mean[i];
                result[i] = Gaussian.FromNatural(mprec, prec);
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorFromArrayOp"]/message_doc[@name="ArrayAverageConditional{GaussianList}(Vector, GaussianList)"]/*'/>
        /// <typeparam name="GaussianList">The type of the outgoing message.</typeparam>
        public static GaussianList ArrayAverageConditional<GaussianList>(Vector vector, GaussianList result)
            where GaussianList : IList<Gaussian>
        {
            return ArrayAverageLogarithm<GaussianList>(vector, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VectorFromArrayOp"]/message_doc[@name="ArrayAverageLogarithm{GaussianList}(Vector, GaussianList)"]/*'/>
        /// <typeparam name="GaussianList">The type of the outgoing message.</typeparam>
        public static GaussianList ArrayAverageLogarithm<GaussianList>(Vector vector, GaussianList result)
            where GaussianList : IList<Gaussian>
        {
            int length = result.Count;
            for (int i = 0; i < length; i++)
            {
                result[i] = Gaussian.PointMass(vector[i]);
            }
            return result;
        }
    }
}

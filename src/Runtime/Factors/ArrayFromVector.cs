// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "ArrayFromVector")]
    [Quality(QualityBand.Stable)]
    public static class ArrayFromVectorOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/message_doc[@name="LogAverageFactor(double[], Vector)"]/*'/>
        public static double LogAverageFactor(double[] array, Vector vector)
        {
            if (array.Length != vector.Count)
                return Double.NegativeInfinity;
            for (int i = 0; i < array.Length; i++)
            {
                if (array[i] != vector[i])
                    return Double.NegativeInfinity;
            }
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/message_doc[@name="LogEvidenceRatio(double[], Vector)"]/*'/>
        public static double LogEvidenceRatio(double[] array, Vector vector)
        {
            return LogAverageFactor(array, vector);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/message_doc[@name="AverageLogFactor(double[], Vector)"]/*'/>
        public static double AverageLogFactor(double[] array, Vector vector)
        {
            return LogAverageFactor(array, vector);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/message_doc[@name="LogAverageFactor(double[], VectorGaussian)"]/*'/>
        public static double LogAverageFactor(double[] array, VectorGaussian vector)
        {
            return vector.GetLogProb(Vector.FromArray(array));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/message_doc[@name="LogEvidenceRatio(double[], VectorGaussian)"]/*'/>
        public static double LogEvidenceRatio(double[] array, VectorGaussian vector)
        {
            return LogAverageFactor(array, vector);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/message_doc[@name="LogAverageFactor(double[], VectorGaussian)"]/*'/>
        public static double LogAverageFactor(double[] array, VectorGaussianMoments vector)
        {
            return vector.GetLogProb(Vector.FromArray(array));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/message_doc[@name="LogEvidenceRatio(double[], VectorGaussian)"]/*'/>
        public static double LogEvidenceRatio(double[] array, VectorGaussianMoments vector)
        {
            return LogAverageFactor(array, vector);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/message_doc[@name="LogAverageFactor(IList{Gaussian}, VectorGaussian, VectorGaussian)"]/*'/>
        public static double LogAverageFactor(IList<Gaussian> array, VectorGaussian vector, [Fresh] VectorGaussian to_vector)
        {
            return vector.GetLogAverageOf(to_vector);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/message_doc[@name="LogEvidenceRatio(IList{Gaussian}, VectorGaussian, VectorGaussian, IList{Gaussian})"]/*'/>
        public static double LogEvidenceRatio(IList<Gaussian> array, VectorGaussian vector, [Fresh] VectorGaussian to_vector, [Fresh] IList<Gaussian> to_array)
        {
            double result = LogAverageFactor(array, vector, to_vector);
            int length = array.Count;
            for (int i = 0; i < length; i++)
            {
                result -= to_array[i].GetLogAverageOf(array[i]);
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/message_doc[@name="LogAverageFactor(IList{Gaussian}, VectorGaussian, VectorGaussian)"]/*'/>
        public static double LogAverageFactor(IList<Gaussian> array, VectorGaussianMoments vector, [Fresh] VectorGaussianMoments to_vector)
        {
            return vector.GetLogAverageOf(to_vector);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/message_doc[@name="LogEvidenceRatio(IList{Gaussian}, VectorGaussian, VectorGaussian, IList{Gaussian})"]/*'/>
        public static double LogEvidenceRatio(IList<Gaussian> array, VectorGaussianMoments vector, [Fresh] VectorGaussianMoments to_vector, [Fresh] IList<Gaussian> to_array)
        {
            double result = LogAverageFactor(array, vector, to_vector);
            int length = array.Count;
            for (int i = 0; i < length; i++)
            {
                result -= to_array[i].GetLogAverageOf(array[i]);
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/message_doc[@name="AverageLogFactor(double[], VectorGaussian)"]/*'/>
        [Skip]
        public static double AverageLogFactor(double[] array, VectorGaussian vector)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/message_doc[@name="AverageLogFactor(IList{Gaussian}, VectorGaussian)"]/*'/>
        [Skip]
        public static double AverageLogFactor(IList<Gaussian> array, VectorGaussian vector)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/message_doc[@name="AverageLogFactor(double[], VectorGaussian)"]/*'/>
        [Skip]
        public static double AverageLogFactor(double[] array, VectorGaussianMoments vector)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/message_doc[@name="AverageLogFactor(IList{Gaussian}, VectorGaussian)"]/*'/>
        [Skip]
        public static double AverageLogFactor(IList<Gaussian> array, VectorGaussianMoments vector)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/message_doc[@name="AverageLogFactor(IList{Gaussian}, Vector)"]/*'/>
        [Skip]
        public static double AverageLogFactor(IList<Gaussian> array, Vector vector)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/message_doc[@name="VectorAverageConditional(IList{Gaussian}, VectorGaussian)"]/*'/>
        public static VectorGaussian VectorAverageConditional([SkipIfAllUniform] IList<Gaussian> array, VectorGaussian result)
        {
            return VectorAverageLogarithm(array, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/message_doc[@name="VectorAverageLogarithm(IList{Gaussian}, VectorGaussian)"]/*'/>
        public static VectorGaussian VectorAverageLogarithm([SkipIfAllUniform] IList<Gaussian> array, VectorGaussian result)
        {
            if (result.Dimension != array.Count)
                throw new ArgumentException("array.Count (" + array.Count + ") != result.Dimension (" + result.Dimension + ")");
            result.Precision.SetAllElementsTo(0.0);
            int length = array.Count;
            for (int i = 0; i < length; i++)
            {
                Gaussian item = array[i];
                result.Precision[i, i] = item.Precision;
                result.MeanTimesPrecision[i] = item.MeanTimesPrecision;
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/message_doc[@name="VectorAverageConditional(IList{Gaussian}, VectorGaussian)"]/*'/>
        public static VectorGaussianMoments VectorAverageConditional([SkipIfAllUniform] IList<Gaussian> array, VectorGaussianMoments result)
        {
            return VectorAverageLogarithm(array, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/message_doc[@name="VectorAverageLogarithm(IList{Gaussian}, VectorGaussian)"]/*'/>
        public static VectorGaussianMoments VectorAverageLogarithm([SkipIfAllUniform] IList<Gaussian> array, VectorGaussianMoments result)
        {
            if (result.Dimension != array.Count)
                throw new ArgumentException("array.Count (" + array.Count + ") != result.Dimension (" + result.Dimension + ")");
            result.Variance.SetAllElementsTo(0.0);
            int length = array.Count;
            for (int i = 0; i < length; i++)
            {
                Gaussian item = array[i];
                double m, v;
                item.GetMeanAndVariance(out m, out v);
                result.Variance[i, i] = v;
                result.Mean[i] = m;
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/message_doc[@name="VectorAverageConditional(double[], VectorGaussian)"]/*'/>
        public static VectorGaussian VectorAverageConditional(double[] array, VectorGaussian result)
        {
            return VectorAverageLogarithm(array, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/message_doc[@name="VectorAverageLogarithm(double[], VectorGaussian)"]/*'/>
        public static VectorGaussian VectorAverageLogarithm(double[] array, VectorGaussian result)
        {
            result.Point = result.Point;
            result.Point.SetTo(array);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/message_doc[@name="VectorAverageConditional(double[], VectorGaussian)"]/*'/>
        public static VectorGaussianMoments VectorAverageConditional(double[] array, VectorGaussianMoments result)
        {
            return VectorAverageLogarithm(array, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/message_doc[@name="VectorAverageLogarithm(double[], VectorGaussian)"]/*'/>
        public static VectorGaussianMoments VectorAverageLogarithm(double[] array, VectorGaussianMoments result)
        {
            result.Point = result.Point;
            result.Point.SetTo(array);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/message_doc[@name="ArrayAverageConditional{GaussianList}(IList{Gaussian}, VectorGaussian, GaussianList)"]/*'/>
        /// <typeparam name="GaussianList">The type of the resulting array.</typeparam>
        public static GaussianList ArrayAverageConditional<GaussianList>(
            [NoInit] IList<Gaussian> array, [SkipIfUniform] VectorGaussian vector, GaussianList result)
            where GaussianList : IList<Gaussian>
        {
            if (result.Count != vector.Dimension)
                throw new ArgumentException("vector.Dimension (" + vector.Dimension + ") != result.Count (" + result.Count + ")");
            int length = result.Count;
            bool allPointMass = array.All(g => g.IsPointMass);
            if (allPointMass)
            {
                // efficient special case
                for (int i = 0; i < length; i++)
                {
                    double x = array[i].Point;
                    // -prec*(x-m) = -prec*x + prec*m
                    double dlogp = vector.MeanTimesPrecision[i];
                    for (int j = 0; j < length; j++)
                    {
                        dlogp -= vector.Precision[i, j] * array[j].Point;
                    }
                    double ddlogp = -vector.Precision[i, i];
                    result[i] = Gaussian.FromDerivatives(x, dlogp, ddlogp, false);
                }
            }
            else if (vector.IsPointMass)
            {
                // efficient special case
                Vector mean = vector.Point;
                for (int i = 0; i < length; i++)
                {
                    result[i] = Gaussian.PointMass(mean[i]);
                }
            }
            else if (vector.IsUniform())
            {
                for (int i = 0; i < length; i++)
                {
                    result[i] = Gaussian.Uniform();
                }
            }
            else if (array.Any(g => g.IsPointMass))
            {
                // Z = N(m1; m2, V1+V2)
                // logZ = -0.5 (m1-m2)'inv(V1+V2)(m1-m2)
                // dlogZ = (m1-m2)'inv(V1+V2) dm2
                // ddlogZ = -dm2'inv(V1+V2) dm2
                Vector mean = Vector.Zero(length);
                PositiveDefiniteMatrix variance = new PositiveDefiniteMatrix(length, length);
                vector.GetMeanAndVariance(mean, variance);
                for (int i = 0; i < length; i++)
                {
                    if (array[i].IsUniform()) continue;
                    double m, v;
                    array[i].GetMeanAndVariance(out m, out v);
                    variance[i, i] += v;
                    mean[i] -= m;
                }
                PositiveDefiniteMatrix precision = variance.Inverse();
                Vector meanTimesPrecision = precision * mean;
                for (int i = 0; i < length; i++)
                {
                    if (array[i].IsUniform())
                    {
                        result[i] = Gaussian.FromMeanAndVariance(mean[i], variance[i, i]);
                    }
                    else
                    {
                        double alpha = meanTimesPrecision[i];
                        double beta = precision[i, i];
                        result[i] = GaussianOp.GaussianFromAlphaBeta(array[i], alpha, beta, false);
                    }
                }
            }
            else
            {
                // Compute inv(V1+V2)*(m1-m2) as inv(V2)*inv(inv(V1) + inv(V2))*(inv(V1)*m1 + inv(V2)*m2) - inv(V2)*m2 = inv(V2)*(m - m2)
                // Compute inv(V1+V2) as inv(V2)*inv(inv(V1) + inv(V2))*inv(V2) - inv(V2)
                PositiveDefiniteMatrix precision = (PositiveDefiniteMatrix)vector.Precision.Clone();
                Vector meanTimesPrecision = vector.MeanTimesPrecision.Clone();
                for (int i = 0; i < length; i++)
                {
                    Gaussian g = array[i];
                    precision[i, i] += g.Precision;
                    meanTimesPrecision[i] += g.MeanTimesPrecision;
                }
                bool fastMethod = true;
                if (fastMethod)
                {
                    bool isPosDef;
                    // this destroys precision
                    LowerTriangularMatrix precisionChol = precision.CholeskyInPlace(out isPosDef);
                    if (!isPosDef) throw new PositiveDefiniteMatrixException();
                    // variance = inv(precisionChol*precisionChol') = inv(precisionChol)'*inv(precisionChol) = varianceChol*varianceChol'
                    // this destroys meanTimesPrecision
                    var mean = meanTimesPrecision.PredivideBy(precisionChol);
                    mean = mean.PredivideByTranspose(precisionChol);
                    var varianceCholTranspose = precisionChol;
                    // this destroys precisionChol
                    varianceCholTranspose.SetToInverse(precisionChol);
                    for (int i = 0; i < length; i++)
                    {
                        Gaussian g = array[i];
                        double variance_ii = GetSquaredLengthOfColumn(varianceCholTranspose, i);
                        // works when g is uniform, but not when g is point mass
                        result[i] = Gaussian.FromMeanAndVariance(mean[i], variance_ii) / g;
                    }
                }
                else
                {
                    // equivalent to above, but slower
                    PositiveDefiniteMatrix variance = precision.Inverse();
                    var mean = variance * meanTimesPrecision;
                    for (int i = 0; i < length; i++)
                    {
                        Gaussian g = array[i];
                        // works when g is uniform, but not when g is point mass
                        result[i] = Gaussian.FromMeanAndVariance(mean[i], variance[i, i]) / g;
                    }
                }
            }
            return result;
        }

        private static double GetSquaredLengthOfRow(LowerTriangularMatrix L, int row)
        {
            double sum = 0;
            for (int col = 0; col <= row; col++)
            {
                double element = L[row, col];
                sum += element * element;
            }
            return sum;
        }

        private static double GetSquaredLengthOfColumn(LowerTriangularMatrix L, int column)
        {
            int rows = L.Rows;
            double sum = 0;
            for (int row = column; row < rows; row++)
            {
                double element = L[row, column];
                sum += element * element;
            }
            return sum;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/message_doc[@name="ArrayAverageConditionalInit(VectorGaussian)"]/*'/>
        [Skip]
        public static DistributionStructArray<Gaussian, double> ArrayAverageConditionalInit([IgnoreDependency] VectorGaussian vector)
        {
            return new DistributionStructArray<Gaussian, double>(vector.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/message_doc[@name="ArrayAverageConditional{GaussianList}(IList{Gaussian}, VectorGaussianMoments, GaussianList)"]/*'/>
        /// <typeparam name="GaussianList">The type of the resulting array.</typeparam>
        public static GaussianList ArrayAverageConditional<GaussianList>(
            [NoInit] IList<Gaussian> array, [SkipIfUniform] VectorGaussianMoments vector, GaussianList result)
            where GaussianList : IList<Gaussian>
        {
            if (result.Count != vector.Dimension)
                throw new ArgumentException("vector.Dimension (" + vector.Dimension + ") != result.Count (" + result.Count + ")");
            int length = result.Count;
            if (array.All(g => g.IsUniform()) || vector.IsUniform() || vector.IsPointMass)
            {
                for (int i = 0; i < length; i++)
                {
                    result[i] = Gaussian.FromMeanAndVariance(vector.Mean[i], vector.Variance[i, i]);
                }
            }
            else
            {
                // Z = N(m1; m2, V1+V2)
                // logZ = -0.5 (m1-m2)'inv(V1+V2)(m1-m2)
                // dlogZ = (m1-m2)'inv(V1+V2) dm2
                // ddlogZ = -dm2'inv(V1+V2) dm2
                VectorGaussianMoments product = new VectorGaussianMoments(length);
                Vector mean = product.Mean;
                PositiveDefiniteMatrix variance = product.Variance;
                vector.GetMeanAndVariance(mean, variance);
                for (int i = 0; i < length; i++)
                {
                    double m, v;
                    array[i].GetMeanAndVariance(out m, out v);
                    variance[i, i] += v;
                    mean[i] -= m;
                }
                double offset = 1e-10;
                for (int i = 0; i < length; i++)
                {
                    variance[i, i] += offset;
                }
                PositiveDefiniteMatrix precision = variance.Inverse();
                Vector meanTimesPrecision = precision * mean;
                double precisionThreshold = 1;
                if (array.Any(g => g.Precision <= precisionThreshold))
                {
                    // compute the posterior mean and variance
                    product.SetToProduct(vector, array);
                    mean = product.Mean;
                    variance = product.Variance;
                }
                for (int i = 0; i < length; i++)
                {
                    if (array[i].Precision <= precisionThreshold)
                    {
                        result[i] = Gaussian.FromMeanAndVariance(mean[i], variance[i, i]) / array[i];
                    }
                    else
                    {
                        double alpha = meanTimesPrecision[i];
                        double beta = precision[i, i];
                        if (double.IsNaN(alpha)) throw new Exception("alpha is NaN");
                        if (double.IsNaN(beta)) throw new Exception("beta is NaN");
                        result[i] = GaussianOp.GaussianFromAlphaBeta(array[i], alpha, beta, false);
                        if (double.IsNaN(result[i].MeanTimesPrecision)) throw new Exception("result is NaN");
                        if (result[i].Precision < 0) throw new Exception("result is improper");
                    }
                }
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ArrayFromVectorOp"]/message_doc[@name="ArrayAverageLogarithm{GaussianList}(VectorGaussian, GaussianList)"]/*'/>
        /// <typeparam name="GaussianList">The type of the resulting array.</typeparam>
        public static GaussianList ArrayAverageLogarithm<GaussianList>([SkipIfUniform] VectorGaussian vector, GaussianList result)
            where GaussianList : IList<Gaussian>
        {
            if (result.Count != vector.Dimension)
                throw new ArgumentException("vector.Dimension (" + vector.Dimension + ") != result.Count (" + result.Count + ")");
            int length = result.Count;
            Vector mean = Vector.Zero(length);
            PositiveDefiniteMatrix variance = new PositiveDefiniteMatrix(length, length);
            vector.GetMeanAndVariance(mean, variance);
            for (int i = 0; i < length; i++)
            {
                result[i] = Gaussian.FromMeanAndVariance(mean[i], variance[i, i]);
            }
            return result;
        }
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Linq;
    using System.Collections.Generic;
    using System.Runtime.Serialization;
    using Math;
    using Utilities;
    using Factors.Attributes;
    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// Represents a multivariate Gaussian distribution.
    /// </summary>
    /// <remarks><para>
    /// The distribution is parameterized by Mean and Variance.
    /// </para><para>
    /// Some special cases:
    /// If the variance is infinite along the diagonal, then the distribution is uniform.
    /// If the variance is zero, then the distribution is a point mass.  The Point property
    /// gives the location of the point mass.
    /// If variance[i,i] is zero, then the distribution is a point mass along dimension i.  Point[i] gives the mean.
    /// The rest of the row and column of variance must be zero.
    /// </para><para>
    /// The formula for the distribution is:
    /// <c>N(x;m,v) = |2*pi*v|^(-d/2) * exp(-0.5 (x-m)' inv(v) (x-m))</c>.
    /// When v=0, this reduces to delta(x-m).
    /// When v=infinity, the density is redefined to be 1.
    /// When v is singular, the density is redefined to be <c>exp(-0.5 (x-m)' inv(v) (x-m))</c>, 
    /// i.e. we drop the terms <c>|2*pi*v|^(-d/2)</c>.
    /// </para></remarks>
    [Serializable]
    [DataContract]
    [Quality(QualityBand.Mature)]
    public class VectorGaussianMoments : IDistribution<Vector>,
                                  SettableTo<VectorGaussianMoments>, SettableToProduct<VectorGaussianMoments>, Diffable, SettableToUniform,
                                  SettableToRatio<VectorGaussianMoments>, SettableToPower<VectorGaussianMoments>, SettableToWeightedSum<VectorGaussianMoments>,
                                  Sampleable<Vector>,
                                  CanGetMean<DenseVector>, CanGetVariance<PositiveDefiniteMatrix>, CanGetMeanAndVariance<Vector, PositiveDefiniteMatrix>,
                                  CanSetMeanAndVariance<Vector, PositiveDefiniteMatrix>,
                                  CanGetLogAverageOf<VectorGaussianMoments>, CanGetLogAverageOfPower<VectorGaussianMoments>,
                                  CanGetAverageLog<VectorGaussianMoments>, CanGetLogNormalizer, CanGetMode<DenseVector>
    {
        [DataMember]
        private DenseVector mean;
        // variance may not always be positive definite, but we give it that type anyway.
        [DataMember]
        private PositiveDefiniteMatrix variance;

        /// <summary>
        /// Gets/Sets the mean vector
        /// </summary>
        public Vector Mean
        {
            get { return mean; }
            set
            {
                if (mean == null)
                    mean = (DenseVector)value;
                else if (!ReferenceEquals(mean, value))
                    mean.SetTo(value);
            }
        }

        /// <summary>
        /// Gets/sets the variance matrix
        /// </summary>
        public PositiveDefiniteMatrix Variance
        {
            get { return variance; }
            set
            {
                if (variance == null)
                    variance = value;
                else if (!ReferenceEquals(variance, value))
                    variance.SetTo(value);
            }
        }

        /// <summary>
        /// Gets the mean and variance of the distribution.
        /// </summary>
        /// <param name="mean">Modified to contain the mean.  Must already be the correct size.</param>
        /// <param name="variance">Modified to contain the covariance matrix.  Must already be the correct size.</param>
        /// <remarks>Because the Gaussian stored its parameters in exponential form, it is more efficient
        /// to compute the mean and variance at the same time rather than separately.</remarks>
        public void GetMeanAndVariance(Vector mean, PositiveDefiniteMatrix variance)
        {
            mean.SetTo(Mean);
            variance.SetTo(Variance);
        }

        /// <summary>
        /// Gets the mean vector and precision matrix of the distribution
        /// </summary>
        /// <param name="mean">Modified to contain the mean.  Must already be the correct size.</param>
        /// <param name="precision">Modified to contain the precision matrix.  Must already be the correct size.</param>
        public void GetMeanAndPrecision(Vector mean, PositiveDefiniteMatrix precision)
        {
            if (IsPointMass)
            {
                precision.SetToIdentityScaledBy(Double.PositiveInfinity);
                mean.SetTo(Point);
            }
            else if (IsUniform())
            {
                precision.SetAllElementsTo(0.0);
                mean.SetAllElementsTo(0.0);
            }
            else
            {
                bool[] wasZero = null;
                bool anyDiagonalIsZero = false;
                for (int i = 0; i < Dimension; i++)
                {
                    if (variance[i, i] == 0) anyDiagonalIsZero = true;
                }
                if (anyDiagonalIsZero)
                {
                    wasZero = new bool[Dimension];
                    for (int i = 0; i < Dimension; i++)
                    {
                        if (variance[i, i] == 0)
                        {
                            wasZero[i] = true;
                            variance[i, i] = double.Epsilon;
                        }
                    }
                }
                precision.SetToInverse(variance);
                mean.SetTo(Mean);
                if (anyDiagonalIsZero)
                {
                    for (int i = 0; i < Dimension; i++)
                    {
                        if (wasZero[i])
                        {
                            mean[i] = Point[i];
                            precision[i, i] = double.PositiveInfinity;
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Sets the mean and variance of the distribution.
        /// </summary>
        /// <param name="mean">The mean vector.  Cannot be the same object as <c>this.MeanTimesPrecision</c>.</param>
        /// <param name="variance">The covariance matrix.  Can be the same object as <c>this.Precision</c>.</param>
        public void SetMeanAndVariance(Vector mean, PositiveDefiniteMatrix variance)
        {
            Mean = mean;
            Variance = variance;
            for (int i = 0; i < Dimension; i++)
            {
                if (variance[i, i] == 0)
                    Point[i] = mean[i];
            }
        }

        /// <summary>
        /// Sets the mean and precision of the distribution.
        /// </summary>
        /// <param name="mean">The mean vector</param>
        /// <param name="precision">The precision matrix</param>
        public void SetMeanAndPrecision(Vector mean, PositiveDefiniteMatrix precision)
        {
            if (precision.EqualsAll(0.0))
            {
                SetToUniform();
            }
            else if (precision.EnumerateDiagonal().All(Double.IsPositiveInfinity))
            {
                Point = mean;
            }
            else
            {
                // check for zeros on the diagonal
                bool hasZeros = false;
                for (int i = 0; i < Dimension; i++)
                {
                    if (precision[i, i] == 0)
                    {
                        hasZeros = true;
                        bool rowColZero = true;
                        for (int j = 0; j < Dimension; j++)
                        {
                            if (precision[i, j] != 0 || precision[j, i] != 0)
                            {
                                rowColZero = false;
                                break;
                            }
                        }
                        if (!rowColZero) throw new PositiveDefiniteMatrixException();
                    }
                }
                if (hasZeros)
                {
                    PositiveDefiniteMatrix precision2 = (PositiveDefiniteMatrix)precision.Clone();
                    for (int i = 0; i < Dimension; i++)
                    {
                        if (precision[i, i] == 0) precision2[i, i] = 1;
                    }
                    Variance.SetToInverse(precision2);
                    for (int i = 0; i < Dimension; i++)
                    {
                        if (precision[i, i] == 0) variance[i, i] = Double.PositiveInfinity;
                    }
                }
                else
                {
                    Variance.SetToInverse(precision);
                }
                Mean = mean;
            }
        }

        /// <summary>
        /// Sets the natural parameters of the distribution (mean times precision, and precision)
        /// </summary>
        /// <param name="meanTimesPrecision">Mean times precision</param>
        /// <param name="precision">The precision matrix</param>
        public void SetNatural(Vector meanTimesPrecision, PositiveDefiniteMatrix precision)
        {
            VectorGaussian vg = VectorGaussian.FromNatural(meanTimesPrecision, precision);
            vg.GetMeanAndVariance(Mean, Variance);
        }

        /// <summary>
        /// The most likely value
        /// </summary>
        /// <returns>A new Vector.</returns>
        public DenseVector GetMode()
        {
            return GetMean();
        }

        /// <summary>
        /// Gets the mean of the distribution.
        /// </summary>
        /// <returns>A new Vector.</returns>
        public DenseVector GetMean()
        {
            return (DenseVector)GetMean(DenseVector.Zero(Dimension));
        }

        /// <summary>
        /// Gets the mean of the distribution
        /// </summary>
        /// <param name="result">Where to place the mean value</param>
        /// <returns>mean</returns>
        public Vector GetMean(Vector result)
        {
            if (IsPointMass) result.SetTo(Point);
            else result.SetTo(mean);
            return result;
        }

        /// <summary>
        /// Gets the variance-covariance matrix of the distribution.
        /// </summary>
        /// <returns>A new PositiveDefiniteMatrix.</returns>
        public PositiveDefiniteMatrix GetVariance()
        {
            return Variance;
        }

        /// <summary>
        /// Gets the variance-covariance matrix of the distribution.
        /// </summary>
        /// <param name="variance">Where to place the variance-covariance</param>
        /// <returns>variance</returns>
        public PositiveDefiniteMatrix GetVariance(PositiveDefiniteMatrix variance)
        {
            variance.SetTo(Variance);
            return variance;
        }

        /// <summary>
        /// Asks whether the instance is a point mass
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public bool IsPointMass
        {
            get
            {
                return variance.EqualsAll(0);
            }
        }

        /// <summary>
        /// Sets this instance to a point mass. The location of the
        /// point mass is the existing mean vector
        /// </summary>
        protected void SetToPointMass()
        {
            variance.SetAllElementsTo(0.0);
        }

        /// <summary>
        /// Sets/gets the instance as a point mass
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public Vector Point
        {
            get
            {
                // The accessor must succeed, even if the distribution is not a point mass.
                //Assert.IsTrue(IsPointMass, "The distribution is not a point mass");
                return mean;
            }
            set
            {
                SetToPointMass();
                Mean = value;
            }
        }

        /// <summary>
        /// The dimension of the VectorGaussian domain
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public int Dimension
        {
            get { return mean.Count; }
        }

        //public bool IsCompatibleWith(IDistribution<Vector> thatd)
        //{
        //  VectorGaussian that = thatd as VectorGaussian;
        //  if (that == null) return false;
        //  return (that.Dimension == Dimension);
        //}
        //bool IDistribution<ICursor>.IsCompatibleWith(IDistribution<ICursor> thatd)
        //{
        //  Gaussian that = thatd as Gaussian;
        //  if (that == null) return false;
        //  return (that.Dimension == Dimension);
        //}

        /// <summary>
        /// Sets this VectorGaussian instance to be a uniform distribution
        /// </summary>
        public void SetToUniform()
        {
            variance.SetToIdentityScaledBy(double.PositiveInfinity);
            mean.SetAllElementsTo(0);
        }

        /// <summary>
        /// Asks whether this instance is uniform
        /// </summary>
        /// <returns>True if uniform, false otherwise</returns>
        public bool IsUniform()
        {
            return variance.EnumerateDiagonal().All(double.IsPositiveInfinity);
        }

        /// <summary>
        /// Evaluates the log of the multivariate Gaussian density.
        /// </summary>
        /// <param name="x">Point to evaluate the density at.</param>
        /// <returns>log p(x)</returns>
        public double GetLogProb(Vector x)
        {
            return VectorGaussian.GetLogProb(x, Mean, Variance);
        }

        /// <summary>
        /// Gets the normalizer for the VectorGaussian density function
        /// </summary>
        /// <returns></returns>
        public double GetLogNormalizer()
        {
            LowerTriangularMatrix varianceChol = new LowerTriangularMatrix(Dimension, Dimension);
            bool isPosDef = varianceChol.SetToCholesky(Variance);
            if (!isPosDef) return 0.0;
            DenseVector meanTimesPrecChol = DenseVector.Zero(Dimension);
            meanTimesPrecChol.SetTo(mean);
            meanTimesPrecChol.PredivideBy(varianceChol);
            // 0.5*LogDeterminant(precision) = 0.5*2*L.TraceLn()
            double result = 0.5 * meanTimesPrecChol.Inner(meanTimesPrecChol);
            for (int i = 0; i < Dimension; i++)
            {
                if (!Double.IsPositiveInfinity(variance[i, i]))
                    result += MMath.LnSqrt2PI + Math.Log(varianceChol[i, i]);
            }
            return result;
        }

        /// <summary>
        /// The expected logarithm of that distribution under this distribution.
        /// </summary>
        /// <param name="that">The distribution to take the logarithm of.</param>
        /// <returns><c>sum_x this.Evaluate(x)*Math.Log(that.Evaluate(x))</c></returns>
        /// <remarks>This is also known as the cross entropy.</remarks>
        public double GetAverageLog(VectorGaussianMoments that)
        {
            if (that.IsPointMass)
            {
                if (this.IsPointMass && this.Point.Equals(that.Point)) return 0.0;
                else return Double.NegativeInfinity;
            }
            else
            {
                // E[-0.5 logDet(2*pi*that.v) -0.5 (x-that.m)' inv(that.v) (x-that.m)]
                // = -0.5 logDet(2*pi*that.v) -0.5 (m-that.m)' inv(that.v) (m-that.m) -0.5 tr(v inv(that.v))
                Vector thatMean = Vector.Zero(Dimension);
                PositiveDefiniteMatrix thatPrecision = new PositiveDefiniteMatrix(Dimension, Dimension);
                that.GetMeanAndPrecision(thatMean, thatPrecision);
                return that.GetLogProb(Mean) - 0.5 * Matrix.TraceOfProduct(Variance, thatPrecision);
            }
        }

        /// <summary>
        /// Log-integral of the product of this VectorGaussian with that VectorGaussian
        /// </summary>
        /// <param name="that">That VectorGaussian</param>
        /// <returns>The log inner product</returns>
        public double GetLogAverageOf(VectorGaussianMoments that)
        {
            if (IsPointMass)
            {
                return that.GetLogProb(Point);
            }
            else if (that.IsPointMass)
            {
                return GetLogProb(that.Point);
            }
            else
            {
                VectorGaussianMoments product = this * that;
                //if (!product.IsProper()) throw new ArgumentException("The product is improper.");
                double result = product.GetLogNormalizer() - this.GetLogNormalizer() - that.GetLogNormalizer();
                return result;
            }
        }

        /// <summary>
        /// Get the integral of this distribution times another distribution raised to a power.
        /// </summary>
        /// <param name="that"></param>
        /// <param name="power"></param>
        /// <returns></returns>
        public double GetLogAverageOfPower(VectorGaussianMoments that, double power)
        {
            if (IsPointMass)
            {
                return power * that.GetLogProb(Point);
            }
            else if (that.IsPointMass)
            {
                if (power < 0) throw new DivideByZeroException("The exponent is negative and the distribution is a point mass");
                return this.GetLogProb(that.Point);
            }
            else
            {
                var product = this * (that ^ power);
                double result = product.GetLogNormalizer() - this.GetLogNormalizer() - power * that.GetLogNormalizer();
                return result;
            }
        }

        /// <summary>
        /// Samples from this VectorGaussian distribution
        /// </summary>
        /// <param name="result">Where to put the result</param>
        /// <returns>The sample value</returns>
        [Stochastic]
        public Vector Sample(Vector result)
        {
            Rand.NormalP(Mean, Variance.Inverse(), result);
            return result;
        }

        /// <summary>
        /// Sample from this VectorGaussian distribution
        /// </summary>
        /// <returns>The sample value</returns>
        [Stochastic]
        public Vector Sample()
        {
            return Sample(Vector.Zero(Dimension));
        }

        /// <summary>
        /// Sets this VectorGaussian instance to have the parameter values of that VectorGaussian instance
        /// </summary>
        /// <param name="that">That VectorGaussian</param>
        public void SetTo(VectorGaussianMoments that)
        {
            if (object.ReferenceEquals(that, null)) SetToUniform();
            else
            {
                Mean.SetTo(that.Mean);
                Variance.SetTo(that.Variance);
            }
        }

        public void SetToProduct(VectorGaussianMoments vector, IList<Gaussian> array)
        {
            int length = array.Count;
            bool useAccurateMethod = false;
            if (useAccurateMethod)
            {
                // This method does not work for improper array.
                bool useSlowMethod = true;
                if (useSlowMethod)
                {
                    // slow method
                    VectorGaussianMoments arrayMoments = new VectorGaussianMoments(length);
                    for (int i = 0; i < length; i++)
                    {
                        double m, v;
                        array[i].GetMeanAndVariance(out m, out v);
                        arrayMoments.Mean[i] = m;
                        arrayMoments.Variance[i, i] = v;
                    }
                    SetToProduct(vector, arrayMoments);
                }
                else
                {
                    // When computing the variance, any axis which is a point mass (in either array or vector) must remain a point mass.
                    // To avoid inverting a singular matrix, we must set these axes to a dummy value (like 1) and then zero them out at the end.
                    PositiveDefiniteMatrix variance = (PositiveDefiniteMatrix)vector.Variance.Clone();
                    PositiveDefiniteMatrix eigenvectors = new PositiveDefiniteMatrix(length, length);
                    eigenvectors.SetToEigenvectorsOfSymmetric(variance);
                    // at this point, vector.Variance = eigenvectors * variance * eigenvectors.Transpose()
                    Vector arrayInvStds = Vector.Zero(length);
                    PositiveDefiniteMatrix precision = variance;
                    PositiveDefiniteMatrix arrayPrecision = new PositiveDefiniteMatrix(length, length);
                    Vector arrayMean = Vector.Zero(length);
                    for (int i = 0; i < length; i++)
                    {
                        double m, v;
                        array[i].GetMeanAndVariance(out m, out v);
                        precision[i, i] = 1 / Math.Max(0, variance[i, i]);
                        arrayPrecision[i, i] = 1 / v;
                        arrayMean[i] = m;
                        arrayInvStds[i] = 1 / Math.Sqrt(v);
                    }
                    var scaledEigenvectors = (PositiveDefiniteMatrix)eigenvectors.Clone();
                    scaledEigenvectors.ScaleRows(arrayInvStds);
                    PositiveDefiniteMatrix rotatedArrayPrecision = new PositiveDefiniteMatrix(length, length);
                    rotatedArrayPrecision.SetToOuterTranspose(scaledEigenvectors);
                    // rotatedArrayPrecision = eigenvectors'*inv(V2)*eigenvectors

                    eigenvectors.SetToTranspose(eigenvectors);
                    Vector rotatedMean = eigenvectors * vector.Mean;
                    Vector extraMean = Vector.Zero(length);
                    for (int i = 0; i < length; i++)
                    {
                        if (double.IsPositiveInfinity(precision[i, i]))
                        {
                            precision[i, i] = 1;
                            // this ensures that the axis is zeroed out in the end.
                            SetRowToZero(eigenvectors, i);
                            SetToSumWithRow(extraMean, rotatedMean[i], eigenvectors, i);
                            rotatedMean[i] = 0;
                        }
                    }
                    var meanTimesPrecision = precision * rotatedMean;
                    Vector rotatedArrayMean = eigenvectors * arrayMean;
                    var arrayMeanTimesPrecision = rotatedArrayPrecision * rotatedArrayMean;
                    meanTimesPrecision.SetToSum(meanTimesPrecision, arrayMeanTimesPrecision);
                    meanTimesPrecision = eigenvectors.Transpose() * meanTimesPrecision;

                    precision.SetToSum(precision, rotatedArrayPrecision);
                    bool isPosDef;
                    var cholesky = precision.CholeskyInPlace(out isPosDef);
                    // inv(L*L') = inv(L')*inv(L)
                    eigenvectors.PredivideBy(cholesky);
                    // variance = inv(inv(V1)+inv(V2)) = H*inv(H'*inv(V1)*H + H'*inv(V2)*H)*H'
                    this.Variance.SetToOuterTranspose(eigenvectors);
                    if (this.Variance.Any(double.IsNaN)) throw new Exception();

                    // mean = variance * (inv(V1)*m1 + inv(V2)*m2) = variance * H*((H'*inv(V1)*H)*(H'*m1) + (H'*inv(V2)*H)*(H'*m2))
                    this.Mean.SetToProduct(this.Variance, meanTimesPrecision);
                    this.Mean.SetToSum(this.Mean, extraMean);
                }
            }
            else 
            {
                double offset = 1e-10;
                PositiveDefiniteMatrix variance = this.Variance;
                variance.SetTo(vector.Variance);
                // add a multiple of the identity to make it invertible.
                for (int i = 0; i < length; i++)
                {
                    variance[i, i] += offset;
                }
                LowerTriangularMatrix cholesky = new LowerTriangularMatrix(length, length);
                bool isPosDef = cholesky.SetToCholesky(variance);
                if (!isPosDef) throw new PositiveDefiniteMatrixException();
                cholesky.SetToInverse(cholesky);
                variance.SetToOuterTranspose(cholesky);
                for (int i = 0; i < length; i++)
                {
                    variance[i, i] += array[i].IsPointMass ? 1 / offset : array[i].Precision;
                }
                // variance = (inv(V1)+inv(V2))
                isPosDef = cholesky.SetToCholesky(variance);
                if (!isPosDef) throw new PositiveDefiniteMatrixException();
                cholesky.SetToInverse(cholesky);
                // this.Variance = inv(inv(V1)+inv(V2))
                this.Variance.SetToOuterTranspose(cholesky);
                if (this.Variance.Any(double.IsNaN)) throw new Exception("result is NaN");
                // mean = variance * (inv(V1)*m1 + inv(V2)*m2) 
                //      = m1 + variance * inv(V2) * (m2 - m1)
                Vector meanDiff = this.Mean;
                for (int i = 0; i < length; i++)
                {
                    if (array[i].IsPointMass)
                    {
                        meanDiff[i] = (array[i].GetMean() - vector.Mean[i]) / offset;
                    }
                    else
                    {
                        meanDiff[i] = array[i].MeanTimesPrecision - vector.Mean[i] * array[i].Precision;
                    }
                }
                if (meanDiff.Any(double.IsNaN)) throw new Exception("result is NaN");
                this.Mean.SetToSum(vector.Mean, this.Variance * meanDiff);
            }
        }

        private static void SetRowAndColumnToZero(Matrix matrix, int dim)
        {
            for (int i = 0; i < matrix.Rows; i++)
            {
                matrix[i, dim] = 0;
                matrix[dim, i] = 0;
            }
        }

        private static void SetRowToZero(Matrix matrix, int dim)
        {
            for (int i = 0; i < matrix.Cols; i++)
            {
                matrix[dim, i] = 0;
            }
        }

        private static void SetColumnToZero(Matrix matrix, int dim)
        {
            for (int i = 0; i < matrix.Rows; i++)
            {
                matrix[i, dim] = 0;
            }
        }

        private static void SetOffDiagonalToZero(Matrix matrix)
        {
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Cols; j++)
                {
                    matrix[i, j] = 0;
                }
            }
        }

        private static void SetToSumWithRow(Vector vector, double scale, Matrix matrix, int row)
        {
            for (int i = 0; i < vector.Count; i++)
            {
                vector[i] += scale * matrix[row, i];
            }
        }

        /// <summary>
        /// Sets the parameters to represent the product of two VectorGaussians.
        /// </summary>
        /// <param name="g1">The first VectorGaussian.  May refer to <c>this</c>.</param>
        /// <param name="g2">The second VectorGaussian.  May refer to <c>this</c>.</param>
        /// <remarks>
        /// The result may not be proper. No error is thrown in this case.
        /// </remarks>
        public void SetToProduct(VectorGaussianMoments g1, VectorGaussianMoments g2)
        {
            int length = g1.Dimension;
            PositiveDefiniteMatrix variance = Variance;
            variance.SetTo(g1.Variance);
            Matrix eigenvectors1 = new Matrix(length, length);
            eigenvectors1.SetToEigenvectorsOfSymmetric(variance);
            DenseVector eigenvalues1 = DenseVector.Zero(length);
            eigenvalues1.SetToDiagonal(variance);
            double maxEigenvalue = eigenvalues1.Max(v => double.IsPositiveInfinity(v) ? 0 : v);
            double small = maxEigenvalue * 1e-15;
            DenseVector invSqrtEigenvalues1 = eigenvalues1;
            for (int i = 0; i < length; i++)
            {
                if (eigenvalues1[i] <= small)
                {
                    invSqrtEigenvalues1[i] = 1;
                    for (int j = 0; j < length; j++)
                    {
                        eigenvectors1[j, i] = 0;
                    }
                }
                else
                {
                    invSqrtEigenvalues1[i] = 1 / Math.Sqrt(eigenvalues1[i]);
                }
            }
            PositiveDefiniteMatrix projection1 = new PositiveDefiniteMatrix(length, length);
            projection1.SetToOuter(eigenvectors1);
            eigenvectors1.ScaleCols(invSqrtEigenvalues1);

            variance.SetTo(g2.Variance);
            Matrix eigenvectors2 = new Matrix(length, length);
            eigenvectors2.SetToEigenvectorsOfSymmetric(variance);
            if (eigenvectors2.Any(double.IsNaN)) throw new Exception("eigenvectors are NaN");
            DenseVector eigenvalues2 = DenseVector.Zero(length);
            eigenvalues2.SetToDiagonal(variance);
            maxEigenvalue = eigenvalues2.Max(v => double.IsPositiveInfinity(v) ? 0 : v);
            small = maxEigenvalue * 1e-15;
            DenseVector invSqrtEigenvalues2 = eigenvalues2;
            for (int i = 0; i < length; i++)
            {
                if (eigenvalues2[i] <= small)
                {
                    invSqrtEigenvalues2[i] = 1;
                    for (int j = 0; j < length; j++)
                    {
                        eigenvectors2[j, i] = 0;
                    }
                }
                else
                {
                    invSqrtEigenvalues2[i] = 1 / Math.Sqrt(eigenvalues2[i]);
                }
            }
            PositiveDefiniteMatrix projection2 = new PositiveDefiniteMatrix(length, length);
            projection2.SetToOuter(eigenvectors2);
            eigenvectors2.ScaleCols(invSqrtEigenvalues2);

            // Find the intersection of subspaces.
            // Reference:
            // "Projectors on intersections of subspaces"
            // Adi Ben–Israel
            // http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf
            Matrix projection = new Matrix(length, length);
            projection.SetToProduct(projection1, projection2);
            Matrix singularValues = (Matrix)projection.Clone();
            Matrix intersection = new Matrix(length, length);
            intersection.SetToRightSingularVectors(singularValues);
            // singularValues is now the left singular vectors, scaled by singular values.

            intersection.SetToTranspose(intersection);
            Matrix rotatedEigenvectors1 = intersection * eigenvectors1;
            PositiveDefiniteMatrix rotatedInvA = new PositiveDefiniteMatrix(length, length);
            rotatedInvA.SetToOuter(rotatedEigenvectors1);
            Matrix rotatedEigenvectors2 = intersection * eigenvectors2;
            PositiveDefiniteMatrix rotatedInvB = new PositiveDefiniteMatrix(length, length);
            rotatedInvB.SetToOuter(rotatedEigenvectors2);
            PositiveDefiniteMatrix sum = rotatedInvA;
            sum.SetToSum(rotatedInvA, rotatedInvB);
            bool[] isNullspace = new bool[length];
            for (int i = 0; i < length; i++)
            {
                double sumOfSquares = 0;
                for (int j = 0; j < length; j++)
                {
                    double element = singularValues[j, i];
                    sumOfSquares += element * element;
                }
                if (sumOfSquares < 1 - 1e-4)
                {
                    isNullspace[i] = true;
                }
            }
            Matrix sumCorner = new Matrix(length, length);
            Matrix nullspace = new Matrix(length, length);
            for (int i = 0; i < length; i++)
            {
                if (isNullspace[i])
                {
                    for (int j = 0; j < length; j++)
                    {
                        nullspace[i, j] = intersection[i, j];
                        intersection[i, j] = 0;
                        sum[i, j] = 0;
                    }
                }
                else
                {
                    for (int j = 0; j < length; j++)
                    {
                        if (isNullspace[j])
                        {
                            sumCorner[i, j] = sum[i, j];
                            sum[i, j] = 0;
                        }
                    }
                }
            }
            var sumNullspace = nullspace * (PositiveDefiniteMatrix.IdentityScaledBy(length, 2) - projection1 - projection2) * nullspace.Transpose();
            sum.SetToSum(sum, sumNullspace);
            bool isPosDef;
            LowerTriangularMatrix cholesky = sum.CholeskyInPlace(out isPosDef);
            if (!isPosDef) throw new PositiveDefiniteMatrixException();
            // sum = cholesky*cholesky'
            // inv(sum) = inv(cholesky')*inv(cholesky)
            sumCorner.PredivideBy(cholesky);
            sumCorner.PredivideBy(cholesky.Transpose());
            nullspace.SetToDifference(nullspace, (intersection.Transpose() * sumCorner).Transpose());

            intersection.PredivideBy(cholesky);
            this.Variance.SetToOuterTranspose(intersection);

            Vector meanDiff = g2.Mean - g1.Mean;
            Vector temp2 = this.Mean;
            Vector temp = Vector.Zero(length);
            temp2.SetToProduct(eigenvectors2.Transpose(), meanDiff);
            temp.SetToProduct(eigenvectors2, temp2);
            temp2.SetToProduct(this.Variance, temp);
            this.Mean.SetToSum(g1.Mean, temp2);

            temp.SetToProduct(projection2, meanDiff);
            meanDiff.SetToDifference(meanDiff, temp);
            temp.SetToProduct(nullspace, meanDiff);
            temp.PredivideBy(cholesky);
            temp.PredivideBy(cholesky.Transpose());
            meanDiff.SetToProduct(nullspace.Transpose(), temp);
            this.Mean.SetToSum(this.Mean, meanDiff);
            if (this.Mean.Any(double.IsNaN)) throw new Exception("result is NaN");
        }

        /// <summary>
        /// Creates a new VectorGaussian which the product of two other VectorGaussians
        /// </summary>
        /// <param name="a">First VectorGaussian</param>
        /// <param name="b">Second VectorGaussian</param>
        /// <returns>Result</returns>
        public static VectorGaussianMoments operator *(VectorGaussianMoments a, VectorGaussianMoments b)
        {
            VectorGaussianMoments result = new VectorGaussianMoments(a.Dimension);
            result.SetToProduct(a, b);
            return result;
        }

        /// <summary>
        /// Sets the parameters to represent the product of two VectorGaussians.
        /// </summary>
        /// <param name="numerator">The numerator VectorGaussian.  Can be the same object as <c>this</c>.</param>
        /// <param name="denominator">The denominator VectorGaussian</param>
        /// <param name="forceProper">If true, the precision of the result is made non-negative definite, under the constraint that result*denominator has the same mean as numerator</param>
        public void SetToRatio(VectorGaussianMoments numerator, VectorGaussianMoments denominator, bool forceProper = false)
        {
            VectorGaussian vg1 = new VectorGaussian(numerator.Mean, numerator.Variance);
            VectorGaussian vg2 = new VectorGaussian(denominator.Mean, denominator.Variance);
            VectorGaussian ratio = new VectorGaussian(this.Dimension);
            ratio.SetToRatio(vg1, vg2, forceProper);
            ratio.GetMeanAndVariance(Mean, Variance);
        }

        /// <summary>
        /// Creates a new VectorGaussian which the ratio of two other VectorGaussians
        /// </summary>
        /// <param name="numerator">numerator VectorGaussian</param>
        /// <param name="denominator">denominator VectorGaussian</param>
        /// <returns>Result</returns>
        public static VectorGaussianMoments operator /(VectorGaussianMoments numerator, VectorGaussianMoments denominator)
        {
            VectorGaussianMoments result = new VectorGaussianMoments(numerator.Dimension);
            result.SetToRatio(numerator, denominator);
            return result;
        }

        /// <summary>
        /// Sets the parameters to represent the power of a source VectorGaussian to some exponent.
        /// </summary>
        /// <param name="dist">The source VectorGaussian</param>
        /// <param name="exponent">The exponent</param>
        public void SetToPower(VectorGaussianMoments dist, double exponent)
        {
            Mean = dist.Mean;
            variance.SetToProduct(dist.variance, 1 / exponent);
            for (int i = 0; i < Dimension; i++)
            {
                if (dist.variance[i, i] == 0)
                {
                    if (exponent == 0)
                    {
                        variance[i, i] = double.PositiveInfinity;
                    }
                    else if (exponent < 0)
                    {
                        throw new DivideByZeroException("The exponent is negative and the distribution is a point mass");
                    }
                }
            }
        }

        /// <summary>
        /// Raises a distribution to a power.
        /// </summary>
        /// <param name="dist">The distribution.</param>
        /// <param name="exponent">The power to raise to.</param>
        /// <returns><paramref name="dist"/> raised to power <paramref name="exponent"/>.</returns>
        public static VectorGaussianMoments operator ^(VectorGaussianMoments dist, double exponent)
        {
            VectorGaussianMoments result = new VectorGaussianMoments(dist.Dimension);
            result.SetToPower(dist, exponent);
            return result;
        }

        /// <summary>
        /// Sets the mean and covariance to match a VectorGaussian mixture.
        /// </summary>
        /// <param name="weight1">First weight</param>
        /// <param name="dist1">First VectorGaussian</param>
        /// <param name="weight2">Second weight</param>
        /// <param name="dist2">Second VectorGaussian</param>
        public void SetToSum(double weight1, VectorGaussianMoments dist1, double weight2, VectorGaussianMoments dist2)
        {
            WeightedSum<VectorGaussianMoments>(this, Dimension, weight1, dist1, weight2, dist2);
        }

        /// <summary>
        /// Creates a distribution of the specified type which matchs the mean and variance/covariance
        /// of a VectorGaussian mixture. The distribution type must implement <see cref="CanGetMeanAndVariance&lt;Vector, PositiveDefiniteMatrix&gt;"/> and
        /// <see cref="CanSetMeanAndVariance&lt;Vector, PositiveDefiniteMatrix&gt;"/>
        /// </summary>
        /// <typeparam name="T">Distribution type for the mixture</typeparam>
        /// <param name="dimension">The dimension of the domain</param>
        /// <param name="weight1">The first weight</param>
        /// <param name="dist1">The first distribution</param>
        /// <param name="weight2">The second weight</param>
        /// <param name="dist2">The second distribution</param>
        /// <param name="result">Resulting distribution</param>
        public static T WeightedSum<T>(T result, int dimension, double weight1, T dist1, double weight2, T dist2)
            where T : CanGetMeanAndVariance<Vector, PositiveDefiniteMatrix>, CanSetMeanAndVariance<Vector, PositiveDefiniteMatrix>, SettableTo<T>, SettableToUniform
        {
            if (weight1 + weight2 == 0) result.SetToUniform();
            else if (weight1 + weight2 < 0)
                throw new ArgumentException("weight1 (" + weight1 + ") + weight2 (" + weight2 + ") < 0");
            else if (weight1 == 0) result.SetTo(dist2);
            else if (weight2 == 0) result.SetTo(dist1);
            // if dist1 == dist2 then we must return dist1, with no roundoff error
            else if (dist1.Equals(dist2)) result.SetTo(dist1);
            else if (double.IsPositiveInfinity(weight1))
            {
                if (double.IsPositiveInfinity(weight2))
                {
                    throw new ArgumentException("both weights are infinity");
                }
                else
                {
                    result.SetTo(dist1);
                }
            }
            else if (double.IsPositiveInfinity(weight2)) result.SetTo(dist2);
            else
            {
                // w = weight1/(weight1 + weight2)
                // m = w*m1 + (1-w)*m2
                // v+m^2 = w*(v1+m1^2) + (1-w)*(v2+m2^2)
                // v = w*v1 + (1-w)*v2 + w*(m1-m)^2 + (1-w)*(m2-m)^2
                Vector m1 = Vector.Zero(dimension);
                Vector m2 = Vector.Zero(dimension);
                Vector m = Vector.Zero(dimension);
                PositiveDefiniteMatrix v1 = new PositiveDefiniteMatrix(dimension, dimension);
                PositiveDefiniteMatrix v2 = new PositiveDefiniteMatrix(dimension, dimension);
                PositiveDefiniteMatrix v = new PositiveDefiniteMatrix(dimension, dimension);
                dist1.GetMeanAndVariance(m1, v1);
                dist2.GetMeanAndVariance(m2, v2);
                double invZ = 1.0 / (weight1 + weight2);
                if (m1.Equals(m2))
                {
                    // catch this to avoid roundoff errors
                    if (v1.Equals(v2)) result.SetTo(dist1);
                    else
                    {
                        v.SetToSum(weight1, v1, weight2, v2);
                        v.Scale(invZ);
                        result.SetMeanAndVariance(m1, v);
                    }
                }
                else
                {
                    m.SetToSum(weight1, m1, weight2, m2);
                    m.Scale(invZ);
                    m1.SetToDifference(m1, m);
                    m2.SetToDifference(m2, m);
                    v.SetToSum(weight1, v1, weight2, v2);
                    v.SetToSumWithOuter(v, weight1, m1, m1);
                    v.SetToSumWithOuter(v, weight2, m2, m2);
                    v.Scale(invZ);
                    result.SetMeanAndVariance(m, v);
                }
            }
            return result;
        }

        /// <summary>
        /// The maximum difference between the parameters of this VectorGaussian
        /// and that VectorGaussian
        /// </summary>
        /// <param name="thatd">That VectorGaussian</param>
        /// <returns>The maximum difference</returns>
        public double MaxDiff(object thatd)
        {
            VectorGaussianMoments that = thatd as VectorGaussianMoments;
            if (that == null) return Double.PositiveInfinity;
            double max = mean.MaxDiff(that.mean);
            double diff = variance.MaxDiff(that.variance);
            if (diff > max) max = diff;
            return max;
        }

        /// <summary>
        /// Override of the Equals method
        /// </summary>
        /// <param name="thatd">The instance to compare to</param>
        /// <returns>True if the two distributions are the same in value, false otherwise</returns>
        /// <exclude/>
        public override bool Equals(object thatd)
        {
            VectorGaussian that = thatd as VectorGaussian;
            if (that == null) return false;
            return (MaxDiff(that) == 0.0);
        }

        /// <summary>
        /// Override of GetHashCode method
        /// </summary>
        /// <returns>The hash code for this instance</returns>
        /// <exclude/>
        public override int GetHashCode()
        {
            return Hash.Combine(mean.GetHashCode(), variance.GetHashCode());
        }

        /// <summary>
        /// Clones this VectorGaussian. 
        /// </summary>
        /// <returns>An object which is a clone of the current instance. This must be cast
        /// if you want to assign the result to a VectorGaussian type</returns>
        public object Clone()
        {
            VectorGaussianMoments result = new VectorGaussianMoments(Dimension);
            result.SetTo(this);
            return result;
        }

        #region Constructors

        /// <summary>
        /// Constructs a new VectorGaussian
        /// </summary>
        protected VectorGaussianMoments()
        {
        }

        /// <summary>
        /// Creates a uniform VectorGaussian of a given dimension
        /// </summary>
        /// <param name="dimension">The dimension</param>
        public VectorGaussianMoments(int dimension)
        {
            // this creates temporary storage which is immediately released
            mean = DenseVector.Zero(dimension);
            variance = new PositiveDefiniteMatrix(dimension, dimension);
        }

        /// <summary>
        /// Sets this VectorGaussian to the value of another
        /// </summary>
        /// <param name="that"></param>
        public VectorGaussianMoments(VectorGaussianMoments that)
            : this(that.Dimension)
        {
            SetTo(that);
        }

        /// <summary>
        /// Creates a copy of a given VectorGaussian
        /// </summary>
        /// <param name="that"></param>
        /// <returns></returns>
        public static VectorGaussianMoments Copy(VectorGaussianMoments that)
        {
            return new VectorGaussianMoments(that);
        }

        /// <summary>
        /// Creates a uniform VectorGaussian of a given dimension
        /// </summary>
        /// <param name="dimension">The dimension</param>
        [Construction("Dimension", UseWhen = "IsUniform"), Skip]
        public static VectorGaussianMoments Uniform(int dimension)
        {
            return new VectorGaussianMoments(dimension);
        }

        /// <summary>
        /// Creates a Gaussian with given mean and variance.
        /// </summary>
        /// <param name="mean">Mean</param>
        /// <param name="variance">Variance</param>
        /// <returns>A new Gaussian instance.</returns>
        /// <remarks>The mean and variance objects are copied into the Gaussian and not referenced afterwards.</remarks>
        public VectorGaussianMoments(Vector mean, PositiveDefiniteMatrix variance)
            : this(mean.Count)
        {
            Assert.IsTrue(variance.Rows == mean.Count, "The mean does not have the same dimension as the variance.");
            SetMeanAndVariance(mean, variance);
        }

        /// <summary>
        /// Creates a VectorGaussian point mass at the specified location
        /// </summary>
        /// <param name="mean">Where to position the point mass</param>
        /// <returns>A new point mass VectorGaussian</returns>
        [Construction("Point", UseWhen = "IsPointMass")]
        public static VectorGaussianMoments PointMass(Vector mean)
        {
            VectorGaussianMoments g = new VectorGaussianMoments(mean.Count);
            g.Point = mean;
            return g;
        }

        /// <summary>
        /// Creates a VectorGaussian point mass where the location is a
        /// vector of identical values
        /// </summary>
        /// <param name="mean">The value for the mean vector</param>
        /// <returns>A new point mass VectorGaussian</returns>
        public static VectorGaussianMoments PointMass(double mean)
        {
            return PointMass(Vector.FromArray(new double[] { mean }));
        }

        /// <summary>
        /// Creates a 1D VectorGaussian with given mean and variance.
        /// </summary>
        /// <param name="mean">Mean</param>
        /// <param name="variance">Variance</param>
        /// <returns>A new VectorGaussian object.</returns>
        public VectorGaussianMoments(double mean, double variance)
            : this(Vector.FromArray(new double[] { mean }),
                   new PositiveDefiniteMatrix(new double[,] { { variance } }))
        {
        }

        /// <summary>
        /// Creates a 1D VectorGaussian with given mean and variance.
        /// </summary>
        /// <param name="mean">Mean</param>
        /// <param name="variance">Variance</param>
        /// <returns>A new VectorGaussian object.</returns>
        public static VectorGaussianMoments FromMeanAndVariance(double mean, double variance)
        {
            VectorGaussianMoments result = new VectorGaussianMoments(1);
            result.SetMeanAndVariance(
                Vector.FromArray(new double[] { mean }),
                new PositiveDefiniteMatrix(new double[,] { { variance } }));
            return result;
        }

        /// <summary>
        /// Create a Gaussian with given mean and variance.
        /// </summary>
        /// <param name="mean">Mean</param>
        /// <param name="variance">Variance</param>
        /// <returns>A new Gaussian instance.</returns>
        /// <remarks>The mean and variance objects are copied into the Gaussian and not referenced afterwards.</remarks>
        public static VectorGaussianMoments FromMeanAndVariance(Vector mean, PositiveDefiniteMatrix variance)
        {
            Assert.IsTrue(variance.Rows == mean.Count, "The mean does not have the same dimension as the variance.");
            VectorGaussianMoments result = new VectorGaussianMoments(mean.Count);
            result.SetMeanAndVariance(mean, variance);
            return result;
        }

        /// <summary>
        /// Creates a 1D VectorGaussian with given mean and precision.
        /// </summary>
        /// <param name="mean">Mean</param>
        /// <param name="precision">Precision</param>
        /// <returns>A new VectorGaussian object.</returns>
        public static VectorGaussianMoments FromMeanAndPrecision(double mean, double precision)
        {
            VectorGaussianMoments result = new VectorGaussianMoments(1);
            result.SetMeanAndPrecision(
                Vector.FromArray(new double[] { mean }),
                new PositiveDefiniteMatrix(new double[,] { { precision } }));
            return result;
        }

        /// <summary>
        /// Create a VectorGaussian with given mean and precision matrix.
        /// </summary>
        /// <param name="mean">Mean</param>
        /// <param name="precision">Precision</param>
        /// <returns>A new VectorGaussian object.</returns>
        public static VectorGaussianMoments FromMeanAndPrecision(Vector mean, PositiveDefiniteMatrix precision)
        {
            VectorGaussianMoments result = new VectorGaussianMoments(mean.Count);
            result.SetMeanAndPrecision(mean, precision);
            return result;
        }

        #endregion

        /// <summary>
        /// Asks whether this VectorGaussian instance is proper or not. A VectorGaussian distribution
        /// is proper only if its precision matrix is positive definite.
        /// </summary>
        /// <returns>True if proper, false otherwise</returns>
        public bool IsProper()
        {
            return Variance.IsPositiveDefinite();
        }

        /// <summary>
        /// ToString override
        /// </summary>
        /// <returns>String representation of the instance</returns>
        /// <exclude/>
        public override string ToString()
        {
            if (IsPointMass)
            {
                return StringUtil.JoinColumns("VectorGaussianMoments.PointMass(", Point, ")");
            }
            else if (IsUniform())
            {
                return "VectorGaussianMoments.Uniform(" + Dimension + ")";
            }
            else
            {
                return StringUtil.JoinColumns("VectorGaussianMoments(", mean, ", ", variance, ")");
            }
        }

        /// <summary>
        /// The marginal distribution of one dimension.
        /// </summary>
        /// <param name="dim"></param>
        /// <returns></returns>
        public Gaussian GetMarginal(int dim)
        {
            return new Gaussian(mean[dim], variance[dim, dim]);
        }

        /// <summary>
        /// The marginal distribution of a subvector.
        /// </summary>
        /// <param name="firstDim">The first dimension of the subvector</param>
        /// <param name="result">A VectorGaussian receiving the result, whose Dimension specifies the length of the subvector.</param>
        /// <returns><paramref name="result"/></returns>
        public VectorGaussianMoments GetMarginal(int firstDim, VectorGaussianMoments result)
        {
            result.Mean.SetToSubvector(mean, firstDim);
            result.Variance.SetToSubmatrix(variance, firstDim, firstDim);
            return result;
        }
    }
}
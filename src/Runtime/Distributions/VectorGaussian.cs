// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Linq;
    using System.Runtime.Serialization;
    using Math;
    using Utilities;
    using Factors.Attributes;
    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// Represents a multivariate Gaussian distribution.
    /// </summary>
    /// <remarks><para>
    /// The distribution is parameterized by MeanTimesPrecision and Precision.
    /// Precision is the inverse of the variance, so a Gaussian with mean m and variance v is
    /// represented as Precision = inv(v), MeanTimesPrecision = inv(v)*m.
    /// </para><para>
    /// Some special cases:
    /// If the precision is zero, then the distribution is uniform.
    /// If the precision is infinite along the diagonal, then the distribution is a point mass.  The Point property
    /// gives the location of the point mass.
    /// If precision[i,i] is infinite, then the distribution is a point mass along dimension i.  Point[i] gives the mean.
    /// The rest of the row and column of precision must be zero.
    /// </para><para>
    /// The formula for the distribution is:
    /// <c>N(x;m,v) = |2*pi*v|^(-d/2) * exp(-0.5 (x-m)' inv(v) (x-m))</c>.
    /// When v=0, this reduces to delta(x-m).
    /// When v=infinity, the density is redefined to be 1.
    /// When inv(v) is singular, the density is redefined to be <c>exp(-0.5 x' inv(v) x + x' inv(v) m)</c>, 
    /// i.e. we drop the terms <c>|2*pi*v|^(-d/2) * exp(-0.5 m' inv(v) m)</c>.
    /// </para></remarks>
    [Serializable]
    [DataContract]
    [Quality(QualityBand.Mature)]
    public class VectorGaussian : IDistribution<Vector>,
                                  SettableTo<VectorGaussian>, SettableToProduct<VectorGaussian>, Diffable, SettableToUniform,
                                  SettableToRatio<VectorGaussian>, SettableToPower<VectorGaussian>, SettableToWeightedSum<VectorGaussian>,
                                  Sampleable<Vector>, CanSamplePrep<VectorGaussian, Vector>, CanGetLogProbPrep<VectorGaussian, Vector>,
                                  CanGetMean<DenseVector>, CanGetVariance<PositiveDefiniteMatrix>, CanGetMeanAndVariance<Vector, PositiveDefiniteMatrix>,
                                  CanSetMeanAndVariance<Vector, PositiveDefiniteMatrix>,
                                  CanGetLogAverageOf<VectorGaussian>, CanGetLogAverageOfPower<VectorGaussian>,
                                  CanGetAverageLog<VectorGaussian>, CanGetLogNormalizer, CanGetMode<DenseVector>
    {
        [DataMember]
        private DenseVector meanTimesPrecision;
        // precision may not always be positive definite, but we give it that type anyway.
        [DataMember]
        private PositiveDefiniteMatrix precision;

        /// <summary>
        /// Gets/Sets Mean times precision
        /// </summary>
        public Vector MeanTimesPrecision
        {
            get { return meanTimesPrecision; }
            set
            {
                if (meanTimesPrecision == null)
                    meanTimesPrecision = (DenseVector)value;
                else meanTimesPrecision.SetTo(value);
            }
        }

        /// <summary>
        /// Gets/sets precision value
        /// </summary>
        public PositiveDefiniteMatrix Precision
        {
            get { return precision; }
            set
            {
                if (precision == null) precision = value;
                else precision.SetTo(value);
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
            if (IsPointMass)
            {
                variance.SetAllElementsTo(0.0);
                mean.SetTo(Point);
            }
            else if (IsUniform())
            {
                variance.SetToIdentityScaledBy(Double.PositiveInfinity);
                mean.SetAllElementsTo(0.0);
            }
            else
            {
                bool[] wasZero = null;
                bool anyDiagonalIsZero = false;
                for (int i = 0; i < Dimension; i++)
                {
                    if (precision[i, i] == 0) anyDiagonalIsZero = true;
                }
                if (anyDiagonalIsZero)
                {
                    wasZero = new bool[Dimension];
                    for (int i = 0; i < Dimension; i++)
                    {
                        if (precision[i, i] == 0)
                        {
                            wasZero[i] = true;
                            precision[i, i] = double.Epsilon;
                        }
                    }
                }
                variance.SetToInverse(precision);
                mean.SetToProduct(variance, meanTimesPrecision);
                for (int i = 0; i < Dimension; i++)
                {
                    if (Double.IsPositiveInfinity(precision[i, i])) mean[i] = Point[i];
                }
                if (anyDiagonalIsZero)
                {
                    for (int i = 0; i < Dimension; i++)
                    {
                        if (wasZero[i])
                        {
                            mean[i] = 0;
                            precision[i, i] = 0;
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Gets the mean vector and precision matrix of the distribution
        /// </summary>
        /// <param name="mean">Modified to contain the mean.  Must already be the correct size.</param>
        /// <param name="precision">Modified to contain the precision matrix.  Must already be the correct size.</param>
        public void GetMeanAndPrecision(Vector mean, PositiveDefiniteMatrix precision)
        {
            precision.SetTo(this.precision);
            GetMean(mean);
        }

        /// <summary>
        /// Sets the mean and variance of the distribution.
        /// </summary>
        /// <param name="mean">The mean vector.  Cannot be the same object as <c>this.MeanTimesPrecision</c>.</param>
        /// <param name="variance">The covariance matrix.  Can be the same object as <c>this.Precision</c>.</param>
        public void SetMeanAndVariance(Vector mean, PositiveDefiniteMatrix variance)
        {
            if (variance.EqualsAll(0.0))
            {
                Point = mean;
            }
            else if (variance.EnumerateDiagonal().All(Double.IsPositiveInfinity))
            {
                SetToUniform();
            }
            else
            {
                // check for zeros on the diagonal
                bool hasZeros = false;
                for (int i = 0; i < Dimension; i++)
                {
                    if (variance[i, i] == 0)
                    {
                        hasZeros = true;
                        bool rowColZero = true;
                        for (int j = 0; j < Dimension; j++)
                        {
                            if (variance[i, j] != 0 || variance[j, i] != 0)
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
                    PositiveDefiniteMatrix variance2 = (PositiveDefiniteMatrix)variance.Clone();
                    for (int i = 0; i < Dimension; i++)
                    {
                        if (variance[i, i] == 0) variance2[i, i] = 1;
                    }
                    precision.SetToInverse(variance2);
                    meanTimesPrecision.SetToProduct(precision, mean);
                    for (int i = 0; i < Dimension; i++)
                    {
                        if (variance[i, i] == 0) precision[i, i] = Double.PositiveInfinity;
                    }
                }
                else
                {
                    precision.SetToInverse(variance);
                    meanTimesPrecision.SetToProduct(precision, mean);
                }
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
                this.precision.SetTo(precision);
                meanTimesPrecision.SetToProduct(precision, mean);
                for (int i = 0; i < Dimension; i++)
                {
                    if (Double.IsPositiveInfinity(precision[i, i]))
                        Point[i] = mean[i];
                }
            }
        }

        /// <summary>
        /// Sets the natural parameters of the distribution (mean times precision, and precision)
        /// </summary>
        /// <param name="meanTimesPrecision">Mean times precision</param>
        /// <param name="precision">The precision matrix</param>
        public void SetNatural(Vector meanTimesPrecision, PositiveDefiniteMatrix precision)
        {
            this.MeanTimesPrecision = meanTimesPrecision;
            this.Precision = precision;
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
            else if (IsUniform()) result.SetAllElementsTo(0.0);
            else
            {
                // TM: Don't need to check for PosDef because Inverse already does.
                //if(!precision.IsPositiveDefinite()) throw new ArgumentException("Improper distribution", "this");
                result.SetTo(meanTimesPrecision);
                result.PredivideBy(this.Precision);
                for (int i = 0; i < Dimension; i++)
                {
                    if (Double.IsPositiveInfinity(precision[i, i])) result[i] = meanTimesPrecision[i];
                }
            }
            return result;
        }

        /// <summary>
        /// Get the mean of the distribution
        /// </summary>
        /// <param name="result">Where to place the mean vector</param>
        /// <param name="variance">The pre-computed inverse of this.Precision</param>
        /// <returns></returns>
        public Vector GetMean(Vector result, PositiveDefiniteMatrix variance)
        {
            if (IsPointMass) result.SetTo(Point);
            else if (IsUniform()) result.SetAllElementsTo(0.0);
            else
            {
                result.SetToProduct(variance, meanTimesPrecision);
                for (int i = 0; i < Dimension; i++)
                {
                    if (Double.IsPositiveInfinity(precision[i, i])) result[i] = meanTimesPrecision[i];
                }
            }
            return result;
        }

        /// <summary>
        /// Gets the variance-covariance matrix of the distribution.
        /// </summary>
        /// <returns>A new PositiveDefiniteMatrix.</returns>
        public PositiveDefiniteMatrix GetVariance()
        {
            return GetVariance(new PositiveDefiniteMatrix(Dimension, Dimension));
        }

        /// <summary>
        /// Gets the variance-covariance matrix of the distribution.
        /// </summary>
        /// <param name="variance">Where to place the variance-covariance</param>
        /// <returns>variance</returns>
        public PositiveDefiniteMatrix GetVariance(PositiveDefiniteMatrix variance)
        {
            if (IsPointMass) variance.SetAllElementsTo(0.0);
            else if (IsUniform()) variance.SetToIdentityScaledBy(Double.PositiveInfinity);
            // TM: Don't need to check for PosDef because Inverse already does.
            //else if(!precision.IsPositiveDefinite()) throw new ArgumentException("Improper distribution", "this");
            else variance.SetToInverse(precision);
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
                //return Enumerable.TrueForAll(precision.EnumerateDiagonal(), Double.IsPositiveInfinity);
                for (int i = 0; i < Dimension; i++)
                {
                    if (!Double.IsPositiveInfinity(precision[i, i])) return false;
                }
                return true;
            }
        }

        /// <summary>
        /// Sets this instance to a point mass. The location of the
        /// point mass is the existing mean vector
        /// </summary>
        protected void SetToPointMass()
        {
            precision.SetAllElementsTo(0.0);
            int dim = Dimension;
            for (int i = 0; i < dim; i++)
            {
                precision[i, i] = Double.PositiveInfinity;
            }
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
                return meanTimesPrecision;
            }
            set
            {
                SetToPointMass();
                meanTimesPrecision.SetTo(value);
            }
        }

        /// <summary>
        /// The dimension of the VectorGaussian domain
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public int Dimension
        {
            get { return meanTimesPrecision.Count; }
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
            precision.SetAllElementsTo(0);
            meanTimesPrecision.SetAllElementsTo(0);
        }

        /// <summary>
        /// Asks whether this instance is uniform
        /// </summary>
        /// <returns>True if uniform, false otherwise</returns>
        public bool IsUniform()
        {
            return precision.EqualsAll(0) && meanTimesPrecision.EqualsAll(0);
        }

        /// <summary>
        /// Evaluate the log of multivariate Gaussian density with specified mean vector
        /// and covariance matrix
        /// </summary>
        /// <param name="x">Where to evaluate the density function</param>
        /// <param name="mean">The mean vector</param>
        /// <param name="variance">A non-singular covariance matrix.</param>
        /// <returns></returns>
        public static double GetLogProb(Vector x, Vector mean, PositiveDefiniteMatrix variance)
        {
            if (variance.EqualsAll(0)) return (x == mean) ? 0 : double.NegativeInfinity;
            int d = x.Count;
            LowerTriangularMatrix varianceChol = new LowerTriangularMatrix(d, d);
            bool isPosDef = varianceChol.SetToCholesky(variance);
            Vector dx = x - mean;
            if (!isPosDef)
            {
                LowerTriangularMatrix A = varianceChol;
                Vector b = dx;
                for (int i = 0; i < A.Rows; i++)
                {
                    double sum = b[i];
                    for (int j = 0; j < i; j++)
                    {
                        sum -= A[i, j] * b[j];
                    }
                    if (sum == 0) b[i] = 0;
                    else if (A[i, i] == 0) return double.NegativeInfinity;
                    else b[i] = sum / A[i, i];
                }
                return -0.5 * dx.Inner(dx);
            }
            else
            {
                dx.PredivideBy(varianceChol);
                // new dx = inv(chol(v))*dx so that
                // (new dx)'*(new dx) = dx'*inv(chol(v))'*inv(chol(v))*dx 
                //                    = dx'*inv(chol(v)*chol(v)')*dx
                //                    = dx'*inv(v)*dx
                return -varianceChol.TraceLn() - d * MMath.LnSqrt2PI - 0.5 * dx.Inner(dx);
            }
        }

        /// <summary>
        /// Gets the normalizer for the VectorGaussian density function
        /// </summary>
        /// <returns></returns>
        public double GetLogNormalizer()
        {
            LowerTriangularMatrix precChol = new LowerTriangularMatrix(Dimension, Dimension);
            bool isPosDef = precChol.SetToCholesky(precision);
            if (!isPosDef) return 0.0;
            Vector meanTimesPrecChol = Vector.Zero(Dimension);
            meanTimesPrecChol.SetTo(meanTimesPrecision);
            meanTimesPrecChol.PredivideBy(precChol);
            // 0.5*LogDeterminant(precision) = 0.5*2*L.TraceLn()
            double result = 0.5 * meanTimesPrecChol.Inner(meanTimesPrecChol);
            for (int i = 0; i < Dimension; i++)
            {
                if (!Double.IsPositiveInfinity(precision[i, i]))
                    result += MMath.LnSqrt2PI - Math.Log(precChol[i, i]);
            }
            return result;
        }

        /// <summary>
        /// Evaluates the log of the multivariate Gaussian density.
        /// </summary>
        /// <param name="x">Point to evaluate the density at.</param>
        /// <param name="meanTimesPrecision">Precision matrix times the mean vector.</param>
        /// <param name="precision">A non-singular precision matrix (inverse of covariance matrix).</param>
        /// <param name="L">Same size as precision.</param>
        /// <param name="iLb">Same size as x.</param>
        /// <returns>log p(x)</returns>
        public static double GetLogProb(Vector x, Vector meanTimesPrecision, PositiveDefiniteMatrix precision,
                                        LowerTriangularMatrix L, Vector iLb)
        {
            // The distribution is
            //  p(x) = (2*pi)^(-d/2)*|precision|^(1/2)*exp(-0.5*(x-m)'precision(x-m))
            // (x-m)'precision(x-m) = x'Ax -2xm + m'Am
            // m = precision \ meanTimesPrecision = L' \ L \ meanTimesPrecision
            // m'Am = m'meanTimesPrecision = (L\meanTimesPrecision)'(L\meanTimesPrecision)
            bool isPosDef = L.SetToCholesky(precision);
            int d = x.Count;
            //double result = -0.5 * precision.QuadraticForm(x) + x.Inner(meanTimesPrecision);
            double result = 0.0;
            for (int i = 0; i < d; i++)
            {
                if (Double.IsPositiveInfinity(precision[i, i]))
                {
                    if (x[i] != meanTimesPrecision[i]) return Double.NegativeInfinity;
                }
                else
                {
                    double sum = 0.0;
                    for (int j = 0; j < d; j++)
                    {
                        sum += precision[i, j] * x[j];
                    }
                    result += x[i] * (meanTimesPrecision[i] - 0.5 * sum);
                }
            }
            if (isPosDef)
            {
                iLb.SetTo(meanTimesPrecision);
                iLb.PredivideBy(L);
                result += -0.5 * iLb.Inner(iLb);
                // 0.5*LogDeterminant(precision) = 0.5*2*L.TraceLn()
                //result += L.TraceLn() - d * MMath.LnSqrt2PI;
                for (int i = 0; i < d; i++)
                {
                    double Lii = L[i, i];
                    if (!Double.IsPositiveInfinity(Lii))
                    {
                        result += Math.Log(Lii) - MMath.LnSqrt2PI;
                    }
                }
            }
            return result;
        }

        /// <summary>
        /// Evaluates the log of the multivariate Gaussian density.
        /// </summary>
        /// <param name="x">Point to evaluate the density at.</param>
        /// <param name="L">Work matrix - same size as Precision</param>
        /// <param name="iLb">Work vector - same size as x</param>
        /// <returns>log p(x)</returns>
        public double GetLogProb(Vector x, LowerTriangularMatrix L, Vector iLb)
        {
            if (IsPointMass)
            {
                return (x == Point) ? 0.0 : Double.NegativeInfinity;
            }
            else if (IsUniform())
            {
                return 0;
            }
            else
            {
                return GetLogProb(x, meanTimesPrecision, precision, L, iLb);
            }
        }

        /// <summary>
        /// Evaluates the log of the multivariate Gaussian density.
        /// </summary>
        /// <param name="x">Point to evaluate the density at.</param>
        /// <returns>log p(x)</returns>
        public double GetLogProb(Vector x)
        {
            LowerTriangularMatrix precL = new LowerTriangularMatrix(Dimension, Dimension);
            Vector iLb = Vector.Zero(Dimension);
            return GetLogProb(x, precL, iLb);
        }

        /// <summary>
        /// Returns an Evaluator delegate which has a pre-allocated workspace
        /// for efficient evaluation calculation. If you are generating many
        /// evaluations, call this method to get an Evaluator, then use the Evaluator
        /// delegate to calculate the evaluations
        /// </summary>
        /// <returns>Evaluator delegate</returns>
        public Evaluator<VectorGaussian, Vector> GetLogProbPrep()
        {
            LowerTriangularMatrix precL = new LowerTriangularMatrix(Dimension, Dimension);
            Vector iLb = Vector.Zero(Dimension);
            return delegate (VectorGaussian dist, Vector x) { return dist.GetLogProb(x, precL, iLb); };
        }

        /// <summary>
        /// The expected logarithm of that distribution under this distribution.
        /// </summary>
        /// <param name="that">The distribution to take the logarithm of.</param>
        /// <returns><c>sum_x this.Evaluate(x)*Math.Log(that.Evaluate(x))</c></returns>
        /// <remarks>This is also known as the cross entropy.</remarks>
        public double GetAverageLog(VectorGaussian that)
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
                Vector m1 = Vector.Zero(Dimension);
                PositiveDefiniteMatrix v1 = new PositiveDefiniteMatrix(Dimension, Dimension);
                GetMeanAndVariance(m1, v1);
                return that.GetLogProb(m1) - 0.5 * Matrix.TraceOfProduct(v1, that.Precision);
            }
        }

        /// <summary>
        /// Log-integral of the product of this VectorGaussian with that VectorGaussian
        /// </summary>
        /// <param name="that">That VectorGaussian</param>
        /// <returns>The log inner product</returns>
        public double GetLogAverageOf(VectorGaussian that)
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
                VectorGaussian product = this * that;
                //if (!product.IsProper()) throw new ArgumentException("The product is improper.");
                double result = product.GetLogNormalizer() - this.GetLogNormalizer() - that.GetLogNormalizer();
                for (int i = 0; i < Dimension; i++)
                {
                    if (Double.IsPositiveInfinity(product.Precision[i, i]))
                    {
                        // -0.5 * precision.QuadraticForm(x) + x.Inner(meanTimesPrecision);
                        if (Double.IsPositiveInfinity(this.Precision[i, i]))
                        {
                            double x = this.Point[i];
                            result += x * that.meanTimesPrecision[i];
                            double sum = 0.0;
                            for (int j = 0; j < Dimension; j++)
                            {
                                if (Double.IsPositiveInfinity(this.Precision[j, j]))
                                    sum += that.precision[i, j] * this.Point[j];
                            }
                            result += -0.5 * sum * x;
                        }
                        else
                        {
                            double x = that.Point[i];
                            result += that.Point[i] * this.meanTimesPrecision[i];
                            double sum = 0.0;
                            for (int j = 0; j < Dimension; j++)
                            {
                                if (Double.IsPositiveInfinity(that.Precision[j, j]))
                                    sum += this.precision[i, j] * that.Point[j];
                            }
                            result += -0.5 * sum * x;
                        }
                    }
                }
                return result;
            }
        }

        /// <summary>
        /// Get the integral of this distribution times another distribution raised to a power.
        /// </summary>
        /// <param name="that"></param>
        /// <param name="power"></param>
        /// <returns></returns>
        public double GetLogAverageOfPower(VectorGaussian that, double power)
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
                for (int i = 0; i < Dimension; i++)
                {
                    if (Double.IsPositiveInfinity(product.Precision[i, i]))
                    {
                        throw new NotImplementedException();
                    }
                }
                return result;
            }
        }

        /// <summary>
        /// Samples from a VectorGaussian distribution with the specified mean and precision
        /// </summary>
        /// <returns>The sample value</returns>
        [Stochastic]
        public static Vector Sample(Vector mean, PositiveDefiniteMatrix precision)
        {
            Vector result = Vector.Zero(mean.Count);
            Rand.NormalP(mean, precision, result);
            return result;
        }

        /// <summary>
        /// Samples from a VectorGaussian distribution with the specified mean and variance
        /// </summary>
        /// <returns>The sample value</returns>
        [Stochastic]
        [ParameterNames("sample", "mean", "variance")]
        public static Vector SampleFromMeanAndVariance(Vector mean, PositiveDefiniteMatrix variance)
        {
            Vector result = Vector.Zero(mean.Count);
            Rand.NormalP(mean, variance.Inverse(), result);
            return result;
        }

        /// <summary>
        /// Samples from this VectorGaussian distribution
        /// </summary>
        /// <param name="result">Where to put the result</param>
        /// <param name="precL">A DxD workspace</param>
        /// <returns>The sample value</returns>
        [Stochastic]
        public Vector Sample(Vector result, LowerTriangularMatrix precL)
        {
            if (IsPointMass)
            {
                result.SetTo(Point);
                return result;
            }

            // algorithm is similar to Rand.NormalPChol()
            // meanTimesPrecision = precision*mean = L*L'*mean
            // thus mean = L' \ L \ meanTimesPrecision
            // x = L' \ (L \ meanTimesPrecision + RandNorm)

            result.SetTo(meanTimesPrecision);
            //Matrix precL = new Matrix(Dimension, Dimension);
            // precL is the Cholesky factor of the precision matrix
            precL.SetToCholesky(precision);
            result.PredivideBy(precL);
            for (int j = 0; j < Dimension; j++) result[j] += Rand.Normal();
            result.PredivideByTranspose(precL);
            for (int j = 0; j < Dimension; j++)
            {
                if (double.IsPositiveInfinity(precL[j, j])) result[j] = meanTimesPrecision[j];
            }
            return result;
        }

        /// <summary>
        /// Samples from this VectorGaussian distribution
        /// </summary>
        /// <param name="result">Where to put the result</param>
        /// <returns>The sample value</returns>
        [Stochastic]
        public Vector Sample(Vector result)
        {
            Assert.IsTrue(result.Count == Dimension);
            LowerTriangularMatrix workspace = new LowerTriangularMatrix(Dimension, Dimension);
            return Sample(result, workspace);
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
        /// Returns a sampler delegate which has a pre-allocated workspace
        /// for efficient sample calculation. If you are generating many
        /// samples, call this method to get a sampler, then use the sampler
        /// delegate to generate samples.
        /// </summary>
        /// <returns>Sampler delegate</returns>
        public Sampler<VectorGaussian, Vector> SamplePrep()
        {
            LowerTriangularMatrix workspace = new LowerTriangularMatrix(Dimension, Dimension);
            return delegate (VectorGaussian dist, Vector result) { return dist.Sample(result, workspace); };
        }

        /// <summary>
        /// Sets this VectorGaussian instance to have the parameter values of that VectorGaussian instance
        /// </summary>
        /// <param name="that">That VectorGaussian</param>
        public void SetTo(VectorGaussian that)
        {
            if (object.ReferenceEquals(that, null)) SetToUniform();
            else
            {
                meanTimesPrecision.SetTo(that.meanTimesPrecision);
                precision.SetTo(that.precision);
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
        public void SetToProduct(VectorGaussian g1, VectorGaussian g2)
        {
            for (int i = 0; i < Dimension; i++)
            {
                if (Double.IsPositiveInfinity(g1.precision[i, i]))
                {
                    if (Double.IsPositiveInfinity(g2.precision[i, i]) && (g1.Point[i] != g2.Point[i]))
                        throw new AllZeroException();
                    Point[i] = g1.Point[i];
                }
                else if (Double.IsPositiveInfinity(g2.precision[i, i]))
                {
                    Point[i] = g2.Point[i];
                }
                else
                {
                    meanTimesPrecision[i] = g1.meanTimesPrecision[i] + g2.meanTimesPrecision[i];
                }
            }
            precision.SetToSum(g1.precision, g2.precision);
            for (int i = 0; i < Dimension; i++)
            {
                if (Double.IsPositiveInfinity(precision[i, i]))
                {
                    // zero out the rest of the row and column
                    for (int j = 0; j < Dimension; j++)
                    {
                        if (j != i)
                        {
                            if (!Double.IsPositiveInfinity(precision[j, j]))
                                meanTimesPrecision[j] -= precision[j, i] * Point[i];
                            precision[i, j] = 0.0;
                            precision[j, i] = 0.0;
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Creates a new VectorGaussian which the product of two other VectorGaussians
        /// </summary>
        /// <param name="a">First VectorGaussian</param>
        /// <param name="b">Second VectorGaussian</param>
        /// <returns>Result</returns>
        public static VectorGaussian operator *(VectorGaussian a, VectorGaussian b)
        {
            VectorGaussian result = new VectorGaussian(a.Dimension);
            result.SetToProduct(a, b);
            return result;
        }

        /// <summary>
        /// Sets the parameters to represent the product of two VectorGaussians.
        /// </summary>
        /// <param name="numerator">The numerator VectorGaussian.  Can be the same object as <c>this</c>.</param>
        /// <param name="denominator">The denominator VectorGaussian</param>
        /// <param name="forceProper">If true, the precision of the result is made non-negative definite, under the constraint that result*denominator has the same mean as numerator</param>
        public void SetToRatio(VectorGaussian numerator, VectorGaussian denominator, bool forceProper = false)
        {
            bool isImproper = false;
            Vector tau = null;
            if (forceProper)
            {
                for (int i = 0; i < Dimension; i++)
                {
                    if (numerator.precision[i, i] < denominator.precision[i, i])
                    {
                        isImproper = true;
                        break;
                    }
                }
                if (isImproper)
                {
                    //throw new NotImplementedException();
                    tau = denominator.precision * numerator.GetMean();
                    Precision.SetAllElementsTo(0);
                    // cannot access numerator.precision after this
                }
            }
            if (tau == null) precision.SetToDifference(numerator.precision, denominator.precision);
            for (int i = 0; i < Dimension; i++)
            {
                if (Double.IsPositiveInfinity(precision[i, i]))
                {
                    Point[i] = numerator.Point[i];
                }
                else if (Double.IsNaN(precision[i, i]))
                {
                    // NaN = Inf - Inf so both numerator and denominator are point masses
                    if (numerator.Point[i] != denominator.Point[i]) throw new DivideByZeroException();
                    meanTimesPrecision[i] = 0.0;
                    precision[i, i] = 1e-100; // dummy value that allows inversion
                }
                else if (Double.IsNegativeInfinity(precision[i, i]))
                {
                    throw new DivideByZeroException();
                }
                else if (isImproper)
                {
                    meanTimesPrecision[i] = tau[i] - denominator.meanTimesPrecision[i];
                }
                else
                {
                    meanTimesPrecision[i] = numerator.meanTimesPrecision[i] - denominator.meanTimesPrecision[i];
                }
            }
            for (int i = 0; i < Dimension; i++)
            {
                if (Double.IsPositiveInfinity(precision[i, i]))
                {
                    // zero out the rest of the row and column
                    for (int j = 0; j < Dimension; j++)
                    {
                        if (j != i)
                        {
                            if (!Double.IsPositiveInfinity(precision[j, j]))
                                meanTimesPrecision[j] -= precision[j, i] * Point[i];
                            precision[i, j] = 0.0;
                            precision[j, i] = 0.0;
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Creates a new VectorGaussian which the ratio of two other VectorGaussians
        /// </summary>
        /// <param name="numerator">numerator VectorGaussian</param>
        /// <param name="denominator">denominator VectorGaussian</param>
        /// <returns>Result</returns>
        public static VectorGaussian operator /(VectorGaussian numerator, VectorGaussian denominator)
        {
            VectorGaussian result = new VectorGaussian(numerator.Dimension);
            result.SetToRatio(numerator, denominator);
            return result;
        }

        /// <summary>
        /// Sets the parameters to represent the power of a source VectorGaussian to some exponent.
        /// </summary>
        /// <param name="dist">The source VectorGaussian</param>
        /// <param name="exponent">The exponent</param>
        public void SetToPower(VectorGaussian dist, double exponent)
        {
            for (int i = 0; i < Dimension; i++)
            {
                if (Double.IsPositiveInfinity(dist.precision[i, i]))
                {
                    if (exponent == 0)
                    {
                        precision[i, i] = 0.0;
                        meanTimesPrecision[i] = 0.0;
                    }
                    else if (exponent < 0)
                    {
                        throw new DivideByZeroException("The exponent is negative and the distribution is a point mass");
                    }
                    else
                    {
                        meanTimesPrecision[i] = dist.meanTimesPrecision[i];
                    }
                }
                else
                {
                    meanTimesPrecision[i] = dist.meanTimesPrecision[i] * exponent;
                }
            }
            precision.SetToProduct(dist.precision, exponent);
        }

        /// <summary>
        /// Raises a distribution to a power.
        /// </summary>
        /// <param name="dist">The distribution.</param>
        /// <param name="exponent">The power to raise to.</param>
        /// <returns><paramref name="dist"/> raised to power <paramref name="exponent"/>.</returns>
        public static VectorGaussian operator ^(VectorGaussian dist, double exponent)
        {
            VectorGaussian result = new VectorGaussian(dist.Dimension);
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
        public void SetToSum(double weight1, VectorGaussian dist1, double weight2, VectorGaussian dist2)
        {
            WeightedSum<VectorGaussian>(this, Dimension, weight1, dist1, weight2, dist2);
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
            VectorGaussian that = thatd as VectorGaussian;
            if (that == null) return Double.PositiveInfinity;
            double max = meanTimesPrecision.MaxDiff(that.meanTimesPrecision);
            double diff = precision.MaxDiff(that.precision);
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
            return Hash.Combine(meanTimesPrecision.GetHashCode(), precision.GetHashCode());
        }

        /// <summary>
        /// Clones this VectorGaussian. 
        /// </summary>
        /// <returns>An object which is a clone of the current instance. This must be cast
        /// if you want to assign the result to a VectorGaussian type</returns>
        public object Clone()
        {
            VectorGaussian result = new VectorGaussian(Dimension);
            result.SetTo(this);
            return result;
        }

        #region Constructors

        /// <summary>
        /// Constructs a new VectorGaussian
        /// </summary>
        protected VectorGaussian()
        {
        }

        /// <summary>
        /// Creates a uniform VectorGaussian of a given dimension
        /// </summary>
        /// <param name="dimension">The dimension</param>
        public VectorGaussian(int dimension)
        {
            // this creates temporary storage which is immediately released
            meanTimesPrecision = DenseVector.Zero(dimension);
            precision = new PositiveDefiniteMatrix(dimension, dimension);
        }

        /// <summary>
        /// Sets this VectorGaussian to the value of another
        /// </summary>
        /// <param name="that"></param>
        public VectorGaussian(VectorGaussian that)
            : this(that.Dimension)
        {
            SetTo(that);
        }

        /// <summary>
        /// Creates a copy of a given VectorGaussian
        /// </summary>
        /// <param name="that"></param>
        /// <returns></returns>
        public static VectorGaussian Copy(VectorGaussian that)
        {
            return new VectorGaussian(that);
        }

        /// <summary>
        /// Creates a uniform VectorGaussian of a given dimension
        /// </summary>
        /// <param name="dimension">The dimension</param>
        [Construction("Dimension", UseWhen = "IsUniform"), Skip]
        public static VectorGaussian Uniform(int dimension)
        {
            return new VectorGaussian(dimension);
        }

        /// <summary>
        /// Creates a Gaussian from Cursor objects.
        /// </summary>
        /// <param name="meanTimesPrecision">Mean times precision</param>
        /// <param name="precision">Precision</param>
        /// <remarks>The cursors will use their existing source array.
        /// The Gaussian will reference the given cursors.</remarks>
        public static VectorGaussian FromCursors(Vector meanTimesPrecision, PositiveDefiniteMatrix precision)
        {
            VectorGaussian g = new VectorGaussian();
            g.meanTimesPrecision = (DenseVector)meanTimesPrecision;
            g.precision = precision;
            //g.CreateSourceArray(1);
            return g;
        }

        /// <summary>
        /// Creates a Gaussian with given mean and variance.
        /// </summary>
        /// <param name="mean">Mean</param>
        /// <param name="variance">Variance</param>
        /// <returns>A new Gaussian instance.</returns>
        /// <remarks>The mean and variance objects are copied into the Gaussian and not referenced afterwards.</remarks>
        public VectorGaussian(Vector mean, PositiveDefiniteMatrix variance)
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
        public static VectorGaussian PointMass(Vector mean)
        {
            VectorGaussian g = new VectorGaussian(mean.Count);
            g.Point = mean;
            return g;
        }

        /// <summary>
        /// Creates a VectorGaussian point mass where the location is a
        /// vector of identical values
        /// </summary>
        /// <param name="mean">The value for the mean vector</param>
        /// <returns>A new point mass VectorGaussian</returns>
        public static VectorGaussian PointMass(double mean)
        {
            return PointMass(Vector.FromArray(new double[] { mean }));
        }

        /// <summary>
        /// Creates a 1D VectorGaussian with given mean and variance.
        /// </summary>
        /// <param name="mean">Mean</param>
        /// <param name="variance">Variance</param>
        /// <returns>A new VectorGaussian object.</returns>
        public VectorGaussian(double mean, double variance)
            : this(Vector.FromArray(new double[] { mean }),
                   new PositiveDefiniteMatrix(new double[,] { { variance } }))
        {
        }

        /// <summary>
        /// Create a new VectorGaussian from its natural parameters (Mean times precision, and precision)
        /// </summary>
        /// <param name="meanTimesPrecision">Mean times precision</param>
        /// <param name="precision">Precision matrix</param>
        /// <returns>A new VectorGaussian object.</returns>
        [Construction("MeanTimesPrecision", "Precision")]
        public static VectorGaussian FromNatural(Vector meanTimesPrecision, PositiveDefiniteMatrix precision)
        {
            VectorGaussian result = new VectorGaussian(meanTimesPrecision.Count);
            result.meanTimesPrecision.SetTo(meanTimesPrecision);
            result.precision.SetTo(precision);
            return result;
        }

        /// <summary>
        /// Construct a Gaussian distribution whose pdf has the given derivatives at a point.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="dlogp">The gradient</param>
        /// <param name="negativeHessian">The negative Hessian matrix</param>
        /// <param name="forceProper">If true and both derivatives cannot be matched, match only the first.</param>
        /// <returns></returns>
        public static VectorGaussian FromDerivatives(Vector x, Vector dlogp, PositiveDefiniteMatrix negativeHessian, bool forceProper)
        {
            // logp = -0.5*r*(x-m)^2 + const.
            // dlogp = -r*(x-m)
            // ddlogp = -r  (but we take negative to get r)
            if (forceProper)
            {
                LowerTriangularMatrix L = new LowerTriangularMatrix(negativeHessian.Rows, negativeHessian.Cols);
                bool isPosDef = L.SetToCholesky(negativeHessian);
                if (!isPosDef)
                {
                    negativeHessian = new PositiveDefiniteMatrix(L.Rows, L.Cols);
                    negativeHessian.SetToOuter(L);
                }
            }
            return VectorGaussian.FromNatural(dlogp + negativeHessian * x, negativeHessian);
        }

        /// <summary>
        /// Creates a 1D VectorGaussian with given mean and variance.
        /// </summary>
        /// <param name="mean">Mean</param>
        /// <param name="variance">Variance</param>
        /// <returns>A new VectorGaussian object.</returns>
        public static VectorGaussian FromMeanAndVariance(double mean, double variance)
        {
            VectorGaussian result = new VectorGaussian(1);
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
        public static VectorGaussian FromMeanAndVariance(Vector mean, PositiveDefiniteMatrix variance)
        {
            Assert.IsTrue(variance.Rows == mean.Count, "The mean does not have the same dimension as the variance.");
            VectorGaussian result = new VectorGaussian(mean.Count);
            result.SetMeanAndVariance(mean, variance);
            return result;
        }

        /// <summary>
        /// Creates a 1D VectorGaussian with given mean and precision.
        /// </summary>
        /// <param name="mean">Mean</param>
        /// <param name="precision">Precision</param>
        /// <returns>A new VectorGaussian object.</returns>
        public static VectorGaussian FromMeanAndPrecision(double mean, double precision)
        {
            VectorGaussian result = new VectorGaussian(1);
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
        public static VectorGaussian FromMeanAndPrecision(Vector mean, PositiveDefiniteMatrix precision)
        {
            VectorGaussian result = new VectorGaussian(mean.Count);
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
            return Precision.IsPositiveDefinite();
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
                return StringUtil.JoinColumns("VectorGaussian.PointMass(", Point, ")");
            }
            else if (IsUniform())
            {
                return "VectorGaussian.Uniform(" + Dimension + ")";
            }
            else if (Precision.SymmetryError() == 0 && IsProper())
            {
                Vector mean = Vector.Zero(Dimension);
                PositiveDefiniteMatrix variance = new PositiveDefiniteMatrix(Dimension, Dimension);
                GetMeanAndVariance(mean, variance);
                return StringUtil.JoinColumns("VectorGaussian(", mean, ", ", variance, ")");
            }
            else
            {
                return StringUtil.JoinColumns("VectorGaussian(pm=", meanTimesPrecision, ", prec=", precision, ")");
            }
        }

        /// <summary>
        /// The marginal distribution of one dimension.
        /// </summary>
        /// <param name="dim"></param>
        /// <returns></returns>
        public Gaussian GetMarginal(int dim)
        {
            if (Dimension == 1) return Gaussian.FromNatural(meanTimesPrecision[0], precision[0, 0]);
            if (Precision[dim, dim] == 0)
                return Gaussian.Uniform();
            Vector mean = Vector.Zero(Dimension);
            PositiveDefiniteMatrix variance = new PositiveDefiniteMatrix(Dimension, Dimension);
            GetMeanAndVariance(mean, variance);
            return new Gaussian(mean[dim], variance[dim, dim]);
        }

        /// <summary>
        /// The marginal distribution of a subvector.
        /// </summary>
        /// <param name="firstDim">The first dimension of the subvector</param>
        /// <param name="result">A VectorGaussian receiving the result, whose Dimension specifies the length of the subvector.</param>
        /// <returns><paramref name="result"/></returns>
        public VectorGaussian GetMarginal(int firstDim, VectorGaussian result)
        {
            Vector mean = Vector.Zero(Dimension);
            PositiveDefiniteMatrix variance = new PositiveDefiniteMatrix(Dimension, Dimension);
            GetMeanAndVariance(mean, variance);
            Vector resultMean = Vector.Zero(result.Dimension);
            PositiveDefiniteMatrix resultVariance = new PositiveDefiniteMatrix(result.Dimension, result.Dimension);
            resultMean.SetToSubvector(mean, firstDim);
            resultVariance.SetToSubmatrix(variance, firstDim, firstDim);
            result.SetMeanAndVariance(resultMean, resultVariance);
            return result;
        }
    }
}
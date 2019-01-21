// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Runtime.Serialization;

    using Math;
    using Utilities;
    using Factors.Attributes;
    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// A Wishart distribution on positive definite matrices.
    /// </summary>
    /// <remarks><para>
    ///  In the matrix case, the distribution is
    ///   <c>p(X) = |X|^(a-(d+1)/2)*exp(-tr(X*B))*|B|^a/Gamma_d(a)</c>.
    /// In this code, the <c>a</c> parameter is called the "Shape" and the <c>B</c> parameter
    /// is called the "Rate".  The distribution is uniform when B=0 and a=(d+1)/2.
    /// The mean of the distribution is <c>a/B</c> and the diagonal variance is 
    /// <c>var(X_ii) = a*C_ii^2</c> where <c>C=inv(B)</c>.  The non-diagonal variances are
    /// <c>var(X_ij) = a*0.5*(C_ij^2 + C_ii*C_jj)</c> where <c>C=inv(B)</c>.
    /// </para><para>
    /// The distribution is represented by a double for <c>a</c> and 
    /// a PositiveDefiniteMatrix for <c>B</c>.  
    /// </para></remarks>
    [Serializable]
    [DataContract]
    [Quality(QualityBand.Mature)]
    public class Wishart : IDistribution<PositiveDefiniteMatrix>,
                           SettableTo<Wishart>, SettableToProduct<Wishart>, Diffable, SettableToUniform,
                           Sampleable<PositiveDefiniteMatrix>, SettableToRatio<Wishart>, SettableToPower<Wishart>,
                           CanGetMean<PositiveDefiniteMatrix>, CanGetVariance<PositiveDefiniteMatrix>, CanGetMeanAndVariance<PositiveDefiniteMatrix, PositiveDefiniteMatrix>,
                           CanSetMeanAndVariance<PositiveDefiniteMatrix, PositiveDefiniteMatrix>, CanGetLogAverageOf<Wishart>, CanGetLogAverageOfPower<Wishart>,
                           SettableToWeightedSum<Wishart>, CanGetAverageLog<Wishart>, CanGetLogNormalizer, CanGetMode<PositiveDefiniteMatrix>
    {
        // rate may not always be positive definite, but we give it that type anyway.
        [DataMember]
        private PositiveDefiniteMatrix rate;

        [DataMember]
        private double shape;

        /// <summary>
        /// Sets/gets the rate matrix
        /// </summary>
        public PositiveDefiniteMatrix Rate
        {
            get
            {
                return rate;
            }
            set
            {
                if (rate == null)
                    rate = value;
                else
                    rate.SetTo(value);
            }
        }

        /// <summary>
        /// Sets/gets the shape value
        /// </summary>
        public double Shape
        {
            get
            {
                return shape;
            }
            set
            {
                shape = value;
            }
        }

        public PositiveDefiniteMatrix GetMode()
        {
            return GetMode(new PositiveDefiniteMatrix(Dimension, Dimension));
        }

        public PositiveDefiniteMatrix GetMode(PositiveDefiniteMatrix result)
        {
            if (IsPointMass)
            {
                result.SetTo(Point);
            }
            else if (Shape <= (Dimension + 1) * 0.5)
            {
                result.SetAllElementsTo(0.0);
            }
            else if (rate.EqualsAll(0))
            {
                result.SetAllElementsTo(Double.PositiveInfinity);
            }
            else
            {
                result.SetToInverse(rate).Scale(Shape - (Dimension + 1) * 0.5);
            }
            return result;
        }

        /// <summary>
        /// Gets the mean of the distribution.
        /// </summary>
        /// <returns>A new PositiveDefiniteMatrix.</returns>
        public PositiveDefiniteMatrix GetMean()
        {
            return GetMean(new PositiveDefiniteMatrix(Dimension, Dimension));
        }

        /// <summary>
        /// Gets the scale matrix
        /// </summary>
        /// <returns>A new PositiveDefiniteMatrix.</returns>
        public PositiveDefiniteMatrix GetScale()
        {
            return Rate.Inverse();
        }

        /// <summary>
        /// Gets the mean of the distribution.
        /// </summary>
        /// <param name="mean">Where to put the mean matrix</param>
        /// <returns>The mean matrix</returns>
        public PositiveDefiniteMatrix GetMean(PositiveDefiniteMatrix mean)
        {
            if (IsPointMass)
            {
                mean.SetTo(Point);
            }
            else if (IsUniform())
            {
                mean.SetAllElementsTo(Double.PositiveInfinity);
            }
            else
            {
                //   E[X] = a/B
                mean.SetToInverse(rate).Scale(Shape);
            }
            return mean;
        }

#if false
        public Vector GetDiagonalVariance()
        {
            // var(X_ii) = a*diag(inv(B))^2
            Vector result = new Vector(Dimension);
            if (!IsPointMass) {
                PositiveDefiniteMatrix iB = rate.Inverse();
                result.SetToDiagonal(iB);
                result.SetToProduct(result, result);
                result.Scale(Shape);
            }
            return result;
        }
#endif

        /// <summary>
        /// Gets the variance of the distribution
        /// </summary>
        /// <returns>The variance matrix</returns>
        public PositiveDefiniteMatrix GetVariance()
        {
            PositiveDefiniteMatrix variance = new PositiveDefiniteMatrix(Dimension, Dimension);
            return GetVariance(variance);
        }

        /// <summary>
        /// Gets the variance of the distribution
        /// </summary>
        /// <param name="variance">Where to put the variance</param>
        /// <returns>The variance matrix</returns>
        public PositiveDefiniteMatrix GetVariance(PositiveDefiniteMatrix variance)
        {
            if (IsPointMass)
            {
                variance.SetAllElementsTo(0.0);
            }
            else if (IsUniform())
            {
                variance.SetToIdentityScaledBy(Double.PositiveInfinity);
            }
            else
            {
                // var(X) = a/2*(c.*c + diag(c)*diag(c)')  where c = inv(B)
                variance.SetToInverse(rate);
                Vector diag = variance.Diagonal();
                variance.SetToElementwiseProduct(variance, variance);
                variance.SetToSumWithOuter(variance, 1, diag, diag);
                variance.Scale(Shape * 0.5);
            }
            return variance;
        }

        /// <summary>
        /// Gets the mean and variance matrices.
        /// </summary>
        /// <param name="mean">Where to put the mean - assumed to be of the correct size</param>
        /// <param name="variance">Where to put the variance - assumed to be of the correct size</param>
        public void GetMeanAndVariance(PositiveDefiniteMatrix mean, PositiveDefiniteMatrix variance)
        {
            if (IsPointMass)
            {
                mean.SetTo(Point);
                variance.SetAllElementsTo(0.0);
            }
            else if (IsUniform())
            {
                mean.SetAllElementsTo(Double.PositiveInfinity);
                variance.SetToIdentityScaledBy(Double.PositiveInfinity);
            }
            else
            {
                variance.SetToInverse(Rate);
                mean.SetToProduct(variance, Shape);
                Vector diag = variance.Diagonal();
                variance.SetToElementwiseProduct(variance, variance);
                variance.SetToSumWithOuter(variance, 1, diag, diag);
                variance.Scale(Shape * 0.5);
            }
        }

        /// <summary>
        /// Sets the parameters to produce a given mean and variance.
        /// </summary>
        /// <param name="mean">Mean</param>
        /// <param name="variance">Variance</param>
        /// <remarks>
        /// The mean is always matched, but the variance may not match exactly, since the distribution
        /// has only one scalar parameter for variance.
        /// </remarks>
        public void SetMeanAndVariance(PositiveDefiniteMatrix mean, PositiveDefiniteMatrix variance)
        {
            // a = (sum_i mean[i,i]*mean[i,i])/(sum_i variance[i,i])
            // B = a*inv(mean)
            double denom = 0.0;
            foreach (double d in variance.EnumerateDiagonal())
            {
                denom += d;
            }
            if (denom == 0.0)
            {
                Point = mean;
            }
            else if (Double.IsPositiveInfinity(denom))
            {
                SetToUniform();
            }
            else
            {
                Vector mDiag = mean.Diagonal();
                mDiag.SetToProduct(mDiag, mDiag);
                Shape = mDiag.Sum() / denom;
                Rate.SetToInverse(mean).Scale(Shape);
            }
        }

        /// <summary>
        /// Sets the shape parameter and the scale matrix parameter for this instance
        /// </summary>
        /// <param name="shape">The shape parameter</param>
        /// <param name="scale">The scale matrix</param>
        public void SetShapeAndScale(double shape, PositiveDefiniteMatrix scale)
        {
            Shape = shape;
            Rate.SetToInverse(scale);
        }

        /// <summary>
        /// Sets the shape parameter and the rate matrix parameter for this instance
        /// </summary>
        /// <param name="shape">The shape parameter</param>
        /// <param name="rate">The rate matrix</param>
        public void SetShapeAndRate(double shape, PositiveDefiniteMatrix rate)
        {
            Shape = shape;
            Rate.SetTo(rate);
        }

        /// <summary>
        /// Gets the mean log determinant
        /// </summary>
        /// <returns>The mean log determinant</returns>
        public double GetMeanLogDeterminant()
        {
            if (IsPointMass)
                return Point.LogDeterminant();
            // E[logdet(X)] = -logdet(B) + sum_{i=0..d-1} digamma(a)
            double s = 0;
            int d = Dimension;
            for (int i = 0; i < d; i++)
                s += MMath.Digamma(Shape - i * 0.5);
            s -= rate.LogDeterminant();
            return s;
        }

        /// <summary>
        /// Asks whether the instance is a point mass
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public bool IsPointMass
        {
            get
            {
                return (Shape == Double.PositiveInfinity);
            }
        }

        /// <summary>
        /// Sets this instance to a point mass. The location of the
        /// point mass is the existing rate matrix
        /// </summary>
        protected void SetToPointMass()
        {
            Shape = Double.PositiveInfinity;
        }

        /// <summary>
        /// Sets/gets this instance as a point-mass
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public PositiveDefiniteMatrix Point
        {
            get
            {
                // The accessor must succeed, even if the distribution is not a point mass.
                //Assert.IsTrue(IsPointMass, "The distribution is not a point mass");
                return rate;
            }
            set
            {
                SetToPointMass();
                rate.SetTo(value);
            }
        }

        /// <summary>
        /// Dimension of this distribution
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public int Dimension
        {
            get
            {
                return rate.Rows;
            }
        }

        /// <summary>
        /// Sets this instance to have uniform distribution
        /// </summary>
        public void SetToUniform()
        {
            shape = 0.5 * (Dimension + 1);
            rate.SetAllElementsTo(0);
        }

        /// <summary>
        /// Asks whether this instance is uniform
        /// </summary>
        /// <returns></returns>
        public bool IsUniform()
        {
            return (shape == 0.5 * (Dimension + 1)) && rate.EqualsAll(0);
        }

        /// <summary>
        /// Asks whether this instance is proper
        /// </summary>
        /// <returns>True if proper, false otherwise</returns>
        public bool IsProper()
        {
            return (Shape > 0.5 * (Dimension - 1)) && Rate.IsPositiveDefinite();
        }

        /// <summary>
        /// Asks whether a Wishart distribution of the specified shape and rate is proper
        /// </summary>
        /// <param name="shape">Shape parameter</param>
        /// <param name="rate">Rate matrix</param>
        /// <returns>True if proper, false otherwise</returns>
        public static bool IsProper(double shape, PositiveDefiniteMatrix rate)
        {
            int dimension = rate.Rows;
            return (shape > 0.5 * (dimension - 1)) && rate.IsPositiveDefinite();
        }

        /// <summary>
        /// Evaluates the logarithm of a Wishart density function at a given point
        /// </summary>
        /// <param name="x">Where to evaluate the density</param>
        /// <param name="shape">Shape parameter</param>
        /// <param name="rate">Rate matrix</param>
        /// <returns>The log density</returns>
        /// <remarks>
        /// The distribution is <c>p(X) = |X|^(a-(d+1)/2)*exp(-tr(X*B))*|B|^a/Gamma_d(a)</c>.
        /// When a &lt;= (d-1)/2 the <c>Gamma_d(a)</c> term is dropped.
        /// When B &lt;= 0 the <c>|B|^a</c> term is dropped.
        /// Thus if shape = (d+1)/2 and rate = 0 the density is 1.
        /// </remarks>
        public static double GetLogProb(PositiveDefiniteMatrix x, double shape, PositiveDefiniteMatrix rate)
        {
            int dimension = x.Rows;
            double result = (shape - 0.5 * (dimension + 1)) * x.LogDeterminant();
            result -= Matrix.TraceOfProduct(x, rate);
            result -= GetLogNormalizer(shape, rate);
            return result;
        }

        /// <summary>
        /// Evaluates the logarithm of this Wishart density function at a given point
        /// </summary>
        /// <param name="X">Where to evaluate the density</param>
        /// <returns>The log density</returns>
        public double GetLogProb(PositiveDefiniteMatrix X)
        {
            if (IsPointMass)
            {
                return (X == Point) ? 0.0 : Double.NegativeInfinity;
            }
            else if (IsUniform())
            {
                return 0.0;
            }
            else
            {
                return GetLogProb(X, Shape, rate);
            }
        }

        /// <summary>
        /// Gets the normalizer for a Wishart density function specified by shape and rate matrix
        /// </summary>
        /// <param name="shape">Shape parameter</param>
        /// <param name="rate">rate matrix</param>
        /// <returns></returns>
        public static double GetLogNormalizer(double shape, PositiveDefiniteMatrix rate)
        {
            int dimension = rate.Rows;
            // inline call to IsProper()
            if (!(shape > 0.5 * (dimension - 1)))
                return 0.0;
            LowerTriangularMatrix rateChol = new LowerTriangularMatrix(dimension, dimension);
            bool isPosDef = rateChol.SetToCholesky(rate);
            if (!isPosDef)
                return 0.0;
            // LogDeterminant(rate) = 2*L.TraceLn()
            return MMath.GammaLn(shape, dimension) - shape * 2 * rateChol.TraceLn();
        }

        /// <summary>
        /// Gets the normalizer for the density function of this Wishart distribution
        /// </summary>
        /// <returns></returns>
        public double GetLogNormalizer()
        {
            return GetLogNormalizer(Shape, rate);
        }

        /// <summary>
        /// The expected logarithm of that distribution under this distribution.
        /// </summary>
        /// <param name="that">The distribution to take the logarithm of.</param>
        /// <returns><c>sum_x this.Evaluate(x)*Math.Log(that.Evaluate(x))</c></returns>
        /// <remarks>This is also known as the cross entropy.</remarks>
        public double GetAverageLog(Wishart that)
        {
            if (that.IsPointMass)
            {
                if (this.IsPointMass && (this.Point == that.Point))
                    return 0.0;
                else
                    return Double.NegativeInfinity;
            }
            else
            {
                // that is not a point mass.
                double result = (that.Shape - 0.5 * (that.Dimension + 1)) * this.GetMeanLogDeterminant();
                result -= Matrix.TraceOfProduct(this.GetMean(), that.Rate);
                result -= that.GetLogNormalizer();
                return result;
            }
        }

#if false
        public static double ProbEqualLn(IEnumerable<Wishart> dists)
        {
            // are any of them point masses?
            PositiveDefiniteMatrix point = null;
            foreach(Wishart dist in dists) {
                if(dist.IsPointMass) {
                    point = dist.Point;
                    break;
                }
            }
            if(point != null) {
                // one is a point mass
                double sum = 0.0;
                foreach(Wishart dist in dists) {
                    sum += dist.EvaluateLn(point);
                }
                return sum;
            } else {
                double sum = 0.0;
                int dimension=0;
                foreach(Wishart dist in dists) {
                    dimension = dist.Dimension;
                    sum += dist.Normalizer();
                }
                Wishart prod = new Wishart(dimension);
                Util.SetToProduct(prod, dists);
                sum -= prod.Normalizer();
                return sum;
            }
        }
        public double Normalizer()
        {
            double ad = Precision+0.5*(Dimension+1);
            return ad*PrecisionTimesMean.LogDeterminant() - MMath.GammaLn(ad,Dimension);
        }
#endif

        /// <summary>
        /// Gets the log-integral of the product of this Wishart with another Wishart
        /// </summary>
        /// <param name="that">The other Wishart</param>
        /// <returns>The log inner product</returns>
        public double GetLogAverageOf(Wishart that)
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
                Wishart product = this * that;
                //if (!product.IsProper()) throw new ArgumentException("The product is improper.");
                return product.GetLogNormalizer() - this.GetLogNormalizer() - that.GetLogNormalizer();
            }
        }

        /// <summary>
        /// Get the integral of this distribution times another distribution raised to a power.
        /// </summary>
        /// <param name="that"></param>
        /// <param name="power"></param>
        /// <returns></returns>
        public double GetLogAverageOfPower(Wishart that, double power)
        {
            if (IsPointMass)
            {
                return power * that.GetLogProb(Point);
            }
            else if (that.IsPointMass)
            {
                if (power < 0)
                    throw new DivideByZeroException("The exponent is negative and the distribution is a point mass");
                return this.GetLogProb(that.Point);
            }
            else
            {
                var product = this * (that ^ power);
                return product.GetLogNormalizer() - this.GetLogNormalizer() - power * that.GetLogNormalizer();
            }
        }

        /// <summary>
        /// Samples this Wishart distribution
        /// </summary>
        /// <param name="result">Where to put the sample</param>
        /// <param name="cholB">A workspace matrix of the same dimension as the distribution</param>
        /// <param name="cholX">A workspace matrix of the same dimension as the distribution</param>
        /// <param name="cholXt">A workspace matrix of the same dimension as the distribution</param>
        /// <returns>The sample</returns>
        [Stochastic]
        public PositiveDefiniteMatrix Sample(PositiveDefiniteMatrix result,
                                             LowerTriangularMatrix cholB, LowerTriangularMatrix cholX, Matrix cholXt)
        {
            // TODO: unify this with SampleFromShapeAndRate
            if (IsPointMass)
            {
                result.SetTo(Point);
                return result;
            }
            // Algorithm:
            //   X = chol(inv(B))*W*chol(inv(B))'  where W is std Wishart
            //   chol(inv(B)) = inv(chol(B)')
            cholB.SetToCholesky(rate);
            Rand.Wishart(Shape, cholX);
            cholX.PredivideByTranspose(cholB);
            // cholX is no longer LowerTriangular
            cholXt.SetToTranspose(cholX);
            result.SetToProduct(cholX, cholXt);
            return result;
        }

        /// <summary>
        /// Samples this Wishart distribution. Workspaces are allocated behind the scenes
        /// </summary>
        /// <param name="result">Where to put the sample</param>
        /// <returns>The sample</returns>
        [Stochastic]
        public PositiveDefiniteMatrix Sample(PositiveDefiniteMatrix result)
        {
            Assert.IsTrue(result.Rows == Dimension);
            LowerTriangularMatrix cholB = new LowerTriangularMatrix(Dimension, Dimension);
            LowerTriangularMatrix cholX = new LowerTriangularMatrix(Dimension, Dimension);
            Matrix cholXt = new Matrix(Dimension, Dimension);
            return Sample(result, cholB, cholX, cholXt);
        }

        /// <summary>
        /// Samples this Wishart distribution. Workspaces and sample matrix are allocated
        /// behind the scenes
        /// </summary>
        /// <returns>The sample</returns>
        [Stochastic]
        public PositiveDefiniteMatrix Sample()
        {
            return Sample(new PositiveDefiniteMatrix(Dimension, Dimension));
        }

        /// <summary>
        /// Samples a Wishart distribution of specified shape and rate.
        /// Workspaces are allocated behind the scenes
        /// </summary>
        /// <param name="shape">The shape parameter</param>
        /// <param name="scale">The scale matrix</param>
        /// <param name="result">Where to put the sample</param>
        /// <returns>The sample</returns>
        [Stochastic]
        public static PositiveDefiniteMatrix Sample(double shape, PositiveDefiniteMatrix scale, PositiveDefiniteMatrix result)
        {
            int dimension = scale.Rows;
            LowerTriangularMatrix cholB = new LowerTriangularMatrix(dimension, dimension);
            cholB.SetToCholesky(scale);
            LowerTriangularMatrix cholX = new LowerTriangularMatrix(dimension, dimension);
            Rand.Wishart(shape, cholX);
            // result = cholB*cholX*cholX'*cholB'
            cholX.SetToProduct(cholB, cholX);
            UpperTriangularMatrix cholXt = cholX.Transpose();
            result.SetToProduct(cholX, cholXt);
            return result;
        }

        /// <summary>
        /// Samples a Wishart distribution of specified shape and scale.
        /// Workspaces are allocated behind the scenes
        /// </summary>
        /// <param name="shape">The shape parameter</param>
        /// <param name="scale">The scale matrix</param>
        /// <returns>The sample</returns>
        [Stochastic]
        [ParameterNames("sample", "shape", "scale")]
        public static PositiveDefiniteMatrix SampleFromShapeAndScale(double shape, PositiveDefiniteMatrix scale)
        {
            var result = PositiveDefiniteMatrix.Identity(scale.Rows);
            return Sample(shape, scale, result);
        }

        /// <summary>
        /// Samples a Wishart distribution of specified shape and rate.
        /// Workspaces are allocated behind the scenes
        /// </summary>
        /// <param name="shape">The shape parameter</param>
        /// <param name="rate">The rate matrix</param>
        /// <returns>The sample</returns>
        [Stochastic]
        [ParameterNames("sample", "shape", "rate")]
        public static PositiveDefiniteMatrix SampleFromShapeAndRate(double shape, PositiveDefiniteMatrix rate)
        {
            var result = PositiveDefiniteMatrix.Identity(rate.Rows);
            return SampleFromShapeAndRate(shape, rate, result);
        }

        /// <summary>
        /// Samples a Wishart distribution of specified shape and rate.
        /// Workspaces are allocated behind the scenes
        /// </summary>
        /// <param name="shape">The shape parameter</param>
        /// <param name="rate">The rate matrix</param>
        /// <param name="result">Receives the sample</param>
        /// <returns><paramref name="result"/></returns>
        [Stochastic]
        public static PositiveDefiniteMatrix SampleFromShapeAndRate(double shape, PositiveDefiniteMatrix rate, PositiveDefiniteMatrix result)
        {
            int dimension = rate.Rows;
            LowerTriangularMatrix cholB = new LowerTriangularMatrix(dimension, dimension);
            cholB.SetToCholesky(rate);
            LowerTriangularMatrix cholX = new LowerTriangularMatrix(dimension, dimension);
            Rand.Wishart(shape, cholX);
            // result = inv(cholB')*cholX*cholX'*inv(cholB)
            cholX.PredivideByTranspose(cholB);
            UpperTriangularMatrix cholXt = cholX.Transpose();
            result.SetToProduct(cholX, cholXt);
            return result;
        }

        /// <summary>
        /// Sets this Wishart instance to have the parameter values of another Wishart instance
        /// </summary>
        /// <param name="that">The other Wishart</param>
        public void SetTo(Wishart that)
        {
            if (object.ReferenceEquals(that, null))
                SetToUniform();
            else
            {
                rate.SetTo(that.rate);
                shape = that.shape;
            }
        }

        /// <summary>
        /// Sets the parameters to represent the product of two Wisharts.
        /// </summary>
        /// <param name="g1">The first Wishart. May refer to <c>this</c>.</param>
        /// <param name="g2">The second Wishart. May refer to <c>this</c>.</param>
        /// <remarks>
        /// The result may not be proper. No error is thrown in this case.
        /// </remarks>
        public void SetToProduct(Wishart g1, Wishart g2)
        {
            if (g1.IsPointMass)
            {
                Point = g1.Point;
                return;
            }
            if (g2.IsPointMass)
            {
                Point = g2.Point;
                return;
            }
            // avoid roundoff errors when a shape is below eps
            if (g1.Shape < g2.Shape)
                shape = g1.shape + (g2.shape - 0.5 * (Dimension + 1));
            else
                shape = g2.shape + (g1.shape - 0.5 * (Dimension + 1));
            rate.SetToSum(g1.rate, g2.rate);
        }

        /// <summary>
        /// Creates a new Wishart which the product of two other Wisharts
        /// </summary>
        /// <param name="a">First Wishart</param>
        /// <param name="b">Second Wishart</param>
        /// <returns>Result</returns>
        public static Wishart operator *(Wishart a, Wishart b)
        {
            Wishart result = new Wishart(a.Dimension);
            result.SetToProduct(a, b);
            return result;
        }

        /// <summary>
        /// Sets the parameters to represent the ratio of two Wisharts.
        /// </summary>
        /// <param name="numerator">The numerator Wishart.  Can be <c>this</c></param>
        /// <param name="denominator">The denominator Wishart</param>
        /// <param name="forceProper">If true, the result shape >= (dimension+1)/2 and rate is non-negative definite</param>
        public void SetToRatio(Wishart numerator, Wishart denominator, bool forceProper = false)
        {
            if (numerator.IsPointMass)
            {
                if (denominator.IsPointMass)
                {
                    if (numerator.Point.Equals(denominator.Point))
                    {
                        SetToUniform();
                    }
                    else
                    {
                        throw new DivideByZeroException();
                    }
                }
                else
                {
                    Point = numerator.Point;
                }
            }
            else if (denominator.IsPointMass)
            {
                throw new DivideByZeroException();
            }
            else
            {
                int dim = Dimension;
                double c = 0.5 * (dim + 1);
                if (forceProper)
                {
                    // constraints are: shape >= (d+1)/2, rate >= 0, ((shape-(d+1)/2)+denominator.shape)/(rate + denominator.rate) = numerator.shape/numerator.rate
                    // let a = shape-(d+1)/2
                    // (a+denominator.shape)/numerator.shape I = (rate + denominator.rate)/numerator.rate
                    // (a+denominator.shape)/numerator.shape I >= denominator.rate/numerator.rate
                    // rate = (a+denominator.shape)/numerator.shape*numerator.rate - denominator.rate
                    // so we compute the largest eigenvalue of denominator.rate/numerator.rate
                    LowerTriangularMatrix nChol = new LowerTriangularMatrix(dim, dim);
                    nChol.SetToCholesky(numerator.rate);
                    LowerTriangularMatrix dChol = new LowerTriangularMatrix(dim, dim);
                    dChol.SetToCholesky(denominator.rate);
                    Matrix indChol = dChol.PredivideBy(nChol);
                    Matrix ind = new Matrix(dim, dim);
                    ind.SetToOuter(indChol);
                    // ind is now denominator.rate/numerator.rate
                    Matrix eigenvectors = indChol;
                    indChol = null;
                    eigenvectors.SetToEigenvectorsOfSymmetric(ind);
                    Matrix eigenvalues = ind;
                    ind = null;
                    double max = 1;
                    for (int i = 0; i < Dimension; i++)
                    {
                        double eig = eigenvalues[i, i];
                        if (eig > max)
                            max = eig;
                    }
                    // a = (numerator.shape * max - denominator.shape)
                    // now apply the constraint (a >= 0)
                    if (numerator.shape * max < denominator.shape)
                    {
                        rate.SetToSum(denominator.shape / numerator.shape, numerator.rate, -1, denominator.rate);
                        shape = c;
                    }
                    else
                    {
                        shape = numerator.shape * max + (c - denominator.shape);
                        rate.SetToSum(max, numerator.rate, -1, denominator.rate);
                    }
                }
                else
                {
                    shape = numerator.shape + (c - denominator.shape);
                    rate.SetToDifference(numerator.rate, denominator.rate);
                }
            }
        }

        /// <summary>
        /// Creates a new Wishart which is the ratio of two other Wishart
        /// </summary>
        /// <param name="numerator">numerator Wishart</param>
        /// <param name="denominator">denominator Wishart</param>
        /// <returns>Result</returns>
        public static Wishart operator /(Wishart numerator, Wishart denominator)
        {
            Wishart result = new Wishart(numerator.Dimension);
            result.SetToRatio(numerator, denominator);
            return result;
        }

        /// <summary>
        /// Sets the parameters to represent the power of a source Wishart to some exponent.
        /// </summary>
        /// <param name="dist">The source Wishart</param>
        /// <param name="exponent">The exponent</param>
        public void SetToPower(Wishart dist, double exponent)
        {
            if (dist.IsPointMass)
            {
                if (exponent == 0)
                {
                    SetToUniform();
                }
                else if (exponent < 0)
                {
                    throw new DivideByZeroException("The exponent is negative and the distribution is a point mass");
                }
                else
                {
                    Point = dist.Point;
                }
                return;
            }
            else
            {
                double d = 0.5 * (Dimension + 1);
                shape = dist.shape * exponent + d * (1 - exponent);
                rate.SetToProduct(dist.rate, exponent);
            }
        }

        /// <summary>
        /// Raises a distribution to a power.
        /// </summary>
        /// <param name="dist">The distribution.</param>
        /// <param name="exponent">The power to raise to.</param>
        /// <returns><paramref name="dist"/> raised to power <paramref name="exponent"/>.</returns>
        public static Wishart operator ^(Wishart dist, double exponent)
        {
            Wishart result = new Wishart(dist.Dimension);
            result.SetToPower(dist, exponent);
            return result;
        }

        /// <summary>
        /// Weighted mixture distribution for two Wisharts
        /// </summary>
        /// <param name="weight1">First weight</param>
        /// <param name="dist1">First Wishart</param>
        /// <param name="weight2">Second weight</param>
        /// <param name="dist2">Second Wishart</param>
        public void SetToSum(double weight1, Wishart dist1, double weight2, Wishart dist2)
        {
            SetTo(WeightedSum<Wishart>(this, Dimension, weight1, dist1, weight2, dist2));
        }

        /// <summary>
        /// Creates a weighted mixture distribution for distributions whose mean and variance are both
        /// of type PositiveDefiniteMatrix. The distribution type must implement <see cref="CanGetMeanAndVariance&lt;PositiveDefiniteMatrix, PositiveDefiniteMatrix&gt;"/> and
        /// <see cref="CanSetMeanAndVariance&lt;PositiveDefiniteMatrix, PositiveDefiniteMatrix&gt;"/>
        /// </summary>
        /// <typeparam name="T">Distribution type for the mixture</typeparam>
        /// <param name="dimension">The dimension of the domain</param>
        /// <param name="weight1">The first weight</param>
        /// <param name="dist1">The first distribution</param>
        /// <param name="weight2">The second weight</param>
        /// <param name="dist2">The second distribution</param>
        /// <param name="result">Resulting distribution</param>
        public static T WeightedSum<T>(T result, int dimension, double weight1, T dist1, double weight2, T dist2)
            where T : CanGetMeanAndVariance<PositiveDefiniteMatrix, PositiveDefiniteMatrix>, CanSetMeanAndVariance<PositiveDefiniteMatrix, PositiveDefiniteMatrix>,
                SettableToUniform, SettableTo<T>
        {
            if (weight1 + weight2 == 0)
                result.SetToUniform();
            else if (weight1 + weight2 < 0)
                throw new ArgumentException("weight1 (" + weight1 + ") + weight2 (" + weight2 + ") < 0");
            else if (weight1 == 0)
                result.SetTo(dist2);
            else if (weight2 == 0)
                result.SetTo(dist1);
            // if dist1 == dist2 then we must return dist1, with no roundoff error
            else if (dist1.Equals(dist2))
                result.SetTo(dist1);
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
            else if (double.IsPositiveInfinity(weight2))
                result.SetTo(dist2);
            else
            {
                // w = weight1/(weight1 + weight2)
                // m = w*m1 + (1-w)*m2
                // v+m^2 = w*(v1+m1^2) + (1-w)*(v2+m2^2)
                // v = w*v1 + (1-w)*v2 + w*(m1-m)^2 + (1-w)*(m2-m)^2
                PositiveDefiniteMatrix m1 = new PositiveDefiniteMatrix(dimension, dimension);
                PositiveDefiniteMatrix v1 = new PositiveDefiniteMatrix(dimension, dimension);
                PositiveDefiniteMatrix m2 = new PositiveDefiniteMatrix(dimension, dimension);
                PositiveDefiniteMatrix v2 = new PositiveDefiniteMatrix(dimension, dimension);
                PositiveDefiniteMatrix m = new PositiveDefiniteMatrix(dimension, dimension);
                PositiveDefiniteMatrix v = new PositiveDefiniteMatrix(dimension, dimension);
                dist1.GetMeanAndVariance(m1, v1);
                dist2.GetMeanAndVariance(m2, v2);
                double invZ = 1.0 / (weight1 + weight2);
                if (m1.Equals(m2))
                {
                    // catch this to avoid roundoff errors
                    if (v1.Equals(v2))
                        result.SetTo(dist1);
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
                    m1.SetToElementwiseProduct(m1, m1);
                    v.SetToSum(1, v, weight1, m1);
                    m2.SetToElementwiseProduct(m2, m2);
                    v.SetToSum(1, v, weight2, m2);
                    v.Scale(invZ);
                    result.SetMeanAndVariance(m, v);
                }
            }
            return result;
        }

        /// <summary>
        /// The maximum difference between the parameters of this Wishart
        /// and that Wishart
        /// </summary>
        /// <param name="thatd">That Wishart</param>
        /// <returns>The maximum difference</returns>
        public double MaxDiff(object thatd)
        {
            Wishart that = thatd as Wishart;
            if (that == null)
                return Double.PositiveInfinity;
            double max = rate.MaxDiff(that.rate);
            double diff = MMath.AbsDiff(shape, that.shape);
            if (diff > max)
                max = diff;
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
            Wishart that = thatd as Wishart;
            if (that == null)
                return false;
            return (MaxDiff(that) == 0.0);
        }

        /// <summary>
        /// Override of GetHashCode method
        /// </summary>
        /// <returns>The hash code for this instance</returns>
        /// <exclude/>
        public override int GetHashCode()
        {
            return Hash.Combine(rate.GetHashCode(), shape.GetHashCode());
        }

        /// <summary>
        /// Clones this Wishart. 
        /// </summary>
        /// <returns>An object which is a clone of the current instance. This must be cast
        /// if you want to assign the result to a Wishart type</returns>
        public object Clone()
        {
            Wishart result = new Wishart(Dimension);
            result.SetTo(this);
            return result;
        }

        #region Constructors

        /// <summary>
        /// Parameterless constructor needed for serialization
        /// </summary>
        protected Wishart()
        {
        }

        /// <summary>
        /// Constructs a uniform Wishart distribution of the given dimension
        /// </summary>
        /// <param name="dimension">The dimension</param>
        public Wishart(int dimension)
        {
            shape = 0.5 * (dimension + 1);
            rate = new PositiveDefiniteMatrix(dimension, dimension);
        }

        /// <summary>
        /// Constructs a uniform Wishart distribution of the given dimension
        /// </summary>
        /// <param name="dimension">The dimension</param>
        [Construction("Dimension", UseWhen = "IsUniform"), Skip]
        public static Wishart Uniform(int dimension)
        {
            return new Wishart(dimension);
        }

#if true
        /// <summary>
        /// Creates a one-dimensional Wishart with given shape and scale
        /// </summary>
        /// <param name="shape">The shape parameter</param>
        /// <param name="scale">The one-dimesional scale</param>
        public Wishart(double shape, double scale)
          : this(1)
        {
            Shape = shape;
            Rate.SetAllElementsTo(1.0 / scale);
        }

        /// <summary>
        /// Constructs a multi-dimensional Wishart with given shape and with
        /// a scale matrix which is set to a scaled identity matrix
        /// </summary>
        /// <param name="dimension">The dimension</param>
        /// <param name="shape">The shape parameter</param>
        /// <param name="scale">Used to scale the identity matrix</param>
        public Wishart(int dimension, double shape, double scale)
          : this(dimension)
        {
            Shape = shape;
            Rate.SetToIdentityScaledBy(1.0 / scale);
        }

        /// <summary>
        /// Constructs a multi-dimensional Wishart with given shape and scale matrix
        /// </summary>
        /// <param name="shape">The shape parameter</param>
        /// <param name="scale">The scale matrix</param>
        public Wishart(double shape, PositiveDefiniteMatrix scale)
          : this(scale.Rows)
        {
            Shape = shape;
            Rate.SetToInverse(scale);
        }
#endif

        /// <summary>
        /// Creates a new multi-dimensional Wishart with given shape and scale matrix
        /// </summary>
        /// <param name="shape">The shape parameter</param>
        /// <param name="scale">The scale matrix</param>
        /// <returns>A new Wishart distribution</returns>
        public static Wishart FromShapeAndScale(double shape, PositiveDefiniteMatrix scale)
        {
            Wishart result = new Wishart(scale.Rows);
            result.SetShapeAndScale(shape, scale);
            return result;
        }

        /// <summary>
        /// Creates a multi-dimensional Wishart with given shape and with
        /// a scale matrix which is set to a scaled identity matrix
        /// </summary>
        /// <param name="dimension">The dimension</param>
        /// <param name="shape">The shape parameter</param>
        /// <param name="scale">Used to scale the identity matrix</param>
        /// <returns>A new Wishart distribution</returns>
        public static Wishart FromShapeAndScale(int dimension, double shape, double scale)
        {
            return FromShapeAndRate(dimension, shape, 1.0 / scale);
        }

        /// <summary>
        /// Creates a multi-dimensional Wishart with given shape and rate matrix
        /// </summary>
        /// <param name="shape">The shape parameter</param>
        /// <param name="rate">The rate matrix</param>
        /// <returns>A new Wishart distribution</returns>
        [Construction("Shape", "Rate")]
        public static Wishart FromShapeAndRate(double shape, PositiveDefiniteMatrix rate)
        {
            Wishart result = new Wishart(rate.Rows);
            result.SetShapeAndRate(shape, rate);
            return result;
        }

        /// <summary>
        /// Creates a multi-dimensional Wishart with given shape and with
        /// a rate matrix which is set to a scaled identity matrix
        /// </summary>
        /// <param name="dimension">The dimension</param>
        /// <param name="shape">The shape parameter</param>
        /// <param name="rate">Used to scale the identity matrix</param>
        /// <returns>A new Wishart distribution</returns>
        public static Wishart FromShapeAndRate(int dimension, double shape, double rate)
        {
            //return FromShapeAndRate(shape, PositiveDefiniteMatrix.IdentityScaledBy(dimension,rate));
            Wishart result = new Wishart(dimension);
            result.Shape = shape;
            result.Rate.SetToIdentityScaledBy(rate);
            return result;
        }

        /// <summary>
        /// Constructs a Wishart distribution with the given mean and mean log determinant.
        /// </summary>
        /// <param name="mean">Desired expected value.</param>
        /// <param name="meanLogDet">Desired expected log determinant.</param>
        /// <param name="result"></param>
        /// <returns>A new Wishart distribution.</returns>
        /// <remarks>This function is equivalent to maximum-likelihood estimation of a Wishart distribution
        /// from data given by sufficient statistics.
        /// This function is significantly slower than the other constructors since it
        /// involves nonlinear optimization. The algorithm is a generalized Newton iteration, 
        /// described in "Estimating a Gamma distribution" by T. Minka, 2002.
        /// </remarks>
        public static Wishart FromMeanAndMeanLogDeterminant(PositiveDefiniteMatrix mean, double meanLogDet, Wishart result = null)
        {
            if (result == null)
                result = new Wishart(mean.Rows);
            // substituting rate = shape/mean gives the log-likelihood:
            // shape*(meanLogDet - logdet(mean) - d) + d*shape*log(shape) - gammaLn(shape,d)
            // now apply Newton's method
            int d = mean.Rows;
            double delta = mean.LogDeterminant() - meanLogDet;
            if (delta <= 2e-16)
                return Wishart.PointMass(mean);
            double shape = 0.5 * d / delta;
            for (int iter = 0; iter < 100; iter++)
            {
                double oldShape = shape;
                double g = d * Math.Log(shape) - delta - MMath.Digamma(shape, d);
                shape /= 1 + g / (d - shape * MMath.Trigamma(shape, d));
                if (Math.Abs(shape - oldShape) < 1e-8)
                    break;
            }
            if (Double.IsNaN(shape))
                throw new Exception("shape is nan");
            result.Shape = shape;
            result.Rate.SetToInverse(mean);
            result.Rate.Scale(shape);
            return result;
        }

        /// <summary>
        /// Modify the parameters so that the pdf has the given derivatives at a point.
        /// </summary>
        /// <param name="xChol">Cholesky factor of the point X</param>
        /// <param name="invX">Inverse of X.  Can be the same object as <c>this.Rate</c></param>
        /// <param name="dlogp">Desired derivative.  Can be the same object as <c>this.Rate</c></param>
        /// <param name="xxddlogp">tr(x tr(x dlogp')/dx)</param>
        /// <param name="forceProper">If true and both derivatives cannot be matched, match only the first.</param>
        /// <param name="shapeOffset"></param>
        /// <returns></returns>
        public void SetDerivatives(LowerTriangularMatrix xChol, PositiveDefiniteMatrix invX, PositiveDefiniteMatrix dlogp, double xxddlogp, bool forceProper, double shapeOffset = 0)
        {
            int dim = dlogp.Rows;
            double c = 0.5 * (dim + 1);
            double a = -xxddlogp / dim;
            if (forceProper)
            {
                // we want to ensure that a*ir - dlogp is nonnegative definite
                Matrix DL = dlogp * xChol;
                PositiveDefiniteMatrix LDL = new PositiveDefiniteMatrix(dim, dim);
                Matrix Lt = xChol.Transpose();
                LDL.SetToProduct(Lt, DL);
                LDL.Symmetrize();
                DL.SetToEigenvectorsOfSymmetric(LDL);
                // LDL is now a diagonal matrix of eigenvalues
                // find the largest eigenvalue
                double max = -shapeOffset;
                for (int i = 0; i < dim; i++)
                {
                    double eig = LDL[i, i];
                    if (eig > max)
                        max = eig;
                }
                if (a < max)
                    a = max;
            }
            Rate.SetToSum(a, invX, -1, dlogp);
            Shape = a + c + shapeOffset;
        }

        /// <summary>
        /// Creates a Wishart point mass at the specified location
        /// </summary>
        /// <param name="mean">The location of the point-mass</param>
        /// <returns>A new point mass Wishart distribution</returns>
        [Construction("Point", UseWhen = "IsPointMass")]
        public static Wishart PointMass(PositiveDefiniteMatrix mean)
        {
            Wishart g = new Wishart(mean.Rows);
            g.Point = mean;
            return g;
        }

        /// <summary>
        /// Creates a Wishart point mass at the specified location
        /// </summary>
        /// <param name="mean">The location of the point-mass is a vector where every element equals this value</param>
        /// <returns>A new point mass Wishart distribution</returns>
        public static Wishart PointMass(double mean)
        {
            return PointMass(new PositiveDefiniteMatrix(new double[,] { { mean } }));
        }

        #endregion

        /// <summary>
        /// ToString override
        /// </summary>
        /// <returns>String representation of the instance</returns>
        /// <exclude/>
        public override string ToString()
        {
            if (IsPointMass)
            {
                return StringUtil.JoinColumns("Wishart.PointMass(", Point, ")");
            }
            else if (IsUniform())
            {
                return "Wishart.Uniform(" + Dimension + ")";
            }
            else if (rate.IsPositiveDefinite())
            {
                return StringUtil.JoinColumns("Wishart(", shape.ToString("g4"), ", ", rate.Inverse(), ")[mean=", GetMean(), "]");
            }
            else
            {
                return StringUtil.JoinColumns("Wishart(", shape.ToString("g4"), ", rate=", rate, ")");
            }
        }
    }
}
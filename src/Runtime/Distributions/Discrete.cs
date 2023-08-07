// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Runtime.Serialization;

    using Microsoft.ML.Probabilistic.Factors.Attributes;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Serialization;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// An arbitrary distribution over integers [0,D-1].
    /// </summary>
    /// <remarks>
    /// The distribution is represented by a normalized Vector of length D.
    /// The Vector may be all zero to indicate an empty distribution such as the product of conflicting point masses.
    /// The probability of value x is available as this[x] or GetLogProb(x).
    /// </remarks>
    [Serializable]
    [Quality(QualityBand.Mature)]
    [DataContract]
    public class Discrete : IDistribution<int>,
        SettableTo<Discrete>, SettableToProduct<Discrete>, Diffable, SettableToUniform, SettableToPartialUniform<Discrete>,
        SettableToRatio<Discrete>, SettableToPower<Discrete>, SettableToWeightedSumExact<Discrete>,
        CanGetLogAverageOf<Discrete>, CanGetLogAverageOfPower<Discrete>, CanGetAverageLog<Discrete>, CanGetLogNormalizer,
        Sampleable<int>, CanGetMean<double>, CanGetVariance<double>, CanGetMode<int>
    {
        private const double UniformEps = 1e-5;

        /// <summary>
        /// Probability of each value (when not a point mass).
        /// </summary>
        /// <remarks>
        /// prob.Length == D.
        /// prob[i] >= 0.  sum_i prob[i] = 1.
        /// </remarks>
        [DataMember]
        protected Vector prob;

        /// <summary>
        /// Gets the dimension of this discrete distribution
        /// </summary>
        public int Dimension
        {
            get { return prob.Count; }
        }

        /// <summary>
        /// Gets the <see cref="Sparsity"/> specification of this Distribution.
        /// </summary>
        public Sparsity Sparsity
        {
            get { return prob.Sparsity; }
        }

        /// <summary>
        /// Clones this discrete distribution. 
        /// </summary>
        /// <returns>An object which is a clone of the current instance. This must be cast
        /// if you want to assign the result to a Discrete type</returns>
        public object Clone()
        {
            return new Discrete(this);
        }

        /// <summary>
        /// Sets/gets this distribution as a point distribution
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public int Point
        {
            get
            {
                if (Dimension <= 1) return 0;
                else
                {
                    return prob.IndexOfMaximum();
                }
            }
            set
            {
                if (Dimension > 1)
                {
                    if (prob.IsDense)
                    {
                        prob = SparseVector.Zero(Dimension);
                    }
                    else
                    {
                        prob.SetAllElementsTo(0);
                    }
                    prob[value] = 1;
                }
            }
        }

        /// <summary>
        /// Indicates whether or not this instance is a point mass.
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public bool IsPointMass
        {
            get
            {
                if (Dimension == 0)
                {
                    // For an empty sequence, All is true but Any is false.
                    return false;
                }
                else if (Dimension == 1)
                {
                    return true;
                }
                else
                {
                    // If exactly one element is not equal to 0 then this is a point mass.
                    int nonzeroCount = prob.CountAll(e => e != 0);
                    return nonzeroCount == 1;
                }
            }
        }

        /// <summary>
        /// Gets the maximum difference between the parameters of this discrete and that discrete
        /// </summary>
        /// <param name="that">That discrete</param>
        /// <returns>The maximum difference</returns>
        public double MaxDiff(object that)
        {
            Discrete thatd = that as Discrete;
            if (thatd == null) return Double.PositiveInfinity;
            if (IsPointMass) return (thatd.IsPointMass && thatd.Point == this.Point) ? 0.0 : double.PositiveInfinity;
            return prob.MaxDiff(thatd.prob); // assumes that PointMass representation is unique
        }

        /// <summary>
        /// Override of the Equals method
        /// </summary>
        /// <param name="that">The instance to compare to</param>
        /// <returns>True if the two distributions are the same in value, false otherwise</returns>
        /// <exclude/>
        public override bool Equals(object that)
        {
            Discrete thatd = that as Discrete;
            if (thatd == null) return false;
            return (MaxDiff(thatd) == 0.0);
        }

        /// <summary>
        /// Override of GetHashCode method
        /// </summary>
        /// <returns>The hash code for this instance</returns>
        /// <exclude/>
        public override int GetHashCode()
        {
            return prob.GetHashCode();
        }

        /// <summary>
        /// Sets this instance to a uniform discrete (i.e. probabilities all equal)
        /// </summary>
        public void SetToUniform()
        {
            this.prob.SetAllElementsTo(1.0 / this.Dimension);
        }

        /// <summary>
        /// Returns whether the discrete distribution is uniform or not
        /// </summary>
        /// <returns>True if uniform</returns>
        public bool IsUniform()
        {
            return this.prob.All(p => Math.Abs(Math.Log(p) + Math.Log(this.Dimension)) < UniformEps);
        }

        /// <summary>
        /// Sets the distribution to be uniform over its support.
        /// </summary>
        public void SetToPartialUniform()
        {
            this.SetToPartialUniformOf(this);
        }

        /// <summary>
        /// Sets the distribution to be uniform over the support of a given distribution.
        /// </summary>
        /// <param name="dist">The distribution which support will be used to setup the current distribution.</param>
        public void SetToPartialUniformOf(Discrete dist)
        {
            int nonZeroCount = dist.prob.CountAll(p => p > 0);
            double uniformProb = 1.0 / nonZeroCount;
            this.prob.SetToFunction(dist.prob, p => p > 0 ? uniformProb : 0);
        }

        /// <summary>
        /// Checks whether the distribution is uniform over its support.
        /// </summary>
        /// <returns>True if the distribution is uniform over its support, false otherwise.</returns>
        public bool IsPartialUniform()
        {
            int nonZeroCount = this.prob.CountAll(p => p > 0);
            return this.prob.All(p => Math.Abs(Math.Log(p) + Math.Log(nonZeroCount)) < UniformEps || p == 0.0);
        }

        /// <summary>
        /// Evaluates the log density at the specified domain value
        /// </summary>
        /// <param name="value">The point at which to evaluate</param>
        /// <returns>The log density</returns>
        public double GetLogProb(int value)
        {
            if (value < 0 || value >= Dimension)
                return double.NegativeInfinity;
            return Math.Log(prob[value]);
        }

        /// <summary>
        /// Gets the log normalizer of the distribution
        /// </summary>
        /// <returns></returns>
        public double GetLogNormalizer()
        {
            return -Math.Log(prob[0]);
        }

        /// <summary>
        /// Evaluates the density at the specified domain value
        /// </summary>
        /// <param name="value">The point at which to evaluate</param>
        /// <returns>The density</returns>
        public double Evaluate(int value)
        {
            if (value < 0 || value >= Dimension)
                return 0.0;
            return prob[value];
        }

        /// <summary>
        /// The expected logarithm of that distribution under this distribution.
        /// </summary>
        /// <param name="that">The distribution to take the logarithm of.</param>
        /// <returns><c>sum_x this.Evaluate(x)*Math.Log(that.Evaluate(x))</c></returns>
        /// <remarks>This is also known as the cross entropy.</remarks>
        public double GetAverageLog(Discrete that)
        {
            if (that.IsPointMass)
            {
                if (this.IsPointMass && Point.Equals(that.Point)) return 0.0;
                else return Double.NegativeInfinity;
            }
            else if (this.IsPointMass)
            {
                return Math.Log(that.prob[Point]);
            }
            else
            {
                double result = prob.Inner(that.prob, x => (x == 0.0) ? 0.0 : Math.Log(x));
                return result;
            }
        }

        /// <summary>
        /// The log of the integral of the product of this discrete and that discrete
        /// </summary>
        /// <param name="that">That discrete distribution</param>
        /// <returns>The log inner product</returns>
        public double GetLogAverageOf(Discrete that)
        {
            if (Dimension == 0) return 0;
            if (that.Dimension == Dimension)
            {
                if (IsUniform() || that.IsUniform())
                {
                    return -Math.Log(Dimension); // the code below does not return this exactly
                }
                // equivalent to:
                // (this*that).GetLogNormalizer() - this.GetLogNormalizer() - that.GetLogNormalizer()
                // but that fails if prob[0] == 0
                return Math.Log(prob.Inner(that.prob));
            }
            else
            {
                double sum = 0.0;
                int min = Math.Min(prob.Count, that.prob.Count);
                for (int i = 0; i < min; i++)
                {
                    sum += prob[i] * that.prob[i];
                }
                return Math.Log(sum);
            }
        }

        /// <summary>
        /// Get the integral of this distribution times another distribution raised to a power.
        /// </summary>
        /// <param name="that"></param>
        /// <param name="power"></param>
        /// <returns></returns>
        public double GetLogAverageOfPower(Discrete that, double power)
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
            else if (Dimension == 0)
            {
                return 0.0;
            }
            else
            {
                return Math.Log(prob.Reduce(0.0, that.prob, (partial, thisp, thatp) => (partial + thisp * Math.Pow(thatp, power))));
            }
        }

        /// <summary>
        /// The integral of the product between this discrete and that discrete. This
        /// is the probability that samples from this instance and that instance are equal
        /// </summary>
        /// <param name="that">That discrete distribution</param>
        /// <returns>The inner product</returns>
        public double ProbEqual(Discrete that)
        {
            if (IsPointMass)
            {
                return that.Evaluate(Point);
            }
            else if (that.IsPointMass)
            {
                return Evaluate(that.Point);
            }
            else
            {
                return prob.Inner(that.prob);
            }
        }

        /// <summary>
        /// Gets or sets the probability at the given index.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public double this[int index]
        {
            get
            {
                if (Dimension == 0)
                    return 1;
                else
                    return prob[index];
            }
        }

        /// <summary>
        /// Gets the probability at each index.
        /// </summary>
        /// <returns>The vector of probabilities</returns>
        public Vector GetProbs()
        {
            Vector v = Vector.Zero(prob.Count, Sparsity);
            return GetProbs(v);
        }

        /// <summary>
        /// Gets the probability at each index.
        /// </summary>
        /// <param name="result">When used internally, can be the same object as prob.</param>
        /// <returns>result</returns>
        public Vector GetProbs(Vector result)
        {
            if (Dimension > 0)
                SetToPadded(result, prob);
            return result;
        }

        /// <summary>
        /// Gets the vector of log probabilities for this distribution.
        /// </summary>
        /// <returns></returns>
        public Vector GetLogProbs()
        {
            Vector v = Vector.Zero(prob.Count, Sparsity);
            v.SetToFunction(prob, Math.Log);
            return v;
        }

        /// <summary>
        /// Gets a Vector of size this.Dimension.
        /// </summary>
        /// <returns>A pointer to the internal probs Vector of the object.</returns>
        /// <remarks>
        /// This function is intended to be used with SetProbs, to avoid allocating a new Vector.
        /// The return value should not be interpreted as a probs vector, but only a workspace filled
        /// with unknown data that can be overwritten.  Until SetProbs is called, the distribution object 
        /// is invalid once this workspace is modified.
        /// </remarks>
        public Vector GetWorkspace()
        {
            return prob;
        }

        /// <summary>
        /// Sets the probability of each index.
        /// </summary>
        /// <param name="probs">A vector of non-negative, finite numbers.  Need not sum to 1.</param>
        /// <remarks>
        /// Instead of allocating your own Vector to pass to SetProbs, you can call <see cref="GetWorkspace"/>,
        /// fill in the resulting Vector, and then pass it to SetProbs.
        /// </remarks>
        public void SetProbs(Vector probs)
        {
            if (probs != prob)
            {
                prob.SetTo(probs);
            }
            Normalize();
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        /// <summary>
        /// Sets the parameters of this instance to the parameters of that instance
        /// </summary>
        /// <param name="value">That instance</param>
        public void SetTo(Discrete value)
        {
            if (value.Dimension == Dimension)
            {
                prob.SetTo(value.prob);
            }
            else if (value.IsPointMass)
            {
                Point = value.Point;
            }
            else if (value.Dimension < Dimension)
            {
                throw new ArgumentException("value.Dimension (" + value.Dimension + ") < this.Dimension (" + Dimension + ")");
                for (int i = 0; i < value.Dimension; i++)
                {
                    prob[i] = value[i];
                }
                for (int i = value.Dimension; i < Dimension; i++)
                {
                    prob[i] = 0.0;
                }
            }
            else
            {
                // value.Dimension > Dimension
                throw new ArgumentException("value.Dimension (" + value.Dimension + ") > this.Dimension (" + Dimension + ")");
                for (int i = 0; i < Dimension; i++)
                {
                    prob[i] = value[i];
                }
                Normalize();
            }
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        /// <summary>
        /// Set this distribution to match the given distribution, but possibly over a larger domain
        /// </summary>
        /// <param name="value"></param>
        public void SetToPadded(Discrete value)
        {
            if (value.Dimension == Dimension)
            {
                prob.SetTo(value.prob);
            }
            else if (value.IsPointMass)
            {
                Point = value.Point;
            }
            else if (value.Dimension < Dimension)
            {
                SetToPadded(prob, value.prob);
            }
            else
            {
                // value.Dimension > Dimension
                throw new ArgumentException("value.Dimension (" + value.Dimension + ") > this.Dimension (" + Dimension + ")");
            }
        }

        private static void SetToPadded(Vector result, Vector value)
        {
            if (result.Count == value.Count)
            {
                result.SetTo(value);
                return;
            }
            if (!result.IsDense)
                result.SetAllElementsTo(0);
            result.SetSubvector(0, value);
            if (result.IsDense)
            {
                int resultCount = result.Count;
                for (int i = value.Count; i < resultCount; i++)
                {
                    result[i] = 0.0;
                }
            }
        }

        /// <summary>
        /// Sets the parameters to represent the product of two discrete distributions.
        /// </summary>
        /// <param name="a">The first discrete distribution</param>
        /// <param name="b">The second discrete distribution</param>
        public void SetToProduct(Discrete a, Discrete b)
        {
            if (a.IsPointMass)
            {
                if (b.IsPointMass && (a.Point != b.Point)) throw new AllZeroException();
                if (b[a.Point] == 0.0) throw new AllZeroException();
                SetTo(a);
            }
            else if (b.IsPointMass)
            {
                if (a[b.Point] == 0.0) throw new AllZeroException();
                SetTo(b);
            }
            else
            {
                prob.SetToProduct(a.prob, b.prob);
                Normalize();
            }
        }

        /// <summary>
        /// Creates a Discrete distribution which is the product of two Discrete distributions
        /// </summary>
        /// <param name="a">The first distribution</param>
        /// <param name="b">The second distribution</param>
        /// <returns>The resulting Discrete distribution</returns>
        public static Discrete operator *(Discrete a, Discrete b)
        {
            Discrete result = Discrete.Uniform(a.Dimension, a.Sparsity);
            result.SetToProduct(a, b);
            return result;
        }

#if false
        /// <summary>
        /// Sets the parameters to represent the ratio of two discrete distributions.
        /// </summary>
        /// <param name="numerator">The numerator discrete distribution</param>
        /// <param name="denominator">The denominator discrete distribution</param>
        public void SetToRatio(Discrete numerator, Discrete denominator)
        {
            if ((Dimension != numerator.Dimension) || (Dimension != denominator.Dimension)) {
                throw new ArgumentException("Discrete distributions have different dimensions");
            }
            if (Dimension == 1) return;
            if (numerator.IsPointMass) {
                if (denominator.IsPointMass) {
                    throw new DivideByZeroException();
                    //if (numerator.Point.Equals(denominator.Point)) {
                    //  SetToUniform();
                    //} else {
                    //  throw new DivideByZeroException();
                    //}
                } else {
                    Point = numerator.Point;
                }
            } else if (denominator.IsPointMass) {
                throw new DivideByZeroException();
            } else {
                prob.SetToRatio(numerator.prob, denominator.prob);
                Normalize();
            }
        }
#else
        public void SetToRatio(Discrete numerator, Discrete denominator, bool forceProper = false)
        {
            if (numerator.IsPointMass)
            {
                if (denominator.IsPointMass)
                {
                    if (numerator.Point.Equals(denominator.Point))
                    {
                        //SetToUniform();
                        Point = numerator.Point;
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
                if ((Dimension != numerator.Dimension) || (Dimension != denominator.Dimension))
                {
                    throw new ArgumentException("Discrete distributions have different dimensions");
                }
                if (denominator.prob.All(x => x > 0.0))
                    prob.SetToRatio(numerator.prob, denominator.prob);
                else
                {
                    for (int i = 0; i < Dimension; ++i)
                    {
                        if (denominator.prob[i] == 0.0)
                        {
                            if (numerator.prob[i] == 0.0) prob[i] = 0.0;
                            else throw new DivideByZeroException();
                        }
                        else prob[i] = numerator.prob[i] / denominator.prob[i];
                    }
                }
                Normalize();
            }
        }
#endif

        /// <summary>
        /// Creates a Discrete distribution which is the ratio of two Discrete distributions
        /// </summary>
        /// <param name="numerator">The numerator distribution</param>
        /// <param name="denominator">The denominator distribution</param>
        /// <returns>The resulting Discrete distribution</returns>
        public static Discrete operator /(Discrete numerator, Discrete denominator)
        {
            Discrete result = Discrete.Uniform(numerator.Dimension, numerator.prob.Sparsity);
            result.SetToRatio(numerator, denominator);
            return result;
        }

        /// <summary>
        /// Sets the parameters to represent the power of a discrete distributions.
        /// </summary>
        /// <param name="dist">The discrete distribution</param>
        /// <param name="exponent">The exponent</param>
        public void SetToPower(Discrete dist, double exponent)
        {
            if (dist.IsPointMass)
            {
                if (exponent == 0)
                {
                    SetToUniform();
                }
                else if (exponent < 0)
                {
                    if (Dimension > 1)
                        throw new DivideByZeroException("The exponent is negative and the distribution is a point mass");
                }
                else
                {
                    Point = dist.Point;
                }
            }
            else
            {
                prob.SetToPower(dist.prob, exponent);
                Normalize();
            }
        }

        /// <summary>
        /// Raises a distribution to a power.
        /// </summary>
        /// <param name="dist">The distribution.</param>
        /// <param name="exponent">The power to raise to.</param>
        /// <returns><paramref name="dist"/> raised to power <paramref name="exponent"/>.</returns>
        public static Discrete operator ^(Discrete dist, double exponent)
        {
            Discrete result = new Discrete(dist.Dimension);
            result.SetToPower(dist, exponent);
            return result;
        }

        /// <summary>
        /// Sets the parameters to represent the weighted sum of two discrete distributions.
        /// </summary>
        /// <param name="dist1">The first discrete distribution.  Can be the same object as <c>this</c></param>
        /// <param name="weight1">The first weight</param>
        /// <param name="dist2">The second discrete distribution.  Cannot be the same object as <c>this</c></param>
        /// <param name="weight2">The second weight</param>
        public void SetToSum(double weight1, Discrete dist1, double weight2, Discrete dist2)
        {
            if (weight1 + weight2 == 0)
                SetToUniform();
            else if (weight1 + weight2 < 0)
                throw new ArgumentException("weight1 (" + weight1 + ") + weight2 (" + weight2 + ") < 0");
            else if (weight1 == 0)
                SetTo(dist2);
            else if (weight2 == 0)
                SetTo(dist1);
            // if dist1 == dist2 then we must return dist1, with no roundoff error
            else if (dist1.Equals(dist2))
                SetTo(dist1);
            else if (double.IsPositiveInfinity(weight1))
            {
                if (double.IsPositiveInfinity(weight2))
                {
                    throw new ArgumentException("both weights are infinity");
                }
                else
                {
                    SetTo(dist1);
                }
            }
            else if (double.IsPositiveInfinity(weight2))
                SetTo(dist2);
            else
            {
                dist1.GetProbs(prob);
                if (dist2.IsPointMass)
                {
                    prob.Scale(weight1);
                    prob[dist2.Point] += weight2;
                }
                else if (dist2.Dimension < this.Dimension)
                {
                    Vector prob2 = Vector.Zero(prob.Count, prob.Sparsity);
                    dist2.GetProbs(prob2);
                    prob.SetToSum(weight1, prob, weight2, prob2);
                }
                else
                {
                    prob.SetToSum(weight1, prob, weight2, dist2.prob);
                }
                Normalize();
            }
        }

        /// <summary>
        /// Normalizes this distribution - i.e. sets the probabilities to sum to 1.
        /// This is called internally after product operations, sum operations etc.
        /// </summary>
        /// <returns>The normalizing factor</returns>
        protected double Normalize()
        {
            if (Dimension == 1)
            {
                double sum = prob[0];
                prob[0] = 1.0;
                return sum;
            }
            else if (Dimension == 0)
                return 1.0;
            else
            {
                double sum = prob.Sum();
                if (double.IsInfinity(sum) || double.IsNaN(sum)) throw new DivideByZeroException();
                if (sum > 0) prob.Scale(1.0 / sum);
                return sum;
            }
        }

        /// <summary>
        /// Returns a sample from this discrete distribution
        /// </summary>
        /// <returns>The sample value</returns>
        [Stochastic]
        public int Sample()
        {
            if (Dimension == 1) return 0;

            if (IsPointMass)
                return Point;
            else
                return Rand.Sample(prob);
        }

        /// <summary>
        /// Returns a sample from this discrete distribution
        /// </summary>
        /// <param name="result">This parameter is ignored and is only present to support the Sampleable interface</param>
        /// <returns>The sample value</returns>
        [Stochastic]
        public int Sample(int result)
        {
            return Sample();
        }


        /// <summary>
        /// Returns a sample from a discrete distribution with the specified probabilities
        /// </summary>
        /// <param name="probs">The parameters of the discrete distribution</param>
        /// <returns>The sample value</returns>
        [Stochastic]
        public static int Sample(Vector probs)
        {
            return Rand.Sample(probs);
        }

        /// <summary>
        /// Parameterless constructor required for serialization 
        /// </summary>
        private Discrete()
        {
        }

        /// <summary>
        /// Creates a uniform Discrete distribution, from 0 to dimension-1.
        /// </summary>
        /// <param name="dimension"></param>
        protected Discrete(int dimension)
        {
            prob = Vector.Zero(dimension);
            SetToUniform();
        }

        /// <summary>
        /// Creates a uniform Discrete distribution with a specified sparsity, from 0 to dimension-1.
        /// </summary>
        /// <param name="dimension">Dimension</param>
        /// <param name="sparsity">Sparsity</param>
        protected Discrete(int dimension, Sparsity sparsity)
        {
            prob = Vector.Zero(dimension, sparsity);
            SetToUniform();
        }

        /// <summary>
        /// Copy constructor
        /// </summary>
        /// <param name="that">The discrete instance to copy</param>
        public Discrete(Discrete that)
        {
            prob = (Vector)that.prob.Clone();
        }

        /// <summary>
        /// Creates a Discrete distribution from the given probabilities.
        /// </summary>
        /// <param name="probs"></param>
        [Construction("GetWorkspace")]
        public Discrete(Vector probs)
        {
            prob = Vector.Copy(probs);
            Normalize();
        }

        /// <summary>
        /// Creates a Discrete distribution from the given probabilities.
        /// </summary>
        /// <param name="probs"></param>
        public Discrete(params double[] probs)
        {
            //if (probs.Length == 0) throw new ArgumentOutOfRangeException("probs.Length == 0");
            prob = Vector.FromArray(probs);
            Normalize();
        }

        /// <summary>
        /// Creates a Discrete distribution which allows only one value.
        /// </summary>
        /// <param name="value">The allowed value.</param>
        /// <param name="numValues">The number of values in the domain.</param>
        /// <returns></returns>
        [Construction("Point", "Dimension", UseWhen = "IsPointMass")]
        public static Discrete PointMass(int value, int numValues)
        {
            if (value < 0 || value >= numValues) throw new ArgumentException(String.Format("value ({0}) is not in the range [0, numValues-1 ({1})]", value, numValues - 1));
            Discrete d = Discrete.Uniform(numValues);
            d.Point = value;
            return d;
        }

        /// <summary>
        /// Creates a uniform Discrete distribution over the values from 0 to numValues-1
        /// </summary>
        /// <returns></returns>
        [Skip]
        public static Discrete Uniform(int numValues)
        {
            Discrete d = new Discrete(numValues);
            return d;
        }

        /// <summary>
        /// Creates a uniform Discrete distribution with a specified sparsity over the values from 0 to numValues-1
        /// </summary>
        /// <param name="numValues">Number of values</param>
        /// <param name="sparsity">Sparsity</param>
        /// <returns></returns>
        [Construction("Dimension", "Sparsity", UseWhen = "IsUniform"), Skip]
        public static Discrete Uniform(int numValues, Sparsity sparsity)
        {
            Discrete d = new Discrete(numValues, sparsity);
            return d;
        }

        /// <summary>
        /// Creates a Discrete distribution which is uniform over values from
        /// start to end inclusive.
        /// </summary>
        /// <param name="numValues">Number of values</param>
        /// <param name="start">The first value included in the distribution</param>
        /// <param name="end">The last value included in the distribution</param>
        /// <returns>Discrete which is uniform over the specified range (and zero elsewhere).</returns>
        public static Discrete UniformInRange(int numValues, int start, int end)
        {
            var probs = PiecewiseVector.Zero(numValues);
            probs.SetToConstantInRange(start, end, 1.0);
            return new Discrete(probs);
        }


        /// <summary>
        /// Creates a Discrete distribution which is uniform over values in
        /// multiple ranges specified by pairs of start and end values. These pairs
        /// are specified as adjacent values in an array whose length must therefore be
        /// even.
        /// </summary>
        /// <param name="numValues">Number of values</param>
        /// <param name="startEndPairs">Sequence of start and end pairs</param>
        /// <returns>Discrete which is uniform over the specified ranges (and zero elsewhere).</returns>
        public static Discrete UniformInRanges(int numValues, params int[] startEndPairs)
        {
            return UniformInRanges(numValues, (IEnumerable<int>)startEndPairs);
        }

        /// <summary>
        /// Creates a Discrete distribution which is uniform over values in
        /// multiple ranges specified by pairs of start and end values. These pairs
        /// are specified as adjacent values in an enumerable whose length must therefore be
        /// even.
        /// </summary>
        /// <param name="numValues">Number of values</param>
        /// <param name="startEndPairs">Sequence of start and end pairs</param>
        /// <returns>Discrete which is uniform over the specified ranges (and zero elsewhere).</returns>
        public static Discrete UniformInRanges(int numValues, IEnumerable<int> startEndPairs)
        {
            var probs = PiecewiseVector.Zero(numValues);
            probs.SetToConstantInRanges(startEndPairs, 1.0);
            return new Discrete(probs);
        }

        public static Discrete Zero(int numValues)
        {
            return new Discrete()
            {
                prob = Vector.Zero(numValues)
            };
        }

        public bool IsZero()
        {
            return prob.EqualsAll(0);
        }

        /// <summary>
        /// Override of ToString method
        /// </summary>
        /// <returns>String representation of this instance</returns>
        /// <exclude/>
        public override string ToString()
        {
            return ToString("g4");
        }

        public virtual string ToString(string format)
        {
            return ToString(format, " ");
        }

        public virtual string ToString(string format, string delimiter)
        {
            if (IsPointMass)
            {
                return "Discrete.PointMass(" + Point + ")";
            }
            return "Discrete(" + prob.ToString(format, delimiter) + ")";
        }

        /// <summary>
        /// Gets the mean of the distribution
        /// </summary>
        /// <returns></returns>
        public double GetMean()
        {
            if (IsPointMass) return (double)Point;
            return prob.SumI();
        }

        /// <summary>
        /// Gets the median of the distribution
        /// </summary>
        /// <returns>The median</returns>
        public int GetMedian()
        {
            if (this.IsPointMass)
            {
                return this.Point;
            }

            int result = this.prob.IndexAtCumulativeSum(0.5);

            const int MedianNotFound = -1;
            Debug.Assert(result != MedianNotFound);

            return result;
        }

        /// <summary>
        /// Gets the mode of the distribution
        /// </summary>
        /// <returns></returns>
        public int GetMode()
        {
            if (IsPointMass) return Point;
            return prob.IndexOfMaximum();
        }

        /// <summary>
        /// Gets the variance of the distribution
        /// </summary>
        /// <returns></returns>
        public double GetVariance()
        {
            if (IsPointMass) return 0.0;
            else
            {
                double mean = GetMean();
                return prob.SumISq() - mean * mean;
            }
        }

        /// <summary>
        /// Creates a distribution with reduced support.
        /// </summary>
        /// <param name="lowerBound">The smallest allowed value.</param>
        /// <param name="upperBound">The largest allowed value.</param>
        /// <returns></returns>
        public Discrete Truncate(int lowerBound, int upperBound)
        {
            Vector probs = this.prob.Subvector(0, Math.Min(upperBound + 1, this.Dimension));
            if (lowerBound > 0)
            {
                probs.SetSubvector(0, Vector.Zero(lowerBound));
            }
            return new Discrete(probs);
        }
    }
}

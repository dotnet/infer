// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Collections.Generic;
    using System.Text;
    using System.Linq;
    using System.Runtime.Serialization;
    using Math;
    using Utilities;
    using Factors.Attributes;
    using Collections;
    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// A generic base class for discrete distributions over a type T.
    /// </summary>
    /// <remarks>
    /// This class makes it straightforward to implement discrete distributions
    /// over types other than int.  It is only necessary to make a subclass of
    /// this abstract class, add a constructor and implement two methods to convert 
    /// the desired type T to and from an integer. 
    /// 
    /// Internally, this class wraps a Discrete distribution and uses it to provide
    /// all functionality.
    /// </remarks>
    /// <typeparam name="T">The domain type of this distribution</typeparam>
    /// <typeparam name="TThis">The type of the subclass (allows factory methods return the correct type)</typeparam>
    [DataContract]
    public abstract class GenericDiscreteBase<T, TThis> : IDistribution<T>, SettableTo<TThis>, SettableToProduct<TThis>,
                                                          SettableToRatio<TThis>, SettableToPower<TThis>, SettableToWeightedSumExact<TThis>, SettableToPartialUniform<TThis>,
                                                          CanGetLogAverageOf<TThis>, CanGetLogAverageOfPower<TThis>,
                                                          CanGetAverageLog<TThis>, Sampleable<T>
        where TThis : GenericDiscreteBase<T, TThis>, new()
    {
        [DataMember]
        protected Discrete disc;

        /// <summary>
        /// The dimension of this discrete distribution.
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public int Dimension
        {
            get { return disc.Dimension; }
        }

        /// <summary>
        /// Creates a discrete
        /// </summary>
        /// <param name="dimension">The dimension of the underlying discrete</param>
        /// <param name="sparsity">The sparsity of the underlying discrete</param>
        protected GenericDiscreteBase(int dimension, Sparsity sparsity)
        {
            disc = Discrete.Uniform(dimension, sparsity);
        }

        /// <summary>
        /// Creates a uniform distribution.
        /// </summary>
        [Construction(UseWhen = "IsUniform"), Skip]
        public static TThis Uniform()
        {
            var res = new TThis();
            return res;
        }

        /// <summary>
        /// Creates a distribution using the probabilities from the given discrete distribution.
        /// </summary>
        /// <param name="d"></param>
        public static TThis FromDiscrete(Discrete d)
        {
            return FromVector(d.GetWorkspace());
        }

        /// <summary>
        /// Creates a distribution from the given probabilities.
        /// </summary>
        /// <param name="probs"></param>
        public static TThis FromProbs(params double[] probs)
        {
            return FromVector(Vector.FromArray(probs));
        }

        /// <summary>
        /// Creates a distribution from the given vector of probabilities.
        /// </summary>
        /// <param name="probs"></param>
        [Construction("GetWorkspace")]
        public static TThis FromVector(Vector probs)
        {
            TThis res = new TThis();
            res.disc.SetProbs(probs);
            return res;
        }

        /// <summary>
        /// Creates a Discrete distribution which is uniform over values from
        /// start to end inclusive.
        /// </summary>
        /// <param name="start">The first value included in the distribution</param>
        /// <param name="end">The last value included in the distribution</param>
        /// <returns>Discrete which is uniform over the specified range (and zero elsewhere).</returns>
        public static TThis UniformInRange(T start, T end)
        {
            TThis res = new TThis();
            var probs = PiecewiseVector.Zero(res.Dimension);
            probs.SetToConstantInRange(res.ConvertToInt(start), res.ConvertToInt(end), 1.0);
            res.disc.SetProbs(probs);
            return res;
        }


        /// <summary>
        /// Creates a Discrete distribution which is uniform over values in
        /// multiple ranges specified by pairs of start and end values. These pairs
        /// are specified as adjacent values in an array whose length must therefore be
        /// even.
        /// </summary>
        /// <param name="startEndPairs">Sequence of start and end pairs</param>
        /// <returns>Discrete which is uniform over the specified range (and zero elsewhere).</returns>
        public static TThis UniformInRanges(params T[] startEndPairs)
        {
            return UniformInRanges((IEnumerable<T>) startEndPairs);
        }

        /// <summary>
        /// Creates a Discrete distribution which is uniform over values in
        /// multiple ranges specified by pairs of start and end values. These pairs
        /// are specified as adjacent values in an enumerable whose length must therefore be
        /// even.
        /// </summary>
        /// <param name="startEndPairs">Sequence of start and end pairs</param>
        /// <returns>Discrete which is uniform over the specified range (and zero elsewhere).</returns>
        public static TThis UniformInRanges(IEnumerable<T> startEndPairs)
        {
            TThis res = new TThis();
            var probs = PiecewiseVector.Zero(res.Dimension);
            probs.SetToConstantInRanges(startEndPairs.Select(res.ConvertToInt), 1.0);
            res.disc.SetProbs(probs);
            return res;
        }

        /// <summary>
        /// Creates a Discrete distribution which is uniform over the specified set of values and zero elsewhere.
        /// </summary>
        /// <param name="values">The values.</param>
        /// <returns>Discrete which is uniform over the specified set of values and zero elsewhere.</returns>
        public static TThis UniformOver(params T[] values)
        {
            return UniformOver((IEnumerable<T>)values);
        }

        /// <summary>
        /// Creates a Discrete distribution which is uniform over the specified set of values and zero elsewhere.
        /// </summary>
        /// <param name="values">The values.</param>
        /// <returns>Discrete which is uniform over the specified set of values and zero elsewhere.</returns>
        public static TThis UniformOver(IEnumerable<T> values)
        {
            TThis res = new TThis();
            var probs = PiecewiseVector.Zero(res.Dimension);
            foreach (T value in values)
            {
                probs[res.ConvertToInt(value)] = 1.0;
            }

            res.disc.SetProbs(probs);
            return res;
        }

        /// <summary>
        /// Creates a point mass distribution.
        /// </summary>
        /// <param name="value">The allowed value.</param>
        /// <returns></returns>
        [Construction("Point", UseWhen = "IsPointMass")]
        public static TThis PointMass(T value)
        {
            var res = new TThis();
            ((HasPoint<T>) res).Point = value;
            return res;
        }

        /// <summary>
        /// Evaluates the log density at the specified domain value
        /// </summary>
        /// <param name="value">The point at which to evaluate</param>
        /// <returns>The log density</returns>
        public double GetLogProb(T value)
        {
            return disc.GetLogProb(ConvertToInt(value));
        }

        /// <summary>
        /// Gets the probability of a given value.
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public double this[T value]
        {
            get { return disc[ConvertToInt(value)]; }
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
            return disc.GetWorkspace();
        }

        /// <summary>
        /// Gets the internal discrete distribution that this distribution wraps.
        /// </summary>
        /// <returns>The internal discrete distribution</returns>
        public Discrete GetInternalDiscrete()
        {
            return disc;
        }

        /// <summary>
        /// Gets the probability at each index.
        /// </summary>
        /// <returns>The vector of probabilities</returns>
        public Vector GetProbs()
        {
            return disc.GetProbs();
        }

        /// <summary>
        /// Gets the log-probability at each index.
        /// </summary>
        /// <returns>The vector of log-probabilities</returns>
        public Vector GetLogProbs()
        {
            return disc.GetLogProbs();
        }

        /// <summary>
        /// Gets the mode of the distribution
        /// </summary>
        /// <returns>The value with the highest probability</returns>
        public T GetMode()
        {
            return ConvertFromInt(disc.GetMode());
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
            disc.SetProbs(probs);
        }

        /// <summary>
        /// Returns a clone of this distribution.
        /// </summary>
        /// <returns></returns>
        public virtual object Clone()
        {
            var clone = new TThis();
            clone.disc.SetProbs(disc.GetProbs());
            return clone;
        }

        /// <summary>
        /// Point property
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public T Point
        {
            get { return ConvertFromInt(disc.Point); }
            set { disc.Point = ConvertToInt(value); }
        }

        /// <summary>
        /// Returns true if the distribution is a point mass.
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public bool IsPointMass
        {
            get { return disc.IsPointMass; }
        }

        /// <summary>
        /// Returns true if the distribution is uniform.
        /// </summary>
        /// <returns>True if uniform</returns>
        public bool IsUniform()
        {
            return disc.IsUniform();
        }

        /// <summary>
        /// Sets this instance to a uniform discrete (i.e. probabilities all equal)
        /// </summary>
        public void SetToUniform()
        {
            disc.SetToUniform();
        }

        /// <summary>
        /// Checks whether the distribution is uniform over its support.
        /// </summary>
        /// <returns>True if the distribution is uniform over its support, false otherwise.</returns>
        public bool IsPartialUniform()
        {
            return this.disc.IsPartialUniform();
        }

        /// <summary>
        /// Sets the distribution to be uniform over its support.
        /// </summary>
        public void SetToPartialUniform()
        {
            this.disc.SetToPartialUniform();
        }

        /// <summary>
        /// Sets the distribution to be uniform over the support of a given distribution.
        /// </summary>
        /// <param name="dist">The distribution which support will be used to setup the current distribution.</param>
        public void SetToPartialUniformOf(TThis dist)
        {
            this.disc.SetToPartialUniformOf(dist.disc);
        }

        /// <summary>
        /// Gets the maximum difference between the parameters of this discrete and that discrete
        /// </summary>
        /// <param name="that">That discrete</param>
        /// <returns>The maximum difference</returns>
        public double MaxDiff(object that)
        {
            var db = that as TThis;
            if (db == null) return double.NegativeInfinity;
            return disc.MaxDiff(db.disc);
        }

        /// <summary>
        /// Converts an integer to an item of type T.
        /// </summary>
        /// <param name="i"></param>
        /// <returns></returns>
        public abstract T ConvertFromInt(int i);

        /// <summary>
        /// Converts an item of type T to an integer.
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public abstract int ConvertToInt(T value);

        /// <summary>
        /// Implements a custom ToString() for items of type T.
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        protected virtual string ToString(T value)
        {
            return value.ToString();
        }

        /// <summary>
        /// Returns the log probability of the value under this distribution.
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        double CanGetLogProb<T>.GetLogProb(T value)
        {
            return disc.GetLogProb(ConvertToInt(value));
        }

        /// <summary>
        /// Sets the parameters of this instance to the parameters of that instance
        /// </summary>
        /// <param name="value">That instance</param>
        public void SetTo(TThis value)
        {
            disc.SetTo(value.disc);
        }

        /// <summary>
        /// Sets the parameters to represent the product of two discrete distributions.
        /// </summary>
        /// <param name="a">The first discrete distribution</param>
        /// <param name="b">The second discrete distribution</param>
        public void SetToProduct(TThis a, TThis b)
        {
            disc.SetToProduct(a.disc, b.disc);
        }

        /// <summary>
        /// Sets the parameters to represent the ratio of two discrete distributions.
        /// </summary>
        /// <param name="numerator">The numerator discrete distribution</param>
        /// <param name="denominator">The denominator discrete distribution</param>
        /// <param name="forceProper">Whether to force the returned distribution to be proper</param>
        public void SetToRatio(TThis numerator, TThis denominator, bool forceProper)
        {
            disc.SetToRatio(numerator.disc, denominator.disc, forceProper);
        }

        /// <summary>
        /// Sets the parameters to represent the power of a discrete distributions.
        /// </summary>
        /// <param name="value">The discrete distribution</param>
        /// <param name="exponent">The exponent</param>
        public void SetToPower(TThis value, double exponent)
        {
            disc.SetToPower(value.disc, exponent);
        }

        /// <summary>
        /// Sets the parameters to represent the weighted sum of two discrete distributions.
        /// </summary>
        /// <param name="value1">The first discrete distribution</param>
        /// <param name="weight1">The first weight</param>
        /// <param name="value2">The second discrete distribution</param>
        /// <param name="weight2">The second weight</param>
        public void SetToSum(double weight1, TThis value1, double weight2, TThis value2)
        {
            disc.SetToSum(weight1, value1.disc, weight2, value2.disc);
        }

        /// <summary>
        /// The log of the integral of the product of this discrete distribution and that discrete distribution
        /// </summary>
        /// <param name="that">That discrete distribution</param>
        /// <returns>The log inner product</returns>
        public double GetLogAverageOf(TThis that)
        {
            return disc.GetLogAverageOf(that.disc);
        }

        /// <summary>
        /// Get the integral of this distribution times another distribution raised to a power.
        /// </summary>
        /// <param name="that"></param>
        /// <param name="power"></param>
        /// <returns></returns>
        public double GetLogAverageOfPower(TThis that, double power)
        {
            return disc.GetLogAverageOfPower(that.disc, power);
        }

        /// <summary>
        /// The expected logarithm of that distribution under this distribution.
        /// </summary>
        /// <param name="that">The distribution to take the logarithm of.</param>
        /// <returns><c>sum_x this.Evaluate(x)*Math.Log(that.Evaluate(x))</c></returns>
        /// <remarks>This is also known as the cross entropy.</remarks>
        public double GetAverageLog(TThis that)
        {
            return disc.GetAverageLog(that.disc);
        }

        /// <summary>
        /// Returns the domain values corresponding to the first and last indices of non-zero probabilities.
        /// </summary>
        public (T, T) ValueRange
        {
            get
            {
                var probs = disc.GetWorkspace();
                int firstIndex = probs.FindFirstIndex(p => p > 0.0);
                int lastIndex = probs.FindLastIndex(p => p > 0.0);
                return (ConvertFromInt(firstIndex), ConvertFromInt(lastIndex));
            }
        }

        public IntPair IndexRange
        {
            get
            {
                var probs = disc.GetWorkspace();
                return new IntPair(probs.FindFirstIndex(p => p > 0.0), probs.FindLastIndex(p => p > 0.0));
            }
        }

        /// <summary>
        /// Returns a string representation of this distribution.
        /// </summary>
        /// <returns></returns>
        public virtual string ToString(string format, string delimiter)
        {
            if (IsPointMass)
            {
                if (format == null) return ToString(Point);
                return "PointMass(" + ToString(Point) + ")";
            }
            var prob = disc.GetWorkspace();
            if (format == null) return prob.ToString(null, delimiter, i => ToString(ConvertFromInt(i)));
            return "Discrete(" + prob.ToString(format, delimiter, i => ToString(ConvertFromInt(i))) + ")";
        }

        public override string ToString()
        {
            return this.ToString("g4", ",");
        }

        public virtual string ToString(string format)
        {
            return this.ToString(format, ",");
        }

        /// <summary>
        /// Returns a sample from the distribution
        /// </summary>
        /// <returns>The sample value</returns>
        [Stochastic]
        public T Sample()
        {
            return ConvertFromInt(disc.Sample());
        }

        /// <summary>
        /// Returns a sample from the distribution
        /// </summary>
        /// <param name="result">Not used</param>
        /// <returns>The sample value</returns>
        [Stochastic]
        public T Sample(T result)
        {
            return Sample();
        }

        /// <summary>
        /// Override of the Equals method
        /// </summary>
        /// <param name="obj">The instance to compare to</param>
        /// <returns>True if the two distributions are the same in value, false otherwise</returns>
        /// <exclude/>
        public override bool Equals(object obj)
        {
            var db = obj as TThis;
            if (db == null) return false;
            return disc.Equals(db.disc);
        }

        /// <summary>
        /// Override of GetHashCode method
        /// </summary>
        /// <returns>The hash code for this instance</returns>
        /// <exclude/>
        public override int GetHashCode()
        {
            return disc.GetHashCode();
        }
    }
}
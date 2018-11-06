// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Runtime.Serialization;

    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Serialization;
    using Microsoft.ML.Probabilistic.Utilities;
    
    /// <summary>
    /// Abstract base class for a homogeneous sparse list of distributions. The class supports
    /// an approximation tolerance which allows elements close to the common value to be
    /// automatically reset to the common value. The list implements the
    /// interfaces which allow these distributions to participate in message passing.
    /// </summary>
    /// <typeparam name="TDist">The distribution type for the elements of the sparse list.</typeparam>
    /// <typeparam name="TDomain">The domain type for the distribution elements.</typeparam>
    /// <typeparam name="TThis">The type of a specialized class derived from this class.</typeparam>
    [Serializable]
    [DataContract]
    public abstract class SparseDistributionList<TDist, TDomain, TThis> :
        ApproximateSparseList<TDist>,
        IDistribution<ISparseList<TDomain>>, 
        Sampleable<ISparseList<TDomain>>, 
        SettableTo<TThis>,
        SettableToProduct<TThis>,
        SettableToPower<TThis>,
        SettableToRatio<TThis>,
        SettableToWeightedSum<TThis>,
        CanGetLogAverageOf<TThis>,
        CanGetLogAverageOfPower<TThis>,
        CanGetAverageLog<TThis>,
        CanGetMean<ISparseList<double>>, CanGetVariance<ISparseList<double>>
        where TDist : struct,
            SettableToUniform,
            Diffable,
            Sampleable<TDomain>,
            SettableToProduct<TDist>,
            SettableToRatio<TDist>,
            SettableToPower<TDist>,
            SettableToWeightedSum<TDist>,
            CanGetLogAverageOf<TDist>,
            CanGetAverageLog<TDist>,
            CanGetLogAverageOfPower<TDist>,
            CanGetMean<double>,
            CanGetVariance<double>,
            IDistribution<TDomain>
        where TThis : SparseDistributionList<TDist, TDomain, TThis>, new()
    {
        /// <summary>
        /// Gets the dimension of the sparse distribution list.
        /// </summary>
        public int Dimension
        {
            get
            {
                return this.Count;
            }
        }

        /// <summary>
        /// Gets or sets the instance as a point mass.
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public ISparseList<TDomain> Point
        {
            get
            {
                return SparseList<TDomain>.FromSparseValues(
                    this.Count,
                    this.CommonValue.Point,
                    SparseValues.Select(x => new ValueAtIndex<TDomain>(x.Index, x.Value.Point)).ToList());
            }

            set
            {
                Argument.CheckIfNotNull(value, "Sparse distribution list cannot be set to a null point mass");

                ISparseList<TDomain> p;
                if (!(value is ISparseList<TDomain>))
                {
                    p = SparseList<TDomain>.Copy(value);
                }
                else
                {
                    p = (ISparseList<TDomain>)value;
                }

                var sen = p.GetSparseEnumerator();
                TDist commonDist = default(TDist);
                commonDist.Point = sen.CommonValue;
                this.CommonValue = commonDist;
                SparseValues.Clear();
                while (sen.MoveNext())
                {
                    TDist dist = default(TDist);
                    dist.Point = sen.Current;
                    SparseValues.Add(new ValueAtIndex<TDist>(sen.CurrentIndex, dist));
                }
            }
        }

        /// <summary>
        /// Gets a value indicating whether the instance is a point mass.
        /// </summary>
        /// <returns>True if all elements of the list are point masses, false otherwise.</returns>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public bool IsPointMass
        {
            get
            {
                return this.All(x => x.IsPointMass);
            }
        }
        
        /// <summary>
        /// Returns a new instance of the <see cref="SparseDistributionList{TDist,TDomain,TThis}"/> class
        /// with the specified number of elements all of which are set to uniform.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        /// <returns>The new instance.</returns>
        public static new TThis FromSize(int size, double tolerance)
        {
            var commonValue = default(TDist);
            commonValue.SetToUniform();
            return Constant(size, commonValue, tolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseDistributionList{TDist,TDomain,TThis}"/> class with the specified number of elements
        /// all of which are set to the specified value.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="commonValue">The common value.</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        /// <returns>The new instance.</returns>
        public static new TThis Constant(int size, TDist commonValue, double tolerance)
        {
            var result = new TThis();
            result.Tolerance = tolerance;
            result.Count = size;
            result.SparseValues = new List<ValueAtIndex<TDist>>();
            result.CommonValue = commonValue;
            return result;
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseDistributionList{TDist,TDomain,TThis}"/> class of a given length
        /// and assigns all elements the given value, except for the specified list of sparse values.
        /// This list is stored internally as is, so MUST be sorted by index and must not be modified
        /// externally after being passed in.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="commonValue">The common value.</param>
        /// <param name="sortedSparseValues">The sorted list of non-common values.</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        /// <returns>The new instance.</returns>
        [Construction("Count", "CommonValue", "SparseValues", "Tolerance")]
        public static new TThis FromSparseValues(
            int size,
            TDist commonValue,
            List<ValueAtIndex<TDist>> sortedSparseValues,
            double tolerance)
        {
            var result = Constant(size, commonValue, tolerance);
            result.SparseValues = sortedSparseValues;
            return result;
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseDistributionList{TDist,TDomain,TThis}"/> class from a sparse list of distributions.
        /// </summary>
        /// <param name="distributions">The sparse list of distributions.</param>
        /// <param name="tolerance">The tolerance.</param>
        /// <returns>The new instance.</returns>
        public static TThis FromSparseList(ISparseList<TDist> distributions, double tolerance)
        {
            var result = new TThis();
            result.Count = distributions.Count;
            result.Tolerance = tolerance;
            result.SetTo(distributions);
            return result;
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseDistributionList{TDist,TDomain,TThis}"/> class
        /// with the specified number of elements all of which are set to uniform.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <returns>The new instance.</returns>
        public static new TThis FromSize(int size)
        {
            var commonValue = default(TDist);
            commonValue.SetToUniform();
            return Constant(size, commonValue);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseDistributionList{TDist,TDomain,TThis}"/> class
        /// with the specified number of elements all of which are set to the specified value.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="commonValue">The common value.</param>
        /// <returns>The instance.</returns>
        public static new TThis Constant(int size, TDist commonValue)
        {
            // Caution - this must not call into the tolerance version of Constant() with the
            // default tolerance, because TThis static constructor may not yet have been invoked.
            // So we must explicitly create the new instance here.
            var result = new TThis();
            result.Count = size;
            result.SparseValues = new List<ValueAtIndex<TDist>>();
            result.CommonValue = commonValue;
            return result;
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseDistributionList{TDist,TDomain,TThis}"/> class
        /// of a given length and assigns all elements the given value,
        /// except for the specified list of sparse values. This list is stored internally as is
        /// so MUST be sorted by index and must not be modified externally after being passed in.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="commonValue">The common value.</param>
        /// <param name="sortedSparseValues">The sorted list of non-common values.</param>
        /// <returns>The new instance.</returns>
        public static new TThis FromSparseValues(
            int size,
            TDist commonValue,
            List<ValueAtIndex<TDist>> sortedSparseValues)
        {
            var result = Constant(size, commonValue);
            result.SparseValues = sortedSparseValues;
            return result;
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseDistributionList{TDist,TDomain,TThis}"/> class from a sparse list of distributions.
        /// </summary>
        /// <param name="distributions">The sparse list of distributions.</param>
        /// <returns>The new instance.</returns>
        public static TThis FromSparseList(ISparseList<TDist> distributions)
        {
            var result = new TThis();
            result.SetTo(distributions);
            return result;
        }
 
        /// <summary>
        /// Clones this <see cref="SparseDistributionList{TDist,TDomain,TThis}"/>.
        /// </summary>
        /// <returns>The clone returned as an <see cref="object"/>.</returns>
        public new object Clone()
        {
            var result = FromSize(this.Dimension);
            result.SetTo(this);
            return result;
        }
        
        /// <summary>
        /// Returns the maximum difference between the parameters of this sparse distribution list
        /// and another.
        /// </summary>
        /// <param name="thatd">The other sparse distribution list.</param>
        /// <returns>The maximum difference.</returns>
        /// <remarks><c>a.MaxDiff(b) == b.MaxDiff(a)</c>.</remarks>
        public double MaxDiff(object thatd)
        {
            if (!(thatd is IEnumerable<TDist>))
            {
                return double.PositiveInfinity;
            }

            var sparseThat = thatd as ISparseEnumerable<TDist>;
            if (sparseThat != null)
            {
                return this.Reduce<double, TDist>(
                    double.NegativeInfinity, sparseThat, (x, y, z) => Math.Max(x, y.MaxDiff(z)), (x, y, z, n) => Math.Max(x, y.MaxDiff(z)));
            }
            else
            {
                var that = (IEnumerable<TDist>)thatd;
                return this.Reduce<double, TDist>(
                    double.NegativeInfinity, that, (x, y, z) => Math.Max(x, y.MaxDiff(z)));
            }
        }

        /// <summary>
        /// Sets this sparse distribution list to be a uniform distribution.
        /// </summary>
        public void SetToUniform()
        {
            CommonValue.SetToUniform();
            SparseValues.Clear();
        }

        /// <summary>
        /// Asks whether this sparse distribution list is uniform.
        /// </summary>
        /// <returns>True if all elements of the list are uniform, false otherwise.</returns>
        public bool IsUniform()
        {
            return this.All(x => x.IsUniform());
        }

        /// <summary>
        /// Evaluates the log of the density function.
        /// </summary>
        /// <param name="value">The point at which to evaluate the density.</param>
        /// <returns>The log probability of the given value>.</returns>
        public double GetLogProb(ISparseList<TDomain> value)
        {
            Argument.CheckIfValid(value.Count == this.Dimension, string.Format("Point value does not match dimension of this distribution. Expected {0}, got {1}.", this.Dimension, value.Count));
            return this.Reduce<double, TDomain>(0.0, value, (x, y, z) => x + y.GetLogProb(z), (x, y, z, n) => x + n * y.GetLogProb(z));
        }

        /// <summary>
        /// Samples from this sparse distribution list.
        /// </summary>
        /// <returns>A sample from this distribution.</returns>
        /// <remarks>This method is inefficient in that the result will be dense even though the return type is sparse.</remarks>
        public ISparseList<TDomain> Sample()
        {
            ISparseList<TDomain> sv = SparseList<TDomain>.FromSize(Count);
            return this.Sample(sv);
        }

        /// <summary>
        /// Samples from this sparse distribution list.
        /// </summary>
        /// <param name="result">Where to put the result.</param>
        /// <returns>A sample from this distribution.</returns>
        /// <remarks>This method is inefficient in that the result will be dense even though the return type is sparse.</remarks>
        public ISparseList<TDomain> Sample(ISparseList<TDomain> result)
        {
            Argument.CheckIfValid(result.Count == this.Dimension, string.Format("Result list does not match dimension of this distribution. Expected {0}, got {1}.", this.Dimension, result.Count));
            IEnumerator<TDist> e = GetEnumerator();
            int i = 0;
            while (e.MoveNext())
            {
                result[i++] = e.Current.Sample();
            }

            return result;
        }
        
        /// <summary>
        /// Sets this sparse distribution list to another sparse distribution list.
        /// </summary>
        /// <param name="value">The other sparse distribution list.</param>
        public void SetTo(TThis value)
        {
            base.SetTo(value);
        }

        /// <summary>
        /// Sets this sparse distribution list to the product of two other sparse distribution lists.
        /// </summary>
        /// <param name="a">Left hand side.</param>
        /// <param name="b">Right hand side.</param>
        public void SetToProduct(TThis a, TThis b)
        {
            this.SetToFunction(
                a, 
                b, 
                (d1, d2) =>
            {
                var result = default(TDist);
                result.SetToProduct(d1, d2);
                return result;
            });
        }

        /// <summary>
        /// Sets this sparse distribution list to the power of another sparse distribution list.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <param name="exponent">The exponent.</param>
        public void SetToPower(TThis value, double exponent)
        {
            this.SetToFunction(
                value, 
                d =>
            {
                var result = default(TDist);
                result.SetToPower(d, exponent);
                return result;
            });
        }

        /// <summary>
        /// Sets this sparse distribution list to the ratio of two other sparse Gaussian lists.
        /// </summary>
        /// <param name="numerator">The numerator.</param>
        /// <param name="denominator">The denominator.</param>
        /// <param name="forceProper">Whether to force the result to be proper.</param>
        public void SetToRatio(TThis numerator, TThis denominator, bool forceProper = false)
        {
            this.SetToFunction(
                numerator, 
                denominator, 
                (n, d) =>
            {
                var result = default(TDist);
                result.SetToRatio(n, d, forceProper);
                return result;
            });
        }

        /// <summary>
        /// Creates a sparse distribution list to the weighted sums of the elements of two other sparse distribution lists.
        /// </summary>
        /// <param name="weight1">The first weight.</param>
        /// <param name="value1">The first sparse distribution list.</param>
        /// <param name="weight2">The second weight.</param>
        /// <param name="value2">The second sparse distribution list.</param>
        public void SetToSum(double weight1, TThis value1, double weight2, TThis value2)
        {
            this.SetToFunction(
                value1, 
                value2, 
                (d1, d2) =>
                {
                    var result = default(TDist);
                    result.SetToSum(weight1, d1, weight2, d2);
                    return result;
                });
        }

        /// <summary>
        /// Returns the log probability that this distribution and another draw the same sample.
        /// </summary>
        /// <param name="that">The other distribution.</param>
        /// <returns>The log probability that this distribution and another draw the same sample.</returns>
        public double GetLogAverageOf(TThis that)
        {
            return this.Reduce<double, TDist>(0.0, that, (x, y, z) => x + y.GetLogAverageOf(z), (x, y, z, n) => x + n * y.GetLogAverageOf(z));
        }

        /// <summary>
        /// Returns the log of the integral of the product of this sparse distribution list and another sparse distribution list raised to a power.
        /// </summary>
        /// <param name="that">The other distribution.</param>
        /// <param name="exponent">The exponent.</param>
        /// <returns>The log average of the power of the second distribution.</returns>
        public double GetLogAverageOfPower(TThis that, double exponent)
        {
            return this.Reduce<double, TDist>(0.0, that, (x, y, z) => x + y.GetLogAverageOfPower(z, exponent), (x, y, z, n) => x + n * y.GetLogAverageOfPower(z, exponent));
        }

        /// <summary>
        /// Returns the expected logarithm of that sparse distribution list under this sparse distribution list.
        /// </summary>
        /// <param name="that">The other distribution.</param>
        /// <returns>The expected logarithm of the other distribution.</returns>
        public double GetAverageLog(TThis that)
        {
            return this.Reduce<double, TDist>(0.0, that, (x, y, z) => x + y.GetAverageLog(z), (x, y, z, n) => x + n * y.GetAverageLog(z));
        }

        /// <summary>
        /// Gets the mean as a sparse list.
        /// </summary>
        /// <returns>The mean of this distribution.</returns>
        public ISparseList<double> GetMean()
        {
            ISparseList<double> result = SparseList<double>.Constant(Count, CommonValue.GetMean());
            var sen = GetSparseEnumerator();
            while (sen.MoveNext())
            {
                result[sen.CurrentIndex] = sen.Current.GetMean();
            }

            return result;
        }

        /// <summary>
        /// Gets the variance as a sparse list.
        /// </summary>
        /// <returns>The variance of this distribution.</returns>
        public ISparseList<double> GetVariance()
        {
            ISparseList<double> result = SparseList<double>.Constant(Count, CommonValue.GetVariance());
            var sen = GetSparseEnumerator();
            while (sen.MoveNext())
            {
                result[sen.CurrentIndex] = sen.Current.GetVariance();
            }

            return result;
        }
    }
}

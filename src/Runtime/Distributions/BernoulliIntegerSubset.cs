// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System.Collections.Generic;
    using System.Runtime.Serialization;
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Factors.Attributes;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Serialization;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// Represents a sparse list of Bernoulli distributions considered as a distribution over a variable-sized list of
    /// integers, which are the indices of elements in the boolean list with value 'true'.
    /// </summary>
    [Quality(QualityBand.Stable)]
    public class BernoulliIntegerSubset : IDistribution<IList<int>>, Sampleable<IList<int>>,
                                          SettableToProduct<BernoulliIntegerSubset>,
                                          SettableTo<BernoulliIntegerSubset>, SettableToPower<BernoulliIntegerSubset>, SettableToRatio<BernoulliIntegerSubset>,
                                          SettableToWeightedSum<BernoulliIntegerSubset>, CanGetLogAverageOf<BernoulliIntegerSubset>,
                                          CanGetLogAverageOfPower<BernoulliIntegerSubset>, CanGetAverageLog<BernoulliIntegerSubset>
    {
        /// <summary>
        /// Embedded <see cref="SparseBernoulliList"/> instance.
        /// </summary>
        private SparseBernoulliList sparseBernoulliList;

        /// <summary>
        /// Initializes a new instance of the <see cref="BernoulliIntegerSubset"/> class.
        /// </summary>
        public BernoulliIntegerSubset()
        {
            SparseBernoulliList = new SparseBernoulliList();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="BernoulliIntegerSubset"/> class
        /// with the specified number of elements all of which are set to uniform.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        protected BernoulliIntegerSubset(int size)
        {
            SparseBernoulliList = SparseBernoulliList.FromSize(size);
        }
        
        /// <summary>
        /// Initializes a new instance of the <see cref="BernoulliIntegerSubset"/> class
        /// of the given size, and assigns all elements to the specified common value.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="commonValue">The common value.</param>
        protected BernoulliIntegerSubset(int size, Bernoulli commonValue)
        {
            SparseBernoulliList = SparseBernoulliList.Constant(size, commonValue);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="BernoulliIntegerSubset"/> class
        /// of a given length and assigns all elements the given value,
        /// except for the specified list of sparse values. This list is stored internally as is,
        /// so MUST be sorted by index and must not be modified externally after being passed in.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="commonValue">The common value.</param>
        /// <param name="sortedSparseValues">The sorted list of non-common values.</param>
        protected BernoulliIntegerSubset(int size, Bernoulli commonValue, List<ValueAtIndex<Bernoulli>> sortedSparseValues)
        {
            SparseBernoulliList = SparseBernoulliList.FromSparseValues(size, commonValue, sortedSparseValues);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="BernoulliIntegerSubset"/> class from a sparse list of <see cref="Bernoulli"/> distributions.
        /// </summary>
        /// <param name="bernoullis">The sparse list of Bernoulli distributions.</param>
        protected BernoulliIntegerSubset(ISparseList<Bernoulli> bernoullis)
        {
            SparseBernoulliList = SparseBernoulliList.FromSparseList(bernoullis);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="BernoulliIntegerSubset"/> class from an instance of the same type.
        /// </summary>
        /// <param name="that">The instance to copy.</param>
        protected BernoulliIntegerSubset(BernoulliIntegerSubset that)
        {
            SparseBernoulliList = (SparseBernoulliList)that.SparseBernoulliList.Clone();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="BernoulliIntegerSubset"/> class from a <see cref="SparseBernoulliList"/> instance.
        /// </summary>
        /// <param name="sparseBernoulliList"><see cref="SparseBernoulliList"/> instance.</param>
        protected BernoulliIntegerSubset(SparseBernoulliList sparseBernoulliList)
        {
            SparseBernoulliList = (SparseBernoulliList)sparseBernoulliList.Clone();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="BernoulliIntegerSubset"/> class
        /// with the specified number of elements all of which are set to uniform.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        protected BernoulliIntegerSubset(int size, double tolerance)
        {
            SparseBernoulliList = SparseBernoulliList.FromSize(size, tolerance);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="BernoulliIntegerSubset"/> class
        /// of the given size, and assigns all elements to the specified common value.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="commonValue">The common value.</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        protected BernoulliIntegerSubset(int size, Bernoulli commonValue, double tolerance)
        {
            SparseBernoulliList = SparseBernoulliList.Constant(size, commonValue, tolerance);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="BernoulliIntegerSubset"/> class
        /// of a given length and assigns all elements the given value,
        /// except for the specified list of sparse values. This list is stored internally as is,
        /// so MUST be sorted by index and must not be modified externally after being passed in.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="commonValue">The common value.</param>
        /// <param name="sortedSparseValues">The sorted list of non-common values.</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        protected BernoulliIntegerSubset(int size, Bernoulli commonValue, List<ValueAtIndex<Bernoulli>> sortedSparseValues, double tolerance)
        {
            SparseBernoulliList = SparseBernoulliList.FromSparseValues(size, commonValue, sortedSparseValues, tolerance);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="BernoulliIntegerSubset"/> class from a sparse list of Bernoulli distributions.
        /// </summary>
        /// <param name="bernoullis">The sparse list of Bernoulli distributions.</param>
        /// <param name="tolerance">The tolerance.</param>
        protected BernoulliIntegerSubset(ISparseList<Bernoulli> bernoullis, double tolerance)
        {
            SparseBernoulliList = SparseBernoulliList.FromSparseList(bernoullis, tolerance);
        }

        /// <summary>
        /// Gets or sets the embedded <see cref="SparseBernoulliList"/> instance which maintains the parameters of this distribution.
        /// </summary>
        public SparseBernoulliList SparseBernoulliList
        {
            get
            {
                return this.sparseBernoulliList;
            }

            set
            {
                Argument.CheckIfNotNull(value, "Sparse Bernoulli list cannot be null reference");
                this.sparseBernoulliList = value;
            }
        }

        /// <summary>
        /// Gets or sets the distribution as a point value integer array. This is a sparse method - the integers specify
        /// the indices where elements are true - all other elements are false.
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public IList<int> Point
        {
            get
            {
                var trueIndices = this.SparseBernoulliList.IndexOfAll(b => b.Point);
                return new List<int>(trueIndices);
            }

            set
            {
                this.SparseBernoulliList.SetAllElementsTo(Bernoulli.PointMass(false));
                for (int i = 0; i < value.Count; i++)
                {
                    this.SparseBernoulliList[value[i]] = Bernoulli.PointMass(true);
                }
            }
        }

        /// <summary>
        /// Gets a value indicating whether this is a point mass or not.
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public bool IsPointMass
        {
            get
            {
                return this.SparseBernoulliList.IsPointMass;
            }
        }

        /// <summary>
        /// Gets a value representing the size of the discrete domain.
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public int Dimension
        {
            get
            {
                return this.SparseBernoulliList.Dimension;
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="BernoulliIntegerSubset"/> class
        /// with the specified number of elements all of which are set to uniform.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        /// <returns>The new <see cref="BernoulliIntegerSubset"/> instance.</returns>
        public static BernoulliIntegerSubset FromSize(int size, double tolerance)
        {
            var result = new BernoulliIntegerSubset();
            result.SparseBernoulliList = SparseBernoulliList.FromSize(size, tolerance);
            return result;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="BernoulliIntegerSubset"/> class
        /// with the specified number of elements all of which are set to the specified <see cref="Bernoulli"/> instance.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="commonValue">The common value.</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        /// <returns>The new <see cref="BernoulliIntegerSubset"/> instance.</returns>
        public static BernoulliIntegerSubset FromBernoulli(int size, Bernoulli commonValue, double tolerance)
        {
            var result = new BernoulliIntegerSubset();
            result.SparseBernoulliList = SparseBernoulliList.Constant(size, commonValue, tolerance);
            return result;
        }

        /// <summary>
        /// Returns a new instance of the <see cref="BernoulliIntegerSubset"/> class of a given length and assigns all elements the given value,
        /// except for the specified list of sparse values. This list is stored internally as is
        /// so MUST be sorted by index and must not be modified externally after being passed in.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="commonValue">The common value.</param>
        /// <param name="sortedSparseValues">The sorted list of non-common values.</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        /// <returns>The new <see cref="BernoulliIntegerSubset"/> instance.</returns>
        public static BernoulliIntegerSubset FromSparseValues(
            int size,
            Bernoulli commonValue,
            List<ValueAtIndex<Bernoulli>> sortedSparseValues,
            double tolerance)
        {
            var result = new BernoulliIntegerSubset();
            result.SparseBernoulliList = SparseBernoulliList.FromSparseValues(size, commonValue, sortedSparseValues, tolerance);
            return result;
        }

        [Construction("SparseBernoulliList")]
        public static BernoulliIntegerSubset FromSparseBernoulliList(SparseBernoulliList list)
        {
            var result = new BernoulliIntegerSubset();
            result.SparseBernoulliList = list;
            return result;
        }

        public override string ToString()
        {
            return string.Format("BernoulliIntegerSubset({0})", this.SparseBernoulliList.ToString());
        }

        /// <summary>
        /// Returns a new instance of the <see cref="BernoulliIntegerSubset"/> class from a sparse list of Bernoulli distributions.
        /// </summary>
        /// <param name="bernoullis">The sparse list of Bernoulli distributions.</param>
        /// <param name="tolerance">The tolerance.</param>
        /// <returns>The new <see cref="BernoulliIntegerSubset"/> instance.</returns>
        public static BernoulliIntegerSubset FromSparseList(ISparseList<Bernoulli> bernoullis, double tolerance)
        {
            var result = new BernoulliIntegerSubset();
            result.SparseBernoulliList = SparseBernoulliList.FromSparseList(bernoullis, tolerance);
            return result;
        }

        /// <summary>
        /// Returns a new instance of the <see cref="BernoulliIntegerSubset"/> class of a given size
        /// with each element having a given probability of true.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="probTrue">The desired probability of true.</param>
        /// <returns>The new <see cref="BernoulliIntegerSubset"/> instance.</returns>
        public static BernoulliIntegerSubset FromProbTrue(
            int size, double probTrue)
        {
            return BernoulliIntegerSubset.FromProbTrue(size, probTrue, SparseBernoulliList.DefaultTolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="BernoulliIntegerSubset"/> class of a given size
        /// with each element having a given log odds.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="logOdds">The desired log odds.</param>
        /// <returns>The new <see cref="BernoulliIntegerSubset"/> instance.</returns>
        public static BernoulliIntegerSubset FromLogOdds(
            int size, double logOdds)
        {
            return BernoulliIntegerSubset.FromLogOdds(size, logOdds, SparseBernoulliList.DefaultTolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="BernoulliIntegerSubset"/> class of a given size
        /// with each element having a given probability of true.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="probTrue">The desired probability of true.</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        /// <returns>The new <see cref="BernoulliIntegerSubset"/> instance.</returns>
        public static BernoulliIntegerSubset FromProbTrue(
            int size, double probTrue, double tolerance)
        {
            var result = new BernoulliIntegerSubset();
            result.SparseBernoulliList = SparseBernoulliList.FromProbTrue(size, probTrue, tolerance);
            return result;
        }

        /// <summary>
        /// Returns a new instance of the <see cref="BernoulliIntegerSubset"/> class of a given size
        /// with each element having a given log odds.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="logOdds">The desired log odds.</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        /// <returns>The new <see cref="BernoulliIntegerSubset"/> instance.</returns>
        public static BernoulliIntegerSubset FromLogOdds(
            int size, double logOdds, double tolerance)
        {
            var result = new BernoulliIntegerSubset();
            result.SparseBernoulliList = SparseBernoulliList.FromLogOdds(size, logOdds, tolerance);
            return result;
        }

        /// <summary>
        /// Returns a new instance of the <see cref="BernoulliIntegerSubset"/> class of a given size
        /// from a sparse list of probability true.
        /// </summary>
        /// <param name="probTrue">The sparse list of probability of true.</param>
        /// <returns>The new <see cref="BernoulliIntegerSubset"/> instance.</returns>
        public static BernoulliIntegerSubset FromProbTrue(
            ISparseList<double> probTrue)
        {
            return BernoulliIntegerSubset.FromProbTrue(probTrue, SparseBernoulliList.DefaultTolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="BernoulliIntegerSubset"/> class of a given size
        /// from a sparse list of log odds
        /// </summary>
        /// <param name="logOdds">The sparse list of log odds.</param>
        /// <returns>The new <see cref="BernoulliIntegerSubset"/> instance.</returns>
        public static BernoulliIntegerSubset FromLogOdds(
            ISparseList<double> logOdds)
        {
            return BernoulliIntegerSubset.FromLogOdds(logOdds, SparseBernoulliList.DefaultTolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="BernoulliIntegerSubset"/> class of a given size
        /// from a sparse list of probability true.
        /// </summary>
        /// <param name="probTrue">The sparse list of probability of true.</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        /// <returns>The new <see cref="BernoulliIntegerSubset"/> instance.</returns>
        public static BernoulliIntegerSubset FromProbTrue(
            ISparseList<double> probTrue, double tolerance)
        {
            BernoulliIntegerSubset result = new BernoulliIntegerSubset();
            result.SparseBernoulliList = SparseBernoulliList.FromProbTrue(probTrue, tolerance);
            return result;
        }

        /// <summary>
        /// Returns a new instance of the <see cref="BernoulliIntegerSubset"/> class of a given size
        /// from a sparse list of log odds.
        /// </summary>
        /// <param name="logOdds">The sparse list of log odds.</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        /// <returns>The new <see cref="BernoulliIntegerSubset"/> instance.</returns>
        public static BernoulliIntegerSubset FromLogOdds(
            ISparseList<double> logOdds, double tolerance)
        {
            BernoulliIntegerSubset result = new BernoulliIntegerSubset();
            result.SparseBernoulliList = SparseBernoulliList.FromLogOdds(logOdds, tolerance);
            return result;
        }

        /// <summary>
        /// Returns a new instance of the <see cref="BernoulliIntegerSubset"/> class
        /// with the specified number of elements all of which are set to uniform.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <returns>The new <see cref="BernoulliIntegerSubset"/> instance.</returns>
        public static BernoulliIntegerSubset FromSize(int size)
        {
            return BernoulliIntegerSubset.FromSize(size, SparseBernoulliList.DefaultTolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="BernoulliIntegerSubset"/> class with the specified number of elements
        /// all of which are set to the specified <see cref="Bernoulli"/> instance.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="commonValue">The common value.</param>
        /// <returns>The new <see cref="BernoulliIntegerSubset"/> instance.</returns>
        public static BernoulliIntegerSubset FromBernoulli(int size, Bernoulli commonValue)
        {
            return BernoulliIntegerSubset.FromBernoulli(size, commonValue, SparseBernoulliList.DefaultTolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="BernoulliIntegerSubset"/> class of a given length
        /// and assigns all elements the given value,
        /// except for the specified list of sparse values. This list is stored internally as is,
        /// so MUST be sorted by index and must not be modified externally after being passed in.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="commonValue">The common value.</param>
        /// <param name="sortedSparseValues">The sorted list of non-common values.</param>
        /// <returns>The new <see cref="BernoulliIntegerSubset"/> instance.</returns>
        public static BernoulliIntegerSubset FromSparseValues(
            int size,
            Bernoulli commonValue,
            List<ValueAtIndex<Bernoulli>> sortedSparseValues)
        {
            return BernoulliIntegerSubset.FromSparseValues(size, commonValue, sortedSparseValues, SparseBernoulliList.DefaultTolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="BernoulliIntegerSubset"/> class from a sparse list of Bernoulli distributions.
        /// </summary>
        /// <param name="bernoullis">The sparse list of Bernoulli distributions.</param>
        /// <returns>The new <see cref="BernoulliIntegerSubset"/> instance.</returns>
        public static BernoulliIntegerSubset FromSparseList(ISparseList<Bernoulli> bernoullis)
        {
            return BernoulliIntegerSubset.FromSparseList(bernoullis, SparseBernoulliList.DefaultTolerance);
        }

        /// <summary>
        /// Samples from a list of Bernoulli distributions with the specified vector of P(true) values.
        /// </summary>
        /// <param name="probTrue">the vector of P(true) values.</param>
        /// <returns>The sample.</returns>
        [Stochastic]
        public static IList<int> Sample(ISparseList<double> probTrue)
        {
            var result = new bool[probTrue.Count];
            var list = new List<int>();
            int i = 0;
            foreach (double d in probTrue)
            {
                if (Bernoulli.Sample(d))
                {
                    list.Add(i);
                }

                i++;
            }

            return list;
        }

        /// <summary>
        /// Converts index subset to a sparse list of boolean values.
        /// </summary>
        /// <param name="indices">The indices.</param>
        /// <param name="length">The length of the boolean list.</param>
        /// <returns>The equivalent sparse list of boolean values.</returns>
        public static ISparseList<bool> SubsetToList(IList<int> indices, int length)
        {
            var result = SparseList<bool>.Constant(length, false);
            foreach (int index in indices)
            {
                result[index] = true;
            }

            return result;
        }
        
        /// <summary>
        /// Gets the log probability of the given value under this distribution.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The log probability of the given value under this distribution.</returns>
        public double GetLogProb(IList<int> value)
        {
            var boolValue = SparseList<bool>.Constant(this.SparseBernoulliList.Count, false);
            foreach (int index in value)
            {
                boolValue[index] = true;
            }

            return this.SparseBernoulliList.GetLogProb(boolValue);
        }

        /// <summary>
        /// Clones this object.
        /// </summary>
        /// <returns>The clone.</returns>
        public object Clone()
        {
            return new BernoulliIntegerSubset(this);
        }

        /// <summary>
        /// Samples a list of integers from this distribution.
        /// </summary>
        /// <returns>The sample.</returns>
        public IList<int> Sample()
        {
            return Sample(new List<int>());
        }

        /// <summary>
        /// Samples a list of integers from this distribution.
        /// </summary>
        /// <param name="result">Where to put the resulting sample.</param>
        /// <returns>The sample.</returns>
        public IList<int> Sample(IList<int> result)
        {
            result.Clear();
            int i = 0;
            foreach (Bernoulli b in this.SparseBernoulliList)
            {
                if (b.Sample())
                {
                    result.Add(i);
                }

                i++;
            }

            return result;
        }

        /// <summary>
        /// Sets this BernoulliIntegerSubset distribution to a product of two other such distributions.
        /// </summary>
        /// <param name="a">The left hand side.</param>
        /// <param name="b">The right hand side.</param>
        public void SetToProduct(BernoulliIntegerSubset a, BernoulliIntegerSubset b)
        {
            this.SparseBernoulliList.SetToProduct(a.SparseBernoulliList, b.SparseBernoulliList);
        }

        /// <summary>
        /// Sets this BernoulliIntegerSubset distribution from an enumerable collection of Bernoulli distribution.
        /// </summary>
        /// <param name="value">The (usually sparse) enumerable collection of Bernoulli distributions.</param>
        public void SetTo(IEnumerable<Bernoulli> value)
        {
            this.SparseBernoulliList.SetTo(value);
        }

        /// <summary>
        /// Sets this BernoulliIntegerSubset distribution to another such distribution.
        /// </summary>
        /// <param name="value">The distribution to set.</param>
        public void SetTo(BernoulliIntegerSubset value)
        {
            this.SparseBernoulliList.SetTo(value.SparseBernoulliList);
        }

        /// <summary>
        /// Sets this BernoulliIntegerSubset distribution to the power another such distribution.
        /// </summary>
        /// <param name="value">The distribution to be raised to a power.</param>
        /// <param name="exponent">The exponent.</param>
        public void SetToPower(BernoulliIntegerSubset value, double exponent)
        {
            this.SparseBernoulliList.SetToPower(value.SparseBernoulliList, exponent);
        }

        /// <summary>
        /// Sets this BernoulliIntegerSubset distribution to the ratio of two other such distributions.
        /// </summary>
        /// <param name="numerator">The numerator.</param>
        /// <param name="denominator">The denominator.</param>
        /// <param name="forceProper">Whether to force the result to be proper.</param>
        public void SetToRatio(BernoulliIntegerSubset numerator, BernoulliIntegerSubset denominator, bool forceProper = false)
        {
            this.SparseBernoulliList.SetToRatio(numerator.SparseBernoulliList, denominator.SparseBernoulliList, forceProper);
        }

        /// <summary>
        /// Sets this BernoulliIntegerSubset distribution to the weighted sum of two other such distributions.
        /// </summary>
        /// <param name="weight1">The first weight.</param>
        /// <param name="value1">The first distribution.</param>
        /// <param name="weight2">The second weight.</param>
        /// <param name="value2">The second distribution.</param>
        /// <remarks>Not yet implemented</remarks>
        public void SetToSum(double weight1, BernoulliIntegerSubset value1, double weight2, BernoulliIntegerSubset value2)
        {
            this.SparseBernoulliList.SetToSum(weight1, value1.SparseBernoulliList, weight2, value2.SparseBernoulliList);
        }

        /// <summary>
        /// Gets the log of the integral of the product of this BernoulliIntegerSubset distribution and another such distribution.
        /// </summary>
        /// <param name="that">The other distribution.</param>
        /// <returns>The log average of the other distribution under this distribution.</returns>
        public double GetLogAverageOf(BernoulliIntegerSubset that)
        {
            return this.SparseBernoulliList.GetLogAverageOf(that.SparseBernoulliList);
        }

        /// <summary>
        /// Returns the log of the integral of the product of BernoulliIntegerSubset distribution and another
        /// BernoulliIntegerSubset distribution raised to a power.
        /// </summary>
        /// <param name="that">The other distribution.</param>
        /// <param name="power">The exponent.</param>
        /// <returns>The average of the power of the other distribution under this distribution.</returns>
        public double GetLogAverageOfPower(BernoulliIntegerSubset that, double power)
        {
            return this.SparseBernoulliList.GetLogAverageOfPower(that.SparseBernoulliList, power);
        }

        /// <summary>
        /// The expected logarithm of that distribution under this distribution.
        /// </summary>
        /// <param name="that">The distribution to take the logarithm of.</param>
        /// <returns><c>sum_x this.Evaluate(x)*Math.Log(that.Evaluate(x))</c>.</returns>
        /// <remarks>This is also known as the cross entropy.</remarks>
        public double GetAverageLog(BernoulliIntegerSubset that)
        {
            return this.SparseBernoulliList.GetAverageLog(that.SparseBernoulliList);
        }

        /// <summary>
        /// The max difference between this distribution and another distribution.
        /// </summary>
        /// <param name="thatd">The other distribution.</param>
        /// <returns>The maximum difference between this distribution and the other.</returns>
        public double MaxDiff(object thatd)
        {
            if (!(thatd is BernoulliIntegerSubset))
            {
                return double.PositiveInfinity;
            }

            var that = (BernoulliIntegerSubset)thatd;
            return this.SparseBernoulliList.MaxDiff(that.SparseBernoulliList);
        }

        /// <summary>
        /// Sets this sparse Bernoulli list to uniform.
        /// </summary>
        public void SetToUniform()
        {
            this.SparseBernoulliList.SetToUniform();
        }

        /// <summary>
        /// Gets a value indicating whether this distribution is uniform or not.
        /// </summary>
        /// <returns>True if uniform, false otherwise.</returns>
        public bool IsUniform()
        {
            return this.SparseBernoulliList.IsUniform();
        }

        /// <summary>
        /// Creates a <see cref="BernoulliIntegerSubset"/> distribution which is the product of two others.
        /// </summary>
        /// <param name="a">The first distribution.</param>
        /// <param name="b">The second distribution.</param>
        /// <returns>The resulting <see cref="BernoulliIntegerSubset"/> distribution.</returns>
        public static BernoulliIntegerSubset operator *(BernoulliIntegerSubset a, BernoulliIntegerSubset b)
        {
            var result = BernoulliIntegerSubset.FromSize(a.Dimension);
            result.SetToProduct(a, b);
            return result;
        }

        /// <summary>
        /// Creates a <see cref="BernoulliIntegerSubset"/> distribution which is the ratio of two others.
        /// </summary>
        /// <param name="numerator">The numerator.</param>
        /// <param name="denominator">The denominator.</param>
        /// <returns>The resulting <see cref="BernoulliIntegerSubset"/> distribution.</returns>
        public static BernoulliIntegerSubset operator /(BernoulliIntegerSubset numerator, BernoulliIntegerSubset denominator)
        {
            var result = BernoulliIntegerSubset.FromSize(numerator.Dimension);
            result.SetToRatio(numerator, denominator);
            return result;
        }

        /// <summary>
        /// Creates a <see cref="BernoulliIntegerSubset"/> distribution which is the power of another.
        /// </summary>
        /// <param name="dist">The other distribution.</param>
        /// <param name="exponent">The exponent.</param>
        /// <returns>The resulting <see cref="BernoulliIntegerSubset"/> distribution.</returns>
        public static BernoulliIntegerSubset operator ^(BernoulliIntegerSubset dist, double exponent)
        {
            var result = BernoulliIntegerSubset.FromSize(dist.Dimension);
            result.SetToPower(dist, exponent);
            return result;
        }
    }
}
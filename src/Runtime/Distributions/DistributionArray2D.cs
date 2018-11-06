// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#define SpecializeInterfaces

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Linq;
    using System.Runtime.Serialization;
    using Utilities;
    using Collections;
    using Math;
    using Factors.Attributes;
    using Microsoft.ML.Probabilistic.Serialization;

    #region DistributionArray2D<T>

    /// <summary>
    /// A distribution over a 2D array, where each element is independent and has distribution type <typeparamref name="T"/>
    /// </summary>
    /// <typeparam name="T">The distribution type.</typeparam>
    [Serializable]
    [DataContract]
    public abstract class DistributionArray2D<T> : Array2D<T>,
                                                   SettableToUniform,
                                                   Diffable
#if !SpecializeInterfaces
        ,SettableTo<DistributionArray2D<T>>,
        SettableToProduct<DistributionArray2D<T>>,
        SettableToRatio<DistributionArray2D<T>>,
        SettableToPower<DistributionArray2D<T>>,
        SettableToWeightedSum<DistributionArray2D<T>>,
        CanGetLogAverageOf<DistributionArray2D<T>>,
        CanGetAverageLog<DistributionArray2D<T>>
#endif
        where T : SettableToUniform,
            Diffable
    {
        /// <summary>
        /// Parameterless constructor needed for serialization
        /// </summary>
        protected DistributionArray2D()
        {
        }

        /// <summary>
        /// Constructs a new 2-D distribution array given the lengths of each dimension
        /// </summary>
        /// <param name="length0"></param>
        /// <param name="length1"></param>
        /// <returns></returns>
        public DistributionArray2D(int length0, int length1)
            : this(length0, length1, new T[length0*length1])
        {
        }

        /// <summary>
        /// Constructs a new 2-D distribution array given the lengths of each dimension,
        /// and an array of values to reference
        /// </summary>
        /// <param name="length0"></param>
        /// <param name="length1"></param>
        /// <param name="valuesRef">Array of values to reference</param>
        /// <returns></returns>
        public DistributionArray2D(int length0, int length1, T[] valuesRef)
            : base(length0, length1, valuesRef)
        {
        }

        /// <summary>
        /// Constructs a new 2-D distribution array given a 2-D array of distributions
        /// </summary>
        /// <param name="array"></param>
        /// <returns></returns>
        public DistributionArray2D(T[,] array)
            : base(array)
        {
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that"></param>
        public DistributionArray2D(Array2D<T> that)
            : base(that)
        {
        }

#if false
    /// <summary>
    /// Clones the array but not the items in the array.
    /// </summary>
    /// <returns></returns>
        public override object Clone()
        {
            return new DistributionArray2D<T>(this);
        }
#endif

        /// <summary>
        /// Set the distribution to uniform
        /// </summary>
        public void SetToUniform()
        {
            SetToFunction(this, delegate(T item)
                {
                    item.SetToUniform();
                    return item;
                });
        }

        /// <summary>
        /// True if the distribution is uniform
        /// </summary>
        /// <returns>True if all uniform, otherwise false</returns>
        public bool IsUniform()
        {
            return this.All(item => item.IsUniform());
        }

        /// <summary>
        /// The maximum difference the parameters of this distribution and that distribution
        /// </summary>
        /// <param name="that">That distribution array</param>
        /// <returns>The maximum difference</returns>
        public virtual double MaxDiff(object that)
        {
            DistributionArray2D<T> thatd = that as DistributionArray2D<T>;
            if ((object) thatd == null) return Double.PositiveInfinity;
            return Distribution.MaxDiff(this.array, thatd.array);
        }

#if !SpecializeInterfaces
        public void SetTo(DistributionArray2D<T> that)
        {
            base.SetTo(that);
        }

        /// <summary>
        /// Sets the current instance to an array of distributions each element
        /// of which is a product of the corresponding distributions in two given
        /// distribution arrays
        /// </summary>
        /// <param name="a">The first distribution array</param>
        /// <param name="b">The second distribution array</param>
        public void SetToProduct(DistributionArray2D<T> a, DistributionArray2D<T> b)
        {
            Distribution.SetToProduct(array, a.array, b.array);
        }

        /// <summary>
        /// Sets the current instance to an array of distributions each element
        /// of which is a ratio of the corrresponding distributions in two given
        /// distribution arrays
        /// </summary>
        /// <param name="numerator">The numerator distribution array</param>
        /// <param name="denominator">The denominator distribution array</param>
        public void SetToRatio(DistributionArray2D<T> numerator, DistributionArray2D<T> denominator)
        {
            Distribution.SetToRatio(array, numerator.array, denominator.array);
        }

        /// <summary>
        /// Sets the current instance to an array of distributions each element
        /// of which is a power of the corresponding element in a source distribution array
        /// </summary>
        /// <param name="a">The source distribution array</param>
        /// <param name="exponent">The exponent</param>
        public void SetToPower(DistributionArray2D<T> a, double exponent)
        {
            Distribution.SetToPower(array, a.array, exponent);
        }

        /// <summary>
        /// Sets the current instance to an array of distributions each element
        /// of which is a weighted sum of the corresponding distributions in two given
        /// distribution arrays
        /// </summary>
        /// <param name="weight1">The first weight</param>
        /// <param name="a">The first distribution array</param>
        /// <param name="weight2">The second weight</param>
        /// <param name="b">The second distribution array</param>
        public void SetToSum(double weight1, DistributionArray2D<T> a, double weight2, DistributionArray2D<T> b)
        {
            Distribution.SetToSum(array, weight1, a.array, weight2, b.array);
        }

        /// <summary>
        /// The log-probability that two distributions would draw the same sample.
        /// </summary>
        /// <param name="that"></param>
        /// <returns><c>Math.Log(sum_x this.Evaluate(x)*that.Evaluate(x))</c></returns>
        /// <remarks>This can be considered a type of inner product between distributions.
        /// Another name might be "LogAverage" to go with "GetAverageLog".
        /// For a DistributionArray, this specializes to:
        /// <c>sum_i Math.Log(sum_x this[i].Evaluate(x)*that[i].Evaluate(x))</c>
        /// = <c>sum_i this[i].GetLogAverageOf(that[i])</c>
        /// </remarks>
        public double GetLogAverageOf(DistributionArray2D<T> that)
        {
            return Distribution.GetLogAverageOf(array, that.array);
        }

        /// <summary>
        /// The expected logarithm of that distribution under this distribution.
        /// </summary>
        /// <param name="that">The distribution to take the logarithm of.</param>
        /// <returns><c>sum_x this.Evaluate(x)*Math.Log(that.Evaluate(x))</c></returns>
        /// <remarks>This is also known as the cross entropy.
        /// For a DistributionArray, this specializes to:
        /// <c>sum_i sum_x this[i].Evaluate(x)*Math.Log(that[i].Evaluate(x))</c>
        /// = <c>sum_i this[i].GetAverageLog(that[i])</c>
        /// </remarks>
        public double GetAverageLog(DistributionArray2D<T> that)
        {
            return Distribution.GetAverageLog(array, that.array);
        }
#endif
    }

    #endregion

    #region DistributionArray2D<T,DomainType>

    /// <summary>
    /// A distribution over an array of type <typeparamref name="DomainType"/>, where each element is independent and has distribution of type <typeparamref name="T"/>
    /// </summary>
    /// <typeparam name="T">Distribution type of an element.  Must be a value type all the way down, i.e. must not hold any references.</typeparam>
    /// <typeparam name="DomainType">Domain type of an element</typeparam>
    /// <remarks>
    /// This is an extension of DistributionArray that implements IDistribution and Sampleable.
    /// </remarks>
    [Serializable]
    [Quality(QualityBand.Mature)]
    [DataContract]
    public abstract class DistributionArray2D<T, DomainType> : DistributionArray2D<T>, IDistribution<DomainType[,]>, Sampleable<DomainType[,]>
        where T : SettableToUniform,
            Diffable,
            Sampleable<DomainType>,
            IDistribution<DomainType>
    {
        /// <summary>
        /// Parameterless constructor needed for serialization
        /// </summary>
        protected DistributionArray2D()
        {
        }

        /// <summary>
        /// Constructs a new 2-D distribution array given the lengths of each dimension
        /// </summary>
        /// <param name="length0"></param>
        /// <param name="length1"></param>
        /// <returns></returns>
        public DistributionArray2D(int length0, int length1)
            : this(length0, length1, new T[length0*length1])
        {
        }

        /// <summary>
        /// Constructs a new 2-D distribution array given the lengths of each dimension,
        /// and an array of values to reference
        /// </summary>
        /// <param name="length0"></param>
        /// <param name="length1"></param>
        /// <param name="valuesRef">Array of values to reference</param>
        /// <returns></returns>
        public DistributionArray2D(int length0, int length1, T[] valuesRef)
            : base(length0, length1, valuesRef)
        {
        }

        /// <summary>
        /// Constructs a new 2-D distribution array given a 2-D array of distributions
        /// </summary>
        /// <param name="array"></param>
        /// <returns></returns>
        public DistributionArray2D(T[,] array)
            : base(array)
        {
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that"></param>
        public DistributionArray2D(Array2D<T> that)
            : base(that)
        {
        }

        /// <summary>
        /// Sets/gets the 2-D distribution array as a point mass
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public DomainType[,] Point
        {
            get
            {
                DomainType[,] result = new DomainType[Length0,Length1];
                for (int i = 0; i < Length0; i++)
                {
                    for (int j = 0; j < Length1; j++)
                    {
                        result[i, j] = this[i, j].Point;
                    }
                }
                return result;
            }
            set
            {
                if (value.GetLength(0) != Length0) throw new ArgumentException("value.GetLength(0) (" + value.GetLength(0) + ") != this.Length0 (" + this.Length0 + ")");
                if (value.GetLength(1) != Length1) throw new ArgumentException("value.GetLength(1) (" + value.GetLength(1) + ") != this.Length1 (" + this.Length1 + ")");
                for (int i = 0; i < Length0; i++)
                {
                    for (int j = 0; j < Length1; j++)
                    {
                        T item = this[i, j];
                        item.Point = value[i, j];
                        this[i, j] = item;
                    }
                }
            }
        }

        /// <summary>
        /// True if the distribution is a point mass
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public bool IsPointMass
        {
            get
            {
                for (int i = 0; i < array.Length; i++)
                {
                    if (!array[i].IsPointMass) return false;
                }
                return true;
            }
        }

        /// <summary>
        /// Gets the log probability density at a point
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public double GetLogProb(DomainType[,] value)
        {
            double sum = 0.0;
            ForEach(value, delegate(T distribution, DomainType item) { sum += distribution.GetLogProb(item); });
            return sum;
        }

        #region Sampleable<DomainType[,]> Members

        /// <summary>
        /// Sample from the distribution
        /// </summary>
        /// <returns>A sample</returns>
        public DomainType[,] Sample()
        {
            DomainType[,] result = new DomainType[Length0,Length1];
            return this.Sample(result);
        }

        /// <summary>
        /// Sample from the distribution
        /// </summary>
        /// <param name="result">Where to put the results</param>
        /// <returns>A sample</returns>
        public abstract DomainType[,] Sample(DomainType[,] result);

        #endregion
    }

    #endregion

    #region DistributionStructArray2D

    /// <summary>
    /// A distribution over a 2D array of type <typeparamref name="DomainType"/>, where each element is independent and has distribution of type <typeparamref name="T"/>
    /// </summary>
    /// <typeparam name="T">Distribution type of an element.  Must be a value type all the way down, i.e. must not hold any references.</typeparam>
    /// <typeparam name="DomainType">Domain type of an element</typeparam>
    /// <remarks>
    /// This is an extension of DistributionArray that requires T to be a value type.
    /// This class only exists for efficiency, to avoid unnecessary cloning.
    /// </remarks>
    [Serializable]
    [Quality(QualityBand.Mature)]
    [DataContract]
    public class DistributionStructArray2D<T, DomainType> : DistributionArray2D<T, DomainType>
#if SpecializeInterfaces
                                                            , SettableTo<DistributionStructArray2D<T, DomainType>>,
                                                            SettableToProduct<DistributionStructArray2D<T, DomainType>>,
                                                            SettableToRatio<DistributionStructArray2D<T, DomainType>>,
                                                            SettableToPower<DistributionStructArray2D<T, DomainType>>,
                                                            SettableToWeightedSum<DistributionStructArray2D<T, DomainType>>,
                                                            CanGetLogAverageOf<DistributionStructArray2D<T, DomainType>>,
                                                            CanGetLogAverageOfPower<DistributionStructArray2D<T, DomainType>>,
                                                            CanGetAverageLog<DistributionStructArray2D<T, DomainType>>
#endif
        where T : struct,
            SettableToUniform,
            Diffable,
            SettableToProduct<T>,
            SettableToRatio<T>,
            SettableToPower<T>,
            SettableToWeightedSum<T>,
            CanGetLogAverageOf<T>,
            CanGetLogAverageOfPower<T>,
            CanGetAverageLog<T>,
            IDistribution<DomainType>,
            Sampleable<DomainType>
    {
        /// <summary>
        /// Parameterless constructor needed for serialization
        /// </summary>
        protected DistributionStructArray2D()
        {
        }

        /// <summary>
        /// Constructs a new 2-D distribution array given the lengths of each dimension
        /// </summary>
        /// <param name="length0"></param>
        /// <param name="length1"></param>
        /// <returns></returns>
        public DistributionStructArray2D(int length0, int length1)
            : this(length0, length1, new T[length0*length1])
        {
        }

        /// <summary>
        /// Constructs a new 2-D distribution array given the lengths of each dimension and initial values.
        /// </summary>
        /// <param name="length0"></param>
        /// <param name="length1"></param>
        /// <param name="init">Function that maps an index to a value.</param>
        /// <returns></returns>
        public DistributionStructArray2D(int length0, int length1, Func<int, int, T> init)
            : this(length0, length1, new T[length0*length1])
        {
            for (int i = 0; i < length0; i++)
            {
                for (int j = 0; j < length1; j++)
                {
                    this[i, j] = init(i, j);
                }
            }
        }

        /// <summary>
        /// Constructs a new 2-D distribution array given the lengths of each dimension,
        /// and an array of values to reference
        /// </summary>
        /// <param name="length0"></param>
        /// <param name="length1"></param>
        /// <param name="valuesRef">Array of values to reference</param>
        /// <returns></returns>
        public DistributionStructArray2D(int length0, int length1, T[] valuesRef)
            : base(length0, length1, valuesRef)
        {
        }

        /// <summary>
        /// Constructs a new 2-D distribution array given the lengths of each dimension, and a value
        /// </summary>
        /// <param name="value"></param>
        /// <param name="length0"></param>
        /// <param name="length1"></param>
        /// <returns></returns>
        public DistributionStructArray2D(T value, int length0, int length1)
            : this(length0, length1)
        {
            SetAllElementsTo(value);
        }

        /// <summary>
        /// Constructs a new 2-D distribution array given a 2-D array of distributions
        /// </summary>
        /// <param name="array"></param>
        /// <returns></returns>
        public DistributionStructArray2D(T[,] array)
            : base(array)
        {
        }

        /// <summary>
        /// Copy constructor
        /// </summary>
        /// <param name="that"></param>
        public DistributionStructArray2D(Array2D<T> that)
            : base(that)
        {
        }

        /// <summary>
        /// Clone the distribution
        /// </summary>
        /// <returns></returns>
        public override object Clone()
        {
            return new DistributionStructArray2D<T, DomainType>(this);
        }

        /// <summary>
        /// Creates a distribution containing a single point.
        /// </summary>
        /// <param name="value">The point</param>
        /// <returns></returns>
        public static DistributionStructArray2D<T,DomainType> PointMass(DomainType[,] value)
        {
            int length0 = value.GetLength(0);
            int length1 = value.GetLength(1);
            return new DistributionStructArray2D<T, DomainType>(length0, length1, (i, j) =>
            {
                T dist = new T();
                dist.Point = value[i, j];
                return dist;
            });
        }

        /// <summary>
        /// The maximum difference in parameters between
        /// this distribution array and that distribution array
        /// </summary>
        /// <param name="that">That distribution array</param>
        /// <returns>The maximum difference</returns>
        public override double MaxDiff(object that)
        {
            DistributionStructArray2D<T, DomainType> thatd = that as DistributionStructArray2D<T, DomainType>;
            if ((object) thatd == null) return Double.PositiveInfinity;
            return Distribution.MaxDiff(this.array, thatd.array);
        }

        /// <summary>
        /// Sample from the distribution 
        /// </summary>
        /// <param name="result">Where to put the result</param>
        /// <returns>A sample</returns>
        public override DomainType[,] Sample(DomainType[,] result)
        {
            if (result.GetLength(0) != Length0) throw new ArgumentException("result.GetLength(0) (" + result.GetLength(0) + ") != this.Length0 (" + this.Length0 + ")");
            if (result.GetLength(1) != Length1) throw new ArgumentException("result.GetLength(1) (" + result.GetLength(1) + ") != this.Length1 (" + this.Length1 + ")");
            for (int i = 0; i < Length0; i++)
            {
                for (int j = 0; j < Length1; j++)
                {
                    result[i, j] = this[i, j].Sample(result[i, j]);
                }
            }
            return result;
        }

#if SpecializeInterfaces
        /// <summary>
        /// Set the parameters of this distribution to match those of the given distribution (by value)
        /// </summary>
        /// <param name="that"></param>
        public void SetTo(DistributionStructArray2D<T, DomainType> that)
        {
            base.SetTo(that);
        }

        /// <summary>
        /// Set the parameters to represent the product of two distributions
        /// </summary>
        /// <param name="a">The first distribution</param>
        /// <param name="b">The second distribution</param>
        public void SetToProduct(DistributionStructArray2D<T, DomainType> a, DistributionStructArray2D<T, DomainType> b)
        {
            Distribution.SetToProduct(array, a.array, b.array);
        }

        /// <summary>
        /// Set the parameters to represent the ratio of two distributions
        /// </summary>
        /// <param name="numerator">The numerator distribution array</param>
        /// <param name="denominator">The denominator distribution array</param>
        /// <param name="forceProper"></param>
        public void SetToRatio(DistributionStructArray2D<T, DomainType> numerator, DistributionStructArray2D<T, DomainType> denominator, bool forceProper)
        {
            Distribution.SetToRatio(array, numerator.array, denominator.array, forceProper);
        }

        /// <summary>
        /// Set the parameters to represent the power of a source distribution to some exponent
        /// </summary>
        /// <param name="a">The source distribution array</param>
        /// <param name="exponent">The exponent</param>
        public void SetToPower(DistributionStructArray2D<T, DomainType> a, double exponent)
        {
            Distribution.SetToPower(array, a.array, exponent);
        }

        /// <summary>
        /// Set the parameters to match the moments of a mixture of two distributions
        /// </summary>
        /// <param name="weight1">The first weight</param>
        /// <param name="a">The first distribution array</param>
        /// <param name="weight2">The second weight</param>
        /// <param name="b">The second distribution array</param>
        public void SetToSum(double weight1, DistributionStructArray2D<T, DomainType> a, double weight2, DistributionStructArray2D<T, DomainType> b)
        {
            Distribution.SetToSum(array, weight1, a.array, weight2, b.array);
        }

        /// <summary>
        /// The log-probability that two distributions would draw the same sample.
        /// </summary>
        /// <param name="that"></param>
        /// <returns><c>Math.Log(sum_x this.Evaluate(x)*that.Evaluate(x))</c></returns>
        /// <remarks>This can be considered a type of inner product between distributions.
        /// Another name might be "LogAverage" to go with "GetAverageLog".
        /// </remarks>
        public double GetLogAverageOf(DistributionStructArray2D<T, DomainType> that)
        {
            return Distribution.GetLogAverageOf(array, that.array);
        }

        /// <summary>
        /// Get the integral of this distribution times another distribution raised to a power.
        /// </summary>
        /// <param name="that"></param>
        /// <param name="power"></param>
        /// <returns></returns>
        public double GetLogAverageOfPower(DistributionStructArray2D<T, DomainType> that, double power)
        {
            return Distribution.GetLogAverageOfPower(array, that.array, power);
        }

        /// <summary>
        /// The expected logarithm of that distribution under this distribution.
        /// </summary>
        /// <param name="that">The distribution to take the logarithm of.</param>
        /// <returns><c>sum_x this.Evaluate(x)*Math.Log(that.Evaluate(x))</c></returns>
        /// <remarks>This is also known as the cross entropy.
        /// </remarks>
        public double GetAverageLog(DistributionStructArray2D<T, DomainType> that)
        {
            return Distribution.GetAverageLog(array, that.array);
        }
#endif
    }

    #endregion

    #region DistributionRefArray2D

    /// <summary>
    /// A distribution over a 2D array of type <typeparamref name="DomainType"/>, where each element is independent and has distribution of type <typeparamref name="T"/>
    /// </summary>
    /// <typeparam name="T">Distribution type of an element</typeparam>
    /// <typeparam name="DomainType">Domain type of an element</typeparam>
    /// <remarks>
    /// This is an extension of DistributionArray that requires T to be a reference type.
    /// The SetTo and CopyTo methods are overriden to use cloning instead of assignment.
    /// </remarks>
    [Serializable]
    [Quality(QualityBand.Mature)]
    [DataContract]
    public class DistributionRefArray2D<T, DomainType> : DistributionArray2D<T, DomainType>
#if SpecializeInterfaces
                                                         , SettableTo<DistributionRefArray2D<T, DomainType>>,
                                                         SettableToProduct<DistributionRefArray2D<T, DomainType>>,
                                                         SettableToRatio<DistributionRefArray2D<T, DomainType>>,
                                                         SettableToPower<DistributionRefArray2D<T, DomainType>>,
                                                         SettableToWeightedSum<DistributionRefArray2D<T, DomainType>>,
                                                         CanGetLogAverageOf<DistributionRefArray2D<T, DomainType>>,
                                                         CanGetLogAverageOfPower<DistributionRefArray2D<T, DomainType>>,
                                                         CanGetAverageLog<DistributionRefArray2D<T, DomainType>>
#endif
        where T : class, ICloneable, SettableTo<T>,
            SettableToUniform,
            Diffable,
            SettableToProduct<T>,
            SettableToRatio<T>,
            SettableToPower<T>,
            SettableToWeightedSum<T>,
            CanGetLogAverageOf<T>,
            CanGetLogAverageOfPower<T>,
            CanGetAverageLog<T>,
            IDistribution<DomainType>,
            Sampleable<DomainType>
    {
        /// <summary>
        /// Parameterless constructor needed for serialization
        /// </summary>
        protected DistributionRefArray2D()
        {
        }

        /// <summary>
        /// Constructs a new 2-D distribution array given the lengths of each dimension
        /// </summary>
        /// <param name="length0"></param>
        /// <param name="length1"></param>
        /// <returns></returns>
        public DistributionRefArray2D(int length0, int length1)
            : this(length0, length1, new T[length0*length1])
        {
        }

        /// <summary>
        /// Constructs a new 2-D distribution array given the lengths of each dimension,
        /// and an array of values to reference
        /// </summary>
        /// <param name="length0"></param>
        /// <param name="length1"></param>
        /// <param name="valuesRef">Array of values to reference</param>
        /// <returns></returns>
        public DistributionRefArray2D(int length0, int length1, T[] valuesRef)
            : base(length0, length1, valuesRef)
        {
        }

        /// <summary>
        /// Constructs a new 2-D distribution array given the lengths of each dimension and initial values.
        /// </summary>
        /// <param name="length0"></param>
        /// <param name="length1"></param>
        /// <param name="init">Function that maps an index to a value.</param>
        /// <returns></returns>
        /// <remarks>
        /// The references returned by <paramref name="init"/> are not copied.  They are placed directly into the array.
        /// </remarks>
        public DistributionRefArray2D(int length0, int length1, Func<int, int, T> init)
            : this(length0, length1, new T[length0*length1])
        {
            for (int i = 0; i < length0; i++)
            {
                for (int j = 0; j < length1; j++)
                {
                    this[i, j] = init(i, j);
                }
            }
        }

        /// <summary>
        /// Constructs a new 2-D distribution array given the lengths of each dimension, and a value
        /// </summary>
        /// <param name="value"></param>
        /// <param name="length0"></param>
        /// <param name="length1"></param>
        /// <returns></returns>
        public DistributionRefArray2D(T value, int length0, int length1)
            : this(length0, length1)
        {
            SetAllElementsTo(value);
        }

        /// <summary>
        /// Constructs a new 2-D distribution array given a 2-D array of distributions
        /// </summary>
        /// <param name="array"></param>
        /// <returns></returns>
        public DistributionRefArray2D(T[,] array)
            : base(array)
        {
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that"></param>
        public DistributionRefArray2D(Array2D<T> that)
            : base(that)
        {
        }

        /// <summary>
        /// Initialise the values in this 2-D distribution array to clones of the values in a given 2-D array of distributions
        /// </summary>
        /// <param name="array"></param>
        /// <remarks>Given array and this array must be the same length in each dimension</remarks>
        protected void InitializeTo(T[,] array)
        {
            if (array.GetLength(0) != Length0) throw new ArgumentException("array.GetLength(0) (" + array.GetLength(0) + ") != this.Length0 (" + this.Length0 + ")");
            if (array.GetLength(1) != Length1) throw new ArgumentException("array.GetLength(1) (" + array.GetLength(1) + ") != this.Length1 (" + this.Length1 + ")");
            for (int i = 0; i < Length0; i++)
            {
                for (int j = 0; j < Length1; j++)
                {
                    this[i, j] = (T) array[i, j].Clone();
                }
            }
        }

        /// <summary>
        /// Set the parameters of this distribution so that the marginals match the given distributions (by value)
        /// </summary>
        /// <param name="array"></param>
        /// <remarks>Given array and this array must be the same length in each dimension</remarks>
        public override void SetTo(T[,] array)
        {
            if (array.GetLength(0) != Length0) throw new ArgumentException("array.GetLength(0) (" + array.GetLength(0) + ") != this.Length0 (" + this.Length0 + ")");
            if (array.GetLength(1) != Length1) throw new ArgumentException("array.GetLength(1) (" + array.GetLength(1) + ") != this.Length1 (" + this.Length1 + ")");
            if (array.Length == 0) return;
            if (this.array[0] == null)
            {
                InitializeTo(array);
            }
            else
            {
#if false
                ModifyAll(array, delegate(T item, T arrayItem) { item.SetTo(arrayItem); return item; });
#else
                for (int i = 0; i < Length0; i++)
                {
                    for (int j = 0; j < Length1; j++)
                    {
                        this[i, j].SetTo(array[i, j]);
                    }
                }
#endif
            }
        }

        /// <summary>
        /// Initialises all the values in this 2-D distribution array to clones of the given value
        /// </summary>
        /// <param name="value"></param>
        protected void InitializeTo(T value)
        {
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = (T) value.Clone();
            }
        }

        /// <summary>
        /// Set the parameters of this distribution so that all marginals equal the given distribution (by value)
        /// </summary>
        /// <param name="value"></param>
        public override void SetAllElementsTo(T value)
        {
            if (array.Length == 0) return;
            if (this.array[0] == null)
            {
                InitializeTo(value);
            }
            else
            {
                for (int i = 0; i < array.Length; i++)
                {
                    array[i].SetTo(value);
                }
            }
        }

        /// <summary>
        /// Set the parameters of array[i,j] to match the marginal distribution of element (i,j), creating a new distribution if array[i,j] was null
        /// </summary>
        /// <param name="array"></param>
        public override void CopyTo(T[,] array)
        {
            if (array.GetLength(0) != Length0) throw new ArgumentException("array.GetLength(0) (" + array.GetLength(0) + ") != this.Length0 (" + this.Length0 + ")");
            if (array.GetLength(1) != Length1) throw new ArgumentException("array.GetLength(1) (" + array.GetLength(1) + ") != this.Length1 (" + this.Length1 + ")");
            if (array.Length == 0) return;
            if (array[0, 0] == null)
            {
                for (int i = 0; i < Length0; i++)
                {
                    for (int j = 0; j < Length1; j++)
                    {
                        array[i, j] = (T) this[i, j].Clone();
                    }
                }
            }
            else
            {
                SetItemsOf(array);
            }
        }

        /// <summary>
        /// Set the parameters of array[i,j] to match the marginal distribution of element (i,j)
        /// </summary>
        /// <param name="array"></param>
        protected void SetItemsOf(T[,] array)
        {
            if (array.GetLength(0) != Length0) throw new ArgumentException("array.GetLength(0) (" + array.GetLength(0) + ") != this.Length0 (" + this.Length0 + ")");
            if (array.GetLength(1) != Length1) throw new ArgumentException("array.GetLength(1) (" + array.GetLength(1) + ") != this.Length1 (" + this.Length1 + ")");
            for (int i = 0; i < Length0; i++)
            {
                for (int j = 0; j < Length1; j++)
                {
                    array[i, j].SetTo(this[i, j]);
                }
            }
        }

        /// <summary>
        /// Set the parameters of array[i] to match the marginal distribution of element i,
        /// starting at a given index
        /// </summary>
        /// <param name="array"></param>
        /// <param name="arrayIndex">The index to start at</param>
        protected void SetItemsOf(T[] array, int arrayIndex)
        {
            for (int i = 0; i < this.array.Length; i++)
            {
                array[i + arrayIndex].SetTo(this.array[i]);
            }
        }

        /// <summary>
        /// Set the parameters of array[i] to match the marginal distribution of element i, creating a new distribution if array[i] was null,
        /// starting at a given index
        /// </summary>
        /// <param name="array"></param>
        /// <param name="arrayIndex">The index to start at</param>
        public override void CopyTo(T[] array, int arrayIndex)
        {
            if (this.array.Length == 0) return;
            if (array[arrayIndex] == null)
            {
                for (int i = 0; i < this.array.Length; i++)
                {
                    array[i + arrayIndex] = (T) this.array[i].Clone();
                }
            }
            else
            {
                SetItemsOf(array, arrayIndex);
            }
        }

        /// <summary>
        /// Clone the distribution
        /// </summary>
        /// <returns></returns>
        public override object Clone()
        {
            return new DistributionRefArray2D<T, DomainType>(this);
        }

        /// <summary>
        /// Get a sample from the distribution
        /// </summary>
        /// <param name="result">Where to put the result</param>
        /// <returns>A sample</returns>
        public override DomainType[,] Sample(DomainType[,] result)
        {
            if (result == null) return Sample();
            if (result.GetLength(0) != Length0) throw new ArgumentException("result.GetLength(0) (" + result.GetLength(0) + ") != this.Length0 (" + this.Length0 + ")");
            if (result.GetLength(1) != Length1) throw new ArgumentException("result.GetLength(1) (" + result.GetLength(1) + ") != this.Length1 (" + this.Length1 + ")");
            for (int i = 0; i < Length0; i++)
            {
                for (int j = 0; j < Length1; j++)
                {
                    if (result[i, j] == null) result[i, j] = (DomainType) this[i, j].Sample();
                    else result[i, j] = (DomainType) this[i, j].Sample(result[i, j]);
                }
            }
            return result;
        }

#if SpecializeInterfaces
        /// <summary>
        /// Set the parameters of this distribution to match those of the given distribution (by value)
        /// </summary>
        /// <param name="that"></param>
        public void SetTo(DistributionRefArray2D<T, DomainType> that)
        {
            base.SetTo(that);
        }

        /// <summary>
        /// Set the parameters to represent the product of two distributions
        /// </summary>
        /// <param name="a">The first distribution array</param>
        /// <param name="b">The second distribution array</param>
        public void SetToProduct(DistributionRefArray2D<T, DomainType> a, DistributionRefArray2D<T, DomainType> b)
        {
            Distribution.SetToProduct(array, a.array, b.array);
        }

        /// <summary>
        /// Set the parameters to represent the ratio of two distributions
        /// </summary>
        /// <param name="numerator">The numerator distribution array</param>
        /// <param name="denominator">The denominator distribution array</param>
        /// <param name="forceProper">Argument passed to T.SetToRatio</param>
        public void SetToRatio(DistributionRefArray2D<T, DomainType> numerator, DistributionRefArray2D<T, DomainType> denominator, bool forceProper)
        {
            Distribution.SetToRatio(array, numerator.array, denominator.array, forceProper);
        }

        /// <summary>
        /// Set the parameters to represent the power of a source distribution to some exponent
        /// </summary>
        /// <param name="a">The source distribution array</param>
        /// <param name="exponent">The exponent</param>
        public void SetToPower(DistributionRefArray2D<T, DomainType> a, double exponent)
        {
            Distribution.SetToPower(array, a.array, exponent);
        }

        /// <summary>
        /// Set the parameters to match the moments of a mixture of two distributions
        /// </summary>
        /// <param name="weight1">The first weight</param>
        /// <param name="a">The first distribution array</param>
        /// <param name="weight2">The second weight</param>
        /// <param name="b">The second distribution array</param>
        public void SetToSum(double weight1, DistributionRefArray2D<T, DomainType> a, double weight2, DistributionRefArray2D<T, DomainType> b)
        {
            Distribution.SetToSum(array, weight1, a.array, weight2, b.array);
        }

        /// <summary>
        /// The log-probability that two distributions would draw the same sample.
        /// </summary>
        /// <param name="that"></param>
        /// <returns><c>Math.Log(sum_x this.Evaluate(x)*that.Evaluate(x))</c></returns>
        /// <remarks>This can be considered a type of inner product between distributions.
        /// Another name might be "LogAverage" to go with "GetAverageLog".
        /// </remarks>
        public double GetLogAverageOf(DistributionRefArray2D<T, DomainType> that)
        {
            return Distribution.GetLogAverageOf(array, that.array);
        }

        /// <summary>
        /// Get the integral of this distribution times another distribution raised to a power.
        /// </summary>
        /// <param name="that"></param>
        /// <param name="power"></param>
        /// <returns></returns>
        public double GetLogAverageOfPower(DistributionRefArray2D<T, DomainType> that, double power)
        {
            return Distribution.GetLogAverageOfPower(array, that.array, power);
        }

        /// <summary>
        /// The expected logarithm of that distribution under this distribution.
        /// </summary>
        /// <param name="that">The distribution to take the logarithm of.</param>
        /// <returns><c>sum_x this.Evaluate(x)*Math.Log(that.Evaluate(x))</c></returns>
        /// <remarks>This is also known as the cross entropy.
        /// </remarks>
        public double GetAverageLog(DistributionRefArray2D<T, DomainType> that)
        {
            return Distribution.GetAverageLog(array, that.array);
        }
#endif
    }

    #endregion
}
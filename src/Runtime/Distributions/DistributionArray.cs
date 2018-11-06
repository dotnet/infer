// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#define SpecializeInterfaces

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Linq;
    using System.Runtime.Serialization;
    using Collections;
    using Math;
    using Factors.Attributes;
    using Utilities;
    using Microsoft.ML.Probabilistic.Serialization;

    #region DistributionArray<T>

    /// <summary>
    /// A distribution over an array, where each element is independent and has distribution type <typeparamref name="T"/>
    /// </summary>
    /// <typeparam name="T">The distribution type of an element</typeparam>
    [Serializable]
    [DataContract]
    public abstract class DistributionArray<T> : Array<T>,
                                                 SettableToUniform,
                                                 Diffable
#if !SpecializeInterfaces
        ,SettableTo<DistributionArray<T>>,
        SettableToProduct<DistributionArray<T>>,
        SettableToRatio<DistributionArray<T>>,
        SettableToPower<DistributionArray<T>>,
        SettableToWeightedSum<DistributionArray<T>>,
        CanGetLogAverageOf<DistributionArray<T>>,
        CanGetAverageLog<DistributionArray<T>>
#endif
        where T : SettableToUniform,
            Diffable
#if !SpecializeInterfaces
        ,SettableToProduct<T>,
        SettableToRatio<T>,
        SettableToPower<T>,
        SettableToWeightedSum<T>,
        CanGetLogAverageOf<T>,
        CanGetAverageLog<T>
#endif
    {
        /// <summary>
        /// Parameterless constructor needed for serialization
        /// </summary>
        protected DistributionArray()
        {
        }

        /// <summary>
        /// Creates a distribution array of a specified length
        /// </summary>
        /// <param name="length"></param>
        public DistributionArray(int length)
            : base(length)
        {
        }

        /// <summary>
        /// Creates a distribution array from an array of distributions
        /// </summary>
        /// <param name="array"></param>
        public DistributionArray(T[] array)
            : base(array)
        {
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that"></param>
        public DistributionArray(Array<T> that)
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
            return new DistributionArray<T>(this);
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
        /// <returns>True if uniform, otherwise false</returns>
        public bool IsUniform()
        {
            return this.All(item => item.IsUniform());
        }

        /// <summary>
        /// The maximum difference in parameter values between this distribution and that distribution
        /// </summary>
        /// <param name="that">That distribution</param>
        /// <returns>The maximum difference</returns>
        public virtual double MaxDiff(object that)
        {
            DistributionArray<T> thatd = that as DistributionArray<T>;
            if ((object) thatd == null) return Double.PositiveInfinity;
            return Distribution.MaxDiff(this.array, thatd.array);
        }

#if !SpecializeInterfaces
        public void SetTo(DistributionArray<T> that)
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
        public void SetToProduct(DistributionArray<T> a, DistributionArray<T> b)
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
        public void SetToRatio(DistributionArray<T> numerator, DistributionArray<T> denominator)
        {
            Distribution.SetToRatio(array, numerator.array, denominator.array);
        }

        /// <summary>
        /// Sets the current instance to an array of distributions each element
        /// of which is a power of the corresponding element in a source distribution array
        /// </summary>
        /// <param name="a">The source distribution array</param>
        /// <param name="exponent">The exponent</param>
        public void SetToPower(DistributionArray<T> a, double exponent)
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
        public void SetToSum(double weight1, DistributionArray<T> a, double weight2, DistributionArray<T> b)
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
        public double GetLogAverageOf(DistributionArray<T> that)
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
        public double GetAverageLog(DistributionArray<T> that)
        {
            return Distribution.GetAverageLog(array, that.array);
        }
#endif
    }

    #endregion

    #region DistributionArray<T, DomainType>

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
    public abstract class DistributionArray<T, DomainType> : DistributionArray<T>, IDistribution<DomainType[]>, Sampleable<DomainType[]>
        where T : SettableToUniform,
            Diffable,
            Sampleable<DomainType>,
#if !SpecializeInterfaces
        SettableToProduct<T>,
        SettableToRatio<T>,
        SettableToPower<T>,
        SettableToWeightedSum<T>,
        CanGetLogAverageOf<T>,
        CanGetAverageLog<T>,
#endif
            IDistribution<DomainType>
    {
        /// <summary>
        /// Parameterless constructor needed for serialization
        /// </summary>
        protected DistributionArray()
        {
        }

        /// <summary>
        /// Creates a distribution array of a specified length
        /// </summary>
        /// <param name="length"></param>
        public DistributionArray(int length)
            : base(length)
        {
        }

        /// <summary>
        /// Creates a distribution array from an array of distributions
        /// </summary>
        /// <param name="array"></param>
        public DistributionArray(T[] array)
            : base(array)
        {
        }

        /// <summary>
        /// Copy constructor
        /// </summary>
        /// <param name="that"></param>
        public DistributionArray(Array<T> that)
            : base(that)
        {
        }

        /// <summary>
        /// Sets/gets the instance as a point mass
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public DomainType[] Point
        {
            get
            {
                DomainType[] result = new DomainType[Count];
                for (int i = 0; i < array.Length; i++)
                {
                    result[i] = array[i].Point;
                }
                return result;
            }
            set
            {
                if (value.Length != this.Count) throw new ArgumentException("value.Length (" + value.Length + ") != this.Count (" + this.Count + ")");
                for (int i = 0; i < array.Length; i++)
                {
                    T item = array[i];
                    item.Point = value[i];
                    array[i] = item;
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
        /// Logarithm of the probability density function
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public double GetLogProb(DomainType[] value)
        {
            double sum = 0.0;
            ForEach(value, delegate(T distribution, DomainType item) { sum += distribution.GetLogProb(item); });
            return sum;
        }

        #region Sampleable<DomainType[]> Members

        /// <summary>
        /// Get a sample from the distribution array
        /// </summary>
        /// <returns>An array of samples</returns>
        public DomainType[] Sample()
        {
            DomainType[] result = new DomainType[Count];
            return this.Sample(result);
        }

        /// <summary>
        /// Get a sample from the distribution array
        /// </summary>
        /// <param name="result">Where to put the results</param>
        /// <returns>An array of samples</returns>
        public abstract DomainType[] Sample(DomainType[] result);

        #endregion
    }

    #endregion

    #region DistributionStructArray

    /// <summary>
    /// A distribution over an array of type <typeparamref name="DomainType"/>, where each element is independent and has distribution of type <typeparamref name="T"/>
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
    public class DistributionStructArray<T, DomainType> : DistributionArray<T, DomainType>, IArrayFactory<T, DistributionStructArray<T, DomainType>>
#if SpecializeInterfaces
                                                          , SettableTo<DistributionStructArray<T, DomainType>>,
                                                          SettableToProduct<DistributionStructArray<T, DomainType>>,
                                                          SettableToRatio<DistributionStructArray<T, DomainType>>,
                                                          SettableToPower<DistributionStructArray<T, DomainType>>,
                                                          SettableToWeightedSum<DistributionStructArray<T, DomainType>>,
                                                          CanGetLogAverageOf<DistributionStructArray<T, DomainType>>,
                                                          CanGetLogAverageOfPower<DistributionStructArray<T, DomainType>>,
                                                          CanGetAverageLog<DistributionStructArray<T, DomainType>>
#endif
        where T : struct,
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
        protected DistributionStructArray()
        {
        }

        /// <summary>
        /// Create a new distribution struct array of a specified length
        /// </summary>
        /// <param name="length"></param>
        [Skip]
        public DistributionStructArray(int length)
            : base(length)
        {
        }

        /// <summary>
        /// Create a new distribution struct array of a specified length and initial values.
        /// </summary>
        /// <param name="length"></param>
        /// <param name="init">Function that maps an index to a value.</param>
        public DistributionStructArray(int length, Func<int, T> init)
            : base(length)
        {
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = init(i);
            }
        }

        public DistributionStructArray<T, DomainType> CreateArray(int length, Func<int, T> init)
        {
            return new DistributionStructArray<T, DomainType>(length, init);
        }

        /// <summary>
        /// Create a new distribution struct array of a specified value and length
        /// </summary>
        /// <param name="value"></param>
        /// <param name="length"></param>
        public DistributionStructArray(T value, int length)
            : this(length)
        {
            SetAllElementsTo(value);
        }

        /// <summary>
        /// Create a new distribution struct array from an array of distributions
        /// </summary>
        /// <param name="array"></param>
        public DistributionStructArray(T[] array)
            : base(array)
        {
        }

        /// <summary>
        /// Copy constructor
        /// </summary>
        /// <param name="that"></param>
        public DistributionStructArray(Array<T> that)
            : base(that)
        {
        }

        /// <summary>
        /// Clone the distribution
        /// </summary>
        /// <returns></returns>
        public override object Clone()
        {
            return new DistributionStructArray<T, DomainType>(this);
        }


        /// <summary>
        /// The maximum difference in parameters between
        /// this distribution array and that distribution array
        /// </summary>
        /// <param name="that">That distribution array</param>
        /// <returns>The maximum difference</returns>
        public override double MaxDiff(object that)
        {
            DistributionStructArray<T, DomainType> thatd = that as DistributionStructArray<T, DomainType>;
            if ((object) thatd == null) return Double.PositiveInfinity;
            return Distribution.MaxDiff(this.array, thatd.array);
        }

        /// <summary>
        /// Get a sample from the distribution
        /// </summary>
        /// <param name="result">Where to put the result</param>
        /// <returns>A sample</returns>
        public override DomainType[] Sample(DomainType[] result)
        {
            if (result == null) return Sample();
            if (result.Length != this.Count) throw new ArgumentException("result.Length (" + result.Length + ") != this.Count (" + this.Count + ")");
            for (int i = 0; i < array.Length; i++)
            {
                result[i] = array[i].Sample(result[i]);
            }
            return result;
        }

#if SpecializeInterfaces
        /// <summary>
        /// Set the parameters of this distribution to match those of the given distribution (by value)
        /// </summary>
        /// <param name="that"></param>
        public void SetTo(DistributionStructArray<T, DomainType> that)
        {
            base.SetTo(that);
        }

        /// <summary>
        /// Set the parameters to represent the product of two distributions
        /// </summary>
        /// <param name="a">The first distribution array</param>
        /// <param name="b">The second distribution array</param>
        public void SetToProduct(DistributionStructArray<T, DomainType> a, DistributionStructArray<T, DomainType> b)
        {
            Distribution.SetToProduct(array, a.array, b.array);
        }

        /// <summary>
        /// Set the parameters to represent the ratio of two distributions
        /// </summary>
        /// <param name="numerator">The numerator distribution array</param>
        /// <param name="denominator">The denominator distribution array</param>
        /// <param name="forceProper"></param>
        public void SetToRatio(DistributionStructArray<T, DomainType> numerator, DistributionStructArray<T, DomainType> denominator, bool forceProper)
        {
            Distribution.SetToRatio(array, numerator.array, denominator.array, forceProper);
        }

        /// <summary>
        /// Set the parameters to represent the power of a source distribution to some exponent
        /// </summary>
        /// <param name="a">The source distribution array</param>
        /// <param name="exponent">The exponent</param>
        public void SetToPower(DistributionStructArray<T, DomainType> a, double exponent)
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
        public void SetToSum(double weight1, DistributionStructArray<T, DomainType> a, double weight2, DistributionStructArray<T, DomainType> b)
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
        public double GetLogAverageOf(DistributionStructArray<T, DomainType> that)
        {
            return Distribution.GetLogAverageOf(array, that.array);
        }

        /// <summary>
        /// Get the integral of this distribution times another distribution raised to a power.
        /// </summary>
        /// <param name="that"></param>
        /// <param name="power"></param>
        /// <returns></returns>
        public double GetLogAverageOfPower(DistributionStructArray<T, DomainType> that, double power)
        {
            return Distribution.GetLogAverageOfPower(array, that.array, power);
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
        public double GetAverageLog(DistributionStructArray<T, DomainType> that)
        {
            return Distribution.GetAverageLog(array, that.array);
        }
#endif
    }

    #endregion

    #region DistributionRefArray

    /// <summary>
    /// A distribution over an array of type <typeparamref name="DomainType"/>, where each element is independent and has distribution of type <typeparamref name="T"/>
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
    public class DistributionRefArray<T, DomainType> : DistributionArray<T, DomainType>, IArrayFactory<T, DistributionRefArray<T, DomainType>>
#if SpecializeInterfaces
                                                       , SettableTo<DistributionRefArray<T, DomainType>>,
                                                       SettableToProduct<DistributionRefArray<T, DomainType>>,
                                                       SettableToRatio<DistributionRefArray<T, DomainType>>,
                                                       SettableToPower<DistributionRefArray<T, DomainType>>,
                                                       SettableToWeightedSum<DistributionRefArray<T, DomainType>>,
                                                       CanGetLogAverageOf<DistributionRefArray<T, DomainType>>,
                                                       CanGetLogAverageOfPower<DistributionRefArray<T, DomainType>>,
                                                       CanGetAverageLog<DistributionRefArray<T, DomainType>>
#endif
        where T : class, ICloneable, SettableTo<T>,
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
        protected DistributionRefArray()
        {
        }

        /// <summary>
        /// Creates a distribution array given a length
        /// </summary>
        /// <param name="length"></param>
        [Skip]
        public DistributionRefArray(int length)
            : base(length)
        {
        }

        /// <summary>
        /// Create a distribution struct array of a specified length and initial values.
        /// </summary>
        /// <param name="length"></param>
        /// <param name="init">Function that maps an index to a value.</param>
        /// <remarks>
        /// The references returned by <paramref name="init"/> are not copied.  They are placed directly into the array.
        /// </remarks>
        public DistributionRefArray(int length, Func<int, T> init)
            : base(length)
        {
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = init(i);
            }
        }

        public DistributionRefArray<T, DomainType> CreateArray(int length, Func<int, T> init)
        {
            return new DistributionRefArray<T, DomainType>(length, init);
        }

        /// <summary>
        /// Creates a new distribution array given a value and a length
        /// </summary>
        /// <param name="value"></param>
        /// <param name="length"></param>
        public DistributionRefArray(T value, int length)
            : this(length)
        {
            SetAllElementsTo(value);
        }

        /// <summary>
        /// Creates a new distribution array given an array of distributions
        /// </summary>
        /// <param name="array"></param>
        public DistributionRefArray(T[] array)
            : base(array)
        {
        }

        /// <summary>
        /// Copy constructor
        /// </summary>
        /// <param name="that"></param>
        public DistributionRefArray(Array<T> that)
            : base(that)
        {
        }

        /// <summary>
        /// Initialise the elements of this distribution array to clones of the given distributions
        /// </summary>
        /// <param name="array"></param>
        /// <remarks>Given array and this array must be the same length</remarks>
        protected void InitializeTo(T[] array)
        {
            if (array.Length != this.Count) throw new ArgumentException("array.Length (" + array.Length + ") != this.Count (" + this.Count + ")");
            for (int i = 0; i < array.Length; i++)
            {
                this[i] = (T) array[i].Clone();
            }
        }

        /// <summary>
        /// Set the parameters of this distribution so that the marginals match the given distributions (by value)
        /// </summary>
        /// <param name="array"></param>
        /// <remarks>Given array and this array must be the same length.</remarks>
        public override void SetTo(T[] array)
        {
            if (array.Length != this.Count) throw new ArgumentException("array.Length (" + array.Length + ") != this.Count (" + this.Count + ")");
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
                for (int i = 0; i < array.Length; i++)
                {
                    this[i].SetTo(array[i]);
                }
#endif
            }
        }

        /// <summary>
        /// Initialise all the values in this array to clones of the given value
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
        /// Set the parameters of array[i] to match the marginal distribution of element i, creating a new distribution if array[i] was null
        /// </summary>
        /// <param name="array">Array to write into</param>
        public override void CopyTo(T[] array)
        {
            if (array.Length != this.Count) throw new ArgumentException("array.Length (" + array.Length + ") != this.Count (" + this.Count + ")");
            if (array.Length == 0) return;
            if (array[0] == null)
            {
                for (int i = 0; i < array.Length; i++)
                {
                    array[i] = (T) this[i].Clone();
                }
            }
            else
            {
                SetItemsOf(array);
            }
        }

        /// <summary>
        /// Set the parameters of array[i] to match the marginal distribution of element i
        /// </summary>
        /// <param name="array"></param>
        protected void SetItemsOf(T[] array)
        {
            if (array.Length != this.Count) throw new ArgumentException("array.Length (" + array.Length + ") != this.Count (" + this.Count + ")");
            for (int i = 0; i < array.Length; i++)
            {
                array[i].SetTo(this[i]);
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
        /// Clones the array and the items in the array.
        /// </summary>
        /// <returns></returns>
        public override object Clone()
        {
            return new DistributionRefArray<T, DomainType>(this);
        }

        /// <summary>
        /// Get a sample from the distribution
        /// </summary>
        /// <param name="result">Where to put the results</param>
        /// <returns>A sample</returns>
        public override DomainType[] Sample(DomainType[] result)
        {
            if (result == null) return Sample();
            if (result.Length != this.Count) throw new ArgumentException("result.Length (" + result.Length + ") != this.Count (" + this.Count + ")");
            for (int i = 0; i < array.Length; i++)
            {
                if (result[i] == null) result[i] = array[i].Sample();
                else result[i] = array[i].Sample(result[i]);
            }
            return result;
        }

#if SpecializeInterfaces
        /// <summary>
        /// Set the parameters of this distribution to match those of the given distribution (by value)
        /// </summary>
        /// <param name="that"></param>
        public void SetTo(DistributionRefArray<T, DomainType> that)
        {
            base.SetTo(that);
        }

        /// <summary>
        /// Set the parameters to represent the product of two distributions
        /// </summary>
        /// <param name="a">The first distribution array</param>
        /// <param name="b">The second distribution array</param>
        public void SetToProduct(DistributionRefArray<T, DomainType> a, DistributionRefArray<T, DomainType> b)
        {
            Distribution.SetToProduct(array, a.array, b.array);
        }

        /// <summary>
        /// Set the parameters to represent the ratio of two distributions
        /// </summary>
        /// <param name="numerator">The numerator distribution array</param>
        /// <param name="denominator">The denominator distribution array</param>
        /// <param name="forceProper">Argument passed to T.SetToRatio</param>
        public void SetToRatio(DistributionRefArray<T, DomainType> numerator, DistributionRefArray<T, DomainType> denominator, bool forceProper)
        {
            Distribution.SetToRatio(array, numerator.array, denominator.array, forceProper);
        }

        /// <summary>
        /// Set the parameters to represent the power of a source distribution to some exponent
        /// </summary>
        /// <param name="a">The source distribution array</param>
        /// <param name="exponent">The exponent</param>
        public void SetToPower(DistributionRefArray<T, DomainType> a, double exponent)
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
        public void SetToSum(double weight1, DistributionRefArray<T, DomainType> a, double weight2, DistributionRefArray<T, DomainType> b)
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
        public double GetLogAverageOf(DistributionRefArray<T, DomainType> that)
        {
            return Distribution.GetLogAverageOf(array, that.array);
        }

        /// <summary>
        /// Get the integral of this distribution times another distribution raised to a power.
        /// </summary>
        /// <param name="that"></param>
        /// <param name="power"></param>
        /// <returns></returns>
        public double GetLogAverageOfPower(DistributionRefArray<T, DomainType> that, double power)
        {
            return Distribution.GetLogAverageOfPower(array, that.array, power);
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
        public double GetAverageLog(DistributionRefArray<T, DomainType> that)
        {
            return Distribution.GetAverageLog(array, that.array);
        }
#endif
    }

    #endregion
}
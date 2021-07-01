// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Collections.Generic;
    using System.Runtime.Serialization;
    using Microsoft.ML.Probabilistic.Factors.Attributes;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;

    // TODO: switch to a collection of interfaces with default method implementations
    // after updating to netstandard2.1+ or net5+ and C# 8.0+

    /// <summary>
    /// A mutable wrapper for an immutable distribution.
    /// 
    /// Considering that typically such wrapper should be a struct containing a single reference
    /// to the wrapped immutable distribution, this class should not be inherited, but instead
    /// used as a reference implementation.
    /// </summary>
    /// <typeparam name="TDomain">The type of objects in the domain, e.g. double or string. Should be an immutable type.</typeparam>
    /// <typeparam name="TWrapped">The type of the wrapped <see cref="IImmutableDistribution{T, TThis}"/> implementation.</typeparam>
    /// <remarks>When Infer.Net's target runtime is updated to one that supports interfaces with default method implementations,
    /// this class should be replaced with a collection of such interfaces, which would be possible to inherit.</remarks>
    [Quality(QualityBand.Experimental)]
    [Serializable]
    internal abstract class MutableWrapper<TDomain, TWrapped> :
        IDistribution<TDomain>,
        SettableTo<TWrapped>, SettableTo<MutableWrapper<TDomain, TWrapped>>,
        SettableToPartialUniform<TWrapped>, SettableToPartialUniform<MutableWrapper<TDomain, TWrapped>>,
        CanGetMode<TDomain>,
        CanGetProbLessThan<TDomain>,
        CanGetQuantile<TDomain>,
        CanGetMean<TDomain>,
        CanSetMean<TDomain>,
        CanGetVariance<TDomain>,
        CanGetMeanAndVarianceOut<TDomain, TDomain>,
        CanSetMeanAndVariance<TDomain, TDomain>,
        CanGetLogNormalizer,
        CanGetLogAverageOf<TWrapped>, CanGetLogAverageOf<MutableWrapper<TDomain, TWrapped>>,
        CanGetLogAverageOfPower<TWrapped>, CanGetLogAverageOfPower<MutableWrapper<TDomain, TWrapped>>,
        CanGetAverageLog<TWrapped>, CanGetAverageLog<MutableWrapper<TDomain, TWrapped>>,
        CanEnumerateSupport<TDomain>,
        SettableToProduct<TWrapped>, SettableToProduct<MutableWrapper<TDomain, TWrapped>>,
        SettableToProduct<TWrapped, MutableWrapper<TDomain, TWrapped>>, SettableToProduct<MutableWrapper<TDomain, TWrapped>, TWrapped>,
        SettableToRatio<TWrapped>, SettableToRatio<MutableWrapper<TDomain, TWrapped>>,
        SettableToRatio<TWrapped, MutableWrapper<TDomain, TWrapped>>, SettableToRatio<MutableWrapper<TDomain, TWrapped>, TWrapped>,
        SettableToPower<TWrapped>, SettableToPower<MutableWrapper<TDomain, TWrapped>>,
        SettableToWeightedSum<TWrapped>, SettableToWeightedSum<MutableWrapper<TDomain, TWrapped>>,
        SettableToWeightedSumExact<TWrapped>, SettableToWeightedSumExact<MutableWrapper<TDomain, TWrapped>>,
        Sampleable<TDomain>
        where TWrapped : 
        IImmutableDistribution<TDomain, TWrapped>,
        CanCreatePartialUniform<TWrapped>,
        CanGetMode<TDomain>,
        CanGetProbLessThan<TDomain>,
        CanGetQuantile<TDomain>,
        CanGetMean<TDomain>,
        CanCopyWithMean<TDomain, TWrapped>,
        CanGetVariance<TDomain>,
        CanGetMeanAndVarianceOut<TDomain, TDomain>,
        CanCopyWithMeanAndVariance<TDomain, TDomain, TWrapped>,
        CanGetLogNormalizer,
        CanGetLogAverageOf<TWrapped>,
        CanGetLogAverageOfPower<TWrapped>,
        CanGetAverageLog<TWrapped>,
        CanEnumerateSupport<TDomain>,
        CanComputeProduct<TWrapped>,
        CanComputeRatio<TWrapped>,
        CanComputePower<TWrapped>,
        Summable<TWrapped>,
        SummableExactly<TWrapped>,
        Sampleable<TDomain>
    {
        #region Data

        [DataMember]
        protected TWrapped WrappedDistribution { get; set; }

        #endregion

        #region Constructors

        protected MutableWrapper(TWrapped wrappedDistribution)
        {
            if (wrappedDistribution == null)
                throw new ArgumentNullException(nameof(wrappedDistribution), "Cannot wrap a null.");

            WrappedDistribution = wrappedDistribution;
        }

        #endregion

        #region object overrides

        public override bool Equals(object obj) => obj is MutableWrapper<TDomain, TWrapped> that && WrappedDistribution.Equals(that.WrappedDistribution);

        public override int GetHashCode() => WrappedDistribution.GetHashCode();

        public override string ToString() => WrappedDistribution.ToString();

        #endregion

        #region IDistribution implementation

        public abstract object Clone();

        public bool IsUniform() => WrappedDistribution.IsUniform();

        public double MaxDiff(object that)
        {
            Argument.CheckIfNotNull(that, nameof(that));
            return that is MutableWrapper<TDomain, TWrapped> thatDist
                ? WrappedDistribution.MaxDiff(thatDist.WrappedDistribution)
                : double.PositiveInfinity;
        }

        public void SetToUniform()
        {
            WrappedDistribution = WrappedDistribution.CreateUniform();
        }

        #endregion

        #region IDistribution<T> implementation

        public TDomain Point
        {
            get => WrappedDistribution.Point;
            set => WrappedDistribution = WrappedDistribution.CreatePointMass(value);
        }

        public bool IsPointMass => WrappedDistribution.IsPointMass;

        public double GetLogProb(TDomain value) => WrappedDistribution.GetLogProb(value);

        #endregion

        #region SettableTo implementation

        public void SetTo(TWrapped value)
        {
            if (value == null)
                throw new ArgumentNullException(nameof(value), "Cannot wrap a null.");

            WrappedDistribution = value;
        }

        public void SetTo(MutableWrapper<TDomain, TWrapped> value)
        {
            WrappedDistribution = value.WrappedDistribution;
        }

        #endregion

        #region SettableToPartialUniform implementation

        public void SetToPartialUniform()
        {
            WrappedDistribution = WrappedDistribution.CreatePartialUniform();
        }

        public void SetToPartialUniformOf(TWrapped dist)
        {
            WrappedDistribution = dist.CreatePartialUniform();
        }

        public bool IsPartialUniform() => WrappedDistribution.IsPartialUniform();

        public void SetToPartialUniformOf(MutableWrapper<TDomain, TWrapped> dist)
        {
            WrappedDistribution = dist.WrappedDistribution.CreatePartialUniform();
        }

        #endregion

        #region CanGetMode implementation

        public TDomain GetMode() => WrappedDistribution.GetMode();

        #endregion

        #region CanGetProbLessThan implementation

        public double GetProbLessThan(TDomain x) => WrappedDistribution.GetProbLessThan(x);

        public double GetProbBetween(TDomain lowerBound, TDomain upperBound) => WrappedDistribution.GetProbBetween(lowerBound, upperBound);

        #endregion

        #region CanGetQuantile implementation

        public TDomain GetQuantile(double probability) => WrappedDistribution.GetQuantile(probability);

        #endregion

        #region CanGetMean implementation

        public TDomain GetMean() => WrappedDistribution.GetMean();

        #endregion

        #region CanSetMean implementation

        public void SetMean(TDomain value)
        {
            WrappedDistribution = WrappedDistribution.WithMean(value);
        }

        #endregion

        #region CanGetVariance implementation

        public TDomain GetVariance() => WrappedDistribution.GetVariance();

        #endregion

        #region CanGetMeanAndVarianceOut implementation

        public void GetMeanAndVariance(out TDomain mean, out TDomain variance) => WrappedDistribution.GetMeanAndVariance(out mean, out variance);

        #endregion

        #region CanSetMeanAndVariance implementation

        public void SetMeanAndVariance(TDomain mean, TDomain variance)
        {
            WrappedDistribution = WrappedDistribution.WithMeanAndVariance(mean, variance);
        }

        #endregion

        #region CanGetLogNormalizer implementation

        public double GetLogNormalizer() => WrappedDistribution.GetLogNormalizer();

        #endregion

        #region CanGetLogAverageOf implementation

        public double GetLogAverageOf(TWrapped that) => WrappedDistribution.GetLogAverageOf(that);

        public double GetLogAverageOf(MutableWrapper<TDomain, TWrapped> that) =>
            WrappedDistribution.GetLogAverageOf(that.WrappedDistribution);

        #endregion

        #region CanGetLogAverageOfPower implementation

        public double GetLogAverageOfPower(TWrapped that, double power) => WrappedDistribution.GetLogAverageOfPower(that, power);

        public double GetLogAverageOfPower(MutableWrapper<TDomain, TWrapped> that, double power) =>
            WrappedDistribution.GetLogAverageOfPower(that.WrappedDistribution, power);

        #endregion

        #region CanGetAverageLog implementation

        public double GetAverageLog(TWrapped that) => WrappedDistribution.GetAverageLog(that);

        public double GetAverageLog(MutableWrapper<TDomain, TWrapped> that) =>
            WrappedDistribution.GetAverageLog(that.WrappedDistribution);

        #endregion

        #region CanEnumerateSupport implementation

        public IEnumerable<TDomain> EnumerateSupport() => WrappedDistribution.EnumerateSupport();

        #endregion

        #region SettableToProduct implementation

        public void SetToProduct(TWrapped a, TWrapped b)
        {
            WrappedDistribution = a.Multiply(b);
        }

        public void SetToProduct(MutableWrapper<TDomain, TWrapped> a, MutableWrapper<TDomain, TWrapped> b)
        {
            WrappedDistribution = a.WrappedDistribution.Multiply(b.WrappedDistribution);
        }

        public void SetToProduct(TWrapped a, MutableWrapper<TDomain, TWrapped> b)
        {
            WrappedDistribution = a.Multiply(b.WrappedDistribution);
        }

        public void SetToProduct(MutableWrapper<TDomain, TWrapped> a, TWrapped b)
        {
            WrappedDistribution = a.WrappedDistribution.Multiply(b);
        }

        #endregion

        #region SettableToRatio implementation

        public void SetToRatio(TWrapped a, TWrapped b, bool forceProper = false)
        {
            WrappedDistribution = a.Divide(b, forceProper);
        }

        public void SetToRatio(MutableWrapper<TDomain, TWrapped> a, MutableWrapper<TDomain, TWrapped> b, bool forceProper = false)
        {
            WrappedDistribution = a.WrappedDistribution.Divide(b.WrappedDistribution, forceProper);
        }

        public void SetToRatio(TWrapped a, MutableWrapper<TDomain, TWrapped> b, bool forceProper = false)
        {
            WrappedDistribution = a.Divide(b.WrappedDistribution, forceProper);
        }

        public void SetToRatio(MutableWrapper<TDomain, TWrapped> a, TWrapped b, bool forceProper = false)
        {
            WrappedDistribution = a.WrappedDistribution.Divide(b, forceProper);
        }

        #endregion

        #region SettableToPower implementation

        public void SetToPower(TWrapped value, double exponent)
        {
            WrappedDistribution = value.Pow(exponent);
        }

        public void SetToPower(MutableWrapper<TDomain, TWrapped> value, double exponent)
        {
            WrappedDistribution = value.WrappedDistribution.Pow(exponent);
        }

        #endregion

        #region SettableToWeightedSum implementation

        public void SetToSum(double weight1, TWrapped value1, double weight2, TWrapped value2)
        {
            WrappedDistribution = value1.Sum(weight1, value2, weight2);
        }

        public void SetToSum(double weight1, MutableWrapper<TDomain, TWrapped> value1, double weight2, MutableWrapper<TDomain, TWrapped> value2)
        {
            WrappedDistribution = value1.WrappedDistribution.Sum(weight1, value2.WrappedDistribution, weight2);
        }

        #endregion

        #region Sampleable implementation

        public TDomain Sample() => WrappedDistribution.Sample();

        public TDomain Sample(TDomain result) => WrappedDistribution.Sample(result);

        #endregion
    }
}

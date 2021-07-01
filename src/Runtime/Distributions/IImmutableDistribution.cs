// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using Microsoft.ML.Probabilistic.Factors.Attributes;
    using Microsoft.ML.Probabilistic.Math;
    using System;

    /// <summary>
    /// Base type for all immutable distributions that doesn't specify over which domain distribution is defined.
    /// </summary>
    /// <remarks>
    /// This interface is useful in generic code where distributions of different types have to be stored
    /// in single container. Container of <see cref="IImmutableDistribution"/> is a more specific type than
    /// container of <see cref="object"/> and adds some type-safety in these cases.
    /// </remarks>
    [Quality(QualityBand.Experimental)]
    public interface IImmutableDistribution : ICloneable, Diffable
    {
    }

    /// <summary>Immutable distribution interface</summary>
    /// <typeparam name="T">The type of objects in the domain, e.g. double or string. Should be an immutable type.</typeparam>
    /// <typeparam name="TThis">The type of the concrete <see cref="IImmutableDistribution{T, TThis}"/> implementation.</typeparam>
    /// <remarks><para>
    /// T should generally have value semantics, i.e. it should override Equals to use value equality.
    /// Otherwise it implies a Distribution over references.
    /// </para><para>
    /// In addition to this interface, Distributions should override Equals to 
    /// use value equality.  
    /// A typical implementation of Equals is: <c>MaxDiff(that) == 0.0</c>
    /// </para><para>
    /// When implementing this interface, its second type parameter should be the type being implemented
    /// itself, e.g. a typical declaration of a type implementing an immutable distribution over double-precision
    /// floating point numbers would look like
    /// <code>public class A : IImmutableDistribution&lt;double, A&gt;</code></para></remarks>
    [Quality(QualityBand.Experimental)]
    public interface IImmutableDistribution<T, TThis> : IImmutableDistribution, CanCreateUniform<TThis>, CanCreatePointMass<T, TThis>, CanGetLogProb<T>
        where TThis : IImmutableDistribution<T, TThis>
    {
    }

    /// <summary>
    /// Whether the distribution can be uniform.
    /// </summary>
    /// <typeparam name="TDist">The type of the distribution.</typeparam>
    [Quality(QualityBand.Experimental)]
    public interface CanCreateUniform<out TDist>
    {
        /// <summary>
        /// Creates a uniform distribution
        /// </summary>
        /// <returns>The created distribution.</returns>
        TDist CreateUniform();

        /// <summary>
        /// Checks whether the current distribution is uniform.
        /// </summary>
        /// <returns><see langword="true"/> if the current distribution is uniform, <see langword="false"/> otherwise.</returns>
        bool IsUniform();
    }

    /// <summary>
    /// Whether the distribution can create another distribution of the same type and with the same support,
    /// but partially uniform over said support.
    /// </summary>
    /// <typeparam name="TDist">The type of the distribution.</typeparam>
    [Quality(QualityBand.Experimental)]
    public interface CanCreatePartialUniform<out TDist>
    {
        /// <summary>
        /// Creates a distribution uniform over the support of the current distribution.
        /// </summary>
        /// <returns>The created distribution.</returns>
        TDist CreatePartialUniform();

        /// <summary>
        /// Checks whether the distribution is uniform over its support.
        /// </summary>
        /// <returns><see langword="true"/> if the distribution is uniform over its support, <see langword="false"/> otherwise.</returns>
        bool IsPartialUniform();
    }

    /// <summary>
    /// Whether the distribution supports being a point mass.
    /// </summary>
    /// <typeparam name="T">The type of objects in the domain, e.g. Vector or Matrix.</typeparam>
    /// <typeparam name="TDist">The type of the distribution.</typeparam>
    [Quality(QualityBand.Experimental)]
    public interface CanCreatePointMass<T, out TDist>
    {
        /// <summary>
        /// Access the location of a point mass.
        /// </summary>
        /// <remarks><para>
        /// If the distribution parameters do not represent a point mass, 
        /// getting this property returns an undefined value (if T is a ValueType) 
        /// or a storage area with undefined contents (if T is a reference type).
        /// It should not throw an exception.
        /// </para><para>
        /// If T is a reference type, then the result is volatile and
        /// its contents expire upon invoking any subsequent distribution method,
        /// unless T is immutable.
        /// </para><para>
        /// Point is a property because it is expected that each distribution would have a 
        /// corresponding data field. Accessing the Point should take constant time.
        /// </para></remarks>
        T Point { get; }

        /// <summary>
        /// Creates a point mass distribution.
        /// </summary>
        /// <param name="point">The location of the point mass.</param>
        /// <returns>The created distribution.</returns>
        TDist CreatePointMass(T point);

        /// <summary>
        /// Gets whether the distribution parameters represent a point mass.
        /// </summary>
        bool IsPointMass { get; }
    }

    /// <summary>
    /// Whether the distribution supports creating copies with different mean value.
    /// </summary>
    /// <typeparam name="MeanType">Type of the mean value.</typeparam>
    /// <typeparam name="TDist">Type of the distribution.</typeparam>
    [Quality(QualityBand.Experimental)]
    public interface CanCopyWithMean<in MeanType, out TDist>
    {
        /// <summary>
        /// Creates a copy of the current distribution with mean set to the provided value.
        /// </summary>
        /// <param name="value">The mean value</param>
        /// <returns>The created distribution.</returns>
        TDist WithMean(MeanType value);
    }

    /// <summary>
    /// Whether the distribution supports creating copies with different mean and variance.
    /// </summary>
    /// <typeparam name="MeanType">Type of the mean value.</typeparam>
    /// <typeparam name="VarType">Type of the variance value.</typeparam>
    /// <typeparam name="TDist">Type of the distribution.</typeparam>
    [Quality(QualityBand.Experimental)]
    public interface CanCopyWithMeanAndVariance<in MeanType, in VarType, out TDist>
    {
        /// <summary>
        /// Creates a copy of the current distribution with mean and variance set to the provided values.
        /// </summary>
        /// <param name="mean">Mean</param>
        /// <param name="variance">Variance</param>
        /// <returns>The created distribution.</returns>
        /// <remarks>In cases where the given mean and variance cannot be matched exactly, the mean
        /// should have priority.</remarks>
        TDist WithMeanAndVariance(MeanType mean, VarType variance);
    }

    /// <summary>
    /// Whether the distribution supports creating copies with all elements set to duplicates of the same value.
    /// </summary>
    /// <typeparam name="T">Type of the element.</typeparam>
    /// <typeparam name="TDist">Type of the distribution.</typeparam>
    [Quality(QualityBand.Experimental)]
    public interface CanCopyWithAllElementsSetTo<in T, out TDist>
    {
        /// <summary>
        /// Creates a copy of the current distribution with all elements set to duplicates of the given value.
        /// </summary>
        /// <param name="value">The value of all elements of the new distribution.</param>
        /// <returns>The created distribution.</returns>
        TDist WithAllElementsSetTo(T value);
    }

    /// <summary>
    /// Supports computing the product of the current instance and another value.
    /// </summary>
    /// <typeparam name="TOther">The type of the second multiplier.</typeparam>
    /// <typeparam name="TResult">The type of the product.</typeparam>
    /// <remarks>Typically the type of the product would be the same as the type of the current instance.</remarks>
    [Quality(QualityBand.Experimental)]
    public interface CanComputeProduct<in TOther, out TResult>
    {
        /// <summary>
        /// Computes the product of the current instance and <paramref name="other"/>.
        /// </summary>
        /// <param name="other">The second multiplier.</param>
        /// <returns>The product.</returns>
        TResult Multiply(TOther other);
    }

    /// <summary>
    /// Supports computing the product of the current instance and another value that has
    /// the same type as the product.
    /// </summary>
    /// <remarks>Typically the type of the second multiplier and the product would be
    /// the same as the type of the current instance.</remarks>
    [Quality(QualityBand.Experimental)]
    public interface CanComputeProduct<T> : CanComputeProduct<T, T>
    {
    }

    /// <summary>
    /// Supports computing the ratio of the current instance and another value.
    /// </summary>
    /// <typeparam name="TDenominator">The type of the denominator.</typeparam>
    /// <typeparam name="TResult">The type of the ratio.</typeparam>
    /// <remarks>Typically the type of the ratio would be the same as the type of the current instance.</remarks>
    [Quality(QualityBand.Experimental)]
    public interface CanComputeRatio<in TDenominator, out TResult>
    {
        /// <summary>
        /// Computes the product of the current instance and <paramref name="denominator"/>.
        /// </summary>
        /// <param name="denominator">The denominator.</param>
        /// <param name="forceProper">If true, the result is modified to have parameters in a safe range</param>
        TResult Divide(TDenominator denominator, bool forceProper = false);
    }

    /// <summary>
    /// Supports computing the ratio of the current instance and another value that has
    /// the same type as the ratio.
    /// </summary>
    /// <remarks>Typically the type of the denominator and the ratio would be
    /// the same as the type of the current instance.</remarks>
    public interface CanComputeRatio<T> : CanComputeRatio<T, T>
    {
    }

    /// <summary>
    /// Supports computing the value of the current instance raised to a power.
    /// </summary>
    /// <typeparam name="TDist">The type of the result.</typeparam>
    /// <remarks>Typically the type of the result would be the same as the type of the current instance.</remarks>
    [Quality(QualityBand.Experimental)]
    public interface CanComputePower<out TDist>
    {
        /// <summary>
        /// Computes the value of the current instance raised to a power.
        /// </summary>
        /// <param name="exponent">The exponent.</param>
        TDist Pow(double exponent);
    }

    /// <summary>
    /// Supports computing the weighted sum of the current instance and another value.
    /// </summary>
    /// <typeparam name="TOther">The type of the second summand.</typeparam>
    /// <typeparam name="TResult">The type of the result.</typeparam>
    /// <remarks>Typically the type of the result would be the same as the type of the current instance.</remarks>
    [Quality(QualityBand.Experimental)]
    public interface Summable<in TOther, out TResult>
    {
        /// <summary>
        /// Computes a distribution with its parameters set to best match a mixture distribution.
        /// </summary>
        /// <param name="weightThis">A finite nonnegative weight.</param>
        /// <param name="other">Second value.</param>
        /// <param name="weightOther">A finite nonnegative weight.</param>
        /// <returns>A distribution of the result type that best matches a mixture distribution.</returns>
        TResult Sum(double weightThis, TOther other, double weightOther);
    }

    /// <summary>
    /// Supports computing the weighted sum of the current instance and another value that has
    /// the same type as the result.
    /// </summary>
    /// <typeparam name="T">The type of the second summand and the result.</typeparam>
    /// <remarks>Typically the type of the second summand and the result would be the same as the type of the current instance.</remarks>
    [Quality(QualityBand.Experimental)]
    public interface Summable<T> : Summable<T, T>
    {
    }

    /// <summary>
    /// Indicates that the computed weighted sum is exact.
    /// </summary>
    /// <typeparam name="TOther">The type of the second summand.</typeparam>
    /// <typeparam name="TResult">The type of the result.</typeparam>
    /// <remarks>Typically the type of the result would be the same as the type of the current instance.</remarks>
    [Quality(QualityBand.Experimental)]
    public interface SummableExactly<in TOther, out TResult> : Summable<TOther, TResult>
    {
    }

    /// <summary>
    /// Indicates that the computed weighted sum is exact.
    /// </summary>
    /// <typeparam name="T">The type of the second summand and the result.</typeparam>
    /// <remarks>Typically the type of the second summand and the result would be the same as the type of the current instance.</remarks>
    [Quality(QualityBand.Experimental)]
    public interface SummableExactly<T> : SummableExactly<T, T>
    {
    }
}

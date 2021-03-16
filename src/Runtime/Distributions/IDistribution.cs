// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Collections.Generic;

    using Math;
    using Factors.Attributes;

    /// <summary>
    /// Base type for all distributions that doesn't specify over which domain distribution is defined.
    /// </summary>
    /// <remarks>
    /// This interface is useful in generic code where distributions of different types have to be stored
    /// in single container. Container of <see cref="IDistribution"/> is a more specific type than
    /// container of <see cref="object"/> and adds some type-safety in these cases.
    /// </remarks>
    [Quality(QualityBand.Mature)]
    public interface IDistribution : ICloneable, Diffable, SettableToUniform
    {
    }

    /// <summary>Distribution interface</summary>
    /// <typeparam name="T">The type of objects in the domain, e.g. Vector or Matrix.</typeparam>
    /// <remarks><para>
    /// T should generally have value semantics, i.e. it should override Equals to use value equality.
    /// Otherwise it implies a Distribution over references.
    /// </para><para>
    /// In addition to this interface, Distributions should override Equals to 
    /// use value equality.  
    /// A typical implementation of Equals is: <c>MaxDiff(that) == 0.0</c>
    /// </para><para>
    /// To be generally used by message-passing algorithms, a distribution should also implement the
    /// following interfaces:
    /// <c>SettableTo, SettableToProduct, SettableToRatio, SettableToPower, SettableToWeightedSum,
    /// CanGetLogAverageOf, CanGetAverageLog</c>
    /// </para></remarks>
    [Quality(QualityBand.Mature)]
    public interface IDistribution<T> : IDistribution, HasPoint<T>, CanGetLogProb<T>
    {
    }

    /// <summary>
    /// Whether the distribution supports evaluation of its density
    /// </summary>
    /// <typeparam name="T">Domain type</typeparam>
    public interface CanGetLogProb<in T>
    {
        /// <summary>
        /// Evaluate the log of the density function at the specified domain value
        /// </summary>
        /// <param name="value">The value at which to calculate the density</param>
        /// <returns>The log density</returns>
        double GetLogProb(T value);
    }

    /// <summary>
    /// Delegate type for evaluating log densities. This is used for distributions such as
    /// <see cref="VectorGaussian"/> which have a large memory footprint. If a distribution
    /// supports <see cref="CanGetLogProbPrep&lt;DistributionType,T&gt;"/>, then it can return a delegate of this type
    /// to do evaluations without recreating a workspace each time.
    /// </summary>
    /// <typeparam name="DistributionType">The distribution type</typeparam>
    /// <typeparam name="T">The domain type of the distribution</typeparam>
    /// <param name="dist">The distribution instance</param>
    /// <param name="value">The value at which to evaluate the log density</param>
    /// <returns>The delegate returns a double</returns>
    public delegate double Evaluator<DistributionType, T>(DistributionType dist, T value);

    /// <summary>
    /// Whether the distribution supports preallocation of a workspace for density evaluation
    /// </summary>
    /// <typeparam name="DistributionType">The distribution type</typeparam>
    /// <typeparam name="T">The domain type of the distribution</typeparam>
    public interface CanGetLogProbPrep<DistributionType, T>
    {
        /// <summary>
        /// Return an evaluator delegate which owns an evaluation workspace
        /// </summary>
        /// <returns>An evaluator delegate</returns>
        Evaluator<DistributionType, T> GetLogProbPrep();
    }

    /// <summary>
    /// Whether the distribution can be set to uniform
    /// </summary>
    public interface SettableToUniform
    {
        /// <summary>
        /// Set the distribution to be uniform
        /// </summary>
        void SetToUniform();

        /// <summary>
        /// Ask whether the distribution instance is uniform
        /// </summary>
        /// <returns>True if uniform</returns>
        bool IsUniform();
    }

    /// <summary>
    /// Whether the distribution can be set to be uniform over the support of another distribution.
    /// </summary>
    public interface SettableToPartialUniform<in TDist>
    {
        /// <summary>
        /// Sets the distribution to be uniform over its support.
        /// </summary>
        void SetToPartialUniform();

        /// <summary>
        /// Sets the distribution to be uniform over the support of a given distribution.
        /// </summary>
        /// <param name="dist">The distribution which support will be used to setup the current distribution.</param>
        void SetToPartialUniformOf(TDist dist);

        /// <summary>
        /// Checks whether the distribution is uniform over its support.
        /// </summary>
        /// <returns>True if the distribution is uniform over its support, false otherwise.</returns>
        bool IsPartialUniform();
    }

    /// <summary>
    /// Whether the distribution supports being a point mass
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface HasPoint<T>
    {
        /// <summary>
        /// Access the location of a point mass.
        /// </summary>
        /// <remarks><para>
        /// If the distribution parameters do not represent a point mass, 
        /// getting this property returns an undefined value (if T is a ValueType) 
        /// or a storage area with undefined contents (if T is a reference type).
        /// It should not throw an exception.
        /// Setting Point will change the distribution into a point mass, overriding any other 
        /// property settings.
        /// Note: Point = Point does have an effect!
        /// </para><para>
        /// If T is a reference type, then the result is volatile.  
        /// Its contents expire upon invoking any subsequent distribution method.
        /// </para><para>
        /// Point is a property because it is expected that each distribution would have a 
        /// corresponding data field.  Accessing the Point should take constant time.
        /// </para></remarks>
        // FIXME: Point.Set should instead be a method SetToPointMass()
        T Point { get; set; }

        /// <summary>
        /// Gets whether the distribution parameters represent a point mass.
        /// </summary>
        bool IsPointMass { get; }
    }

    /// <summary>
    /// Whether the distribution supports sampling
    /// </summary>
    /// <typeparam name="T">The sample type (i.e. the domain type)</typeparam>
    public interface Sampleable<T>
    {
        /// <summary>
        /// Sample the distribution
        /// </summary>
        /// <returns>The sample value</returns>
        T Sample();

        /// <summary>
        /// Sample the distribution and provide a place to put the result.
        /// </summary>
        /// <param name="result">Where to put the result. This is ignored for value-type domains</param>
        /// <returns></returns>
        T Sample(T result);
    }

    /// <summary>
    /// Delegate type for sampling
    /// </summary>
    /// <typeparam name="T">Domain type</typeparam>
    /// <param name="result">Where to put the result</param>
    /// <returns></returns>
    public delegate T Sampler<T>(T result);

    /// <summary>
    /// Delegate type for sampling a distribution. This is used for distributions such as
    /// <see cref="VectorGaussian"/> which have a large memory footprint. If a distribution
    /// supports <see cref="CanSamplePrep&lt;DistributionType,T&gt;"/>, then it can return a delegate of this type
    /// to do successive sampling without recreating a workspace each time.
    /// </summary>
    /// <typeparam name="DistributionType">The distribution type</typeparam>
    /// <typeparam name="T">The domain type of the distribution</typeparam>
    /// <param name="distribution">The distribution instance</param>
    /// <param name="result">The value at which to evaluate the log density</param>
    /// <returns>The delegate's return type is the domain type</returns>
    public delegate T Sampler<DistributionType, T>(DistributionType distribution, T result);

    /// <summary>
    /// Whether the distribution supports preallocation of a workspace for sampling
    /// </summary>
    /// <typeparam name="DistributionType">The distribution type</typeparam>
    /// <typeparam name="T">The domain type of the distribution</typeparam>
    public interface CanSamplePrep<DistributionType, T>
    {
        /// <summary>
        /// Return an sampler delegate which owns an sampling workspace
        /// </summary>
        /// <returns>An sampler delegate</returns>
        Sampler<DistributionType, T> SamplePrep();
    }

    /// <summary>
    /// Whether the distribution supports retrieval of the most probable value
    /// </summary>
    /// <typeparam name="ModeType">Type of the most probable value</typeparam>
    public interface CanGetMode<out ModeType>
    {
        /// <summary>
        /// The most probable value
        /// </summary>
        /// <returns>The most probable value</returns>
        /// <remarks>
        /// This is not a property because it is not reasonable to expect that distributions
        /// would generally have their mode as a data member.  Computing the mode could take a long
        /// time for some distributions.
        /// </remarks>
        ModeType GetMode();
    }

    public interface CanGetProbLessThan<in T>
    {
        /// <summary>
        /// Returns the probability of drawing a sample less than x.
        /// </summary>
        /// <param name="x">Any number.</param>
        /// <returns>A real number in [0,1].</returns>
        double GetProbLessThan(T x);

        /// <summary>
        /// Returns the probability mass in an interval.
        /// </summary>
        /// <param name="lowerBound">Inclusive</param>
        /// <param name="upperBound">Exclusive</param>
        /// <returns>A number between 0 and 1, inclusive.</returns>
        double GetProbBetween(T lowerBound, T upperBound);
    }

    public interface CanGetQuantile<out T>
    {
        /// <summary>
        /// Returns the largest value x such that GetProbLessThan(x) &lt;= probability.
        /// </summary>
        /// <param name="probability">A real number in [0,1].</param>
        /// <returns>A number</returns>
        T GetQuantile(double probability);
    }

    /// <summary>
    /// Whether the distribution supports retrieval of a mean value
    /// </summary>
    /// <typeparam name="MeanType">Type of the mean value</typeparam>
    public interface CanGetMean<out MeanType>
    {
        /// <remarks><para>
        /// </para><para>
        /// This is not a property because it is not reasonable to expect that distributions
        /// would generally have their mean as a data member.  Computing the mean could take a long
        /// time for some distributions.
        /// </para></remarks>
        MeanType GetMean();

        //MeanType GetMean(MeanType result);
    }

    /// <summary>
    /// Whether the distribution supports setting of its mean value
    /// </summary>
    /// <typeparam name="MeanType">Type of the mean value</typeparam>
    public interface CanSetMean<in MeanType>
    {
        /// <summary>
        /// Method to set the mean
        /// </summary>
        /// <param name="value">The mean value</param>
        void SetMean(MeanType value);
    }

    /// <summary>
    /// Whether the distribution supports retrieval of a variance value
    /// </summary>
    /// <typeparam name="VarType">Type of the variance value</typeparam>
    public interface CanGetVariance<out VarType>
    {
        /// <summary>
        /// Method to get the variance
        /// </summary>
        /// <returns>The variance</returns>
        VarType GetVariance();
    }

    /// <summary>
    /// Whether the distribution supports the joint getting of mean and variance
    /// where the mean and variance are returned as 'out' argiments
    /// </summary>
    /// <typeparam name="MeanType">Mean type</typeparam>
    /// <typeparam name="VarType">Variance type</typeparam>
    public interface CanGetMeanAndVarianceOut<MeanType, VarType>
    {
        /// <summary>
        /// Get the mean and variance
        /// </summary>
        /// <param name="mean">Where to put the mean</param>
        /// <param name="variance">Where to put the variance</param>
        void GetMeanAndVariance(out MeanType mean, out VarType variance);
    }

    /// <summary>
    /// Whether the distribution supports the joint getting of mean and variance
    /// where the mean and variance are reference types
    /// </summary>
    /// <typeparam name="MeanType">Mean type</typeparam>
    /// <typeparam name="VarType">Variance type</typeparam>
    public interface CanGetMeanAndVariance<MeanType, VarType>
    {
        /// <summary>
        /// Get the mean and variance
        /// </summary>
        /// <param name="mean">Where to put the mean</param>
        /// <param name="variance">Where to put the variance</param>
        void GetMeanAndVariance(MeanType mean, VarType variance);
    }

    /// <summary>
    /// Whether the distribution supports the joint setting of mean and variance
    /// </summary>
    /// <typeparam name="MeanType"></typeparam>
    /// <typeparam name="VarType"></typeparam>
    public interface CanSetMeanAndVariance<in MeanType, in VarType>
    {
        /// <summary>
        /// Set the parameters to produce a given mean and variance.
        /// </summary>
        /// <param name="mean">Mean</param>
        /// <param name="variance">Variance</param>
        /// <remarks>In cases where the given mean and variance cannot be matched exactly, the mean
        /// should have priority.</remarks>
        void SetMeanAndVariance(MeanType mean, VarType variance);
    }

    /// <summary>
    /// Whether the distribution can compute its normalizer.
    /// </summary>
    public interface CanGetLogNormalizer
    {
        /// <summary>
        /// The logarithm of the distribution's normalizer, i.e. the integral of its minimal exponential-family representation.
        /// </summary>
        /// <returns></returns>
        double GetLogNormalizer();
    }

    /// <summary>
    /// Whether the distribution can compute the expectation of another distribution's value.
    /// </summary>
    /// <typeparam name="T">The other distribution type</typeparam>
    public interface CanGetLogAverageOf<in T>
    {
        /// <summary>
        /// The log-probability that two distributions would draw the same sample.
        /// </summary>
        /// <param name="that"></param>
        /// <returns><c>Math.Log(sum_x this.Evaluate(x)*that.Evaluate(x))</c></returns>
        /// <remarks>This can be considered a type of inner product between distributions.
        /// </remarks>
        double GetLogAverageOf(T that);
    }

    /// <summary>
    /// Whether the distribution can compute the expectation of another distribution raised to a power.
    /// </summary>
    /// <typeparam name="T">The distribution type</typeparam>
    public interface CanGetLogAverageOfPower<in T>
    {
        /// <summary>
        /// The log-integral of one distribution times another raised to a power.
        /// </summary>
        /// <param name="that">The other distribution</param>
        /// <param name="power">The exponent</param>
        /// <returns><c>Math.Log(sum_x this.Evaluate(x)*Math.Pow(that.Evaluate(x),power))</c></returns>
        /// <remarks>This is not the same as GetLogAverageOf(that^power) because it includes the normalization constant of that.
        /// </remarks>
        double GetLogAverageOfPower(T that, double power);
    }

    /// <summary>
    /// Whether the distribution supports the expected logarithm of one instance under another
    /// </summary>
    /// <typeparam name="T">The distribution type</typeparam>
    public interface CanGetAverageLog<in T>
    {
        /// <summary>
        /// The expected logarithm of that distribution under this distribution.
        /// </summary>
        /// <param name="that">The distribution to take the logarithm of.</param>
        /// <returns><c>sum_x this.Evaluate(x)*Math.Log(that.Evaluate(x))</c></returns>
        /// <remarks>This is also known as the cross entropy.</remarks>
        double GetAverageLog(T that);
    }

    /// <summary>
    /// Whether the distribution supports enumeration over the support - i.e. enumeration
    /// over the domain values with non-zero mass.
    /// </summary>
    /// <typeparam name="T">The domain type.</typeparam>
    public interface CanEnumerateSupport<T>
    {
        /// <summary>
        /// Enumerates over the support of the distribution instance.
        /// </summary>
        /// <returns>The domain values with non-zero mass.</returns>
        IEnumerable<T> EnumerateSupport();
    }
}
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Math
{
    /// <summary>
    /// Supports setting an instance to a value
    /// </summary>
    /// <typeparam name="T">The type of value</typeparam>
    public interface SettableTo<in T>
    {
        /// <summary>
        /// Set the instance to have the same value as given
        /// </summary>
        /// <param name="value">The value to set to</param>
        void SetTo(T value);
    }

    /// <summary>
    /// Supports setting all elements to duplicates of the same value
    /// </summary>
    /// <typeparam name="T">The type</typeparam>
    public interface CanSetAllElementsTo<in T>
    {
        /// <summary>
        /// Set all elements to duplicates of the given value
        /// </summary>
        /// <param name="value">The value to set to</param>
        void SetAllElementsTo(T value);
    }

    /// <summary>
    /// Supports setting an instance to the product of
    /// two values of different types
    /// </summary>
    /// <typeparam name="T">The first type</typeparam>
    /// <typeparam name="U">The second type</typeparam>
    public interface SettableToProduct<in T, in U>
    {
        /// <summary>
        /// Set this to the product of a and b
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        void SetToProduct(T a, U b);
    }

    /// <summary>
    /// Supports setting an instance to the product of two values of the same type
    /// </summary>
    /// <typeparam name="T">The type</typeparam>
    public interface SettableToProduct<in T> : SettableToProduct<T, T>
    {
    }

    /// <summary>
    /// Supports setting an instance to the ratio of
    /// two values of different types
    /// </summary>
    /// <typeparam name="T">The first type</typeparam>
    /// <typeparam name="U">The second type</typeparam>
    public interface SettableToRatio<in T, in U>
    {
        /// <summary>
        /// Set this to the ratio of a and b
        /// </summary>
        /// <param name="numerator"></param>
        /// <param name="denominator"></param>
        /// <param name="forceProper">If true, the result is modified to have parameters in a safe range</param>
        void SetToRatio(T numerator, U denominator, bool forceProper = false);
    }

    /// <summary>
    /// Supports setting an instance to the ratio of
    /// two values of the same type
    /// </summary>
    /// <typeparam name="T">The type</typeparam>
    public interface SettableToRatio<in T> : SettableToRatio<T, T>
    {
    }

    /// <summary>
    /// Supports setting an instance to a value raised to a power
    /// </summary>
    /// <typeparam name="T">The value type</typeparam>
    public interface SettableToPower<in T>
    {
        /// <summary>
        /// Set this to the given value raised to the given power
        /// </summary>
        /// <param name="value"></param>
        /// <param name="exponent"></param>
        void SetToPower(T value, double exponent);
    }

    /// <summary>
    /// Supports setting an instance to the weighted sum of
    /// two values of the same type
    /// </summary>
    /// <typeparam name="T">The type</typeparam>
    public interface SettableToWeightedSum<in T>
    {
        /// <summary>
        /// Set the parameters to best match a mixture distribution.
        /// </summary>
        /// <param name="weight1">A finite nonnegative weight.</param>
        /// <param name="weight2">A finite nonnegative weight.</param>
        /// <param name="value1">First value.  Can be the same object as <c>this</c></param>
        /// <param name="value2">Second value</param>
        void SetToSum(double weight1, T value1, double weight2, T value2);
    }

    /// <summary>
    /// Indicates that a distribution can represent weighted sum
    /// of distributions of type <typeparamref name="T"/> exactly.
    /// </summary>
    /// <typeparam name="T">The type of the distributions which sum can be represented exactly.</typeparam>
    public interface SettableToWeightedSumExact<in T> : SettableToWeightedSum<T>
    {
    }

    /// <summary>
    /// Supports calculating the maximum difference between
    /// this instance and another object (not necessarily of the same type)
    /// </summary>
    public interface Diffable
    {
        /// <summary>
        /// The maximum difference between this instance and the given
        /// </summary>
        /// <param name="that"></param>
        /// <returns></returns>
        double MaxDiff(object that);
    }
}
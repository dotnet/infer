// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Diagnostics;
    using System.Globalization;
    using System.Runtime.Serialization;

    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Serialization;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// Represent an element of the automata weight semiring.
    /// </summary>
    /// <remarks>
    /// Weights are stored in log-domain so that weights of tiny magnitude can be handled without the loss of precision.
    /// </remarks>
    [Serializable]
    [DataContract]
    public struct Weight
    {
        /// <summary>
        /// The logarithm of the weight value.
        /// </summary>
        [DataMember]
        private readonly double logValue;

        /// <summary>
        /// Initializes a new instance of the <see cref="Weight"/> struct.
        /// </summary>
        /// <param name="logValue">The logarithm of the weight value.</param>
        private Weight(double logValue)
        {
            Debug.Assert(!double.IsNaN(logValue), "A valid weight must be given.");

            this.logValue = logValue;
        }

        /// <summary>
        /// Gets the zero weight.
        /// </summary>
        public static Weight Zero => new Weight(double.NegativeInfinity);

        /// <summary>
        /// Gets the unit weight.
        /// </summary>
        public static Weight One => new Weight(0);

        /// <summary>
        /// Gets the infinite weight.
        /// </summary>
        public static Weight Infinity => new Weight(double.PositiveInfinity);

        /// <summary>
        /// Gets the logarithm of the weight value.
        /// </summary>
        public double LogValue => this.logValue;

        /// <summary>
        /// Gets the weight value.
        /// </summary>
        public double Value => Math.Exp(this.LogValue);

        /// <summary>
        /// Gets a value indicating whether the weight is zero.
        /// </summary>
        public bool IsZero => double.IsNegativeInfinity(this.LogValue);

        /// <summary>
        /// Creates a weight from the logarithm of its value.
        /// </summary>
        /// <param name="logValue">The logarithm of the weight value.</param>
        /// <returns>The created weight.</returns>
        [Construction("LogValue")]
        public static Weight FromLogValue(double logValue)
        {
            Argument.CheckIfValid(!double.IsNaN(logValue), "logValue", "A valid weight value must be provided.");
            
            return new Weight(logValue);
        }

        /// <summary>
        /// Creates a weight from its value.
        /// </summary>
        /// <param name="value">The weight value.</param>
        /// <returns>The created weight.</returns>
        public static Weight FromValue(double value)
        {
            Argument.CheckIfValid(!double.IsNaN(value), "value", "A valid weight value must be provided.");
            Argument.CheckIfValid(value >= 0, "value", "Weight values must not be negative.");
            
            return new Weight(Math.Log(value));
        }

        /// <summary>
        /// Compute the sum of given weights.
        /// </summary>
        /// <param name="weight1">The first weight.</param>
        /// <param name="weight2">The second weight.</param>
        /// <returns>The computed sum.</returns>
        public static Weight Sum(Weight weight1, Weight weight2)
        {
            return new Weight(MMath.LogSumExp(weight1.LogValue, weight2.LogValue));
        }

        /// <summary>
        /// Compute the difference of given weights.
        /// </summary>
        /// <param name="weight1">The first weight.</param>
        /// <param name="weight2">The second weight.</param>
        /// <returns>The computed difference.</returns>
        public static Weight AbsoluteDifference(Weight weight1, Weight weight2)
        {
            double min = Math.Min(weight1.LogValue, weight2.LogValue);
            double max = Math.Max(weight1.LogValue, weight2.LogValue);
            return new Weight(MMath.LogDifferenceOfExp(max, min));
        }

        /// <summary>
        /// Compute the product of given weights.
        /// </summary>
        /// <param name="weight1">The first weight.</param>
        /// <param name="weight2">The second weight.</param>
        /// <returns>The computed product.</returns>
        public static Weight Product(Weight weight1, Weight weight2)
        {
            if (weight1.IsZero || weight2.IsZero)
            {
                return Zero;
            }
            
            return new Weight(weight1.LogValue + weight2.LogValue);
        }

        /// <summary>
        /// Compute the product of given weights.
        /// </summary>
        /// <param name="weight1">The first weight.</param>
        /// <param name="weight2">The second weight.</param>
        /// <param name="weight3">The third weight.</param>
        /// <returns>The computed product.</returns>
        public static Weight Product(Weight weight1, Weight weight2, Weight weight3)
        {
            if (weight1.IsZero || weight2.IsZero || weight3.IsZero)
            {
                return Zero;
            }

            return new Weight(weight1.LogValue + weight2.LogValue + weight3.LogValue);
        }

        /// <summary>
        /// Compute the product of given weights.
        /// </summary>
        /// <param name="weights">The weights.</param>
        /// <returns>The computed product.</returns>
        public static Weight Product(params Weight[] weights)
        {
            double resultLogValue = 0;
            for (int i = 0; i < weights.Length; ++i)
            {
                if (weights[i].IsZero)
                {
                    return Weight.Zero;
                }

                resultLogValue += weights[i].LogValue;
            }

            return new Weight(resultLogValue);
        }

        /// <summary>
        /// Computes the inverse of a given weight.
        /// </summary>
        /// <param name="weight">The weight.</param>
        /// <returns>The inverse <c>I</c> such that <paramref name="weight"/>*I=1.</returns>
        public static Weight Inverse(Weight weight)
        {
            return new Weight(-weight.LogValue);
        }

        /// <summary>
        /// Computes the sum of a geometric series <c>1 + w + w^2 + w^3 + ...</c>,
        /// where <c>w</c> is a given weight.
        /// </summary>
        /// <param name="weight">The weight.</param>
        /// <returns>The computed sum, or <see cref="Infinity"/> if the sum diverges.</returns>
        public static Weight Closure(Weight weight)
        {
            const double Eps = 1e-20;
            if (weight.LogValue < -Eps)
            {
                // The series converges
                return new Weight(-MMath.Log1MinusExp(weight.logValue));
            }

            // The series diverges
            return Weight.Infinity;
        }

        /// <summary>
        /// Computes the sum of a geometric series <c>1 + w + w^2 + w^3 + ...</c>,
        /// where <c>w</c> is a given weight.
        /// If the sum diverges, replaces the infinite sum by a finite sum with a lot of terms.
        /// </summary>
        /// <param name="weight">The weight.</param>
        /// <returns>The computed sum.</returns>
        public static Weight ApproximateClosure(Weight weight)
        {
            const double Eps = 1e-20;
            const double TermCount = 10000;

            if (weight.LogValue < -Eps)
            {
                // The series converges
                return new Weight(-MMath.Log1MinusExp(weight.LogValue));
            }

            if (weight.LogValue < Eps)
            {
                // The series diverges, geometric progression formula does not apply
                return new Weight(Math.Log(TermCount));
            }

            // Compute geometric progression with a lot of terms
            return new Weight(MMath.LogExpMinus1(weight.LogValue * TermCount) - MMath.LogExpMinus1(weight.LogValue));
        }

        /// <summary>
        /// The weight equality operator.
        /// </summary>
        /// <param name="weight1">The first weight.</param>
        /// <param name="weight2">The second weight.</param>
        /// <returns>
        /// <see langword="true"/> if the given weights are equal,
        /// <see langword="false"/> otherwise.
        /// </returns>
        public static bool operator ==(Weight weight1, Weight weight2)
        {
            return weight1.logValue == weight2.logValue;
        }

        /// <summary>
        /// The weight inequality operator.
        /// </summary>
        /// <param name="weight1">The first weight.</param>
        /// <param name="weight2">The second weight.</param>
        /// <returns>
        /// <see langword="true"/> if the given weights are not equal,
        /// <see langword="false"/> otherwise.
        /// </returns>
        public static bool operator !=(Weight weight1, Weight weight2)
        {
            return !(weight1 == weight2);
        }

        /// <summary>
        /// Compute the product of given weights.
        /// </summary>
        /// <param name="weight1">The first weight.</param>
        /// <param name="weight2">The second weight.</param>
        /// <returns>The computed product.</returns>
        public static Weight operator *(Weight weight1, Weight weight2) => Product(weight1, weight2);

        /// <summary>
        /// Compute the sum of given weights.
        /// </summary>
        /// <param name="weight1">The first weight.</param>
        /// <param name="weight2">The second weight.</param>
        /// <returns>The computed sum.</returns>
        public static Weight operator +(Weight weight1, Weight weight2) => Sum(weight1, weight2);

        /// <summary>
        /// Checks if this instance is equal to a given object.
        /// </summary>
        /// <param name="obj">The object.</param>
        /// <returns>
        /// <see langword="true"/> if this instance is equal to <paramref name="obj"/>,
        /// <see langword="false"/> otherwise.
        /// </returns>
        public override bool Equals(object obj)
        {
            if (obj == null || GetType() != obj.GetType())
            {
                return false;
            }

            return this == (Weight)obj;
        }

        /// <summary>
        /// Gets the hash code of this instance.
        /// </summary>
        /// <returns>The hash code.</returns>
        public override int GetHashCode()
        {
            return this.logValue.GetHashCode();
        }

        /// <summary>
        /// Gets a string representation of this object.
        /// </summary>
        /// <returns>A string representation of this object.</returns>
        public override string ToString()
        {
            return this.ToString(CultureInfo.CurrentCulture);
        }

        /// <summary>
        /// Gets a string representation of this object.
        /// </summary>
        /// <param name="provider">A format provider.</param>
        /// <returns>A string representation of this object.</returns>
        public string ToString(IFormatProvider provider)
        {
            return this.Value.ToString(provider);
        }


        /// <summary>
        /// Writes the weight.
        /// </summary>
        public void Write(Action<double> writeDouble) => writeDouble(this.logValue);

        /// <summary>
        /// Reads a weight.
        /// </summary>
        public static Weight Read(Func<double> readDouble) => new Weight(readDouble());
    }
}

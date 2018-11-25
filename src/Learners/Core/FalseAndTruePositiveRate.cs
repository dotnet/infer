// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// Struct which holds both the FPR and TPR
    /// </summary>
    public struct FalseAndTruePositiveRate
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="FalseAndTruePositiveRate"/> struct.
        /// </summary>
        /// <param name="falsePositiveRate">The false positive rate (FPR)</param>
        /// <param name="truePositiveRate">The true positive rate (TPR)</param>
        public FalseAndTruePositiveRate(double falsePositiveRate, double truePositiveRate)
            : this()
        {
            this.FalsePositiveRate = falsePositiveRate;
            this.TruePositiveRate = truePositiveRate;
        }

        /// <summary>
        /// Gets the <see cref="FalsePositiveRate"/>.
        /// </summary>
        public readonly double FalsePositiveRate;

        /// <summary>
        /// Gets the <see cref="TruePositiveRate"/>
        /// </summary>
        public readonly double TruePositiveRate;

        /// <summary>
        /// Gets the string representation of this <see cref="FalseAndTruePositiveRate"/>.
        /// </summary>
        /// <returns>The string representation of the <see cref="FalseAndTruePositiveRate"/>.</returns>
        public override string ToString()
        {
            return $"{this.FalsePositiveRate}, {this.TruePositiveRate}";
        }

        /// <summary>
        /// Checks if this object is equal to <paramref name="obj"/>.
        /// </summary>
        /// <param name="obj">The object to compare this object with.</param>
        /// <returns>
        /// <see langword="true"/> if this object is equal to <paramref name="obj"/>,
        /// <see langword="false"/> otherwise.
        /// </returns>
        public override bool Equals(object obj)
        {
            if (obj is FalseAndTruePositiveRate receiverOperatingCharacteristic)
            {
                return object.Equals(this.FalsePositiveRate, receiverOperatingCharacteristic.FalsePositiveRate) && object.Equals(this.TruePositiveRate, receiverOperatingCharacteristic.TruePositiveRate);
            }

            return false;
        }
        /// <summary>
        /// Computes the hash code of this object.
        /// </summary>
        /// <returns>The computed hash code.</returns>
        public override int GetHashCode()
        {
            int result = Hash.Start;

            result = Hash.Combine(result, this.FalsePositiveRate.GetHashCode());
            result = Hash.Combine(result, this.TruePositiveRate.GetHashCode());

            return result;
        }
    }
}
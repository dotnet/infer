// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// Struct which holds the precision and recall
    /// </summary>
    public struct PrecisionRecall
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="PrecisionRecall"/> struct.
        /// </summary>
        /// <param name="precision">The precision (positive predictive value)</param>
        /// <param name="recall">The recall (sensitivity)</param>
        public PrecisionRecall(double precision, double recall)
            : this()
        {
            this.Precision = precision;
            this.Recall = recall;
        }

        /// <summary>
        /// Gets the <see cref="Precision"/>
        /// </summary>
        public readonly double Precision;

        /// <summary>
        /// Gets the <see cref="Recall"/>.
        /// </summary>
        public readonly double Recall;

        /// <summary>
        /// Gets the string representation of this <see cref="PrecisionRecall"/>.
        /// </summary>
        /// <returns>The string representation of the <see cref="PrecisionRecall"/>.</returns>
        public override string ToString()
        {
            return $"{this.Precision}, {this.Recall}";
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
            if (obj is PrecisionRecall precisionRecall)
            {
                return object.Equals(this.Precision, precisionRecall.Precision) && object.Equals(this.Recall, precisionRecall.Recall);
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

            result = Hash.Combine(result, this.Precision.GetHashCode());
            result = Hash.Combine(result, this.Recall.GetHashCode());

            return result;
        }
    }
}
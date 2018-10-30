// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// Struct which holds empirical and predicted probabilities for use in calibration.
    /// </summary>
    public struct CalibrationPair
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="CalibrationPair"/> struct.
        /// </summary>
        /// <param name="empiricalProbability">The empirical probability</param>
        /// <param name="predictedProbability">The predicted probability</param>
        public CalibrationPair(double empiricalProbability, double predictedProbability)
            : this()
        {
            this.EmpiricalProbability = empiricalProbability;
            this.PredictedProbability = predictedProbability;
        }

        /// <summary>
        /// Gets the empirical probability.
        /// </summary>
        public readonly double EmpiricalProbability;

        /// <summary>
        /// Gets the predicted probability.
        /// </summary>
        public readonly double PredictedProbability;
        
        /// <summary>
        /// Gets the string representation of this <see cref="CalibrationPair"/>.
        /// </summary>
        /// <returns>The string representation of the <see cref="CalibrationPair"/>.</returns>
        public override string ToString()
        {
            return $"{this.EmpiricalProbability}, {this.PredictedProbability}";
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
            if (obj is CalibrationPair calibrationPair)
            {
                return object.Equals(this.EmpiricalProbability, calibrationPair.EmpiricalProbability) && object.Equals(this.PredictedProbability, calibrationPair.PredictedProbability);
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

            result = Hash.Combine(result, this.EmpiricalProbability.GetHashCode());
            result = Hash.Combine(result, this.PredictedProbability.GetHashCode());
            
            return result;
        }
    }
}
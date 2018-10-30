// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Collections
{
    using Microsoft.ML.Probabilistic.Utilities;

    public struct EmpiricalProbabilityCalibrationPoint
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="EmpiricalProbabilityCalibrationPoint"/> struct.
        /// </summary>
        /// <param name="empiricalProbability">The empirical probability</param>
        /// <param name="predictedProbability">The predicted probability</param>
        public EmpiricalProbabilityCalibrationPoint(double empiricalProbability, double predictedProbability)
            : this()
        {
            this.EmpiricalProbability = empiricalProbability;
            this.PredictedProbability = predictedProbability;
        }

        /// <summary>
        /// Gets or sets the empirical probability.
        /// </summary>
        public readonly double EmpiricalProbability;

        /// <summary>
        /// Gets or sets the predicted probability.
        /// </summary>
        public readonly double PredictedProbability;
        
        /// <summary>
        /// Gets the string representation of this <see cref="EmpiricalProbabilityCalibrationPoint"/>.
        /// </summary>
        /// <returns>The string representation of the <see cref="EmpiricalProbabilityCalibrationPoint"/>.</returns>
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
            if (obj is EmpiricalProbabilityCalibrationPoint calibrationPoint)
            {
                return object.Equals(this.EmpiricalProbability, calibrationPoint.EmpiricalProbability) && object.Equals(this.PredictedProbability, calibrationPoint.PredictedProbability);
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
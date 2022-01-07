// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using Math;
    using Factors.Attributes;

    /// <summary>
    /// A discrete distribution over the values of an enum.
    /// </summary>
    /// <typeparam name="TEnum"></typeparam>
    [Quality(QualityBand.Preview)]
    public class DiscreteEnum<TEnum> : GenericDiscreteBase<TEnum, DiscreteEnum<TEnum>>
    {
        private static Array values = Enum.GetValues(typeof (TEnum));

        /// <summary>
        /// Creates a uniform distribution over the enum values.
        /// </summary>
        public DiscreteEnum() :
            base(values.Length, Sparsity.Dense)
        {
        }

        /// <summary>
        /// Creates a distribution over the enum values using the specified probabilities.
        /// </summary>
        public DiscreteEnum(params double[] probs) :
            base(values.Length, Sparsity.Dense)
        {
            disc.SetProbs(Vector.FromArray(probs));
        }

        /// <summary>
        /// Creates a distribution over the enum values using the specified probabilities.
        /// </summary>
        public DiscreteEnum(Vector probs) :
            base(values.Length, Sparsity.Dense)
        {
            disc.SetProbs(probs);
        }

        /// <summary>
        /// Converts from an integer to an enum value
        /// </summary>
        /// <param name="i"></param>
        /// <returns></returns>
        public override TEnum ConvertFromInt(int i)
        {
            return (TEnum) values.GetValue(i);
        }

        /// <summary>
        /// Converts the enum value to an integer
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public override int ConvertToInt(TEnum value)
        {
            return (int) (object) value;
        }
    }
}
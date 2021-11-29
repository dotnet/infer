// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;
    using System.IO;
    using System.Runtime.Serialization;

    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// Settings of the Matchbox recommender which affect prediction.
    /// </summary>
    /// <remarks>
    /// These settings can be modified after training.
    /// </remarks>
    [Serializable]
    public class MatchboxRecommenderPredictionSettings : ICustomSerializable
    {
        /// <summary>
        /// The default loss function.
        /// </summary>
        public const LossFunction LossFunctionDefault = LossFunction.ZeroOne;

        /// <summary>
        /// The current custom binary serialization version of the <see cref="MatchboxRecommenderPredictionSettings"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// The custom binary serialization <see cref="Guid"/> of the <see cref="MatchboxRecommenderPredictionSettings"/> class.
        /// </summary>
        private readonly Guid customSerializationGuid = new Guid("E652AC97-69F3-4752-B5BE-8C7198106606");

        /// <summary>
        /// The loss function applied when converting an uncertain prediction into a point prediction.
        /// </summary>
        private LossFunction lossFunction;

        /// <summary>
        /// The custom loss function provided by the user.
        /// </summary>
        [NonSerialized]
        private Func<int, int, double> customLossFunction;

        /// <summary>
        /// Initializes a new instance of the <see cref="MatchboxRecommenderPredictionSettings"/> class.
        /// </summary>
        public MatchboxRecommenderPredictionSettings()
        {
            this.lossFunction = LossFunctionDefault;
            this.customLossFunction = null;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="MatchboxRecommenderPredictionSettings"/> class
        /// from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the prediction settings from.</param>
        public MatchboxRecommenderPredictionSettings(IReader reader)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            reader.VerifySerializationGuid(
                this.customSerializationGuid, "The binary stream does not contain the prediction settings of an Infer.NET Matchbox recommender.");

            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);
            if (deserializedVersion == CustomSerializationVersion)
            {
                // Note that a customLossFunction may need to be reset by the user.
                string lossFunctionString = reader.ReadString();
                if (!Enum.TryParse(lossFunctionString, false, out this.lossFunction))
                {
                    throw new SerializationException("Unknown loss function '" + lossFunctionString + "'.");
                }
            }
        }

        /// <summary>
        /// Sets the loss function which determines how a prediction in the form of a distribution is converted into a point prediction.
        /// </summary>
        /// <param name="lossFunction">The loss function.</param>
        /// <param name="customLossFunction">
        /// An optional custom loss function. This can only be set when <paramref name="lossFunction"/> is set to 'Custom'. 
        /// The custom loss function returns the loss incurred when choosing an estimate instead of the true value, 
        /// where the first argument is the true value and the second argument is the estimate of the true value.
        /// </param>
        public void SetPredictionLossFunction(LossFunction lossFunction, Func<int, int, double> customLossFunction = null)
        {
            if (lossFunction == LossFunction.Custom)
            {
                if (customLossFunction == null)
                {
                    throw new ArgumentNullException(nameof(customLossFunction));
                }
            }
            else
            {
                if (customLossFunction != null)
                {
                    throw new InvalidOperationException("Loss function must be set to '" + LossFunction.Custom + "' when providing a custom loss function.");
                }
            }

            this.customLossFunction = customLossFunction;
            this.lossFunction = lossFunction;
        }

        /// <summary>
        /// Gets the loss function which determines how a prediction in the form of a distribution is converted into a point prediction.
        /// </summary>
        /// <param name="customLossFunction">
        /// The custom loss function on integers. This is <c>null</c> unless the returned <see cref="LossFunction"/> is 'Custom'. 
        /// </param>
        /// <returns>The <see cref="LossFunction"/>.</returns>
        /// <remarks>
        /// A loss function returns the loss incurred when choosing an estimate instead of the true value, 
        /// where the first argument is the true value and the second argument is the estimate of the true value.
        /// </remarks>
        public LossFunction GetPredictionLossFunction(out Func<int, int, double> customLossFunction)
        {
            customLossFunction = this.customLossFunction;
            return this.lossFunction;
        }

        /// <summary>
        /// Saves the prediction settings of the Matchbox recommender using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the prediction settings to.</param>
        public void SaveForwardCompatible(IWriter writer)
        {
            writer.Write(this.customSerializationGuid);
            writer.Write(CustomSerializationVersion);
            writer.Write(this.lossFunction.ToString());
        }
    }
}

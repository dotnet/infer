// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;
    using System.IO;

    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// Settings of the Matchbox recommender (settable by the developer).
    /// </summary>
    [Serializable]
    public class MatchboxRecommenderSettings : ISettings, ICustomSerializable
    {
        /// <summary>
        /// The current custom binary serialization version of the <see cref="MatchboxRecommenderSettings"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// The custom binary serialization <see cref="Guid"/> of the <see cref="MatchboxRecommenderSettings"/> class.
        /// </summary>
        private readonly Guid customSerializationGuid = new Guid("E59E452A-B2FB-46C8-87AE-D96A6D7234A0");

        /// <summary>
        /// Initializes a new instance of the <see cref="MatchboxRecommenderSettings"/> class. 
        /// </summary>
        /// <param name="isTrained">Indicates whether the Matchbox recommender is trained.</param>
        public MatchboxRecommenderSettings(Func<bool> isTrained)
        {
            this.Training = new MatchboxRecommenderTrainingSettings(isTrained);
            this.Prediction = new MatchboxRecommenderPredictionSettings();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="MatchboxRecommenderSettings"/> class
        /// from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the settings of the Matchbox recommender from.</param>
        /// <param name="isTrained">Indicates whether the Matchbox recommender is trained.</param>
        public MatchboxRecommenderSettings(IReader reader, Func<bool> isTrained)
        {
            reader.VerifySerializationGuid(
                this.customSerializationGuid, "The binary stream does not contain the settings of an Infer.NET Matchbox recommender.");

            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);
            if (deserializedVersion == CustomSerializationVersion)
            {
                this.Training = new MatchboxRecommenderTrainingSettings(reader, isTrained);
                this.Prediction = new MatchboxRecommenderPredictionSettings(reader);
            }
        }

        /// <summary>
        /// Gets the settings of the Matchbox recommender which affect training.
        /// </summary>
        public MatchboxRecommenderTrainingSettings Training { get; private set; }

        /// <summary>
        /// Gets the settings of the Matchbox recommender which affect prediction.
        /// </summary>
        public MatchboxRecommenderPredictionSettings Prediction { get; private set; }

        /// <summary>
        /// Saves the settings of the Matchbox recommender using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the settings to.</param>
        public void SaveForwardCompatible(IWriter writer)
        {
            writer.Write(this.customSerializationGuid);
            writer.Write(CustomSerializationVersion);
            this.Training.SaveForwardCompatible(writer);
            this.Prediction.SaveForwardCompatible(writer);
        }
    }
}
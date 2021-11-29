// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;
    using System.IO;

    using Microsoft.ML.Probabilistic.Utilities;
    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// Settings of the Matchbox recommender which affect training.
    /// Cannot be set after training.
    /// </summary>
    [Serializable]
    public class MatchboxRecommenderTrainingSettings : ICustomSerializable
    {
        #region Default values

        /// <summary>
        /// The default value indicating whether user features will be used.
        /// </summary>
        public const bool UseUserFeaturesDefault = false;

        /// <summary>
        /// The default value indicating whether item features will be used.
        /// </summary>
        public const bool UseItemFeaturesDefault = false;

        /// <summary>
        /// The default number of traits.
        /// </summary>
        public const int TraitCountDefault = 4;

        /// <summary>
        /// The default number of inference iterations.
        /// </summary>
        public const int IterationCountDefault = 20;

        /// <summary>
        /// The default number of data batches.
        /// </summary>
        public const int BatchCountDefault = 1;

        /// <summary>
        /// The default value indicating whether shared user thresholds will be used.
        /// </summary>
        public const bool UseSharedUserThresholdsDefault = false;

        #endregion

        #region Constants and fields

        /// <summary>
        /// The current custom binary serialization version of the <see cref="MatchboxRecommenderTrainingSettings"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 2;

        /// <summary>
        /// The custom binary serialization <see cref="Guid"/> of the <see cref="MatchboxRecommenderTrainingSettings"/> class.
        /// </summary>
        private readonly Guid customSerializationGuid = new Guid("E6115B64-3C64-4DD8-B766-F497CFE562AF");

        /// <summary>
        /// Guards the training Matchbox recommender settings from being changed after training.
        /// </summary>
        private readonly SettingsGuard trainingSettingsGuard;

        /// <summary>
        /// Indicates whether to use explicit user features.
        /// </summary>
        private bool useUserFeatures;

        /// <summary>
        /// Indicates whether to use explicit item features.
        /// </summary>
        private bool useItemFeatures;

        /// <summary>
        /// The number of implicit user or item features (traits) to learn.
        /// </summary>
        private int traitCount;

        /// <summary>
        /// The number of batches the training data is split into.
        /// </summary>
        private int batchCount;

        /// <summary>
        /// The number of inference iterations to run.
        /// </summary>
        private int iterationCount;

        /// <summary>
        /// Indicates whether to use shared user thresholds.
        /// </summary>
        private bool useSharedUserThresholds;

        #endregion

        #region Constructors

        /// <summary>
        /// Initializes a new instance of the <see cref="MatchboxRecommenderTrainingSettings"/> class.
        /// </summary>
        /// <param name="isTrained">Indicates whether the Matchbox recommender is trained.</param>
        internal MatchboxRecommenderTrainingSettings(Func<bool> isTrained)
        {
            this.trainingSettingsGuard = new SettingsGuard(isTrained, "This setting cannot be changed after training.");

            this.Advanced = new MatchboxRecommenderAdvancedTrainingSettings(isTrained);
            this.useUserFeatures = UseUserFeaturesDefault;
            this.useItemFeatures = UseItemFeaturesDefault;
            this.traitCount = TraitCountDefault;
            this.batchCount = BatchCountDefault;
            this.iterationCount = IterationCountDefault;
            this.useSharedUserThresholds = UseSharedUserThresholdsDefault;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="MatchboxRecommenderTrainingSettings"/> class
        /// from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the training settings from.</param>
        /// <param name="isTrained">Indicates whether the Matchbox recommender is trained.</param>
        internal MatchboxRecommenderTrainingSettings(IReader reader, Func<bool> isTrained)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            reader.VerifySerializationGuid(
                this.customSerializationGuid, "The binary stream does not contain the training settings of an Infer.NET Matchbox recommender.");

            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);

            if (deserializedVersion == 1)
            {
                this.trainingSettingsGuard = new SettingsGuard(reader, isTrained);
                this.ReadVersion1(reader, isTrained);
            }
            else if (deserializedVersion == CustomSerializationVersion)
            {
                this.trainingSettingsGuard = new SettingsGuard(reader, isTrained);
                this.ReadVersion2(reader, isTrained);
            }
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets the advanced settings of the Matchbox recommender.
        /// </summary>
        public MatchboxRecommenderAdvancedTrainingSettings Advanced { get; private set; }

        /// <summary>
        /// Gets or sets a value indicating whether to use explicit user features.
        /// </summary>
        public bool UseUserFeatures
        { 
            get
            {
                return this.useUserFeatures;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                this.useUserFeatures = value;
            }
        }

        /// <summary>
        /// Gets or sets a value indicating whether to use explicit item features.
        /// </summary>
        public bool UseItemFeatures
        {
            get
            {
                return this.useItemFeatures;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                this.useItemFeatures = value;
            }
        }

        /// <summary>
        /// Gets or sets the number of implicit user or item features (traits) to learn.
        /// </summary>
        public int TraitCount
        {
            get
            {
                return this.traitCount;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                Argument.CheckIfInRange(value >= 0, "value", "The number of traits must be non-negative.");
                this.traitCount = value;
            }
        }

        /// <summary>
        /// Gets or sets the number of batches the training data is split into.
        /// </summary>
        public int BatchCount
        {
            get
            {
                return this.batchCount;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                Argument.CheckIfInRange(value > 0, "value", "The number of batches must be positive.");
                this.batchCount = value;
            }
        }

        /// <summary>
        /// Gets or sets the number of inference iterations to run.
        /// </summary>
        public int IterationCount
        {
            get
            {
                return this.iterationCount;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                Argument.CheckIfInRange(value > 0, "value", "The number of inference iterations must be positive.");
                this.iterationCount = value;
            }
        }

        /// <summary>
        /// Gets or sets a value indicating whether to use shared user thresholds.
        /// </summary>
        public bool UseSharedUserThresholds
        {
            get
            {
                return this.useSharedUserThresholds;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                this.useSharedUserThresholds = value;
            }
        }

        #endregion

        #region ICustomSerializable implementation

        /// <summary>
        /// Saves the training settings of the Matchbox recommender using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the training settings to.</param>
        public void SaveForwardCompatible(IWriter writer)
        {
            writer.Write(this.customSerializationGuid);
            writer.Write(CustomSerializationVersion);

            this.trainingSettingsGuard.SaveForwardCompatible(writer);
            this.Advanced.SaveForwardCompatible(writer);

            writer.Write(this.useUserFeatures);
            writer.Write(this.useItemFeatures);
            writer.Write(this.traitCount);
            writer.Write(this.batchCount);
            writer.Write(this.iterationCount);
            writer.Write(this.useSharedUserThresholds);
        }

        #endregion

        #region Helper methods

        /// <summary>
        /// Reads the training settings for serialization version 1.
        /// </summary>
        /// <param name="reader">The binary reader to read the training settings from.</param>
        /// <param name="isTrained">Indicates whether the Matchbox recommender is trained.</param>
        private void ReadVersion1(IReader reader, Func<bool> isTrained)
        {
            this.Advanced = new MatchboxRecommenderAdvancedTrainingSettings(reader, isTrained);
            this.useUserFeatures = reader.ReadBoolean();
            this.useItemFeatures = reader.ReadBoolean();
            this.traitCount = reader.ReadInt32();
            this.batchCount = reader.ReadInt32();
            this.iterationCount = reader.ReadInt32();
        }

        /// <summary>
        /// Reads the training settings for serialization version 2.
        /// </summary>
        /// <param name="reader">The binary reader to read the training settings from.</param>
        /// <param name="isTrained">Indicates whether the Matchbox recommender is trained.</param>
        private void ReadVersion2(IReader reader, Func<bool> isTrained)
        {
            this.ReadVersion1(reader, isTrained);
            this.useSharedUserThresholds = reader.ReadBoolean();
        }

        #endregion
    }
}

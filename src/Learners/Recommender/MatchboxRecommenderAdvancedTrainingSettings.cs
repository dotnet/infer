// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;
    using System.IO;

    using Microsoft.ML.Probabilistic.Learners.MatchboxRecommenderInternal;
    using Microsoft.ML.Probabilistic.Serialization;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// Advanced settings of the Matchbox recommender which affect training.
    /// Cannot be set after training.
    /// </summary>
    [Serializable]
    public class MatchboxRecommenderAdvancedTrainingSettings : ICustomSerializable
    {
        /// <summary>
        /// The current custom binary serialization version of the <see cref="MatchboxRecommenderAdvancedTrainingSettings"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// The custom binary serialization <see cref="Guid"/> of the <see cref="MatchboxRecommenderAdvancedTrainingSettings"/> class.
        /// </summary>
        private readonly Guid customSerializationGuid = new Guid("EAD566B9-7005-4D0F-98D8-6DB83EC2717E");

        /// <summary>
        /// Guards the training Matchbox recommender settings from being changed after training.
        /// </summary>
        private readonly SettingsGuard trainingSettingsGuard;

        /// <summary>
        /// Initializes a new instance of the <see cref="MatchboxRecommenderAdvancedTrainingSettings"/> class.
        /// </summary>
        /// <param name="isTrained">Indicates whether the Matchbox recommender is trained.</param>
        internal MatchboxRecommenderAdvancedTrainingSettings(Func<bool> isTrained)
        {
            this.trainingSettingsGuard = new SettingsGuard(isTrained, "This setting cannot be changed after training.");

            this.User = new UserHyperparameters();
            this.Item = new ItemHyperparameters();
            this.UserFeature = new FeatureHyperparameters();
            this.ItemFeature = new FeatureHyperparameters();
            this.Noise = new NoiseHyperparameters();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="MatchboxRecommenderAdvancedTrainingSettings"/> class.
        /// from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the advanced training settings from.</param>
        /// <param name="isTrained">Indicates whether the Matchbox recommender is trained.</param>
        internal MatchboxRecommenderAdvancedTrainingSettings(IReader reader, Func<bool> isTrained)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            reader.VerifySerializationGuid(
                this.customSerializationGuid, "The binary stream does not contain the advanced training settings of an Infer.NET Matchbox recommender.");

            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);
            if (deserializedVersion == CustomSerializationVersion)
            {
                this.trainingSettingsGuard = new SettingsGuard(reader, isTrained);

                this.User = new UserHyperparameters(reader);
                this.Item = new ItemHyperparameters(reader);
                this.UserFeature = new FeatureHyperparameters(reader);
                this.ItemFeature = new FeatureHyperparameters(reader);
                this.Noise = new NoiseHyperparameters(reader);
            }
        }

        #region User hyper-parameter properties

        /// <summary>
        /// Gets or sets the variance of the user traits.
        /// </summary>
        public double UserTraitVariance
        {
            get
            {
                return this.User.TraitVariance;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                Argument.CheckIfInRange(value >= 0, "value", "The user trait variance must be non-negative.");
                this.User.TraitVariance = value;
            }
        }

        /// <summary>
        /// Gets or sets the variance of the user bias.
        /// </summary>
        public double UserBiasVariance
        {
            get
            {
                return this.User.BiasVariance;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                Argument.CheckIfInRange(value >= 0, "value", "The user bias variance must be non-negative.");
                this.User.BiasVariance = value;
            }
        }

        /// <summary>
        /// Gets or sets the variance of the prior distribution of the user threshold.
        /// </summary>
        public double UserThresholdPriorVariance
        {
            get
            {
                return this.User.ThresholdPriorVariance;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                Argument.CheckIfInRange(value >= 0, "value", "The user threshold prior variance must be non-negative.");
                this.User.ThresholdPriorVariance = value;
            }
        }

        #endregion

        #region Item hyper-parameter properties

        /// <summary>
        /// Gets or sets the variance of the item traits.
        /// </summary>
        public double ItemTraitVariance
        {
            get
            {
                return this.Item.TraitVariance;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                Argument.CheckIfInRange(value >= 0, "value", "The item trait variance must be non-negative.");
                this.Item.TraitVariance = value;
            }
        }

        /// <summary>
        /// Gets or sets the variance of the item bias.
        /// </summary>
        public double ItemBiasVariance
        {
            get
            {
                return this.Item.BiasVariance;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                Argument.CheckIfInRange(value >= 0, "value", "The item bias variance must be non-negative.");
                this.Item.BiasVariance = value;
            }
        }

        #endregion

        #region User feature hyper-parameter properties

        /// <summary>
        /// Gets or sets the variance of the prior distribution over feature weights used to compute user traits.
        /// </summary>
        public double UserTraitFeatureWeightPriorVariance
        {
            get
            {
                return this.UserFeature.TraitWeightPriorVariance;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                Argument.CheckIfInRange(value >= 0, "value", "The user trait feature weight prior variance must be non-negative.");
                this.UserFeature.TraitWeightPriorVariance = value;
            }
        }

        /// <summary>
        /// Gets or sets the variance of the prior distribution over feature weights used to compute user bias.
        /// </summary>
        public double UserBiasFeatureWeightPriorVariance
        {
            get
            {
                return this.UserFeature.BiasWeightPriorVariance;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                Argument.CheckIfInRange(value >= 0, "value", "The user bias feature weight prior variance must be non-negative.");
                this.UserFeature.BiasWeightPriorVariance = value;
            }
        }

        #endregion

        #region Item feature hyper-parameter properties

        /// <summary>
        /// Gets or sets the variance of the prior distribution over feature weights used to compute item traits.
        /// </summary>
        public double ItemTraitFeatureWeightPriorVariance
        {
            get
            {
                return this.ItemFeature.TraitWeightPriorVariance;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                Argument.CheckIfInRange(value >= 0, "value", "The item trait feature weight prior variance must be non-negative.");
                this.ItemFeature.TraitWeightPriorVariance = value;
            }
        }

        /// <summary>
        /// Gets or sets the variance of the prior distribution over feature weights used to compute item bias.
        /// </summary>
        public double ItemBiasFeatureWeightPriorVariance
        {
            get
            {
                return this.ItemFeature.BiasWeightPriorVariance;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                Argument.CheckIfInRange(value >= 0, "value", "The item bias feature weight prior variance must be non-negative.");
                this.ItemFeature.BiasWeightPriorVariance = value;
            }
        }

        #endregion

        #region Noise hyper-parameters properties

        /// <summary>
        /// Gets or sets the variance of the noise of the user thresholds.
        /// </summary>
        public double UserThresholdNoiseVariance
        {
            get
            {
                return this.Noise.UserThresholdVariance;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                Argument.CheckIfInRange(value >= 0, "value", "The variance of the noise of the user thresholds must be non-negative.");
                this.Noise.UserThresholdVariance = value;
            }
        }

        /// <summary>
        /// Gets or sets the variance of the affinity noise.
        /// </summary>
        public double AffinityNoiseVariance
        {
            get
            {
                return this.Noise.AffinityVariance;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                Argument.CheckIfInRange(value >= 0, "value", "The variance of the affinity noise must be non-negative.");
                this.Noise.AffinityVariance = value;
            }
        }

        #endregion

        #region Internal properties

        /// <summary>
        /// Gets the user hyper-parameters.
        /// </summary>
        internal UserHyperparameters User { get; private set; }

        /// <summary>
        /// Gets the item hyper-parameters.
        /// </summary>
        internal ItemHyperparameters Item { get; private set; }

        /// <summary>
        /// Gets the user feature hyper-parameters.
        /// </summary>
        internal FeatureHyperparameters UserFeature { get; private set; }

        /// <summary>
        /// Gets the item feature hyper-parameters.
        /// </summary>
        internal FeatureHyperparameters ItemFeature { get; private set; }

        /// <summary>
        /// Gets the noise hyper-parameters.
        /// </summary>
        internal NoiseHyperparameters Noise { get; private set; }

        #endregion

        #region ICustomSerializable implementation

        /// <summary>
        /// Saves the advanced training settings of a Matchbox recommender using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the advanced training settings to.</param>
        public void SaveForwardCompatible(IWriter writer)
        {
            writer.Write(this.customSerializationGuid);
            writer.Write(CustomSerializationVersion);

            this.trainingSettingsGuard.SaveForwardCompatible(writer);

            this.User.SaveForwardCompatible(writer);
            this.Item.SaveForwardCompatible(writer);
            this.UserFeature.SaveForwardCompatible(writer);
            this.ItemFeature.SaveForwardCompatible(writer);
            this.Noise.SaveForwardCompatible(writer);
        }

        #endregion
    }
}
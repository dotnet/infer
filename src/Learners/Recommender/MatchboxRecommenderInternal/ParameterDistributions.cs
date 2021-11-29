// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.MatchboxRecommenderInternal
{
    using System;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Serialization;

    using GaussianArray = Distributions.DistributionStructArray<Microsoft.ML.Probabilistic.Distributions.Gaussian, double>;
    using GaussianMatrix = Distributions.DistributionRefArray<Distributions.DistributionStructArray<Microsoft.ML.Probabilistic.Distributions.Gaussian, double>, double[]>;

    /// <summary>
    /// Contains distributions over the parameters of the Matchbox model.
    /// </summary>
    [Serializable]
    internal class ParameterDistributions : ICloneable, ICustomSerializable
    {
        /// <summary>
        /// The current custom binary serialization version of the <see cref="ParameterDistributions"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// Initializes a new instance of the <see cref="ParameterDistributions"/> class.
        /// </summary>
        public ParameterDistributions()
        {
            this.UserTraitDistribution = new GaussianMatrix(0);
            this.UserBiasDistribution = new GaussianArray(0);
            this.UserThresholdDistribution = new GaussianMatrix(0);
            this.ItemTraitDistribution = new GaussianMatrix(0);
            this.ItemBiasDistribution = new GaussianArray(0);

            this.UserFeature = new FeatureParameterDistribution();
            this.ItemFeature = new FeatureParameterDistribution();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ParameterDistributions"/> class
        /// from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the distributions over parameters from.</param>
        public ParameterDistributions(IReader reader)
        {
            Debug.Assert(reader != null, "The reader must not be null.");

            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);

            if (deserializedVersion == CustomSerializationVersion)
            {
                this.UserTraitDistribution = reader.ReadGaussianMatrix();
                this.UserBiasDistribution = reader.ReadGaussianArray();
                this.UserThresholdDistribution = reader.ReadGaussianMatrix();
                this.ItemTraitDistribution = reader.ReadGaussianMatrix();
                this.ItemBiasDistribution = reader.ReadGaussianArray();

                this.UserFeature = new FeatureParameterDistribution(reader);
                this.ItemFeature = new FeatureParameterDistribution(reader);
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ParameterDistributions"/> class.
        /// Each contained distribution is initialized to uniform.
        /// </summary>
        /// <param name="metaData">The instance meta data.</param>
        /// <param name="traitCount">The number of traits.</param>
        public ParameterDistributions(InstanceMetadata metaData, int traitCount)
        {
            // Check input data
            Debug.Assert(metaData.UserCount > 0, "The number of users must be positive.");
            Debug.Assert(metaData.ItemCount > 0, "The number of items must be positive.");
            Debug.Assert(metaData.RatingCount + 1 > 0, "The number of user thresholds must be positive.");
            Debug.Assert(metaData.UserFeatures.FeatureCount >= 0, "The number of user features must be non-negative.");
            Debug.Assert(metaData.ItemFeatures.FeatureCount >= 0, "The number of item features must be non-negative.");
            Debug.Assert(traitCount >= 0, "The number of traits must be non-negative.");

            // The value to which each element of the contained distributions will be initialized
            var initialValue = Gaussian.Uniform();

            // Initialize contained distributions.
            this.UserTraitDistribution = new GaussianMatrix(new GaussianArray(initialValue, traitCount), metaData.UserCount);
            this.UserBiasDistribution = new GaussianArray(initialValue, metaData.UserCount);
            this.UserThresholdDistribution = new GaussianMatrix(
                new GaussianArray(initialValue, metaData.RatingCount + 1), metaData.UserCount);
            this.ItemTraitDistribution = new GaussianMatrix(new GaussianArray(initialValue, traitCount), metaData.ItemCount);
            this.ItemBiasDistribution = new GaussianArray(initialValue, metaData.ItemCount);

            this.UserFeature = new FeatureParameterDistribution(traitCount, metaData.UserFeatures.FeatureCount, initialValue);
            this.ItemFeature = new FeatureParameterDistribution(traitCount, metaData.ItemFeatures.FeatureCount, initialValue);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ParameterDistributions"/> class.
        /// </summary>
        /// <param name="userTraitDistribution">The distribution over user traits.</param>
        /// <param name="userBiasDistribution">The distribution over user biases.</param>
        /// <param name="userThresholdDistribution">The distribution over user thresholds.</param>
        /// <param name="itemTraitDistribution">The distribution over item traits.</param>
        /// <param name="itemBiasDistribution">The distribution over item biases.</param>
        /// <param name="userTraitFeatureWeightDistribution">The distribution over user trait feature weights.</param>
        /// <param name="userBiasFeatureWeightDistribution">The distribution over user bias feature weights.</param>
        /// <param name="itemTraitFeatureWeightDistribution">The distribution over item trait feature weights.</param>
        /// <param name="itemBiasFeatureWeightDistribution">The distribution over item bias feature weights.</param>
        /// <remarks>The feature parameters can be null.</remarks>
        public ParameterDistributions(
            GaussianMatrix userTraitDistribution,
            GaussianArray userBiasDistribution,
            GaussianMatrix userThresholdDistribution,
            GaussianMatrix itemTraitDistribution,
            GaussianArray itemBiasDistribution,
            GaussianMatrix userTraitFeatureWeightDistribution,
            GaussianArray userBiasFeatureWeightDistribution,
            GaussianMatrix itemTraitFeatureWeightDistribution,
            GaussianArray itemBiasFeatureWeightDistribution)
        {
            Debug.Assert(userTraitDistribution.Count > 0, "Parameters for at least one user must be provided.");
            Debug.Assert(itemTraitDistribution.Count > 0, "Parameters for at least one item must be provided.");
            Debug.Assert(
                userTraitDistribution.Count == userBiasDistribution.Count && userTraitDistribution.Count == userThresholdDistribution.Count,
                "User counts in various distribution arrays must be consistent.");
            Debug.Assert(
                itemTraitDistribution.Count == itemBiasDistribution.Count,
                "Item counts in various distribution arrays must be consistent.");
            Debug.Assert(
                userTraitFeatureWeightDistribution == null ||
                userTraitFeatureWeightDistribution.Count == userTraitDistribution[0].Count,
                "The number of traits must be consistent between various distribution arrays.");
            Debug.Assert(
                itemTraitFeatureWeightDistribution == null ||
                itemTraitFeatureWeightDistribution.Count == userTraitDistribution[0].Count,
                "The number of traits must be consistent between various distribution arrays.");
            Debug.Assert(
                userTraitDistribution.All(t => t.Count == userTraitDistribution[0].Count),
                "The number of traits must be consistent between various distribution arrays.");
            Debug.Assert(
                itemTraitDistribution.All(t => t.Count == userTraitDistribution[0].Count),
                "The number of traits must be consistent between various distribution arrays.");
            Debug.Assert(
                itemTraitFeatureWeightDistribution == null ||
                itemTraitFeatureWeightDistribution.Count == userTraitDistribution[0].Count,
                "The number of traits must be consistent between various distribution arrays.");
            Debug.Assert(
                userThresholdDistribution.All(t => t.Count == userThresholdDistribution[0].Count),
                "The number of user thresholds must be consistent between various distribution arrays.");

            this.UserTraitDistribution = userTraitDistribution;
            this.UserBiasDistribution = userBiasDistribution;
            this.UserThresholdDistribution = userThresholdDistribution;
            this.ItemTraitDistribution = itemTraitDistribution;
            this.ItemBiasDistribution = itemBiasDistribution;
            
            this.UserFeature = new FeatureParameterDistribution(userTraitFeatureWeightDistribution, userBiasFeatureWeightDistribution);
            this.ItemFeature = new FeatureParameterDistribution(itemTraitFeatureWeightDistribution, itemBiasFeatureWeightDistribution);
        }

        /// <summary>
        /// Gets the distribution over user traits.
        /// </summary>
        public GaussianMatrix UserTraitDistribution { get; private set; }

        /// <summary>
        /// Gets the distribution over user biases.
        /// </summary>
        public GaussianArray UserBiasDistribution { get; private set; }

        /// <summary>
        /// Gets the distribution over user thresholds.
        /// </summary>
        public GaussianMatrix UserThresholdDistribution { get; private set; }

        /// <summary>
        /// Gets the distribution over item traits.
        /// </summary>
        public GaussianMatrix ItemTraitDistribution { get; private set; }

        /// <summary>
        /// Gets the distribution over item biases.
        /// </summary>
        public GaussianArray ItemBiasDistribution { get; private set; }

        /// <summary>
        /// Gets the number of traits used in the model.
        /// </summary>
        public int TraitCount
        {
            get { return this.UserTraitDistribution[0].Count; }
        }

        /// <summary>
        /// Gets the number of user thresholds used in the model.
        /// </summary>
        public int UserThresholdCount
        {
            get { return this.UserThresholdDistribution[0].Count; }
        }

        /// <summary>
        /// Gets the number of users used in the model.
        /// </summary>
        public int UserCount
        {
            get { return this.UserTraitDistribution.Count;  }
        }

        /// <summary>
        /// Gets the number of items used in the model.
        /// </summary>
        public int ItemCount
        {
            get { return this.ItemTraitDistribution.Count; }
        }

        /// <summary>
        /// Gets the distribution over user feature related parameters.
        /// </summary>
        public FeatureParameterDistribution UserFeature { get; private set; }

        /// <summary>
        /// Gets the distribution over item feature related parameters.
        /// </summary>
        public FeatureParameterDistribution ItemFeature { get; private set; }

        /// <summary>
        /// Gets a distribution over the parameters of a given user.
        /// </summary>
        /// <param name="userId">The user identifier.</param>
        /// <returns>The distribution over user parameters.</returns>
        public UserParameterDistribution ForUser(int userId)
        {
            Debug.Assert(userId >= 0 && userId < this.UserTraitDistribution.Count, "An unknown user identifier specified.");

            return new UserParameterDistribution(
                this.UserTraitDistribution[userId], this.UserBiasDistribution[userId], this.UserThresholdDistribution[userId]);
        }

        /// <summary>
        /// Gets a distribution over the parameters of a given item.
        /// </summary>
        /// <param name="itemId">The item identifier.</param>
        /// <returns>The distribution over user parameters.</returns>
        public ItemParameterDistribution ForItem(int itemId)
        {
            Debug.Assert(itemId >= 0 && itemId < this.ItemTraitDistribution.Count, "An unknown item identifier specified.");

            return new ItemParameterDistribution(this.ItemTraitDistribution[itemId], this.ItemBiasDistribution[itemId]);
        }

        /// <summary>
        /// Sets the traits, biases and thresholds to uniform distributions.
        /// </summary>
        /// <remarks>This method does not affect the user and item features.</remarks>
        public void SetEntityParametersToUniform()
        {
            var uniformGaussian = Gaussian.Uniform();

            foreach (var array in this.UserTraitDistribution)
            {
                array.SetAllElementsTo(uniformGaussian);
            }

            this.UserBiasDistribution.SetAllElementsTo(uniformGaussian);

            foreach (var array in this.UserThresholdDistribution)
            {
                array.SetAllElementsTo(uniformGaussian);
            }

            foreach (var array in this.ItemTraitDistribution)
            {
                array.SetAllElementsTo(uniformGaussian);
            }
            
            this.ItemBiasDistribution.SetAllElementsTo(uniformGaussian);
        }

        /// <summary>
        /// Sets the traits, biases and thresholds to represent the ratio of two parameter distributions.
        /// </summary>
        /// <param name="numerator">The numerator parameter distribution.</param>
        /// <param name="denominator">The denominator parameter distribution.</param>
        /// <param name="forceProper">A flag indicating whether or not to force proper messages.</param>
        /// <remarks>This method does not affect the user and item features.</remarks>
        public void SetEntityParametersToRatio(ParameterDistributions numerator, ParameterDistributions denominator, bool forceProper = true)
        {
            this.UserTraitDistribution.SetToRatio(numerator.UserTraitDistribution, denominator.UserTraitDistribution, forceProper);
            this.UserBiasDistribution.SetToRatio(numerator.UserBiasDistribution, denominator.UserBiasDistribution, forceProper);
            this.UserThresholdDistribution.SetToRatio(numerator.UserThresholdDistribution, denominator.UserThresholdDistribution, forceProper);
            this.ItemTraitDistribution.SetToRatio(numerator.ItemTraitDistribution, denominator.ItemTraitDistribution, forceProper);
            this.ItemBiasDistribution.SetToRatio(numerator.ItemBiasDistribution, denominator.ItemBiasDistribution, forceProper);
        }

        /// <summary>
        /// Sets the traits, biases and thresholds to the product of two parameter distributions.
        /// </summary>
        /// <param name="firstFactor">The first factor.</param>
        /// <param name="secondFactor">The second factor.</param>
        /// <remarks>This method does not affect the user and item features.</remarks>
        public void SetEntityParametersToProduct(ParameterDistributions firstFactor, ParameterDistributions secondFactor)
        {
            this.UserTraitDistribution.SetToProduct(firstFactor.UserTraitDistribution, secondFactor.UserTraitDistribution);
            this.UserBiasDistribution.SetToProduct(firstFactor.UserBiasDistribution, secondFactor.UserBiasDistribution);
            this.UserThresholdDistribution.SetToProduct(firstFactor.UserThresholdDistribution, secondFactor.UserThresholdDistribution);
            this.ItemTraitDistribution.SetToProduct(firstFactor.ItemTraitDistribution, secondFactor.ItemTraitDistribution);
            this.ItemBiasDistribution.SetToProduct(firstFactor.ItemBiasDistribution, secondFactor.ItemBiasDistribution);
        }

        /// <summary>
        /// Sets the traits, biases and thresholds to the power of a parameter distributions.
        /// </summary>
        /// <param name="value">The base.</param>
        /// <param name="exponent">The exponent.</param>
        /// <remarks>This method does not affect the user and item features.</remarks>
        public void SetEntityParametersToPower(ParameterDistributions value, double exponent)
        {
            this.UserTraitDistribution.SetToPower(value.UserTraitDistribution, exponent);
            this.UserBiasDistribution.SetToPower(value.UserBiasDistribution, exponent);
            this.UserThresholdDistribution.SetToPower(value.UserThresholdDistribution, exponent);
            this.ItemTraitDistribution.SetToPower(value.ItemTraitDistribution, exponent);
            this.ItemBiasDistribution.SetToPower(value.ItemBiasDistribution, exponent);
        }

        /// <summary>
        /// Creates a new object that is a copy of the current instance.
        /// </summary>
        /// <returns>A new object that is a copy of this instance.</returns>
        public object Clone()
        {
            return new ParameterDistributions
            {
                UserTraitDistribution = (GaussianMatrix)this.UserTraitDistribution.Clone(),
                UserBiasDistribution = (GaussianArray)this.UserBiasDistribution.Clone(),
                UserThresholdDistribution = (GaussianMatrix)this.UserThresholdDistribution.Clone(),
                ItemTraitDistribution = (GaussianMatrix)this.ItemTraitDistribution.Clone(),
                ItemBiasDistribution = (GaussianArray)this.ItemBiasDistribution.Clone(),
                UserFeature = (FeatureParameterDistribution)this.UserFeature.Clone(),
                ItemFeature = (FeatureParameterDistribution)this.ItemFeature.Clone()
            };
        }

        /// <summary>
        /// Saves the distributions over parameters using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the distributions over parameters to.</param>
        public void SaveForwardCompatible(IWriter writer)
        {
            writer.Write(CustomSerializationVersion);

            writer.Write(this.UserTraitDistribution);
            writer.Write(this.UserBiasDistribution);
            writer.Write(this.UserThresholdDistribution);
            writer.Write(this.ItemTraitDistribution);
            writer.Write(this.ItemBiasDistribution);

            this.UserFeature.SaveForwardCompatible(writer);
            this.ItemFeature.SaveForwardCompatible(writer);
        }
    }
}

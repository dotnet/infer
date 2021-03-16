// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.MatchboxRecommenderInternal
{
    using System.Collections.Generic;

    using Microsoft.ML.Probabilistic.Factors;
    using Microsoft.ML.Probabilistic.Models;
    using Microsoft.ML.Probabilistic.Models.Attributes;
    using GaussianArray = Distributions.DistributionStructArray<Microsoft.ML.Probabilistic.Distributions.Gaussian, double>;
    using GaussianMatrix = Distributions.DistributionRefArray<Distributions.DistributionStructArray<Microsoft.ML.Probabilistic.Distributions.Gaussian, double>, double[]>;

    /// <summary>
    /// Defines the Matchbox training and prediction models.
    /// </summary>
    internal static class Models
    {
        /// <summary>
        /// Constructs the Matchbox model with the specified settings.
        /// </summary>
        /// <param name="buildTrainingModel">Specifies whether the method should build a training or a prediction model.</param>
        /// <param name="breakTraitSymmetry">Specifies whether the constraints for breaking trait symmetry should be specified.</param>
        /// <param name="usePreAdjustedUserParameters">Specifies whether user parameters should be sampled from custom priors (without using features).</param>
        /// <param name="usePreAdjustedItemParameters">Specifies whether item parameters should be sampled from custom priors (without using features).</param>
        /// <returns>The variables to infer from the built model.</returns>
        public static IVariable[] BuildModel(
            bool buildTrainingModel,
            bool breakTraitSymmetry,
            bool usePreAdjustedUserParameters,
            bool usePreAdjustedItemParameters)
        {
            // Counts
            var userCount = Variable.Observed(default(int)).Named("UserCount");
            var itemCount = Variable.Observed(default(int)).Named("ItemCount");
            var userThresholdCount = Variable.Observed(default(int)).Named("UserThresholdCount");
            var observationCount = Variable.Observed(default(int)).Named("ObservationCount");
            var traitCount = Variable.Observed(default(int)).Named("TraitCount");

            // Ranges
            var user = new Range(userCount).Named("user");
            var item = new Range(itemCount).Named("item");
            var trait = new Range(traitCount).Named("trait");
            var observation = new Range(observationCount).Named("observation");
            var userThreshold = new Range(userThresholdCount).Named("userThreshold");
            var ratingValue = new Range(userThresholdCount - 1).Named("ratingValue");

            if (buildTrainingModel)
            {
                // Use a sequential schedule
                user.AddAttribute(new Sequential());
                item.AddAttribute(new Sequential());
                observation.AddAttribute(new Sequential());
            }

            // Latent variables
            var userTraits = Variable.Array(Variable.Array<double>(trait), user).Named("UserTraits");
            var itemTraits = Variable.Array(Variable.Array<double>(trait), item).Named("ItemTraits");
            var userBias = Variable.Array<double>(user).Named("UserBias");
            var itemBias = Variable.Array<double>(item).Named("ItemBias");
            var userThresholds = Variable.Array(Variable.Array<double>(userThreshold), user).Named("UserThresholds");

            // User latent variable definitions
            var userFeatureCount = Variable.Observed(default(int)).Named("UserFeatureCount");
            var userFeature = new Range(userFeatureCount).Named("userFeature");
            var userTraitFeatureWeights = Variable.Array(Variable.Array<double>(userFeature), trait).Named("UserTraitFeatureWeights");
            var userBiasFeatureWeights = Variable.Array<double>(userFeature).Named("UserBiasFeatureWeights");

            if (usePreAdjustedUserParameters)
            {
                var userTraitsPrior = Variable.Observed(default(GaussianMatrix)).Named("UserTraitsPrior");
                var userBiasPrior = Variable.Observed(default(GaussianArray)).Named("UserBiasPrior");
                var userThresholdsPrior = Variable.Observed(default(GaussianMatrix)).Named("UserThresholdsPrior");

                userTraits.SetTo(Variable<double[][]>.Random(userTraitsPrior));
                userBias.SetTo(Variable<double[]>.Random(userBiasPrior));
                userThresholds.SetTo(Variable<double[][]>.Random(userThresholdsPrior));
            }
            else
            {
                // Features
                var nonZeroUserFeatureCounts = Variable.Observed(default(IReadOnlyList<int>), user).Named("NonZeroUserFeatureCounts");
                var nonZeroUserFeature = new Range(nonZeroUserFeatureCounts[user]).Named("nonZeroUserFeature");
                var nonZeroUserFeatureIndices = Variable.Observed(default(IReadOnlyList<IReadOnlyList<int>>), user, nonZeroUserFeature).Named("NonZeroUserFeatureIndices");
                var nonZeroUserFeatureValues = Variable.Observed(default(IReadOnlyList<IReadOnlyList<double>>), user, nonZeroUserFeature).Named("NonZeroUserFeatureValues");

                // Feature weights
                var userTraitFeatureWeightVariance = Variable.Observed(default(double)).Named("UserTraitFeatureWeightPriorVariance");
                var userBiasFeatureWeightVariance = Variable.Observed(default(double)).Named("UserBiasFeatureWeightPriorVariance");
                userTraitFeatureWeights[trait][userFeature] =
                    Variable.GaussianFromMeanAndVariance(0.0, userTraitFeatureWeightVariance).ForEach(userFeature).ForEach(trait);
                userBiasFeatureWeights[userFeature] =
                    Variable.GaussianFromMeanAndVariance(0.0, userBiasFeatureWeightVariance).ForEach(userFeature);

                // Traits, biases and thresholds
                var userTraitVariance = Variable.Observed(default(double)).Named("UserTraitVariance");
                var userBiasVariance = Variable.Observed(default(double)).Named("UserBiasVariance");
                var userThresholdPriorMean = Variable.Observed(default(double[]), userThreshold).Named("UserThresholdPriorMean");
                var userThresholdPriorVariance = Variable.Observed(default(double)).Named("UserThresholdPriorVariance");

                var middleUserThresholdIndex = Variable.Observed(default(int)).Named("MiddleUserThresholdIndex");
                using (var userThresholdIterationBlock = Variable.ForEach(userThreshold))
                {
                    var middleThreshold = userThresholdIterationBlock.Index == middleUserThresholdIndex;
                    var userThresholdPriorVarianceForThreshold = Variable.New<double>().Named("UserThresholdPriorVarianceForThreshold");
                    using (Variable.If(middleThreshold))
                    {
                        // Break symmetry between the user thresholds and the user bias
                        userThresholdPriorVarianceForThreshold.SetTo(0.0);
                    }
                    using (Variable.IfNot(middleThreshold))
                    {
                        userThresholdPriorVarianceForThreshold.SetTo(userThresholdPriorVariance);
                    }

                    userThresholds[user][userThresholdIterationBlock.Index] = Variable.GaussianFromMeanAndVariance(
                        userThresholdPriorMean[userThresholdIterationBlock.Index], userThresholdPriorVarianceForThreshold).ForEach(user);
                }

                using (Variable.ForEach(user))
                {
                    var nonZeroUserTraitFeatureWeights = Variable.Subarray(userTraitFeatureWeights[trait], nonZeroUserFeatureIndices[user]);
                    nonZeroUserTraitFeatureWeights.Name = "nonZeroUserTraitFeatureWeights";
                    var nonZeroUserTraitFeatureWeightProducts = Variable.Array(Variable.Array<double>(nonZeroUserFeature), trait).Named("nonZeroUserTraitFeatureWeightProducts");
                    nonZeroUserTraitFeatureWeightProducts[trait][nonZeroUserFeature] =
                        nonZeroUserTraitFeatureWeights[nonZeroUserFeature] * nonZeroUserFeatureValues[user][nonZeroUserFeature];
                    var userTraitMean = Variable.Sum(nonZeroUserTraitFeatureWeightProducts[trait]).Named("userTraitMean");
                    userTraits[user][trait] = Variable.GaussianFromMeanAndVariance(userTraitMean, userTraitVariance);

                    var nonZeroUserBiasFeatureWeights = Variable.Subarray(userBiasFeatureWeights, nonZeroUserFeatureIndices[user]);
                    nonZeroUserBiasFeatureWeights.Name = "nonZeroUserBiasFeatureWeights";
                    var nonZeroUserBiasFeatureWeightProducts = Variable.Array<double>(nonZeroUserFeature).Named("nonZeroUserBiasFeatureWeightProducts");
                    nonZeroUserBiasFeatureWeightProducts[nonZeroUserFeature] = 
                        nonZeroUserBiasFeatureWeights[nonZeroUserFeature] * nonZeroUserFeatureValues[user][nonZeroUserFeature];
                    var userBiasMean = Variable.Sum(nonZeroUserBiasFeatureWeightProducts).Named("userBiasMean");
                    userBias[user] = Variable.GaussianFromMeanAndVariance(userBiasMean, userBiasVariance);
                }

                // Trait, bias and threshold messages
                var userTraitsMessage = Variable.Observed(default(GaussianMatrix)).Named("UserTraitsMessage");
                var userBiasMessage = Variable.Observed(default(GaussianArray)).Named("UserBiasMessage");
                var userThresholdsMessage = Variable.Observed(default(GaussianMatrix)).Named("UserThresholdsMessage");
                Variable.ConstrainEqualRandom(userTraits, userTraitsMessage);
                Variable.ConstrainEqualRandom(userBias, userBiasMessage);
                Variable.ConstrainEqualRandom(userThresholds, userThresholdsMessage);
                userTraits.AddAttribute(QueryTypes.MarginalDividedByPrior);
                userBias.AddAttribute(QueryTypes.MarginalDividedByPrior);
                userThresholds.AddAttribute(QueryTypes.MarginalDividedByPrior);
                userTraits.AddAttribute(QueryTypes.Marginal);
                userBias.AddAttribute(QueryTypes.Marginal);
                userThresholds.AddAttribute(QueryTypes.Marginal);
                //// User thresholds do not have the initialize backward attribute because we want the 
                //// schedule to send their backward message before the affinity backward message.

                if (buildTrainingModel)
                {
                    // Trait, bias and threshold initializers
                    var userTraitsInitializer = Variable.Observed(default(GaussianMatrix)).Named("UserTraitsInitializer");
                    var userBiasInitializer = Variable.Observed(default(GaussianArray)).Named("UserBiasInitializer");
                    var userThresholdsInitializer = Variable.Observed(default(GaussianMatrix)).Named("UserThresholdsInitializer");
                    userTraits.InitialiseTo(userTraitsInitializer);
                    userBias.InitialiseTo(userBiasInitializer);
                    userThresholds.InitialiseTo(userThresholdsInitializer);
                }
            }

            // Item latent variable definitions
            var itemFeatureCount = Variable.Observed(default(int)).Named("ItemFeatureCount");
            var itemFeature = new Range(itemFeatureCount).Named("itemFeature");
            var itemTraitFeatureWeights = Variable.Array(Variable.Array<double>(itemFeature), trait).Named("ItemTraitFeatureWeights");
            var itemBiasFeatureWeights = Variable.Array<double>(itemFeature).Named("ItemBiasFeatureWeights");

            if (usePreAdjustedItemParameters)
            {
                // Define using custom priors
                var itemTraitsPrior = Variable.Observed(default(GaussianMatrix)).Named("ItemTraitsPrior");
                var itemBiasPrior = Variable.Observed(default(GaussianArray)).Named("ItemBiasPrior");

                itemTraits.SetTo(Variable<double[][]>.Random(itemTraitsPrior));
                itemBias.SetTo(Variable<double[]>.Random(itemBiasPrior));
            }
            else
            {
                // Features
                var nonZeroItemFeatureCounts = Variable.Observed(default(IReadOnlyList<int>), item).Named("NonZeroItemFeatureCounts");
                var nonZeroItemFeature = new Range(nonZeroItemFeatureCounts[item]).Named("nonZeroItemFeature");
                var nonZeroItemFeatureIndices = Variable.Observed(default(IReadOnlyList<IReadOnlyList<int>>), item, nonZeroItemFeature).Named("NonZeroItemFeatureIndices");
                var nonZeroItemFeatureValues = Variable.Observed(default(IReadOnlyList<IReadOnlyList<double>>), item, nonZeroItemFeature).Named("NonZeroItemFeatureValues");

                // Feature weights
                var itemTraitFeatureWeightVariance = Variable.Observed(default(double)).Named("ItemTraitFeatureWeightPriorVariance");
                var itemBiasFeatureWeightVariance = Variable.Observed(default(double)).Named("ItemBiasFeatureWeightPriorVariance");
                itemTraitFeatureWeights[trait][itemFeature] =
                    Variable.GaussianFromMeanAndVariance(0.0, itemTraitFeatureWeightVariance).ForEach(itemFeature).ForEach(trait);
                itemBiasFeatureWeights[itemFeature] =
                    Variable.GaussianFromMeanAndVariance(0.0, itemBiasFeatureWeightVariance).ForEach(itemFeature);

                // Traits and biases
                var itemTraitVariance = Variable.Observed(default(double)).Named("ItemTraitVariance");
                var itemBiasVariance = Variable.Observed(default(double)).Named("ItemBiasVariance");

                using (Variable.ForEach(item))
                {
                    var nonZeroItemTraitFeatureWeights = Variable.Subarray(itemTraitFeatureWeights[trait], nonZeroItemFeatureIndices[item]);
                    nonZeroItemTraitFeatureWeights.Name = "nonZeroItemTraitFeatureWeights";
                    var nonZeroItemTraitFeatureWeightProducts = Variable.Array(Variable.Array<double>(nonZeroItemFeature), trait);
                    nonZeroItemTraitFeatureWeightProducts[trait][nonZeroItemFeature] =
                        nonZeroItemTraitFeatureWeights[nonZeroItemFeature] * nonZeroItemFeatureValues[item][nonZeroItemFeature];
                    var itemTraitMean = Variable.Sum(nonZeroItemTraitFeatureWeightProducts[trait]).Named("itemTraitMean");
                    itemTraits[item][trait] = Variable.GaussianFromMeanAndVariance(itemTraitMean, itemTraitVariance);

                    var nonZeroItemBiasFeatureWeights = Variable.Subarray(itemBiasFeatureWeights, nonZeroItemFeatureIndices[item]);
                    nonZeroItemBiasFeatureWeights.Name = "nonZeroItemBiasFeatureWeights";
                    var nonZeroItemBiasFeatureWeightProducts = Variable.Array<double>(nonZeroItemFeature);
                    nonZeroItemBiasFeatureWeightProducts[nonZeroItemFeature] =
                        nonZeroItemBiasFeatureWeights[nonZeroItemFeature] * nonZeroItemFeatureValues[item][nonZeroItemFeature];
                    var itemBiasMean = Variable.Sum(nonZeroItemBiasFeatureWeightProducts).Named("itemBiasMean");
                    itemBias[item] = Variable.GaussianFromMeanAndVariance(itemBiasMean, itemBiasVariance);
                }

                // Trait and bias messages
                var itemTraitsMessage = Variable.Observed(default(GaussianMatrix)).Named("ItemTraitsMessage");
                var itemBiasMessage = Variable.Observed(default(GaussianArray)).Named("ItemBiasMessage");
                Variable.ConstrainEqualRandom(itemTraits, itemTraitsMessage);
                Variable.ConstrainEqualRandom(itemBias, itemBiasMessage);
                itemTraits.AddAttribute(QueryTypes.MarginalDividedByPrior);
                itemBias.AddAttribute(QueryTypes.MarginalDividedByPrior);
                itemTraits.AddAttribute(QueryTypes.Marginal);
                itemBias.AddAttribute(QueryTypes.Marginal);

                if (buildTrainingModel)
                {
                    // Trait and bias initializers
                    var itemTraitsInitializer = Variable.Observed(default(GaussianMatrix)).Named("ItemTraitsInitializer");
                    var itemBiasInitializer = Variable.Observed(default(GaussianArray)).Named("ItemBiasInitializer");
                    itemTraits.InitialiseTo(itemTraitsInitializer);
                    itemBias.InitialiseTo(itemBiasInitializer);
                }
            }

            // Initialize first min(itemCount, traitCount) item traits to identity matrix to break symmetry
            if (breakTraitSymmetry)
            {
                // Note that this code affects evidence computation
                using (var traitIterationBlock1 = Variable.ForEach(trait))
                {
                    var traitCopy = trait.Clone();
                    using (var traitIterationBlock2 = Variable.ForEach(traitCopy))
                    {
                        using (Variable.If(traitIterationBlock1.Index < itemCount & traitIterationBlock2.Index < itemCount))
                        {
                            var diagonalElement = traitIterationBlock1.Index == traitIterationBlock2.Index;

                            using (Variable.If(diagonalElement))
                            {
                                Variable.ConstrainEqual(itemTraits[traitIterationBlock1.Index][traitIterationBlock2.Index], 1);
                            }

                            using (Variable.IfNot(diagonalElement))
                            {
                                Variable.ConstrainEqual(itemTraits[traitIterationBlock1.Index][traitIterationBlock2.Index], 0);
                            }
                        }
                    }
                }
            }

            // Observation data
            var userIds = Variable.Observed(default(IReadOnlyList<int>), observation).Named("UserIds");
            var itemIds = Variable.Observed(default(IReadOnlyList<int>), observation).Named("ItemIds");
            IJaggedVariableArray<Variable<int>> ratings;
            if (buildTrainingModel)
            {
                var observedRatings = Variable.Observed(default(IReadOnlyList<int>), observation).Named("Ratings");
                observedRatings.SetValueRange(ratingValue);
                ratings = observedRatings;
            }
            else
            {
                ratings = Variable.Array<int>(observation).Named("Ratings");
                ratings[observation] = Variable.DiscreteUniform(ratingValue).ForEach(observation);
            }

            // Noise variances
            var affinityNoiseVariance = Variable.Observed(default(double)).Named("AffinityNoiseVariance");
            var thresholdNoiseVariance = Variable.Observed(default(double)).Named("UserThresholdNoiseVariance");

            // Connect everything to observations
            // Send VMP messages here for correct message reconstruction in batching
            using (Variable.ForEach(observation))
            {
                var userId = userIds[observation];
                var itemId = itemIds[observation];

                // Products of user and item traits (as if running VMP)
                // Make a copy so that we can attach attributes to the argument of Factor.Product_SHG09, not the original itemTraits array.
                var itemTrait = Variable.Copy(itemTraits[itemId][trait]).Named("itemTrait");
                var userTrait = Variable.Copy(userTraits[userId][trait]).Named("userTrait");
                var traitProducts = Variable.Array<double>(trait);
                traitProducts[trait] = Variable<double>.Factor(Factor.Product_SHG09, userTrait, itemTrait);

                // Affinity
                var userBiasObs = Variable.Copy(userBias[userId]).Named("userBiasObs");
                var itemBiasObs = Variable.Copy(itemBias[itemId]).Named("itemBiasObs");
                var affinity =
                    Variable<double>.Factor(Factor.Product_SHG09, userBiasObs, 1.0)
                    + Variable<double>.Factor(Factor.Product_SHG09, itemBiasObs, 1.0)
                    + Variable.Sum(traitProducts);
                var noisyAffinity = Variable.GaussianFromMeanAndVariance(affinity, affinityNoiseVariance);

                // During prediction, we don't want to update the parameters. Variable.Cut ensures this.
                if (!buildTrainingModel)
                {
                    noisyAffinity = Variable.Cut(noisyAffinity);
                }

                // User thresholds
                var useSharedUserThresholds = Variable.Observed(default(bool)).Named("UseSharedUserThresholds");
                var userThresholdsObs = Variable.Array<double>(userThreshold).Named("UserThresholdsObs");
                using (Variable.If(useSharedUserThresholds))
                {
                    userThresholdsObs.SetTo(Variable.Copy(userThresholds[0]));
                }
                using (Variable.IfNot(useSharedUserThresholds))
                {
                    userThresholdsObs.SetTo(Variable.Copy(userThresholds[userId]));
                }

                var noisyUserThresholds = Variable.Array<double>(userThreshold).Named("NoisyUserThresholds");
                var noisyUserThreshold = 
                    Variable.GaussianFromMeanAndVariance(
                    Variable<double>.Factor(Factor.Product_SHG09, userThresholdsObs[userThreshold], 1.0),
                    thresholdNoiseVariance);
                if (!buildTrainingModel)
                {
                    noisyUserThreshold = Variable.Cut(noisyUserThreshold);
                }
                noisyUserThresholds[userThreshold] = noisyUserThreshold;

                if (buildTrainingModel)
                {
                    // Every argument to Factor.Product_SHG09 should have an InitialiseBackward attribute, for the special first iteration.
                    // They should also be initialized to the current marginal distribution.
                    // This ensures that the factor receives the marginal during the special first iteration.
                    itemTrait.AddAttribute(new InitialiseBackward());
                    userTrait.AddAttribute(new InitialiseBackward());
                    userBiasObs.AddAttribute(new InitialiseBackward());
                    itemBiasObs.AddAttribute(new InitialiseBackward());
                    userThresholdsObs.AddAttribute(new InitialiseBackward());
                }

                // Rating
                var rating = ratings[observation];
                using (Variable.Switch(rating))
                {
                    // This hack allows indexing the thresholds with the ratingValue range instead of the userThreshold range
                    var currentRating = (rating + 0).Named("CurrentRating");
                    var nextRating = (rating + 1).Named("NextRating");

                    Variable.ConstrainBetween(
                        noisyAffinity, noisyUserThresholds[currentRating], noisyUserThresholds[nextRating]);
                }
            }

            IVariable[] result;
            if (buildTrainingModel)
            {
                // Community training
                result = new IVariable[]
                             {
                                 userTraits, userBias, userThresholds, itemTraits, itemBias, userTraitFeatureWeights,
                                 userBiasFeatureWeights, itemTraitFeatureWeights, itemBiasFeatureWeights
                             };
            }
            else
            {
                // Prediction
                result = new IVariable[] { ratings };
            }

            return result;
        }
    }
}
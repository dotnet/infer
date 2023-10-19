// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests
{
#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    using System.Runtime.Serialization;
    using Xunit;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Models;
    using Microsoft.ML.Probabilistic.Utilities;
    using Microsoft.ML.Probabilistic.Serialization;
    using Assert = Xunit.Assert;
    using GaussianArray = Microsoft.ML.Probabilistic.Distributions.DistributionStructArray<Microsoft.ML.Probabilistic.Distributions.Gaussian, double>;
    using GaussianArrayArray =
        Microsoft.ML.Probabilistic.Distributions.DistributionRefArray<Microsoft.ML.Probabilistic.Distributions.DistributionStructArray<Microsoft.ML.Probabilistic.Distributions.Gaussian, double>, double[]>;
    using Microsoft.ML.Probabilistic.Models.Attributes;
    using Range = Microsoft.ML.Probabilistic.Models.Range;

    /// <summary>
    /// Matchbox recommender tests
    /// </summary>
    public class MatchboxTests
    {
        // Test a case where SHG finds a poor solution
        [Trait("Category", "OpenBug")]
        [Fact]
        public void MatchboxTest2()
        {
            var model = new MatchboxRecommender();
            model.fixAllDegreesOfFreedom = true;
            model.observationSequentialAttribute = true;
            //model.observationSequentialAttribute = false; // works
            model.Run(47);
            Assert.True(ValidateInferredTraits(model.trueItemTraits, model.learnedItemTraits));
        }

        /// <summary>
        /// Tests if Matchbox model is able to recover the true parameter values in different modes of operation.
        /// </summary>
        [Fact]
        public void MatchboxTest()
        {
            var model = new MatchboxRecommender();
            if (false)
            {
                // for debugging
                model.fixAllDegreesOfFreedom = true;
                model.observationSequentialAttribute = true;
                model.useArrayPriors = false;
                model.useFeatures = false;
                model.useSparseFeatures = false;
                model.initializeBackward = false;
                model.useBetween = true;
                model.learnThresholds = false;
                model.useInstanceWeights = false;
                model.Run(46);
                Assert.True(ValidateInferredTraits(model.trueItemTraits, model.learnedItemTraits));
                return;
            }

            for (int seed = 46; seed <= 46; seed++)
            {
                for (int useBetweenFlag = 1; useBetweenFlag < 2; useBetweenFlag++)
                {
                    // some cases do not work without observation sequential
                    for (int observationSequentialAttributeFlag = 1; observationSequentialAttributeFlag < 2; observationSequentialAttributeFlag++)
                    {
                        for (int useArrayPriorsFlag = 0; useArrayPriorsFlag < 2; useArrayPriorsFlag++)
                        {
                            for (int useFeaturesFlag = 0; useFeaturesFlag < 2; useFeaturesFlag++)
                            {
                                for (int initializeBackwardFlag = 0; initializeBackwardFlag < 2; initializeBackwardFlag++)
                                {
                                    if (initializeBackwardFlag == 1 && observationSequentialAttributeFlag == 0)
                                        continue;  // Not supported
                                    for (int instanceWeightsFlag = 0; instanceWeightsFlag < 2; instanceWeightsFlag++)
                                    {
                                        if (useArrayPriorsFlag > 0 && useFeaturesFlag > 0)
                                        {
                                            continue; // Not supported
                                        }
                                        model = new MatchboxRecommender();
                                        model.useBetween = useBetweenFlag > 0;
                                        model.fixAllDegreesOfFreedom = true;
                                        model.observationSequentialAttribute = observationSequentialAttributeFlag > 0;
                                        model.useArrayPriors = useArrayPriorsFlag > 0;
                                        model.useFeatures = useFeaturesFlag > 0;
                                        model.initializeBackward = initializeBackwardFlag > 0;
                                        model.useInstanceWeights = instanceWeightsFlag > 0;
                                        // sequential=false and learnThresholds=true doesn't work, even for 200 iterations
                                        // gets stuck in a bad local minimum
                                        //model.learnThresholds = true;

                                        Trace.WriteLine($"seed = {seed}");
                                        model.Run(seed);
                                        Assert.True(ValidateInferredTraits(model.trueItemTraits, model.learnedItemTraits));
                                        Trace.WriteLine("");
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Fixing only numTraits*(numTraits+1)/2 degrees of freedom fails in 179 cases out of 1000
            //(new MatchboxRecommender()).Run(83, false, false, out trueItemTraits, out trueUserFeatureWeights, out learnedItemTraits, out learnedUserFeatureWeights);
            //Assert.False(CompareTraits(trueItemTraits, learnedItemTraits));

            // Failure when not enough degrees of freedom are fixed is invariant of the Sequential attribute
            //(new MatchboxRecommender()).Run(83, false, true, out trueItemTraits, out learnedItemTraits);
            //Assert.False(compareTraits(trueItemTraits, learnedItemTraits));
        }

        /// <summary>
        /// Compares true parameters to learned ones
        /// </summary>
        /// <param name="trueTraits">Model data should be generated from these parameters</param>
        /// <param name="learnedTraits">Learned parameters array should be of the same size as the true one</param>
        /// <returns>Whether the learned parameters approximate well the ground truth</returns>
        private static bool ValidateInferredTraits(double[][] trueTraits, Gaussian[][] learnedTraits)
        {
            double maxDiff = double.NegativeInfinity;
            for (int i = 0; i < trueTraits.GetLength(0); ++i)
            {
                for (int j = 0; j < trueTraits[i].Length; ++j)
                {
                    double learnedTraitMean = learnedTraits[i][j].GetMean();
                    if (double.IsNaN(learnedTraitMean))
                    {
                        return false;
                    }

                    double deltaTraits = Math.Abs(trueTraits[i][j] - learnedTraitMean);
                    if (deltaTraits > maxDiff)
                    {
                        maxDiff = deltaTraits;
                    }
                }
            }
            Trace.WriteLine($"maxDiff = {maxDiff:0.0}");
            return (maxDiff < 0.7);
        }

        /// <summary>
        /// Represents the implementation of the Matchbox model.
        /// </summary>
        public class MatchboxRecommender
        {
            public int numUsers = 50;
            public int numItems = 50;
            public int numTraits = 2;
            public int numLevels = 4;
            // affinity noise must not be too large since this makes the problem too easy
            // affinity noise must not be too small since this slows down convergence
            public double affinityNoiseVariance = 0.1;
            public double thresholdNoiseVariance = 0.0;
            public double traitVariance;
            public double biasVariance = 0; //1.0;
            public int numUserFeatures;
            public int numItemFeatures;
            public bool fixAllDegreesOfFreedom;
            public bool observationSequentialAttribute;
            public bool useArrayPriors;
            public bool useFeatures;
            public bool initializeBackward;
            public bool useBetween;
            public bool learnThresholds;
            public bool useSparseFeatures;
            public bool useInstanceWeights;
            public bool uniqueObservations = true;
            public double[][] trueItemTraits, trueUserFeatureWeights;
            public Gaussian[][] learnedItemTraits, learnedUserTraitFeatureWeights;
            public int[] generatedUserData;
            public int[] generatedItemData;
            public bool[][] generatedRatingData;
            public double[][] userFeatures;
            public double[][] itemFeatures;

            public void WriteSettings()
            {
                if (useBetween)
                    Trace.WriteLine("useBetween");
                if (fixAllDegreesOfFreedom)
                    Trace.WriteLine("fixAllDegreesOfFreedom");
                if (observationSequentialAttribute)
                    Trace.WriteLine("observationSequentialAttribute");
                if (useArrayPriors)
                    Trace.WriteLine("useArrayPriors");
                if (useFeatures)
                    Trace.WriteLine($"useFeatures {(useSparseFeatures ? "sparse" : "")}");
                if (initializeBackward)
                    Trace.WriteLine("initializeBackward");
                if (learnThresholds)
                    Trace.WriteLine("learnThresholds");
                if (useInstanceWeights)
                    Trace.WriteLine("useInstanceWeights");
            }

            /// <summary>
            /// Trains and runs a Matchbox recommender
            /// </summary>
            /// <param name="randomSeed">The random seed to be initially set</param>
            public void Run(int randomSeed)
            {
                Debug.Assert(
                    !(useArrayPriors && useFeatures),
                    "This test is not set up to work with both array priors and features.");

                Rand.Restart(randomSeed);
                WriteSettings();

                // Define counts
                Variable<int> numObservations = Variable.Observed(numUsers * numItems).Named("numObservations");
                if (!useFeatures)
                {
                    numUserFeatures = 0;
                    numItemFeatures = 0;
                    traitVariance = 1.0;
                }
                else
                {
                    // must have at least as many features as traits in order to get recovery
                    numUserFeatures = numTraits;
                    numItemFeatures = numTraits;
                    traitVariance = 1.0 / (numItemFeatures + 1);
                }

                var evidence = Variable.Bernoulli(0.5).Named("evidence");
                evidence.ObservedValue = true;
                evidence.IsReadOnly = true;
                var block = Variable.If(evidence);

                // Define ranges
                Range user = new Range(numUsers).Named("user");
                Range item = new Range(numItems).Named("item");
                Range trait = new Range(numTraits).Named("trait");
                Range observation = new Range(numObservations).Named("observation");
                Range level = new Range(numLevels).Named("level");
                Range userFeature = new Range(Variable.Observed(numUserFeatures)).Named("userFeature");
                Range itemFeature = new Range(Variable.Observed(numItemFeatures)).Named("itemFeature");

                // Latent variables
                var userTraits = Variable.Array(Variable.Array<double>(trait), user).Named("userTraits");
                var itemTraits = Variable.Array(Variable.Array<double>(trait), item).Named("itemTraits");
                var userBias = Variable.Array<double>(user).Named("userBias");
                var itemBias = Variable.Array<double>(item).Named("itemBias");
                var userThresholds = Variable.Array(Variable.Array<double>(level), user).Named("userThresholds");

                // Feature-related variables
                var userTraitFeatureWeights = Variable.Array(Variable.Array<double>(userFeature), trait).Named("userTraitFeatureWeights");
                var itemTraitFeatureWeights = Variable.Array(Variable.Array<double>(itemFeature), trait).Named("itemTraitFeatureWeights");
                var userBiasFeatureWeights = Variable.Array<double>(userFeature).Named("userBiasFeatureWeights");
                var itemBiasFeatureWeights = Variable.Array<double>(itemFeature).Named("itemBiasFeatureWeights");

                // Training data variables
                var userData = Variable.Array<int>(observation).Named("userData");
                var itemData = Variable.Array<int>(observation).Named("itemData");
                var ratingData = Variable.Array(Variable.Array<bool>(level), observation).Named("ratingData");
                var ratingInt = Variable.Array<int>(observation).Named("ratingInt");
                var userFeatureData = Variable.Array(Variable.Array<double>(userFeature), user).Named("userFeatureData");
                var itemFeatureData = Variable.Array(Variable.Array<double>(itemFeature), item).Named("itemFeatureData");
                var instanceWeights = Variable.Array<double>(observation).Named("instanceWeights");
                instanceWeights[observation] = Variable.Constant(1.0);
                var nonZeroItemFeatureCounts = Variable.Array<int>(item).Named("nonZeroItemFeatureCounts");
                var nonZeroItemFeature = new Range(nonZeroItemFeatureCounts[item]).Named("nonZeroItemFeature");
                var nonZeroItemFeatureIndices = Variable.Array(Variable.Array<int>(nonZeroItemFeature), item).Named("nonZeroItemFeatureIndices");
                var nonZeroItemFeatureValues = Variable.Array(Variable.Array<double>(nonZeroItemFeature), item).Named("nonZeroItemFeatureValues");
                var nonZeroUserFeatureCounts = Variable.Array<int>(user).Named("nonZeroUserFeatureCounts");
                var nonZeroUserFeature = new Range(nonZeroUserFeatureCounts[user]).Named("nonZeroUserFeature");
                var nonZeroUserFeatureIndices = Variable.Array(Variable.Array<int>(nonZeroUserFeature), user).Named("nonZeroUserFeatureIndices");
                var nonZeroUserFeatureValues = Variable.Array(Variable.Array<double>(nonZeroUserFeature), user).Named("nonZeroUserFeatureValues");

                Gaussian traitWeightPrior = Gaussian.FromMeanAndVariance(0.0, 1.0 / (numItemFeatures + 1));
                Gaussian biasWeightPrior = Gaussian.FromMeanAndVariance(0.0, 0.0);
                double thresholdPriorVariance = learnThresholds ? 0.1 : 0;
                Gaussian[] userThresholdPrior = Util.ArrayInit(numLevels, l => Gaussian.FromMeanAndVariance(l - (numLevels / 2.0) + 0.5, thresholdPriorVariance));
                if (useBetween)
                {
                    if (numLevels < 3)
                        throw new ArgumentException("numLevels (" + numLevels + ") < 3");
                    userThresholdPrior[0] = Gaussian.PointMass(double.NegativeInfinity);
                    userThresholdPrior[numLevels - 1] = Gaussian.PointMass(double.PositiveInfinity);
                }

                // Regular priors
                var userThresholdsPrior = Variable.Array(Variable.Array<Gaussian>(level), user).Named("userThresholdsPrior");
                var userTraitFeatureWeightPrior = Variable.Array(Variable.Array<Gaussian>(userFeature), trait).Named("userTraitFeatureWeightPrior");
                var userBiasFeatureWeightPrior = Variable.Array<Gaussian>(userFeature).Named("userBiasFeatureWeightPrior");
                var itemFeatureWeightsPrior = Variable.Array(Variable.Array<Gaussian>(itemFeature), trait).Named("itemFeatureWeightsPrior");
                var itemFeatureBiasWeightsPrior = Variable.Array<Gaussian>(itemFeature).Named("itemFeatureBiasWeightsPrior");

                // Array priors
                var userTraitsArrayPrior = Variable.New<GaussianArrayArray>().Named("userTraitsArrayPrior");
                var itemTraitsArrayPrior = Variable.New<GaussianArrayArray>().Named("itemTraitsArrayPrior");
                var userBiasArrayPrior = Variable.New<GaussianArray>().Named("userBiasArrayPrior");
                var itemBiasArrayPrior = Variable.New<GaussianArray>().Named("itemBiasArrayPrior");
                var userThresholdsArrayPrior = Variable.New<GaussianArrayArray>().Named("userThresholdsArrayPrior");

                // Initialise regular priors
                userThresholdsPrior.ObservedValue = Util.ArrayInit(numUsers, u => userThresholdPrior);
                userTraitFeatureWeightPrior.ObservedValue = Util.ArrayInit(numTraits, t => Util.ArrayInit(numUserFeatures, f => traitWeightPrior));
                userBiasFeatureWeightPrior.ObservedValue = Util.ArrayInit(numUserFeatures, f => biasWeightPrior);
                itemFeatureWeightsPrior.ObservedValue = Util.ArrayInit(numTraits, t => Util.ArrayInit(numItemFeatures, f => traitWeightPrior));
                itemFeatureBiasWeightsPrior.ObservedValue = Util.ArrayInit(numItemFeatures, f => biasWeightPrior);

                // Initialize array priors
                userTraitsArrayPrior.ObservedValue = new GaussianArrayArray(numUsers, u => new GaussianArray(numTraits, t => new Gaussian(0, traitVariance)));
                itemTraitsArrayPrior.ObservedValue = new GaussianArrayArray(numItems, i => new GaussianArray(numTraits, t => new Gaussian(0, traitVariance)));
                userBiasArrayPrior.ObservedValue = new GaussianArray(numUsers, u => new Gaussian(0, biasVariance));
                itemBiasArrayPrior.ObservedValue = new GaussianArray(numItems, i => new Gaussian(0, biasVariance));
                userThresholdsArrayPrior.ObservedValue = new GaussianArrayArray(numUsers, u => new GaussianArray(userThresholdPrior));

                // Define feature weights
                userTraitFeatureWeights[trait][userFeature] = Variable<double>.Random(userTraitFeatureWeightPrior[trait][userFeature]);
                itemTraitFeatureWeights[trait][itemFeature] = Variable<double>.Random(itemFeatureWeightsPrior[trait][itemFeature]);
                userBiasFeatureWeights[userFeature] = Variable<double>.Random(userBiasFeatureWeightPrior[userFeature]);
                itemBiasFeatureWeights[itemFeature] = Variable<double>.Random(itemFeatureBiasWeightsPrior[itemFeature]);

                // Define latent variables
                if (useArrayPriors)
                {
                    userTraits.SetTo(Variable<double[][]>.Random(userTraitsArrayPrior));
                    itemTraits.SetTo(Variable<double[][]>.Random(itemTraitsArrayPrior));
                    userBias.SetTo(Variable<double[]>.Random(userBiasArrayPrior));
                    itemBias.SetTo(Variable<double[]>.Random(itemBiasArrayPrior));
                    userThresholds.SetTo(Variable<double[][]>.Random(userThresholdsArrayPrior));
                }
                else
                {
                    const double stepsize = 1;
                    // Define user latent variables
                    using (Variable.ForEach(user))
                    {
                        const double UserTraitFeatureWeightDamping = stepsize;
                        const double UserBiasFeatureWeightDamping = stepsize;

                        if (useSparseFeatures)
                        {
                            var nonZeroUserTraitFeatureWeights = Variable.Subarray(userTraitFeatureWeights[trait], nonZeroUserFeatureIndices[user]);
                            var userTraitFeatureWeightsDamped = Variable<double>.Factor<double, double>(
                                Damp.Backward, nonZeroUserTraitFeatureWeights[nonZeroUserFeature], UserTraitFeatureWeightDamping);
                            var userTraitFeatureWeightProducts = Variable.Array(Variable.Array<double>(nonZeroUserFeature), trait);
                            userTraitFeatureWeightProducts[trait][nonZeroUserFeature] = userTraitFeatureWeightsDamped * nonZeroUserFeatureValues[user][nonZeroUserFeature];
                            userTraits[user][trait] = Variable.GaussianFromMeanAndVariance(Variable.Sum(userTraitFeatureWeightProducts[trait]), traitVariance);

                            var nonZeroUserBiasFeatureWeights = Variable.Subarray(userBiasFeatureWeights, nonZeroUserFeatureIndices[user]);
                            var userBiasFeatureWeightsDamped = Variable<double>.Factor<double, double>(
                                Damp.Backward, nonZeroUserBiasFeatureWeights[nonZeroUserFeature], UserBiasFeatureWeightDamping);
                            var userBiasFeatureWeightProducts = Variable.Array<double>(nonZeroUserFeature);
                            userBiasFeatureWeightProducts[nonZeroUserFeature] = userBiasFeatureWeightsDamped * nonZeroUserFeatureValues[user][nonZeroUserFeature];
                            userBias[user] = Variable.GaussianFromMeanAndVariance(Variable.Sum(userBiasFeatureWeightProducts), biasVariance);
                        }
                        else if (useFeatures)
                        {
                            var userTraitFeatureWeightsDamped = Variable<double>.Factor<double, double>(
                                Damp.Backward, userTraitFeatureWeights[trait][userFeature], UserTraitFeatureWeightDamping);
                            var userTraitFeatureWeightProducts = Variable.Array(Variable.Array<double>(userFeature), trait);
                            userTraitFeatureWeightProducts[trait][userFeature] = userTraitFeatureWeightsDamped * userFeatureData[user][userFeature];
                            userTraits[user][trait] = Variable.GaussianFromMeanAndVariance(Variable.Sum(userTraitFeatureWeightProducts[trait]), traitVariance);

                            var userBiasFeatureWeightsDamped = Variable<double>.Factor<double, double>(
                                Damp.Backward, userBiasFeatureWeights[userFeature], UserBiasFeatureWeightDamping);
                            var userBiasFeatureWeightProducts = Variable.Array<double>(userFeature).Named("userBiasFeatureWeightProducts");
                            userBiasFeatureWeightProducts[userFeature] = userBiasFeatureWeightsDamped * userFeatureData[user][userFeature];
                            var userBiasMean = Variable.Sum(userBiasFeatureWeightProducts).Named("userBiasMean");
                            userBias[user] = Variable.GaussianFromMeanAndVariance(userBiasMean, biasVariance);
                        }
                        else
                        {
                            userTraits[user][trait] = Variable.GaussianFromMeanAndVariance(0, traitVariance).ForEach(trait);
                            userBias[user] = Variable.GaussianFromMeanAndVariance(0, biasVariance);
                        }

                        userThresholds[user][level] = Variable<double>.Random(userThresholdsPrior[user][level]);
                    }

                    // Define item latent variables
                    using (Variable.ForEach(item))
                    {
                        const double ItemTraitFeatureWeightDamping = stepsize;
                        const double ItemBiasFeatureWeightDamping = stepsize;

                        if (useSparseFeatures)
                        {
                            var nonZeroItemTraitFeatureWeights = Variable.Subarray(itemTraitFeatureWeights[trait], nonZeroItemFeatureIndices[item]);
                            var itemTraitFeatureWeightsDamped = Variable<double>.Factor<double, double>(
                                Damp.Backward, nonZeroItemTraitFeatureWeights[nonZeroItemFeature], ItemTraitFeatureWeightDamping);
                            var itemTraitFeatureWeightProducts = Variable.Array(Variable.Array<double>(nonZeroItemFeature), trait);
                            itemTraitFeatureWeightProducts[trait][nonZeroItemFeature] = itemTraitFeatureWeightsDamped * nonZeroItemFeatureValues[item][nonZeroItemFeature];
                            itemTraits[item][trait] = Variable.GaussianFromMeanAndVariance(Variable.Sum(itemTraitFeatureWeightProducts[trait]), traitVariance);

                            var nonZeroItemBiasFeatureWeights = Variable.Subarray(itemBiasFeatureWeights, nonZeroItemFeatureIndices[item]);
                            var itemBiasFeatureWeightsDamped = Variable<double>.Factor<double, double>(
                                Damp.Backward, nonZeroItemBiasFeatureWeights[nonZeroItemFeature], ItemBiasFeatureWeightDamping);
                            var itemBiasFeatureWeightProducts = Variable.Array<double>(nonZeroItemFeature).Named("itemBiasFeatureWeightProducts");
                            itemBiasFeatureWeightProducts[nonZeroItemFeature] = itemBiasFeatureWeightsDamped * nonZeroItemFeatureValues[item][nonZeroItemFeature];
                            itemBias[item] = Variable.GaussianFromMeanAndVariance(Variable.Sum(itemBiasFeatureWeightProducts), biasVariance);
                        }
                        else if (useFeatures)
                        {
                            var itemTraitFeatureWeightsDamped = Variable<double>.Factor<double, double>(
                                Damp.Backward, itemTraitFeatureWeights[trait][itemFeature], ItemTraitFeatureWeightDamping);
                            var itemTraitFeatureWeightProducts = Variable.Array(Variable.Array<double>(itemFeature), trait);
                            itemTraitFeatureWeightProducts[trait][itemFeature] = itemTraitFeatureWeightsDamped * itemFeatureData[item][itemFeature];
                            itemTraits[item][trait] = Variable.GaussianFromMeanAndVariance(Variable.Sum(itemTraitFeatureWeightProducts[trait]), traitVariance);

                            var itemBiasFeatureWeightsDamped = Variable<double>.Factor<double, double>(
                                Damp.Backward, itemBiasFeatureWeights[itemFeature], ItemBiasFeatureWeightDamping);
                            var itemBiasFeatureWeightProducts = Variable.Array<double>(itemFeature).Named("itemBiasFeatureWeightProducts");
                            itemBiasFeatureWeightProducts[itemFeature] = itemBiasFeatureWeightsDamped * itemFeatureData[item][itemFeature];
                            itemBias[item] = Variable.GaussianFromMeanAndVariance(Variable.Sum(itemBiasFeatureWeightProducts), biasVariance);
                        }
                        else
                        {
                            itemTraits[item][trait] = Variable.GaussianFromMeanAndVariance(0, traitVariance).ForEach(trait);
                            itemBias[item] = Variable.GaussianFromMeanAndVariance(0, biasVariance);
                        }
                    }
                }

                var userTraitsTransposed = Variable.Array(Variable.Array<double>(user), trait).Named("userTraitsTransposed");
                userTraitsTransposed[trait][user] = Variable.Copy(userTraits[user][trait]);
                var itemTraitsTransposed = Variable.Array(Variable.Array<double>(item), trait).Named("itemTraitsTransposed");
                itemTraitsTransposed[trait][item] = Variable.Copy(itemTraits[item][trait]);

                // Model
                using (Variable.ForEach(observation))
                {
                    RepeatBlock repeatBlock = null;
                    if (useInstanceWeights)
                        repeatBlock = Variable.Repeat(instanceWeights[observation]);
                    var userId = userData[observation];
                    var itemId = itemData[observation];

                    VariableArray<double> products = Variable.Array<double>(trait).Named("products");
                    if (observationSequentialAttribute)
                    {
                        var itemTrait = Variable.Copy(itemTraits[itemId][trait]).Named("itemTrait");
                        var userTrait = Variable.Copy(userTraits[userId][trait]).Named("userTrait");
                        if (initializeBackward)
                        {
                            itemTrait.AddAttribute(new InitialiseBackward());
                            userTrait.AddAttribute(new InitialiseBackward());
                        }
                        products[trait] = Variable<double>.Factor(Factor.Product_SHG09, userTrait, itemTrait);
                        //products[trait] = userTraits[userId][trait] * itemTraits[itemId][trait];
                        //var alg = new VariationalMessagePassing();
                        //var attr = new Algorithm(alg);
                        //userTraits.AddAttribute(attr);
                        //itemTraits.AddAttribute(attr);
                        //products.AddAttribute(new FactorAlgorithm(alg));
                    }
                    else
                        products[trait] = Variable<double>.Factor(Factor.Product_SHG09, userTraitsTransposed[trait][userId], itemTraitsTransposed[trait][itemId]);
                    var itemBiasObs = Variable.Copy(itemBias[itemId]).Named("itemBiasObs");
                    var userBiasObs = Variable.Copy(userBias[userId]).Named("userBiasObs");
                    if (initializeBackward)
                    {
                        itemBiasObs.AddAttribute(new InitialiseBackward());
                        userBiasObs.AddAttribute(new InitialiseBackward());
                    }
                    //var bias = Variable<double>.Factor(Factor.Product_SHG09, userBiasObs, 1.0)
                    //            + Variable<double>.Factor(Factor.Product_SHG09, itemBiasObs, 1.0);
                    var bias = userBiasObs + itemBiasObs;
                    bias.Name = "bias";
                    // damping products allows us to use a non-sequential schedule for observations
                    var productsDamped = Variable<double>.Array(trait);
                    productsDamped[trait] = Variable<double>.Factor(Damp.Forward<double>, products[trait], observationSequentialAttribute ? 1.0 : 2 * 0.5);
                    var affinity = (bias + Variable.Sum(products).Named("productSum")).Named("affinity");
                    var noisyAffinity = Variable.GaussianFromMeanAndVariance(affinity, affinityNoiseVariance).Named("noisyAffinity");
                    var noisyThresholds = Variable.Array<double>(level).Named("noisyThresholds");
                    noisyThresholds[level] = Variable.GaussianFromMeanAndVariance(userThresholds[userId][level], thresholdNoiseVariance);
                    if (useBetween)
                    {
                        var r = ratingInt[observation];
                        var ratingPlus1 = (r + 1).Named("ratingPlus1");
                        Variable.ConstrainBetween(noisyAffinity, noisyThresholds[r], noisyThresholds[ratingPlus1]);
                    }
                    else
                    {
                        ratingData[observation][level] = (noisyAffinity - noisyThresholds[level]).Named("diff") > 0;
                    }
                    if (useInstanceWeights)
                        repeatBlock.CloseBlock();
                }

                // Generate and observe training data
                this.GenerateData(
                    numObservations.ObservedValue,
                    userThresholdsPrior.ObservedValue,
                    userTraitFeatureWeightPrior.ObservedValue, itemFeatureWeightsPrior.ObservedValue,
                    userBiasFeatureWeightPrior.ObservedValue, itemFeatureBiasWeightsPrior.ObservedValue);

                userData.ObservedValue = generatedUserData;
                itemData.ObservedValue = generatedItemData;
                ratingData.ObservedValue = generatedRatingData;
                if (useBetween)
                {
                    ratingInt.ObservedValue = Util.ArrayInit(numObservations.ObservedValue, i => RatingFromBools(generatedRatingData[i]) - 1);
                }

                if (useFeatures)
                {
                    userFeatureData.ObservedValue = userFeatures;
                    itemFeatureData.ObservedValue = itemFeatures;
                    nonZeroUserFeatureCounts.ObservedValue = Util.ArrayInit(numUsers, i => numUserFeatures);
                    nonZeroUserFeatureIndices.ObservedValue = Util.ArrayInit(numUsers, i => Util.ArrayInit(numUserFeatures, f => f));
                    nonZeroUserFeatureValues.ObservedValue = Util.ArrayInit(numUsers, i => Util.ArrayInit(numUserFeatures, f => userFeatures[i][f]));
                    nonZeroItemFeatureCounts.ObservedValue = Util.ArrayInit(numItems, i => numItemFeatures);
                    nonZeroItemFeatureIndices.ObservedValue = Util.ArrayInit(numItems, i => Util.ArrayInit(numItemFeatures, f => f));
                    nonZeroItemFeatureValues.ObservedValue = Util.ArrayInit(numItems, i => Util.ArrayInit(numItemFeatures, f => itemFeatures[i][f]));
                }
                else
                {
                    userFeatureData.ObservedValue = Util.ArrayInit(numUsers, u => new double[0]);
                    itemFeatureData.ObservedValue = Util.ArrayInit(numItems, i => new double[0]);
                }

                // Break trait symmetry
                for (int i = 0; i < Math.Min(numTraits, numItems); ++i)
                {
                    int maxJ = fixAllDegreesOfFreedom ? Math.Min(numTraits, numItems) : i + 1;
                    for (int j = 0; j < maxJ; ++j)
                    {
                        if (useArrayPriors)
                        {
                            itemTraitsArrayPrior.ObservedValue[i][j] = Gaussian.PointMass(trueItemTraits[i][j]);
                        }
                        else
                        {
                            Variable.ConstrainEqual(itemTraits[i][j], trueItemTraits[i][j]);
                        }
                    }
                }

                block.CloseBlock();

                // Create engine and set its parameters
                InferenceEngine engine = new InferenceEngine();
                engine.ModelName = "MatchboxTest";
                engine.ShowProgress = false;
                engine.Compiler.FreeMemory = false;
                // the test also passes with these operators:
                //engine.Compiler.GivePriorityTo(typeof(GaussianProductOp_Slow));
                //engine.Compiler.GivePriorityTo(typeof(GaussianProductOp_Laplace2));
                if (observationSequentialAttribute)
                {
                    user.AddAttribute(new Sequential());
                    item.AddAttribute(new Sequential());
                    observation.AddAttribute(new Sequential());
                }
                else
                {
                    engine.Compiler.GivePriorityTo(typeof(SumOp_SHG09));
                    //trait.AddAttribute(new Sequential());
                }
                var toInfer = new List<IVariable>() { userTraits, userBias, itemTraits, itemBias, userThresholds, evidence, userTraitFeatureWeights, itemTraitFeatureWeights };
                engine.OptimiseForVariables = toInfer;
                if (initializeBackward)
                {
                    if (false)
                    {
                        var userTraitsInitializer = Variable.Observed(default(GaussianArrayArray)).Named("UserTraitsInitializer");
                        var userBiasInitializer = Variable.Observed(default(GaussianArray)).Named("UserBiasInitializer");
                        var userThresholdsInitializer = Variable.Observed(default(GaussianArrayArray)).Named("UserThresholdsInitializer");
                        userTraits.InitialiseTo(userTraitsInitializer);
                        userBias.InitialiseTo(userBiasInitializer);
                        userThresholds.InitialiseTo(userThresholdsInitializer);
                        var itemTraitsInitializer = Variable.Observed(default(GaussianArrayArray)).Named("ItemTraitsInitializer");
                        var itemBiasInitializer = Variable.Observed(default(GaussianArray)).Named("ItemBiasInitializer");
                        itemTraits.InitialiseTo(itemTraitsInitializer);
                        itemBias.InitialiseTo(itemBiasInitializer);
                    }

                    engine.Compiler.UseSpecialFirstIteration = true; // reconstruct backward messages whenever the data changes
                    engine.Compiler.AllowSerialInitialisers = true; // required by UseSpecialFirst
                }
                //itemTraits.AddAttribute(new PointEstimate());
                //userTraits.AddAttribute(new PointEstimate());
                //userBias.AddAttribute(new PointEstimate());
                //itemBias.AddAttribute(new PointEstimate());

                // Run inference
                IDistribution<double[][]> prevItemTraitsPost = null, prevItemFeatureWeightsPost = null;
                object itemTraitsPost1 = null;
                Stopwatch watch = new Stopwatch();
                for (int iter = 1; iter < 1000; iter++)
                {
                    engine.NumberOfIterations = iter;
                    watch.Reset();
                    watch.Start();
                    var itemTraitsPost = engine.Infer<IDistribution<double[][]>>(itemTraits);
                    var itemTraitFeatureWeightPost = engine.Infer<IDistribution<double[][]>>(itemTraitFeatureWeights);
                    watch.Stop();
                    if (iter > 2)
                    {
                        double delta = itemTraitsPost.MaxDiff(prevItemTraitsPost);
                        delta = Math.Max(delta, itemTraitFeatureWeightPost.MaxDiff(prevItemFeatureWeightsPost));
                        Trace.WriteLine($"{iter}: delta = {delta.ToString("g4")} time = {watch.ElapsedMilliseconds}ms");
                        if (delta < 1e-3)
                            break;
                        //Trace.WriteLine("{0} {1}", ((GaussianArrayArray)itemTraitsPost)[10][0], ((GaussianArrayArray)prevItemTraitsPost)[10][0]);
                    }
                    prevItemTraitsPost = (IDistribution<double[][]>)itemTraitsPost.Clone();
                    prevItemFeatureWeightsPost = (IDistribution<double[][]>)itemTraitFeatureWeightPost.Clone();
                    if (iter == 1) itemTraitsPost1 = prevItemTraitsPost;
                }
                var userTraitsPosterior = engine.Infer<Gaussian[][]>(userTraits);
                var itemTraitsPosterior = engine.Infer<Gaussian[][]>(itemTraits);
                learnedItemTraits = itemTraitsPosterior;
                var userBiasPosterior = engine.Infer<Gaussian[]>(userBias);
                var itemBiasPosterior = engine.Infer<Gaussian[]>(itemBias);
                var userThresholdsPosterior = engine.Infer<Gaussian[][]>(userThresholds);
                var userTraitFeatureWeightPosterior = engine.Infer<Gaussian[][]>(userTraitFeatureWeights);
                learnedUserTraitFeatureWeights = userTraitFeatureWeightPosterior;

                if (false)
                {
                    Trace.WriteLine("Learned user traits: ");
                    for (int i = 0; i < Math.Min(numUsers, 4); ++i)
                    {
                        for (int j = 0; j < numTraits; ++j)
                            Trace.Write(userTraitsPosterior[i][j].GetMean() + "\t");
                        Trace.WriteLine("");
                    }
                }
                if (numUserFeatures > 0)
                {
                    Trace.WriteLine("Learned user trait feature weights:");
                    for (int i = 0; i < Math.Min(numUserFeatures, 4); ++i)
                    {
                        for (int j = 0; j < numTraits; ++j)
                        {
                            Trace.Write(userTraitFeatureWeightPosterior[j][i].GetMean() + "\t");
                        }
                        Trace.WriteLine("");
                    }
                }
                Trace.WriteLine("Learned item traits: ");
                for (int i = 0; i < Math.Min(4, numItems); ++i)
                {
                    for (int j = 0; j < numTraits; ++j)
                        Trace.Write(itemTraitsPosterior[i][j].GetMean() + "\t");
                    Trace.WriteLine("");
                }
                if (learnThresholds)
                {
                    Trace.WriteLine("Learned thresholds:");
                    for (int i = 0; i < Math.Min(numUsers, 4); ++i)
                    {
                        for (int j = 0; j < numLevels; ++j)
                            Trace.Write(userThresholdsPosterior[i][j].GetMean() + "\t");
                        Trace.WriteLine("");
                    }
                }
                Trace.WriteLine($"evidence = {engine.Infer<Bernoulli>(evidence).LogOdds}");

                if (false)
                {
                    for (int i = 0; i < numItems; i++)
                    {
                        for (int j = 0; j < numItems; j++)
                        {
                            var prod0 = GaussianProductVmpOp.ProductAverageLogarithm(userTraitsPosterior[i][0], itemTraitsPosterior[j][0]);
                            var prod1 = GaussianProductVmpOp.ProductAverageLogarithm(userTraitsPosterior[i][1], itemTraitsPosterior[j][1]);
                            Trace.WriteLine($"affinity({i},{j}) = {DoublePlusOp.SumAverageConditional(prod0, prod1)}");
                        }
                    }
                }

                if (false)
                {
                    var serializer = new DataContractSerializer(typeof(DistributionRefArray<DistributionStructArray<Gaussian, double>, double[]>), new DataContractSerializerSettings { DataContractResolver = new InferDataContractResolver() });
                    using (var writer = new FileStream("userTraits.bin", FileMode.Create))
                    {
                        serializer.WriteObject(writer, (DistributionRefArray<DistributionStructArray<Gaussian, double>, double[]>)engine.Infer(userTraits));
                    }
                    using (var writer = new FileStream("itemTraits.bin", FileMode.Create))
                    {
                        serializer.WriteObject(writer, (DistributionRefArray<DistributionStructArray<Gaussian, double>, double[]>)engine.Infer(itemTraits));
                    }
                }

                // test resetting inference
                if (engine.Compiler.ReturnCopies)
                {
                    engine.NumberOfIterations = 1;
                    var itemTraitsPost2 = engine.Infer<Diffable>(itemTraits);
                    Assert.True(itemTraitsPost2.MaxDiff(itemTraitsPost1) < 1e-10);
                }
            }

            /// <summary>
            /// Generates data from the Matchbox model.
            /// </summary>
            private void GenerateData(
                int numObservations,
                Gaussian[][] userThresholdsPrior,
                Gaussian[][] userTraitFeatureWeightPrior,
                Gaussian[][] itemTraitFeatureWeightPrior,
                Gaussian[] userBiasFeatureWeightPrior,
                Gaussian[] itemBiasFeatureWeightPrior)
            {
                generatedUserData = new int[numObservations];
                generatedItemData = new int[numObservations];
                generatedRatingData = new bool[numObservations][];

                // Generate feature weights
                double[][] userFeatureWeights = Util.ArrayInit(numTraits, t => Util.ArrayInit(numUserFeatures, f => userTraitFeatureWeightPrior[t][f].Sample()));
                double[][] itemFeatureWeights = Util.ArrayInit(numTraits, t => Util.ArrayInit(numItemFeatures, f => itemTraitFeatureWeightPrior[t][f].Sample()));
                double[] userBiasFeatureWeights = Util.ArrayInit(numUserFeatures, f => userBiasFeatureWeightPrior[f].Sample());
                double[] itemBiasFeatureWeights = Util.ArrayInit(numItemFeatures, f => itemBiasFeatureWeightPrior[f].Sample());

                // Generate features
                userFeatures = Util.ArrayInit(numUsers, u => Util.ArrayInit(numUserFeatures, f => Rand.Double()));
                itemFeatures = Util.ArrayInit(numItems, i => Util.ArrayInit(numItemFeatures, f => Rand.Double()));

                // Sample model parameters from the priors
                double[][] userTraits = Util.ArrayInit(
                    numUsers,
                    u => Util.ArrayInit(
                        numTraits,
                        t => Gaussian.Sample(Util.ArrayInit(numUserFeatures, f => (userFeatureWeights[t][f] * userFeatures[u][f])).Sum(), 1.0 / traitVariance)));
                double[][] itemTraits = Util.ArrayInit(
                    numItems,
                    i => Util.ArrayInit(
                        numTraits,
                        t => Gaussian.Sample(Util.ArrayInit(numItemFeatures, f => itemFeatureWeights[t][f] * itemFeatures[i][f]).Sum(), 1.0 / traitVariance)));

                double[] userBias = Util.ArrayInit(
                    numUsers,
                    u => Gaussian.Sample(Util.ArrayInit(numUserFeatures, f => userBiasFeatureWeights[f] * userFeatures[u][f]).Sum(), 1.0 / biasVariance));
                double[] itemBias = Util.ArrayInit(
                    numItems,
                    i => Gaussian.Sample(Util.ArrayInit(numItemFeatures, f => itemBiasFeatureWeights[f] * itemFeatures[i][f]).Sum(), 1.0 / biasVariance));
                double[][] userThresholds = Util.ArrayInit(numUsers, u => Util.ArrayInit(userThresholdsPrior[u].Length, l => userThresholdsPrior[u][l].Sample()));

                trueUserFeatureWeights = userFeatureWeights;
                trueItemTraits = itemTraits;
                if (true)
                {
                    MeanVarianceAccumulator mva = new MeanVarianceAccumulator();
                    for (int i = 0; i < itemTraits.Length; i++)
                    {
                        for (int j = 0; j < itemTraits[i].Length; j++)
                        {
                            mva.Add(itemTraits[i][j]);
                        }
                    }
                    Trace.WriteLine($"itemTraits variance = {mva.Variance}");
                }

                if (false)
                {
                    Trace.WriteLine("True user traits: ");
                    for (int i = 0; i < Math.Min(numUsers, 4); ++i)
                    {
                        for (int j = 0; j < numTraits; ++j)
                            Trace.Write(userTraits[i][j] + "\t");
                        Trace.WriteLine("");
                    }
                }
                if (numUserFeatures > 0)
                {
                    Trace.WriteLine("True user trait feature weights: ");
                    for (int i = 0; i < Math.Min(numUserFeatures, 4); ++i)
                    {
                        for (int j = 0; j < numTraits; ++j)
                        {
                            Trace.Write(userFeatureWeights[j][i] + "\t");
                        }
                        Trace.WriteLine("");
                    }
                }
                Trace.WriteLine("True item traits: ");
                for (int i = 0; i < Math.Min(4, numItems); ++i)
                {
                    for (int j = 0; j < numTraits; ++j)
                        Trace.Write(itemTraits[i][j] + "\t");
                    Trace.WriteLine("");
                }
                if (learnThresholds)
                {
                    Trace.WriteLine("True user thresholds: ");
                    for (int i = 0; i < Math.Min(4, numUsers); ++i)
                    {
                        for (int j = 0; j < numLevels; ++j)
                            Trace.Write(userThresholds[i][j] + "\t");
                        Trace.WriteLine("");
                    }
                }
                if (false)
                {
                    for (int i = 0; i < numItems; i++)
                    {
                        for (int j = 0; j < numItems; j++)
                        {
                            Trace.WriteLine($"affinity({i},{j}) = {userTraits[i][0] * itemTraits[j][0] + userTraits[i][1] * itemTraits[j][1]}");
                        }
                    }
                }

                // Repeat the model with fixed parameters
                HashSet<int> visited = new HashSet<int>();
                for (int observation = 0; observation < numObservations; observation++)
                {
                    int user = Rand.Int(numUsers);
                    int item = Rand.Int(numItems);

                    int userItemPairID = user * numItems + item; // pair encoding
                    if (visited.Contains(userItemPairID) && uniqueObservations) // duplicate generated
                    {
                        observation--; // reject pair
                        continue;
                    }
                    visited.Add(userItemPairID);

                    double[] products = Util.ArrayInit(numTraits, t => userTraits[user][t] * itemTraits[item][t]);
                    double bias = userBias[user] + itemBias[item];

                    double affinity = bias + products.Sum();
                    double noisyAffinity = new Gaussian(affinity, affinityNoiseVariance).Sample();
                    double[] noisyThresholds = Util.ArrayInit(userThresholds[user].Length, l => new Gaussian(userThresholds[user][l], 0.0).Sample());

                    generatedUserData[observation] = user;
                    generatedItemData[observation] = item;
                    generatedRatingData[observation] = Util.ArrayInit(numLevels, l => noisyAffinity > noisyThresholds[l]);
                }
                //WriteData(@"..\..\..\..\Prototypes\Matchbox\data.csv");
            }

            public void WriteData(string filename)
            {
                using (StreamWriter writer = new StreamWriter(filename))
                {
                    for (int i = 0; i < generatedRatingData.Length; i++)
                    {
                        writer.Write(generatedItemData[i]);
                        writer.Write(",");
                        writer.Write(generatedUserData[i]);
                        writer.Write(",");
                        int r = RatingFromBools(generatedRatingData[i]) + 1;
                        writer.WriteLine(r);
                    }
                }
            }

            public static int RatingFromBools(bool[] greaterThan)
            {
                for (int i = 0; i < greaterThan.Length; i++)
                {
                    if (!greaterThan[i])
                        return i;
                }
                return greaterThan.Length;
            }

            /// <summary>
            /// Randomly perturbs a Gaussian distribution.
            /// </summary>
            /// <param name="dist">The distribution to perturb.</param>
            /// <returns>The perturbed distribution.</returns>
            public static Gaussian Perturb(Gaussian dist)
            {
                //return Gaussian.FromNatural(dist.MeanTimesPrecision + Rand.Normal() / 100, Rand.Double());
                return dist * Gaussian.FromMeanAndVariance(Rand.Normal(), 1.0);
            }
        }
    }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif
}
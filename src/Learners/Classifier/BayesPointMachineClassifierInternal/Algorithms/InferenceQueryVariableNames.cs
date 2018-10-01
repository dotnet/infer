// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.BayesPointMachineClassifierInternal
{
    /// <summary>
    /// The names of the variables used in different types of inference queries on Bayes point machine classifier models.
    /// </summary>
    internal static class InferenceQueryVariableNames
    {
        /// <summary>
        /// The name of the variable for the number of classes.
        /// </summary>
        public const string ClassCount = "ClassCount";

        /// <summary>
        /// The name of the variable for the number of features.
        /// </summary>
        public const string FeatureCount = "FeatureCount";

        /// <summary>
        /// The name of the variable for the number of instances.
        /// </summary>
        public const string InstanceCount = "InstanceCount";

        /// <summary>
        /// The name of the variable for the number of features present for an instance.
        /// </summary>
        public const string InstanceFeatureCounts = "InstanceFeatureCounts";

        /// <summary>
        /// The name of the variable for the array of instance counts for features with value zero.
        /// </summary>
        /// <remarks>
        /// Required in models with sparse feature representations to anchor the weight precision rates of the compound prior.
        /// </remarks>
        public const string ZeroFeatureValueInstanceCounts = "ZeroFeatureValueInstanceCounts";

        /// <summary>
        /// The name of the variable for the feature values.
        /// </summary>
        public const string FeatureValues = "FeatureValues";

        /// <summary>
        /// The name of the variable for the feature indexes.
        /// </summary>
        public const string FeatureIndexes = "FeatureIndexes";

        /// <summary>
        /// The name of the variable for the labels.
        /// </summary>
        public const string Labels = "Labels";

        /// <summary>
        /// The name of the variable for the weights.
        /// </summary>
        public const string Weights = "Weights";

        /// <summary>
        /// The name of the variable for the prior distributions over weights.
        /// </summary>
        public const string WeightPriors = "WeightPriors";

        /// <summary>
        /// The name of the variable for the constraint distributions of the weights.
        /// </summary>
        public const string WeightConstraints = "WeightConstraints";

        /// <summary>
        /// The name of the variable for the distributions over weight precision rates.
        /// </summary>
        public const string WeightPrecisionRates = "WeightPrecisionRates";

        /// <summary>
        /// The name of the variable for the constraint distributions of the weight precision rates.
        /// </summary>
        public const string WeightPrecisionRateConstraints = "WeightPrecisionRateConstraints";

        /// <summary>
        /// The name of the variable for model selection.
        /// </summary>
        public const string ModelSelector = "ModelSelector";
    }
}

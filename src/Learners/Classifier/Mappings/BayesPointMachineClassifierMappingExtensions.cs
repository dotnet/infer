// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Mappings
{
    using System.Collections.Generic;
    using System.Linq;

    /// <summary>
    /// Extension methods for the 
    /// <see cref="IBayesPointMachineClassifierMapping{TInstanceSource, TInstance, TLabelSource, TLabel}"/> interface.
    /// </summary>
    public static class BayesPointMachineClassifierMappingExtensions
    {
        /// <summary>
        /// Safely retrieves the total number of features for the specified instance source.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of the instance source.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of the label source.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="mapping">The mapping.</param>
        /// <param name="instanceSource">An optional instance source.</param>
        /// <returns>The total number of features.</returns>
        /// <exception cref="MappingException">Thrown if the number of features is negative.</exception>
        public static int GetFeatureCountSafe<TInstanceSource, TInstance, TLabelSource, TLabel>(
            this IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel> mapping,
            TInstanceSource instanceSource = default(TInstanceSource))
        {
            int featureCount = mapping.GetFeatureCount(instanceSource);
            if (featureCount < 0)
            {
                throw new MappingException("The total number of features must not be negative.");
            }

            return featureCount;
        }

        /// <summary>
        /// Safely retrieves the number of classes that the Bayes point machine classifier is used for.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of the instance source.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of the label source.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="mapping">The mapping.</param>
        /// <param name="instanceSource">An optional instance source.</param>
        /// <param name="labelSource">An optional label source.</param>
        /// <returns>The number of classes that the Bayes point machine classifier is used for.</returns>
        /// <exception cref="MappingException">Thrown if there are less than three classes.</exception>
        public static int GetClassCountSafe<TInstanceSource, TInstance, TLabelSource, TLabel>(
            this IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel> mapping,
            TInstanceSource instanceSource = default(TInstanceSource),
            TLabelSource labelSource = default(TLabelSource))
        {
            int classCount = mapping.GetClassCount(instanceSource, labelSource);
            if (classCount < 2)
            {
                throw new MappingException("There must be at least two classes.");
            }

            return classCount;
        }

        /// <summary>
        /// Safely retrieves the feature values for a specified instance.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of the instance source.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of the label source.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="mapping">The mapping.</param>
        /// <param name="instance">The instance.</param>
        /// <param name="instanceSource">An optional instance source.</param>
        /// <returns>The feature values for the specified instance.</returns>
        /// <exception cref="MappingException">Thrown if the feature values are null, infinite or NaN.</exception>
        public static double[] GetSingleFeatureVectorValuesSafe<TInstanceSource, TInstance, TLabelSource, TLabel>(
            this IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel> mapping,
            TInstance instance,
            TInstanceSource instanceSource = default(TInstanceSource))
        {
            double[] featureValues = mapping.GetFeatureValues(instance, instanceSource);
            if (featureValues == null)
            {
                throw new MappingException("The feature values must not be null.");
            }

            if (featureValues.Any(value => double.IsInfinity(value) || double.IsNaN(value)))
            {
                throw new MappingException("A feature value must not be infinite or NaN.");
            }

            return featureValues;
        }

        /// <summary>
        /// Safely retrieves the feature indexes for a specified instance.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of the instance source.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of the label source.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="mapping">The mapping.</param>
        /// <param name="instance">The instance.</param>
        /// <param name="instanceSource">An optional instance source.</param>
        /// <returns>The feature indexes for the specified instance.</returns>
        /// <exception cref="MappingException">Thrown if the feature indexes contain duplicates or negative indexes.</exception>
        public static int[] GetSingleFeatureVectorIndexesSafe<TInstanceSource, TInstance, TLabelSource, TLabel>(
            this IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel> mapping,
            TInstance instance,
            TInstanceSource instanceSource = default(TInstanceSource))
        {
            int[] featureIndexes = mapping.GetFeatureIndexes(instance, instanceSource);
            if (featureIndexes == null)
            {
                return null;
            }
            
            // Check that indexes are not negative
            if (featureIndexes.Any(index => index < 0))
            {
                throw new MappingException("The feature indexes must not be negative.");
            }

            // Check for duplicate indexes
            var featureIndexSet = new HashSet<int>(featureIndexes);
            if (featureIndexSet.Count != featureIndexes.Length)
            {
                throw new MappingException("The feature indexes must not contain duplicates.");
            }

            return featureIndexes;
        }

        /// <summary>
        /// Safely retrieves the feature values of all instances from a specified batch of the instance source.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of the instance source.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of the label source.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="mapping">The mapping.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="batchNumber">An optional batch number. Defaults to 0 and is used only if the instance source is divided into batches.</param>
        /// <returns>The feature values provided by the specified batch of the instance source.</returns>
        /// <exception cref="MappingException">
        /// Thrown if the feature values are null, empty or contain infinite values or NaNs.
        /// </exception>
        public static double[][] GetAllFeatureVectorValuesSafe<TInstanceSource, TInstance, TLabelSource, TLabel>(
            this IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel> mapping, 
            TInstanceSource instanceSource,
            int batchNumber = 0)
        {
            if (batchNumber < 0)
            {
                throw new MappingException("The number of a batch must not be negative.");
            }

            double[][] featureValues = mapping.GetFeatureValues(instanceSource, batchNumber);
            if (featureValues == null)
            {
                throw new MappingException("The feature values must not be null.");
            }

            if (featureValues.Length == 0)
            {
                throw new MappingException("The feature values must not be empty.");
            }

            if (featureValues.Any(values => values == null))
            {
                throw new MappingException("The feature values must not be null for any instance.");
            }

            if (featureValues.Any(values => values.Any(value => double.IsInfinity(value) || double.IsNaN(value))))
            {
                throw new MappingException("A feature value must not be infinite or NaN.");
            }

            return featureValues;
        }

        /// <summary>
        /// Safely retrieves the feature indexes of all instances from a specified batch of the instance source.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of the instance source.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of the label source.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="mapping">The mapping.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="batchNumber">An optional batch number. Defaults to 0 and is used only if the instance source is divided into batches.</param>
        /// <returns>The feature indexes provided by the specified batch of the instance source.</returns>
        /// <exception cref="MappingException">
        /// Thrown if the feature indexes are empty or contain instances with duplicate, negative or null indexes. 
        /// </exception>
        public static int[][] GetAllFeatureVectorIndexesSafe<TInstanceSource, TInstance, TLabelSource, TLabel>(
            this IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel> mapping,
            TInstanceSource instanceSource,
            int batchNumber = 0)
        {
            if (batchNumber < 0)
            {
                throw new MappingException("The number of a batch must not be negative.");
            }

            int[][] featureIndexes = mapping.GetFeatureIndexes(instanceSource, batchNumber);
            if (featureIndexes == null)
            {
                return null;
            }

            if (featureIndexes.Length == 0)
            {
                throw new MappingException("The feature indexes must not be empty.");
            }

            if (featureIndexes.Any(indexes => indexes == null))
            {
                throw new MappingException("The feature indexes must not be null for any instance.");
            }

            for (int instance = 0; instance < featureIndexes.Length; instance++)
            {
                // Check that indexes are not negative
                if (featureIndexes[instance].Any(index => index < 0))
                {
                    throw new MappingException("The feature indexes must not be negative.");
                }

                // Check for duplicate indexes
                var featureIndexSet = new HashSet<int>(featureIndexes[instance]);
                if (featureIndexSet.Count != featureIndexes[instance].Length)
                {
                    throw new MappingException("The feature indexes must not contain duplicates.");
                }
            }

            return featureIndexes;
        }

        /// <summary>
        /// Safely retrieves an array of labels for the specified batch of the instance source.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of the instance source.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of the label source.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="mapping">The mapping.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">An optional label source.</param>
        /// <param name="batchNumber">An optional batch number. Defaults to 0 and is used only if the instance and label sources are divided into batches.</param>
        /// <returns>The labels provided by the specified batch of the sources.</returns>
        /// <exception cref="MappingException">Thrown if the array of labels is null or empty.</exception>
        public static TLabel[] GetAllLabelsSafe<TInstanceSource, TInstance, TLabelSource, TLabel>(
            this IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel> mapping,
            TInstanceSource instanceSource,
            TLabelSource labelSource = default(TLabelSource),
            int batchNumber = 0)
        {
            if (batchNumber < 0)
            {
                throw new MappingException("The number of a batch must not be negative.");
            }

            TLabel[] labels = mapping.GetLabels(instanceSource, labelSource, batchNumber);
            if (labels == null)
            {
                throw new MappingException("The labels must not be null.");
            }

            if (labels.Length == 0)
            {
                throw new MappingException("The labels must not be empty.");
            }

            return labels;
        }
    }
}

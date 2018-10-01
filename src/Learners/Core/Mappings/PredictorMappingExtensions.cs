// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Mappings
{
    using System.Collections.Generic;

    /// <summary>
    /// Extension methods for the <see cref="IPredictorMapping{TInstanceSource, TInstance, TLabelSource, TLabel, TFeatures}"/> interface.
    /// </summary>
    public static class PredictorMappingExtensions
    {
        /// <summary>
        /// Safely retrieves instances for a given instance source.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TFeatures">The type of the features.</typeparam>
        /// <param name="mapping">The mapping.</param>
        /// <param name="instanceSource">The source of instances.</param>
        /// <returns>The instances provided by the instance source.</returns>
        /// <exception cref="MappingException">Thrown if the retrieved instances are null.</exception>
        public static IEnumerable<TInstance> GetInstancesSafe<TInstanceSource, TInstance, TLabelSource, TLabel, TFeatures>(
            this IPredictorMapping<TInstanceSource, TInstance, TLabelSource, TLabel, TFeatures> mapping, 
            TInstanceSource instanceSource)
        {
            IEnumerable<TInstance> instances = mapping.GetInstances(instanceSource);

            if (instances == null)
            {
                throw new MappingException("The instances must not be null.");
            }

            return instances;
        }

        /// <summary>
        /// Safely retrieves the features for a given instance.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TFeatures">The type of the features.</typeparam>
        /// <param name="mapping">The mapping.</param>
        /// <param name="instance">The instance to provide features for.</param>
        /// <param name="instanceSource">An optional source of instances.</param>
        /// <returns>The features for the given instance.</returns>
        /// <exception cref="MappingException">Thrown if the retrieved feature values are null.</exception>
        public static TFeatures GetFeaturesSafe<TInstanceSource, TInstance, TLabelSource, TLabel, TFeatures>(
            this IPredictorMapping<TInstanceSource, TInstance, TLabelSource, TLabel, TFeatures> mapping,
            TInstance instance, 
            TInstanceSource instanceSource = default(TInstanceSource))
        {
            TFeatures featureValues = mapping.GetFeatures(instance, instanceSource);

            if (featureValues == null)
            {
                throw new MappingException("The features must not be null.");
            }

            return featureValues;
        }

        /// <summary>
        /// Safely retrieves the label for a given instance.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TFeatures">The type of the features.</typeparam>
        /// <param name="mapping">The mapping.</param>
        /// <param name="instance">The instance to provide the label for.</param>
        /// <param name="instanceSource">An optional source of instances.</param>
        /// <param name="labelSource">An optional source of labels.</param>
        /// <returns>The label of the given instance.</returns>
        /// <exception cref="MappingException">Thrown if the retrieved label is null.</exception>
        public static TLabel GetLabelSafe<TInstanceSource, TInstance, TLabelSource, TLabel, TFeatures>(
            this IPredictorMapping<TInstanceSource, TInstance, TLabelSource, TLabel, TFeatures> mapping,
            TInstance instance,
            TInstanceSource instanceSource = default(TInstanceSource),
            TLabelSource labelSource = default(TLabelSource))
        {
            TLabel label = mapping.GetLabel(instance, instanceSource, labelSource);

            if (label == null)
            {
                throw new MappingException("The label must not be null.");
            }

            return label;
        }
    }
}

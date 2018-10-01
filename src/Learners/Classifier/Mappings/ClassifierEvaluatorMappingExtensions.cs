// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Mappings
{
    using System.Collections.Generic;
    using System.Linq;

    /// <summary>
    /// Extension methods for the <see cref="IClassifierEvaluatorMapping{TInstanceSource, TInstance, TLabelSource, TLabel}"/> interface.
    /// </summary>
    public static class ClassifierEvaluatorMappingExtensions
    {
        /// <summary>
        /// Safely retrieves instances for a given instance source.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="mapping">The mapping.</param>
        /// <param name="instanceSource">The source of instances.</param>
        /// <returns>The instances provided by the instance source.</returns>
        /// <exception cref="MappingException">Thrown if the retrieved instances are null.</exception>
        public static IEnumerable<TInstance> GetInstancesSafe<TInstanceSource, TInstance, TLabelSource, TLabel>(
            this IClassifierEvaluatorMapping<TInstanceSource, TInstance, TLabelSource, TLabel> mapping,
            TInstanceSource instanceSource)
        {
            IEnumerable<TInstance> instances = mapping.GetInstances(instanceSource);

            if (instances == null)
            {
                throw new MappingException("The instances must not be null.");
            }

            if (instances.Any(instance => instance == null))
            {
                throw new MappingException("An instance must not be null.");
            }

            return instances;
        }

        /// <summary>
        /// Safely retrieves the label for a given instance.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="mapping">The mapping.</param>
        /// <param name="instance">The instance to provide the label for.</param>
        /// <param name="instanceSource">An optional source of instances.</param>
        /// <param name="labelSource">An optional source of labels.</param>
        /// <returns>The label of the given instance.</returns>
        /// <exception cref="MappingException">Thrown if the retrieved label is null.</exception>
        public static TLabel GetLabelSafe<TInstanceSource, TInstance, TLabelSource, TLabel>(
            this IClassifierEvaluatorMapping<TInstanceSource, TInstance, TLabelSource, TLabel> mapping,
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

        /// <summary>
        /// Safely gets all class labels.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="mapping">The mapping.</param>
        /// <param name="instanceSource">An optional instance source.</param>
        /// <param name="labelSource">An optional label source.</param>
        /// <returns>All possible values of a label.</returns>
        /// <exception cref="MappingException">Thrown if the class labels are null, empty, identical or not unique.</exception>
        public static IEnumerable<TLabel> GetClassLabelsSafe<TInstanceSource, TInstance, TLabelSource, TLabel>(
            this IClassifierEvaluatorMapping<TInstanceSource, TInstance, TLabelSource, TLabel> mapping,
            TInstanceSource instanceSource = default(TInstanceSource),
            TLabelSource labelSource = default(TLabelSource))
        {
            IEnumerable<TLabel> classLabels = mapping.GetClassLabels(instanceSource, labelSource);
            if (classLabels == null)
            {
                throw new MappingException("The class labels must not be null.");
            }

            IList<TLabel> classLabelList = classLabels.ToList();

            if (classLabelList.Any(label => label == null))
            {
                throw new MappingException("A class label must not be null.");
            }

            var classLabelSet = new HashSet<TLabel>(classLabelList);
            int uniqueClassLabelCount = classLabelSet.Count;
            if (uniqueClassLabelCount != classLabelList.Count)
            {
                throw new MappingException("All class labels must be unique.");
            }

            if (uniqueClassLabelCount < 2)
            {
                throw new MappingException("There must be at least two distinct class labels.");
            }

            return classLabelList;
        }
    }
}

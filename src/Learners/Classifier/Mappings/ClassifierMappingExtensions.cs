// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Mappings
{
    using System.Collections.Generic;
    using System.Linq;

    /// <summary>
    /// Extension methods for the <see cref="IClassifierMapping{TInstanceSource, TInstance, TLabelSource, TLabel, TFeatures}"/> interface.
    /// </summary>
    public static class ClassifierMappingExtensions
    {
        /// <summary>
        /// Safely gets all class labels.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TFeatures">The type of the features.</typeparam>
        /// <param name="mapping">The mapping.</param>
        /// <param name="instanceSource">An optional instance source.</param>
        /// <param name="labelSource">An optional label source.</param>
        /// <returns>All possible values of a label.</returns>
        /// <exception cref="MappingException">Thrown if the class labels are null, empty, identical or not unique.</exception>
        public static IReadOnlyList<TLabel> GetClassLabelsSafe<TInstanceSource, TInstance, TLabelSource, TLabel, TFeatures>(
            this IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, TFeatures> mapping,
            TInstanceSource instanceSource = default(TInstanceSource),
            TLabelSource labelSource = default(TLabelSource))
        {
            IEnumerable<TLabel> classLabels = mapping.GetClassLabels(instanceSource, labelSource);
            if (classLabels == null)
            {
                throw new MappingException("The class labels must not be null.");
            }

            IReadOnlyList<TLabel> classLabelList = classLabels.OrderBy(classLabel => classLabel).ToList();

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

        /// <summary>
        /// Chains a given classifier mapping with an evaluator mapping,
        /// making it possible to evaluate a classifier on the mapped data.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TFeatures">The type of the features.</typeparam>
        /// <param name="mapping">The classifier mapping to chain with the evaluator mapping.</param>
        /// <returns>The classifier evaluator mapping.</returns>
        public static ClassifierEvaluatorMapping<TInstanceSource, TInstance, TLabelSource, TLabel, TFeatures>
            ForEvaluation<TInstanceSource, TInstance, TLabelSource, TLabel, TFeatures>(
            this IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, TFeatures> mapping)
        {
            return new ClassifierEvaluatorMapping<TInstanceSource, TInstance, TLabelSource, TLabel, TFeatures>(mapping);
        }
    }
}

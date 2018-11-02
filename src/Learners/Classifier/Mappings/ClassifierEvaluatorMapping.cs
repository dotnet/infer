// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Mappings
{
    using System;
    using System.Collections.Generic;

    /// <summary>
    /// A mapping used by an evaluator of classifiers.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
    /// <typeparam name="TInstance">The type of an instance.</typeparam>
    /// <typeparam name="TLabelSource">The type of a source of ground truth labels.</typeparam>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    /// <typeparam name="TFeatureValues">The type of the feature values.</typeparam>
    public class ClassifierEvaluatorMapping<TInstanceSource, TInstance, TLabelSource, TLabel, TFeatureValues>
        : IClassifierEvaluatorMapping<TInstanceSource, TInstance, TLabelSource, TLabel>
    {
        #region Fields

        /// <summary>
        /// The wrapped classifier mapping.
        /// </summary>
        private readonly IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, TFeatureValues> classifierMapping;

        #region Caching

        /// <summary>
        /// The cached instance source.
        /// </summary>
        private TInstanceSource cachedInstanceSource;

        /// <summary>
        /// The cached label source.
        /// </summary>
        private TLabelSource cachedLabelSource;

        /// <summary>
        /// The cached instances.
        /// </summary>
        private IEnumerable<TInstance> cachedInstances;

        /// <summary>
        /// The cached labels.
        /// </summary>
        private Dictionary<TInstance, TLabel> cachedLabels;

        #endregion

        #endregion

        #region Constructor

        /// <summary>
        /// Initializes a new instance of the <see cref="ClassifierEvaluatorMapping{TInstanceSource,TInstance,TLabelSource,TLabel,TFeatureValues}"/> class.
        /// </summary>
        /// <param name="mapping">A classifier mapping.</param>
        public ClassifierEvaluatorMapping(IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, TFeatureValues> mapping)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            this.classifierMapping = mapping;

            this.cachedLabels = new Dictionary<TInstance, TLabel>();
        }

        #endregion

        #region IClassifierEvaluatorMapping implementation

        /// <summary>
        /// Provides the instances for a given instance source.
        /// </summary>
        /// <param name="instanceSource">The source of instances.</param>
        /// <returns>The instances provided by the instance source.</returns>
        /// <remarks>Assumes that the same instance source always provides the same instances.</remarks>
        public IEnumerable<TInstance> GetInstances(TInstanceSource instanceSource)
        {
            if (instanceSource == null)
            {
                throw new ArgumentNullException(nameof(instanceSource));
            }

            if (!this.IsInstanceSourceCached(instanceSource))
            {
                // Cache instances
                this.cachedInstances = this.classifierMapping.GetInstancesSafe(instanceSource);
                this.cachedInstanceSource = instanceSource;
            }

            return this.cachedInstances;
        }

        /// <summary>
        /// Provides the label for a given instance.
        /// </summary>
        /// <param name="instance">The instance to provide the label for.</param>
        /// <param name="instanceSource">An optional source of instances.</param>
        /// <param name="labelSource">An optional source of labels.</param>
        /// <returns>The label of the given instance.</returns>
        /// <remarks>Assumes that the same sources always provide the same label for a given instance.</remarks>
        public TLabel GetLabel(
            TInstance instance,
            TInstanceSource instanceSource = default(TInstanceSource),
            TLabelSource labelSource = default(TLabelSource))
        {
            if (instance == null)
            {
                throw new ArgumentNullException(nameof(instance));
            }

            TLabel label;

            if ((labelSource != null && !this.IsLabelSourceCached(labelSource)) ||
                (instanceSource != null && !this.IsInstanceSourceCached(instanceSource)))
            {
                if (instanceSource != null && !this.IsInstanceSourceCached(instanceSource))
                {
                    // Cache instances!
                    this.cachedInstances = this.classifierMapping.GetInstancesSafe(instanceSource);
                    this.cachedInstanceSource = instanceSource;
                }

                // Clear label cache, insert new label and return it
                this.cachedLabels = new Dictionary<TInstance, TLabel>();
                label = this.classifierMapping.GetLabelSafe(instance, instanceSource, labelSource);
                this.cachedLabels.Add(instance, label);
                this.cachedLabelSource = labelSource;

                return label;
            }

            // Instance and label sources null or cached, but is label cached?
            if (!this.cachedLabels.TryGetValue(instance, out label))
            {
                // Cache label
                label = this.classifierMapping.GetLabelSafe(instance, instanceSource, labelSource);
                this.cachedLabels.Add(instance, label);
            }

            return label;
        }

        /// <summary>
        /// Gets all class labels.
        /// </summary>
        /// <param name="instanceSource">An optional instance source.</param>
        /// <param name="labelSource">An optional label source.</param>
        /// <returns>All possible values of a label.</returns>
        public IEnumerable<TLabel> GetClassLabels(
            TInstanceSource instanceSource = default(TInstanceSource), TLabelSource labelSource = default(TLabelSource))
        {
            if (instanceSource != null && !this.IsInstanceSourceCached(instanceSource))
            {
                // Cache instances!
                this.cachedInstances = this.classifierMapping.GetInstancesSafe(instanceSource);
                this.cachedInstanceSource = instanceSource;
            }

            // Class labels cannot be cached: They might change even if both instance and label source are unchanged
            return this.classifierMapping.GetClassLabelsSafe(instanceSource, labelSource);
        }

        #endregion

        #region Helper methods

        /// <summary>
        /// Returns true if the specified instance source is cached and false otherwise.
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <returns>True if the specified instance source is cached and false otherwise.</returns>
        private bool IsInstanceSourceCached(TInstanceSource instanceSource)
        {
            return ReferenceEquals(instanceSource, this.cachedInstanceSource);
        }

        /// <summary>
        /// Returns true if the specified label source is cached and false otherwise.
        /// </summary>
        /// <param name="labelSource">The label source.</param>
        /// <returns>True if the specified label source is cached and false otherwise.</returns>
        private bool IsLabelSourceCached(TLabelSource labelSource)
        {
            return ReferenceEquals(labelSource, this.cachedLabelSource);
        }

        #endregion
    }
}

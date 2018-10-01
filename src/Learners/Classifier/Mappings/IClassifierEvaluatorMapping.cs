// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Mappings
{
    using System.Collections.Generic;

    /// <summary>
    /// A mapping used by the <see cref="ClassifierEvaluator{TInstanceSource, TInstance, TLabelSource, TLabel}"/> class.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
    /// <typeparam name="TInstance">The type of an instance.</typeparam>
    /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    public interface IClassifierEvaluatorMapping<in TInstanceSource, TInstance, in TLabelSource, out TLabel>
    {
        /// <summary>
        /// Provides the instances for a given instance source.
        /// </summary>
        /// <param name="instanceSource">The source of instances.</param>
        /// <returns>The instances provided by the instance source.</returns>
        /// <remarks>Assumes that the same instance source always provides the same instances.</remarks>
        IEnumerable<TInstance> GetInstances(TInstanceSource instanceSource);

        /// <summary>
        /// Provides the label for a given instance.
        /// </summary>
        /// <param name="instance">The instance to provide the label for.</param>
        /// <param name="instanceSource">An optional source of instances.</param>
        /// <param name="labelSource">An optional source of labels.</param>
        /// <returns>The label of the given instance.</returns>
        /// <remarks>Assumes that the same sources always provide the same label for a given instance.</remarks>
        TLabel GetLabel(TInstance instance, TInstanceSource instanceSource = default(TInstanceSource), TLabelSource labelSource = default(TLabelSource));

        /// <summary>
        /// Gets all class labels.
        /// </summary>
        /// <param name="instanceSource">An optional instance source.</param>
        /// <param name="labelSource">An optional label source.</param>
        /// <returns>All possible values of a label.</returns>
        IEnumerable<TLabel> GetClassLabels(TInstanceSource instanceSource = default(TInstanceSource), TLabelSource labelSource = default(TLabelSource));
    }
}
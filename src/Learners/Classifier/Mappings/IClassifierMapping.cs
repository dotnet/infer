// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Mappings
{
    using System.Collections.Generic;

    /// <summary>
    /// A mapping used by classifier implementations of the
    /// <see cref="IPredictor{TInstanceSource, TInstance, TLabelSource, TResult, TResultDist}"/>
    /// interface to convert user data to an appropriate format.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
    /// <typeparam name="TInstance">The type of an instance.</typeparam>
    /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    /// <typeparam name="TFeatures">The type of the feature values.</typeparam>
    public interface IClassifierMapping<in TInstanceSource, TInstance, in TLabelSource, out TLabel, out TFeatures> 
        : IPredictorMapping<TInstanceSource, TInstance, TLabelSource, TLabel, TFeatures>
    {
        /// <summary>
        /// Gets all class labels.
        /// </summary>
        /// <param name="instanceSource">An optional instance source.</param>
        /// <param name="labelSource">An optional label source.</param>
        /// <returns>All possible values of a label.</returns>
        IEnumerable<TLabel> GetClassLabels(TInstanceSource instanceSource = default(TInstanceSource), TLabelSource labelSource = default(TLabelSource));
    }
}

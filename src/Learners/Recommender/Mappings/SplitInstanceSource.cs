// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Mappings
{
    using System.Diagnostics.CodeAnalysis;

    /// <summary>
    /// Factory for the <see cref="SplitInstanceSource"/> class.
    /// </summary>
    public static class SplitInstanceSource
    {
        /// <summary>
        /// Creates an instance source restricted to the instances in the training set of a given instance source.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of the unrestricted instance source.</typeparam>
        /// <param name="instanceSource">The unrestricted instance source.</param>
        /// <returns>The instance source restricted to the training set instances.</returns>
        public static SplitInstanceSource<TInstanceSource> Training<TInstanceSource>(TInstanceSource instanceSource)
        {
            // TODO: created instances can be cached in order to avoid mapping chain invalidation
            return new SplitInstanceSource<TInstanceSource>(true, instanceSource);
        }

        /// <summary>
        /// Creates an instance source restricted to the instances in the test set of a given instance source.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of the unrestricted instance source.</typeparam>
        /// <param name="instanceSource">The unrestricted instance source.</param>
        /// <returns>The instance source restricted to the test set instances.</returns>
        public static SplitInstanceSource<TInstanceSource> Test<TInstanceSource>(TInstanceSource instanceSource)
        {
            // TODO: created instances can be cached in order to avoid mapping chain invalidation
            return new SplitInstanceSource<TInstanceSource>(false, instanceSource);
        }
    }
    
    /// <summary>
    /// Splits an instance source
    /// produced by the <see cref="TrainTestSplittingRecommenderMapping{TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues}"/>
    /// to instances in the training set or the test set.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of the instance source.</typeparam>
    [SuppressMessage("StyleCop.CSharp.MaintainabilityRules", "SA1402:FileMayOnlyContainASingleClass", Justification = "One of the classes is a factory for another one and has the same name.")]
    public class SplitInstanceSource<TInstanceSource>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="SplitInstanceSource{TInstanceSource}"/> class.
        /// </summary>
        /// <param name="restrictToTrainingSet">
        /// Whether the new instance source is restricted to the training set or the test set of the given instance source.
        /// </param>
        /// <param name="instanceSource">The unrestricted instance source.</param>
        internal SplitInstanceSource(bool restrictToTrainingSet, TInstanceSource instanceSource)
        {
            this.IsTrainingSet = restrictToTrainingSet;
            this.InstanceSource = instanceSource;
        }
        
        /// <summary>
        /// Gets a value indicating whether this instance source is restricted to the training set of the unrestricted instance source.
        /// </summary>
        public bool IsTrainingSet { get; private set; }

        /// <summary>
        /// Gets the unrestricted instance source.
        /// </summary>
        public TInstanceSource InstanceSource { get; private set; }
    }
}

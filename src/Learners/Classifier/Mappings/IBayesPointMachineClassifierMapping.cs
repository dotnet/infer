// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Mappings
{
    /// <summary>
    /// A mapping to access data in the native format of implementations of the
    /// <see cref="IBayesPointMachineClassifier{TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings}"/>
    /// interface.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of the instance source.</typeparam>
    /// <typeparam name="TInstance">The type of an instance.</typeparam>
    /// <typeparam name="TLabelSource">The type of the label source.</typeparam>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    /// <remarks>
    /// This interface allows to provide features in one of two representations: dense and sparse. 
    /// To provide features in a dense representation, all arrays over feature indexes must be null 
    /// and all instances must have the same number of feature values. To provide features 
    /// in a sparse representation, each single instance must have the same number of feature 
    /// indexes and values (arrays over feature indexes must not be null).
    /// </remarks>
    public interface IBayesPointMachineClassifierMapping<in TInstanceSource, in TInstance, in TLabelSource, out TLabel>
    {
        /// <summary>
        /// Indicates whether the feature representation provided by the instance source is sparse or dense.
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <returns>True, if the feature representation is sparse and false if it is dense.</returns>
        bool IsSparse(TInstanceSource instanceSource);

        /// <summary>
        /// Provides the total number of features for the specified instance source.
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <returns>The total number of features.</returns>
        int GetFeatureCount(TInstanceSource instanceSource);

        /// <summary>
        /// Provides the number of classes that the Bayes point machine classifier is used for.
        /// </summary>
        /// <param name="instanceSource">An optional instance source.</param>
        /// <param name="labelSource">An optional label source.</param>
        /// <returns>The number of classes that the Bayes point machine classifier is used for.</returns>
        int GetClassCount(TInstanceSource instanceSource = default(TInstanceSource), TLabelSource labelSource = default(TLabelSource));

        /// <summary>
        /// Provides the feature values for a specified instance.
        /// </summary>
        /// <param name="instance">The instance.</param>
        /// <param name="instanceSource">An optional instance source.</param>
        /// <returns>The feature values for the specified instance.</returns>
        double[] GetFeatureValues(TInstance instance, TInstanceSource instanceSource = default(TInstanceSource));

        /// <summary>
        /// Provides the feature indexes for a specified instance.
        /// </summary>
        /// <param name="instance">The instance.</param>
        /// <param name="instanceSource">An optional instance source.</param>
        /// <returns>The feature indexes for the specified instance. Null if feature values are in a dense representation.</returns>
        int[] GetFeatureIndexes(TInstance instance, TInstanceSource instanceSource = default(TInstanceSource));

        /// <summary>
        /// Provides the feature values of all instances from the specified batch of the instance source.
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="batchNumber">An optional batch number. Defaults to 0 and is used only if the instance source is divided into batches.</param>
        /// <returns>The feature values provided by the specified batch of the instance source.</returns>
        double[][] GetFeatureValues(TInstanceSource instanceSource, int batchNumber = 0);

        /// <summary>
        /// Provides the feature indexes of all instances from the specified batch of the instance source.
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="batchNumber">An optional batch number. Defaults to 0 and is used only if the instance source is divided into batches.</param>
        /// <returns>
        /// The feature indexes provided by the specified batch of the instance source. Null if feature values are in a dense representation.
        /// </returns>
        int[][] GetFeatureIndexes(TInstanceSource instanceSource, int batchNumber = 0);

        /// <summary>
        /// Provides the labels of all instances from the specified batch of the instance source.
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">An optional label source.</param>
        /// <param name="batchNumber">An optional batch number. Defaults to 0 and is used only if the instance and label sources are divided into batches.</param>
        /// <returns>The labels provided by the specified batch of the sources.</returns>
        TLabel[] GetLabels(TInstanceSource instanceSource, TLabelSource labelSource = default(TLabelSource), int batchNumber = 0);
    }
}
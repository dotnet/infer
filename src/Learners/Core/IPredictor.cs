// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System.Collections.Generic;

    /// <summary>
    /// Interface to a learner that acts on some data to predict a label.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
    /// <typeparam name="TInstance">The type of an instance.</typeparam>
    /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
    /// <typeparam name="TResult">The type of a prediction for a single instance.</typeparam>
    /// <typeparam name="TResultDist">The type of an uncertain prediction for a single instance.</typeparam>
    /// <remarks>
    /// Intended usage:
    /// <para>
    /// An instance refers to the data that can be attributed to a label. It may include the label itself if it is known. 
    /// Typically, an instance provides the feature values which characterize the label. In spam detection, for example, an instance 
    /// might refer to an email which may be labeled 'spam' or 'no spam'. In finance, an instance might refer to a commodity 
    /// and the label may be its price.
    /// </para><para>
    /// An instance source provides all instances of interest.
    /// </para><para>
    /// A label source provides the known labels of some given instances.
    /// </para>
    /// Labels may hence be provided by label source or instance source.
    /// </remarks>
    public interface IPredictor<in TInstanceSource, in TInstance, in TLabelSource, out TResult, out TResultDist> : ILearner
    {
        /// <summary>
        /// Gets the capabilities of the predictor.
        /// </summary>
        new IPredictorCapabilities Capabilities { get; }

        /// <summary>
        /// Trains the predictor on the specified instances.
        /// </summary>
        /// <param name="instanceSource">The source of instances.</param>
        /// <param name="labelSource">The source of labels.</param>
        void Train(TInstanceSource instanceSource, TLabelSource labelSource = default(TLabelSource));

        /// <summary>
        /// Makes a prediction for the specified instance. Uncertainty in the prediction is characterized by the returned distribution type.
        /// </summary>
        /// <param name="instance">The instance to make predictions for.</param>
        /// <param name="instanceSource">The source of instances which provides <paramref name="instance"/>.</param>
        /// <returns>The prediction for <paramref name="instance"/>, with uncertainty.</returns>
        TResultDist PredictDistribution(TInstance instance, TInstanceSource instanceSource = default(TInstanceSource));
        
        /// <summary>
        /// Makes predictions for the specified instances. Uncertainty in the predictions is characterized by the returned distribution type.
        /// </summary>
        /// <param name="instanceSource">The source of instances to make predictions for.</param>
        /// <returns>The predictions of all instances provided by <paramref name="instanceSource"/>, with uncertainty.</returns>
        IEnumerable<TResultDist> PredictDistribution(TInstanceSource instanceSource);

        /// <summary>
        /// Makes a prediction for the specified instance. Uncertainty in the prediction is discarded and a "best" prediction is returned.
        /// </summary>
        /// <param name="instance">The instance to make predictions for.</param>
        /// <param name="instanceSource">The source of instances which provides <paramref name="instance"/>.</param>
        /// <returns>The "best" prediction for <paramref name="instance"/>, discarding uncertainty.</returns>
        /// <remarks>The definition of "best" depends on the particular implementation of the predictor and its settings.</remarks>
        TResult Predict(TInstance instance, TInstanceSource instanceSource = default(TInstanceSource));
        
        /// <summary>
        /// Makes predictions for the specified instances. Uncertainty in the predictions is discarded and the "best" predictions are returned.
        /// </summary>
        /// <param name="instanceSource">The source of instances to make predictions for.</param>
        /// <returns>The "best" prediction for every instance in <paramref name="instanceSource"/>, discarding uncertainty.</returns>
        /// <remarks>The definition of "best" depends on the particular implementation of the predictor and its settings.</remarks>
        IEnumerable<TResult> Predict(TInstanceSource instanceSource);
    }
}
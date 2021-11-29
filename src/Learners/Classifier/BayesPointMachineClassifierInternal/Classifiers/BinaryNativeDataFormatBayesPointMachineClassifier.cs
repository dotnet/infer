// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.BayesPointMachineClassifierInternal
{
    using System;
    using System.Collections.Generic;
    using System.IO;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// An abstract base class for a binary Bayes point machine classifier which operates on 
    /// data in the native format of the underlying Infer.NET model.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of the instance source.</typeparam>
    /// <typeparam name="TInstance">The type of an instance.</typeparam>
    /// <typeparam name="TLabelSource">The type of the label source.</typeparam>
    /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
    [Serializable]
    internal abstract class BinaryNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TTrainingSettings> :
        NativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, bool, Bernoulli, TTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<bool>>
        where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
    {
        #region Fields, constructor, properties

        /// <summary>
        /// The current custom binary serialization version of the
        /// <see cref="BinaryNativeDataFormatBayesPointMachineClassifier{TInstanceSource, TInstance, TLabelSource, TTrainingSettings}"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// Initializes a new instance of the 
        /// <see cref="BinaryNativeDataFormatBayesPointMachineClassifier{TInstanceSource, TInstance, TLabelSource, TTrainingSettings}"/> 
        /// class.
        /// </summary>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        protected BinaryNativeDataFormatBayesPointMachineClassifier(IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, bool> mapping)
            : base(mapping)
        {
        }

        /// <summary>
        /// Initializes a new instance of the 
        /// <see cref="BinaryNativeDataFormatBayesPointMachineClassifier{TInstanceSource, TInstance, TLabelSource, TTrainingSettings}"/> 
        /// class from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the state of the binary Bayes point machine classifier from.</param>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        protected BinaryNativeDataFormatBayesPointMachineClassifier(IReader reader, IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, bool> mapping)
            : base(reader, mapping)
        {
            reader.ReadSerializationVersion(CustomSerializationVersion);

            // Nothing to deserialize
        }

        /// <summary>
        /// Gets the point estimator for the label distribution.
        /// </summary>
        protected override Func<Bernoulli, bool> PointEstimator
        {
            get
            {
                Func<bool, bool, double> customLossFunction;
                LossFunction lossFunction = this.Settings.Prediction.GetPredictionLossFunction(out customLossFunction);
                return (lossFunction == LossFunction.Custom) ? 
                    Learners.PointEstimator.ForBernoulli(customLossFunction) : Learners.PointEstimator.ForBernoulli(lossFunction);
            }
        }

        #endregion

        #region ICustomSerializable implementation

        /// <summary>
        /// Saves the state of the Bayes point machine classifier using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the state of the Bayes point machine classifier to.</param>
        public override void SaveForwardCompatible(IWriter writer)
        {
            base.SaveForwardCompatible(writer);

            writer.Write(CustomSerializationVersion);
        }

        #endregion

        #region Template methods for inference algorithms

        /// <summary>
        /// Runs the prediction algorithm on the specified data.
        /// </summary>
        /// <param name="featureValues">The feature values.</param>
        /// <param name="featureIndexes">The feature indexes.</param>
        /// <returns>The predictive distribution over labels.</returns>
        protected override IEnumerable<Bernoulli> PredictDistribution(double[][] featureValues, int[][] featureIndexes)
        {
            return this.InferenceAlgorithms.PredictDistribution(featureValues, featureIndexes, 1);
        }

        #endregion
    }
}
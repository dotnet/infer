// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.BayesPointMachineClassifierInternal
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// An abstract base class for a multi-class Bayes point machine classifier which operates on 
    /// data in the native format of the underlying Infer.NET model.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of the instance source.</typeparam>
    /// <typeparam name="TInstance">The type of an instance.</typeparam>
    /// <typeparam name="TLabelSource">The type of the label source.</typeparam>
    /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
    [Serializable]
    internal abstract class MulticlassNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TTrainingSettings> :
        NativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, int, Discrete, TTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<int>>
        where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
    {
        #region Fields, constructor, properties

        /// <summary>
        /// The current custom binary serialization version of the
        /// <see cref="MulticlassNativeDataFormatBayesPointMachineClassifier{TInstanceSource, TInstance, TLabelSource, TTrainingSettings}"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// Initializes a new instance of the 
        /// <see cref="MulticlassNativeDataFormatBayesPointMachineClassifier{TInstanceSource, TInstance, TLabelSource, TTrainingSettings}"/> 
        /// class.
        /// </summary>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        protected MulticlassNativeDataFormatBayesPointMachineClassifier(IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, int> mapping)
            : base(mapping)
        {
        }

        /// <summary>
        /// Initializes a new instance of the 
        /// <see cref="MulticlassNativeDataFormatBayesPointMachineClassifier{TInstanceSource, TInstance, TLabelSource, TTrainingSettings}"/> 
        /// class from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the state of the multi-class Bayes point machine classifier from.</param>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        protected MulticlassNativeDataFormatBayesPointMachineClassifier(IReader reader, IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, int> mapping)
            : base(reader, mapping)
        {
            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);

            if (deserializedVersion == CustomSerializationVersion)
            {
                this.ClassCount = reader.ReadInt32();
            }
        }

        /// <summary>
        /// Gets or sets the number of classes used in the multi-class Bayes point machine classifier.
        /// </summary>
        protected int ClassCount { get; set; }

        /// <summary>
        /// Gets the point estimator for the label distribution.
        /// </summary>
        protected override Func<Discrete, int> PointEstimator
        {
            get
            {
                Func<int, int, double> customLossFunction;
                LossFunction lossFunction = this.Settings.Prediction.GetPredictionLossFunction(out customLossFunction);
                return (lossFunction == LossFunction.Custom) ? 
                    Learners.PointEstimator.ForDiscrete(customLossFunction) : Learners.PointEstimator.ForDiscrete(lossFunction);
            }
        }

        #endregion

        #region IBayesPointMachineClassifier implementation

        /// <summary>
        /// Incrementally trains the multi-class Bayes point machine classifier on the specified instances.
        /// </summary>
        /// <param name="instanceSource">The source of instances.</param>
        /// <param name="labelSource">An optional source of labels.</param>
        public override void TrainIncremental(TInstanceSource instanceSource, TLabelSource labelSource = default(TLabelSource))
        {
            // Check that the mapping has the correct number of classes.
            int mappingClassCount = this.Mapping.GetClassCountSafe(instanceSource, labelSource);

            if (!this.IsTrained)
            {
                if (mappingClassCount == 2)
                {
                    throw new BayesPointMachineClassifierException(
                        "Please use the binary Bayes point machine classifier for data with two classes.");
                }

                if (mappingClassCount < 2)
                {
                    throw new BayesPointMachineClassifierException("The multi-class Bayes point machine classifier requires data with more than two classes.");
                }
                
                this.ClassCount = mappingClassCount;
            }
            else
            {
                if (mappingClassCount != this.ClassCount)
                {
                    throw new BayesPointMachineClassifierException("The number of classes is inconsistent with the inference algorithms.");
                }
            }

            base.TrainIncremental(instanceSource, labelSource);
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
            writer.Write(this.ClassCount);
        }

        #endregion

        #region Template methods for inference algorithms

        /// <summary>
        /// Runs the prediction algorithm on the specified data.
        /// </summary>
        /// <param name="featureValues">The feature values.</param>
        /// <param name="featureIndexes">The feature indexes.</param>
        /// <returns>The predictive distribution over labels.</returns>
        protected override IEnumerable<Discrete> PredictDistribution(double[][] featureValues, int[][] featureIndexes)
        {
            return this.InferenceAlgorithms.PredictDistribution(featureValues, featureIndexes, this.Settings.Prediction.IterationCount);
        }

        #endregion

        #region Helper methods

        /// <summary>
        /// Verifies that the data returned by the native mapping is consistent.
        /// </summary>
        /// <param name="isSparse">If true, the mapping uses a sparse feature representation. If false, the representation is dense.</param>
        /// <param name="featureCount">The total number of features accessible through the mapping in either representation.</param>
        /// <param name="featureValues">The feature values returned by the mapping.</param>
        /// <param name="featureIndexes">The feature indexes returned by the mapping.</param>
        /// <param name="labels">The labels returned by the mapping.</param>
        /// <exception cref="BayesPointMachineClassifierException">Thrown if the training data is inconsistent.</exception>
        protected override void CheckDataConsistency(bool isSparse, int featureCount, double[][] featureValues, int[][] featureIndexes, int[] labels)
        {
            base.CheckDataConsistency(isSparse, featureCount, featureValues, featureIndexes, labels);

            if (labels.Any(label => label < 0 || label >= this.ClassCount))
            {
                throw new BayesPointMachineClassifierException("The data contains unknown labels.");
            }
        }

        #endregion
    }
}
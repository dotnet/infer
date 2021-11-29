// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.BayesPointMachineClassifierInternal
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// An abstract Bayes point machine classifier which operates on 
    /// data in the native format of the underlying Infer.NET model.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of the instance source.</typeparam>
    /// <typeparam name="TInstance">The type of an instance.</typeparam>
    /// <typeparam name="TLabelSource">The type of the label source.</typeparam>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
    /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
    /// <typeparam name="TPredictionSettings">The type of the settings for prediction.</typeparam>
    [Serializable]
    internal abstract class NativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings> :
        IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings>
        where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        where TPredictionSettings : IBayesPointMachineClassifierPredictionSettings<TLabel>
    {
        #region Fields, constructors, events

        /// <summary>
        /// The current custom binary serialization version of the
        /// <see cref="NativeDataFormatBayesPointMachineClassifier{TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings}"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// The maximum number of iterations to run the training algorithm on a single batch of instances.
        /// </summary>
        private const int MaxBatchedIterationCount = 25;

        /// <summary>
        /// The custom binary serialization <see cref="Guid"/> of the
        /// <see cref="NativeDataFormatBayesPointMachineClassifier{TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings}"/> class.
        /// </summary>
        private readonly Guid customSerializationGuid = new Guid("B7409D2F-3AAF-4C2C-89DF-01E9EAFDA66C");

        /// <summary>
        /// The capabilities of this Bayes point machine classifier.
        /// </summary>
        private readonly BayesPointMachineClassifierCapabilities capabilities;

        /// <summary>
        /// The natural logarithm of the evidence for the Bayes point machine classifier model.
        /// </summary>
        private double logModelEvidence;

        /// <summary>
        /// Indicates whether the Bayes point machine classifier is incrementally trained.
        /// </summary>
        private bool isIncrementallyTrained;

        /// <summary>
        /// Initializes a new instance of the 
        /// <see cref="NativeDataFormatBayesPointMachineClassifier{TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings}"/> 
        /// class.
        /// </summary>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        protected NativeDataFormatBayesPointMachineClassifier(
            IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel> mapping)
        {
            Debug.Assert(mapping != null, "The mapping must not be null.");

            this.Mapping = mapping;
            this.capabilities = new BayesPointMachineClassifierCapabilities();
            this.IsTrained = false;
            this.isIncrementallyTrained = false;
        }

        /// <summary>
        /// Initializes a new instance of the 
        /// <see cref="NativeDataFormatBayesPointMachineClassifier{TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings}"/> 
        /// class from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the state of the Bayes point machine classifier from.</param>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        protected NativeDataFormatBayesPointMachineClassifier(
            IReader reader, 
            IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel> mapping)
        {
            Debug.Assert(reader != null, "The reader must not be null.");
            Debug.Assert(mapping != null, "The mapping must not be null.");

            this.Mapping = mapping;
            this.capabilities = new BayesPointMachineClassifierCapabilities();

            reader.VerifySerializationGuid(
                this.customSerializationGuid, "The stream does not contain an Infer.NET Bayes point machine classifier.");

            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);
            if (deserializedVersion == CustomSerializationVersion)
            {
                this.IsTrained = reader.ReadBoolean();
                this.isIncrementallyTrained = reader.ReadBoolean();
                this.logModelEvidence = reader.ReadDouble();
            }
        }

        /// <summary>
        /// The event that is fired at the end of each iteration of the Bayes point machine classifier training algorithm.
        /// </summary>
        /// <remarks>
        /// Subscribing a handler to this event may have negative effects on the performance of the training algorithm in terms of 
        /// both memory consumption and execution speed.
        /// </remarks>
        [field: NonSerialized]
        public event EventHandler<BayesPointMachineClassifierIterationChangedEventArgs> IterationChanged;
        
        #endregion

        #region Properties

        /// <summary>
        /// Gets the capabilities of the learner.
        /// </summary>
        ICapabilities ILearner.Capabilities
        {
            get { return this.capabilities; }
        }

        /// <summary>
        /// Gets the settings of the learner.
        /// </summary>
        ISettings ILearner.Settings
        {
            get { return this.Settings; }
        }

        /// <summary>
        /// Gets the capabilities of the predictor.
        /// </summary>
        public IPredictorCapabilities Capabilities
        {
            get { return this.capabilities; }
        }

        /// <summary>
        /// Gets or sets the settings of the Bayes point machine classifier.
        /// </summary>
        public IBayesPointMachineClassifierSettings<TLabel, TTrainingSettings, TPredictionSettings> Settings { get; protected set; }

        /// <summary>
        /// Gets the natural logarithm of the evidence for the Bayes point machine classifier model. 
        /// Use this for model selection.
        /// </summary>
        public double LogModelEvidence
        {
            get
            {
                if (!this.Settings.Training.ComputeModelEvidence)
                {
                    throw new InvalidOperationException("Evidence is only computed during training when Settings.ComputeModelEvidence is set to true.");
                }

                if (!this.IsTrained)
                {
                    throw new InvalidOperationException("Evidence can only be obtained from a trained model.");
                }

                if (this.isIncrementallyTrained)
                {
                    throw new InvalidOperationException("Evidence cannot be obtained from an incrementally trained model.");
                }

                return this.logModelEvidence;
            }
        }

        /// <summary>
        /// Gets the posterior distributions over the weights of the Bayes point machine classifier.
        /// </summary>
        public IReadOnlyList<IReadOnlyList<Gaussian>> WeightPosteriorDistributions
        {
            get
            {
                if (!this.IsTrained)
                {
                    throw new InvalidOperationException(
                        "The posterior distributions over weights can only be obtained from a trained classifier.");
                }

                return this.InferenceAlgorithms.WeightDistributions;
            }
        }

        /// <summary>
        /// Gets a value indicating whether the Bayes point machine classifier is trained.
        /// </summary>
        public bool IsTrained { get; private set; }

        /// <summary>
        /// Gets the mapping used for accessing data in the native format.
        /// </summary>
        protected IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel> Mapping { get; private set; }

        /// <summary>
        /// Gets the point estimator for the label distribution.
        /// </summary>
        protected abstract Func<TLabelDistribution, TLabel> PointEstimator { get; }

        /// <summary>
        /// Gets or sets the generated inference algorithms for training and prediction.
        /// </summary>
        protected IInferenceAlgorithms<TLabel, TLabelDistribution> InferenceAlgorithms { get; set; }

        #endregion

        #region IPredictor implementation

        /// <summary>
        /// Trains the Bayes point machine classifier on the specified instances.
        /// </summary>
        /// <param name="instanceSource">The source of instances.</param>
        /// <param name="labelSource">An optional source of labels.</param>
        public void Train(TInstanceSource instanceSource, TLabelSource labelSource = default(TLabelSource))
        {
            if (this.IsTrained)
            {
                throw new InvalidOperationException("This classifier has already been trained. For incremental training, use TrainIncremental().");
            }

            this.TrainIncremental(instanceSource, labelSource);
        }

        /// <summary>
        /// Makes a prediction for the specified instance. Uncertainty in the prediction is characterized in the form of a distribution.
        /// </summary>
        /// <param name="instance">The instance to make a prediction for.</param>
        /// <param name="instanceSource">An optional source of instances which provides <paramref name="instance"/>.</param>
        /// <returns>The prediction for <paramref name="instance"/>, with uncertainty.</returns>
        public TLabelDistribution PredictDistribution(TInstance instance, TInstanceSource instanceSource = default(TInstanceSource))
        {
            if (!this.IsTrained)
            {
                throw new InvalidOperationException("This classifier is not trained.");
            }

            if (instance == null)
            {
                throw new ArgumentNullException(nameof(instance));
            }

            // Retrieve the data through the mapping and check consistency (through mapping)
            double[][] featureValues = { this.Mapping.GetSingleFeatureVectorValuesSafe(instance, instanceSource) };
            int[] indexes = this.Mapping.GetSingleFeatureVectorIndexesSafe(instance, instanceSource);
            int[][] featureIndexes = indexes == null ? null : new[] { indexes };

            // Check features are consistent
            CheckFeatureConsistency(this.InferenceAlgorithms.UseSparseFeatures, this.InferenceAlgorithms.FeatureCount, featureValues, featureIndexes);

            // Predict the label
            return this.PredictDistribution(featureValues, featureIndexes).First();
        }

        /// <summary>
        /// Makes predictions for the specified instances. Uncertainty in a prediction is characterized in the form of a distribution.
        /// </summary>
        /// <param name="instanceSource">The source of instances to make predictions for.</param>
        /// <returns>The prediction for every instance provided by <paramref name="instanceSource"/>, with uncertainty.</returns>
        public IEnumerable<TLabelDistribution> PredictDistribution(TInstanceSource instanceSource)
        {
            if (!this.IsTrained)
            {
                throw new InvalidOperationException("This classifier is not trained.");
            }

            if (instanceSource == null)
            {
                throw new ArgumentNullException(nameof(instanceSource));
            }

            // Retrieve the data through the mapping and check consistency
            bool isSparse = this.Mapping.IsSparse(instanceSource);
            int featureCount = this.Mapping.GetFeatureCountSafe(instanceSource);
            this.CheckFeatureAlgorithmConsistency(isSparse, featureCount);

            double[][] featureValues = this.Mapping.GetAllFeatureVectorValuesSafe(instanceSource);
            int[][] featureIndexes = this.Mapping.GetAllFeatureVectorIndexesSafe(instanceSource);
            CheckFeatureConsistency(isSparse, featureCount, featureValues, featureIndexes);

            // Predict the labels
            return this.PredictDistribution(featureValues, featureIndexes);
        }

        /// <summary>
        /// Makes a prediction for the specified instance. Uncertainty is discarded and a "best" label is returned.
        /// </summary>
        /// <param name="instance">The instance to make a prediction for.</param>
        /// <param name="instanceSource">An optional source of instances which provides <paramref name="instance"/>.</param>
        /// <returns>The "best" label for <paramref name="instance"/>, discarding uncertainty.</returns>
        /// <remarks>The definition of "best" depends on the setting of <see cref="LossFunction"/>.</remarks>
        public TLabel Predict(TInstance instance, TInstanceSource instanceSource = default(TInstanceSource))
        {
            return this.PointEstimator(this.PredictDistribution(instance, instanceSource));
        }

        /// <summary>
        /// Makes predictions for the specified instances. Uncertainty is discarded and "best" labels are returned.
        /// </summary>
        /// <param name="instanceSource">The source of instances to make predictions for.</param>
        /// <returns>The "best" label for every instance provided by <paramref name="instanceSource"/>.</returns>
        /// <remarks>The definition of "best" depends on the setting of <see cref="LossFunction"/>.</remarks>
        public IEnumerable<TLabel> Predict(TInstanceSource instanceSource)
        {
            return this.PredictDistribution(instanceSource).Select(this.PointEstimator);
        }

        #endregion

        #region IBayesPointMachineClassifier implementation

        /// <summary>
        /// Incrementally trains the Bayes point machine classifier on the specified instances.
        /// </summary>
        /// <param name="instanceSource">The source of instances.</param>
        /// <param name="labelSource">An optional source of labels.</param>
        public virtual void TrainIncremental(TInstanceSource instanceSource, TLabelSource labelSource = default(TLabelSource))
        {
            if (instanceSource == null)
            {
                throw new ArgumentNullException(nameof(instanceSource));
            }

            if (this.Settings.Training.BatchCount == 1)
            {
                this.TrainOnSingleBatch(instanceSource, labelSource);
            }
            else
            {
                this.TrainOnMultipleBatches(instanceSource, labelSource);
            }
        }

        #endregion

        #region ICustomSerializable implementation

        /// <summary>
        /// Saves the state of the Bayes point machine classifier using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the state of the Bayes point machine classifier to.</param>
        public virtual void SaveForwardCompatible(IWriter writer)
        {
            writer.Write(this.customSerializationGuid);
            writer.Write(CustomSerializationVersion);

            writer.Write(this.IsTrained);
            writer.Write(this.isIncrementallyTrained);
            writer.Write(this.logModelEvidence);

            // No need to write capabilities as they can be created from scratch

            // Write settings and inference algorithms in derived classes
        }

        #endregion

        #region Template methods for inference algorithms

        /// <summary>
        /// Creates the inference algorithms for the specified feature representation.
        /// </summary>
        /// <param name="useSparseFeatures">If true, the inference algorithms expect features in a sparse representation.</param>
        /// <param name="featureCount">The number of features that the inference algorithms use.</param>
        /// <returns>The inference algorithms for the specified feature representation.</returns>
        protected abstract IInferenceAlgorithms<TLabel, TLabelDistribution> CreateInferenceAlgorithms(bool useSparseFeatures, int featureCount);

        /// <summary>
        /// Runs the prediction algorithm on the specified data.
        /// </summary>
        /// <param name="featureValues">The feature values.</param>
        /// <param name="featureIndexes">The feature indexes.</param>
        /// <returns>The predictive distribution over labels.</returns>
        protected abstract IEnumerable<TLabelDistribution> PredictDistribution(double[][] featureValues, int[][] featureIndexes);

        #endregion

        #region Helper methods

        /// <summary>
        /// Verifies that the data returned by the native mapping is consistent.
        /// </summary>
        /// <param name="isSparse">If true, a sparse feature representation is expected. If false, the representation is dense.</param>
        /// <param name="featureCount">The total number of features accessible through the mapping in either representation.</param>
        /// <param name="featureValues">The feature values returned by the mapping.</param>
        /// <param name="featureIndexes">The feature indexes returned by the mapping.</param>
        /// <param name="labels">The labels returned by the mapping.</param>
        /// <exception cref="BayesPointMachineClassifierException">Thrown if the training data is inconsistent.</exception>
        protected virtual void CheckDataConsistency(bool isSparse, int featureCount, double[][] featureValues, int[][] featureIndexes, TLabel[] labels)
        {
            CheckFeatureConsistency(isSparse, featureCount, featureValues, featureIndexes);

            if (featureValues.Length != labels.Length)
            {
                throw new BayesPointMachineClassifierException("There must be the same number of feature values and labels.");
            }
        }

        /// <summary>
        /// Verifies that the features returned by the native mapping are consistent.
        /// </summary>
        /// <param name="isSparse">If true, a sparse feature representation is expected. If false, the representation is dense.</param>
        /// <param name="featureCount">The total number of features accessible through the mapping in either representation.</param>
        /// <param name="featureValues">The feature values returned by the mapping.</param>
        /// <param name="featureIndexes">The feature indexes returned by the mapping.</param>
        /// <exception cref="BayesPointMachineClassifierException">Thrown if the features are inconsistent.</exception>
        private static void CheckFeatureConsistency(bool isSparse, int featureCount, double[][] featureValues, int[][] featureIndexes)
        {
            Debug.Assert(featureCount >= 0, "The number of features must not be negative.");
            Debug.Assert(featureValues != null, "The feature values must not be null.");
            
            if (isSparse)
            {
                if (featureIndexes == null)
                {
                    throw new BayesPointMachineClassifierException("In a sparse feature representation, the feature indexes must not be null.");
                }

                if (featureValues.Length != featureIndexes.Length)
                {
                    throw new BayesPointMachineClassifierException("In a sparse feature representation, there must be the same number of feature values and feature indexes.");
                }

                for (int instance = 0; instance < featureValues.Length; instance++)
                {
                    if (featureValues[instance].Length != featureIndexes[instance].Length)
                    {
                        throw new BayesPointMachineClassifierException("In a sparse feature representation, each single instances must have the same number of feature values and feature indexes.");
                    }

                    if (featureIndexes[instance].Any(index => index >= featureCount))
                    {
                        throw new BayesPointMachineClassifierException("In a sparse feature representation, no index must be greater than the total number of features.");
                    }
                }
            }
            else
            {
                if (featureIndexes != null)
                {
                    throw new BayesPointMachineClassifierException("In a dense feature representation, the feature indexes must be null.");
                }

                if (featureValues.Any(values => values.Length != featureCount))
                {
                    throw new BayesPointMachineClassifierException("In a dense feature representation, all instances must have the same number of feature values.");
                }
            }
        }

        /// <summary>
        /// Runs the training algorithm on a single batch of instances.
        /// </summary>
        /// <param name="instanceSource">The source of instances.</param>
        /// <param name="labelSource">An optional source of labels.</param>
        private void TrainOnSingleBatch(TInstanceSource instanceSource, TLabelSource labelSource)
        {
            // Retrieve the data through the mapping
            bool isSparse = this.Mapping.IsSparse(instanceSource);
            int featureCount = this.Mapping.GetFeatureCountSafe(instanceSource);
            double[][] featureValues = this.Mapping.GetAllFeatureVectorValuesSafe(instanceSource);
            int[][] featureIndexes = this.Mapping.GetAllFeatureVectorIndexesSafe(instanceSource);
            TLabel[] labels = this.Mapping.GetAllLabelsSafe(instanceSource, labelSource);

            // Check training data is self-consistent
            this.CheckDataConsistency(isSparse, featureCount, featureValues, featureIndexes, labels);

            // Create the inference algorithms, if required
            if (!this.IsTrained)
            {
                this.InferenceAlgorithms = this.CreateInferenceAlgorithms(isSparse, featureCount);
            }

            // Set the number of training data batches and reset per-batch output messages
            this.InferenceAlgorithms.SetBatchCount(this.Settings.Training.BatchCount);

            // Check features are consistent with the inference algorithms
            this.CheckFeatureAlgorithmConsistency(isSparse, featureCount);

            // Subscribe to the IterationChanged event on the training algorithm
            bool subscribed = this.TrySubscribeToIterationChangedEvent();

            // Run the training algorithm
            this.logModelEvidence = this.InferenceAlgorithms.Train(
                featureValues, featureIndexes, labels, this.Settings.Training.IterationCount);

            // Unsubscribe from IterationChanged event
            this.UnsubscribeFromIterationChangedEvent(subscribed);

            // Mark classifier as trained
            if (this.IsTrained)
            {
                this.isIncrementallyTrained = true;
            }

            this.IsTrained = true;
        }

        /// <summary>
        /// Runs the training algorithm on multiple batches of instances.
        /// </summary>
        /// <param name="instanceSource">The source of instances.</param>
        /// <param name="labelSource">An optional source of labels.</param>
        private void TrainOnMultipleBatches(TInstanceSource instanceSource, TLabelSource labelSource)
        {
            double logEvidence = 0.0;

            // Retrieve data properties through the mapping
            bool isSparse = this.Mapping.IsSparse(instanceSource);
            int featureCount = this.Mapping.GetFeatureCountSafe(instanceSource);

            // Create the inference algorithms, if required
            if (!this.IsTrained)
            {
                this.InferenceAlgorithms = this.CreateInferenceAlgorithms(isSparse, featureCount);
            }

            // Set the number of training data batches and reset per-batch constraint distributions
            this.InferenceAlgorithms.SetBatchCount(this.Settings.Training.BatchCount);

            for (int iteration = 1; iteration <= this.Settings.Training.IterationCount; iteration++)
            {
                for (int batch = 0; batch < this.Settings.Training.BatchCount; batch++)
                {
                    // Retrieve the feature values, indexes and labels through the mapping
                    double[][] featureValues = this.Mapping.GetAllFeatureVectorValuesSafe(instanceSource, batch);
                    int[][] featureIndexes = this.Mapping.GetAllFeatureVectorIndexesSafe(instanceSource, batch);
                    TLabel[] labels = this.Mapping.GetAllLabelsSafe(instanceSource, labelSource, batch);

                    // Check training data is self-consistent
                    this.CheckDataConsistency(isSparse, featureCount, featureValues, featureIndexes, labels);

                    // Check features are consistent with the inference algorithms
                    this.CheckFeatureAlgorithmConsistency(isSparse, featureCount);

                    // Run the training algorithm on the specified batch
                    int batchedIterationCount = iteration < MaxBatchedIterationCount ? iteration : MaxBatchedIterationCount;
                    double logBatchEvidence = this.InferenceAlgorithms.Train(
                        featureValues, featureIndexes, labels, batchedIterationCount, batch);

                    // Last iteration: compute evidence on the specified batch
                    if (iteration == this.Settings.Training.IterationCount)
                    {
                        logEvidence += logBatchEvidence;
                    }
                }

                // Raise IterationChanged event
                this.OnBatchedIterationChanged(iteration);
            }

            this.logModelEvidence += logEvidence;

            // Mark classifier as trained
            if (this.IsTrained)
            {
                this.isIncrementallyTrained = true;
            }

            this.IsTrained = true;
        }

        /// <summary>
        /// Verifies that the features are compatible with the inference algorithms.
        /// </summary>
        /// <param name="isSparse">If true, the mapping uses a sparse feature representation. If false, the representation is dense.</param>
        /// <param name="featureCount">The total number of features accessible through the mapping in either representation.</param>
        /// <exception cref="BayesPointMachineClassifierException">
        /// Thrown if the features are incompatible with the inference algorithms.
        /// </exception>
        private void CheckFeatureAlgorithmConsistency(bool isSparse, int featureCount)
        {
            Debug.Assert(featureCount >= 0, "The number of features must not be negative.");
            Debug.Assert(this.InferenceAlgorithms != null, "The inference algorithms must not be null.");

            if (isSparse != this.InferenceAlgorithms.UseSparseFeatures || featureCount != this.InferenceAlgorithms.FeatureCount)
            {
                throw new BayesPointMachineClassifierException("The features are incompatible with the inference algorithms.");
            }
        }

        /// <summary>
        /// Subscribes the <see cref="IterationChanged"/> event handler to the <see cref="IterationChanged"/> event 
        /// on inference algorithms.
        /// </summary>
        /// <returns>True, if the subscription of the event handler was successful and false otherwise.</returns>
        private bool TrySubscribeToIterationChangedEvent()
        {
            EventHandler<BayesPointMachineClassifierIterationChangedEventArgs> handler = this.IterationChanged;
            if (handler == null)
            {
                return false;
            }

            this.InferenceAlgorithms.IterationChanged += this.OnIterationChanged;
            return true;
        }

        /// <summary>
        /// Unsubscribes the <see cref="IterationChanged"/> event handler from the <see cref="IterationChanged"/> event 
        /// on inference algorithms.
        /// </summary>
        /// <param name="unsubscribeEventHandler">If true, the event handler will be unsubscribed, otherwise not.</param>
        private void UnsubscribeFromIterationChangedEvent(bool unsubscribeEventHandler)
        {
            if (unsubscribeEventHandler)
            {
                this.InferenceAlgorithms.IterationChanged -= this.OnIterationChanged;
            }
        }

        /// <summary>
        /// Fires at the end of each iteration of the Bayes point machine classifier training algorithm.
        /// </summary>
        /// <param name="sender">The sender.</param>
        /// <param name="iterationChangedEventArgs">The information describing the change in iterations.</param>
        private void OnIterationChanged(object sender, BayesPointMachineClassifierIterationChangedEventArgs iterationChangedEventArgs)
        {
            this.IterationChanged?.Invoke(this, iterationChangedEventArgs);
        }

        /// <summary>
        /// Delivers the current marginal distributions over weights.
        /// </summary>
        /// <param name="completedIteration">The completed training algorithm iteration.</param>
        private void OnBatchedIterationChanged(int completedIteration)
        {
            // Raise IterationChanged event
            this.IterationChanged?.Invoke(this, new BayesPointMachineClassifierIterationChangedEventArgs(completedIteration, this.InferenceAlgorithms.WeightDistributions));
        }

        #endregion
    }
}
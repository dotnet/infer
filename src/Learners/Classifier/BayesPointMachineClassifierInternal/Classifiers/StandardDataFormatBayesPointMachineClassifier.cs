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
    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// An abstract base class for a Bayes point machine classifier which operates on data in the standard format.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of the instance source.</typeparam>
    /// <typeparam name="TInstance">The type of an instance.</typeparam>
    /// <typeparam name="TLabelSource">The type of the label source.</typeparam>
    /// <typeparam name="TStandardLabel">The type of a label in standard data format.</typeparam>
    /// <typeparam name="TNativeLabel">The type of a label in native data format.</typeparam>
    /// <typeparam name="TNativeLabelDistribution">The type of a distribution over labels in native data format.</typeparam>
    /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
    /// <typeparam name="TPredictionSettings">The type of the settings for prediction.</typeparam>
    /// <typeparam name="TNativePredictionSettings">The type of the settings for prediction on data in native data.</typeparam>
    [Serializable]
    internal abstract class StandardDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TStandardLabel, TNativeLabel, TNativeLabelDistribution, TTrainingSettings, TPredictionSettings, TNativePredictionSettings> :
        IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TStandardLabel, IDictionary<TStandardLabel, double>, TTrainingSettings, TPredictionSettings>
        where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        where TPredictionSettings : IBayesPointMachineClassifierPredictionSettings<TStandardLabel>
        where TNativePredictionSettings : IBayesPointMachineClassifierPredictionSettings<TNativeLabel>
    {
        #region Constants, constructors, events

        /// <summary>
        /// The current custom binary serialization version of the
        /// <see cref="StandardDataFormatBayesPointMachineClassifier{TInstanceSource,TInstance,TLabelSource,TStandardLabel,TNativeLabel,TNativeLabelDistribution,TTrainingSettings,TPredictionSettings,TNativePredictionSettings}"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// The custom binary serialization <see cref="Guid"/> of the
        /// <see cref="StandardDataFormatBayesPointMachineClassifier{TInstanceSource,TInstance,TLabelSource,TStandardLabel,TNativeLabel,TNativeLabelDistribution,TTrainingSettings,TPredictionSettings,TNativePredictionSettings}"/> class.
        /// </summary>
        private readonly Guid customSerializationGuid = new Guid("BBAC0187-0E7E-49AD-A39F-915B67BB5AC6");

        /// <summary>
        /// Initializes a new instance of the 
        /// <see cref="StandardDataFormatBayesPointMachineClassifier{TInstanceSource,TInstance,TLabelSource,TStandardLabel,TNativeLabel,TNativeLabelDistribution,TTrainingSettings,TPredictionSettings,TNativePredictionSettings}"/> class.
        /// </summary>
        protected StandardDataFormatBayesPointMachineClassifier()
        {
        }

        /// <summary>
        /// Initializes a new instance of the 
        /// <see cref="StandardDataFormatBayesPointMachineClassifier{TInstanceSource,TInstance,TLabelSource,TStandardLabel,TNativeLabel,TNativeLabelDistribution,TTrainingSettings,TPredictionSettings,TNativePredictionSettings}"/> 
        /// class from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the state of the Bayes point machine classifier from.</param>
        protected StandardDataFormatBayesPointMachineClassifier(IReader reader)
        {
            Debug.Assert(reader != null, "The reader must not be null.");

            reader.VerifySerializationGuid(
                this.customSerializationGuid, "The stream does not contain an Infer.NET Bayes point machine classifier.");
            reader.ReadSerializationVersion(CustomSerializationVersion);

            // Nothing to deserialize
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
            get { return this.Classifier.Capabilities; }
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
            get { return this.Classifier.Capabilities; }
        }

        /// <summary>
        /// Gets or sets the settings of the Bayes point machine classifier.
        /// </summary>
        public IBayesPointMachineClassifierSettings<TStandardLabel, TTrainingSettings, TPredictionSettings> Settings { get; protected set; }

        /// <summary>
        /// Gets the natural logarithm of the model evidence. Use this for model selection.
        /// </summary>
        public double LogModelEvidence
        {
            get { return this.Classifier.LogModelEvidence; }
        }

        /// <summary>
        /// Gets the posterior distributions of the weights of the Bayes point machine classifier.
        /// </summary>
        public IReadOnlyList<IReadOnlyList<Gaussian>> WeightPosteriorDistributions
        {
            get { return this.Classifier.WeightPosteriorDistributions; }
        }

        /// <summary>
        /// Gets or sets the Bayes point machine classifier which operates on data in the native format.
        /// </summary>
        protected NativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TNativeLabel, TNativeLabelDistribution, TTrainingSettings, TNativePredictionSettings> Classifier { get; set; }

        /// <summary>
        /// Gets the point estimator for the label distribution.
        /// </summary>
        private Func<IDictionary<TStandardLabel, double>, TStandardLabel> PointEstimator
        {
            get
            {
                Func<TStandardLabel, TStandardLabel, double> customLossFunction;
                LossFunction lossFunction = this.Settings.Prediction.GetPredictionLossFunction(out customLossFunction);

                if (lossFunction == LossFunction.Squared || lossFunction == LossFunction.Absolute)
                {
                    throw new InvalidOperationException("Please specify a custom loss function for labels in standard format.");
                }

                return (lossFunction == LossFunction.Custom) ? 
                    Learners.PointEstimator.ForDiscrete(customLossFunction) : Learners.PointEstimator.ForDiscrete<TStandardLabel>(lossFunction);
            }
        }

        #endregion

        #region IPredictor implementation

        /// <summary>
        /// Trains the Bayes point machine classifier on the specified instances.
        /// </summary>
        /// <param name="instanceSource">The source of instances.</param>
        /// <param name="labelSource">The source of labels.</param>
        public void Train(TInstanceSource instanceSource, TLabelSource labelSource = default(TLabelSource))
        {
            if (this.Classifier.IsTrained)
            {
                throw new InvalidOperationException("This classifier has already been trained. For incremental training, use TrainIncremental().");
            }

            this.TrainIncremental(instanceSource, labelSource);
        }

        /// <summary>
        /// Makes a prediction for the specified instance. Uncertainty in the prediction is characterized in the form 
        /// of a discrete distribution over labels.
        /// </summary>
        /// <param name="instance">The instance to make a prediction for.</param>
        /// <param name="instanceSource">The source of instances which provides <paramref name="instance"/>.</param>
        /// <returns>The prediction for <paramref name="instance"/>, with uncertainty.</returns>
        public IDictionary<TStandardLabel, double> PredictDistribution(TInstance instance, TInstanceSource instanceSource = default(TInstanceSource))
        {
            this.UpdateNativePredictionSettings();
            return this.GetStandardLabelDistribution(this.Classifier.PredictDistribution(instance, instanceSource));
        }

        /// <summary>
        /// Makes predictions for the specified instances. Uncertainty in a prediction is characterized in the form
        /// of a discrete distribution over labels.
        /// </summary>
        /// <param name="instanceSource">The source of instances to make predictions for.</param>
        /// <returns>The prediction for every instance provided by <paramref name="instanceSource"/>, with uncertainty.</returns>
        public IEnumerable<IDictionary<TStandardLabel, double>> PredictDistribution(TInstanceSource instanceSource)
        {
            this.UpdateNativePredictionSettings();
            IEnumerable<TNativeLabelDistribution> nativeLabelDistributions = this.Classifier.PredictDistribution(instanceSource);
            return nativeLabelDistributions.Select(this.GetStandardLabelDistribution);
        }

        /// <summary>
        /// Makes a prediction for the specified instance. Uncertainty is discarded and a "best" label is returned.
        /// </summary>
        /// <param name="instance">The instance to make a prediction for.</param>
        /// <param name="instanceSource">The source of instances which provides <paramref name="instance"/>.</param>
        /// <returns>The "best" label for <paramref name="instance"/>, discarding uncertainty.</returns>
        /// <remarks>The definition of "best" depends on the setting of <see cref="LossFunction"/>.</remarks>
        public TStandardLabel Predict(TInstance instance, TInstanceSource instanceSource = default(TInstanceSource))
        {
            return this.PointEstimator(this.PredictDistribution(instance, instanceSource)); 
        }

        /// <summary>
        /// Makes predictions for the specified instances. Uncertainty is discarded and "best" labels are returned.
        /// </summary>
        /// <param name="instanceSource">The source of instances to make predictions for.</param>
        /// <returns>The "best" label for every instance provided by <paramref name="instanceSource"/>.</returns>
        /// <remarks>The definition of "best" depends on the setting of <see cref="LossFunction"/>.</remarks>
        public IEnumerable<TStandardLabel> Predict(TInstanceSource instanceSource)
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
        public void TrainIncremental(TInstanceSource instanceSource, TLabelSource labelSource = default(TLabelSource))
        {
            if (instanceSource == null)
            {
                throw new ArgumentNullException(nameof(instanceSource));
            }

            if (!this.Classifier.IsTrained)
            {
                // Update evidence setting of native classifier
                this.UpdateGuardedNativeTrainingSettings();
                this.SetClassLabels(instanceSource, labelSource);
            }
            else
            {
                this.CheckClassLabelConsistency(instanceSource, labelSource);
            }

            // Update training settings of native classifier
            this.UpdateUnguardedNativeTrainingSettings();
            
            // Configure mapping
            this.SetBatchCount(this.Settings.Training.BatchCount);

            bool subscribed = this.TrySubscribeToIterationChangedEvent();
            this.Classifier.TrainIncremental(instanceSource, labelSource);
            this.UnsubscribeFromIterationChangedEvent(subscribed);

            this.SetBatchCount(1); // No batching during prediction
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
        }

        #endregion

        #region Helper methods

        /// <summary>
        /// Sets the number of batches the training data is split into and resets constraints on the weight distributions.
        /// </summary>
        /// <param name="value">The number of batches to use.</param>
        protected abstract void SetBatchCount(int value);

        /// <summary>
        /// Gets the distribution over class labels in the standard data format.
        /// </summary>
        /// <param name="nativeLabelDistribution">The distribution over class labels in the native data format.</param>
        /// <returns>The class label distribution in standard data format.</returns>
        protected abstract IDictionary<TStandardLabel, double> GetStandardLabelDistribution(TNativeLabelDistribution nativeLabelDistribution);

        /// <summary>
        /// Gets the class labels in the standard data format.
        /// </summary>
        /// <param name="nativeLabel">The class labels in the native data format.</param>
        /// <returns>The class label in standard data format.</returns>
        protected abstract TStandardLabel GetStandardLabel(TNativeLabel nativeLabel);

        /// <summary>
        /// Sets the class labels.
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">An optional label source.</param>
        protected abstract void SetClassLabels(TInstanceSource instanceSource, TLabelSource labelSource = default(TLabelSource));

        /// <summary>
        /// Checks that the class labels provided by the mapping are consistent.
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">An optional label source.</param>
        protected abstract void CheckClassLabelConsistency(TInstanceSource instanceSource, TLabelSource labelSource = default(TLabelSource));

        /// <summary>
        /// Updates the prediction settings of the native data format classifier with those of the standard data format classifier.
        /// </summary>
        protected virtual void UpdateNativePredictionSettings()
        {
        }

        /// <summary>
        /// Updates the guarded training settings of the native data format classifier with those of the standard data format classifier.
        /// </summary>
        private void UpdateGuardedNativeTrainingSettings()
        {
            this.Classifier.Settings.Training.ComputeModelEvidence = this.Settings.Training.ComputeModelEvidence;
        }

        /// <summary>
        /// Updates the unguarded training settings of the native data format classifier with those of the standard data format classifier.
        /// </summary>
        private void UpdateUnguardedNativeTrainingSettings()
        {
            this.Classifier.Settings.Training.BatchCount = this.Settings.Training.BatchCount;
            this.Classifier.Settings.Training.IterationCount = this.Settings.Training.IterationCount;
        }

        /// <summary>
        /// Subscribes the <see cref="IterationChanged"/> event handler to the <see cref="IterationChanged"/> event 
        /// on the Bayes point machine classifier operating on data in the native format.
        /// </summary>
        /// <returns>True, if the subscription of the event handler was successful and false otherwise.</returns>
        private bool TrySubscribeToIterationChangedEvent()
        {
            EventHandler<BayesPointMachineClassifierIterationChangedEventArgs> handler = this.IterationChanged;
            if (handler == null)
            {
                return false;
            }

            this.Classifier.IterationChanged += this.OnIterationChanged;
            return true;
        }

        /// <summary>
        /// Unsubscribes the <see cref="IterationChanged"/> event handler from the <see cref="IterationChanged"/> event 
        /// on the Bayes point machine classifier operating on data in the native format.
        /// </summary>
        /// <param name="unsubscribeEventHandler">If true, the event handler will be unsubscribed, otherwise not.</param>
        private void UnsubscribeFromIterationChangedEvent(bool unsubscribeEventHandler)
        {
            if (unsubscribeEventHandler)
            {
                this.Classifier.IterationChanged -= this.OnIterationChanged;
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

        #endregion
    }
}

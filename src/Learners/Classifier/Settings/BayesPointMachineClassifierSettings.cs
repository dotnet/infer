// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;
    using System.IO;

    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// Abstract settings of the Bayes point machine classifier.
    /// </summary>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
    /// <typeparam name="TPredictionSettings">The type of the settings for prediction.</typeparam>
    [Serializable]
    public abstract class BayesPointMachineClassifierSettings<TLabel, TTrainingSettings, TPredictionSettings> :
        IBayesPointMachineClassifierSettings<TLabel, TTrainingSettings, TPredictionSettings>
        where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        where TPredictionSettings : IBayesPointMachineClassifierPredictionSettings<TLabel>
    {
        /// <summary>
        /// The current custom binary serialization version of the 
        /// <see cref="BayesPointMachineClassifierSettings{TLabel,TTrainingSettings,TPredictionSettings}"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// The custom binary serialization <see cref="Guid"/> of the
        /// <see cref="BayesPointMachineClassifierSettings{TLabel,TTrainingSettings,TPredictionSettings}"/> class.
        /// </summary>
        private readonly Guid customSerializationGuid = new Guid("E80D4612-EF7B-48DC-9BFE-92DB0858982D");

        /// <summary>
        /// Initializes a new instance of the <see cref="BayesPointMachineClassifierSettings{TLabel,TTrainingSettings,TPredictionSettings}"/> class. 
        /// </summary>
        protected BayesPointMachineClassifierSettings()
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="BayesPointMachineClassifierSettings{TLabel,TTrainingSettings,TPredictionSettings}"/> class 
        /// from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the settings of the Bayes point machine classifier from.</param>
        protected BayesPointMachineClassifierSettings(IReader reader)
        {
            reader.VerifySerializationGuid(
                this.customSerializationGuid, "The binary stream does not contain the settings of an Infer.NET Bayes point machine classifier.");
            reader.ReadSerializationVersion(CustomSerializationVersion);

            // Nothing to deserialize.
        }

        /// <summary>
        /// Gets or sets the settings of the Bayes point machine classifier which affect training.
        /// </summary>
        public TTrainingSettings Training { get; protected set; }

        /// <summary>
        /// Gets or sets the settings of the Bayes point machine classifier which affect prediction.
        /// </summary>
        public TPredictionSettings Prediction { get; protected set; }

        /// <summary>
        /// Saves the settings of the Bayes point machine classifier using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the settings to.</param>
        public virtual void SaveForwardCompatible(IWriter writer)
        {
            writer.Write(this.customSerializationGuid);
            writer.Write(CustomSerializationVersion);
        }
    }
}
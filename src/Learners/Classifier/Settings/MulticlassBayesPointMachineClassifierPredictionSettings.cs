// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;
    using System.IO;

    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// Settings for the multi-class Bayes point machine classifier which affect prediction.
    /// </summary>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    /// <remarks>
    /// These settings can be modified after training.
    /// </remarks>
    [Serializable]
    public class MulticlassBayesPointMachineClassifierPredictionSettings<TLabel> : BayesPointMachineClassifierPredictionSettings<TLabel>
    {
        /// <summary>
        /// The default number of iterations of the prediction algorithm.
        /// </summary>
        public const int IterationCountDefault = 10;

        /// <summary>
        /// The current custom binary serialization version of the <see cref="MulticlassBayesPointMachineClassifierPredictionSettings{TLabel}"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// The number of iterations of the prediction algorithm.
        /// </summary>
        private int iterationCount;

        /// <summary>
        /// Initializes a new instance of the <see cref="MulticlassBayesPointMachineClassifierPredictionSettings{TLabel}"/> class.
        /// </summary>
        internal MulticlassBayesPointMachineClassifierPredictionSettings()
        {
            this.iterationCount = IterationCountDefault;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="MulticlassBayesPointMachineClassifierPredictionSettings{TLabel}"/> class
        /// from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the prediction settings from.</param>
        internal MulticlassBayesPointMachineClassifierPredictionSettings(IReader reader) 
            : base(reader)
        {
            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);

            if (deserializedVersion == CustomSerializationVersion)
            {
                this.iterationCount = reader.ReadInt32();
            }
        }

        /// <summary>
        /// Gets or sets the number of iterations of the prediction algorithm.
        /// </summary>
        public int IterationCount
        {
            get
            {
                return this.iterationCount;
            }

            set
            {
                if (value <= 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(value), "The number of iterations of the prediction algorithm must be positive.");
                }

                this.iterationCount = value;
            }
        }

        /// <summary>
        /// Saves the prediction settings of the multi-class Bayes point machine classifier using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the prediction settings to.</param>
        public override void SaveForwardCompatible(IWriter writer)
        {
            base.SaveForwardCompatible(writer);

            writer.Write(CustomSerializationVersion);
            writer.Write(this.iterationCount);
        }
    }
}
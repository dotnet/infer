// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;

    /// <summary>
    /// Defines the capabilities of the Bayes point machine classifier.
    /// </summary>
    [Serializable]
    public class BayesPointMachineClassifierCapabilities : IPredictorCapabilities
    {
        #region ICapabilities implementation

        /// <summary>
        /// Gets a value indicating whether the Bayes point machine classifier is precompiled.
        /// </summary>
        /// <remarks>This is currently only relevant for Infer.NET.</remarks>
        public bool IsPrecompiled
        {
            get { return true; }
        }

        /// <summary>
        /// Gets a value indicating whether the Bayes point machine classifier supports missing data.
        /// </summary>
        public bool SupportsMissingData
        {
            get { return false; }
        }

        /// <summary>
        /// Gets a value indicating whether the Bayes point machine classifier supports sparse data.
        /// </summary>
        public bool SupportsSparseData
        {
            get { return true; }
        }

        /// <summary>
        /// Gets a value indicating whether the Bayes point machine classifier supports streamed data.
        /// </summary>
        public bool SupportsStreamedData
        {
            get { return false; }
        }

        /// <summary>
        /// Gets a value indicating whether the Bayes point machine classifier supports training on batched data.
        /// </summary>
        public bool SupportsBatchedTraining
        {
            get { return true; }
        }

        /// <summary>
        /// Gets a value indicating whether the Bayes point machine classifier supports distributed training.
        /// </summary>
        public bool SupportsDistributedTraining
        {
            get { return false; }
        }

        /// <summary>
        /// Gets a value indicating whether the Bayes point machine classifier supports incremental training.
        /// </summary>
        public bool SupportsIncrementalTraining
        {
            get { return true; }
        }

        /// <summary>
        /// Gets a value indicating whether the Bayes point machine classifier can compute 
        /// how well it matches the training data (usually for a specified set of hyper-parameters).
        /// </summary>
        public bool SupportsModelEvidenceComputation
        {
            get { return true; }
        }

        #endregion

        #region IPredictorCapabilities implementation

        /// <summary>
        /// Gets a value indicating whether the Bayes point machine classifier can compute predictive point estimates 
        /// via a user-defined loss function.
        /// </summary>
        public bool SupportsCustomPredictionLossFunction 
        {
            get { return true; }
        }

        #endregion
    }
}

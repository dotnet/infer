// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    /// <summary>
    /// Interface to predictor capabilities.
    /// </summary>
    public interface IPredictorCapabilities : ICapabilities
    {
        /// <summary>
        /// Gets a value indicating whether the predictor can compute predictive point estimates from a user-defined loss function.
        /// </summary>
        bool SupportsCustomPredictionLossFunction { get; }
    }
}

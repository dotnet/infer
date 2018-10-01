// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Probabilistic.Models
{
    /// <summary>
    /// Delegate for handlers of inference progress events.
    /// </summary>
    /// <param name="engine">The inference engine which invoked the inference query</param>
    /// <param name="progress">The progress object describing the progress of the inference algorithm</param>
    public delegate void InferenceProgressEventHandler(InferenceEngine engine, InferenceProgressEventArgs progress);

    /// <summary>
    /// Provides information about the progress of the inference algorithm, as it
    /// is being executed.
    /// </summary>
    public class InferenceProgressEventArgs : EventArgs
    {
        /// <summary>
        /// The iteration of inference that has just been completed.
        /// </summary>
        public int Iteration { get; internal set; }

        /// <summary>
        /// The compiled algorithm which is performing the inference.
        /// </summary>
        public IGeneratedAlgorithm Algorithm { get; internal set; }
    }
}
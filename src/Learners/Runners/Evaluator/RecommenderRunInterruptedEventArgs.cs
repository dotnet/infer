// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Diagnostics;

    /// <summary>
    /// The arguments of the <see cref="RecommenderRun.Interrupted"/> event.
    /// </summary>
    public class RecommenderRunInterruptedEventArgs : EventArgs
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="RecommenderRunInterruptedEventArgs"/> class.
        /// </summary>
        /// <param name="exception">The exception which caused the interruption.</param>
        public RecommenderRunInterruptedEventArgs(Exception exception)
        {
            Debug.Assert(exception != null, "A valid exception should be provided.");

            this.Exception = exception;
        }

        /// <summary>
        /// Gets the exception which caused the interruption.
        /// </summary>
        public Exception Exception { get; private set; }
    }
}
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Probabilistic
{
    /// <summary>
    /// Provides information about the progress of the inference algorithm, as it
    /// is being executed.
    /// </summary>
    public class ProgressChangedEventArgs : EventArgs
    {
        /// <summary>
        /// The iteration of inference that has just been completed (the first iteration is 0).
        /// </summary>
        public int Iteration
        {
            get { return iteration; }
        }

        private int iteration;

        /// <summary>
        /// Create a ProgressChangedEventArgs with the given iteration number
        /// </summary>
        /// <param name="iteration"></param>
        public ProgressChangedEventArgs(int iteration)
        {
            this.iteration = iteration;
        }
    }
}
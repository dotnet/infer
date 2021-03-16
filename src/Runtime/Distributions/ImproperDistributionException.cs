// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Runtime.Serialization;

    /// <summary>
    /// Exception thrown when a distribution is improper and its expectations need to be computed.
    /// </summary>
    [Serializable]
    public class ImproperDistributionException : Exception
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ImproperDistributionException"/> class.
        /// </summary>
        public ImproperDistributionException()
        {
        }

        /// <summary>
        /// Create a new Improper Distribution exception
        /// </summary>
        /// <param name="distribution"></param>
        public ImproperDistributionException(object distribution)
            : base("The distribution is improper (" + distribution + "). Cannot compute expectations.")
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ImproperDistributionException"/> class with a specified error message 
        /// and a reference to the inner exception that is the cause of this exception. 
        /// </summary>
        /// <param name="message">The error message that explains the reason for the exception.</param>
        /// <param name="inner">The exception that is the cause of the current exception.</param>
        public ImproperDistributionException(string message, Exception inner)
            : base(message, inner)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ImproperDistributionException"/> class.
        /// </summary>
        /// <param name="info">The object that holds the serialized object data.</param>
        /// <param name="context">The contextual information about the source or destination.</param>
        protected ImproperDistributionException(SerializationInfo info, StreamingContext context)
            : base(info, context)
        {
        }
    }
}
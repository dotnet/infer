// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Probabilistic.Factors
{
    using System.Runtime.Serialization;

    /// <summary>
    /// Improper message exception
    /// </summary>
    [Serializable]
    public class ImproperMessageException : Exception
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ImproperMessageException"/> class.
        /// </summary>
        public ImproperMessageException()
        {
        }

        /// <summary>
        /// Creates an improper message exception with the specified distribution
        /// </summary>
        /// <param name="distribution">Distribution instance</param>
        public ImproperMessageException(object distribution)
            : base("Improper distribution during inference (" + distribution + ").  Cannot perform inference on this model.")
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ImproperMessageException"/> class with a specified error message 
        /// and a reference to the inner exception that is the cause of this exception. 
        /// </summary>
        /// <param name="message">The error message that explains the reason for the exception.</param>
        /// <param name="inner">The exception that is the cause of the current exception.</param>
        public ImproperMessageException(string message, Exception inner)
            : base(message, inner)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ImproperMessageException"/> class.
        /// </summary>
        /// <param name="info">The object that holds the serialized object data.</param>
        /// <param name="context">The contextual information about the source or destination.</param>
        protected ImproperMessageException(SerializationInfo info, StreamingContext context)
            : base(info, context)
        {
        }
    }
}

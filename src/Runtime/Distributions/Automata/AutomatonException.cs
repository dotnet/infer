// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Runtime.Serialization;

    /// <summary>
    /// The exception that is thrown when an automaton-related error occurs.
    /// </summary>
    [Serializable]
    public class AutomatonException : Exception
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="AutomatonException"/> class.
        /// </summary>
        public AutomatonException()
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="AutomatonException"/> class with a specified error message.
        /// </summary>
        /// <param name="message">The error message that explains the reason for the exception.</param>
        public AutomatonException(string message)
            : base(message)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="AutomatonException"/> class
        /// with a specified error message and a reference to the inner exception that is the cause of this exception.
        /// </summary>
        /// <param name="message">The error message that explains the reason for the exception.</param>
        /// <param name="innerException">The exception that is the cause of the current exception.
        /// If the <paramref name="innerException"/> parameter is not a <see langword="null"/> reference,
        /// the current exception is raised in a <see langword="catch"/> block that handles the inner exception.</param>
        public AutomatonException(string message, Exception innerException)
            : base(message, innerException)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="AutomatonException"/> class.
        /// </summary>
        /// <param name="info">The object that holds the serialized object data.</param>
        /// <param name="context">The contextual information about the source or destination.</param>
        protected AutomatonException(SerializationInfo info, StreamingContext context)
            : base(info, context)
        {
        }
    }
}
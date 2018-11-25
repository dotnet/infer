// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Runtime.Serialization;

    /// <summary>
    /// The exception that is thrown when an enumeration becomes too large.
    /// </summary>
    [Serializable]
    public class AutomatonEnumerationCountException : AutomatonException
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="AutomatonEnumerationCountException"/> class.
        /// </summary>
        public AutomatonEnumerationCountException()
            : this("An enumeration across the paths of an automaton has exceeded its limit.")
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="AutomatonEnumerationCountException"/> class with a specified error message.
        /// </summary>
        /// <param name="message">The error message that explains the reason for the exception.</param>
        public AutomatonEnumerationCountException(string message)
            : base(message)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="AutomatonEnumerationCountException"/> class
        /// with a specified error message and a reference to the inner exception that is the cause of this exception.
        /// </summary>
        /// <param name="message">The error message that explains the reason for the exception.</param>
        /// <param name="innerException">The exception that is the cause of the current exception.
        /// If the <paramref name="innerException"/> parameter is not a <see langword="null"/> reference,
        /// the current exception is raised in a <see langword="catch"/> block that handles the inner exception.</param>
        public AutomatonEnumerationCountException(string message, Exception innerException)
            : base(message, innerException)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="AutomatonEnumerationCountException"/> class.
        /// </summary>
        /// <param name="maxEnumerationCount">The maximum enumeration count that has been exceeded.</param>
        public AutomatonEnumerationCountException(int maxEnumerationCount)
            : this($"An enumeration across the paths of an automaton has exceeded its limit of {maxEnumerationCount}")
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="AutomatonEnumerationCountException"/> class with serialized data.
        /// </summary>
        /// <param name="info">The object that holds the serialized object data.</param>
        /// <param name="context">The contextual information about the source or destination.</param>
        protected AutomatonEnumerationCountException(SerializationInfo info, StreamingContext context)
            : base(info, context)
        {
        }
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Runtime.Serialization;

    /// <summary>
    /// The exception that is thrown for invalid file formats.
    /// </summary>
    [Serializable]
    public class InvalidClassifierFileFormatException : Exception
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="InvalidClassifierFileFormatException"/> class.
        /// </summary>
        public InvalidClassifierFileFormatException()
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="InvalidClassifierFileFormatException"/> class with a specified error message.
        /// </summary>
        /// <param name="message">The error message.</param>
        public InvalidClassifierFileFormatException(string message)
            : base(message)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="InvalidClassifierFileFormatException"/> class
        /// with a specified error message and a reference to the inner exception that is the cause of this exception. 
        /// </summary>
        /// <param name="message">The error message that explains the reason for the exception.</param>
        /// <param name="inner">The exception that is the cause of the current exception.</param>
        public InvalidClassifierFileFormatException(string message, Exception inner)
            : base(message, inner)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="InvalidClassifierFileFormatException"/> class.
        /// </summary>
        /// <param name="info">The object that holds the serialized object data.</param>
        /// <param name="context">The contextual information about the source or destination.</param>
        protected InvalidClassifierFileFormatException(SerializationInfo info, StreamingContext context)
            : base(info, context)
        {
        }
    }
}

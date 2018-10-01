// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler
{
    using System;
    using System.Runtime.Serialization;

    /// <summary>
    /// The exception that is thrown in the case of an issue encountered by the Infer.NET Compiler.
    /// </summary>
    public class InferCompilerException : Exception
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="InferCompilerException"/> class.
        /// </summary>
        public InferCompilerException()
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="InferCompilerException"/> class with a specified error message.
        /// </summary>
        /// <param name="message">The error message.</param>
        public InferCompilerException(string message)
            : base(message)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="InferCompilerException"/> class with a specified error message 
        /// and a reference to the inner exception that is the cause of this exception. 
        /// </summary>
        /// <param name="message">The error message that explains the reason for the exception.</param>
        /// <param name="inner">The exception that is the cause of the current exception.</param>
        public InferCompilerException(string message, Exception inner)
            : base(message, inner)
        {
        }

        // This constructor is needed for serialization.
        protected InferCompilerException(SerializationInfo info, StreamingContext context) : base(info, context)
        {
        }
    }
}

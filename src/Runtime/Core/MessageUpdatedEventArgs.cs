// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Probabilistic
{
    /// <summary>
    /// Delegate for handlers of message updated events.
    /// </summary>
    /// <param name="algorithm">The generated algorithm which is being executed</param>
    /// <param name="messageEvent">The event object describing the message that has been computed</param>
    public delegate void MessageUpdatedEventHandler(IGeneratedAlgorithm algorithm, MessageUpdatedEventArgs messageEvent);

    /// <summary>
    /// Provides information about a message that has just been updated, in the course
    /// of executing an inference algorithm.
    /// </summary>
    public class MessageUpdatedEventArgs : EventArgs
    {
        /// <summary>
        /// The name of the variable holding the message
        /// in the generated code.
        /// </summary>
        public string MessageId
        {
            get;
            internal set;
        }

        /// <summary>
        /// The message that was computed.  Note: this is the actual message
        /// and not a copy, so it should not be modified.
        /// </summary>
        public object Message
        {
            get;
            internal set;
        }

        public override string ToString()
        {
            return MessageId + ": " + Message;
            ;
        }
    }
}

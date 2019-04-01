// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Diagnostics;
    using Attributes;
    using Utilities;

    public static class Tracing
    {
        /// <summary>
        /// Used in generated code to write a value with descriptive text
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="input"></param>
        /// <param name="text"></param>
        /// <returns><paramref name="input"/></returns>
        public static T Trace<T>([IsReturned] T input, string text)
        {
            System.Diagnostics.Trace.WriteLine(StringUtil.JoinColumns(text, ": ", input));
            return input;
        }

        /// <summary>
        /// Used in generated code to fire an event when a message is updated (and optionally writes a trace as well).
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="input"></param>
        /// <param name="text"></param>
        /// <param name="onMessageUpdated"></param>
        /// <param name="doTraceAsWell"></param>
        /// <returns></returns>
        public static T FireEvent<T>([IsReturned] T input, string text, Action<MessageUpdatedEventArgs> onMessageUpdated, bool doTraceAsWell)
        {
            if (doTraceAsWell)
                Trace(input, text);
            var messageEvent = new MessageUpdatedEventArgs
            {
                MessageId = text,
                Message = input
            };
            onMessageUpdated(messageEvent);
            return input;
        }

    }
}

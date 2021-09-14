// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Utilities
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Runtime.Serialization;

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Thrown when Assert.IsTrue fails.
    /// </summary>
    [Serializable]
    internal class AssertFailedException : Exception
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="AssertFailedException"/> class.
        /// </summary>
        public AssertFailedException()
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="AssertFailedException"/> class with a specified error message.
        /// </summary>
        /// <param name="message">The error message.</param>
        public AssertFailedException(string message)
            : base(message)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="AssertFailedException"/> class with a specified error message 
        /// and a reference to the inner exception that is the cause of this exception. 
        /// </summary>
        /// <param name="message">The error message that explains the reason for the exception.</param>
        /// <param name="inner">The exception that is the cause of the current exception.</param>
        public AssertFailedException(string message, Exception inner)
            : base(message, inner)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="AssertFailedException"/> class.
        /// </summary>
        /// <param name="info">The object that holds the serialized object data.</param>
        /// <param name="context">The contextual information about the source or destination.</param>
        protected AssertFailedException(SerializationInfo info, StreamingContext context)
            : base(info, context)
        {
        }
    }

    /// <summary>
    /// Assertion methods for debugging.
    /// </summary>
    /// <remarks>To catch assertion failures in the debugger:
    /// Debug -> Exceptions -> Add
    /// Microsoft.VisualStudio.TestTools.UnitTesting.AssertFailedException
    /// </remarks>
    [DebuggerNonUserCode()]
    internal static class Assert
    {
        /// <summary>
        /// Checks if a condition is true; if not, an exception is thrown with an error message.
        /// </summary>
        /// <param name="condition">Condition that must be true</param>
        /// <param name="message">Message to be output by the exception</param>
        // TM: This routine should not have [ConditionalAttribute("DEBUG")]
        public static void IsTrue(bool condition, string message)
        {
            if (!condition)
            {
#if false
    // Check to see if we're running in a unit test environment - if so
    // call the unit test assert. If not, throw an app exception, as we
    // don't want to be shipping test assemblies
                if (System.AppDomain.CurrentDomain.FriendlyName.Contains("UnitTest"))
                    throw new Microsoft.VisualStudio.TestTools.UnitTesting.AssertFailedException(message);
                else
#endif
                throw new AssertFailedException(message);
            }
        }

        /// <summary>
        /// Checks if a condition is true; if not, an exception  without error message is thrown.
        /// </summary>
        /// <param name="condition">Condition that must be true</param>
        // TM: This routine should not have [ConditionalAttribute("DEBUG")]
        public static void IsTrue(bool condition)
        {
            if (!condition)
            {
                throw new AssertFailedException();
            }
        }
    }

    /// <summary>
    /// Assertion methods which are stripped out in release mode.
    /// </summary>
    /// <remarks>To catch assertion failures in the debugger:
    /// Debug -> Exceptions -> Add
    /// Microsoft.VisualStudio.TestTools.UnitTesting.AssertFailedException
    /// </remarks>
    [DebuggerNonUserCode()]
    public static class AssertWhenDebugging
    {
        /// <summary>
        /// Checks if a condition is true; if not, an exception is thrown with an error message.
        /// </summary>
        /// <param name="condition">Condition that must be true</param>
        /// <param name="message">Message to be output by the exception</param>
        [Conditional("DEBUG")]
        public static void IsTrue(bool condition, string message)
        {
            if (!condition)
            {
                throw new AssertFailedException(message);
            }
        }

        /// <summary>
        /// Checks if a condition is true; if not, an exception  without error message is thrown.
        /// </summary>
        /// <param name="condition">Condition that must be true</param>
        [Conditional("DEBUG")]
        public static void IsTrue(bool condition)
        {
            if (!condition)
            {
                throw new AssertFailedException();
            }
        }

        /// <summary>
        /// Throws an exception if the collection has duplicate elements.
        /// </summary>
        /// <param name="collection">Any collection</param>
        [Conditional("DEBUG")]
        public static void Distinct(IEnumerable<int> collection)
        {
            HashSet<int> set = new HashSet<int>();
            foreach (var item in collection)
            {
                if (set.Contains(item)) throw new AssertFailedException($"duplicate array element: {item}");
                set.Add(item);
            }
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}
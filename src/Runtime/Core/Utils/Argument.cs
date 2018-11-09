// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Utilities
{
    using System;
    using System.Diagnostics;

    /// <summary>
    /// Useful routines for checking arguments of a function for correctness.
    /// </summary>
    public static class Argument
    {
        /// <summary>
        /// Throws an <see cref="ArgumentNullException"/> if the argument is <see langword="null"/>.
        /// </summary>
        /// <typeparam name="T">The type of the argument.</typeparam>
        /// <param name="argument">The argument.</param>
        /// <param name="argumentName">The name of the argument.</param>
        /// <remarks>
        /// There is no "class" restriction on T because in generic code it may be convinient to pass
        /// value types into <see cref="Argument.CheckIfNotNull{T}(T, string)"/>.
        /// Even if it is a no-op for them.
        /// </remarks>
        [DebuggerStepThrough]
        public static void CheckIfNotNull<T>(T argument, string argumentName)
        {
            if (argument == null)
            {
                throw new ArgumentNullException(argumentName);
            }
        }

        /// <summary>
        /// Throws an <see cref="ArgumentNullException"/> with the specified message if the argument is <see langword="null"/>.
        /// </summary>
        /// <typeparam name="T">The type of the argument.</typeparam>
        /// <param name="argument">The argument.</param>
        /// <param name="argumentName">The name of the argument.</param>
        /// <param name="message">The exception message.</param>
        /// <remarks>
        /// There is no "class" restriction on T because in generic code it may be convinient to pass
        /// value types into <see cref="Argument.CheckIfNotNull{T}(T, string, string)"/>.
        /// Even if it is a no-op for them.
        /// </remarks>
        [DebuggerStepThrough]
        public static void CheckIfNotNull<T>(T argument, string argumentName, string message)
        {
            if (argument == null)
            {
                throw new ArgumentNullException(argumentName, message);
            }
        }

        /// <summary>
        /// Throws an <see cref="ArgumentOutOfRangeException"/> if the <paramref name="inRangeCondition"/> is <see langword="false"/>.
        /// </summary>
        /// <param name="inRangeCondition"><see langword="true"/> if the argument is in its valid range, <see langword="false"/> otherwise.</param>
        /// <param name="argumentName">The name of the argument.</param>
        /// <param name="message">The exception message.</param>
        [DebuggerStepThrough]
        public static void CheckIfInRange(bool inRangeCondition, string argumentName, string message)
        {
            if (!inRangeCondition)
            {
                throw new ArgumentOutOfRangeException(argumentName, message);
            }
        }

        /// <summary>
        /// Throws an <see cref="ArgumentException"/> if the <paramref name="isValidCondition"/> is <see langword="false"/>.
        /// </summary>
        /// <param name="isValidCondition"><see langword="true"/> if the argument has a valid value, <see langword="false"/> otherwise.</param>
        /// <param name="argumentName">The name of the argument.</param>
        /// <param name="message">The exception message.</param>
        [DebuggerStepThrough]
        public static void CheckIfValid(bool isValidCondition, string argumentName, string message)
        {
            if (!isValidCondition)
            {
                throw new ArgumentException(message, argumentName);
            }
        }

        /// <summary>
        /// Throws an <see cref="ArgumentException"/> without specifying the argument
        /// if the <paramref name="isValidCondition"/> is <see langword="false"/>.
        /// </summary>
        /// <param name="isValidCondition"><see langword="true"/> if the exception should not be thrown, <see langword="false"/> otherwise.</param>
        /// <param name="message">The exception message.</param>
        [DebuggerStepThrough]
        public static void CheckIfValid(bool isValidCondition, string message)
        {
            if (!isValidCondition)
            {
                throw new ArgumentException(message);
            }
        }
    }
}
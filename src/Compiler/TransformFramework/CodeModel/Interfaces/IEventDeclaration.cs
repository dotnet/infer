// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Property declaration
    /// </summary>
    public interface IEventDeclaration : IMemberDeclaration, IEventReference
    {
        /// <summary>
        /// Reference to method which allows client code to trigger the event
        /// </summary>
        IMethodReference InvokeMethod { get; set; }

#if false
        /// <summary>
        /// Reference to method which allows client code to add a handler. If null
        /// then <see cref="RemoveMethod"/> must be null.
        /// </summary>
        IMethodReference AddMethod { get; set; }

        /// <summary>
        /// Reference to method which allows client code to remove a handler. If null
        /// then <see cref="AddMethod"/> must be null.
        /// </summary>
        IMethodReference RemoveMethod { get; set; }
#endif
    }
}
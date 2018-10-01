// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Event reference
    /// </summary>
    public interface IEventReference : IMemberReference
    {
        /// <summary>
        /// Delegate type for the event
        /// </summary>
        ITypeReference EventType { get; set; }

        /// <summary>
        /// Generic event reference
        /// </summary>
        IEventReference GenericEvent { get; set; }

        /// <summary>
        /// Resolve the reference to an event declaration
        /// </summary>
        /// <returns>Event declaration</returns>
        IEventDeclaration Resolve();
    }
}
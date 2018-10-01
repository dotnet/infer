// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Event reference expression
    /// </summary>
    public interface IEventReferenceExpression : IExpression
    {
        /// <summary>
        /// The event reference
        /// </summary>
        IEventReference Event { get; set; }

        /// <summary>
        /// The instance
        /// </summary>
        IExpression Target { get; set; }
    }
}
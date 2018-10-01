// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Variable reference
    /// </summary>
    public interface IVariableReference
    {
        /// <summary>
        /// Variable declaration
        /// </summary>
        IVariableDeclaration Variable { get; set; }

        /// <summary>
        /// Resolve
        /// </summary>
        /// <returns></returns>
        IVariableDeclaration Resolve();
    }
}
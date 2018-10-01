// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Required modifier
    /// </summary>
    public interface IRequiredModifier : IType
    {
        /// <summary>
        /// Element type
        /// </summary>
        IType ElementType { get; set; }

        /// <summary>
        /// Modifier type reference
        /// </summary>
        ITypeReference Modifier { get; set; }
    }
}
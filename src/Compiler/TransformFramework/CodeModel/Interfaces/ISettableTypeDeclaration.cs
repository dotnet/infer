// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Whether the type declaration is settable - for example in a type reference
    /// </summary>
    public interface ISettableTypeDeclaration
    {
        /// <summary>
        /// The type declaration
        /// </summary>
        ITypeDeclaration Declaration { set; }
    }
}
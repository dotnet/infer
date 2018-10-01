// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Namespace
    /// </summary>
    public interface INamespace
    {
        /// <summary>
        /// Name of the namespace
        /// </summary>
        string Name { get; set; }

        /// <summary>
        /// Type declarations
        /// </summary>
        List<ITypeDeclaration> Types { get; }
    }
}
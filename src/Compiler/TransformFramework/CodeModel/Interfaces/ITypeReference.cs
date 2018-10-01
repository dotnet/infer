// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Type reference
    /// </summary>
    public interface ITypeReference : IType, IGenericArgumentProvider
    {
        /// <summary>
        /// The generic type
        /// </summary>
        ITypeReference GenericType { get; set; }

        /// <summary>
        /// Name of the reference
        /// </summary>
        string Name { get; set; }

        /// <summary>
        /// Namespace of the reference
        /// </summary>
        string Namespace { get; set; }

        /// <summary>
        /// Owner of the reference
        /// </summary>
        object Owner { get; set; }

        /// <summary>
        /// Whether this is a value type
        /// </summary>
        bool ValueType { get; set; }

        /// <summary>
        /// Resolve to a type declaration
        /// </summary>
        /// <returns></returns>
        ITypeDeclaration Resolve();
    }
}
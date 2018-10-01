// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Property declaration
    /// </summary>
    public interface IPropertyDeclaration : IMemberDeclaration, IPropertyReference
    {
        /// <summary>
        /// Get method reference
        /// </summary>
        IMethodReference GetMethod { get; set; }

        /// <summary>
        /// Initializer expression
        /// </summary>
        IExpression Initializer { get; set; }

        /// <summary>
        /// Set method reference
        /// </summary>
        IMethodReference SetMethod { get; set; }
    }
}
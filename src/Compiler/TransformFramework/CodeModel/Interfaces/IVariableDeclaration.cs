// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Variable declaration
    /// </summary>
    public interface IVariableDeclaration : IVariableReference
    {
        /// <summary>
        /// 
        /// </summary>
        int Identifier { get; set; }

        /// <summary>
        /// Variable name
        /// </summary>
        string Name { get; set; }

        /// <summary>
        /// Variable type
        /// </summary>
        IType VariableType { get; set; }
    }
}
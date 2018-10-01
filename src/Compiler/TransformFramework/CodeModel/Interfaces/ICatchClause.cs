// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Catch clause
    /// </summary>
    public interface ICatchClause
    {
        /// <summary>
        /// The body of the clause
        /// </summary>
        IBlockStatement Body { get; set; }

        /// <summary>
        /// Condition expression for catch clause
        /// </summary>
        IExpression Condition { get; set; }

        /// <summary>
        /// Variable declaration for catch clause
        /// </summary>
        IVariableDeclaration Variable { get; set; }
    }
}
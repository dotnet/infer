// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// For each statement
    /// </summary>
    public interface IForEachStatement : IStatement
    {
        /// <summary>
        /// The statements in the body of the for each statement
        /// </summary>
        IBlockStatement Body { get; set; }

        /// <summary>
        /// Expression in the foreach statement
        /// </summary>
        IExpression Expression { get; set; }

        /// <summary>
        /// Variable declaration in the for each statement
        /// </summary>
        IVariableDeclaration Variable { get; set; }
    }
}
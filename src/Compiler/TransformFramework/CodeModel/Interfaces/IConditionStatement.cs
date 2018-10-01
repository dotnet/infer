// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Condition statement
    /// </summary>
    public interface IConditionStatement : IStatement
    {
        /// <summary>
        /// If condition
        /// </summary>
        IExpression Condition { get; set; }

        /// <summary>
        /// Else block of statements
        /// </summary>
        IBlockStatement Else { get; set; }

        /// <summary>
        /// Then block of statements
        /// </summary>
        IBlockStatement Then { get; set; }
    }
}
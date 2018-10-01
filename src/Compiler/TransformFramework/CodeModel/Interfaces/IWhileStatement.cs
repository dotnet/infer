// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// While statement
    /// </summary>
    public interface IWhileStatement : IStatement
    {
        /// <summary>
        /// Body of while statement
        /// </summary>
        IBlockStatement Body { get; set; }

        /// <summary>
        /// Condition for while statement
        /// </summary>
        IExpression Condition { get; set; }
    }
}
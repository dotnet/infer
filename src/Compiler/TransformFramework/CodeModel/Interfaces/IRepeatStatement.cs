// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// For statement
    /// </summary>
    public interface IRepeatStatement : IStatement
    {
        /// <summary>
        /// The body of the if statement
        /// </summary>
        IBlockStatement Body { get; set; }

        /// <summary>
        /// The count of the repeat statement
        /// </summary>
        IExpression Count { get; set; }
    }
}
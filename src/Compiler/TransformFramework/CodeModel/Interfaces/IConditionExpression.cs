// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Condition expression
    /// </summary>
    public interface IConditionExpression : IExpression
    {
        /// <summary>
        /// If condition
        /// </summary>
        IExpression Condition { get; set; }

        /// <summary>
        /// Else expression
        /// </summary>
        IExpression Else { get; set; }

        /// <summary>
        /// Then expression
        /// </summary>
        IExpression Then { get; set; }
    }
}
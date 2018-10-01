// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Assignment expression
    /// </summary>
    public interface IAssignExpression : IExpression
    {
        /// <summary>
        /// Expression being assigned
        /// </summary>
        IExpression Expression { get; set; }

        /// <summary>
        /// Target of the assignment
        /// </summary>
        IExpression Target { get; set; }
    }
}
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Interface for an 'is' expression
    /// </summary>
    public interface ICanCastExpression : IExpression
    {
        /// <summary>
        /// Target type
        /// </summary>
        IType TargetType { get; set; }

        /// <summary>
        /// The expression
        /// </summary>
        IExpression Expression { get; set; }
    }
}
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Cast expression
    /// </summary>
    public interface ICastExpression : IExpression
    {
        /// <summary>
        /// The expression
        /// </summary>
        IExpression Expression { get; set; }

        /// <summary>
        /// The target type for the cast
        /// </summary>
        IType TargetType { get; set; }
    }
}
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// A literal expression
    /// </summary>
    public interface ILiteralExpression : IExpression
    {
        /// <summary>
        /// The value of the literal expression
        /// </summary>
        object Value { get; set; }
    }
}
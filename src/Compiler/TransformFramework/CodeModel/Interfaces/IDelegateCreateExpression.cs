// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Delegate create expressions
    /// </summary>
    public interface IDelegateCreateExpression : IExpression
    {
        /// <summary>
        /// Delegate type
        /// </summary>
        ITypeReference DelegateType { get; set; }

        /// <summary>
        /// Delegate method reference
        /// </summary>
        IMethodReference Method { get; set; }

        /// <summary>
        /// Expression for creating delegate
        /// </summary>
        IExpression Target { get; set; }
    }
}
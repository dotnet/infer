// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Interface for delegate invoke expression
    /// </summary>
    public interface IDelegateInvokeExpression : IExpression
    {
        /// <summary>
        /// Arguments for delegate invoke expression
        /// </summary>
        IList<IExpression> Arguments { get; }

        /// <summary>
        /// Target for delegate invoke expression
        /// </summary>
        IExpression Target { get; set; }
    }
}
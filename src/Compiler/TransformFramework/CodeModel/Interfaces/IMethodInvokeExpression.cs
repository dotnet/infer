// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Method invoke expression
    /// </summary>
    public interface IMethodInvokeExpression : IExpression
    {
        /// <summary>
        /// The arguments
        /// </summary>
        IList<IExpression> Arguments { get; }

        /// <summary>
        /// The method
        /// </summary>
        IMethodReferenceExpression Method { get; set; }
    }
}
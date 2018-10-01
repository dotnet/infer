// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Method reference expression
    /// </summary>
    public interface IMethodReferenceExpression : IExpression
    {
        /// <summary>
        /// Method reference
        /// </summary>
        IMethodReference Method { get; set; }

        /// <summary>
        /// Instance expression
        /// </summary>
        IExpression Target { get; set; }
    }
}
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler
{
    /// <summary>
    /// Expression transform interface
    /// </summary>
    public interface IExpressionTransform
    {
        /// <summary>
        /// Takes an expression and transforms it
        /// </summary>
        /// <param name="expr">Exression</param>
        IExpression ConvertExpression(IExpression expr);

        /// <summary>
        /// Context
        /// </summary>
        ICodeTransformContext Context { get; set; }
    }
}

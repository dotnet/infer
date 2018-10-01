// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// All expressions derive from this
    /// </summary>
    public interface IExpression
    {
        /// <summary>
        /// Returns the static type of this expression.
        /// </summary>
        /// <returns>The type of the expression</returns>
        Type GetExpressionType();
    }
}
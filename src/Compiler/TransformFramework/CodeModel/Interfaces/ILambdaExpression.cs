// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Interface for a lambda expression
    /// </summary>
    public interface ILambdaExpression : IExpression
    {
        /// <summary>
        /// The body of the expression
        /// </summary>
        IExpression Body { get; set; }

        /// <summary>
        /// The list of parameters
        /// </summary>
        IList<IVariableDeclaration> Parameters { get; }
    }
}
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Method return statement
    /// </summary>
    public interface IMethodReturnStatement : IStatement
    {
        /// <summary>
        /// Expression to return
        /// </summary>
        IExpression Expression { get; set; }
    }
}
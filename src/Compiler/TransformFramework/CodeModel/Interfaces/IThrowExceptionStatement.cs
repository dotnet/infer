// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Interface for throw exception statement
    /// </summary>
    public interface IThrowExceptionStatement : IStatement
    {
        /// <summary>
        /// The expression thrown
        /// </summary>
        IExpression Expression { get; set; }
    }
}
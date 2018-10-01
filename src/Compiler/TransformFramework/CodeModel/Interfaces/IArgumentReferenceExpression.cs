// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Argument reference expression
    /// </summary>
    public interface IArgumentReferenceExpression : IExpression
    {
        /// <summary>
        /// The argument
        /// </summary>
        IParameterReference Parameter { get; set; }
    }
}
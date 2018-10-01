// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Address out expression
    /// </summary>
    public interface IAddressOutExpression : IExpression
    {
        /// <summary>
        /// The source expression
        /// </summary>
        IExpression Expression { get; set; }
    }
}
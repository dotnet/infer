// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// A member initializer expression
    /// </summary>
    public interface IMemberInitializerExpression : IExpression
    {
        /// <summary>
        /// The member reference
        /// </summary>
        IMemberReference Member { get; set; }

        /// <summary>
        /// The initializer expression
        /// </summary>
        IExpression Value { get; set; }
    }
}
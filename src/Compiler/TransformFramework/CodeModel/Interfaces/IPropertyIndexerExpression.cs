// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Property indexer expression
    /// </summary>
    public interface IPropertyIndexerExpression : IExpression
    {
        /// <summary>
        /// Index expression collection
        /// </summary>
        IList<IExpression> Indices { get; }

        /// <summary>
        /// Property reference expression
        /// </summary>
        IPropertyReferenceExpression Target { get; set; }
    }
}
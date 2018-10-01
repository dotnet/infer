// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Comment statement
    /// </summary>
    public interface ICommentStatement : IStatement
    {
        /// <summary>
        /// Text for comment statement
        /// </summary>
        IComment Comment { get; set; }
    }
}
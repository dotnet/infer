// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Array type
    /// </summary>
    public interface IArrayType : IType
    {
        /// <summary>
        /// Rank of the array
        /// </summary>
        int Rank { get; }

        /// <summary>
        /// Element type
        /// </summary>
        IType ElementType { get; set; }
    }
}
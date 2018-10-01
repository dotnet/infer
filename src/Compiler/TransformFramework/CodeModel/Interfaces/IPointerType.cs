// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Pointer type
    /// </summary>
    public interface IPointerType : IType
    {
        /// <summary>
        /// Element type for pointer type
        /// </summary>
        IType ElementType { get; set; }
    }
}
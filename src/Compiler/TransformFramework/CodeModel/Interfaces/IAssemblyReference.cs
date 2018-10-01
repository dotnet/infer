// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Assembly reference
    /// </summary>
    public interface IAssemblyReference : IComparable
    {
        /// <summary>
        /// Name of the assembly
        /// </summary>
        string Name { get; set; }
    }
}
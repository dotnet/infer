// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Member reference
    /// </summary>
    public interface IMemberReference : IComparable
    {
        /// <summary>
        /// Declaring type for the member
        /// </summary>
        IType DeclaringType { get; set; }

        /// <summary>
        /// Name of the member
        /// </summary>
        string Name { get; set; }
    }
}
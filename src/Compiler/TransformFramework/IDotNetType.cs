// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Probabilistic.Compiler
{
    /// <summary>
    /// Interface that exposes the dotNET type of the code model type
    /// </summary>
    public interface IDotNetType
    {
        /// <summary>
        /// The dotNET type
        /// </summary>
        Type DotNetType { get; set; }
    }
}
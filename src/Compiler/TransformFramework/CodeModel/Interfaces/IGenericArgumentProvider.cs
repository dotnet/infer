// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Generic Argument Provider
    /// </summary>
    public interface IGenericArgumentProvider : IComparable
    {
        /// <summary>
        /// The provided type collection
        /// </summary>
        IList<IType> GenericArguments { get; }
    }
}
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler
{
    /// <summary>
    /// Declaration provider
    /// </summary>
    public interface IDeclarationProvider
    {
        /// <summary>
        /// Get type declaration from type
        /// </summary>
        /// <param name="t">The dotNET type</param>
        /// <param name="translate"></param>
        /// <returns></returns>
        ITypeDeclaration GetTypeDeclaration(Type t, bool translate);
    }
}
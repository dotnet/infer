// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Reflection;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Method signature
    /// </summary>
    public interface IMethodSignature
    {
        /// <summary>
        /// Parameter declarations
        /// </summary>
        IList<IParameterDeclaration> Parameters { get; }

        /// <summary>
        /// Return type
        /// </summary>
        IMethodReturnType ReturnType { get; set; }

        /// <summary>
        /// dotNet method info
        /// </summary>
        MethodBase MethodInfo { get; set; }
    }
}
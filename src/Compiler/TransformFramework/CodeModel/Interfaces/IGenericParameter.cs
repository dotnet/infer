// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// A generic parameter
    /// </summary>
    public interface IGenericParameter : IGenericArgument, ICustomAttributeProvider
    {
        /// <summary>
        /// A set of type constraints for the generic parameter
        /// </summary>
        IList<IType> Constraints { get; }

        /// <summary>
        /// The name of the generic parameter
        /// </summary>
        string Name { get; set; }
    }
}
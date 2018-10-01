// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Field reference
    /// </summary>
    public interface IFieldReference : IMemberReference
    {
        /// <summary>
        /// Field type
        /// </summary>
        IType FieldType { get; set; }

        /// <summary>
        /// Generic field reference
        /// </summary>
        IFieldReference GenericField { get; set; }

        /// <summary>
        /// Resolve to a field declaration
        /// </summary>
        /// <returns></returns>
        IFieldDeclaration Resolve();
    }
}
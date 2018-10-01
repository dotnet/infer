// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Method visibility
    /// </summary>
    public enum MethodVisibility
    {
        /// <summary>
        /// 
        /// </summary>
        PrivateScope = 0,

        /// <summary>
        /// Private
        /// </summary>
        Private = 1,

        /// <summary>
        /// Private visibility
        /// </summary>
        FamilyAndAssembly = 2,

        /// <summary>
        /// Visible within the assembly
        /// </summary>
        Assembly = 3,

        /// <summary>
        /// 
        /// </summary>
        Family = 4,

        /// <summary>
        /// 
        /// </summary>
        FamilyOrAssembly = 5,

        /// <summary>
        /// Public visibility
        /// </summary>
        Public = 6,
    }

    /// <summary>
    /// Method declaration
    /// </summary>
    public interface IMethodDeclaration : IMethodReference, IMemberDeclaration
    {
        /// <summary>
        /// Abstract flag
        /// </summary>
        bool Abstract { get; set; }

        /// <summary>
        /// Method body
        /// </summary>
        IBlockStatement Body { get; set; }

        /// <summary>
        /// Final flag
        /// </summary>
        bool Final { get; set; }

        /// <summary>
        /// For a virtual method, indicates whether the method is 'new' versus 'override'
        /// </summary>
        bool NewSlot
        {
            get;
            set;
        }

        /// <summary>
        /// Override flag
        /// </summary>
        bool Overrides { get; set; }

        /// <summary>
        /// Static flag
        /// </summary>
        bool Static { get; set; }

        /// <summary>
        /// Virtual flag
        /// </summary>
        bool Virtual { get; set; }

        /// <summary>
        /// Visibility
        /// </summary>
        MethodVisibility Visibility { get; set; }

    }
}
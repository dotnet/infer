// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.


using System;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Represents a reference to an assembly
    /// </summary>
    internal class XAssemblyReference : IAssemblyReference
    {
        #region Fields

        private string name;

        #endregion

        #region Constructor

        public XAssemblyReference()
        {
            this.name = string.Empty;
        }

        #endregion

        #region IAssemblyReference Members

        /// <summary>
        /// Assembly name
        /// </summary>
        public string Name
        {
            get { return this.name; }
            set { this.name = value; }
        }

        #endregion

        #region IComparable Members

        public int CompareTo(object obj)
        {
            IAssemblyReference assemblyReference = obj as IAssemblyReference;
            if (assemblyReference == null)
                throw new NotSupportedException();
            return String.Compare(this.Name, assemblyReference.Name, StringComparison.InvariantCulture);
        }

        #endregion
    }
}
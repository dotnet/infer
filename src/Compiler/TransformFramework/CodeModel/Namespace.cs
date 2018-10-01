// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Namespace
    /// </summary>
    internal class XNamespace : INamespace
    {
        #region Fields

        private string name;
        private List<ITypeDeclaration> types;

        #endregion

        #region INamespace Members

        /// <summary>
        /// The namespace name
        /// </summary>
        public string Name
        {
            get { return this.name; }
            set { this.name = value; }
        }

        /// <summary>
        /// The type collection
        /// </summary>
        public List<ITypeDeclaration> Types
        {
            get
            {
                if (types == null)
                    types = new List<ITypeDeclaration>();
                return types;
            }
        }

        #endregion

        #region Object Overrides

        public override string ToString()
        {
            return Name;
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            INamespace namesp = obj as INamespace;
            if (namesp == null ||
                (!Name.Equals(namesp.Name)) ||
                Types.Count != namesp.Types.Count)
                return false;

            for (int i = 0; i < Types.Count; i++)
                if (!(Types[i].Equals(namesp.Types[i])))
                    return false;

            return true;
        }

        public override int GetHashCode()
        {
            return this.Name.GetHashCode();
        }

        #endregion Object Overrides
    }
}
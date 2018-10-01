// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Type reference
    /// </summary>
    internal class XTypeReference : ITypeReference, ISettableTypeDeclaration
    {
        #region Fields

        private IList<IType> genericArguments;
        private object owner;
        private ITypeDeclaration typeDeclaration = null;
        private Type dotNetType;

        #endregion

        #region ITypeReference Members

        public ITypeReference GenericType
        {
            get { return null; }
            set { throw new NotSupportedException("Use TypeInstanceReference type to represent a specialization of a generic type"); }
        }

        public string Name
        {
            get
            {
                if (this.dotNetType != null)
                {
                    string nm = this.dotNetType.Name;
                    // Trim everything from the first occurrence of '`' onwards
                    int idx = nm.IndexOf('`');
                    if (idx >= 0)
                        nm = nm.Substring(0, idx);

                    return nm;
                }
                else
                    return "";
            }
            set { throw new NotSupportedException("Cannot set name for a type reference"); }
        }

        public string Namespace
        {
            get
            {
                if (this.dotNetType != null)
                {
                    return this.dotNetType.Namespace;
                }
                else
                    return "";
            }
            set { throw new NotSupportedException("Cannot set name space for a type reference"); }
        }

        public object Owner
        {
            get { return this.owner; }
            set { this.owner = value; }
        }

        public ITypeDeclaration Resolve()
        {
            return this.typeDeclaration;
        }

        public bool ValueType
        {
            get
            {
                if (this.dotNetType != null)
                {
                    return this.dotNetType.IsValueType;
                }
                else
                    return false;
            }
            set { throw new NotSupportedException("Cannot set value type for a type reference"); }
        }

        #endregion

        #region IComparable Members

        public int CompareTo(object obj)
        {
            return CompareItems.CompareTypeReferences(this, obj as ITypeReference);
        }

        #endregion

        #region IGenericArgumentProvider Members

        /// <summary>
        /// Generic arguments
        /// </summary>
        public IList<IType> GenericArguments
        {
            get
            {
                if (this.genericArguments == null)
                    this.genericArguments = new List<IType>();
                return this.genericArguments;
            }
        }

        #endregion

        #region IDotNetType Members

        /// <summary>
        /// Dot net type
        /// </summary>
        public Type DotNetType
        {
            get { return this.dotNetType; }
            set { this.dotNetType = value; }
        }

        #endregion

        #region Object Overrides

        public override string ToString()
        {
            ILanguageWriter writer = new CSharpWriter() as ILanguageWriter;
            return writer.TypeSource(this);
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IDotNetType type = obj as IDotNetType;
            if (type == null)
                return false;

            return (this.DotNetType == type.DotNetType);
        }

        public override int GetHashCode()
        {
            if (DotNetType == null) return 0;
            else return DotNetType.GetHashCode();
        }

        #endregion Object Overrides

        #region ISettableTypeDeclaration Members

        public ITypeDeclaration Declaration
        {
            set { typeDeclaration = value; }
        }

        #endregion
    }
}
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// This class models a reference to a specialization of a generic type
    /// </summary>
    internal class XTypeInstanceReference : ITypeReference, ISettableTypeDeclaration
    {
        #region Fields

        private ITypeReference genericType;
        private IList<IType> genericArguments;
        private ITypeDeclaration typeDeclaration = null;
        private Type dotNetType;

        #endregion

        #region Statics

        /// <summary>
        /// Static method to see if two TypeInstanceReferences are equal
        /// </summary>
        /// <param name="itr1">First TypeInstanceReference</param>
        /// <param name="itr2">Second TypInstanceReference</param>
        /// <returns></returns>
        internal static bool AreEqual(ITypeReference itr1, ITypeReference itr2)
        {
            if (itr1.DotNetType != null)
                return itr1.DotNetType == itr2.DotNetType;
            else
                return false;
        }

        /// <summary>
        /// Static method to compare two TypeInstanceReference instances
        /// </summary>
        /// <param name="itr1">First TypeInstanceReference</param>
        /// <param name="itr2">Second TypInstanceReference</param>
        /// <returns>-1, 0 or 1</returns>
        internal static int Compare(ITypeReference itr1, ITypeReference itr2)
        {
            // We just need to compare the generic types that these specializations
            // derive from, and, if these are equal, compare the generic arguments
            int ret = itr1.GenericType.CompareTo(itr2.GenericType);

            if (ret != 0)
                return ret;

            if ((ret = itr1.GenericArguments.Count.CompareTo(itr2.GenericArguments.Count)) != 0)
                return ret;

            for (int i = 0; i < itr1.GenericArguments.Count; i++)
            {
                if ((ret = itr1.GenericArguments[i].CompareTo(itr2.GenericArguments[i])) != 0)
                    return ret;
            }
            return 0;
        }

        #endregion

        #region ITypeReference Members

        /// <summary>
        /// Generic type of which this is a specialization (if any)
        /// </summary>
        public ITypeReference GenericType
        {
            get { return this.genericType; }
            set
            {
                if (object.ReferenceEquals(value, this))
                {
                    throw new ArgumentException("genericType cannot be equal to this");
                }
                if (object.ReferenceEquals(value, null))
                {
                    throw new NullReferenceException("genericType cannot be null");
                }
                this.genericType = value;
            }
        }

        /// <summary>
        /// Name of type specialization
        /// </summary>
        public string Name
        {
            get
            {
                if (genericType == null) return null;
                return this.genericType.Name;
            }
            set { throw new Exception("The method or operation is not implemented."); }
        }

        /// <summary>
        /// Namespace
        /// </summary>
        public string Namespace
        {
            get { return this.genericType.Namespace; }
            set { throw new Exception("The method or operation is not implemented."); }
        }

        /// <summary>
        /// Owner
        /// </summary>
        public object Owner
        {
            get { return this.genericType.Owner; }
            set { throw new Exception("The method or operation is not implemented."); }
        }

        /// <summary>
        /// Resolve this specialization
        /// </summary>
        /// <returns></returns>
        public ITypeDeclaration Resolve()
        {
            return this.typeDeclaration;
        }

        /// <summary>
        /// WHether this is a value type
        /// </summary>
        public bool ValueType
        {
            get { return this.genericType.ValueType; }
            set { throw new Exception("The method or operation is not implemented."); }
        }

        #endregion

        #region IComparable Members

        public int CompareTo(object obj)
        {
            ITypeReference itr = obj as ITypeReference;
            if (itr == null)
                return -1;
            return Compare(this, itr);
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

            ITypeReference reference = obj as ITypeReference;
            IMethodReference owner = obj as IMethodReference;
            IGenericArgument argument = obj as IGenericArgument;
            if (reference == null)
            {
                // If this is not a type reference, it might be an argument
                // of a generic function. 
                if (argument != null && (owner == null || owner.GenericMethod != null))
                {
                    ITypeReference resolvedRef = argument.Resolve() as ITypeReference;
                    if (resolvedRef == null)
                        return false;
                    else
                        return AreEqual(this, resolvedRef);
                }
                else
                    return false;
            }
            else
            {
                return AreEqual(this, reference);
            }
        }

        public override int GetHashCode()
        {
            return this.Name.GetHashCode();
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
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// A field reference
    /// </summary>
    internal class XFieldReference : IFieldReference
    {
        #region Fields

        private IFieldReference genericField;
        private IType declaringType;
        private IType fieldType;
        private string name;
        private WeakReference weakRef;

        #endregion

        #region Constructor

        public XFieldReference()
        {
        }

        #endregion

        #region IFieldReference Members

        /// <summary>
        /// The type of the field
        /// </summary>
        public IType FieldType
        {
            get { return this.fieldType; }
            set { this.fieldType = value; }
        }

        /// <summary>
        /// A reference to the generic field for which this field is a specialization (if any)
        /// </summary>
        public IFieldReference GenericField
        {
            get { return this.genericField; }
            set { this.genericField = value; }
        }

        /// <summary>
        /// Resolve to a field declaration
        /// </summary>
        /// <returns></returns>
        public IFieldDeclaration Resolve()
        {
            if (this.weakRef == null || !this.weakRef.IsAlive)
            {
                ITypeDeclaration itd = (this.DeclaringType as ITypeReference).Resolve();
                if (itd == null) return null;
                string name = this.name;
                foreach (IFieldDeclaration ifd in itd.Fields)
                {
                    if (name == ifd.Name && this.FieldType.Equals(ifd.FieldType))
                    {
                        this.weakRef = new WeakReference(ifd);
                        return (IFieldDeclaration) this.weakRef.Target;
                    }
                }
                return null;
            }
            else
                return (IFieldDeclaration) this.weakRef.Target;
        }

        #endregion

        #region IMemberReference Members

        public IType DeclaringType
        {
            get { return this.declaringType; }
            set { this.declaringType = value; }
        }

        public string Name
        {
            get { return this.name; }
            set { this.name = value; }
        }

        #endregion

        #region IComparable Members

        public int CompareTo(object obj)
        {
            IFieldReference reference = obj as IFieldReference;
            if (reference == null)
            {
                throw new NotSupportedException();
            }
            int ret = this.DeclaringType.CompareTo(reference.DeclaringType);
            if (0 == ret)
                ret = String.Compare(this.Name, reference.Name, StringComparison.InvariantCulture);
            if (0 == ret)
                ret = this.FieldType.CompareTo(reference.FieldType);
            return ret;
        }

        #endregion

        #region Object Overrides

        public override string ToString()
        {
            return this.Name;
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IFieldReference reference = obj as IFieldReference;
            if (reference == null)
                return false;

            if (this.DeclaringType.Equals(reference.DeclaringType) &&
                this.Name.Equals(reference.Name) &&
                this.FieldType.Equals(reference.FieldType))
                return true;
            else
                return false;
        }

        public override int GetHashCode()
        {
            return this.Name.GetHashCode();
        }

        #endregion Object Overrides
    }
}
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Field declaraion
    /// </summary>
    internal class XFieldDeclaration : IFieldDeclaration
    {
        #region Fields

        private List<ICustomAttribute> attributes;
        private IExpression initializer;
        private IFieldReference genericField;
        private IType declaringType;
        private IType fieldType;
        private string documentation;
        private string name;
        private bool literal;
        private bool sttc;
        private bool readOnly;
        private FieldVisibility visibility;

        #endregion

        #region Constructor

        public XFieldDeclaration()
        {
        }

        #endregion

        #region IFieldDeclaration Members

        /// <summary>
        /// Initializer for the field
        /// </summary>
        public IExpression Initializer
        {
            get { return this.initializer; }
            set { this.initializer = value; }
        }

        /// <summary>
        /// Whether the field is a literal
        /// </summary>
        public bool Literal
        {
            get { return this.literal; }
            set { this.literal = value; }
        }

        /// <summary>
        /// Whether the field is read only
        /// </summary>
        public bool ReadOnly
        {
            get { return this.readOnly; }
            set { this.readOnly = value; }
        }

        /// <summary>
        /// Whether this is a static field
        /// </summary>
        public bool Static
        {
            get { return this.sttc; }
            set { this.sttc = value; }
        }

        /// <summary>
        /// Visibility of the field
        /// </summary>
        public FieldVisibility Visibility
        {
            get { return (this.visibility & (FieldVisibility.Public | FieldVisibility.Private)); }
            set { this.visibility = value; }
        }

        #endregion

        #region IFieldReference Members

        /// <summary>
        /// Type of the field
        /// </summary>
        public IType FieldType
        {
            get { return this.fieldType; }
            set { this.fieldType = value; }
        }

        /// <summary>
        /// The generic field from which this field is derived (if any)
        /// </summary>
        public IFieldReference GenericField
        {
            get { return this.genericField; }
            set { this.genericField = value; }
        }

        /// <summary>
        /// Resolve - just returns this as it is already a declaration
        /// </summary>
        /// <returns></returns>
        public IFieldDeclaration Resolve()
        {
            return this;
        }

        #endregion

        #region IMemberReference Members

        /// <summary>
        /// Declaring type of this field
        /// </summary>
        public IType DeclaringType
        {
            get { return this.declaringType; }
            set { this.declaringType = value; }
        }

        /// <summary>
        /// The name of this field
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
            IFieldDeclaration reference = obj as IFieldDeclaration;
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

        #region ICustomAttributeProvider Members

        /// <summary>
        /// The custome attributes attached to this field
        /// </summary>
        public List<ICustomAttribute> Attributes
        {
            get
            {
                if (this.attributes == null)
                    this.attributes = new List<ICustomAttribute>();
                return this.attributes;
            }
        }

        #endregion

        #region IDocumentationProvider Members

        /// <summary>
        /// The documentation for this field
        /// </summary>
        public string Documentation
        {
            get { return this.documentation; }
            set { this.documentation = value; }
        }

        #endregion

        #region Object Overrides

        public override string ToString()
        {
            ILanguageWriter writer = new CSharpWriter() as ILanguageWriter;
            return writer.FieldDeclarationSource(this);
            //return this.Name;
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IFieldDeclaration reference = obj as IFieldDeclaration;
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
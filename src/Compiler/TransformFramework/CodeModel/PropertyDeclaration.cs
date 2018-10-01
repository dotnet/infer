// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Property declaration
    /// </summary>
    internal class XPropertyDeclaration : IPropertyDeclaration
    {
        #region Fields

        private List<ICustomAttribute> attributes;
        private IExpression initializer;
        private IMethodReference getMethod;
        private IMethodReference setMethod;
        private IList<IParameterDeclaration> parameters;
        private IPropertyReference genericProperty;
        private IType propertyType;
        private IType declaringType;
        private string name;
        private string documentation;

        #endregion

        #region IPropertyDeclaration Members

        public IMethodReference GetMethod
        {
            get { return this.getMethod; }
            set { this.getMethod = value; }
        }

        public IExpression Initializer
        {
            get { return this.initializer; }
            set { this.initializer = value; }
        }

        public IMethodReference SetMethod
        {
            get { return this.setMethod; }
            set { this.setMethod = value; }
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
            IPropertyReference reference = obj as IPropertyReference;

            if (reference == null)
                throw new NotSupportedException();

            return CompareItems.ComparePropertyReferences(this, reference);
        }

        #endregion

        #region ICustomAttributeProvider Members

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

        public string Documentation
        {
            get { return this.documentation; }
            set { this.documentation = value; }
        }

        #endregion

        #region IPropertyReference Members

        public IPropertyReference GenericProperty
        {
            get { return this.genericProperty; }
            set { this.genericProperty = value; }
        }

        public IList<IParameterDeclaration> Parameters
        {
            get
            {
                if (this.parameters == null)
                    this.parameters = new List<IParameterDeclaration>();
                return this.parameters;
            }
        }

        public IType PropertyType
        {
            get { return this.propertyType; }
            set { this.propertyType = value; }
        }

        public IPropertyDeclaration Resolve()
        {
            return this;
        }

        #endregion

        #region Object Overrides

        /// <summary>
        /// ToString override
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            ILanguageWriter writer = new CSharpWriter() as ILanguageWriter;
            return writer.PropertyDeclarationSource(this);
        }

        /// <summary>
        /// Determines if this method reference is equal to another (representing the method declaration)
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IPropertyReference that = obj as IPropertyReference;
            if (that == null)
                return false;

            return CompareItems.PropertyReferencesAreEqual(this, that);
        }

        /// <summary>
        /// GetHashCode override
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            return this.Name.GetHashCode();
        }

        #endregion Object Overrides
    }
}
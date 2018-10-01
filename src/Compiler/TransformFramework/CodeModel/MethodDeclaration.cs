// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// This class models a method declaration. It is not the specialization of a generic type
    /// but may be a generic method definition
    /// </summary>
    internal class XMethodDeclaration : IMethodDeclaration
    {
        #region Fields

        private List<ICustomAttribute> attributes;
        private IMethodReturnType returnType = new XMethodReturnType();
        private IList<IParameterDeclaration> parameters;
        private IType declaringType;
        private IList<IType> genericArguments;
        private IBlockStatement body;
        private string name;
        private string documentation;
        private bool abstrct;
        private bool final;
        private bool sttc;
        private bool vrtl;
        private MethodVisibility visibility;

        #endregion

        #region IMethodDeclaration Members

        /// <summary>
        /// Wther the method is abstract
        /// </summary>
        public bool Abstract
        {
            get { return this.abstrct; }
            set { this.abstrct = value; }
        }

        /// <summary>
        /// The body of the method
        /// </summary>
        public IBlockStatement Body
        {
            get { return this.body; }
            set { this.body = value; }
        }

        /// <summary>
        /// Whether the method is final - i.e. it is an override which cannot be overridden itself
        /// </summary>
        public bool Final
        {
            get { return this.final; }
            set { this.final = value; }
        }

        /// <summary>
        /// For a virtual method, indicates whether the method is 'new' versus 'override'
        /// </summary>
        public bool NewSlot
        {
            get;
            set;
        }

        /// <summary>
        /// Override flag
        /// </summary>
        public bool Overrides
        {
            get;
            set;
        }

        /// <summary>
        /// Whether this is a static method
        /// </summary>
        public bool Static
        {
            get { return this.sttc; }
            set { this.sttc = value; }
        }

        /// <summary>
        /// Whether this method can be overridden
        /// </summary>
        public bool Virtual
        {
            get { return this.vrtl; }
            set { this.vrtl = value; }
        }

        /// <summary>
        /// Visibility of the method
        /// </summary>
        public MethodVisibility Visibility
        {
            get { return this.visibility & (MethodVisibility.Public | MethodVisibility.Private); }
            set { this.visibility = value & (MethodVisibility.Public | MethodVisibility.Private); }
        }

        #endregion

        #region IMethodReference Members

        /// <summary>
        /// Always returns null and cannot be set
        /// </summary>
        public IMethodReference GenericMethod
        {
            get
            {
                // MethodDeclaration is not specialization of a generic method declaration
                // though it may be a generic method declaration itself
                return null;
            }
            set { throw new Exception("The method or operation is not implemented."); }
        }

        /// <summary>
        /// Always returns this
        /// </summary>
        /// <returns></returns>
        public IMethodDeclaration Resolve()
        {
            // We're at the end of the line
            return this;
        }

        #endregion

        #region IMemberReference Members

        /// <summary>
        /// The declaring type for this method
        /// </summary>
        public IType DeclaringType
        {
            get { return this.declaringType; }
            set { this.declaringType = value; }
        }

        /// <summary>
        /// The name of this method
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
            IMethodReference imr = obj as IMethodReference;
            if (imr == null)
                return -1;
            return CompareItems.CompareMethodReferences(this, imr);
        }

        #endregion

        #region IMethodSignature Members

        /// <summary>
        /// Method Info
        /// </summary>
        public MethodBase MethodInfo { get; set; }

        /// <summary>
        /// Parameters declarations
        /// </summary>
        public IList<IParameterDeclaration> Parameters
        {
            get
            {
                if (this.parameters == null)
                    this.parameters = new List<IParameterDeclaration>();
                return this.parameters;
            }
        }

        /// <summary>
        /// Return type of the method
        /// </summary>
        public IMethodReturnType ReturnType
        {
            get { return this.returnType; }
            set { this.returnType = value; }
        }

        #endregion

        #region IGenericArgumentProvider Members

        /// <summary>
        /// Generic types, if this is derived from a generic method
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

        #region ICustomAttributeProvider Members

        /// <summary>
        /// The custom attributes attached to this method
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
        /// The documentation for this method
        /// </summary>
        public string Documentation
        {
            get { return this.documentation; }
            set { this.documentation = value; }
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
            return writer.MethodDeclarationSource(this);
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

            IMethodReference that = obj as IMethodReference;
            if (that == null)
                return false;

            return CompareItems.MethodReferencesAreEqual(this, that);
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
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
    /// This class models a method declaration which is a specialization
    /// of a generic model declaration as given by the GenericMethod property
    /// </summary>
    internal class XMethodInstanceDeclaration : IMethodDeclaration
    {
        #region Fields

        private IList<IParameterDeclaration> parameters;
        private IMethodDeclaration genericMethod;
        private IType declaringType;
        private IList<IType> genericArguments;

        #endregion

        #region IMethodDeclaration Members

        /// <summary>
        /// Whether the method is abstract - this is the same as for the GenericMethod.
        /// For this reason, it cannot be set in this class
        /// </summary>
        public bool Abstract
        {
            get { return this.genericMethod.Abstract; }
            set { throw new NotSupportedException(); }
        }

        /// <summary>
        /// The body of this specialization
        /// </summary>
        public IBlockStatement Body
        {
            get { throw new NotImplementedException(); }
            set { throw new NotSupportedException(); }
        }

        /// <summary>
        /// Whether the method is final (i.e. can no longer be overridden) - this is the
        /// same as for the GenericMethod. For this reason, it cannot be set in this class
        /// </summary>
        public bool Final
        {
            get { return this.genericMethod.Final; }
            set { throw new NotSupportedException(); }
        }

        /// <summary>
        /// For a virtual method, indicates whether the method is 'new' versus 'override'.
        /// This is the same as for the GenericMethod. For this reason, it cannot be set in this class.
        /// </summary>
        public bool NewSlot
        {
            get
            {
                return this.genericMethod.NewSlot;
            }
            set
            {
                throw new NotSupportedException();
            }
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
        /// Whether the method is static - this is the same as for the GenericMethod.
        /// For this reason, it cannot be set in this class
        /// </summary>
        public bool Static
        {
            get { return this.genericMethod.Static; }
            set { throw new NotSupportedException(); }
        }

        /// <summary>
        /// Whether the method is virtual - this is the same as for the GenericMethod.
        /// For this reason, it cannot be set in this class
        /// </summary>
        public bool Virtual
        {
            get { return this.genericMethod.Virtual; }
            set { throw new NotSupportedException(); }
        }

        /// <summary>
        /// The visibility of the method - this is the same as for the GenericMethod.
        /// For this reason, it cannot be set in this class
        /// </summary>
        public MethodVisibility Visibility
        {
            get { return this.genericMethod.Visibility; }
            set { throw new NotSupportedException(); }
        }

        #endregion

        #region IMethodReference Members

        /// <summary>
        /// The generic method for which this is a specialization.
        /// Setting this property will trigger a resolve of the method
        /// reference to its declaration
        /// </summary>
        public IMethodReference GenericMethod
        {
            get { return this.genericMethod; }
            set { this.genericMethod = value as IMethodDeclaration; }
        }

        /// <summary>
        /// Just returns this - it is already a declaration
        /// </summary>
        /// <returns></returns>
        public IMethodDeclaration Resolve()
        {
            return this;
        }

        #endregion

        #region IMemberReference Members

        /// <summary>
        /// The type of the declaring class
        /// </summary>
        public IType DeclaringType
        {
            get { return this.declaringType; }
            set { this.declaringType = value; }
        }

        /// <summary>
        /// The method's name - this is the same as for the GenericMethod.
        /// For this reason, it cannot be set in this class
        /// </summary>
        public string Name
        {
            get { return this.genericMethod.Name; }
            set { throw new NotSupportedException(); }
        }

        #endregion

        #region IComparable Members

        public int CompareTo(object obj)
        {
            if (this == obj)
                return 0;

            IMethodReference imr = obj as IMethodReference;
            if (imr == null)
                throw new ArgumentException("Cannot compare a MethodInstanceDeclaration to a null reference");
            return CompareItems.CompareMethodReferences(this, imr);
        }

        #endregion

        #region IMethodSignature Members

        /// <summary>
        /// Method Info
        /// </summary>
        public MethodBase MethodInfo { get; set; }

        /// <summary>
        /// Parameter declarations
        /// </summary>
        public IList<IParameterDeclaration> Parameters
        {
            get
            {
                if (this.parameters == null)
                {
                    this.parameters = new List<IParameterDeclaration>();
                }
                return this.parameters;
            }
        }

        public IMethodReturnType ReturnType
        {
            get
            {
                IMethodReturnType returnType = new XMethodReturnType();
                returnType.Attributes.AddRange(this.GenericMethod.ReturnType.Attributes);
                //returnType.Type = new XTypeReference();
                //MethodInfo mi = this.MethodInfo as MethodInfo;
                //if (mi != null)
                //{
                //    returnType.Type.DotNetType = mi.ReturnType;
                //}
                returnType.Type = Generics.GetType(
                    this.GenericMethod.ReturnType.Type,
                    this.DeclaringType as IGenericArgumentProvider,
                    this);
                return returnType;
            }
            set { throw new NotSupportedException(); }
        }

        #endregion

        #region IGenericArgumentProvider Members

        /// <summary>
        /// Any remaining generic arguments needed by this method
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
        /// The custom attributes for this method - these are the same as for the GenericMethod.
        /// </summary>
        public List<ICustomAttribute> Attributes
        {
            get { return this.genericMethod.Attributes; }
        }

        #endregion

        #region IDocumentationProvider Members

        /// <summary>
        /// The documentation for the method - this is the same as for the GenericMethod.
        /// For this reason, it cannot be set in this class
        /// </summary>
        public string Documentation
        {
            get { return this.genericMethod.Documentation; }
            set { throw new NotSupportedException(); }
        }

        #endregion

        #region Object Overrides

        public override string ToString()
        {
            ILanguageWriter writer = new CSharpWriter() as ILanguageWriter;
            return writer.MethodDeclarationSource(this);
        }

        #endregion
    }
}
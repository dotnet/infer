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
    /// This class models a reference to a method
    /// </summary>
    internal class XMethodReference : IMethodReference
    {
        #region Fields

        private List<ICustomAttribute> attributes;
        private IMethodReference genericMethod;
        private IMethodReturnType returnType = new XMethodReturnType();
        private IList<IParameterDeclaration> parameters;
        private IType declaringType;
        private IList<IType> genericArguments;
        private string name;
        private WeakReference methodDeclaration = null;

        #endregion

        #region Constructor

        public XMethodReference()
        {
        }

        public XMethodReference(IMethodDeclaration methodDecl)
        {
            this.methodDeclaration = new WeakReference(methodDecl);
            this.name = methodDecl.Name;
            this.declaringType = methodDecl.DeclaringType;
            this.returnType = methodDecl.ReturnType;
            this.parameters = methodDecl.Parameters;
        }

        #endregion

        #region Custom Attributes

        /// <summary>
        /// Custom attributes for the method
        /// </summary>
        public List<ICustomAttribute> Attributes
        {
            get
            {
                if (this.attributes == null)
                {
                    this.attributes = new List<ICustomAttribute>();
                }
                return this.attributes;
            }
        }

        #endregion

        #region IMethodReference Members

        /// <summary>
        /// Reference to the generic method if this method is a specialisation of a generic method
        /// </summary>
        public IMethodReference GenericMethod
        {
            get { return this.genericMethod; }
            set { this.genericMethod = value; }
        }

        /// <summary>
        /// Resolve the method reference
        /// </summary>
        /// <returns>The declaration for the reference</returns>
        public IMethodDeclaration Resolve()
        {
            // If we have already resolved this, and the reference is still
            // alive, then return it.
            if (this.methodDeclaration != null && this.methodDeclaration.IsAlive)
                return (IMethodDeclaration) this.methodDeclaration.Target;

            // Get the declaring type reference
            ITypeReference declrType = this.DeclaringType as ITypeReference;

            // Search up the inheritance chain for the method declaration
            while (declrType != null)
            {
                // Resolve the declaring type
                ITypeDeclaration decl = declrType.Resolve();
                if (decl == null)
                    return null;

                // Look through the declaring type's methods
                foreach (IMethodDeclaration imd in decl.Methods)
                {
                    if (CompareItems.MethodReferencesAreEqualInner(this, imd))
                    {
                        this.methodDeclaration = new WeakReference(imd);
                        return (IMethodDeclaration) (this.methodDeclaration.Target);
                    }
                }

                // Go up an inheritance level
                declrType = decl.BaseType;
            }

            // Nothing doing
            return null;
        }

        #endregion

        #region IMemberReference Members

        /// <summary>
        /// The declaring type for this member method
        /// </summary>
        public IType DeclaringType
        {
            get { return this.declaringType; }
            set { this.declaringType = value; }
        }

        /// <summary>
        /// Method name
        /// </summary>
        public string Name
        {
            get { return this.name; }
            set { this.name = value; }
        }

        #endregion

        #region IComparable Members

        /// <summary>
        /// Compare one method reference to another
        /// </summary>
        /// <param name="obj">Method reference to compare to</param>
        /// <returns>-1, 0 or 1</returns>
        public int CompareTo(object obj)
        {
            if (this == obj)
                return 0;

            IMethodReference imr = obj as IMethodReference;
            if (imr == null)
                throw new NotSupportedException("Cannot compare an IMethodReference to a null reference");
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

        /// <summary>
        /// Method return type
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
                {
                    this.genericArguments = new List<IType>();
                }
                return this.genericArguments;
            }
        }

        #endregion

        #region Object Overrides

        /// <summary>
        /// ToString override
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return this.Name ?? base.ToString();
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
            if (this.Name == null) return DeclaringType.GetHashCode();
            else return this.Name.GetHashCode();
        }

        #endregion Object Overrides
    }
}
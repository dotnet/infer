// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Reflection;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// This class models a reference to a specialization of a generic method
    /// </summary>
    internal class XMethodInstanceReference : IMethodReference
    {
        #region Fields

        private IList<IParameterDeclaration> parameters;
        private IMethodReference genericMethod;
        private IList<IType> genericArguments;
        private IType declaringType;
        private WeakReference methodDeclaration = null;

        #endregion

        #region Constructor

        public XMethodInstanceReference()
        {
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

        public IMethodDeclaration Resolve()
        {
            // If we have already resolved this, and the reference is still
            // alive, then return it.
            if (this.methodDeclaration != null && this.methodDeclaration.IsAlive)
                return (IMethodDeclaration) this.methodDeclaration.Target;

            // Try resolving the generic method for which this is a specialization
            IMethodReference genericDecl = this.GenericMethod.Resolve();

            if (genericDecl == null)
                return null;

            // Build the instance declaration
            IMethodDeclaration instDecl = new XMethodInstanceDeclaration();
            instDecl.GenericMethod = genericDecl;
            instDecl.GenericArguments.AddRange(this.GenericArguments);
            ITypeReference declaringType = this.DeclaringType as ITypeReference;
            if (declaringType != null)
                instDecl.DeclaringType = declaringType.Resolve();
            // Are there other properties of instDecl that we should set?

            this.methodDeclaration = new WeakReference(instDecl);
            return (IMethodDeclaration) this.methodDeclaration.Target;
        }

        #endregion

        #region IMemberReference Members

        /// <summary>
        /// The declaring type for this method instance
        /// </summary>
        public IType DeclaringType
        {
            get { return this.declaringType; }
            set { this.declaringType = value; }
        }

        /// <summary>
        /// Method instance name
        /// </summary>
        public string Name
        {
            get { return this.GenericMethod.Name; }
            set { throw new NotSupportedException("Cannot set MethodInstanceReference name"); }
        }

        #endregion

        #region IComparable Members

        /// <summary>
        /// Compare this to an IMethodReference
        /// </summary>
        /// <param name="obj">Method reference to comapre to</param>
        /// <returns>-1, 0 or 1</returns>
        public int CompareTo(object obj)
        {
            if (this == obj)
                return 0;

            IMethodReference imr = obj as IMethodReference;
            if (imr == null)
                throw new ArgumentException("Cannot compare a MethodInstanceReference to a null reference");
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
        /// Return type of the method
        /// </summary>
        public IMethodReturnType ReturnType
        {
            get
            {
                IMethodReturnType imrt = new XMethodReturnType();
                imrt.Attributes.AddRange(this.GenericMethod.ReturnType.Attributes);
                imrt.Type = Generics.GetType(
                    this.GenericMethod.ReturnType.Type,
                    this.DeclaringType as IGenericArgumentProvider,
                    this);
                return imrt;
            }
            set { throw new Exception("The method or operation is not implemented."); }
        }

        #endregion

        #region IGenericArgumentProvider Members

        /// <summary>
        /// Generic argument collection
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

        #region Object Overrides

        public override string ToString()
        {
            StringBuilder genericArgString = new StringBuilder();
            if (GenericArguments.Count > 0)
            {
                genericArgString.Append("<");
                genericArgString.Append(StringUtil.CollectionToString(GenericArguments, ","));
                genericArgString.Append(">");
            }
            return "MethodInstanceReference: " + GenericMethod + genericArgString + Util.CollectionToString(Parameters);
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            XMethodInstanceReference mie = obj as XMethodInstanceReference;
            if (mie == null)
                return false;

            return DeclaringType.Equals(mie.DeclaringType) &&
                   ((GenericMethod == mie.GenericMethod) ||
                    (GenericMethod != null && GenericMethod.Equals(mie.GenericMethod))) &&
                   (GenericArguments.Equals(mie.genericArguments)) &&
                   (Parameters.Equals(mie.Parameters));
        }

        public override int GetHashCode()
        {
            int hash = Hash.Start;
            hash = Hash.Combine(hash, DeclaringType.GetHashCode());
            hash = Hash.Combine(hash, (GenericMethod == null ? Hash.Start : GenericMethod.GetHashCode()));
            hash = Hash.Combine(hash, GenericArguments.GetHashCode());
            hash = Hash.Combine(hash, Parameters.GetHashCode());
            return this.Parameters.GetHashCode();
        }

        #endregion Object Overrides
    }
}
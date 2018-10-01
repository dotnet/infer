// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Property reference
    /// </summary>
    internal class XPropertyReference : IPropertyReference
    {
        #region Fields

        private IList<IParameterDeclaration> parameters;
        private IPropertyReference genericProperty;
        private IType declaringType;
        private IType propertyType;
        private string name;
        private WeakReference declaration;

        #endregion

        #region IPropertyReference Members

        /// <summary>
        /// Generic property of which this is a specialization (if any)
        /// </summary>
        public IPropertyReference GenericProperty
        {
            get { return this.genericProperty; }
            set { this.genericProperty = value; }
        }

        /// <summary>
        /// Parameters of the property
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
        /// Type of the property
        /// </summary>
        public IType PropertyType
        {
            get { return this.propertyType; }
            set { this.propertyType = value; }
        }

        /// <summary>
        /// Resolve the property reference
        /// </summary>
        /// <remarks>
        /// This does not deal with generic properties
        /// </remarks>
        /// <returns>The property declaration for this reference</returns>
        public IPropertyDeclaration Resolve()
        {
            // If we have already resolved this, and the reference is still
            // alive, then return it.
            if (this.declaration != null && this.declaration.IsAlive)
                return (IPropertyDeclaration) this.declaration.Target;

            // Get the declaring type reference
            ITypeReference declrType = this.DeclaringType as ITypeReference;

            // Get the properties from the declaring type
            List<IPropertyDeclaration> properties = declrType.Resolve().Properties;

            // Search through the properties to see if we can resolve
            foreach (IPropertyDeclaration ipd in properties)
            {
                if (this.Name.Equals(ipd.Name) && this.PropertyType.Equals(ipd.PropertyType))
                {
                    this.declaration = new WeakReference(ipd);
                    return (IPropertyDeclaration) (this.declaration.Target);
                }
            }

            // Nothing doing
            return null;
        }

        #endregion

        #region IMemberReference Members

        /// <summary>
        /// Declaring type for this property
        /// </summary>
        public IType DeclaringType
        {
            get { return this.declaringType; }
            set { this.declaringType = value; }
        }

        /// <summary>
        /// Name of this property
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
            IPropertyReference reference = obj as IPropertyReference;

            if (reference == null)
                throw new NotSupportedException();

            return CompareItems.ComparePropertyReferences(this, reference);
        }

        #endregion

        #region Object Overrides

        /// <summary>
        /// ToString override
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return this.Name;
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
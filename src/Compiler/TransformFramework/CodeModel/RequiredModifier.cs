// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// A required modifier
    /// </summary>
    internal class XRequiredModifier : IRequiredModifier
    {
        #region IRequiredModifier Members

        #region Fields

        private IType elementType;
        private ITypeReference modifier;
        private Type dotNetType; // cached dotnet type

        #endregion

        #region IRequiredModifier Members

        /// <summary>
        /// The element type
        /// </summary>
        public IType ElementType
        {
            get { return this.elementType; }
            set
            {
                if (this.elementType != value)
                    this.elementType = value;
            }
        }

        /// <summary>
        /// The modifier
        /// </summary>
        public ITypeReference Modifier
        {
            get { return this.modifier; }
            set
            {
                if (this.modifier != value)
                    this.modifier = value;
            }
        }

        #endregion

        #region IComparable Members

        public int CompareTo(object obj)
        {
            IRequiredModifier requiredModifier = obj as IRequiredModifier;
            if (requiredModifier == null)
                return -1;

            return this.ElementType.CompareTo(requiredModifier.ElementType);
        }

        #endregion

        #region Object Overrides

        public override string ToString()
        {
            return this.ElementType.ToString() + this.Modifier.ToString();
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IRequiredModifier requiredModifier = obj as IRequiredModifier;
            if (requiredModifier == null)
                return false;

            return
                (this.ElementType.Equals(requiredModifier.ElementType) &&
                 this.Modifier.Equals(requiredModifier.Modifier));
        }

        public override int GetHashCode()
        {
            return Hash.Combine(ElementType.GetHashCode(), Modifier.GetHashCode());
        }

        #endregion Object Overrides

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
    }
}
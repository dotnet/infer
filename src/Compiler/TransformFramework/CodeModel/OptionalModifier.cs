// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// An optional modifier
    /// </summary>
    internal class XOptionalModifier : IOptionalModifier
    {
        #region Fields

        private IType elementType;
        private ITypeReference modifier;
        private Type dotNetType; // cached dotnet type

        #endregion

        #region IOptionalModifier Members

        /// <summary>
        /// Element type
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
        /// Modifier
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
            IOptionalModifier optionalModifier = obj as IOptionalModifier;
            if (optionalModifier == null)
                return -1;

            int ret = this.ElementType.CompareTo(optionalModifier.ElementType);
            if (ret != 0)
                return ret;

            return this.Modifier.CompareTo(optionalModifier.Modifier);
        }

        #endregion

        #region Object Overrides

        public override string ToString()
        {
            // RTODO
            return this.ElementType.ToString() + this.Modifier.ToString();
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IOptionalModifier optionalModifier = obj as IOptionalModifier;
            if (optionalModifier == null)
                return false;

            return
                (this.ElementType.Equals(optionalModifier.ElementType) &&
                 this.Modifier.Equals(optionalModifier.Modifier));
        }

        public override int GetHashCode()
        {
            return Hash.Combine(ElementType.GetHashCode(), Modifier.GetHashCode());
        }

        #endregion Object Overrides

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
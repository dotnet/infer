// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// A pointer type
    /// </summary>
    internal class XPointerType : IPointerType
    {
        #region Fields

        private IType elementType;
        private Type dotNetType; // cached dotnet type

        #endregion

        #region IPointerType Members

        /// <summary>
        /// The associated element type
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

        #endregion

        #region IComparable Members

        public int CompareTo(object obj)
        {
            IPointerType pointerType = obj as IPointerType;
            if (pointerType == null)
                return -1;
            return this.ElementType.CompareTo(pointerType.ElementType);
        }

        #region Object Overrides

        public override string ToString()
        {
            return this.ElementType.ToString();
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IPointerType pointerType = obj as IPointerType;
            if (pointerType == null)
                return false;

            return this.ElementType.Equals(pointerType.ElementType);
        }

        public override int GetHashCode()
        {
            return this.ElementType.GetHashCode();
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
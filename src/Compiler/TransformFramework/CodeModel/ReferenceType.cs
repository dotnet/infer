// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Reference type
    /// </summary>
    internal class XReferenceType : IReferenceType
    {
        #region Fields

        private IType elementType;
        private Type dotNetType;

        #endregion

        #region Constructor

        public XReferenceType()
        {
        }

        #endregion

        #region IReferenceType Members

        /// <summary>
        /// The underlying element type
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

        #region IDotNetType Members

        /// <summary>
        /// The dotNet type
        /// </summary>
        public Type DotNetType
        {
            get { return this.dotNetType; }
            set { this.dotNetType = value; }
        }

        #endregion

        #region IComparable Members

        public int CompareTo(object obj)
        {
            IReferenceType type = obj as IReferenceType;
            if (type == null)
            {
                return -1;
            }
            else
            {
                return this.ElementType.CompareTo(type.ElementType);
            }
        }

        #endregion

        #region Object Overrides

        public override string ToString()
        {
            ILanguageWriter writer = new CSharpWriter() as ILanguageWriter;
            return writer.TypeSource(this);
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IDotNetType type = obj as IDotNetType;
            if (type == null)
                return false;

            return (this.DotNetType == type.DotNetType);
        }

        public override int GetHashCode()
        {
            return this.DotNetType.GetHashCode();
        }

        #endregion Object Overrides
    }
}
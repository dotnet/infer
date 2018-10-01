// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Represents an array type
    /// </summary>
    internal class XArrayType : IArrayType
    {
        #region Fields

        private IType elementType;
        private Type dotNetType; // cached dotnet type

        #endregion

        #region IArrayType Members

        /// <summary>
        /// The rank of the array
        /// </summary>
        public int Rank
        {
            get
            {
                if (dotNetType != null && dotNetType.IsArray)
                    return dotNetType.GetArrayRank();
                else
                    throw new InferCompilerException("Unable to get rank of array");
            }
        }

        /// <summary>
        /// The element type of the array
        /// </summary>
        public IType ElementType
        {
            get { return this.elementType; }
            set { this.elementType = value; }
        }

        #endregion

        #region IComparable Members

        public int CompareTo(object obj)
        {
            IArrayType type = obj as IArrayType;
            if (type == null)
                return -1;
            return CompareItems.CompareArrayTypes(this, type);
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

            IArrayType arrayType = obj as IArrayType;
            if (arrayType == null)
                return false;

            return CompareItems.ArrayTypesAreEqual(this, arrayType);
        }

        public override int GetHashCode()
        {
            return this.ElementType.GetHashCode();
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
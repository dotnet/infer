// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// This class represents a generic type argument for a generic type or
    /// a generic method
    /// </summary>
    internal class XGenericArgument : IGenericArgument
    {
        #region Fields

        private IGenericArgumentProvider owner;
        private int position;
        private Type dotNetType; // cached dotnet type

        #endregion

        #region IGenericArgument Members

        /// <summary>
        /// The type reference or method reference which specifies this argument
        /// </summary>
        public IGenericArgumentProvider Owner
        {
            get { return this.owner; }
            set { this.owner = value; }
        }

        /// <summary>
        /// The position of this argument
        /// </summary>
        public int Position
        {
            get { return this.position; }
            set { this.position = value; }
        }

        /// <summary>
        /// Resolve the argument - return the type of the argument
        /// </summary>
        /// <returns></returns>
        public IType Resolve()
        {
            return this.Owner.GenericArguments[this.position];
        }

        #endregion

        #region IComparable Members

        public int CompareTo(object obj)
        {
            throw new Exception("The method or operation is not implemented.");
        }

        #endregion

        #region Object Overrides

        public override string ToString()
        {
            ILanguageWriter writer = new CSharpWriter() as ILanguageWriter;
            return writer.TypeSource(this);
            //// RTODO
            //IType it = this.Resolve();
            //if (it != null)
            //    return it.ToString();
            //else
            //    return base.ToString();
        }

        /// <summary>
        /// Test whether one generic argument equals another
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IGenericArgument iga = obj as IGenericArgument;
            if (iga == null)
                return false;

            return
                this.Position == iga.Position;
        }

        public override int GetHashCode()
        {
            return this.Position.GetHashCode();
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
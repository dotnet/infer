// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    internal class XGenericParameter : IGenericParameter
    {
        #region Fields

        private List<ICustomAttribute> attributes;
        private IList<IType> constraints;
        private string name;
        private IGenericArgumentProvider owner;
        private int position;
        private Type dotNetType; // cached dotnet type

        #endregion

        #region IGenericParameter Members

        public IList<IType> Constraints
        {
            get
            {
                if (this.constraints == null)
                {
                    this.constraints = new List<IType>();
                }
                return this.constraints;
            }
        }

        public string Name
        {
            get { return this.name; }
            set { this.name = value; }
        }

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

        public IType Resolve()
        {
            return null;
        }

        #endregion

        #region IComparable Members

        public int CompareTo(object obj)
        {
            throw new Exception("The method or operation is not implemented.");
        }

        #endregion

        #region ICustomAttributeProvider Members

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

            IGenericParameter igp = obj as IGenericParameter;
            if (igp == null)
                return false;

            if (this.Name != igp.Name ||
                //(!this.Owner.Equals(igp.Owner)) ||
                this.Position != igp.Position ||
                this.Constraints.Count != igp.Constraints.Count ||
                this.Attributes.Count != igp.Attributes.Count)
                return false;

            for (int i = 0; i < this.Constraints.Count; i++)
                if (!this.Constraints[i].Equals(igp.Constraints[i]))
                    return false;

            for (int i = 0; i < this.Attributes.Count; i++)
                if (!this.Attributes[i].Equals(igp.Attributes[i]))
                    return false;

            return true;
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
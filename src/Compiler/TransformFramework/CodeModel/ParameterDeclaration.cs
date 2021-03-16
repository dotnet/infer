// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Parameter declaration
    /// </summary>
    internal class XParameterDeclaration : IParameterDeclaration
    {
        #region Fields

        private List<ICustomAttribute> attributes;
        private IType parameterType;
        private string name;

        #endregion

        #region IParameterDeclaration Members

        /// <summary>
        /// Type of the parameter
        /// </summary>
        public IType ParameterType
        {
            get { return this.parameterType; }
            set
            {
                if (this.parameterType != value)
                    this.parameterType = value;
            }
        }

        #endregion

        #region IParameterReference Members

        /// <summary>
        /// Parmater name
        /// </summary>
        public string Name
        {
            get { return this.name; }
            set
            {
                if (this.name != value)
                    this.name = value;
            }
        }

        /// <summary>
        /// Always returns this
        /// </summary>
        /// <returns></returns>
        public IParameterDeclaration Resolve()
        {
            return this;
        }

        #endregion

        #region ICustomAttributeProvider Members

        /// <summary>
        /// Attributes attached to the parameter
        /// </summary>
        public List<ICustomAttribute> Attributes
        {
            get
            {
                if (this.attributes == null)
                    this.attributes = new List<ICustomAttribute>();
                return this.attributes;
            }
        }

        #endregion

        #region Object overrides

        /// <summary>
        /// ToString override
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            var writer = new CSharpWriter();
            return writer.ParameterDeclarationSource(this);
        }

        /// <summary>
        /// Test whether one parameter declaration equals another
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IParameterDeclaration ipd = obj as IParameterDeclaration;
            if (ipd == null)
                return false;

            if (this.Name != ipd.Name ||
                (!this.ParameterType.Equals(ipd.ParameterType)) ||
                this.Attributes.Count != ipd.Attributes.Count)
                return false;

            for (int i = 0; i < Attributes.Count; i++)
                if (!Equals(Attributes[i], ipd.Attributes[i]))
                    return false;

            return true;
        }

        /// <summary>
        /// Hash code override
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            int hash = Hash.Start;
            hash = Hash.Combine(hash, parameterType.GetHashCode());
            hash = Hash.Combine(hash, (name == null ? Hash.Start : name.GetHashCode()));
            return hash;
        }

        #endregion
    }
}
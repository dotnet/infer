// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Method return type
    /// </summary>
    internal class XMethodReturnType : IMethodReturnType
    {
        #region Fields

        private List<ICustomAttribute> attributes;
        private IType type = new XTypeReference() {DotNetType = typeof (void)};

        #endregion

        #region IMethodReturnType Members

        /// <summary>
        /// The type
        /// </summary>
        public IType Type
        {
            get { return this.type; }
            set
            {
                if (this.type != value)
                    this.type = value;
            }
        }

        #endregion

        #region ICustomAttributeProvider Members

        /// <summary>
        /// The attributes of the return type
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
    }
}
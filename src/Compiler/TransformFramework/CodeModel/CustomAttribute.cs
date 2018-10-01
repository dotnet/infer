// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// A custom attribute
    /// </summary>
    internal class XCustomAttribute : ICustomAttribute
    {
        private IList<IExpression> arguments;
        private IMethodReference constructor;

        #region ICustomAttribute Members

        /// <summary>
        /// The attribute's arguments
        /// </summary>
        public IList<IExpression> Arguments
        {
            get
            {
                if (this.arguments == null)
                    this.arguments = new List<IExpression>();
                return this.arguments;
            }
        }

        /// <summary>
        /// Attribute constructor
        /// </summary>
        public IMethodReference Constructor
        {
            get { return this.constructor; }
            set { this.constructor = value; }
        }

        #endregion
    }
}
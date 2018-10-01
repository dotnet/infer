// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// A constructor declaration
    /// </summary>
    internal class XConstructorDeclaration : XMethodDeclaration, IConstructorDeclaration
    {
        #region Fields

        private IMethodInvokeExpression initializer;

        #endregion

        #region IConstructorDeclaration Members

        /// <summary>
        /// The method call
        /// </summary>
        public IMethodInvokeExpression Initializer
        {
            get { return this.initializer; }
            set
            {
                if (this.initializer != value)
                    this.initializer = value;
            }
        }

        #endregion
    }
}
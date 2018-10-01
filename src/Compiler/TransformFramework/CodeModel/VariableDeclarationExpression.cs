// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Variable declaration expression
    /// </summary>
    internal class XVariableDeclarationExpression : XExpression, IVariableDeclarationExpression
    {
        #region Fields

        private IVariableDeclaration variable;

        #endregion

        #region IVariableDeclarationExpression Members

        /// <summary>
        /// The variable declaration
        /// </summary>
        public IVariableDeclaration Variable
        {
            get { return this.variable; }
            set { this.variable = value; }
        }

        #endregion

        #region Object Overrides

        public override string ToString()
        {
            // Base class will do the right thing
            return base.ToString();
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IVariableDeclarationExpression expressionDecl = obj as IVariableDeclarationExpression;
            if (expressionDecl == null)
                return false;

            return this.Variable.Resolve().Equals(expressionDecl.Variable.Resolve());
        }

        public override int GetHashCode()
        {
            if (this.Variable != null)
                return this.Variable.GetHashCode();
            else
                return 433494437;
        }

        #endregion Object Overrides

        /// <summary>
        /// Get expression type
        /// </summary>
        public override Type GetExpressionType()
        {
            IVariableDeclaration ivd = this.Variable;
            return ivd.VariableType.DotNetType;
        }
    }
}
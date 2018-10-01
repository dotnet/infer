// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Variable reference expression
    /// </summary>
    internal class XVariableReferenceExpression : XExpression, IVariableReferenceExpression
    {
        #region Fields

        private IVariableReference variable;

        #endregion

        #region IVariableReferenceExpression Members

        /// <summary>
        /// The variable
        /// </summary>
        public IVariableReference Variable
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

            IVariableReferenceExpression expressionRef = obj as IVariableReferenceExpression;
            if (expressionRef == null)
                return false;
            return this.Variable.Equals(expressionRef.Variable.Resolve());
        }

        public override int GetHashCode()
        {
            if (this.Variable != null)
                return this.Variable.GetHashCode();
            else
                return 39916801;
        }

        #endregion Object Overrides

        /// <summary>
        /// Get expression type
        /// </summary>
        public override Type GetExpressionType()
        {
            IVariableDeclaration ivd = this.Variable.Resolve();
            return ivd.VariableType.DotNetType;
        }
    }
}
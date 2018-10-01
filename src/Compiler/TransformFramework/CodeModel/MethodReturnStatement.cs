// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Method return statement
    /// </summary>
    internal class XMethodReturnStatement : XStatement, IMethodReturnStatement
    {
        #region Fields

        private IExpression expression;

        #endregion

        #region IMethodReturnStatement Members

        /// <summary>
        /// The expression in the statement
        /// </summary>
        public IExpression Expression
        {
            get { return this.expression; }
            set { this.expression = value; }
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

            IMethodReturnStatement statement = obj as IMethodReturnStatement;
            if (statement == null)
                return false;

            return (this.Expression == statement.Expression);
        }

        public override int GetHashCode()
        {
            int hash = "XMethodReturnStatement".GetHashCode();
            if (this.expression != null) hash += this.expression.GetHashCode();
            return hash;
        }

        #endregion Object Overrides
    }
}
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// An expression statement
    /// </summary>
    internal class XExpressionStatement : XStatement, IExpressionStatement
    {
        #region Fields

        private IExpression expression;

        #endregion

        #region Constructor

        public XExpressionStatement()
        {
        }

        #endregion

        #region IExpressionStatement Members

        /// <summary>
        /// The expression
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
            return this.Expression.ToString();
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IExpressionStatement statement = obj as IExpressionStatement;
            if (statement == null)
                return false;

            if (this.Expression.Equals(statement.Expression))
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        public override int GetHashCode()
        {
            return this.Expression.GetHashCode();
        }

        #endregion
    }
}
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Using statement
    /// </summary>
    internal class XUsingStatement : XStatement, IUsingStatement
    {
        #region Fields

        private IBlockStatement body;
        private IExpression expression;

        #endregion

        #region IUsingStatement Members

        /// <summary>
        /// The body of the using statement
        /// </summary>
        public IBlockStatement Body
        {
            get { return this.body; }
            set { this.body = value; }
        }

        /// <summary>
        /// The expression in the using statement
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

            IUsingStatement statement = obj as IUsingStatement;
            if (statement == null)
                return false;

            return
                this.Body.Equals(statement.Body) &&
                this.Expression.Equals(statement.Expression);
        }

        public override int GetHashCode()
        {
            return this.Expression.GetHashCode() + this.Body.GetHashCode();
        }

        #endregion Object Overrides
    }
}
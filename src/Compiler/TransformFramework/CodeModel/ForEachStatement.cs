// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// A Foreach statement
    /// </summary>
    internal class XForEachStatement : XStatement, IForEachStatement
    {
        #region Fields

        // Elements of the for statement
        private IBlockStatement body;
        private IExpression expression;
        private IVariableDeclaration variable;

        #endregion

        #region IForStatement Members

        /// <summary>
        /// The Foreach statement code body
        /// </summary>
        public IBlockStatement Body
        {
            get { return this.body; }
            set { this.body = value; }
        }

        /// <summary>
        /// The expression 
        /// </summary>
        public IExpression Expression
        {
            get { return this.expression; }
            set { this.expression = value; }
        }

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
            // RTODO
            return base.ToString();
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IForEachStatement statement = obj as IForEachStatement;
            if (statement == null)
                return false;

            return
                Variable.Equals(statement.Variable) &&
                Expression.Equals(statement.Expression) &&
                Body.Equals(statement.Body);
        }

        public override int GetHashCode()
        {
            return this.Body.GetHashCode();
        }

        #endregion Object Overrides
    }
}
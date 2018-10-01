// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// A condition statement
    /// </summary>
    internal class XConditionStatement : XStatement, IConditionStatement
    {
        #region Fields

        private IExpression condition;
        private IBlockStatement thenBlock;
        private IBlockStatement elseBlock;

        #endregion

        #region IConditionStatement Members

        /// <summary>
        /// Condition expression
        /// </summary>
        public IExpression Condition
        {
            get { return this.condition; }
            set { this.condition = value; }
        }

        /// <summary>
        /// The block of 'else' statements
        /// </summary>
        public IBlockStatement Else
        {
            get { return this.elseBlock; }
            set { this.elseBlock = value; }
        }

        /// <summary>
        /// The block of 'then' statements
        /// </summary>
        public IBlockStatement Then
        {
            get { return this.thenBlock; }
            set { this.thenBlock = value; }
        }

        #endregion

        #region Object Overrides

        public override string ToString()
        {
            // Base statement will deal with this correctly
            return base.ToString();
        }

        public override bool Equals(object obj)
        {
            if (this == obj) return true;
            IConditionStatement statement = obj as IConditionStatement;
            if (statement == null) return false;
            if ((Condition == null) ? (Condition != statement.Condition) : !Condition.Equals(statement.Condition))
                return false;
            if ((Then == null) ? (Then != statement.Then) : !Then.Equals(statement.Then)) return false;
            if ((Else == null) ? (Else != statement.Else) : !Else.Equals(statement.Else)) return false;
            return true;
        }

        public override int GetHashCode()
        {
            return this.Condition.GetHashCode();
        }

        #endregion Object Overrides
    }
}
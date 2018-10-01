// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    internal class XCatchClause : ICatchClause
    {
        #region Fields

        private IBlockStatement body;
        private IExpression condition;
        private IVariableDeclaration variable;

        #endregion

        #region ICatchClause Members

        /// <summary>
        /// The body of the catch clause - a body of statements
        /// </summary>
        public IBlockStatement Body
        {
            get { return this.body; }
            set { this.body = value; }
        }

        // Condition
        public IExpression Condition
        {
            get { return this.condition; }
            set { this.condition = value; }
        }

        // The variable declaration for the catch clause
        public IVariableDeclaration Variable
        {
            get { return this.variable; }
            set { this.variable = value; }
        }

        #endregion

        #region Object Overrides

        public override string ToString()
        {
            return base.ToString();
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            ICatchClause clause = obj as ICatchClause;
            if (clause == null)
                return false;

            if (!this.Variable.Equals(clause.Variable))
                return false;

            if (this.Condition == null)
            {
                if (clause.Condition != null)
                    return false;
            }
            else
            {
                if (!this.Condition.Equals(clause.Condition))
                    return false;
            }

            if (!this.Body.Equals(clause.Body))
                return false;

            return true;
        }

        public override int GetHashCode()
        {
            return this.body.GetHashCode();
        }

        #endregion Object Overrides
    }
}
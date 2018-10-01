// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// While statement
    /// </summary>
    internal class XWhileStatement : XStatement, IWhileStatement
    {
        #region Fields

        private IBlockStatement body;
        private IExpression condition;

        #endregion

        #region IWhileStatement Members

        /// <summary>
        /// The body of the while statement
        /// </summary>
        public IBlockStatement Body
        {
            get { return this.body; }
            set { this.body = value; }
        }

        /// <summary>
        /// The condition expression
        /// </summary>
        public IExpression Condition
        {
            get { return this.condition; }
            set { this.condition = value; }
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

            IWhileStatement statement = obj as IWhileStatement;
            if (statement == null)
                return false;

            return
                this.Body.Equals(statement.Body) &&
                this.Condition.Equals(statement.Condition);
        }

        public override int GetHashCode()
        {
            return this.Condition.GetHashCode() + this.Body.GetHashCode();
        }

        #endregion Object Overrides
    }
}
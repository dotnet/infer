// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// For loop
    /// </summary>
    internal class XForStatement : XStatement, IForStatement
    {
        #region Fields

        private IBlockStatement body;
        private IExpression condition;
        private IStatement increment;
        private IStatement initializer;

        #endregion

        #region IForStatement Members

        /// <summary>
        /// The body of the for loop
        /// </summary>
        public IBlockStatement Body
        {
            get { return this.body; }
            set { this.body = value; }
        }

        /// <summary>
        /// The condition expression for the for loop
        /// </summary>
        public IExpression Condition
        {
            get { return this.condition; }
            set { this.condition = value; }
        }

        /// <summary>
        /// The increment statement for the for loop
        /// </summary>
        public IStatement Increment
        {
            get { return this.increment; }
            set { this.increment = value; }
        }

        /// <summary>
        /// The initializer statement for the for loop
        /// </summary>
        public IStatement Initializer
        {
            get { return this.initializer; }
            set { this.initializer = value; }
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

            IForStatement statement = obj as IForStatement;
            if (statement == null)
                return false;

            return
                Initializer.Equals(statement.Initializer) &&
                Increment.Equals(statement.Increment) &&
                Body.Equals(statement.Body) &&
                Condition.Equals(statement.Condition);
        }

        public override int GetHashCode()
        {
            return this.Body.GetHashCode();
        }

        #endregion Object Overrides
    }
}
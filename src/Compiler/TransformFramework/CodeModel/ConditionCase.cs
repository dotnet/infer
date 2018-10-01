// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// A condition block
    /// </summary>
    internal class XConditionCase : IConditionCase
    {
        #region Fields

        private IExpression condition;
        private IBlockStatement body;

        #endregion

        #region IConditionCase Members

        /// <summary>
        /// An expression giving the condition
        /// </summary>
        public IExpression Condition
        {
            get { return this.condition; }
            set { this.condition = value; }
        }

        #endregion

        #region ISwitchCase Members

        /// <summary>
        /// The body of statements
        /// </summary>
        public IBlockStatement Body
        {
            get { return this.body; }
            set { this.body = value; }
        }

        #endregion

        #region Object Overrides

        public override string ToString()
        {
            return Body.ToString();
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            XConditionCase that = obj as XConditionCase;
            if (that == null)
                return false;

            return
                this.Body.Equals(that.Body) &&
                this.Condition.Equals(that.Condition);
        }

        public override int GetHashCode()
        {
            return this.Condition.GetHashCode() + this.Body.GetHashCode();
        }

        #endregion Object Overrides
    }
}
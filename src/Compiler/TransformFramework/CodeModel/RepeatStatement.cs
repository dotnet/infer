// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Repeat block
    /// </summary>
    internal class XRepeatStatement : XStatement, IRepeatStatement
    {
        #region Fields

        private IBlockStatement body;
        private IExpression count;

        #endregion

        #region IRepeatStatement Members

        /// <summary>
        /// The body of the repeat block
        /// </summary>
        public IBlockStatement Body
        {
            get { return this.body; }
            set { this.body = value; }
        }

        /// <summary>
        /// The count expression for the repeat block
        /// </summary>
        public IExpression Count
        {
            get { return this.count; }
            set { this.count = value; }
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

            IRepeatStatement statement = obj as IRepeatStatement;
            if (statement == null)
                return false;

            return
                Body.Equals(statement.Body) &&
                Count.Equals(statement.Count);
        }

        public override int GetHashCode()
        {
            return this.Body.GetHashCode();
        }

        #endregion Object Overrides
    }
}
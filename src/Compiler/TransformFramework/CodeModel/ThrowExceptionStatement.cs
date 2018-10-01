// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    internal class XThrowExceptionStatement : IThrowExceptionStatement
    {
        #region Fields

        private IExpression expression;

        #endregion

        #region IThrowExceptionStatement Members

        /// <summary>
        /// The expression thrown
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
            // Base statement will deal with this correctly
            return base.ToString();
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IThrowExceptionStatement stmt = obj as IThrowExceptionStatement;
            if (stmt == null)
                return false;

            return Expression.Equals(stmt.Expression);
        }

        public override int GetHashCode()
        {
            return Expression.GetHashCode();
        }

        #endregion Object Overrides
    }
}
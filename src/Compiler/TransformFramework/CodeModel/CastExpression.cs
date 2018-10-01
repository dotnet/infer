// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// A cast expression
    /// </summary>
    internal class XCastExpression : XExpression, ICastExpression
    {
        #region Fields

        private IExpression expression;
        private IType targetType;

        #endregion

        #region Constructor

        public XCastExpression()
        {
        }

        #endregion

        #region ICastExpression Members

        /// <summary>
        /// The expression
        /// </summary>
        public IExpression Expression
        {
            get { return this.expression; }
            set { this.expression = value; }
        }

        /// <summary>
        /// The type of the cast
        /// </summary>
        public IType TargetType
        {
            get { return this.targetType; }
            set { this.targetType = value; }
        }

        #endregion

        #region Object Overrides

        public override string ToString()
        {
            // Base expression will deal with this correctly
            return base.ToString();
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            ICastExpression expression = obj as ICastExpression;
            if (expression == null)
                return false;

            return this.Expression.Equals(expression.Expression) &&
                this.TargetType.Equals(expression.TargetType);
        }

        public override int GetHashCode()
        {
            return this.Expression.GetHashCode();
        }

        #endregion Object Overrides

        /// <summary>
        /// Get expression type
        /// </summary>
        public override Type GetExpressionType()
        {
            return this.TargetType.DotNetType;
        }
    }
}
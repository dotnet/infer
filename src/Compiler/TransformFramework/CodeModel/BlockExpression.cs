// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    ///  A block of expressions
    /// </summary>
    internal class XBlockExpression : XExpression, IBlockExpression
    {
        #region Fields

        private IList<IExpression> expressions;

        #endregion

        #region IBlockExpression Members

        /// <summary>
        /// The expressions in the block
        /// </summary>
        public IList<IExpression> Expressions
        {
            get
            {
                if (this.expressions == null)
                    this.expressions = new List<IExpression>();
                return this.expressions;
            }
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

            IBlockExpression expression = obj as IBlockExpression;
            if (expression == null ||
                Expressions.Count != expression.Expressions.Count)
                return false;

            for (int i = 0; i < Expressions.Count; i++)
                if (!(Expressions[i].Equals(expression.Expressions[i])))
                    return false;

            return true;
        }

        public override int GetHashCode()
        {
            return this.Expressions.GetHashCode();
        }

        #endregion Object Overrides

        /// <summary>
        /// Get expression type
        /// </summary>
        public override Type GetExpressionType()
        {
            return null;
        }
    }
}
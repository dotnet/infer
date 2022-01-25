// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Method reference expression
    /// </summary>
    internal class XMethodReferenceExpression : XExpression, IMethodReferenceExpression
    {
        #region Fields

        private IMethodReference method;
        private IExpression target;

        #endregion

        #region IMethodReferenceExpression Members

        /// <summary>
        /// The method
        /// </summary>
        public IMethodReference Method
        {
            get { return this.method; }
            set { this.method = value; }
        }

        /// <summary>
        /// The target
        /// </summary>
        public IExpression Target
        {
            get { return this.target; }
            set { this.target = value; }
        }

        #endregion

        #region Object Overrides

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IMethodReferenceExpression expression = obj as IMethodReferenceExpression;
            if (expression == null)
                return false;

            return
                Method.Equals(expression.Method) &&
                Target.Equals(expression.Target);
        }

        public override int GetHashCode()
        {
            return this.Method.GetHashCode();
        }

        #endregion Object Overrides

        /// <summary>
        /// Get expression type
        /// </summary>
        public override Type GetExpressionType()
        {
            return typeof(Delegate);
        }
    }
}
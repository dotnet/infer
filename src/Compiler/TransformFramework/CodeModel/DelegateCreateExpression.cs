// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    internal class XDelegateCreateExpression : XExpression, IDelegateCreateExpression
    {
        #region Fields

        private ITypeReference delegateType;
        private IMethodReference method;
        private IExpression target;

        #endregion

        #region IDelegateCreateExpression Members

        public ITypeReference DelegateType
        {
            get { return this.delegateType; }
            set { this.delegateType = value; }
        }

        public IMethodReference Method
        {
            get { return this.method; }
            set { this.method = value; }
        }

        public IExpression Target
        {
            get { return this.target; }
            set { this.target = value; }
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

            IDelegateCreateExpression expression = obj as IDelegateCreateExpression;
            if (expression == null)
                return false;

            return
                this.DelegateType.Equals(expression.DelegateType) &&
                this.Method.Equals(expression.Method) &&
                this.Target.Equals(expression.Target);
        }

        public override int GetHashCode()
        {
            return this.DelegateType.GetHashCode();
        }

        #endregion Object Overrides

        /// <summary>
        /// Get expression type
        /// </summary>
        public override Type GetExpressionType()
        {
            if (delegateType == null) return null;
            return delegateType.DotNetType;
        }
    }
}
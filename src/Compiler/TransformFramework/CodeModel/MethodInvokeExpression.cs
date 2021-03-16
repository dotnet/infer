// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Method invoke expression
    /// </summary>
    internal class XMethodInvokeExpression : XExpression, IMethodInvokeExpression
    {
        #region Fields

        private IList<IExpression> arguments;
        private IMethodReferenceExpression method;

        #endregion

        #region IMethodInvokeExpression Members

        /// <summary>
        /// Arguments of the method
        /// </summary>
        public IList<IExpression> Arguments
        {
            get
            {
                if (this.arguments == null)
                    this.arguments = new List<IExpression>();
                return this.arguments;
            }
        }

        /// <summary>
        /// The method
        /// </summary>
        public IMethodReferenceExpression Method
        {
            get { return this.method; }
            set { this.method = value; }
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

            IMethodInvokeExpression expression = obj as IMethodInvokeExpression;
            if (expression == null ||
                !Method.Equals(expression.Method) ||
                Arguments.Count != expression.Arguments.Count)
                return false;

            for (int i = 0; i < Arguments.Count; i++)
                if (!(Arguments[i].Equals(expression.Arguments[i])))
                    return false;

            return true;
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
            IMethodReferenceExpression imre = this.Method;
            if (imre != null)
            {
                MethodInfo mi = (MethodInfo) imre.Method.MethodInfo;
                return mi.ReturnType;
            }
            return null;
        }
    }
}
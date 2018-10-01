// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Represents an argument reference expression
    /// </summary>
    internal class XArgumentReferenceExpression : XExpression, IArgumentReferenceExpression
    {
        #region Fields

        private IParameterReference parameter;

        #endregion

        #region IArgumentReferenceExpression Members

        /// <summary>
        /// Parameter reference
        /// </summary>
        public IParameterReference Parameter
        {
            get { return this.parameter; }
            set { this.parameter = value; }
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

            IArgumentReferenceExpression expression = obj as IArgumentReferenceExpression;
            if (expression == null)
                return false;

            return Parameter.Equals(expression.Parameter);
        }

        public override int GetHashCode()
        {
            return this.Parameter.GetHashCode();
        }

        #endregion Object Overrides

        /// <summary>
        /// Get expression type
        /// </summary>
        public override Type GetExpressionType()
        {
            Type type = this.Parameter.Resolve().ParameterType.DotNetType;
            if (type.IsByRef) return type.GetElementType();
            return type;
        }
    }
}
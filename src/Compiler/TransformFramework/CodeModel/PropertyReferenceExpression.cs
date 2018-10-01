// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Property reference expression
    /// </summary>
    internal class XPropertyReferenceExpression : XExpression, IPropertyReferenceExpression
    {
        #region Fields

        private IPropertyReference property;
        private IExpression target;

        #endregion

        #region IPropertyReferenceExpression Members

        /// <summary>
        /// The property reference
        /// </summary>
        public IPropertyReference Property
        {
            get { return this.property; }
            set { this.property = value; }
        }

        /// <summary>
        /// The expression
        /// </summary>
        public IExpression Target
        {
            get { return this.target; }
            set { this.target = value; }
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

            IPropertyReferenceExpression expression = obj as IPropertyReferenceExpression;
            if (expression == null)
                return false;

            return
                this.Target.Equals(expression.Target) &&
                this.Property.Equals(expression.Property);
        }

        public override int GetHashCode()
        {
            return this.Property.GetHashCode();
        }

        #endregion Object Overrides

        /// <summary>
        /// Get expression type
        /// </summary>
        public override Type GetExpressionType()
        {
            return this.Property.PropertyType.DotNetType;
        }
    }
}
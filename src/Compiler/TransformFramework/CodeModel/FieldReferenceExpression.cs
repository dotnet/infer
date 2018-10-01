// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// A field reference expression
    /// </summary>
    internal class XFieldReferenceExpression : XExpression, IFieldReferenceExpression
    {
        #region Fields

        private IExpression target;
        private IFieldReference field;

        #endregion

        #region Constructor

        public XFieldReferenceExpression()
        {
        }

        #endregion

        #region IFieldReferenceExpression Members

        /// <summary>
        /// The field reference
        /// </summary>
        public IFieldReference Field
        {
            get { return this.field; }
            set
            {
                if (this.field != value)
                    this.field = value;
            }
        }

        /// <summary>
        /// The target expression
        /// </summary>
        public IExpression Target
        {
            get { return this.target; }
            set
            {
                if (this.target != value)
                    this.target = value;
            }
        }

        #endregion

        #region Object Overrides

        public override string ToString()
        {
            string s = this.Field.Name;
            if (this.Target != null) s = Target.ToString() + "." + s;
            return s;
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IFieldReferenceExpression expression = obj as IFieldReferenceExpression;
            if (expression == null)
                return false;

            if (this.Target.Equals(expression.Target) &&
                this.Field.Equals(expression.Field))
                return true;
            else
                return false;
        }

        public override int GetHashCode()
        {
            return this.Field.GetHashCode();
        }

        #endregion Object Overrides

        /// <summary>
        /// Get expression type
        /// </summary>
        public override Type GetExpressionType()
        {
            return this.Field.FieldType.DotNetType;
        }
    }
}
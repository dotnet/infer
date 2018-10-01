// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    internal class XDefaultExpression : XExpression, IDefaultExpression
    {
        private IType type;

        /// <summary>
        /// The type
        /// </summary>
        public IType Type
        {
            get { return this.type; }
            set { this.type = value; }
        }

        public override Type GetExpressionType()
        {
            return type.DotNetType;
        }

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

            IDefaultExpression stmt = obj as IDefaultExpression;
            if (stmt == null)
                return false;

            return Type.Equals(stmt.Type);
        }

        public override int GetHashCode()
        {
            return Type.GetHashCode();
        }

        #endregion Object Overrides
    }
}
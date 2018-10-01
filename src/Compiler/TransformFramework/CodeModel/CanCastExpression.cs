// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    internal class XCanCastExpression : XExpression, ICanCastExpression
    {
        public IType TargetType { get; set; }
        public IExpression Expression { get; set; }

        #region Object Overrides

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            ICanCastExpression that = obj as ICanCastExpression;
            if (that == null)
                return false;

            return this.TargetType.Equals(that.TargetType) && this.Expression.Equals(that.Expression);
        }

        public override int GetHashCode()
        {
            return Hash.Combine(TargetType.GetHashCode(), Expression.GetHashCode());
        }

        #endregion Object Overrides

        public override Type GetExpressionType()
        {
            return typeof (bool);
        }
    }
}
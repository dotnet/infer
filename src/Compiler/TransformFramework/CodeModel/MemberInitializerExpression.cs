// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    internal class XMemberInitializerExpression : XExpression, IMemberInitializerExpression
    {
        #region Fields

        private IMemberReference member;
        private IExpression expression;

        #endregion

        #region IMemberInitializerExpression Members

        public IMemberReference Member
        {
            get { return this.member; }
            set { this.member = value; }
        }

        public IExpression Value
        {
            get { return this.expression; }
            set { this.expression = value; }
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

            IMemberInitializerExpression imie = obj as IMemberInitializerExpression;
            if (imie == null)
                return false;

            return
                this.Member.Equals(imie.Member) &&
                this.Value.Equals(imie.Value);
        }

        public override int GetHashCode()
        {
            return this.Member.GetHashCode();
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
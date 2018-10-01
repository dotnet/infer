// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// A event reference expression
    /// </summary>
    internal class XEventReferenceExpression : XExpression, IEventReferenceExpression
    {
        #region Fields

        private IExpression target;
        private IEventReference theEvent;

        #endregion

        #region Constructor

        public XEventReferenceExpression()
        {
        }

        #endregion

        #region IEventReferenceExpression Members

        /// <summary>
        /// The event reference
        /// </summary>
        public IEventReference Event
        {
            get { return this.theEvent; }
            set
            {
                if (this.theEvent != value)
                    this.theEvent = value;
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
            string s = this.Event.Name;
            if (this.Target != null) s = Target.ToString() + "." + s;
            return s;
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IEventReferenceExpression expression = obj as IEventReferenceExpression;
            if (expression == null)
                return false;

            if (this.Target.Equals(expression.Target) &&
                this.Event.Equals(expression.Event))
                return true;
            else
                return false;
        }

        public override int GetHashCode()
        {
            return this.Event.GetHashCode();
        }

        #endregion Object Overrides

        /// <summary>
        /// Get expression type
        /// </summary>
        public override Type GetExpressionType()
        {
            return this.Event.EventType.DotNetType;
        }
    }
}
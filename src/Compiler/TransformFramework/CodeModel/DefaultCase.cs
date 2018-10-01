// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// The default case in a switch case
    /// </summary>
    internal class XDefaultCase : IDefaultCase
    {
        #region Fields

        private IBlockStatement body;

        #endregion

        #region ISwitchCase Members

        /// <summary>
        /// The block of statements for the default case
        /// </summary>
        public IBlockStatement Body
        {
            get { return this.body; }
            set { this.body = value; }
        }

        #endregion

        #region Object Overrides

        public override string ToString()
        {
            return Body.ToString();
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            XDefaultCase statement = obj as XDefaultCase;
            if (statement == null)
                return false;

            return this.Body.Equals(statement.Body);
        }

        public override int GetHashCode()
        {
            return this.Body.GetHashCode();
        }

        #endregion Object Overrides
    }
}
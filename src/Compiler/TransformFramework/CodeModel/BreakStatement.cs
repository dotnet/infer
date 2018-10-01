// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Break statement
    /// </summary>
    internal class XBreakStatement : XStatement, IBreakStatement
    {
        #region Object Overrides

        public override string ToString()
        {
            // Base expression will deal with this correctly
            return base.ToString();
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IBreakStatement statement = obj as IBreakStatement;
            if (statement == null)
                return false;

            return true;
        }

        public override int GetHashCode()
        {
            // hash code must be a constant due to behaviour of Equals
            return 524287;
        }

        #endregion Object Overrides
    }
}
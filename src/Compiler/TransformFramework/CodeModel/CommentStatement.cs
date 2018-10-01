// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    ///  A comment statement
    /// </summary>
    internal class XCommentStatement : ICommentStatement
    {
        #region Fields

        private IComment comment;

        #endregion

        #region ICommentStatement Members

        /// <summary>
        /// The comment
        /// </summary>
        public IComment Comment
        {
            get { return this.comment; }
            set { this.comment = value; }
        }

        #endregion

        #region Object Overrides

        public override string ToString()
        {
            return "// " + comment.ToString();
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            ICommentStatement statement = obj as ICommentStatement;
            if (statement == null) return false;
            return (Comment.Equals(statement.Comment));
        }

        public override int GetHashCode()
        {
            return this.Comment.GetHashCode();
        }

        #endregion Object Overrides
    }
}
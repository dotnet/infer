// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// A comment in the code
    /// </summary>
    internal class XComment : IComment
    {
        #region Fields

        private string text;

        #endregion

        #region IComment Members

        /// <summary>
        /// The text of the comment
        /// </summary>
        public string Text
        {
            get { return this.text; }
            set { this.text = value; }
        }

        #endregion

        #region Object Overrides

        public override string ToString()
        {
            // RTODO
            return text;
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IComment comment = obj as IComment;
            if (comment == null)
                return false;

            return this.Text.Equals(comment.Text);
        }

        public override int GetHashCode()
        {
            return this.Text.GetHashCode();
        }

        #endregion Object Overrides
    }
}
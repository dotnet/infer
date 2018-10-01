// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Base calss for all statements
    /// </summary>
    public abstract class XStatement : IStatement
    {
        /// <summary>
        /// 
        /// </summary>
        protected XStatement()
        {
        }

        /// <summary>
        /// ToString() override
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            ILanguageWriter writer = new CSharpWriter() as ILanguageWriter;
            return writer.StatementSource(this);
        }
    }
}
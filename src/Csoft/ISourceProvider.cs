// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Microsoft.ML.Probabilistic.Compiler
{
    public class SourceCode
    {
        public string FilePath { get; }
        public string SourceText { get; }

        public SourceCode(string filePath, string sourceText)
        {
            FilePath = filePath;
            SourceText = sourceText;
        }
    }

    public interface ISourceProvider
    {
        SourceCode TryGetSource(Type t);
    }
}

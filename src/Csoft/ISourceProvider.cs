// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Immutable;

namespace Microsoft.ML.Probabilistic.Compiler
{
    public class SourceFile
    {
        public string FilePath { get; }
        public string SourceText { get; }

        public SourceFile(string filePath, string sourceText)
        {
            FilePath = filePath;
            SourceText = sourceText;
        }
    }

    public class SourceCode
    {
        /// <summary>
        /// Source file that contains the primary definition.
        /// </summary>
        public SourceFile PrimaryFile { get; }
        /// <summary>
        /// Additional files that contain the definitions of some of the types referenced in the primary file.
        /// Should be used to build a more complete semantic model.
        /// </summary>
        public ImmutableArray<SourceFile> AddtionalFiles { get; }

        public SourceCode(SourceFile primaryFile, ImmutableArray<SourceFile> additionalFiles)
        {
            PrimaryFile = primaryFile;
            AddtionalFiles = additionalFiles;
        }
    }

    public interface ISourceProvider
    {
        SourceCode TryGetSource(Type t);
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Microsoft.ML.Probabilistic.Compiler
{
    public class EmbeddedResourceSourceProvider : ISourceProvider
    {
        public SourceCode TryGetSource(Type t)
        {
            bool IsPrimaryFile(SourceFile sf)
            {
                var tree = CSharpSyntaxTree.ParseText(sf.SourceText, null, sf.FilePath);
                var root = tree.GetRoot();
                var typeDecl = root.DescendantNodes()
                                 .OfType<NamespaceDeclarationSyntax>()
                                 .Where(md => md.Name.ToString().Equals(t.Namespace))
                                 .FirstOrDefault()
                                 ?.DescendantNodes()
                                 .OfType<TypeDeclarationSyntax>()
                                 .Where(md => md.Identifier.ValueText.Equals(t.Name))
                                 .FirstOrDefault();
                return typeDecl != null;
            }
            var asm = t.Assembly;
            SourceFile primaryFile = null;
            List<SourceFile> additionalFiles = new List<SourceFile>();
            foreach (var s in asm.GetManifestResourceNames().Where(s => s.EndsWith(".cs")))
            {
                SourceFile sf;
                var stream = asm.GetManifestResourceStream(s);
                using (var reader = new StreamReader(stream))
                {
                    sf = new SourceFile(s, reader.ReadToEnd());
                }
                if (primaryFile == null && IsPrimaryFile(sf))
                    primaryFile = sf;
                else
                    additionalFiles.Add(sf);
            }
            return primaryFile == null ? null : new SourceCode(primaryFile, additionalFiles.ToImmutableArray());
        }
    }
}

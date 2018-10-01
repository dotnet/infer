// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Microsoft.ML.Probabilistic.Compiler
{
    public class EmbeddedResourceSourceProvider : ISourceProvider
    {
        public SourceCode TryGetSource(Type t)
        {
            var asm = t.Assembly;
            return asm.GetManifestResourceNames().Where(s => s.EndsWith(".cs")).Select(s =>
            {
                using (var stream = asm.GetManifestResourceStream(s))
                {
                    var reader = new StreamReader(stream);
                    return new SourceCode(s, reader.ReadToEnd());
                }
            }).FirstOrDefault(code =>
            {
                var tree = CSharpSyntaxTree.ParseText(code.SourceText, null, code.FilePath);
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
            });
        }
    }
}

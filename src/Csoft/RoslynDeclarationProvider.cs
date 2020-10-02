// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler
{
    public class RoslynDeclarationProvider : IDeclarationProvider
    {
        public static RoslynDeclarationProvider Instance = new RoslynDeclarationProvider(new EmbeddedResourceSourceProvider());

        public ISourceProvider SourceProvider { get; set; }

        public RoslynDeclarationProvider(ISourceProvider sourceProvider)
        {
            SourceProvider = sourceProvider;
        }

        /// <summary>
        /// Returns TypeDeclaration for a given type. Some methods in the returned declaration may miss their bodies, because they contain unsupported constructions.
        /// </summary>
        public ITypeDeclaration GetTypeDeclaration(Type t, bool translate)
        {
            // false by default since, for some reason, excessive tracing seems to cause failure of CompilerOptions test
            bool traceFailures = false;
            // This used to be a cache peersistent between GetTypeDeclaration calls, but
            // it didn't work, because of GetTypeDeclaration's mutability
            Dictionary<string, ITypeDeclaration> declarations = new Dictionary<string, ITypeDeclaration>();
            // To be uncommented when/if ITypeDeclaration will become immutable
            //if (declarations.ContainsKey(t.FullName))
            //{
            //    return declarations[t.FullName];
            //}

            // Collecting all the assemblies, t can possibly need to compile
            var referencedAssemblies = new HashSet<Assembly>(t.Assembly.GetReferencedAssemblies().Select(Assembly.Load));
            Stack<Assembly> assemblyStack = new Stack<Assembly>();
            foreach (Assembly assembly in referencedAssemblies)
            {
                assemblyStack.Push(assembly);
            }
            while (assemblyStack.Count != 0)
            {
                Assembly assembly = assemblyStack.Pop();
                foreach (AssemblyName name in assembly.GetReferencedAssemblies())
                {
                    try
                    {
                        Assembly referenced = Assembly.Load(name);
                        if (!referencedAssemblies.Contains(referenced))
                        {
                            referencedAssemblies.Add(referenced);
                            assemblyStack.Push(referenced);
                        }
                    }
                    catch (FileNotFoundException) { } // Assembly was referenced but is not present => hopefully, not really needed
                    // Often happens when running code compiled on Windows machine on Linux with Mono
                }
            }
            var references = referencedAssemblies.Select(ra => MetadataReference.CreateFromFile(ra.Location));
            var source = SourceProvider.TryGetSource(t);
            if (source == null)
                throw new InvalidOperationException($"Cannot find source code for the type {t.Name}");
            var primaryTree = CSharpSyntaxTree.ParseText(source.PrimaryFile.SourceText, null, source.PrimaryFile.FilePath, Encoding.UTF8);
            var allTrees = source.AddtionalFiles.Select(f => CSharpSyntaxTree.ParseText(f.SourceText, null, f.FilePath, Encoding.UTF8)).ToList();
            allTrees.Add(primaryTree);
            
            var options = new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary);
            var targetAssemblyName = Path.GetFileNameWithoutExtension(primaryTree.FilePath);
            var asmName = t.Assembly.GetName();
            var pk = asmName.GetPublicKey();
            if (pk != null && pk.Length > 0)
            {
                // assembly containing target type is signed
                // spoofing name and signature (via public sign) in order to keep 'make internals visible' working
                targetAssemblyName = asmName.Name;
                options = options
                    .WithPublicSign(true)
                    .WithCryptoPublicKey(ImmutableArray.Create(pk));
            }
            var compilation = CSharpCompilation.Create(targetAssemblyName, allTrees, references, options);
            var model = compilation.GetSemanticModel(primaryTree);

            var errors = compilation.GetDiagnostics().ToList();

            // First pass that just converts structure
            var methodBodies = new DeclarationTreeBuilder(model, t.Assembly, declarations).Build();

            // Convert bodies as declarations are all now in place
            foreach (var kvp in methodBodies)
            {
                var methodDecl = kvp.Key;
                var methodSyntax = kvp.Value;
                try
                {
                    MethodBodySynthesizer.SynthesizeBody(methodDecl, methodSyntax, model, declarations);
                }
                catch (Exception e) 
                {
                    if (traceFailures)
                    {
                        var methodName = methodSyntax.Identifier.ToString();
                        Trace.TraceWarning($"Failed to synthesize the body for method {methodName}: {e}");
                    }

                    // All kinds of compilation issues can arise because we're compiling a single file rather than the project -
                    //   Referencing other classes in the same project
                    //   Non-model methods may have all kinds of syntax that we don't need to or can't support
                }
            }

            // TODO replace this hack
            declarations.Values.Where(td => td.Owner == null).ToList().ForEach(td => td.Owner = t.Assembly);

            var decl = declarations[t.FullName];
            return decl;
        }
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

//#define TestLanguageWriter

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using System.Diagnostics;
using System.IO;

namespace Microsoft.ML.Probabilistic.Compiler
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// A chain of transformers, used to apply a series of transforms.
    /// </summary>
    internal class TransformerChain
    {
        public List<CodeTransformer> transformers = new List<CodeTransformer>();

        public void AddTransform(ICodeTransform ict)
        {
            transformers.Add(new CodeTransformer(ict));
        }

        public List<ITypeDeclaration> TransformToDeclaration(ITypeDeclaration itd, AttributeRegistry<object, ICompilerAttribute> inputAttributes, bool trackTransform, bool showProgress,
                                                             out List<TransformError> warnings, bool catchExceptions = false, bool treatWarningsAsErrors = false)
        {
            List<ITypeDeclaration> res = new List<ITypeDeclaration>();
            res.Add(itd);
            AttributeRegistry<object, ICompilerAttribute> attributes = inputAttributes;
            warnings = new List<TransformError>();
            foreach (CodeTransformer ct in transformers)
            {
                string name = ct.Transform.Name;
                Stopwatch watch = null;
                if (showProgress)
                {
                    Console.Write(name + " ");
                    watch = Stopwatch.StartNew();
                }
                ct.Transform.Context.InputAttributes = attributes;
                ct.TrackTransform = trackTransform;
                if (catchExceptions)
                {
                    try
                    {
                        res = ct.TransformToDeclaration(res[0]);
                    }
                    catch (Exception ex)
                    {
                        ((BasicTransformContext) ct.Transform.Context).Error("Uncaught exception in transform", ex);
                    }
                }
                else
                {
                    res = ct.TransformToDeclaration(res[0]);
                }
                if (showProgress)
                {
                    watch.Stop();
                    Console.WriteLine("({0}ms)", watch.ElapsedMilliseconds);
                }
#if TestLanguageWriter
                try
                {
                    ILanguageWriter lw = new CSharpWriter() as ILanguageWriter;
                    lw.GenerateSource(res[0]);
                }
                catch
                {
                    Console.WriteLine("Language writer error for " + res[0].Name);
                }
#endif
                TransformResults tr = ct.Transform.Context.Results;
                tr.ThrowIfErrors(name + " failed", treatWarningsAsErrors);
                if (tr.WarningCount > 0)
                {
                    foreach (TransformError te in tr.ErrorsAndWarnings)
                    {
                        if(te.IsWarning) warnings.Add(te);
                    }
                }
                attributes = ct.Transform.Context.OutputAttributes;
            }
            return res;
        }

        // write all transform outputs to a file.  used for debugging.
        internal void WriteAllOutputs(string outputFolder)
        {
            var dir = Directory.CreateDirectory(outputFolder);
            // clean the directory
            foreach (var entry in dir.EnumerateFileSystemInfos())
            {
                entry.Delete();
            }
            LanguageWriter languageWriter = new CSharpWriter();
            int count = 0;
            foreach (var transformer in transformers)
            {
                if (transformer.OutputEqualsInput)
                    continue;
                count++;
                string filename = Path.Combine(outputFolder, count.ToString() + " " + transformer.Transform.Name + ".txt");
                using (var writer = new StreamWriter(filename))
                {
                    foreach (ITypeDeclaration itd in transformer.transformMap.Values)
                    {
                        SourceNode node = languageWriter.GeneratePartialSource(itd);
                        LanguageWriter.WriteSourceNode(writer, node);
                    }
                }
            }
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}
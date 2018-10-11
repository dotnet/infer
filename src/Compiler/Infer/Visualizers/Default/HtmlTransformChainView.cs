// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Utilities;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;

namespace Microsoft.ML.Probabilistic.Compiler.Visualizers
{
    /// <summary>
    /// Visualizer of generated source using HTML.
    /// </summary>
    internal class HtmlTransformChainView
    {
        //transformers and their file names
        private Dictionary<CodeTransformer, string> fileNames = new Dictionary<CodeTransformer, string>();

        /// <summary>
        /// Transformer, selected for the first view.
        /// </summary>
        public CodeTransformer SelectedTransformer { get; set; } = null;

        /// <summary>
        /// Add a new transformer for visualizing.
        /// </summary>
        /// <param name="transformer"></param>
        public void AddTransformer(CodeTransformer transformer)
        {
            if (fileNames.ContainsKey(transformer))
            {
                return;
            }

            fileNames.Add(transformer, GetFileName(transformer.Transform.Name));
        }

        private string GetFileName(string transformName)
        {
            return fileNames.Count + "_" + transformName + ".html";
        }

        /// <summary>
        /// Save source code in HTML files to a given folder and open them in a browser.
        /// </summary>
        /// <param name="path">The name of the folder for files.</param>
        public void Visualize(string path)
        {
            var dir = Directory.CreateDirectory(path);
            foreach (var f in dir.GetFileSystemInfos())
            {
                f.Delete();
            }

            foreach (var transformer in fileNames.Keys)
            {
                WriteFile(path, fileNames[transformer], transformer);
            }

            OpenFiles(path);
        }

        //if transformer equals null create a file without source code viewer
        private void WriteFile(string path, string fileName, CodeTransformer transformer = null)
        {
            LanguageWriter languageWriter = new CSharpWriter();
            HtmlWriteHelper htmlWriter = new HtmlWriteHelper();

            htmlWriter.WriteLine("<!DOCTYPE html>");
            htmlWriter.WriteLine();

            htmlWriter.OpenTag("head");

            //link styles
            htmlWriter.OpenTag("style", "type=\"text/css\"");
            htmlWriter.WriteLine("a:link { text-decoration: none; }");
            htmlWriter.WriteLine("a:visited { text-decoration: none; }");
            htmlWriter.WriteLine("a:active { text-decoration: none; }");
            htmlWriter.WriteLine("a:active { text-decoration: underline; color: red; }");
            htmlWriter.WriteLine("pre { margin: 0px; }");
            htmlWriter.WriteLine("code { margin: 0px; }");
            htmlWriter.CloseTag();
            htmlWriter.CloseTag();

            htmlWriter.OpenTag("body", "style=\"overflow-x: hidden; overflow-y: hidden\"");

            //main grid
            htmlWriter.OpenTag("div", "style=\"display: grid; grid-template-columns: auto auto; grid-column-gap: 10px; justify-content: start\"");

            //file names list
            // "display: grid" makes it expand to fill the vertical space
            htmlWriter.OpenTag("ol", "style=\"grid-column: 1; display: grid; overflow-y: auto;\"");

            int count = 1;
            foreach (var tuple in fileNames)
            {
                htmlWriter.OpenTag("li");
                if (tuple.Key != transformer)
                {
                    htmlWriter.OpenTag("a", $"style=\"link\" href=\"{Path.Combine(Path.GetFullPath(path), tuple.Value)}\"");
                }
                else
                {
                    htmlWriter.OpenTag("a", $"style=\"color: red\" href=\"{Path.Combine(Path.GetFullPath(path), tuple.Value)}\"");
                }
                htmlWriter.WriteLine(tuple.Key.GetFriendlyName());
                htmlWriter.CloseTag();
                htmlWriter.CloseTag();

                count++;
            }
            htmlWriter.CloseTag();
            htmlWriter.WriteLine();

            //code viewer
            // height is 99vh to make room for the scrollbar
            htmlWriter.OpenTag("div", "class=\"container\" style=\"grid-column: 2; display: grid; overflow: auto; max-height: 99vh; max-width: 100vw\"");

            //write source code
            if (transformer != null)
            {
                htmlWriter.OpenTag("pre");
                htmlWriter.OpenTag("code", "class=\"language-csharp\"");

                if (transformer.Transform.Context.Results.ErrorCount > 0)
                {
                    foreach (var error in transformer.Transform.Context.Results.errorsAndWarnings)
                    {
                        foreach(var line in StringUtil.Lines(error.ToString()))
                        {
                            htmlWriter.Append("// ");
                            htmlWriter.Append(ToHtmlContent(line));
                            htmlWriter.Append(Environment.NewLine);
                        }
                        htmlWriter.Append(Environment.NewLine);
                    }
                }

                foreach (ITypeDeclaration itd in transformer.transformMap.Values)
                {
                    using (var writer = new StringWriter())
                    {
                        SourceNode node = languageWriter.GeneratePartialSource(itd);
                        LanguageWriter.WriteSourceNode(writer, node);
                        string code = ToHtmlContent(writer.ToString().Trim());
                        htmlWriter.Append(code);
                    }
                }

                htmlWriter.CloseTag();
                htmlWriter.CloseTag();
                htmlWriter.CloseTag();
                htmlWriter.CloseTag();

                //highlighting
                htmlWriter.OpenTag("link", "rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.12.0/styles/vs2015.min.css\"", true);
                htmlWriter.OpenTag("script", "src=\"https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.12.0/highlight.min.js\"", true);
                htmlWriter.OpenTag("script");
                htmlWriter.WriteLine("hljs.initHighlightingOnLoad();");
            }
            else
            {
                htmlWriter.CloseTag();
            }

            htmlWriter.CloseTag();
            htmlWriter.CloseTag();

            htmlWriter.SaveToFile(Path.Combine(path, fileName));
        }

        private static string ToHtmlContent(string text)
        {
            return text.Replace("&", "&amp;")
                       .Replace("<", "&lt;")
                       .Replace(">", "&gt;");
        }

        private void OpenFiles(string path)
        {
            string firstOpenFileName;

            if (SelectedTransformer == null || !fileNames.ContainsKey(SelectedTransformer))
            {
                firstOpenFileName = Path.Combine(path, "origin.html");
                WriteFile(path, "origin.html");
            }
            else
            {
                firstOpenFileName = Path.Combine(path, fileNames[SelectedTransformer]);
            }

            var p = new Process();
            p.StartInfo = new ProcessStartInfo(firstOpenFileName);
            p.StartInfo.UseShellExecute = true;

            try
            {
                p.Start();
            }
            catch (Exception e)
            {
                Console.WriteLine("Problem with opening source files in HTML format");
                Console.WriteLine($"Exception message: {e.Message}");
                Console.WriteLine($"\nFiles are saved to {Path.GetFullPath(path)}");
            }
        }
    }
}

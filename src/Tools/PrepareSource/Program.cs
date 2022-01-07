// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tools.PrepareSource
{
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Xml.Linq;
    using System.Xml.XPath;

    internal static class Program
    {
        private static void Main(string[] args)
        {
            try
            {
                MainWrapped(args);
            }
            catch (Exception e)
            {
                Error(e.Message);
            }
        }

        private static void MainWrapped(string[] args)
        {
            if (args.Length != 1)
            {
                Error("Usage: {0} <src_folder>", Environment.GetCommandLineArgs()[0]);
            }

            string sourceFolder = args[0];
            string destinationFolder = args[0]; // This tool now works in place
            if (!Directory.Exists(sourceFolder))
            {
                Error("Unknown directory: {0}", sourceFolder);
            }

            var loadedDocFiles = new Dictionary<string, XDocument>();
            foreach (string sourceFileName in Directory.EnumerateFiles(sourceFolder, "*.cs", SearchOption.AllDirectories))
            {
                string temporaryFile = Path.GetRandomFileName();
                string destinationFileName = Path.Combine(destinationFolder, temporaryFile);
                ProcessFile(sourceFileName, destinationFileName, loadedDocFiles);
                File.Delete(sourceFileName);
                File.Move(destinationFileName, sourceFileName);
            }
        }

        private static void Error(string format, params object[] args)
        {
            Console.WriteLine(format, args);
            Environment.Exit(1);
        }

        private static void ProcessFile(string sourceFileName, string destinationFileName, Dictionary<string, XDocument> loadedDocFiles)
        {
            using var reader = new StreamReader(sourceFileName);
            using var writer = new StreamWriter(destinationFileName);
            string line;
            int lineNumber = 0;
            while ((line = reader.ReadLine()) != null)
            {
                ++lineNumber;

                string trimmedLine = line.Trim();
                if (!trimmedLine.StartsWith("/// <include", StringComparison.InvariantCulture))
                {
                    // Not a line with an include directive
                    writer.WriteLine(line);
                    continue;
                }

                string includeString = trimmedLine.Substring("/// ".Length);
                var includeDoc = XDocument.Parse(includeString);

                XAttribute fileAttribute = includeDoc.Root.Attribute("file");
                XAttribute pathAttribute = includeDoc.Root.Attribute("path");
                if (fileAttribute == null || pathAttribute == null)
                {
                    Error("An ill-formed include directive at {0}:{1}", sourceFileName, lineNumber);
                }

                string fullDocFileName = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(sourceFileName), fileAttribute.Value));
                XDocument docFile;
                if (!loadedDocFiles.TryGetValue(fullDocFileName, out docFile))
                {
                    docFile = XDocument.Load(fullDocFileName);
                    loadedDocFiles.Add(fullDocFileName, docFile);
                }

                XElement[] docElements = ((IEnumerable)docFile.XPathEvaluate(pathAttribute.Value)).Cast<XElement>().ToArray();
                if (docElements.Length == 0)
                {
                    Console.WriteLine("WARNING: nothing to include for the include directive at {0}:{1}", sourceFileName, lineNumber);
                }
                else
                {
                    int indexOfDocStart = line.IndexOf("/// <include", StringComparison.InvariantCulture);
                    foreach (XElement docElement in docElements)
                    {
                        string[] docElementStringLines = docElement.ToString().Split(new[] { Environment.NewLine }, StringSplitOptions.None);
                        string indentation = new string(' ', indexOfDocStart);
                        foreach (string docElementStringLine in docElementStringLines)
                        {
                            writer.WriteLine("{0}/// {1}", indentation, docElementStringLine);
                        }
                    }
                }
            }
        }
    }
}

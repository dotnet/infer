// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler.Transforms;
using System;
using System.IO;
using System.Text;

namespace Microsoft.ML.Probabilistic.Tools.BuildFactorDoc
{
    /// <summary>
    /// Cases of starting program.
    /// </summary>
    enum StartStatus { Normal, DefaultFile, NoArgument, NotExistingFile, Help }

    class Program
    {
        private static readonly string defaultXmlDocFilePath = Path.Combine("..", "..", "..", "..", "..", "Runtime", "Factors", "FactorDocs.xml");

        static int Main(string[] args)
        {
            int exitCode = 0;
            StartStatus status = GetStatus(args);

            if(status == StartStatus.Help)
            {
                ShowReference();
            }
            else if(status == StartStatus.Normal || status == StartStatus.DefaultFile)
            {
                string fileName = status == StartStatus.DefaultFile ? defaultXmlDocFilePath : args[0];
                bool isFactorExists = CheckOrUpdate(fileName);

                if (!isFactorExists)
                {
                    exitCode = 1;
                    Console.WriteLine($"A factor documentation is updated");
                }
                else
                {
                    Console.WriteLine($"A factor documentation is correct");
                }
            }
            else
            {
                ShowErrorMessage(status);
                exitCode = 2;
            }

            return exitCode;
        }

        private static StartStatus GetStatus(string[] args)
        {
            if(args.Length == 0)
            {
                if(!File.Exists(defaultXmlDocFilePath))
                {
                    return StartStatus.NoArgument;
                }

                return StartStatus.DefaultFile;
            }

            if("-h".Equals(args[0]) || "--help".Equals(args[0]))
            {
                return StartStatus.Help;
            }

            if(!File.Exists(args[0]))
            {
                return StartStatus.NotExistingFile;
            }

            return StartStatus.Normal;
        }

        private static void ShowReference()
        {
            Console.WriteLine($@"Factor builder. Updates factor documentation if it necessary.

Usage:
    Program gets a file path as a command line argument.
    If argument is not given, program uses default documentation file (if it exists):

    ""{Path.GetFullPath(defaultXmlDocFilePath)}""

    If argument equals ""-h"" or ""--help"", reference is showing.

Returns:
    0 - if factor documentation is up to date
    1 - if factor documentation has been updated
    2 - if fails");
        }

        //returns true if factor documentation is up to date
        private static bool CheckOrUpdate(string path)
        {
            string tempFile = Path.GetTempFileName();

            FactorDocumentationWriter.WriteFactorDocumentation(tempFile);
            var generated = File.ReadAllText(tempFile);
            var current = File.ReadAllText(path);
            bool flag = generated.Equals(current);

            if(!flag)
            {
                using (var writer = new StreamWriter(path, false, Encoding.UTF8))
                {
                    writer.Write(generated);
                }
            }

            File.Delete(tempFile);

            return flag;
        }

        private static void ShowErrorMessage(StartStatus status)
        {
            string msg = status switch
            {
                StartStatus.NoArgument => $"No given argument and default file (\"{Path.GetFullPath(defaultXmlDocFilePath)}\") is not found!\n",
                StartStatus.NotExistingFile => "File does not exist!\n",
                _ => "",
            };
            msg += "Give '-h' or '--help' argument to get reference";
            Console.WriteLine(msg);
        }
    }
}

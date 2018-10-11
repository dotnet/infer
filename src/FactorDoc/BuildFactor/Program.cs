// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler.Transforms;
using System;
using System.IO;

namespace Microsoft.ML.Probabilistic.FactorDocs.BuildFactor
{
    /// <summary>
    /// Cases of running program.
    /// </summary>
    enum Status { Run, NoArgument, NotExistingFile, Help, CorrectDoc, Updated }

    class Program
    {
        static int Main(string[] args)
        {
            Status status = GetStatus(args);

            if(status == Status.Help)
            {
                ShowReference();
            }
            else if(status == Status.Run)
            {
                bool isFactorExists = CheckOrUpdate(args[0]);

                if (!isFactorExists)
                {
                    status = Status.Updated;
                    Console.WriteLine($"A factor documentation is updated");
                }
                else
                {
                    status = Status.CorrectDoc;
                    Console.WriteLine($"A factor documentation is correct");
                }
            }
            else
            {
                ShowErrorMessage(status);
            }

            return GetExitCode(status);
        }

        private static Status GetStatus(string[] args)
        {
            if(args.Length == 0)
            {
                return Status.NoArgument;
            }

            if("-h".Equals(args[0]) || "--help".Equals(args[0]))
            {
                return Status.Help;
            }

            if(!File.Exists(args[0]))
            {
                return Status.NotExistingFile;
            }

            return Status.Run;
        }

        private static void ShowReference()
        {
            Console.WriteLine($@"Factor builder. Updates factor documentation if it necessary.

Usage:
    {Path.GetFileName(Environment.GetCommandLineArgs()[0])} <path to the factor documentation file>

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
            var generated = File.ReadAllLines(tempFile);
            var current = File.ReadAllLines(path);
            bool flag = true;

            for(int i = 0; i < generated.Length; i++)
            {
                if(i < current.Length)
                {
                    if(!generated[i].Equals(current[i]))
                    {
                        flag = false;
                    }
                }
                else
                {
                    flag = false;
                }

                if(!flag)
                {
                    break;
                }
            }

            if(!flag)
            {
                File.Delete(path);
                File.Move(tempFile, path);
            }

            return flag;
        }

        private static void ShowErrorMessage(Status status)
        {
            string msg;

            switch(status)
            {
                case Status.NoArgument:
                    msg = "No given argument!\n";
                    break;
                case Status.NotExistingFile:
                    msg = "File is not exist!\n";
                    break;
                default:
                    msg = "";
                    break;
            }

            msg += "Give '-h' or '--help' argument to get reference";
            Console.WriteLine(msg);
        }

        private static int GetExitCode(Status status)
        {
            if(status == Status.Help || status == Status.CorrectDoc)
            {
                return 0;
            }
            else if(status == Status.Updated)
            {
                return 1;
            }

            return 2;
        }
    }
}

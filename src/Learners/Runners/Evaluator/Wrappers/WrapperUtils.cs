// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Diagnostics;
#if !NETFRAMEWORK
    using System.Runtime.InteropServices;
#endif

    /// <summary>
    /// Contains useful utility functions for implementing wrappers.
    /// </summary>
    internal static class WrapperUtils
    {
        /// <summary>
        /// Executes a given command.
        /// </summary>
        /// <param name="command">The command to execute.</param>
        /// <exception cref="ExternalCommandExecutionException">Thrown in the command has returned non-zero exit code.</exception>
        public static void ExecuteExternalCommand(string command)
        {
            using (var process = new Process())
            {
                if (DetectedOS == OS.Windows)
                {
                    process.StartInfo.FileName = "cmd.exe";
                    process.StartInfo.Arguments = "/c " + command;
                }
                else
                {
                    process.StartInfo.FileName = "sh";
                    // triple quotes to ensure that command will be passed as a single argument
                    process.StartInfo.Arguments = $@"-c """"""{command}""""""";
                }
                process.StartInfo.UseShellExecute = false;
                process.StartInfo.RedirectStandardOutput = true;
                process.StartInfo.RedirectStandardError = true;
                process.Start();

                string outputStreamContents = process.StandardOutput.ReadToEnd();
                string errorStreamContents = process.StandardError.ReadToEnd();
                process.WaitForExit();

                if (process.ExitCode != 0)
                {
                    throw new ExternalCommandExecutionException(errorStreamContents);
                }
            }
        }

        public static OS DetectedOS
        {
            get
            {
                if (detectedOSDescription == null)
                    detectedOS = DetectOS(out detectedOSDescription);
                return detectedOS;
            }
        }

        public static string DetectedOSDescription
        {
            get
            {
                if (detectedOSDescription == null)
                    detectedOS = DetectOS(out detectedOSDescription);
                return detectedOSDescription;
            }
        }

        public enum OS { Windows, Linux, OSX, Other }

        private static OS detectedOS;
        private static string detectedOSDescription = null;
        private static OS DetectOS(out string description)
        {
#if NETFRAMEWORK
            description = Environment.OSVersion.Platform.ToString();
            switch (Environment.OSVersion.Platform)
            {
                case PlatformID.Win32NT:
                    return OS.Windows;
                case PlatformID.Unix:
                    return OS.Linux;
                case PlatformID.MacOSX:
                    return OS.OSX;
                default:
                    return OS.Other;
            }
#else
            description = RuntimeInformation.OSDescription;
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return OS.Windows;
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                return OS.Linux;
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                return OS.OSX;
            else
                return OS.Other;
#endif
        }
    }
}

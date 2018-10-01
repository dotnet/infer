// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System.Diagnostics;

    /// <summary>
    /// Tracks the name and the current line of the file being parsed.
    /// </summary>
    public class FileParsingContext
    {
        /// <summary>
        /// The name of the file.
        /// </summary>
        private readonly string fileName;

        /// <summary>
        /// The current line of the file.
        /// </summary>
        private string line;

        /// <summary>
        /// The number of the line currently being parsed.
        /// </summary>
        private int lineNumber;

        /// <summary>
        /// Initializes a new instance of the <see cref="FileParsingContext"/> class.
        /// </summary>
        /// <param name="fileName">The name of the file.</param>
        public FileParsingContext(string fileName)
        {
            this.fileName = fileName;
        }

        /// <summary>
        /// Advances context to the next line.
        /// </summary>
        /// <param name="nextLine">The next line of the file.</param>
        public void NextLine(string nextLine)
        {
            Debug.Assert(nextLine != null, "A valid file line must be specified.");

            ++this.lineNumber;
            this.line = nextLine;
        }

        /// <summary>
        /// Throws <see cref="InvalidFileFormatException"/> with given message and details about the current line of the file being parsed.
        /// </summary>
        /// <param name="format">The format string.</param>
        /// <param name="args">The arguments for the format string.</param>
        /// <exception cref="InvalidFileFormatException">Always thrown.</exception>
        public void RaiseError(string format, params object[] args)
        {
            Debug.Assert(this.lineNumber != 0, "File parsing must have started.");

            string errorMessage = string.Format(format, args);
            string fullErrorMessage = string.Format(
                "Error parsing line '{0}' at {1}:{2}. {3}", this.line, this.fileName, this.lineNumber, errorMessage);
            throw new InvalidFileFormatException(fullErrorMessage);
        }

        /// <summary>
        /// Throws <see cref="InvalidFileFormatException"/> with given message and details about the files being parsed.
        /// </summary>
        /// <param name="format">The format string.</param>
        /// <param name="args">The arguments for the format string.</param>
        /// <exception cref="InvalidFileFormatException">Always thrown.</exception>
        public void RaiseGlobalError(string format, params object[] args)
        {
            string errorMessage = string.Format(format, args);
            string fullErrorMessage = string.Format("Error parsing file {0}. {1}", this.fileName, errorMessage);
            throw new InvalidFileFormatException(fullErrorMessage);
        }
    }
}

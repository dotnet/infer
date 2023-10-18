// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;
    using System.IO;
    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// A writer that takes a delegate when writing objects.
    /// </summary>
    public class DelegatingWrappedBinaryWriter : IWriter, IDisposable
    {
        readonly BinaryWriter binaryWriter;
        readonly Func<object, string> writeObject;

        /// <summary>
        /// Initialize a new instance of <see cref="DelegatingWrappedBinaryWriter"/>.
        /// </summary>
        /// <param name="binaryWriter">A binary writer.</param>
        /// <param name="writeObject">An object formatter.</param>
        public DelegatingWrappedBinaryWriter(
            BinaryWriter binaryWriter,
            Func<object, string> writeObject)
        {
            this.binaryWriter = binaryWriter;
            this.writeObject = writeObject;
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            binaryWriter.Dispose();
        }

        /// <inheritdoc/>
        public void Write(bool value)
        {
            binaryWriter.Write(value);
        }

        /// <inheritdoc/>
        public void Write(int value)
        {
            binaryWriter.Write(value);
        }

        /// <inheritdoc/>
        public void Write(string value)
        {
            binaryWriter.Write(value);
        }

        /// <inheritdoc/>
        public void Write(double value)
        {
            binaryWriter.Write(value);
        }

        /// <inheritdoc/>
        public void Write(Guid value)
        {
            binaryWriter.Write(value);
        }

        /// <inheritdoc/>
        public void WriteObject(object value)
        {
            var objectToWrite = writeObject(value);
            Write(objectToWrite);
        }
    }
}

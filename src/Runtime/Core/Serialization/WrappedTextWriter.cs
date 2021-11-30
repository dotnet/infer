using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Microsoft.ML.Probabilistic.Serialization
{
    public class WrappedTextWriter : IWriter, IDisposable
    {
        StreamWriter writer;

        public WrappedTextWriter(StreamWriter writer)
        {
            this.writer = writer;
        }

        public void Dispose()
        {
            writer.Dispose();
        }

        public void Write(bool value)
        {
            writer.WriteLine(value);
        }

        public void Write(int value)
        {
            writer.WriteLine(value);
        }

        public void Write(string value)
        {
            writer.WriteLine(value);
        }

        public void Write(double value)
        {
            writer.WriteLine(value);
        }

        public void Write(Guid value)
        {
            writer.WriteLine(value);
        }

        public void WriteObject(object value)
        {
            if (value is string s)
            {
                writer.WriteLine(s);
            }
            else
            {
                throw new ArgumentException("value is not a string", nameof(value));
            }
        }
    }
}

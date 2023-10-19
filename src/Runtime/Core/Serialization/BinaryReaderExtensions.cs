// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Serialization
{
    using System;
    using System.IO;
    using System.Runtime.Serialization;

    using Microsoft.ML.Probabilistic.Distributions;

    using GammaArray = Microsoft.ML.Probabilistic.Distributions.DistributionStructArray<Distributions.Gamma, double>;
    using GaussianArray = Microsoft.ML.Probabilistic.Distributions.DistributionStructArray<Distributions.Gaussian, double>;
    using GaussianMatrix = Distributions.DistributionRefArray<Distributions.DistributionStructArray<Microsoft.ML.Probabilistic.Distributions.Gaussian, double>, double[]>;

    /// <summary>
    /// Provides extension methods for <see cref="IReader"/>.
    /// </summary>
    public static class BinaryReaderExtensions
    {
        /// <summary>
        /// The number of bytes of a <see cref="Guid"/>.
        /// </summary>
        private const int GuidByteCount = 16;

        /// <summary>
        /// Deserializes a <see cref="Guid"/> from a reader of a binary stream and verifies that it equals the expected <see cref="Guid"/>.
        /// </summary>
        /// <param name="reader">The reader to deserialize the <see cref="Guid"/> from.</param>
        /// <param name="expectedSerializationGuid">The expected <see cref="Guid"/>.</param>
        /// <param name="exceptionMessage">An optional message for the <see cref="SerializationException"/>.</param>
        /// <exception cref="SerializationException">If the deserialized <see cref="Guid"/> is invalid.</exception>
        public static void VerifySerializationGuid(
            this IReader reader,
            Guid expectedSerializationGuid,
            string exceptionMessage = null)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            Guid deserializedGuid = reader.ReadGuid();
            if (deserializedGuid != expectedSerializationGuid)
            {
                if (exceptionMessage == null)
                {
                    exceptionMessage = $"The deserialized guid {deserializedGuid} is invalid. Custom serialization of this object expects the guid to be {expectedSerializationGuid}.";
                }

                throw new SerializationException(exceptionMessage);
            }
        }

        /// <summary>
        /// Gets the serialization version from a reader of a binary stream and throws a <see cref="SerializationException"/> with 
        /// the specified message if the version is less than 1 or greater than <paramref name="maxPermittedSerializationVersion"/>.
        /// </summary>
        /// <param name="reader">The reader to deserialize the version from.</param>
        /// <param name="maxPermittedSerializationVersion">The maximum allowed serialization version.</param>
        /// <param name="exceptionMessage">An optional message for the <see cref="SerializationException"/>.</param>
        /// <returns>The deserialized version.</returns>
        /// <exception cref="SerializationException">
        /// If the deserialized version is less than 1 or greater than <paramref name="maxPermittedSerializationVersion"/>.
        /// </exception>
        public static int ReadSerializationVersion(this IReader reader, int maxPermittedSerializationVersion, string exceptionMessage = null)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            var deserializedVersion = reader.ReadInt32();
            if (deserializedVersion < 1 || deserializedVersion > maxPermittedSerializationVersion)
            {
                if (exceptionMessage == null)
                {
                    exceptionMessage = string.Format(
                        "The deserialized version {0} is invalid. Custom binary serialization of this object requires a version between 1 and {1}.",
                        deserializedVersion,
                        maxPermittedSerializationVersion);
                }

                throw new SerializationException(exceptionMessage);
            }

            return deserializedVersion;
        }

        /// <summary>
        /// Reads a <see cref="Guid"/> from the reader of a binary stream.
        /// </summary>
        /// <param name="reader">The reader of the binary stream.</param>
        /// <returns>The <see cref="Guid"/> constructed from the binary stream.</returns>
        public static Guid ReadGuid(this BinaryReader reader)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            return new Guid(reader.ReadBytes(GuidByteCount));
        }

        /// <summary>
        /// Reads an array of bytes from the reader of a binary stream.
        /// </summary>
        /// <param name="reader">The reader of the binary stream.</param>
        /// <returns>The byte array constructed from the binary stream.</returns>
        public static byte[] ReadByteArray(this BinaryReader reader)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            int length = reader.ReadInt32();
            if (length < 0)
            {
                throw new SerializationException("Cannot read a byte array of negative length.");
            }

            return reader.ReadBytes(length);
        }

        /// <summary>
        /// Reads an array of 4-byte signed integers from the reader of a binary stream.
        /// </summary>
        /// <param name="reader">The reader of the binary stream.</param>
        /// <returns>The integer array constructed from the binary stream.</returns>
        public static int[] ReadInt32Array(this IReader reader)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            int length = reader.ReadInt32();
            if (length < 0)
            {
                throw new SerializationException("Cannot read an integer array of negative length.");
            }

            var array = new int[length];
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = reader.ReadInt32();
            }

            return array;
        }

        /// <summary>
        /// Reads a <see cref="Gaussian"/> distribution from the reader of a binary stream.
        /// </summary>
        /// <param name="reader">The reader of the binary stream.</param>
        /// <returns>The <see cref="Gaussian"/> distribution constructed from the binary stream.</returns>
        public static Gaussian ReadGaussian(this IReader reader)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            return Gaussian.FromNatural(reader.ReadDouble(), reader.ReadDouble());
        }

        /// <summary>
        /// Reads an array of <see cref="Gaussian"/> distributions from the reader of a binary stream.
        /// </summary>
        /// <param name="reader">The reader of the binary stream.</param>
        /// <returns>The array of <see cref="Gaussian"/> distributions constructed from the binary stream.</returns>
        public static GaussianArray ReadGaussianArray(this IReader reader)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            int length = reader.ReadInt32();
            if (length < 0)
            {
                throw new SerializationException("Cannot read a Gaussian array of negative length.");
            }

            return new GaussianArray(length, i => ReadGaussian(reader));
        }

        /// <summary>
        /// Reads a jagged array of <see cref="Gaussian"/> distributions from the reader of a binary stream.
        /// </summary>
        /// <param name="reader">The reader of the binary stream.</param>
        /// <returns>The jagged array of <see cref="Gaussian"/> distributions constructed from the binary stream.</returns>
        public static GaussianMatrix ReadGaussianMatrix(this IReader reader)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            int length = reader.ReadInt32();
            if (length < 0)
            {
                throw new SerializationException("Cannot read a negative number of Gaussian arrays.");
            }

            return new GaussianMatrix(length, i => ReadGaussianArray(reader));
        }

        /// <summary>
        /// Reads a <see cref="Gamma"/> distribution from the reader of a binary stream.
        /// </summary>
        /// <param name="reader">The reader of the binary stream.</param>
        /// <returns>The <see cref="Gamma"/> distribution constructed from the binary stream.</returns>
        public static Gamma ReadGamma(this IReader reader)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            return Gamma.FromShapeAndRate(reader.ReadDouble(), reader.ReadDouble());
        }

        /// <summary>
        /// Reads an array of <see cref="Gamma"/> distributions from the reader of a binary stream.
        /// </summary>
        /// <param name="reader">The reader of the binary stream.</param>
        /// <returns>The array of <see cref="Gamma"/> distributions constructed from the binary stream.</returns>
        public static GammaArray ReadGammaArray(this IReader reader)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            int length = reader.ReadInt32();
            if (length < 0)
            {
                throw new SerializationException("Cannot read a Gamma array of negative length.");
            }

            return new GammaArray(length, i => ReadGamma(reader));
        }
    }
}

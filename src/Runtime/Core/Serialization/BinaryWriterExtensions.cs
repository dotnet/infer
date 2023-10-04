// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Serialization
{
    using System;
    using System.Collections.Generic;
    using System.IO;

    using Microsoft.ML.Probabilistic.Distributions;

    using GaussianMatrix = Distributions.DistributionRefArray<Distributions.DistributionStructArray<Microsoft.ML.Probabilistic.Distributions.Gaussian, double>, double[]>;

    /// <summary>
    /// Provides extension methods for <see cref="IReader"/>.
    /// </summary>
    public static class BinaryWriterExtensions
    {
        /// <summary>
        /// Writes the specified <see cref="Guid"/> in binary to the stream.
        /// </summary>
        /// <param name="writer">The binary writer.</param>
        /// <param name="guid">The <see cref="Guid"/> to write to the stream.</param>
        public static void Write(this BinaryWriter writer, Guid guid)
        {
            if (writer == null)
            {
                throw new ArgumentNullException(nameof(writer));
            }

            if (guid == null)
            {
                throw new ArgumentNullException(nameof(guid));
            }

            writer.Write(guid.ToByteArray()); // A guid has 16 bytes
        }

        /// <summary>
        /// Writes the specified array of integers in binary to the stream.
        /// </summary>
        /// <param name="writer">The binary writer.</param>
        /// <param name="array">The array of integers to write to the stream.</param>
        public static void Write(this IWriter writer, IList<int> array)
        {
            if (writer == null)
            {
                throw new ArgumentNullException(nameof(writer));
            }

            if (array == null)
            {
                throw new ArgumentNullException(nameof(array));
            }

            writer.Write(array.Count);
            foreach (var value in array)
            {
                writer.Write(value);
            }
        }

        /// <summary>
        /// Writes the specified <see cref="Gaussian"/> distribution in binary to the stream.
        /// </summary>
        /// <param name="writer">The binary writer.</param>
        /// <param name="distribution">The <see cref="Gaussian"/> distribution to write to the stream.</param>
        public static void Write(this IWriter writer, Gaussian distribution)
        {
            if (writer == null)
            {
                throw new ArgumentNullException(nameof(writer));
            }

            if (distribution == null)
            {
                throw new ArgumentNullException(nameof(distribution));
            }

            writer.Write(distribution.MeanTimesPrecision);
            writer.Write(distribution.Precision);
        }

        /// <summary>
        /// Writes the specified array of <see cref="Gaussian"/> distributions in binary to the stream.
        /// </summary>
        /// <param name="writer">The binary writer.</param>
        /// <param name="array">The array of <see cref="Gaussian"/> distributions to write to the stream.</param>
        public static void Write(this IWriter writer, IList<Gaussian> array)
        {
            if (writer == null)
            {
                throw new ArgumentNullException(nameof(writer));
            }

            if (array == null)
            {
                throw new ArgumentNullException(nameof(array));
            }
            
            writer.Write(array.Count);
            foreach (Gaussian element in array)
            {
                writer.Write(element);
            }
        }

        /// <summary>
        /// Writes the specified jagged array of <see cref="Gaussian"/> distributions in binary to the stream.
        /// </summary>
        /// <param name="writer">The binary writer.</param>
        /// <param name="gaussians">The jagged array of <see cref="Gaussian"/> distributions to write to the stream.</param>
        public static void Write(this IWriter writer, GaussianMatrix gaussians)
        {
            if (writer == null)
            {
                throw new ArgumentNullException(nameof(writer));
            }

            if (gaussians == null)
            {
                throw new ArgumentNullException(nameof(gaussians));
            }

            writer.Write(gaussians.Count);
            foreach (IList<Gaussian> array in gaussians)
            {
                writer.Write(array);
            }
        }

        /// <summary>
        /// Writes the specified <see cref="Gamma"/> distribution in binary to the stream.
        /// </summary>
        /// <param name="writer">The binary writer.</param>
        /// <param name="distribution">The <see cref="Gamma"/> distribution to write to the stream.</param>
        public static void Write(this IWriter writer, Gamma distribution)
        {
            if (writer == null)
            {
                throw new ArgumentNullException(nameof(writer));
            }

            if (distribution == null)
            {
                throw new ArgumentNullException(nameof(distribution));
            }

            writer.Write(distribution.Shape);
            writer.Write(distribution.Rate);
        }

        /// <summary>
        /// Writes the specified array of <see cref="Gamma"/> distributions in binary to the stream.
        /// </summary>
        /// <param name="writer">The binary writer.</param>
        /// <param name="array">The array of <see cref="Gamma"/> distributions to write to the stream.</param>
        public static void Write(this IWriter writer, IList<Gamma> array)
        {
            if (writer == null)
            {
                throw new ArgumentNullException(nameof(writer));
            }

            if (array == null)
            {
                throw new ArgumentNullException(nameof(array));
            }

            writer.Write(array.Count);
            foreach (Gamma element in array)
            {
                writer.Write(element);
            }
        }
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Reflection;
    using System.Runtime.Serialization;
    using System.Runtime.Serialization.Formatters.Binary;
    using System.Text;

    using Serialization;

    /// <summary>
    /// Implements various utilities for all learners.
    /// </summary>
    public static class Utilities
    {
        #region Serialization

        #region Save

        /// <summary>
        /// Persists an object that controls its binary serialization to a file with the specified name.
        /// </summary>
        /// <param name="obj">The object to be serialized.</param>
        /// <param name="fileName">The name of the file.</param>
        /// <param name="fileMode">The mode used to open the file. Defaults to <see cref="FileMode.Create"/>.</param>
        /// <remarks>To load a saved learner, you can use the factory methods whose names start with LoadBackwardCompatible.</remarks>
        public static void SaveForwardCompatible(this ICustomSerializable obj, string fileName, FileMode fileMode = FileMode.Create)
        {
            if (obj == null)
            {
                throw new ArgumentNullException(nameof(obj));
            }

            if (fileName == null)
            {
                throw new ArgumentNullException(nameof(fileName));
            }

            using (Stream stream = File.Open(fileName, fileMode))
            {
                if (fileName.EndsWith(".bin"))
                {
                    obj.SaveForwardCompatibleAsBinary(stream);
                }
                else
                {
                    obj.SaveForwardCompatibleAsText(stream);
                }
            }
        }

        /// <summary>
        /// Persists an object that controls its binary serialization to the specified stream.
        /// </summary>
        /// <param name="obj">The object to be serialized.</param>
        /// <param name="stream">The serialization stream.</param>
        /// <remarks>To load a saved learner, you can use the factory methods whose names start with LoadBackwardCompatible.</remarks>
        public static void SaveForwardCompatibleAsBinary(this ICustomSerializable obj, Stream stream)
        {
            if (obj == null)
            {
                throw new ArgumentNullException(nameof(obj));
            }

            if (stream == null)
            {
                throw new ArgumentNullException(nameof(stream));
            }

            using (var writer = new WrappedBinaryWriter(new BinaryWriter(stream, Encoding.UTF8, true)))
            {
                obj.SaveForwardCompatible(writer);
            }
        }

        /// <summary>
        /// Persists an object that controls its binary serialization to the specified stream.
        /// </summary>
        /// <param name="obj">The object to be serialized.</param>
        /// <param name="stream">The serialization stream.</param>
        /// <remarks>To load a saved learner, you can use the factory methods whose names start with LoadBackwardCompatible.</remarks>
        public static void SaveForwardCompatibleAsText(this ICustomSerializable obj, Stream stream)
        {
            if (obj == null)
            {
                throw new ArgumentNullException(nameof(obj));
            }

            if (stream == null)
            {
                throw new ArgumentNullException(nameof(stream));
            }

            using (var writer = new WrappedTextWriter(new StreamWriter(stream)))
            {
                obj.SaveForwardCompatible(writer);
            }
        }

        #endregion

        #endregion

        #region Batching

        /// <summary>
        /// Gets the specified batch of elements from a given array.
        /// </summary>
        /// <typeparam name="T">The type of an array element.</typeparam>
        /// <param name="batch">The zero-indexed batch to get the array elements for.</param>
        /// <param name="array">An array to get the batch of elements from.</param>
        /// <param name="totalBatchCount">The total number of batches.</param>
        /// <returns>The array elements for the specified batch.</returns>
        public static IReadOnlyList<T> GetBatch<T>(int batch, T[] array, int totalBatchCount)
        {
            if (array == null)
            {
                throw new ArgumentNullException(nameof(array));
            }

            if (totalBatchCount < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(totalBatchCount), "The total number of batches must be greater than 0.");
            }

            if (batch < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(batch), "The requested batch number must not be negative.");
            }

            if (batch >= totalBatchCount)
            {
                throw new ArgumentOutOfRangeException(nameof(batch), "The requested batch number must not be greater than the total number of batches.");
            }

            // Avoid the batching performance overhead when the data is not batched.
            if (array.Length == 0)
            {
                return array;
            }

            if (totalBatchCount == 1)
            {
                return array;
            }

            if (array.Length < totalBatchCount)
            {
                throw new ArgumentException("There are not enough elements for the requested number of batches.");
            }

            // Compute start and size of requested batch
            int batchStart = 0;
            int batchSize = 0;
            int instanceCount = array.Length;
            for (int b = 0; b <= batch; b++)
            {
                batchStart += batchSize;
                batchSize = (int)Math.Ceiling((double)instanceCount / (totalBatchCount - b));
                instanceCount -= batchSize;
            }

            return new ArraySegment<T>(array, batchStart, batchSize);
        }

        #endregion

        #region Helper methods

        /// <summary>
        /// Appends learner metadata required for version checking to a given serialization stream.
        /// </summary>
        /// <param name="type">The type of the learner.</param>
        /// <param name="stream">The serialization stream.</param>
        /// <param name="formatter">The formatter.</param>
        private static void AppendVersionMetadata(Type type, Stream stream, IFormatter formatter)
        {
            // Serializing type's assembly qualified name, because Types are not serializable on net core
            formatter.Serialize(stream, type.AssemblyQualifiedName);
            formatter.Serialize(stream, GetSerializationVersion(type));
        }

        /// <summary>
        /// Checks the current version of a learner matches the one in a given serialization stream.
        /// </summary>
        /// <param name="stream">The stream containing the required metadata.</param>
        /// <param name="formatter">The formatter.</param>
        /// <remarks>This method modifies the given stream.</remarks>
        private static void CheckVersion(Stream stream, IFormatter formatter)
        {
            if (formatter == null)
            {
                throw new ArgumentNullException(nameof(formatter));
            }

            // Reconstructing type by its serialized assembly qualified name
            var type = Type.GetType((string)formatter.Deserialize(stream));
            var expectedSerializationVersion = GetSerializationVersion(type);

            var actualSerializationVersion = (int)formatter.Deserialize(stream);

            if (expectedSerializationVersion != actualSerializationVersion)
            {
                throw new SerializationException(
                    string.Format(
                        "Serialization version mismatch. Expected: {0}, actual: {1}.",
                        expectedSerializationVersion,
                        actualSerializationVersion));
            }
        }

        /// <summary>
        /// Gets the serialization version of a learner.
        /// </summary>
        /// <param name="memberInfo">The member info of the learner.</param>
        /// <returns>The serialization version of the learner.</returns>
        private static int GetSerializationVersion(MemberInfo memberInfo)
        {
            var serializationVersionAttribute = 
                (SerializationVersionAttribute)Attribute.GetCustomAttribute(memberInfo, typeof(SerializationVersionAttribute));

            if (serializationVersionAttribute == null)
            {
                throw new SerializationException(
                    string.Format(
                        "The {0} must be applied to the learner for serialization and deserialization.",
                        typeof(SerializationVersionAttribute).Name));
            }

            return serializationVersionAttribute.SerializationVersion;
        }

        #endregion
    }
}

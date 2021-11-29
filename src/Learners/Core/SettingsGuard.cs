// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;
    using System.IO;

    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// Guards settings from being changed.
    /// </summary>
    [Serializable]
    public class SettingsGuard : ICustomSerializable
    {
        /// <summary>
        /// The current custom binary serialization version of the <see cref="SettingsGuard"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// A backward reference to the property indicating whether the setting is changeable or not.
        /// </summary>
        [NonSerialized]
        private readonly Func<bool> isImmutable;

        /// <summary>
        /// The message shown when trying to change an immutable setting.
        /// </summary>
        private readonly string message;

        /// <summary>
        /// Initializes a new instance of the <see cref="SettingsGuard"/> class.
        /// </summary>
        /// <param name="isImmutable">If true, the setting cannot be changed.</param>
        /// <param name="message">The message shown when trying to change an immutable setting.</param>
        public SettingsGuard(Func<bool> isImmutable, string message = "This setting cannot be changed.")
        {
            this.isImmutable = isImmutable;
            this.message = message;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="SettingsGuard"/> class
        /// from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The reader to load the <see cref="SettingsGuard"/> from.</param>
        /// <param name="isImmutable">If true, the setting cannot be changed.</param>
        public SettingsGuard(IReader reader, Func<bool> isImmutable)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);

            if (deserializedVersion == CustomSerializationVersion)
            {
                this.message = reader.ReadString();
                this.isImmutable = isImmutable;
            }
        }

        /// <summary>
        /// Saves the state of the <see cref="SettingsGuard"/> using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the state of the <see cref="SettingsGuard"/> to.</param>
        public void SaveForwardCompatible(IWriter writer)
        {
            writer.Write(CustomSerializationVersion);
            writer.Write(this.message);
        }

        /// <summary>
        /// Performs actions required before the value of a training setting is about to be changed.
        /// </summary>
        public void OnSettingChanging()
        {
            if (this.isImmutable != null && this.isImmutable())
            {
                throw new InvalidOperationException(this.message);
            }
        }
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Runtime.Serialization;
    using System.Text;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Learners.BayesPointMachineClassifierInternal;
    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// The Bayes point machine classifier factory.
    /// </summary>
    public static class BayesPointMachineClassifier
    {
        #region Public factory methods

        /// <summary>
        /// Creates a binary Bayes point machine classifier from a native data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        /// <returns>The binary Bayes point machine classifier instance.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, bool, Bernoulli, BayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<bool>>
            CreateBinaryClassifier<TInstanceSource, TInstance, TLabelSource>(
                IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, bool> mapping)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            return new CompoundBinaryNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource>(mapping);
        }

        /// <summary>
        /// Creates a multi-class Bayes point machine classifier from a native data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        /// <returns>The multi-class Bayes point machine classifier instance.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, int, Discrete, BayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<int>>
            CreateMulticlassClassifier<TInstanceSource, TInstance, TLabelSource>(
                IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, int> mapping)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            return new CompoundMulticlassNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource>(mapping);
        }

        /// <summary>
        /// Creates a binary Bayes point machine classifier from a standard data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="mapping">The mapping used for accessing data in the standard format.</param>
        /// <returns>The binary Bayes point machine classifier instance.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, IDictionary<TLabel, double>, BayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>
            CreateBinaryClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(
                IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, Vector> mapping)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            return new CompoundBinaryStandardDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(mapping);
        }

        /// <summary>
        /// Creates a multi-class Bayes point machine classifier from a standard data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="mapping">The mapping used for accessing data in the standard format.</param>
        /// <returns>The multi-class Bayes point machine classifier instance.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, IDictionary<TLabel, double>, BayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>
            CreateMulticlassClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(
                IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, Vector> mapping)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            return new CompoundMulticlassStandardDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(mapping);
        }

        #endregion  

        #region .Net binary deserialization

        /// <summary>
        /// Deserializes a Bayes point machine classifier from a file.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <typeparam name="TPredictionSettings">The type of the settings for prediction.</typeparam>
        /// <param name="fileName">The file name.</param>
        /// <returns>The deserialized Bayes point machine classifier object.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings>
            Load<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings>(string fileName)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
            where TPredictionSettings : IBayesPointMachineClassifierPredictionSettings<TLabel>
        {
            return Utilities.Load<IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings>>(fileName);
        }

        /// <summary>
        /// Deserializes a Bayes point machine classifier from a stream and formatter.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <typeparam name="TPredictionSettings">The type of the settings for prediction.</typeparam>
        /// <param name="stream">The stream.</param>
        /// <param name="formatter">The formatter.</param>
        /// <returns>The deserialized Bayes point machine classifier object.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings>
            Load<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings>(Stream stream, IFormatter formatter)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
            where TPredictionSettings : IBayesPointMachineClassifierPredictionSettings<TLabel>
        {
            return Utilities.Load<IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings>>(stream, formatter);
        }

        /// <summary>
        /// Deserializes a binary Bayes point machine classifier from a file.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="fileName">The file name.</param>
        /// <returns>The deserialized binary Bayes point machine classifier object.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadBinaryClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings>(string fileName)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            return Utilities.Load<IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>>(fileName);
        }

        /// <summary>
        /// Deserializes a binary Bayes point machine classifier from a stream and formatter.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="stream">The stream.</param>
        /// <param name="formatter">The formatter.</param>
        /// <returns>The deserialized binary Bayes point machine classifier object.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadBinaryClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings>(Stream stream, IFormatter formatter)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            return Utilities.Load<IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>>(stream, formatter);
        }

        /// <summary>
        /// Deserializes a multi-class Bayes point machine classifier from a file.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="fileName">The file name.</param>
        /// <returns>The deserialized multi-class Bayes point machine classifier object.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadMulticlassClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings>(string fileName)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            return Utilities.Load<IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>>(fileName);
        }

        /// <summary>
        /// Deserializes a multi-class Bayes point machine classifier from a stream and a formatter.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="stream">The stream.</param>
        /// <param name="formatter">The formatter.</param>
        /// <returns>The deserialized multi-class Bayes point machine classifier object.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadMulticlassClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings>(Stream stream, IFormatter formatter)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            return Utilities.Load<IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>>(stream, formatter);
        }

        /// <summary>
        /// Deserializes a binary Bayes point machine classifier from a file.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <param name="fileName">The file name.</param>
        /// <returns>The deserialized binary Bayes point machine classifier object.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, BayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadBinaryClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution>(string fileName)
        {
            return Utilities.Load<IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, BayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>>(fileName);
        }

        /// <summary>
        /// Deserializes a binary Bayes point machine classifier from a stream and format.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <param name="stream">The stream.</param>
        /// <param name="formatter">The formatter.</param>
        /// <returns>The deserialized binary Bayes point machine classifier object.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, BayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadBinaryClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution>(Stream stream, IFormatter formatter)
        {
            return Utilities.Load<IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, BayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>>(stream, formatter);
        }

        /// <summary>
        /// Deserializes a multi-class Bayes point machine classifier from a file.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <param name="fileName">The file name.</param>
        /// <returns>The deserialized multi-class Bayes point machine classifier object.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, BayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadMulticlassClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution>(string fileName)
        {
            return Utilities.Load<IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, BayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>>(fileName);
        }

        /// <summary>
        /// Deserializes a multi-class Bayes point machine classifier from a stream and formatter.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <param name="stream">The stream.</param>
        /// <param name="formatter">The formatter.</param>
        /// <returns>The deserialized multi-class Bayes point machine classifier object.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, BayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadMulticlassClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution>(Stream stream, IFormatter formatter)
        {
            return Utilities.Load<IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, BayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>>(stream, formatter);
        }

        #endregion

        #region Custom deserialization

        /// <summary>
        /// Deserializes a binary Bayes point machine classifier from a reader to a stream and a native data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <param name="reader">The reader to a stream of a serialized binary Bayes point machine classifier.</param>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        /// <returns>The binary Bayes point machine classifier instance.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, bool, Bernoulli, BayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<bool>>
            LoadBackwardCompatibleBinaryClassifier<TInstanceSource, TInstance, TLabelSource>(
                IReader reader, IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, bool> mapping)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            return new CompoundBinaryNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource>(reader, mapping);
        }

        /// <summary>
        /// Deserializes a binary Bayes point machine classifier from a file with the specified name and a native data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <param name="fileName">The name of the file of a serialized binary Bayes point machine classifier.</param>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        /// <returns>The binary Bayes point machine classifier instance.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, bool, Bernoulli, BayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<bool>>
            LoadBackwardCompatibleBinaryClassifier<TInstanceSource, TInstance, TLabelSource>(
                string fileName, IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, bool> mapping)
        {
            if (fileName == null)
            {
                throw new ArgumentNullException(nameof(fileName));
            }

            return WithReader(fileName, reader =>
            {
                return LoadBackwardCompatibleBinaryClassifier(reader, mapping);
            });
        }

        /// <summary>
        /// Deserializes a multi-class Bayes point machine classifier from a reader to a stream and a native data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <param name="reader">The reader to a stream of a serialized multi-class Bayes point machine classifier.</param>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        /// <returns>The multi-class Bayes point machine classifier instance.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, int, Discrete, BayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<int>>
            LoadBackwardCompatibleMulticlassClassifier<TInstanceSource, TInstance, TLabelSource>(
                IReader reader, IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, int> mapping)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            return new CompoundMulticlassNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource>(reader, mapping);
        }

        /// <summary>
        /// Deserializes a multi-class Bayes point machine classifier from a file with the specified name and a native data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <param name="fileName">The name of the file of a serialized multi-class Bayes point machine classifier.</param>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        /// <returns>The multi-class Bayes point machine classifier instance.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, int, Discrete, BayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<int>>
            LoadBackwardCompatibleMulticlassClassifier<TInstanceSource, TInstance, TLabelSource>(
                string fileName, IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, int> mapping)
        {
            if (fileName == null)
            {
                throw new ArgumentNullException(nameof(fileName));
            }

            return WithReader(fileName, reader =>
            {
                return LoadBackwardCompatibleMulticlassClassifier(reader, mapping);
            });
        }

        /// <summary>
        /// Deserializes a binary Bayes point machine classifier from a reader to a stream and a standard data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="reader">The reader to a stream of a serialized binary Bayes point machine classifier.</param>
        /// <param name="mapping">The mapping used for accessing data in the standard format.</param>
        /// <returns>The binary Bayes point machine classifier instance.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, IDictionary<TLabel, double>, BayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadBackwardCompatibleBinaryClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(
                IReader reader, IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, Vector> mapping)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            return new CompoundBinaryStandardDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(reader, mapping);
        }

        /// <summary>
        /// Deserializes a binary Bayes point machine classifier from a file with the specified name and a standard data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="fileName">The name of the file of a serialized binary Bayes point machine classifier.</param>
        /// <param name="mapping">The mapping used for accessing data in the standard format.</param>
        /// <returns>The binary Bayes point machine classifier instance.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, IDictionary<TLabel, double>, BayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadBackwardCompatibleBinaryClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(
                string fileName, IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, Vector> mapping)
        {
            if (fileName == null)
            {
                throw new ArgumentNullException(nameof(fileName));
            }

            return WithReader(fileName, reader =>
            {
                return LoadBackwardCompatibleBinaryClassifier(reader, mapping);
            });
        }

        /// <summary>
        /// Deserializes a multi-class Bayes point machine classifier from a reader to a stream and a standard data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="reader">The reader to a stream of a serialized multi-class Bayes point machine classifier.</param>
        /// <param name="mapping">The mapping used for accessing data in the standard format.</param>
        /// <returns>The multi-class Bayes point machine classifier instance.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, IDictionary<TLabel, double>, BayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadBackwardCompatibleMulticlassClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(
                IReader reader, IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, Vector> mapping)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            return new CompoundMulticlassStandardDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(reader, mapping);
        }

        /// <summary>
        /// Deserializes a multi-class Bayes point machine classifier from a file with the specified name and a standard data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="fileName">The name of the file of a serialized multi-class Bayes point machine classifier.</param>
        /// <param name="mapping">The mapping used for accessing data in the standard format.</param>
        /// <returns>The multi-class Bayes point machine classifier instance.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, IDictionary<TLabel, double>, BayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadBackwardCompatibleMulticlassClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(
                string fileName, IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, Vector> mapping)
        {
            if (fileName == null)
            {
                throw new ArgumentNullException(nameof(fileName));
            }

            return WithReader(fileName, reader =>
            {
                return LoadBackwardCompatibleMulticlassClassifier(reader, mapping);
            });
        }

        #endregion

        #region Internal factory methods

        /// <summary>
        /// Creates a binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions
        /// over factorized weights from a native data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        /// <returns>The binary Bayes point machine classifier instance.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, bool, Bernoulli, GaussianBayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<bool>>
            CreateGaussianPriorBinaryClassifier<TInstanceSource, TInstance, TLabelSource>(
                IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, bool> mapping)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            return new GaussianBinaryNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource>(mapping);
        }

        /// <summary>
        /// Creates a multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions
        /// over factorized weights from a native data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        /// <returns>The multi-class Bayes point machine classifier instance.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, int, Discrete, GaussianBayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<int>>
            CreateGaussianPriorMulticlassClassifier<TInstanceSource, TInstance, TLabelSource>(
                IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, int> mapping)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            return new GaussianMulticlassNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource>(mapping);
        }

        /// <summary>
        /// Creates a binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions
        /// over factorized weights from a standard data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="mapping">The mapping used for accessing data in the standard format.</param>
        /// <returns>The binary Bayes point machine classifier instance.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, IDictionary<TLabel, double>, GaussianBayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>
            CreateGaussianPriorBinaryClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(
                IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, Vector> mapping)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            return new GaussianBinaryStandardDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(mapping);
        }

        /// <summary>
        /// Creates a multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions
        /// over factorized weights from a standard data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="mapping">The mapping used for accessing data in the standard format.</param>
        /// <returns>The multi-class Bayes point machine classifier instance.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, IDictionary<TLabel, double>, GaussianBayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>
            CreateGaussianPriorMulticlassClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(
                IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, Vector> mapping)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            return new GaussianMulticlassStandardDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(mapping);
        }

        #endregion

        #region Internal .Net binary deserialization

        /// <summary>
        /// Deserializes a binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions
        /// over factorized weights from a file.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <param name="fileName">The file name.</param>
        /// <returns>The deserialized binary Bayes point machine classifier object.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, GaussianBayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadGaussianPriorBinaryClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution>(string fileName)
        {
            return Utilities.Load<IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, GaussianBayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>>(fileName);
        }

        /// <summary>
        /// Deserializes a binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions
        /// over factorized weights from a stream and formatter.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <param name="stream">The stream.</param>
        /// <param name="formatter">The formatter.</param>
        /// <returns>The deserialized binary Bayes point machine classifier object.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, GaussianBayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadGaussianPriorBinaryClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution>(Stream stream, IFormatter formatter)
        {
            return Utilities.Load<IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, GaussianBayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>>(stream, formatter);
        }

        /// <summary>
        /// Deserializes a multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions
        /// over factorized weights from a file.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <param name="fileName">The file name.</param>
        /// <returns>The deserialized multi-class Bayes point machine classifier object.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, GaussianBayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadBackwardCompatibleGaussianPriorMulticlassClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution>(string fileName)
        {
            return Utilities.Load<IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, GaussianBayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>>(fileName);
        }

        /// <summary>
        /// Deserializes a multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions
        /// over factorized weights from a stream and formatter.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <param name="stream">The stream.</param>
        /// <param name="formatter">The formatter.</param>
        /// <returns>The deserialized multi-class Bayes point machine classifier object.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, GaussianBayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadBackwardCompatibleGaussianPriorMulticlassClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution>(Stream stream, IFormatter formatter)
        {
            return Utilities.Load<IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, GaussianBayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>>(stream, formatter);
        }

        #endregion

        #region Internal custom deserialization

        /// <summary>
        /// Deserializes a binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions
        /// over factorized weights from a reader to a stream and a native data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <param name="reader">The reader to a stream of a serialized binary Bayes point machine classifier.</param>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        /// <returns>The binary Bayes point machine classifier instance.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, bool, Bernoulli, GaussianBayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<bool>>
            LoadBackwardCompatibleGaussianPriorBinaryClassifier<TInstanceSource, TInstance, TLabelSource>(
                IReader reader, IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, bool> mapping)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            return new GaussianBinaryNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource>(reader, mapping);
        }

        /// <summary>
        /// Deserializes a binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions
        /// over factorized weights from a file with the specified name and a native data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <param name="fileName">The name of the file of a serialized binary Bayes point machine classifier.</param>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        /// <returns>The binary Bayes point machine classifier instance.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, bool, Bernoulli, GaussianBayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<bool>>
            LoadBackwardCompatibleGaussianPriorBinaryClassifier<TInstanceSource, TInstance, TLabelSource>(
                string fileName, IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, bool> mapping)
        {
            if (fileName == null)
            {
                throw new ArgumentNullException(nameof(fileName));
            }

            return WithReader(fileName, reader =>
            {
                return LoadBackwardCompatibleGaussianPriorBinaryClassifier(reader, mapping);
            });
        }

        /// <summary>
        /// Deserializes a multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions
        /// over factorized weights from a reader to a stream and a native data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <param name="reader">The reader to a stream of a serialized multi-class Bayes point machine classifier.</param>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        /// <returns>The multi-class Bayes point machine classifier instance.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, int, Discrete, GaussianBayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<int>>
            LoadBackwardCompatibleGaussianPriorMulticlassClassifier<TInstanceSource, TInstance, TLabelSource>(
                IReader reader, IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, int> mapping)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            return new GaussianMulticlassNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource>(reader, mapping);
        }

        /// <summary>
        /// Deserializes a multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions
        /// over factorized weights from a file with the specified name and a native data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <param name="fileName">The name of the file of a serialized multi-class Bayes point machine classifier.</param>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        /// <returns>The multi-class Bayes point machine classifier instance.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, int, Discrete, GaussianBayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<int>>
            LoadBackwardCompatibleGaussianPriorMulticlassClassifier<TInstanceSource, TInstance, TLabelSource>(
                string fileName, IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, int> mapping)
        {
            if (fileName == null)
            {
                throw new ArgumentNullException(nameof(fileName));
            }

            return WithReader(fileName, reader =>
            {
                return LoadBackwardCompatibleGaussianPriorMulticlassClassifier(reader, mapping);
            });
        }

        /// <summary>
        /// Deserializes a binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions
        /// over factorized weights from a reader to a stream and a standard data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="reader">The reader to a stream of a serialized binary Bayes point machine classifier.</param>
        /// <param name="mapping">The mapping used for accessing data in the standard format.</param>
        /// <returns>The binary Bayes point machine classifier instance.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, IDictionary<TLabel, double>, GaussianBayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadBackwardCompatibleGaussianPriorBinaryClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(
                IReader reader, IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, Vector> mapping)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            return new GaussianBinaryStandardDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(reader, mapping);
        }

        /// <summary>
        /// Deserializes a binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions
        /// over factorized weights from a file with the specified name and a standard data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="fileName">The name of the file of a serialized binary Bayes point machine classifier.</param>
        /// <param name="mapping">The mapping used for accessing data in the standard format.</param>
        /// <returns>The binary Bayes point machine classifier instance.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, IDictionary<TLabel, double>, GaussianBayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadBackwardCompatibleGaussianPriorBinaryClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(
                string fileName, IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, Vector> mapping)
        {
            if (fileName == null)
            {
                throw new ArgumentNullException(nameof(fileName));
            }

            return WithReader(fileName, reader =>
            {
                return LoadBackwardCompatibleGaussianPriorBinaryClassifier(reader, mapping);
            });
        }

        /// <summary>
        /// Deserializes a multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions
        /// over factorized weights from a reader to a stream and a standard data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="reader">The reader to a stream of a serialized multi-class Bayes point machine classifier.</param>
        /// <param name="mapping">The mapping used for accessing data in the standard format.</param>
        /// <returns>The multi-class Bayes point machine classifier instance.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, IDictionary<TLabel, double>, GaussianBayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadBackwardCompatibleGaussianPriorMulticlassClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(
                IReader reader, IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, Vector> mapping)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            return new GaussianMulticlassStandardDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(reader, mapping);
        }

        /// <summary>
        /// Deserializes a multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions
        /// over factorized weights from a file with the specified name and a standard data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="fileName">The name of the file of a serialized multi-class Bayes point machine classifier.</param>
        /// <param name="mapping">The mapping used for accessing data in the standard format.</param>
        /// <returns>The multi-class Bayes point machine classifier instance.</returns>
        public static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, IDictionary<TLabel, double>, GaussianBayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>
            LoadBackwardCompatibleGaussianPriorMulticlassClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(
                string fileName, IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, Vector> mapping)
        {
            if (fileName == null)
            {
                throw new ArgumentNullException(nameof(fileName));
            }

            return WithReader(fileName, reader =>
            {
                return LoadBackwardCompatibleGaussianPriorMulticlassClassifier(reader, mapping);
            });
        }

        #endregion

        internal static T WithReader<T>(string fileName, Func<IReader, T> action)
        {
            using (var stream = File.Open(fileName, FileMode.Open))
            {
                if (fileName.EndsWith(".bin"))
                {
                    using (var reader = new WrappedBinaryReader(new BinaryReader(stream)))
                    {
                        return action(reader);
                    }
                }
                else
                {
                    using (var reader = new WrappedTextReader(new StreamReader(stream)))
                    {
                        return action(reader);
                    }
                }
            }
        }
    }
}

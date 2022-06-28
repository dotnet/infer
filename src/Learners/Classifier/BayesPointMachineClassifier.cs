// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.IO.Compression;
    using System.Linq;
    using System.Runtime.Serialization;
    using System.Text;
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Learners.BayesPointMachineClassifierInternal;
    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Learners.Runners;
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
            where TPredictionSettings : IBayesPointMachineClassifierPredictionSettings<TLabel> =>
            ReadFile(
                fileName,
                reader => LoadBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings>(reader));

        private static IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings>
            LoadBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings>(IReader reader)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
            where TPredictionSettings : IBayesPointMachineClassifierPredictionSettings<TLabel>
        {
            var type = reader.ReadString();
            switch (type)
            {
                case nameof(CompoundBinaryNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource>):
                    {
                        var mapping = LoadBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, bool>(reader);
                        return (IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings>)new CompoundBinaryNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource>(reader, mapping);
                    }
                case nameof(CompoundBinaryStandardDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>):
                    {
                        var standardMapping = LoadClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel>(reader);
                        return (IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings>)new CompoundBinaryStandardDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(reader, standardMapping);
                    }
                case nameof(CompoundMulticlassNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource>):
                    {
                        var mapping = LoadBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, int>(reader);
                        return (IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings>)new CompoundMulticlassNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource>(reader, mapping);
                    }
                case nameof(CompoundMulticlassStandardDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>):
                    {
                        var standardMapping = LoadClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel>(reader);
                        return (IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings>)new CompoundMulticlassStandardDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(reader, standardMapping);
                    }
                case nameof(GaussianBinaryNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource>):
                    {
                        var mapping = LoadBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, bool>(reader);
                        return (IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings>)new GaussianBinaryNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource>(reader, mapping);
                    }
                case nameof(GaussianBinaryStandardDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>):
                    {
                        var standardMapping = LoadClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel>(reader);
                        return (IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings>)new GaussianBinaryStandardDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(reader, standardMapping);
                    }
                case nameof(GaussianMulticlassNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource>):
                    {
                        var mapping = LoadBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, int>(reader);
                        return (IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings>)new GaussianMulticlassNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource>(reader, mapping);
                    }
                case nameof(GaussianMulticlassStandardDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>):
                    {
                        var standardMapping = LoadClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel>(reader);
                        return (IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, TPredictionSettings>)new GaussianMulticlassStandardDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel>(reader, standardMapping);
                    }
                default:
                    throw new InvalidOperationException($"Unrecognised type: {type}.");
            }
        }

        private static IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel> LoadBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel>(IReader reader)
        {
            var type = reader.ReadString();
            switch (type)
            {
                case nameof(BinaryNativeClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel>):
                    {
                        var standardMapping = LoadClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel>(reader);
                        return (IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel>)new BinaryNativeClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel>(reader, standardMapping);
                    }
                case nameof(MulticlassNativeClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel>):
                    {
                        var standardMapping = LoadClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel>(reader);
                        return (IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel>)new MulticlassNativeClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel>(reader, standardMapping);
                    }
                case nameof(MulticlassNativeBayesPointMachineClassifierTestMapping):
                    {
                        return (IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel>)new MulticlassNativeBayesPointMachineClassifierTestMapping(reader);
                    }
                case nameof(BinaryNativeBayesPointMachineClassifierTestMapping):
                    {
                        return (IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel>)new BinaryNativeBayesPointMachineClassifierTestMapping(reader);
                    }
                default:
                    throw new InvalidOperationException($"Unrecognised type: {type}.");
            }
        }

        private static IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, Vector> LoadClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel>(IReader reader)
        {
            var type = reader.ReadString();
            switch (type)
            {
                case nameof(InternalClassifierMapping):
                    return (IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, Vector>)new InternalClassifierMapping();
                case nameof(BinarySimpleTestMapping<TLabel>):
                    return (IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, Vector>)new BinarySimpleTestMapping<TLabel>();
                case nameof(MulticlassSimpleBayesPointMachineClassifierTestMapping<TLabel>):
                    return (IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, Vector>)new MulticlassSimpleBayesPointMachineClassifierTestMapping<TLabel>();
                case nameof(BinaryStandardBayesPointMachineClassifierTestMapping):
                    return (IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, Vector>)new BinaryStandardBayesPointMachineClassifierTestMapping(reader);
                case nameof(MulticlassStandardBayesPointMachineClassifierTestMapping):
                    return (IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, Vector>)new MulticlassStandardBayesPointMachineClassifierTestMapping(reader);
                case nameof(CsvGzipMapping):
                    return (IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, Vector>)new CsvGzipMapping();
                case nameof(Microsoft.ML.Probabilistic.Learners.Runners.ClassifierMapping):
                    {
                        var featureSet = new IndexedSet<string>(reader, innerReader => innerReader.ReadString());
                        return (IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, Vector>)new Microsoft.ML.Probabilistic.Learners.Runners.ClassifierMapping(featureSet);
                    }
                default:
                    throw new InvalidOperationException($"Unrecognised type: {type}.");
            }
        }

        /// <summary>
        /// The classifier mapping.
        /// </summary>
        internal class InternalClassifierMapping : ClassifierMapping<IEnumerable<IDictionary<string, double>>, IDictionary<string, double>, IEnumerable<IDictionary<string, double>>, string, Vector>
        {
            /// <summary>
            /// Provides the instances for a given instance source.
            /// </summary>
            /// <param name="instanceSource">The source of instances.</param>
            /// <returns>The instances provided by the instance source.</returns>
            /// <remarks>Assumes that the same instance source always provides the same instances.</remarks>
            public override IEnumerable<IDictionary<string, double>> GetInstances(IEnumerable<IDictionary<string, double>> instanceSource)
            {
                if (instanceSource == null)
                {
                    throw new ArgumentNullException(nameof(instanceSource));
                }

                return instanceSource;
            }

            /// <summary>
            /// Provides the features for a given instance.
            /// </summary>
            /// <param name="instance">The instance to provide features for.</param>
            /// <param name="instanceSource">An optional source of instances.</param>
            /// <returns>The features for the given instance.</returns>
            /// <remarks>Assumes that the same instance source always provides the same features for a given instance.</remarks>
            public override Vector GetFeatures(IDictionary<string, double> instance, IEnumerable<IDictionary<string, double>> instanceSource = null)
            {
                throw new NotImplementedException("Features are not required in evaluation.");
            }

            /// <summary>
            /// Provides the label for a given instance.
            /// </summary>
            /// <param name="instance">The instance to provide the label for.</param>
            /// <param name="instanceSource">An optional source of instances.</param>
            /// <param name="labelSource">An optional source of labels.</param>
            /// <returns>The label of the given instance.</returns>
            /// <remarks>Assumes that the same sources always provide the same label for a given instance.</remarks>
            public override string GetLabel(IDictionary<string, double> instance, IEnumerable<IDictionary<string, double>> instanceSource = null, IEnumerable<IDictionary<string, double>> labelSource = null)
            {
                if (instance == null)
                {
                    throw new ArgumentNullException(nameof(instance));
                }

                // Use zero-one loss function to determine point estimate (mode of distribution)
                string mode = string.Empty;
                double maximum = double.NegativeInfinity;
                foreach (var element in instance)
                {
                    if (element.Value > maximum)
                    {
                        maximum = element.Value;
                        mode = element.Key;
                    }
                }

                return mode;
            }

            /// <summary>
            /// Gets all class labels.
            /// </summary>
            /// <param name="instanceSource">An optional instance source.</param>
            /// <param name="labelSource">An optional label source.</param>
            /// <returns>All possible values of a label.</returns>
            public override IEnumerable<string> GetClassLabels(IEnumerable<IDictionary<string, double>> instanceSource = null, IEnumerable<IDictionary<string, double>> labelSource = null)
            {
                if (instanceSource == null)
                {
                    throw new ArgumentNullException(nameof(instanceSource));
                }

                return new HashSet<string>(instanceSource.SelectMany(instance => instance.Keys));
            }
        }

        internal class BinarySimpleTestMapping<TLabel> : ClassifierMapping<IReadOnlyList<Vector>, int, IReadOnlyList<TLabel>, TLabel, Vector>
        {
            public override IEnumerable<int> GetInstances(IReadOnlyList<Vector> featureVectors)
            {
                for (int instance = 0; instance < featureVectors.Count; instance++)
                {
                    yield return instance;
                }
            }

            public override Vector GetFeatures(int instance, IReadOnlyList<Vector> featureVectors)
            {
                return featureVectors[instance];
            }

            public override TLabel GetLabel(int instance, IReadOnlyList<Vector> featureVectors, IReadOnlyList<TLabel> labels)
            {
                return labels[instance];
            }

            public override IEnumerable<TLabel> GetClassLabels(
                IReadOnlyList<Vector> featureVectors = null, IReadOnlyList<TLabel> labels = null)
            {
                if (true is TLabel t && false is TLabel f)
                {
                    // This array is intentionally out of order
                    return new[] { t, f };
                }
                else if (0 is TLabel zero && 1 is TLabel one)
                {
                    // This array is intentionally out of order
                    return new[] { one, zero };
                }
                else throw new NotImplementedException();
            }
        }

        /// <summary>
        /// Represents a dataset in the standard format of the Bayes point machine classifier.
        /// </summary>
        internal class StandardDataset : IDataset
        {
            /// <summary>
            /// The label of the negative class.
            /// </summary>
            private const string NegativeClassLabel = "A";

            /// <summary>
            /// The multi-class labels.
            /// </summary>
            private Dictionary<string, string> multiclassLabels;

            /// <summary>
            /// Initializes a new instance of the <see cref="StandardDataset"/> class.
            /// </summary>
            public StandardDataset()
            {
                this.BinaryClassLabels = new[] { "False", "True" };
            }

            /// <summary>
            /// Gets the number of instances.
            /// </summary>
            public int InstanceCount
            {
                get
                {
                    if (this.FeatureVectors.Count != this.Labels.Count)
                    {
                        throw new Exception($"The number of feature vectors ({this.FeatureVectors.Count}) must be the same as the number of labels ({this.Labels.Count}).");
                    }

                    return this.FeatureVectors.Count;
                }
            }

            /// <summary>
            /// Gets or sets the feature vectors for instances.
            /// </summary>
            public Dictionary<string, Vector> FeatureVectors { get; set; }

            /// <summary>
            /// Gets or sets the multi-class labels.
            /// </summary>
            public Dictionary<string, string> Labels
            {
                get
                {
                    return this.multiclassLabels;
                }

                set
                {
                    this.SetLabels(value);
                }
            }

            /// <summary>
            /// Gets the binary labels.
            /// </summary>
            public Dictionary<string, string> BinaryLabels { get; private set; }

            /// <summary>
            /// Gets or sets the class labels of the multi-class classifier.
            /// </summary>
            public string[] ClassLabels { get; set; }

            /// <summary>
            /// Gets or sets the class labels of the binary classifier.
            /// </summary>
            public string[] BinaryClassLabels { get; set; }

            /// <summary>
            /// Sets both multi-class and binary labels.
            /// </summary>
            /// <param name="labels">The multi-class labels.</param>
            /// <remarks>
            /// The binary labels are created from the multi-class labels where
            /// the string "A" is interpreted as false and everything else as true.
            /// </remarks>
            private void SetLabels(Dictionary<string, string> labels)
            {
                this.multiclassLabels = labels;

                if (labels == null)
                {
                    this.BinaryLabels = null;
                }
                else
                {
                    this.BinaryLabels = new Dictionary<string, string>();
                    foreach (KeyValuePair<string, string> label in labels)
                    {
                        string binaryLabel = null;
                        if (label.Value != null)
                        {
                            binaryLabel = label.Value.Equals(NegativeClassLabel) ? this.BinaryClassLabels[0] : this.BinaryClassLabels[1];
                        }

                        this.BinaryLabels.Add(label.Key, binaryLabel);
                    }
                }
            }
        }

        /// <summary>
        /// An abstract base class implementation of <see cref="IClassifierMapping{TInstanceSource, TInstance, TLabelSource, TLabel, TFeatureValues}"/> 
        /// for <see cref="StandardDataset"/>.
        /// </summary>
        [Serializable]
        internal abstract class StandardBayesPointMachineClassifierTestMapping :
            ClassifierMapping<StandardDataset, string, StandardDataset, string, Vector>, ICustomSerializable
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="StandardBayesPointMachineClassifierTestMapping"/> class.
            /// </summary>
            protected StandardBayesPointMachineClassifierTestMapping()
            {
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="StandardBayesPointMachineClassifierTestMapping"/> class.
            /// </summary>
            /// <param name="reader">The reader to load the mapping from.</param>
            protected StandardBayesPointMachineClassifierTestMapping(IReader reader)
            {
            }

            /// <summary>
            /// Retrieves a list of instances from a given source.
            /// </summary>
            /// <param name="instanceSource">The source of instances.</param>
            /// <returns>The retrieved list of instances.</returns>
            public override IEnumerable<string> GetInstances(StandardDataset instanceSource)
            {
                if (instanceSource == null)
                {
                    return null;
                }

                if (instanceSource.FeatureVectors == null)
                {
                    return instanceSource.Labels == null ? null : instanceSource.Labels.Keys;
                }

                if (instanceSource.Labels == null)
                {
                    return instanceSource.FeatureVectors.Keys;
                }

                return instanceSource.FeatureVectors.Keys.Count > instanceSource.Labels.Keys.Count
                           ? (IEnumerable<string>)instanceSource.FeatureVectors.Keys
                           : instanceSource.Labels.Keys;
            }

            /// <summary>
            /// Provides the features for a given instance.
            /// </summary>
            /// <param name="instance">The instance to provide features for.</param>
            /// <param name="instanceSource">The source of instances.</param>
            /// <returns>The features for the given instance.</returns>
            public override Vector GetFeatures(string instance, StandardDataset instanceSource)
            {
                if (instance == null || instanceSource == null || instanceSource.FeatureVectors == null)
                {
                    return null;
                }

                Vector featureVector;

                if (!instanceSource.FeatureVectors.TryGetValue(instance, out featureVector))
                {
                    // Return null for test purposes
                    return null;
                }

                return featureVector;
            }

            /// <summary>
            /// Saves the state of the standard data mapping using a writer to a binary stream.
            /// </summary>
            /// <param name="writer">The writer to save the state of the standard data mapping to.</param>
            public void SaveForwardCompatible(IWriter writer)
            {
                // Nothing to serialize.
            }
        }

        /// <summary>
        /// An implementation of <see cref="IClassifierMapping{TInstanceSource, TInstance, TLabelSource, TLabel, TFeatureValues}"/> 
        /// for <see cref="StandardDataset"/> and multiple class labels.
        /// </summary>
        [Serializable]
        internal class MulticlassSimpleBayesPointMachineClassifierTestMapping<TLabel> : BinarySimpleTestMapping<TLabel>
        {
            public override IEnumerable<TLabel> GetClassLabels(
                IReadOnlyList<Vector> featureVectors = null, IReadOnlyList<TLabel> labels = null)
            {
                if (0 is TLabel zero && 1 is TLabel one && 2 is TLabel two)
                {
                    // This array is intentionally out of order
                    return new[] { one, two, zero };
                }
                else throw new NotImplementedException();
            }
        }

        /// <summary>
        /// An implementation of <see cref="IClassifierMapping{TInstanceSource, TInstance, TLabelSource, TLabel, TFeatureValues}"/> 
        /// for <see cref="StandardDataset"/> and binary class labels.
        /// </summary>
        [Serializable]
        internal class BinaryStandardBayesPointMachineClassifierTestMapping : StandardBayesPointMachineClassifierTestMapping
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="BinaryStandardBayesPointMachineClassifierTestMapping"/> class.
            /// </summary>
            public BinaryStandardBayesPointMachineClassifierTestMapping()
            {
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="BinaryStandardBayesPointMachineClassifierTestMapping"/> class.
            /// </summary>
            /// <param name="reader">The reader to load the mapping from.</param>
            public BinaryStandardBayesPointMachineClassifierTestMapping(IReader reader) : base(reader)
            {
            }

            /// <summary>
            /// Gets all class labels.
            /// </summary>
            /// <param name="instanceSource">The instance source.</param>
            /// <param name="labelSource">The label source.</param>
            /// <returns>All possible values of a label.</returns>
            public override IEnumerable<string> GetClassLabels(StandardDataset instanceSource, StandardDataset labelSource)
            {
                if (instanceSource == null)
                {
                    return null;
                }

                return instanceSource.BinaryClassLabels;
            }

            /// <summary>
            /// Provides the label for a given instance.
            /// </summary>
            /// <param name="instance">The instance to provide the label for.</param>
            /// <param name="instanceSource">The source of instances.</param>
            /// <param name="labelSource">The source of labels.</param>
            /// <returns>The label of the given instance.</returns>
            public override string GetLabel(string instance, StandardDataset instanceSource, StandardDataset labelSource)
            {
                if (instance == null || instanceSource == null || instanceSource.Labels == null)
                {
                    return null;
                }

                string label;
                if (!instanceSource.BinaryLabels.TryGetValue(instance, out label))
                {
                    // Return null for test purposes
                    return null;
                }

                return label;
            }
        }

        /// <summary>
        /// An implementation of <see cref="IClassifierMapping{TInstanceSource, TInstance, TLabelSource, TLabel, TFeatureValues}"/> 
        /// for <see cref="StandardDataset"/> and multiple class labels.
        /// </summary>
        [Serializable]
        internal class MulticlassStandardBayesPointMachineClassifierTestMapping : StandardBayesPointMachineClassifierTestMapping
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="MulticlassStandardBayesPointMachineClassifierTestMapping"/> class.
            /// </summary>
            public MulticlassStandardBayesPointMachineClassifierTestMapping()
            {
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="MulticlassStandardBayesPointMachineClassifierTestMapping"/> class.
            /// </summary>
            /// <param name="reader">The reader to load the mapping from.</param>
            public MulticlassStandardBayesPointMachineClassifierTestMapping(IReader reader) : base(reader)
            {
            }

            /// <summary>
            /// Gets all class labels.
            /// </summary>
            /// <param name="instanceSource">The instance source.</param>
            /// <param name="labelSource">The label source.</param>
            /// <returns>All possible values of a label.</returns>
            public override IEnumerable<string> GetClassLabels(StandardDataset instanceSource, StandardDataset labelSource)
            {
                if (instanceSource == null)
                {
                    return null;
                }

                return instanceSource.ClassLabels;
            }

            /// <summary>
            /// Provides the label for a given instance.
            /// </summary>
            /// <param name="instance">The instance to provide the label for.</param>
            /// <param name="instanceSource">The source of instances.</param>
            /// <param name="labelSource">The source of labels.</param>
            /// <returns>The label of the given instance.</returns>
            public override string GetLabel(string instance, StandardDataset instanceSource, StandardDataset labelSource)
            {
                if (instance == null || instanceSource == null || instanceSource.Labels == null)
                {
                    return null;
                }

                string label;
                if (!instanceSource.Labels.TryGetValue(instance, out label))
                {
                    // Return null for test purposes
                    return null;
                }

                return label;
            }
        }

        internal class CsvGzipMapping : ClassifierMapping<string, string, string, string, Vector>
        {
            public static IEnumerable<string> ReadLinesGzip(string fileName)
            {
                using (Stream stream = File.Open(fileName, FileMode.Open))
                {
                    var gz = new GZipStream(stream, CompressionMode.Decompress);
                    using (var streamReader = new StreamReader(gz))
                    {
                        while (true)
                        {
                            string line = streamReader.ReadLine();
                            if (line == null)
                                break;
                            yield return line;
                        }
                    }
                }
            }

            public override IEnumerable<string> GetInstances(string instanceSource)
            {
                return ReadLinesGzip(instanceSource);
            }

            public override Vector GetFeatures(string instance, string instanceSource = null)
            {
                var array = instance.Split(",".ToCharArray()).Skip(1).Select(Convert.ToDouble).ToArray();
                return Vector.FromArray(array);
            }

            public override string GetLabel(string instance, string instanceSource = null, string labelSource = null)
            {
                return instance.Split(",".ToCharArray()).First();
            }

            public override IEnumerable<string> GetClassLabels(string instanceSource, string labelSource = null)
            {
                return ReadLinesGzip(instanceSource).Select(l => l.Split(",".ToCharArray()).First()).Distinct();
            }
        }

        /// <summary>
        /// An implementation of <see cref="IBayesPointMachineClassifierMapping{TInstanceSource, TInstance, TLabelSource, TLabel}"/> 
        /// for <see cref="NativeDataset"/> and <see cref="bool"/> labels.
        /// </summary>
        [Serializable]
        internal class BinaryNativeBayesPointMachineClassifierTestMapping :
            NativeBayesPointMachineClassifierTestMapping<bool>
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="BinaryNativeBayesPointMachineClassifierTestMapping"/> class.
            /// </summary>
            public BinaryNativeBayesPointMachineClassifierTestMapping()
            {
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="BinaryNativeBayesPointMachineClassifierTestMapping"/> class.
            /// </summary>
            /// <param name="reader">The reader to load the mapping from.</param>
            public BinaryNativeBayesPointMachineClassifierTestMapping(IReader reader) : base(reader)
            {
            }

            /// <summary>
            /// Provides the labels of all instances from the specified batch of the instance source.
            /// </summary>
            /// <param name="instanceSource">The instance source.</param>
            /// <param name="labelSource">A label source.</param>
            /// <param name="batchNumber">An optional batch number. Defaults to 0 and is used only if the instance and label sources are divided into batches.</param>
            /// <returns>The labels provided by the specified batch of the sources.</returns>
            public override bool[] GetLabels(NativeDataset instanceSource, NativeDataset labelSource, int batchNumber = 0)
            {
                return instanceSource == null || instanceSource.BinaryLabels == null ? null : Utilities.GetBatch(batchNumber, instanceSource.BinaryLabels, this.BatchCount).ToArray();
            }
        }

        /// <summary>
        /// An implementation of <see cref="IBayesPointMachineClassifierMapping{TInstanceSource, TInstance, TLabelSource, TLabel}"/> 
        /// for <see cref="NativeDataset"/> and <see cref="int"/> labels.
        /// </summary>
        [Serializable]
        internal class MulticlassNativeBayesPointMachineClassifierTestMapping :
            NativeBayesPointMachineClassifierTestMapping<int>
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="MulticlassNativeBayesPointMachineClassifierTestMapping"/> class.
            /// </summary>
            public MulticlassNativeBayesPointMachineClassifierTestMapping()
            {
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="MulticlassNativeBayesPointMachineClassifierTestMapping"/> class.
            /// </summary>
            /// <param name="reader">The reader to load the mapping from.</param>
            public MulticlassNativeBayesPointMachineClassifierTestMapping(IReader reader) : base(reader)
            {
            }

            /// <summary>
            /// Provides the labels of all instances from the specified batch of the instance source.
            /// </summary>
            /// <param name="instanceSource">The instance source.</param>
            /// <param name="labelSource">A label source.</param>
            /// <param name="batchNumber">An optional batch number. Defaults to 0 and is used only if the instance and label sources are divided into batches.</param>
            /// <returns>The labels provided by the specified batch of the sources.</returns>
            public override int[] GetLabels(NativeDataset instanceSource, NativeDataset labelSource, int batchNumber = 0)
            {
                return instanceSource == null || instanceSource.Labels == null ? null : Utilities.GetBatch(batchNumber, instanceSource.Labels, this.BatchCount).ToArray();
            }
        }

        /// <summary>
        /// Interface to a dataset.
        /// </summary>
        internal interface IDataset
        {
            /// <summary>
            /// Gets the number of instances of the dataset.
            /// </summary>
            int InstanceCount { get; }
        }

        /// <summary>
        /// Represents a dataset in the native format of the Bayes point machine classifier.
        /// </summary>
        internal class NativeDataset : IDataset
        {
            /// <summary>
            /// The label of the negative class.
            /// </summary>
            private const int NegativeClassLabel = 0;

            /// <summary>
            /// The multi-class labels.
            /// </summary>
            private int[] multiclassLabels;

            /// <summary>
            /// Gets or sets a value indicating whether the features are in a sparse or a dense representation.
            /// </summary>
            public bool IsSparse { get; set; }

            /// <summary>
            /// Gets the number of instances.
            /// </summary>
            public int InstanceCount
            {
                get
                {
                    if (this.FeatureValues.Length != this.Labels.Length)
                    {
                        throw new Exception($"There should be a matching number of features ({this.FeatureValues.Length}) and labels ({this.Labels.Length}).");
                    }

                    return this.FeatureValues.Length;
                }
            }

            /// <summary>
            /// Gets or sets the total number of features.
            /// </summary>
            public int FeatureCount { get; set; }

            /// <summary>
            /// Gets or sets the number of classes of the classifier.
            /// </summary>
            public int ClassCount { get; set; }

            /// <summary>
            /// Gets or sets the feature values.
            /// </summary>
            public double[][] FeatureValues { get; set; }

            /// <summary>
            /// Gets or sets the feature indexes.
            /// </summary>
            public int[][] FeatureIndexes { get; set; }

            /// <summary>
            /// Gets or sets the multi-class labels.
            /// </summary>
            public int[] Labels
            {
                get
                {
                    return this.multiclassLabels;
                }

                set
                {
                    this.SetLabels(value);
                }
            }

            /// <summary>
            /// Gets the binary labels.
            /// </summary>
            public bool[] BinaryLabels { get; private set; }

            /// <summary>
            /// Sets both multi-class and binary labels.
            /// </summary>
            /// <param name="labels">The multi-class labels.</param>
            /// <remarks>
            /// The binary labels are created from the multi-class labels 
            /// where the value 0 is interpreted as false and everything else as true.
            /// </remarks>
            private void SetLabels(int[] labels)
            {
                this.multiclassLabels = labels;

                if (labels == null)
                {
                    this.BinaryLabels = null;
                }
                else
                {
                    this.BinaryLabels = new bool[labels.Length];
                    for (int i = 0; i < labels.Length; i++)
                    {
                        if (labels[i] != NegativeClassLabel)
                        {
                            this.BinaryLabels[i] = true;
                        }
                        else
                        {
                            this.BinaryLabels[i] = false;
                        }
                    }
                }
            }
        }

        /// <summary>
        /// An abstract base class implementation of <see cref="IBayesPointMachineClassifierMapping{TInstanceSource, TInstance, TLabelSource, TLabel}"/> 
        /// for <see cref="NativeDataset"/>.
        /// </summary>
        /// <typeparam name="TLabel">The type of a label in native data format.</typeparam>
        [Serializable]
        internal abstract class NativeBayesPointMachineClassifierTestMapping<TLabel>
            : IBayesPointMachineClassifierMapping<NativeDataset, int, NativeDataset, TLabel>, ICustomSerializable
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="NativeBayesPointMachineClassifierTestMapping{TLabel}"/> class.
            /// </summary>
            protected NativeBayesPointMachineClassifierTestMapping()
            {
                this.BatchCount = 1;
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="NativeBayesPointMachineClassifierTestMapping{TLabel}"/> class.
            /// </summary>
            /// <param name="reader">The reader to load the mapping from.</param>
            protected NativeBayesPointMachineClassifierTestMapping(IReader reader)
            {
                this.BatchCount = reader.ReadInt32();
            }

            /// <summary>
            /// Gets or sets the number of batches.
            /// </summary>
            public int BatchCount { get; set; }

            /// <summary>
            /// Indicates whether the feature representation provided by the instance source is sparse or dense.
            /// </summary>
            /// <param name="instanceSource">The instance source.</param>
            /// <returns>True, if the feature representation is sparse and false if it is dense.</returns>
            public bool IsSparse(NativeDataset instanceSource)
            {
                return instanceSource.IsSparse;
            }

            /// <summary>
            /// Provides the total number of features for the specified instance source.
            /// </summary>
            /// <param name="instanceSource">The instance source.</param>
            /// <returns>The total number of features.</returns>
            public int GetFeatureCount(NativeDataset instanceSource)
            {
                return instanceSource.FeatureCount;
            }

            /// <summary>
            /// Provides the number of classes that the Bayes point machine classifier is used for.
            /// </summary>
            /// <param name="instanceSource">The instance source.</param>
            /// <param name="labelSource">The label source.</param>
            /// <returns>The number of classes that the Bayes point machine classifier is used for.</returns>
            public int GetClassCount(NativeDataset instanceSource, NativeDataset labelSource)
            {
                return instanceSource.ClassCount;
            }

            /// <summary>
            /// Provides the feature values for a specified instance.
            /// </summary>
            /// <param name="instance">The instance.</param>
            /// <param name="instanceSource">The instance source.</param>
            /// <returns>The feature values for the specified instance.</returns>
            public double[] GetFeatureValues(int instance, NativeDataset instanceSource)
            {
                if (instanceSource == null || instanceSource.FeatureValues == null)
                {
                    return null;
                }

                return instanceSource.FeatureValues[instance];
            }

            /// <summary>
            /// Provides the feature indexes for a specified instance.
            /// </summary>
            /// <param name="instance">The instance.</param>
            /// <param name="instanceSource">The instance source.</param>
            /// <returns>The feature indexes for the specified instance. Null if feature values are in a dense representation.</returns>
            public int[] GetFeatureIndexes(int instance, NativeDataset instanceSource)
            {
                if (instanceSource == null || instanceSource.FeatureIndexes == null)
                {
                    return null;
                }

                return instanceSource.FeatureIndexes[instance];
            }

            /// <summary>
            /// Provides the feature values of all instances from the specified batch of the instance source.
            /// </summary>
            /// <param name="instanceSource">The instance source.</param>
            /// <param name="batchNumber">An optional batch number. Defaults to 0 and is used only if the instance source is divided into batches.</param>
            /// <returns>The feature values provided by the specified batch of the instance source.</returns>
            public double[][] GetFeatureValues(NativeDataset instanceSource, int batchNumber = 0)
            {
                return instanceSource == null || instanceSource.FeatureValues == null ? null : Utilities.GetBatch(batchNumber, instanceSource.FeatureValues, this.BatchCount).ToArray();
            }

            /// <summary>
            /// Provides the feature indexes of all instances from the specified batch of the instance source.
            /// </summary>
            /// <param name="instanceSource">The instance source.</param>
            /// <param name="batchNumber">An optional batch number. Defaults to 0 and is used only if the instance source is divided into batches.</param>
            /// <returns>
            /// The feature indexes provided by the specified batch of the instance source. Null if feature values are in a dense representation.
            /// </returns>
            public int[][] GetFeatureIndexes(NativeDataset instanceSource, int batchNumber = 0)
            {
                return instanceSource == null || instanceSource.FeatureIndexes == null ? null : Utilities.GetBatch(batchNumber, instanceSource.FeatureIndexes, this.BatchCount).ToArray();
            }

            /// <summary>
            /// Provides the labels of all instances from the specified batch of the instance source.
            /// </summary>
            /// <param name="instanceSource">The instance source.</param>
            /// <param name="labelSource">A label source.</param>
            /// <param name="batchNumber">An optional batch number. Defaults to 0 and is used only if the instance and label sources are divided into batches.</param>
            /// <returns>The labels provided by the specified batch of the sources.</returns>
            public abstract TLabel[] GetLabels(NativeDataset instanceSource, NativeDataset labelSource, int batchNumber = 0);

            /// <summary>
            /// Saves the state of the native data mapping using a writer to a binary stream.
            /// </summary>
            /// <param name="writer">The writer to save the state of the native data mapping to.</param>
            public void SaveForwardCompatible(IWriter writer)
            {
                writer.Write(this.BatchCount);
            }
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
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings =>
            ReadFile(
                fileName,
                reader => LoadBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>(reader));

        private static T ReadFile<T>(string fileName, Func<IReader, T> read)
        {
            if (fileName == null)
            {
                throw new ArgumentNullException(nameof(fileName));
            }

            using (Stream stream = File.Open(fileName, FileMode.Open))
            using (var reader = new StreamReader(stream))
            {
                var textReader = new WrappedTextReader(reader);
                return read(textReader);
            }
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
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings =>
            ReadFile(
                fileName,
                reader => LoadBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>(reader));

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
            LoadBinaryClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution>(string fileName) =>
            ReadFile(
                fileName,
                reader => LoadBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, BayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>(reader));

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
            LoadMulticlassClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution>(string fileName) =>
            ReadFile(
                fileName,
                reader => LoadBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, BayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>(reader));

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
            LoadGaussianPriorBinaryClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution>(string fileName) =>
            ReadFile(
                fileName,
                reader => LoadBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, GaussianBayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>(reader));

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
            LoadBackwardCompatibleGaussianPriorMulticlassClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution>(string fileName) =>
            ReadFile(
                fileName,
                reader => LoadBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, GaussianBayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>(reader));

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
                    throw new InvalidOperationException("Binary format is no longer supported. Please use a previous version of Infer.NET.");
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

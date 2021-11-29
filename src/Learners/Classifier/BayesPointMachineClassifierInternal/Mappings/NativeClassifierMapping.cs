// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.BayesPointMachineClassifierInternal
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// An abstract base class for the data mapping into the native format of the Bayes point machine classifier. 
    /// Chained with the data mapping into the standard format.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of the instance source.</typeparam>
    /// <typeparam name="TInstance">The type of an instance.</typeparam>
    /// <typeparam name="TLabelSource">The type of the label source.</typeparam>
    /// <typeparam name="TStandardLabel">The type of a label in standard data format.</typeparam>
    /// <typeparam name="TNativeLabel">The type of a label in native data format.</typeparam>
    [Serializable]
    internal abstract class NativeClassifierMapping<TInstanceSource, TInstance, TLabelSource, TStandardLabel, TNativeLabel> 
        : IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, TNativeLabel>, ICustomSerializable
    {
        #region Fields, constructors, properties

        /// <summary>
        /// The current custom binary serialization version of the
        /// <see cref="NativeClassifierMapping{TInstanceSource,TInstance,TLabelSource,TStandardLabel,TNativeLabel}"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// If true, the features provided by the standard format are in a sparse representation.
        /// </summary>
        private bool isSparseFeatureRepresentation;

        /// <summary>
        /// The total number of features in either feature representation.
        /// </summary>
        private int featureCount;

        /// <summary>
        /// The number of batches the instance source is split into.
        /// </summary>
        private int batchCount;

        #region Caching

        /// <summary>
        /// The last batch number seen in training or prediction.
        /// </summary>
        private int lastBatchNumber;

        /// <summary>
        /// The last number of batches the instance source has been split into.
        /// </summary>
        private int lastBatchCount;

        /// <summary>
        /// The last instance source seen in training or prediction.
        /// </summary>
        [NonSerialized]
        private TInstanceSource lastInstanceSource;

        /// <summary>
        /// The last label source seen in training.
        /// </summary>
        [NonSerialized]
        private TLabelSource lastLabelSource;

        /// <summary>
        /// The cached instances.
        /// </summary>
        [NonSerialized]
        private TInstance[] instances;

        /// <summary>
        /// The cached feature values.
        /// </summary>
        [NonSerialized]
        private double[][] featureValues;

        /// <summary>
        /// The cached feature indexes.
        /// </summary>
        [NonSerialized]
        private int[][] featureIndexes;

        /// <summary>
        /// The cached labels.
        /// </summary>
        [NonSerialized]
        private TNativeLabel[] labels;

        /// <summary>
        /// True, if the features provided by the instance source are in the cache.
        /// </summary>
        [NonSerialized]
        private bool featuresCached;

        /// <summary>
        /// True, if the labels provided by the instance source are in the cache.
        /// </summary>
        [NonSerialized]
        private bool labelsCached;

        #endregion

        /// <summary>
        /// Initializes a new instance of the <see cref="NativeClassifierMapping{TInstanceSource,TInstance,TLabelSource,TStandardLabel,TNativeLabel}"/> class.
        /// </summary>
        /// <param name="standardMapping">The mapping for accessing data in standard format.</param>
        protected NativeClassifierMapping(IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TStandardLabel, Vector> standardMapping)
        {
            this.StandardMapping = standardMapping;
            this.batchCount = 1;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="NativeClassifierMapping{TInstanceSource,TInstance,TLabelSource,TStandardLabel,TNativeLabel}"/> 
        /// class from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the mapping from.</param>
        /// <param name="standardMapping">The mapping for accessing data in standard format.</param>
        protected NativeClassifierMapping(IReader reader, IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TStandardLabel, Vector> standardMapping)
        {
            this.StandardMapping = standardMapping;

            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);

            if (deserializedVersion == CustomSerializationVersion)
            {
                this.isSparseFeatureRepresentation = reader.ReadBoolean();
                this.featureCount = reader.ReadInt32();
                this.batchCount = reader.ReadInt32();
                this.lastBatchCount = reader.ReadInt32();
                this.lastBatchNumber = reader.ReadInt32();
            }
        }

        /// <summary>
        /// Gets the data mapping into the standard format.
        /// </summary>
        protected IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TStandardLabel, Vector> StandardMapping { get; private set; }

        #endregion

        #region IBayesPointMachineClassifierMapping implementation

        /// <summary>
        /// Indicates whether the feature representation provided by this mapping is sparse or dense.
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <returns>True, if the feature representation is sparse and false if it is dense.</returns>
        public bool IsSparse(TInstanceSource instanceSource)
        {
            Debug.Assert(instanceSource != null, "The instance source must not be null.");
            
            this.CacheFeatures(instanceSource);
            return this.isSparseFeatureRepresentation;
        }

        /// <summary>
        /// Provides the total number of features for the specified instance source.
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <returns>The total number of features.</returns>
        public int GetFeatureCount(TInstanceSource instanceSource)
        {
            Debug.Assert(instanceSource != null, "The instance source must not be null.");

            this.CacheFeatures(instanceSource);
            return this.featureCount;
        }

        /// <summary>
        /// Provides the number of classes that the Bayes point machine classifier is used for.
        /// </summary>
        /// <param name="instanceSource">An optional instance source.</param>
        /// <param name="labelSource">An optional label source.</param>
        /// <returns>The number of classes that the Bayes point machine classifier is used for.</returns>
        public abstract int GetClassCount(TInstanceSource instanceSource = default(TInstanceSource), TLabelSource labelSource = default(TLabelSource));

        /// <summary>
        /// Provides the feature values for a specified instance.
        /// </summary>
        /// <param name="instance">The instance.</param>
        /// <param name="instanceSource">An optional instance source.</param>
        /// <returns>The feature values for the specified instance.</returns>
        public double[] GetFeatureValues(TInstance instance, TInstanceSource instanceSource = default(TInstanceSource))
        {
            Debug.Assert(instance != null, "The instance must not be null.");

            return this.GetFeatureValues(this.StandardMapping.GetFeaturesSafe(instance, instanceSource));
        }

        /// <summary>
        /// Provides the feature indexes for a specified instance.
        /// </summary>
        /// <param name="instance">The instance.</param>
        /// <param name="instanceSource">An optional instance source.</param>
        /// <returns>The feature indexes for the specified instance. Null if feature values are in a dense representation.</returns>
        public int[] GetFeatureIndexes(TInstance instance, TInstanceSource instanceSource = default(TInstanceSource))
        {
            Debug.Assert(instance != null, "The instance must not be null.");

            return this.GetFeatureIndexes(this.StandardMapping.GetFeaturesSafe(instance, instanceSource));
        }

        /// <summary>
        /// Provides the feature values of all instances from the specified batch of the instance source.
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="batchNumber">An optional batch number. Defaults to 0 and is used only if the instance source is divided into batches.</param>
        /// <returns>The feature values provided by the specified batch of the instance source.</returns>
        public double[][] GetFeatureValues(TInstanceSource instanceSource, int batchNumber = 0)
        {
            Debug.Assert(instanceSource != null, "The instance source must not be null.");

            this.CacheFeatures(instanceSource, batchNumber);
            return this.featureValues;
        }

        /// <summary>
        /// Provides the feature indexes of all instances from the specified batch of the instance source.
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="batchNumber">An optional batch number. Defaults to 0 and is used only if the instance source is divided into batches.</param>
        /// <returns>
        /// The feature indexes provided by the specified batch of the instance source. Null if feature values are in a dense representation.
        /// </returns>
        public int[][] GetFeatureIndexes(TInstanceSource instanceSource, int batchNumber = 0)
        {
            Debug.Assert(instanceSource != null, "The instance source must not be null.");

            this.CacheFeatures(instanceSource, batchNumber);
            return this.featureIndexes;
        }

        /// <summary>
        /// Provides the labels of all instances from the specified batch of the instance source.
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">An optional label source.</param>
        /// <param name="batchNumber">An optional batch number. Defaults to 0 and is used only if the instance and label sources are divided into batches.</param>
        /// <returns>The labels provided by the specified batch of the sources.</returns>
        public TNativeLabel[] GetLabels(TInstanceSource instanceSource, TLabelSource labelSource = default(TLabelSource), int batchNumber = 0)
        {
            Debug.Assert(instanceSource != null, "The instance source must not be null.");

            this.CacheLabels(instanceSource, labelSource, batchNumber);
            return this.labels;
        }

        #endregion

        #region ICustomSerializable implementation

        /// <summary>
        /// Saves the state of the data mapping using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the state of the data mapping to.</param>
        public virtual void SaveForwardCompatible(IWriter writer)
        {
            writer.Write(CustomSerializationVersion);
            writer.Write(this.isSparseFeatureRepresentation);
            writer.Write(this.featureCount);
            writer.Write(this.batchCount);
            writer.Write(this.lastBatchCount);
            writer.Write(this.lastBatchNumber);
        }

        #endregion

        #region Helper methods

        /// <summary>
        /// Gets or sets the number of batches the instance source is split into.
        /// </summary>
        /// <param name="value">The number of batches to use.</param>
        public void SetBatchCount(int value)
        {
            this.batchCount = value;
        }

        /// <summary>
        /// Gets the labels for the specified instances in native data format.
        /// </summary>
        /// <param name="instances">The instances to get the labels for.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">An optional label source.</param>
        /// <returns>The labels in native data format.</returns>
        protected abstract TNativeLabel[] GetNativeLabels(
            IReadOnlyList<TInstance> instances,
            TInstanceSource instanceSource,
            TLabelSource labelSource = default(TLabelSource));

        /// <summary>
        /// Gets the array of non-zero values in a given vector.
        /// </summary>
        /// <param name="vector">The vector to extract non-zero values from.</param>
        /// <returns>The array of non-zero values.</returns>
        private static double[] GetNonZeroValues(Vector vector)
        {
            return vector.FindAll(x => x != 0.0).Select(vi => vi.Value).ToArray();
        }

        /// <summary>
        /// Gets the array of non-zero value indexes in a given vector.
        /// </summary>
        /// <param name="vector">The vector to extract non-zero value indexes from.</param>
        /// <returns>The array of non-zero value indexes.</returns>
        private static int[] GetNonZeroValueIndexes(Vector vector)
        {
            return vector.IndexOfAll(x => x != 0.0).ToArray();
        }

        /// <summary>
        /// Returns true if the specified batch of the instance source is currently cached and false otherwise.
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="batchNumber">The batch number.</param>
        /// <returns>True if the specified instance source is currently cached and false otherwise.</returns>
        private bool IsInstanceSourceCached(TInstanceSource instanceSource, int batchNumber)
        {
            Debug.Assert(instanceSource != null, "The instance source must not be null.");
            Debug.Assert(batchNumber < this.batchCount, "The batch number must be smaller than the total number of batches.");
            Debug.Assert(batchNumber >= 0, "The batch number must not be negative.");

            return ReferenceEquals(instanceSource, this.lastInstanceSource) 
                && batchNumber == this.lastBatchNumber 
                && this.batchCount == this.lastBatchCount;
        }

        /// <summary>
        /// Returns true if the specified batch of the label source is currently cached and false otherwise.
        /// </summary>
        /// <param name="labelSource">The label source.</param>
        /// <param name="batchNumber">The batch number.</param>
        /// <returns>True if the specified instance source is currently cached and false otherwise.</returns>
        private bool IsLabelSourceCached(TLabelSource labelSource, int batchNumber)
        {
            Debug.Assert(labelSource != null, "The label source must not be null.");
            Debug.Assert(batchNumber < this.batchCount, "The batch number must be smaller than the total number of batches.");
            Debug.Assert(batchNumber >= 0, "The batch number must not be negative.");

            return ReferenceEquals(labelSource, this.lastLabelSource) && batchNumber == this.lastBatchNumber;
        }

        /// <summary>
        /// Gets the features values from a given feature vector.
        /// </summary>
        /// <param name="featureVector">The feature vector to extract the feature values from.</param>
        /// <returns>The array of feature values.</returns>
        private double[] GetFeatureValues(Vector featureVector)
        {
            if (featureVector.Count != this.featureCount)
            {
                throw new BayesPointMachineClassifierException("The feature vector has an inconsistent number of features.");
            }

            return this.isSparseFeatureRepresentation ? GetNonZeroValues(featureVector) : featureVector.ToArray();
        }

        /// <summary>
        /// Gets the features indexes from a given feature vector.
        /// </summary>
        /// <param name="featureVector">The feature vector to extract the feature indexes from.</param>
        /// <returns>The array of feature indexes.</returns>
        private int[] GetFeatureIndexes(Vector featureVector)
        {
            if (featureVector.Count != this.featureCount)
            {
                throw new BayesPointMachineClassifierException("The feature vector has an inconsistent number of features.");
            }

            return this.isSparseFeatureRepresentation ? GetNonZeroValueIndexes(featureVector) : null;
        }

        /// <summary>
        /// Caches all features provided by the instance source in the native format.
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="batchNumber">
        /// An optional batch number. Defaults to 0 and is used only if the instance source is divided into batches.
        /// </param>
        /// <remarks>Call this method every time before accessing feature data.</remarks>
        private void CacheFeatures(TInstanceSource instanceSource, int batchNumber = 0)
        {
            Debug.Assert(instanceSource != null, "The instance source must not be null.");

            if (!this.IsInstanceSourceCached(instanceSource, batchNumber))
            {
                this.featuresCached = false;
                this.labelsCached = false;

                // Cache instances
                this.instances = this.StandardMapping.GetInstancesSafe(instanceSource).ToArray();

                // Cache all feature vectors in native format
                this.CacheAllFeatureVectors(Utilities.GetBatch(batchNumber, this.instances, this.batchCount), instanceSource);

                this.lastBatchNumber = batchNumber;
                this.lastBatchCount = this.batchCount;
                this.lastInstanceSource = instanceSource;
                this.featuresCached = true;
            }
            else
            {
                if (!this.featuresCached)
                {
                    // Instances in cache, now cache all feature vectors in native format
                    this.CacheAllFeatureVectors(Utilities.GetBatch(batchNumber, this.instances, this.batchCount), instanceSource);

                    this.featuresCached = true;
                }
            }
        }

        /// <summary>
        /// Caches all features provided by the instance source in the native format.
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">An optional label source.</param>
        /// <param name="batchNumber">
        /// An optional batch number. Defaults to 0 and is used only if the instance source is divided into batches.
        /// </param>
        /// <remarks>Call this method every time before accessing label data.</remarks>
        private void CacheLabels(TInstanceSource instanceSource, TLabelSource labelSource = default(TLabelSource), int batchNumber = 0)
        {
            Debug.Assert(instanceSource != null, "The instance source must not be null.");

            if (!this.IsInstanceSourceCached(instanceSource, batchNumber))
            {
                this.labelsCached = false;
                this.featuresCached = false;

                // Get instances and labels and cache both
                this.instances = this.StandardMapping.GetInstancesSafe(instanceSource).ToArray();
                this.labels = this.GetNativeLabels(Utilities.GetBatch(batchNumber, this.instances, this.batchCount), instanceSource, labelSource);

                this.lastBatchNumber = batchNumber;
                this.lastBatchCount = this.batchCount;
                this.lastInstanceSource = instanceSource;
                this.lastLabelSource = labelSource;
                this.labelsCached = true;
            }
            else
            {
                // Label source might have changed while instance source has not
                if (labelSource != null && !this.IsLabelSourceCached(labelSource, batchNumber))
                {
                    this.labelsCached = false;
                }

                if (!this.labelsCached)
                {
                    // Instances in cache, now cache labels
                    this.labels = this.GetNativeLabels(Utilities.GetBatch(batchNumber, this.instances, this.batchCount), instanceSource, labelSource);

                    if (labelSource != null)
                    {
                        this.lastLabelSource = labelSource;
                    }

                    this.labelsCached = true;
                }
            }
        }

        /// <summary>
        /// Caches all feature vectors of the specified instances.
        /// </summary>
        /// <param name="instances">The instances to cache the feature vectors for.</param>
        /// <param name="instanceSource">An optional instance source.</param>
        private void CacheAllFeatureVectors(IReadOnlyList<TInstance> instances, TInstanceSource instanceSource = default(TInstanceSource))
        {
            Debug.Assert(instances != null, "The instances must not be null.");
            Debug.Assert(instances.All(instance => instance != null), "An instance must not be null.");

            if (instances.Count == 0)
            {
                this.isSparseFeatureRepresentation = false;
                this.featureCount = 0;
                this.featureValues = new double[0][];
                this.featureIndexes = null;
            }
            else
            {
                // Setup the feature representation based on the type of the feature vector of the first instance
                Vector featureVector = this.StandardMapping.GetFeaturesSafe(instances[0], instanceSource);
                this.isSparseFeatureRepresentation = featureVector.IsSparse;
                this.featureCount = featureVector.Count;

                this.featureValues = new double[instances.Count][];
                this.featureIndexes = new int[instances.Count][];
                this.featureValues[0] = this.GetFeatureValues(featureVector);
                this.featureIndexes[0] = this.GetFeatureIndexes(featureVector);

                for (int i = 1; i < instances.Count; i++)
                {
                    featureVector = this.StandardMapping.GetFeaturesSafe(instances[i], instanceSource);
                    this.featureValues[i] = this.GetFeatureValues(featureVector);
                    this.featureIndexes[i] = this.GetFeatureIndexes(featureVector);
                }

                if (!this.isSparseFeatureRepresentation)
                {
                    this.featureIndexes = null;
                }
            }
        }

        #endregion
    }
}
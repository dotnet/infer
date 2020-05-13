// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#define SpecializeInterfaces

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.IO;
    using System.Linq;
    using Collections;
    using Math;
    using Factors.Attributes;
    using Utilities;
    using Microsoft.ML.Probabilistic.Serialization;
    using System.Runtime.Serialization;
    using System.Collections.Generic;

    /// <summary>
    /// A distribution over an array of type <typeparamref name="DomainType"/>, where each element is independent and has distribution of type <typeparamref name="T"/>, all stored in a file.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <typeparam name="DomainType"></typeparam>
    [Serializable]
    [Quality(QualityBand.Experimental)]
    public class DistributionFileArray<T, DomainType> : FileArray<T>, IDistribution<DomainType[]>, Sampleable<DomainType[]>, HasPoint<IList<DomainType>>
#if SpecializeInterfaces
                                                        , SettableTo<DistributionFileArray<T, DomainType>>,
                                                        SettableToProduct<DistributionFileArray<T, DomainType>>,
                                                        SettableToRatio<DistributionFileArray<T, DomainType>>,
                                                        SettableToPower<DistributionFileArray<T, DomainType>>,
                                                        SettableToWeightedSum<DistributionFileArray<T, DomainType>>,
                                                        CanGetLogAverageOf<DistributionFileArray<T, DomainType>>,
                                                        CanGetLogAverageOfPower<DistributionFileArray<T, DomainType>>,
                                                        CanGetAverageLog<DistributionFileArray<T, DomainType>>
#endif
        where T : ICloneable, SettableTo<T>,
            SettableToProduct<T>,
            SettableToRatio<T>,
            SettableToPower<T>,
            SettableToWeightedSum<T>,
            CanGetLogAverageOf<T>,
            CanGetLogAverageOfPower<T>,
            CanGetAverageLog<T>,
            IDistribution<DomainType>,
            Sampleable<DomainType>
    {
        public DistributionFileArray(string prefix, int count, [SkipIfUniform] Func<int, T> init)
            : base(prefix, count, init)
        {
        }

        [Skip]
        public DistributionFileArray(string prefix, int count)
            : base(prefix, count)
        {
        }

        [Skip]
        public DistributionFileArray([IgnoreDeclaration] FileArray<DistributionFileArray<T, DomainType>> parent, int index, int count)
            : this(parent.GetItemFolder(index), count)
        {
            parent.containsFileArrays = true;
            this.doNotDelete = true;
            parent.StoreItem(index, this);
        }

        public override object Clone()
        {
            if (containsFileArrays) throw new NotImplementedException();
            string folder = prefix.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            return new DistributionFileArray<T, DomainType>(folder + "_clone", count, i => this[i]);
        }

        public void SetTo(DistributionFileArray<T, DomainType> value)
        {
            for (int i = 0; i < count; i++)
            {
                this[i] = value[i];
            }
        }

        public void SetToProduct(DistributionFileArray<T, DomainType> a, DistributionFileArray<T, DomainType> b)
        {
            for (int i = 0; i < count; i++)
            {
                T item = this[i];
                item.SetToProduct(a[i], b[i]);
                this[i] = item;
            }
        }

        public void SetToRatio(DistributionFileArray<T, DomainType> numerator, DistributionFileArray<T, DomainType> denominator, bool forceProper)
        {
            throw new NotImplementedException();
        }

        public void SetToPower(DistributionFileArray<T, DomainType> value, double exponent)
        {
            throw new NotImplementedException();
        }

        public void SetToSum(double weight1, DistributionFileArray<T, DomainType> value1, double weight2, DistributionFileArray<T, DomainType> value2)
        {
            throw new NotImplementedException();
        }

        public double GetLogAverageOf(DistributionFileArray<T, DomainType> that)
        {
            throw new NotImplementedException();
        }

        public double GetLogAverageOfPower(DistributionFileArray<T, DomainType> that, double power)
        {
            throw new NotImplementedException();
        }

        public double GetAverageLog(DistributionFileArray<T, DomainType> that)
        {
            throw new NotImplementedException();
        }

        public void SetToUniform()
        {
            for (int i = 0; i < count; i++)
            {
                T item = this[i];
                item.SetToUniform();
                this[i] = item;
            }
        }

        public bool IsUniform()
        {
            return this.All(item => item.IsUniform());
        }

        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public DomainType[] Point
        {
            get => Util.ArrayInit(count, i => this[i].Point);
            set { ((HasPoint<IList<DomainType>>)this).Point = value; }
        }

        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public bool IsPointMass
        {
            get => Enumerable.Range(0, count).All(i => this[i].IsPointMass);
        }

        IList<DomainType> HasPoint<IList<DomainType>>.Point
        {
            get => ((HasPoint<DomainType[]>)this).Point;
            set
            {
                for (int i = 0; i < count; i++)
                {
                    T item = this[i];
                    item.Point = value[i];
                    this[i] = item;
                }
            }
        }

        bool HasPoint<IList<DomainType>>.IsPointMass => ((HasPoint<DomainType[]>)this).IsPointMass;

        public double MaxDiff(object that)
        {
            throw new NotImplementedException();
        }

        public double GetLogProb(DomainType[] value)
        {
            throw new NotImplementedException();
        }

        public DomainType[] Sample()
        {
            throw new NotImplementedException();
        }

        public DomainType[] Sample(DomainType[] result)
        {
            throw new NotImplementedException();
        }
    }
}
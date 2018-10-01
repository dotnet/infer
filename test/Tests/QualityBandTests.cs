// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Xunit;
using Microsoft.ML.Probabilistic.Distributions;
using System.Reflection;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Factors.Attributes;
using Microsoft.ML.Probabilistic.Models;

namespace Microsoft.ML.Probabilistic.Tests
{
    using Assert = Xunit.Assert;

    /// <summary>
    /// Summary description for QualityBandTests
    /// </summary>
    public class QualityBandTests
    {
        public QualityBandTests()
        {
            //
            // TODO: Add constructor logic here
            //
        }

        private class DBase : IDistribution<double>
        {
            public object Clone()
            {
                throw new NotImplementedException();
            }

            public double Point
            {
                get { throw new NotImplementedException(); }
                set { throw new NotImplementedException(); }
            }

            public bool IsPointMass
            {
                get { throw new NotImplementedException(); }
            }

            public double MaxDiff(object that)
            {
                throw new NotImplementedException();
            }

            public void SetToUniform()
            {
                throw new NotImplementedException();
            }

            public bool IsUniform()
            {
                throw new NotImplementedException();
            }

            public double GetLogProb(double value)
            {
                throw new NotImplementedException();
            }
        }

        [Quality(QualityBand.Mature)]
        private class DMature : DBase
        {
        }

        [Quality(QualityBand.Stable)]
        private class DStable : DBase
        {
        }

        [Quality(QualityBand.Preview)]
        private class DPreview : DBase
        {
        }

        [Quality(QualityBand.Experimental)]
        private class DExperimental : DBase
        {
        }

        private class DUnmarked : DBase
        {
        }

        private class CBase : IDistribution<double>
        {
            public object Clone()
            {
                throw new NotImplementedException();
            }

            public double Point
            {
                get { throw new NotImplementedException(); }
                set { throw new NotImplementedException(); }
            }

            public bool IsPointMass
            {
                get { throw new NotImplementedException(); }
            }

            public double MaxDiff(object that)
            {
                throw new NotImplementedException();
            }

            public void SetToUniform()
            {
                throw new NotImplementedException();
            }

            public bool IsUniform()
            {
                throw new NotImplementedException();
            }

            public double GetLogProb(double value)
            {
                throw new NotImplementedException();
            }

            [Quality(QualityBand.Mature)]
            public virtual void MMature()
            {
            }

            [Quality(QualityBand.Stable)]
            public virtual void MStable()
            {
            }

            [Quality(QualityBand.Preview)]
            public virtual void MPreview()
            {
            }

            [Quality(QualityBand.Experimental)]
            public virtual void MExperimental()
            {
            }

            public virtual void MUnmarked()
            {
            }
        }

        [Quality(QualityBand.Mature)]
        private class CMature<T1, T2> : CBase
        {
        }

        [Quality(QualityBand.Stable)]
        private class CStable<T1, T2> : CBase
        {
        }

        [Quality(QualityBand.Preview)]
        private class CPreview<T1, T2> : CBase
        {
        }

        [Quality(QualityBand.Experimental)]
        private class CExperimental<T1, T2> : CBase
        {
        }

        private class CUnmarked<T1, T2> : CBase
        {
        }

        [Fact]
        public void QualityBandDistributionTest()
        {
            // Expected unknown
            List<Type> unknownTypes = new List<Type>();
            unknownTypes.Add(typeof (CUnmarked<DExperimental, double>));
            unknownTypes.Add(typeof (CUnmarked<double, DExperimental>));
            unknownTypes.Add(typeof (CUnmarked<CPreview<DExperimental, double>, double>));
            unknownTypes.Add(typeof (CUnmarked<double, CPreview<DExperimental, double>>));
            unknownTypes.Add(typeof (CUnmarked<CMature<DExperimental, double>, CPreview<DExperimental, double>>));
            unknownTypes.Add(typeof (CExperimental<DUnmarked, double>));
            unknownTypes.Add(typeof (CExperimental<double, DUnmarked>));
            unknownTypes.Add(typeof (CExperimental<CPreview<DUnmarked, double>, double>));
            unknownTypes.Add(typeof (CExperimental<double, CUnmarked<DMature, double>>));
            unknownTypes.Add(typeof (CExperimental<CMature<DMature, double>, CPreview<DUnmarked, double>>));
            unknownTypes.Add(typeof (CExperimental<CUnmarked<DMature, double>, CPreview<DUnmarked, double>>));
            int uexp = unknownTypes.Count;
            for (int i = 0; i < uexp; i++)
                unknownTypes.Add(unknownTypes[i].MakeArrayType(i + 1));
            foreach (Type ut in unknownTypes)
            {
                Assert.True(Distribution.HasDistributionType(ut));
                Assert.Equal(QualityBand.Unknown, Distribution.GetQualityBand(ut));
            }

            // Expected experimental
            List<Type> experimentalTypes = new List<Type>();
            experimentalTypes.Add(typeof (CStable<DExperimental, double>));
            experimentalTypes.Add(typeof (CStable<double, DExperimental>));
            experimentalTypes.Add(typeof (CStable<CPreview<DExperimental, double>, double>));
            experimentalTypes.Add(typeof (CStable<double, CPreview<DExperimental, double>>));
            experimentalTypes.Add(typeof (CStable<CMature<DExperimental, double>, CPreview<DExperimental, double>>));
            experimentalTypes.Add(typeof (CExperimental<DStable, double>));
            experimentalTypes.Add(typeof (CExperimental<double, DStable>));
            experimentalTypes.Add(typeof (CExperimental<CPreview<DMature, double>, double>));
            experimentalTypes.Add(typeof (CExperimental<double, CPreview<DMature, double>>));
            experimentalTypes.Add(typeof (CExperimental<CMature<DMature, double>, CPreview<DStable, double>>));
            int nexp = experimentalTypes.Count;
            for (int i = 0; i < nexp; i++)
                experimentalTypes.Add(experimentalTypes[i].MakeArrayType(i + 1));
            foreach (Type et in experimentalTypes)
            {
                Assert.True(Distribution.HasDistributionType(et));
                Assert.Equal(QualityBand.Experimental, Distribution.GetQualityBand(et));
            }
            // Expected stable
            List<Type> stableTypes = new List<Type>();
            stableTypes.Add(typeof (CStable<DStable, double>));
            stableTypes.Add(typeof (CStable<double, DMature>));
            stableTypes.Add(typeof (CStable<CMature<DStable, double>, double>));
            stableTypes.Add(typeof (CStable<double, CStable<DMature, double>>));
            stableTypes.Add(typeof (CStable<CMature<DStable, double>, CStable<DStable, double>>));
            stableTypes.Add(typeof (CMature<DStable, double>));
            stableTypes.Add(typeof (CMature<double, DStable>));
            stableTypes.Add(typeof (CMature<CStable<DMature, double>, double>));
            stableTypes.Add(typeof (CMature<double, CStable<DMature, double>>));
            stableTypes.Add(typeof (CMature<CMature<DMature, double>, CStable<DStable, double>>));
            int nstb = stableTypes.Count;
            for (int i = 0; i < nstb; i++)
                stableTypes.Add(stableTypes[i].MakeArrayType(i + 1));
            foreach (Type st in stableTypes)
            {
                Assert.True(Distribution.HasDistributionType(st));
                Assert.Equal(QualityBand.Stable, Distribution.GetQualityBand(st));
            }
            // Expected Mature
            List<Type> matureTypes = new List<Type>();
            matureTypes.Add(typeof (CMature<DMature, double>));
            matureTypes.Add(typeof (CMature<double, DMature>));
            matureTypes.Add(typeof (CMature<CMature<DMature, double>, double>));
            matureTypes.Add(typeof (CMature<double, CMature<DMature, double>>));
            matureTypes.Add(typeof (CMature<CMature<DMature, double>, CMature<DMature, double>>));
            int nmat = matureTypes.Count;
            for (int i = 0; i < nmat; i++)
                matureTypes.Add(matureTypes[i].MakeArrayType(i + 1));
            foreach (Type mt in matureTypes)
            {
                Assert.True(Distribution.HasDistributionType(mt));
                Assert.Equal(QualityBand.Mature, Distribution.GetQualityBand(mt));
            }
        }

        [Fact]
        public void QualityBandMethodTest()
        {
            MemberInfo miMatureInMature = typeof (CMature<double, double>).GetMethod("MMature");
            Assert.Equal(QualityBand.Mature, Quality.GetQualityBand(miMatureInMature));
            MemberInfo miStableInMature = typeof (CMature<double, double>).GetMethod("MStable");
            Assert.Equal(QualityBand.Stable, Quality.GetQualityBand(miStableInMature));
            MemberInfo miPreviewInMature = typeof (CMature<double, double>).GetMethod("MPreview");
            Assert.Equal(QualityBand.Preview, Quality.GetQualityBand(miPreviewInMature));
            MemberInfo miExperimentalInMature = typeof (CMature<double, double>).GetMethod("MExperimental");
            Assert.Equal(QualityBand.Experimental, Quality.GetQualityBand(miExperimentalInMature));
            MemberInfo miUnmarkedInMature = typeof (CMature<double, double>).GetMethod("MUnmarked");
            Assert.Equal(QualityBand.Mature, Quality.GetQualityBand(miUnmarkedInMature));

            MemberInfo miMatureInStable = typeof (CStable<double, double>).GetMethod("MMature");
            Assert.Equal(QualityBand.Mature, Quality.GetQualityBand(miMatureInStable));
            MemberInfo miStableInStable = typeof (CStable<double, double>).GetMethod("MStable");
            Assert.Equal(QualityBand.Stable, Quality.GetQualityBand(miStableInStable));
            MemberInfo miPreviewInStable = typeof (CStable<double, double>).GetMethod("MPreview");
            Assert.Equal(QualityBand.Preview, Quality.GetQualityBand(miPreviewInStable));
            MemberInfo miExperimentalInStable = typeof (CStable<double, double>).GetMethod("MExperimental");
            Assert.Equal(QualityBand.Experimental, Quality.GetQualityBand(miExperimentalInStable));
            MemberInfo miUnmarkedInStable = typeof (CStable<double, double>).GetMethod("MUnmarked");
            Assert.Equal(QualityBand.Stable, Quality.GetQualityBand(miUnmarkedInStable));

            MemberInfo miMatureInPreview = typeof (CPreview<double, double>).GetMethod("MMature");
            Assert.Equal(QualityBand.Mature, Quality.GetQualityBand(miMatureInPreview));
            MemberInfo miStableInPreview = typeof (CPreview<double, double>).GetMethod("MStable");
            Assert.Equal(QualityBand.Stable, Quality.GetQualityBand(miStableInPreview));
            MemberInfo miPreviewInPreview = typeof (CPreview<double, double>).GetMethod("MPreview");
            Assert.Equal(QualityBand.Preview, Quality.GetQualityBand(miPreviewInPreview));
            MemberInfo miExperimentalInPreview = typeof (CPreview<double, double>).GetMethod("MExperimental");
            Assert.Equal(QualityBand.Experimental, Quality.GetQualityBand(miExperimentalInPreview));
            MemberInfo miUnmarkedInPreview = typeof (CPreview<double, double>).GetMethod("MUnmarked");
            Assert.Equal(QualityBand.Preview, Quality.GetQualityBand(miUnmarkedInPreview));

            MemberInfo miMatureInExperimental = typeof (CExperimental<double, double>).GetMethod("MMature");
            Assert.Equal(QualityBand.Mature, Quality.GetQualityBand(miMatureInExperimental));
            MemberInfo miStableInExperimental = typeof (CExperimental<double, double>).GetMethod("MStable");
            Assert.Equal(QualityBand.Stable, Quality.GetQualityBand(miStableInExperimental));
            MemberInfo miPreviewInExperimental = typeof (CExperimental<double, double>).GetMethod("MPreview");
            Assert.Equal(QualityBand.Preview, Quality.GetQualityBand(miPreviewInExperimental));
            MemberInfo miExperimentalInExperimental = typeof (CExperimental<double, double>).GetMethod("MExperimental");
            Assert.Equal(QualityBand.Experimental, Quality.GetQualityBand(miExperimentalInExperimental));
            MemberInfo miUnmarkedInExperimental = typeof (CExperimental<double, double>).GetMethod("MUnmarked");
            Assert.Equal(QualityBand.Experimental, Quality.GetQualityBand(miUnmarkedInExperimental));

            MemberInfo miMatureInUnmarked = typeof (CUnmarked<double, double>).GetMethod("MMature");
            Assert.Equal(QualityBand.Mature, Quality.GetQualityBand(miMatureInUnmarked));
            MemberInfo miStableInUnmarked = typeof (CUnmarked<double, double>).GetMethod("MStable");
            Assert.Equal(QualityBand.Stable, Quality.GetQualityBand(miStableInUnmarked));
            MemberInfo miPreviewInUnmarked = typeof (CUnmarked<double, double>).GetMethod("MPreview");
            Assert.Equal(QualityBand.Preview, Quality.GetQualityBand(miPreviewInUnmarked));
            MemberInfo miExperimentalInUnmarked = typeof (CUnmarked<double, double>).GetMethod("MExperimental");
            Assert.Equal(QualityBand.Experimental, Quality.GetQualityBand(miExperimentalInUnmarked));
            MemberInfo miUnmarkedInUnmarked = typeof (CUnmarked<double, double>).GetMethod("MUnmarked");
            Assert.Equal(QualityBand.Unknown, Quality.GetQualityBand(miUnmarkedInUnmarked));
        }

        [Fact]
        public void QualityBandDefaultsTest()
        {
            InferenceEngine ie = new InferenceEngine();
            Assert.Equal(QualityBand.Experimental, ie.Compiler.RequiredQuality);
            Assert.Equal(QualityBand.Preview, ie.Compiler.RecommendedQuality);

            // Required should latch up Recommended
            ie.Compiler.RequiredQuality = QualityBand.Stable;
            Assert.Equal(QualityBand.Stable, ie.Compiler.RequiredQuality);
            Assert.Equal(QualityBand.Stable, ie.Compiler.RecommendedQuality);

            // Recommended should latch down Required
            ie.Compiler.RequiredQuality = QualityBand.Mature;
            ie.Compiler.RecommendedQuality = QualityBand.Stable;

            Assert.Equal(QualityBand.Stable, ie.Compiler.RequiredQuality);
            Assert.Equal(QualityBand.Stable, ie.Compiler.RecommendedQuality);

            ie.Compiler.RequiredQuality = QualityBand.Preview;
            ie.Compiler.RecommendedQuality = QualityBand.Stable;

            Assert.Equal(QualityBand.Preview, ie.Compiler.RequiredQuality);
            Assert.Equal(QualityBand.Stable, ie.Compiler.RecommendedQuality);

            InferenceEngine ie2 = new InferenceEngine();
            ie2.SetTo(ie);

            Assert.Equal(QualityBand.Preview, ie2.Compiler.RequiredQuality);
            Assert.Equal(QualityBand.Stable, ie2.Compiler.RecommendedQuality);
        }
    }
}
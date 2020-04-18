using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Serialization;
using Xunit;

namespace Microsoft.ML.Probabilistic.Tests.Core
{
    using Assert = Xunit.Assert;

    public class MatlabSerializationTests
    {
        [Fact]
        //[DeploymentItem(@"Data\IRT2PL_10_250.mat", "Data")]
        public void MatlabReaderTest2()
        {
            Dictionary<string, object> dict = MatlabReader.Read(Path.Combine(TestUtils.DataFolderPath, "IRT2PL_10_250.mat"));
            Assert.Equal(5, dict.Count);
            Matrix m = (Matrix)dict["Y"];
            Assert.True(m.Rows == 250);
            Assert.True(m.Cols == 10);
            Assert.True(m[0, 1] == 0.0);
            Assert.True(m[1, 0] == 1.0);
            m = (Matrix)dict["difficulty"];
            Assert.True(m.Rows == 10);
            Assert.True(m.Cols == 1);
            Assert.True(MMath.AbsDiff(m[1], 0.7773) < 2e-4);
        }

        [Fact]
        ////[DeploymentItem(@"Data\test.mat", "Data")]
        public void MatlabReaderTest()
        {
            MatlabReaderTester(Path.Combine(TestUtils.DataFolderPath, "test.mat"));
        }

        private void MatlabReaderTester(string fileName)
        {
            Dictionary<string, object> dict = MatlabReader.Read(fileName);
            Assert.Equal(12, dict.Count);
            Matrix aScalar = (Matrix)dict["aScalar"];
            Assert.Equal(1, aScalar.Rows);
            Assert.Equal(1, aScalar.Cols);
            Assert.Equal(5.0, aScalar[0, 0]);
            Assert.Equal("string", (string)dict["aString"]);
            MatlabReader.ComplexMatrix aComplexScalar = (MatlabReader.ComplexMatrix)dict["aComplexScalar"];
            Assert.Equal(5.0, aComplexScalar.Real[0, 0]);
            Assert.Equal(3.0, aComplexScalar.Imaginary[0, 0]);
            MatlabReader.ComplexMatrix aComplexVector = (MatlabReader.ComplexMatrix)dict["aComplexVector"];
            Assert.Equal(1.0, aComplexVector.Real[0, 0]);
            Assert.Equal(2.0, aComplexVector.Imaginary[0, 0]);
            Assert.Equal(3.0, aComplexVector.Real[0, 1]);
            Assert.Equal(4.0, aComplexVector.Imaginary[0, 1]);
            var aStruct = (Dictionary<string, object>)dict["aStruct"];
            Assert.Equal(2, aStruct.Count);
            Assert.Equal(1.0, ((Matrix)aStruct["field1"])[0]);
            Assert.Equal("two", (string)aStruct["field2"]);
            object[,] aCell = (object[,])dict["aCell"];
            Assert.Equal(1.0, ((Matrix)aCell[0, 0])[0]);
            int[] intArray = (int[])dict["intArray"];
            Assert.Equal(1, intArray[0]);
            int[] uintArray = (int[])dict["uintArray"];
            Assert.Equal(1, uintArray[0]);
            bool[] aLogical = (bool[])dict["aLogical"];
            Assert.True(aLogical[0]);
            Assert.True(aLogical[1]);
            Assert.False(aLogical[2]);
            object[,,] aCell3D = (object[,,])dict["aCell3D"];
            Assert.Null(aCell3D[0, 0, 0]);
            Assert.Equal(7.0, ((Matrix)aCell3D[0, 0, 1])[0, 0]);
            Assert.Equal(6.0, ((Matrix)aCell3D[0, 1, 0])[0, 0]);
            double[,,,] array4D = (double[,,,])dict["array4D"];
            Assert.Equal(4.0, array4D[0, 0, 1, 0]);
            Assert.Equal(5.0, array4D[0, 0, 0, 1]);
            long[] aLong = (long[])dict["aLong"];
            Assert.Equal(1234567890123456789L, aLong[0]);
        }

        [Fact]
        //[DeploymentItem(@"Data\test.mat", "Data")]
        public void MatlabWriterTest()
        {
            Dictionary<string, object> dict = MatlabReader.Read(Path.Combine(TestUtils.DataFolderPath, "test.mat"));
            string fileName = $"{System.IO.Path.GetTempPath()}MatlabWriterTest{Environment.CurrentManagedThreadId}.mat";
            using (MatlabWriter writer = new MatlabWriter(fileName))
            {
                foreach (var entry in dict)
                {
                    writer.Write(entry.Key, entry.Value);
                }
            }
            MatlabReaderTester(fileName);
        }

        [Fact]
        public void MatlabWriteStringDictionaryTest()
        {
            Dictionary<string, string> dictString = new Dictionary<string, string>();
            dictString["a"] = "a";
            dictString["b"] = "b";
            string fileName = $"{System.IO.Path.GetTempPath()}MatlabWriteStringDictionaryTest{Environment.CurrentManagedThreadId}.mat";
            using (MatlabWriter writer = new MatlabWriter(fileName))
            {
                writer.Write("dictString", dictString);
            }
            Dictionary<string, object> vars = MatlabReader.Read(fileName);
            Dictionary<string, object> dict = (Dictionary<string, object>)vars["dictString"];
            foreach (var entry in dictString)
            {
                Assert.Equal(dictString[entry.Key], dict[entry.Key]);
            }
        }

        [Fact]
        public void MatlabWriteStringListTest()
        {
            List<string> strings = new List<string>();
            strings.Add("a");
            strings.Add("b");
            string fileName = $"{System.IO.Path.GetTempPath()}MatlabWriteStringListTest{Environment.CurrentManagedThreadId}.mat";
            using (MatlabWriter writer = new MatlabWriter(fileName))
            {
                writer.Write("strings", strings);
            }
            Dictionary<string, object> vars = MatlabReader.Read(fileName);
            string[] array = (string[])vars["strings"];
            for (int i = 0; i < array.Length; i++)
            {
                Assert.Equal(strings[i], array[i]);
            }
        }

        [Fact]
        public void MatlabWriteEmptyArrayTest()
        {
            string fileName = $"{System.IO.Path.GetTempPath()}MatlabWriteEmptyArrayTest{Environment.CurrentManagedThreadId}.mat";
            using (MatlabWriter writer = new MatlabWriter(fileName))
            {
                writer.Write("ints", new int[0]);
            }
            Dictionary<string, object> vars = MatlabReader.Read(fileName);
            int[] ints = (int[])vars["ints"];
            Assert.Empty(ints);
        }

        [Fact]
        public void MatlabWriteNumericNameTest()
        {
            string fileName = $"{System.IO.Path.GetTempPath()}MatlabWriteNumericNameTest{Environment.CurrentManagedThreadId}.mat";
            using (MatlabWriter writer = new MatlabWriter(fileName))
            {
                writer.Write("24", new int[0]);
            }
            Dictionary<string, object> vars = MatlabReader.Read(fileName);
            int[] ints = (int[])vars["24"];
            Assert.Empty(ints);
        }
    }
}

using Loki.Shared;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Infer.Loki.Mappings
{
    public static class AutomatonMethods
    {
        public static int GetWeightHighBits<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>(ulong operationId, DoubleWithTransformingPrecision logWeight)
        {
            return (int)(BitConverter.DoubleToInt64Bits(logWeight.ToDouble()) >> 32);
        }

        public static int GetHashCodeFromLogNorm<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>(ulong operationId, DoubleWithTransformingPrecision logNorm)
        {
            return (BitConverter.DoubleToInt64Bits(logNorm.ToDouble()) >> 31).GetHashCode();
        }
    }
}

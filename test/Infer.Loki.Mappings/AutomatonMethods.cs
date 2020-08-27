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
        public static int GetWeightHighBits<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>(DoubleWithTransformingPrecision logWeight)
        {
            return (int)(BitConverter.DoubleToInt64Bits(logWeight.ToDouble()) >> 32);
        }
    }
}

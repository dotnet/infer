using Infer.Loki.Mappings;
using Loki.Mapping;
using Loki.Mapping.Maps;
using Loki.Shared;
using Microsoft.ML.Probabilistic.Distributions.Automata;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Infer.Loki.Tests
{
    class AutomatonMap : IMap
    {
        public void MapAll(Mappers mappers)
        {
            // TODO: uncomment, when maps for generics are implemented in Loki
            //mappers.PermanentMethodMapper.CreateMap(
            //    typeof(Automaton<,,,,>).GetNestedType("Determinization", System.Reflection.BindingFlags.NonPublic).GetNestedType("WeightedState").GetMethod("GetWeightHighBits", System.Reflection.BindingFlags.NonPublic),
            //    new Func<DoubleWithTransformingPrecision, int>(AutomatonMethods.GetWeightHighBits).Method);
        }
    }
}

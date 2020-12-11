using Infer.Loki.Mappings;
using Loki.Mapping;
using Loki.Mapping.Maps;
using Loki.Shared;
using Microsoft.ML.Probabilistic.Distributions.Automata;
using System;
using System.Reflection;

namespace Infer.Loki.Tests
{
    class AutomatonMap : IMap
    {
        public void MapAll(Mappers mappers)
        {
            mappers.DirectMethodMapper.CreateMap(
                typeof(Automaton<,,,,>).GetNestedType("Determinization", BindingFlags.NonPublic).GetNestedType("WeightedState").GetMethod("GetWeightHighBits", BindingFlags.NonPublic | BindingFlags.Static),
                new Func<ulong, DoubleWithTransformingPrecision, int>(AutomatonMethods.GetWeightHighBits<int,int,int,int,int>).Method);

            mappers.DirectMethodMapper.CreateMap(
                typeof(Automaton<,,,,>).GetMethod("GetHashCodeFromLogNorm", BindingFlags.NonPublic | BindingFlags.Static),
                new Func<ulong, DoubleWithTransformingPrecision, int>(AutomatonMethods.GetHashCodeFromLogNorm<int, int, int, int, int>).Method);
        }
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using Microsoft.ML.Probabilistic.Collections;

    public static class RelativeStatesCache
    {
        public static AutomatonData<TElement, TElementDistribution>.RelativeState TryGetState<TElement, TElementDistribution>(
            AutomatonData<TElement, TElementDistribution>.Transition transition, Weight stateEndWeight)
        where TElementDistribution: IDistribution<TElement>, HasPoint<TElement>
        {
            if (transition.Weight != Weight.One ||
                transition.IsEpsilon ||
                transition.DestinationStateIndex != 1 ||
                stateEndWeight != Weight.Zero ||
                !transition.ElementDistribution.Value.IsPointMass)
            {
                return null;
            }

            if (!(transition is AutomatonData<char, DiscreteChar>.Transition charTransition))
            {
                return null;
            }

            var index = (int)charTransition.ElementDistribution.Value.Point;
            if (CharsHolder.States[index] == null)
            {
                var transitions = ImmutableArray.Create(charTransition);
                CharsHolder.States[index] = new AutomatonData<char, DiscreteChar>.RelativeState(transitions, stateEndWeight);
            }

            return (AutomatonData<TElement, TElementDistribution>.RelativeState)(object)CharsHolder.States[index];
        }

        private static class CharsHolder
        {
            public static AutomatonData<char, DiscreteChar>.RelativeState[] States { get; } =
                new AutomatonData<char, DiscreteChar>.RelativeState[char.MaxValue + 1];
        }
    }
}

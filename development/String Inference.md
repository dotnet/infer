---
layout: default
---
[Infer.NET development](index.md)

## String inference

Infer.NET represents distributions on strings via probabilistic automata.  Operations on automata are described in [Belief Propagation with Strings](https://www.microsoft.com/en-us/research/publication/belief-propagation-strings/), with more details in [Notes on String operators](StringInference.pdf).  The Infer.NET implementation is described in [String inference API design](String inference API design.md).

### Viewing automata

You can create a GraphViz file from a SequenceDistribution by calling `ToString` on the distribution using the   `SequenceDistributionFormats.GraphViz` format and writing to file.

You can use [http://dot-graphics1.appspot.com/](http://dot-graphics1.appspot.com/) to view small automata by copying and pasting the string from the file to the window.

To view large automata, install GraphViz, and run `dot.exe`. The following example creates a jpeg from the file containing the graph description:

```shell
dot -Tjpeg myautomaton.txt > myautomaton.jpeg
```

### Reading

*   [Introduction to Automata Theory](https://mcdtu.files.wordpress.com/2017/03/introduction-to-automata-theory.pdf) - a book covering basics
*   [Weighted Finite-State Transducer Algorithms: An Overview](https://cs.nyu.edu/~mohri/pub/fla.pdf) - a survey of algorithms on WFSA covering intersection, determinization, weight pushing and minimization.
*   [Finite-State Transducers in Language and Speech Processing](http://www.aclweb.org/anthology/J97-2003) - a more detailed survey
*   [Weighted Automata Algorithms](https://cs.nyu.edu/~mohri/pub/hwa.pdf) - one more survey, with a focus on algorithms

### Automata libraries

*   [ASTL](http://astl.sourceforge.net/) - C++ automata library with STL-like design: algorithms are implemented in terms of automata iterators.
*   [Jolt.NET](https://archive.codeplex.com/?p=jolt) - .NET DFA/NFA library
*   [OpenFST](http://www.openfst.org/twiki/bin/view/FST/WebHome) - weighted finite-state transducer C++ library

### Useful facts from automata theory

*   Transducer operations
    *   Intersection - regular relations aren't closed under intersection, complement and subtraction \[[link](https://web.archive.org/web/20130606165805/https:/courses.cit.cornell.edu/ling4424/regular-relations-oct5.pdf)\]
        *   What would happen if one applies the joint traversal as for automata intersection?
*   DFA (deterministic finite automaton) minimization
    *   [Survey](https://arxiv.org/pdf/1010.5318.pdf)
    *   Basic idea: remove unreachable states, merge indistinguishable states \[[link](https://mcdtu.files.wordpress.com/2017/03/introduction-to-automata-theory.pdf), p.159\]
    *   Brzozowski's algorithm \[[link](https://homepage.tudelft.nl/c9d1n/talks/brz-coin.pdf)\]
        *   Invert all edges of the DFA to get an NDFA for the reversed language
        *   Apply the power set construction to the NDFA to get a minimal DFA for the reversed language
        *   Invert the edges of the minimal DFA back and apply the power set construction to get the minimal DFA for the original language
        *   Applies to DFA as well as NFA
        *   Is claimed to be efficient in practice
        *   Can be justified from the algebraic point of view \[[link](http://www.alexandrasilva.org/files/RechabilityObservability.pdf)\]
*   NFA (non-deterministic finite automaton) minimization
    *   Can determinize, then minimize
        *   Can't the minimal DFA have even more states than the original NFA?
    *   Can apply Brzozowski's algorithm
    *   Minimization of NFA is PSPACE-complete (that is, very hard) \[[discussion link](https://cstheory.stackexchange.com/questions/10829/computing-the-minimal-nfa-for-a-dfa?rq=1)\]
    *   What if one gives up on the optimality guarantees? \[[discussion link](https://cstheory.stackexchange.com/questions/18074/simplification-of-weighted-nfa)\]
        *   One can introduce a state equivalence relation stronger than the language equality relation and merge states based on it. Examples include equivalence based on simulation relations \[[link](https://arxiv.org/pdf/1210.6624v1.pdf)\].
        *   There exist some attempts to generalize this technique to the weighted case \[[link](https://people.cs.umu.se/johanna/bisimulation/hogmalmay07c.pdf), [link](https://www.sciencedirect.com/science/article/pii/S0304397507008614)\]
*   Weighted NFA determinization \[[link](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.437.2637&rep=rep1&type=pdf)\]
    *   Requires an extension of power set construction since it matters not only in which state you are, but also how much weight you carry
    *   Extended power set construction may not halt for some automata
        *   Some weighted NFA don't even have an equivalent weighted DFA
        *   Weighted NFA is always determinizable if there are no loops
        *   Having only self-loops doesn't save you
    *   There is a criteria to determine if an automata is determinizable, so-called twins property, but it works only for tropical semirings
        *   Apparently, there are works that generalize it to arbitrary semirings \[[link](http://adambuchsbaum.com/papers/det-sicomp.pdf)\]
    *   Weighted NFA over a tropical semiring can be made determinizable by adding new transitions with special symbols \[[link](https://cs.nyu.edu/~mohri/pub/tcs3.pdf)\]
    *   There exist algorithms for approximate determinization \[[link](http://www.faculty.idc.ac.il/udiboker/files/ApproxDetF.pdf)\]
    *   Disambiguation is an alternative. It applies to a wider class of automata and can lead to exponentially smaller results \[[link](https://cs.nyu.edu/~mohri/pub/dis.pdf)\]. There are algorithms for testing it \[[link](https://cs.nyu.edu/~mohri/pub/namb.pdf)\]
*   Weighted DFA minimization \[[link](https://cs.nyu.edu/~mohri/pub/fla.pdf)\]
		○ Push weights (corresponds to global probability normalization in our case)
		○ Minimize the automata using the regular DFA minimization algorithm, considering (weight, label) as a single edge label
*   Computing mode
    *   Is easy for Viterbi semiring, but NP-hard for the probabilistic semiring \[[link](http://aclweb.org/anthology/W/W13/W13-18.pdf)\]
    *   Should be easy for unambiguous automata (see determinization)
    *   There are some works on the algorithms for the general case
*   Learning probabilistic automata from samples
    *   A WFA-based distribution over sequences is identifiable in the limit from samples \[[link](https://pdfs.semanticscholar.org/aaae/e4a7f71f030536d67aa801dd07f2532838ee.pdf)\]
    *   Learning can be seen as a matrix completion problem \[[link](http://papers.nips.cc/paper/4697-spectral-learning-of-general-weighted-automata-via-constrained-matrix-completion.pdf)\]
    *   There was a competition, lots of datasets and results are available \[[link](http://ai.cs.umbc.edu/icgi2012/challenge/Pautomac/index.php)\]
*   Distances between automata
    *   KL-divergence between unambiguous (no two successful paths are labelled with the same string) probabilistic WFSA can be efficiently computed \[[link](https://cs.nyu.edu/~mohri/pub/kl.pdf)\]
    *   L2 distance between arbitrary weighted automata can be computed in polynomial time  \[[link](https://cs.nyu.edu/~mohri/pub/lpnorm.pdf)\]

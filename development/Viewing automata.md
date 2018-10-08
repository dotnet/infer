---
layout: default
---
[Infer.NET development](index.md)

## Viewing automata

You can create a GraphViz file from a SequenceDistribution by calling `ToString` on the distribution using the   `SequenceDistributionFormats.GraphViz` format and writing to file.

You can use [http://dot-graphics1.appspot.com/](http://dot-graphics1.appspot.com/) to view small automata by copying and pasting the string from the file to the window.

To view large automata, install GraphViz, and run `dot.exe`. The following example creates a jpeg from the file containing the graph description:

```shell
dot -Tjpeg myautomaton.txt > myautomaton.jpeg
```
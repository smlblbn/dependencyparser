# dependencyparser
Graph dependency parser

In this project, graph dependency parser was implemented. The parser works in three steps. 
The first one is to assign (unlabelled) relation scores between the words in a sentence.
The second the step is to extract the maximum spanning tree which will be the dependency tree based on these scores. 
The final step is to label the relations.

There are two models in this project. 
The first one assign scores to the possible relations (edges in the graph).
The second one classify those edges with dependency relations.

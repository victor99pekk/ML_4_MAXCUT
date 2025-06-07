
# Maximum cut: <br>`Heuristic algorithm` vs `Deep Learning`
In this experiment we compare the performance of a `heuristic approximation` algorithm for maximum cut called `Goemans-Williamson`, with two different `deep learning` approaches to solve the maxmimum cut problem the best we can. A thourugh description about this experiment can be found in our [report](report/report.pdf). But below is a brief description.

## `Deep Learning` approaches
We used 2 different deep learning approaches. The code for the two models can be found in [neural_network/models](neural_network/models). We describe the architectures of our deep learning models throughly in our [paper](report/report.pdf), but below is a brief description is:

The first model we implemented was a `LSTM`-model, where one of the LSTM-cells was used as an encoder and the other was used as a decoder. The LSTM approach was inspired from the paper ().

We then had the natural idea of improving this LSTM model with a `transformer` model. The transformer model used the same intuition to solve and model the maxcut problem but with a transformer instead



## Graphs

Both the graphs on the data `training` data and `test` data were generated the same way. Since solving the maximum cut is NP-hard, it will be hard to generate a large number of training examples since we always need the target for every data point, we would need to solve a NP-hard problem to get the target for every datapoint. 

We therefore try to `plant a solution` using an approach from (), where we try to transform the {-1, 1} quadratic programming problem into the maxcut problem. The code for doing this can be found in [`data/generate_data/maxcut_ml.py`](data/generate_data/maxcut_ml.py).

__Potential Benchmark:__ The Biq Mac Library contains a diverse collection of benchmark instances for the Max-Cut problem, and depending on the subdirectory or generator, the probability that two nodes are connected depends on the instance type and its associated edge density parameter.

Link: `https://biqmac.aau.at/biqmaclib.html`

---
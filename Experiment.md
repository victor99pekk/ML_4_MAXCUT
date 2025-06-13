
# Experiment Outline

Link to experiment report: https://www.overleaf.com/project/683f1e355a717aa8e1eac98f

## Test Data
We will try different different maximum cut algorithms on different sized graphs where the size ranges from 5 to 100. Every graph will be represented by an adjacency matrix where every entry will have 25% chance of being a 1, and 25% or being a 0. The graph size will take on the values $$sizes = \{5, 10, 20, 30, 50, 70, 100\}$$

There will be 1000 different graphs of each size.

### Path to test Data
`data/test/n=x` where $x \in sizes$ and $y \in [0, 99]$
row `i` will correspond to the `i`:th graph of that size

## Results
save the output for algorithm `z` in the path as a `csv` file

path: `results/z/n=x/`  where $x \in sizes$ and $y \in [0, 99]$ 

The row `i` of file `results/z/n=x` will correspond to the maximum cut for algorithm `z` produced for the `i`:th graph of size `x`.

__Bonus:__ save plots and other data too if possible


## Evaluation
- `neural network` vs `SDP`. win or loose
- divide by optimal cut
- inference time
- training (nn)
- perfect optimal obtained (..percentage of times)


## Methods
1. neural network
2. brute force
3. goemanss williamson 
   - preferably in c so we can check the inference difference
   - (potentially the improved one too)


## Report
1. introduction
2. theory
   - max cut
   - SDP, theory
   - Neural Network, theory
3. experiments
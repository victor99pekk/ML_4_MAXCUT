
# Experiment Outline

Link to experiment report: https://www.overleaf.com/project/683f1e355a717aa8e1eac98f

## Test Data
we will try on different type of graphs

### planted solutions
3 graphs, of sizes 10, 30, 50, for both graphs that contain planted solutions



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
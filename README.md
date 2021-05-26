# Adaboost
Machine Learning course - Implementing Adaboost algorithm

Given data rectangle.txt, that looks like that: 

![](https://github.com/HilaShoshan/Adaboost/blob/main/pictures/rectangle_data.png)

Each pair of points can define a line that passes through the two points. 
The set of all such lines forms is our hypothesis set, that is set of rules.

We've implement Adaboost using the above set of rules. 

One run of Adaboost containing: 
- spliting the data randomly into 0.5 test (T) and 0.5 train (S). 
- identifing the 8 most important lines hi on S, and their respective weights 𝛼𝑖.
(Adaboost's k parameter is k=8)
- for eack t=1,..8 computing the empirical error of the function Ht on the training set, and the true error of Ht on the test set.

We've execute 40 runs of Adaboost, and here's the errors results: 

![](https://github.com/HilaShoshan/Adaboost/blob/main/pictures/errors_lists_results.jpeg)

![](https://github.com/HilaShoshan/Adaboost/blob/main/pictures/errors_graph_results.jpeg)

The 𝑒̅(𝐻𝑘) and 𝑒(𝐻𝑘) averaged over the 40 runs: 
Emprirical errors mean: 0.14
True errors mean: 0.25

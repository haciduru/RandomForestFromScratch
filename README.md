Random-Forest-From-Scratch
I appreciate all the excellent work and effort of people who have created machine learning libraries such as SciKit Learn, PyTorch, et cetera. However, I still don't like to use other people's codes until I know what is going on in the background. Therefore, I usually implement my own tools as a learning experience and also to understand the logic and the mechanism of machine learning. Here, I will present my own random forest and decision tree models that I used to predict survival in Titanic data.

I assume that everybody here knows what a random forest and what a decision tree is. If not, there are excellent tutorials out there that you can read (I am not sure if I can share their links here). Both random forests and decision trees are analytical tools that we use to solve classification problems. A forest is an aggregate of decision trees. Each tree in the forest makes a prediction and casts a vote based on that prediction. The forest aggregates the votes and makes the final decision. Thus, the real job is done by the decision tree. But, what is a decision tree?

A decision tree is a map of the training dataset that looks just like a tree; it has branches and leaves. Leaves hang on branches. Actually, we can call leaves terminal branches. Leaves are also sets of decision rules. We make predictions about the testing dataset using those rules. Then, growing a decision tree is essentially creating sets of rules that can be used to make predictions. In this text, I will explain how you can grow a decision tree using the Titanic Disaster Dataset. You can find the dataset here: https://www.kaggle.com/c/titanic/data.

In the Titanic case, we want to predict who among the passengers would survive and who would die. Thus, survival is our outcome (or dependent) variable. We make that prediction using the passengers' characteristics, such as their sex, age, passenger class, fare, et cetera. These characteristics are called the predictor (or independent) variables.

Growing a decision tree
While growing a decision tree, we, iteratively, split the data into subsets based on the independent variables. The collection of subsets after the final iteration is the leaves of the tree. We use those leaves (i.e., decision rules) to make our prediction. Here is how it happens. Let's assume that we only have information on two of the independent variables: (1) passenger class and (2) sex. We can split the data into two subsets using one of these two variables--say, we do that using the passenger class. If we do that, we will have the following three new branches, because there are three passenger classes in Titanic.

Branch 1: passenger class = 1, N = 216

Branch 1: passenger class = 2, N = 184

Branch 3: passenger class = 3, N = 491
Then we can split these branches into subsets using the sex variable. Since this is the final iteration, the collection of subsets after this iteration will be the leaves of our tree. Note that we don't have to split all three of the branches. Finally, we calculate the proportion of passengers who survived in each leaf. Here are the leaves:

Leaf 1: passenger class = 1, sex = female, N = 94, 97% survived

Leaf 2: passenger class = 2, sex = female, N = 76, 92% survived

Leaf 3: passenger class = 3, sex = female, N = 144, 50% survived

Leaf 4: passenger class = 1, sex = male, N = 122, 37% survived

Leaf 5: passenger class = 2, sex = male, N = 108, 16% survived

Leaf 6: passenger class = 3, sex = male, N = 347, 14% survived
We create these rules using the training data. We use them to make predictions in the testing data. Here are the rules for each leaf:

Decision rule 1: if the passenger is in passenger class 1 and is a female, then the prediction = survived

Decision rule 2: if the passenger is in passenger class 2 and is a female, then the prediction = survived

Decision rule 3: if the passenger is in passenger class 3 and is a female, then the prediction = ?

Decision rule 4: if the passenger is in passenger class 1 and is a male, then the prediction = died

Decision rule 5: if the passenger is in passenger class 2 and is a male, then the prediction = died

Decision rule 6: if the passenger is in passenger class 3 and is a male, then the prediction = died
Here are two important questions to answer. How many leaves (i.e., decision rules) should we create? And, does it matter whether we start splitting the data into branches using variable A vs. variable B, for example, passenger class versus sex in the Titanic example. The answer to both of these questions is: it depends. It depends on how deep you want to go.

In our example, there are only two variables (passenger class and sex). There are three unique values in passenger class and two unique values in sex. Thus, the maximum number of leaves that we can create is six. Because six is a small number, we can go until the end and create all six of the leaves. If this is the case, it does not matter whether we start splitting the data using passenger class or sex variables. What if we had more than two variables? For example, if we had five variables, two of them coded as yes/no, two of them with three unique values each (e.g., high, medium, low), and one of them with five unique values (e.g., age 0-12, 13-18, 19-32, 33-45, 46+). In that case, the maximum number of leaves we could create would be 2 2 3 3 4 = 144.

We usually have more than five variables, and the maximum number of leaves that we can create increases exponentially with each additional variable. If we have ten variables with two unique values each (e.g., yes/no), the maximum number of leaves we can create is 2^10 = 1,024. It is neither practical nor necessary to create that many leaves. And when we do not create all possible leaves, it matters whether we start splitting the data with variable A vs. variable B. Then, the question becomes, how do we decide where to start?

There are methods to help us make that decision. But, before talking about that, let us think about the leaves. Remember that each leave is also a decision rule. What kind of leaves do we want to have? Consider the following three leaves:

Leaf 1: Passenger class = 1, sex = female, N = 94, 97% survived

Leaf 2: Passenger class = 3, sex = female, N = 144, 50% survived

Leaf 3: Passenger class = 3, sex = male, N = 347, 13% survived
Which one of these leaves (or decision rules) is the most helpful?

If I see a female passenger in passenger class 1 in the testing dataset, I can easily say that she probably survived (see Leaf 1). That is, the first leaf above (the first decision rule) is very helpful. Likewise, if I see a male passenger in passenger class 3 in the testing dataset, I can easily say that he probably died (see Leaf 3). Thus, the third leaf is also very helpful. However, if I see a female passenger in passenger class 3, I cannot say whether she survived or died (see Leaf 2). The second leaf is not helpful because it says that the passengers in this group have a 50% chance of survival. I can make the same prediction using the flip of a fair coin. We want to create leaves like the first and the third ones. We do not want leaves like the second one.

Entropy
The first and third leaves above have one thing in common. They are more homogenous regarding the dependent variable compared to the second leaf. That is, the passengers within these two leaves are similar to each other. The probability of two random passengers within the leaf to be similar to each other is high. On the other hand, the likelihood of two random passengers in the second leaf to be similar to each other is lower. A measure of this homogeneity/heterogeneity is called entropy. Higher entropy means higher heterogeneity. The formula for entropy is:

entropy = –(p log2(p)) –(q log2(q))
In the Titanic case, p = proportion survived, q = 1 - p (i.e., proportion died)

According to this formula, the above three leaves have the following entropy values:

Leaf 1: -(.97 log2(.97)) -(.03 log2(.03)) = .19

Leaf 2: -(.5 log2(.5)) -(.5 log2(.5)) = .1

Leaf 3: -(.13 log2(.13)) -(.87 log2(.87)) = .56
Information gain
When we split branches into sub-branches or leaves, we try to do that in such a way that the new branches/leaves have low entropy. The goal is to have leaves that have the lowest possible entropy. Another measure that we use to achieve this goal is "information gain." Information gain is the difference between the parent branch's entropy value and the weighted totals of the child branches' entropy values. Below is an example of how we calculate information gain.

In total, 38% of the Titanic passengers in the training dataset survived. Thus, the entropy value for the entire training dataset is (i.e., the parent branch):

-(.38 log2(.38)) -(.62 log2(.62)) = .96
If we split the data into three sub-branches on the passenger class variable, the sizes and entropy values of the branches will be:

Branch 1:

N = 216, 63% survived

entropy = -(.63 log2(.63)) -(.37 log2(.37)) = .95
Branch 2:

N = 184, 47% survived

entropy = -(.47 log2(.47)) -(.53 log2(.53)) = .98
Branch 3:

N = 491, 24% survided

entropy = -(.24 log2(.24)) -(.76 log2(.76)) = .80
Information gain = .96 - ( (216/891 .95) + (184/891 .98) + (491/891 * .80)) = .09

Note that we weighted the sub-branches' entropy values by multiplying them with the proportion of passengers in each branch. For example, we multiplied the entropy value of Branch 1 by 216/891, because 216 of the 891 passengers were in this sub-branch (i.e., passenger class = 1).

If we split the entire training dataset into branches on the passenger class variable, the amount of information that we will gain will be .05. If we split the entire training dataset into branches on sex, the amount of information that we will gain will be .22 (you can calculate the information gain as an exercise).

Remember the two questions that I asked above. First, how many leaves (i.e., decision rules) should we create? Second, does it matter whether we start splitting the data into branches using this variable vs. that variable (e.g., passenger class vs. sex)? It depends. If we are going to split the data until there are no more splits, i.e., we want to create the maximum number of leaves possible, then it does not matter whether we start with this or that variable. However, it is neither practical nor necessary to create the maximum number of leaves possible. Moreover, if we create too many leaves, we will probably overfit the data to the training set, and our tree will not perform well in the testing set. Thus, we do not want to create many leaves. We want to create a few (no more than enough) leaves. Therefore, we use the information gain measure to decide which variables we will use to create new branches. We split each branch into subbranches using the variable that produces the highest information gain.

When we calculated the information gain above, we weighted the entropy values of child branches by their size. Why? The logic is very simple; there are more cases in larger branches. That makes sense, right?

Entropy and information gain are the most fundamental ideas about decision trees. If you know what they are, then you know what a decision tree is. All the rest is a coding practice. Below, I will show you how to create a decision tree using R.

First, let us import the data. Because we will rund the code step by step so that we can see what happens every step.

In [1]:
library(tidyverse)
data = read.csv("../input/titanictrainset/train.csv")
head(data)
── Attaching packages ─────────────────────────────────────── tidyverse 1.3.0 ──

✔ ggplot2 3.3.2.9000     ✔ purrr   0.3.4     
✔ tibble  3.0.1          ✔ dplyr   1.0.0     
✔ tidyr   1.1.0          ✔ stringr 1.4.0     
✔ readr   1.3.1          ✔ forcats 0.5.0     

── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
✖ dplyr::filter() masks stats::filter()
✖ dplyr::lag()    masks stats::lag()

A data.frame: 6 × 12
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
<int>	<int>	<int>	<fct>	<fct>	<dbl>	<int>	<int>	<fct>	<dbl>	<fct>	<fct>
1	1	0	3	Braund, Mr. Owen Harris	male	22	1	0	A/5 21171	7.2500		S
2	2	1	1	Cumings, Mrs. John Bradley (Florence Briggs Thayer)	female	38	1	0	PC 17599	71.2833	C85	C
3	3	1	3	Heikkinen, Miss. Laina	female	26	0	0	STON/O2. 3101282	7.9250		S
4	4	1	1	Futrelle, Mrs. Jacques Heath (Lily May Peel)	female	35	1	0	113803	53.1000	C123	S
5	5	0	3	Allen, Mr. William Henry	male	35	0	0	373450	8.0500		S
6	6	0	3	Moran, Mr. James	male	NA	0	0	330877	8.4583		Q
We don't need most of the columns for now. Let us keep the ones that we need.

In [2]:
data = with(data, data.frame(Survived = Survived, Pclass = Pclass, Sex = Sex))
head(data)
A data.frame: 6 × 3
Survived	Pclass	Sex
<int>	<int>	<fct>
1	0	3	male
2	1	1	female
3	1	3	female
4	1	1	female
5	0	3	male
6	0	3	male
To create a decision tree, first we need a function to plant the tree. This is a short function and it is below. The function gets a dataframe or a matrix as the input value.

In [3]:
## -----------------------------------------
# This function initializes a decision tree.
# input:
#    dta (matrix/data frame): data
# returns:
#    tree (list): a list of tree branches.
plant = function(data) {

    tree = list(vars = c(), data = data)
    tree = create_branches(tree)
    return(tree)

}
The first line of the function

tree = list(vars = c(), data = data)

creates a list that has two items. This list is the trunk of the tree. It is also the first parent branch. The first item in the list is an empty vector, for now. The second item is the data. The second line of the function

tree = create_branches(tree)

creates the first branches of the tree from its trunk. Since we haven't created the create_branches function, we cannot run this plant function yet.

We also need a function to calculate entropy. This function takes a vector of dummy coded (i.e., 1/0) output values. In our example, it is the first column of the Titanic dataset. If all cases are 1 or 0, there is no entropy, and the function returns 0. Otherwise, it calculates the entropy using the formula and returns the entropy value.

In [4]:
## -----------------------------------------------------------------------------
# This function calculates information entropy.
# input: 
#     var (vector): of dummy values.
# returns:
#     entropy (float): entropy
entropy = function(var) {

    p = mean(var)
    if (p == 1 | p == 0) {
        return(0)
    } else {
        q = 1 - p
        return(
            -p * log(p, 2)    # -p(1) * log2(p(1)) 
            -q * log(q, 2)    # -p(2) * log2(p(2))
            )
    }

}
Let us try and see how this function works.

In [5]:
entropy(data[,1])
0.960707901875647
The entropy function says that the amount of entropy of the tree trunk is .96.

Next, we need a function that will help us find the variable that will produce the most information gain when we create new sub-branches from an existing branch. This function must take a subset of the dataset (i.e., an existing branch) as the input value. Then, it must calculate the information gain for each variable. Finally, it must suggest the next split variable, the variable that offers the highest information gain if the data is split on that variable. The following function does that.

In [6]:
## -------------------------------------------------------
# This function finds the variable that offers the highest
# information gain if the data is split on that variable.
# input:
#     data (matrix/data frame): data
# returns:
#    ii (int): index of the variable with smallest entropy.
find_branch_to_split = function(data) {

    min_entropy = 1
    ii = 0
    if (ncol(data) > 1) {
        for (i in 2:ncol(data)) {
            levels = unique(data[,i])
            if (length(levels) > 0) {
                new_entropy = 0
                for (j in 1:length(levels)) {
                    level = data[which(data[,i] == levels[j]), 1]
                    new_entropy = new_entropy + entropy(level) * length(level) / nrow(data)
                    if (!is.na(new_entropy) & new_entropy < min_entropy) { 
                        min_entropy = new_entropy
                        ii = i
                    }
                }
            }
        }
    }
    return(ii)

}
Let us see what this function does step by step.

First, the function sets the minimum entropy value to 1 with the following line.

min_entropy = 1

The following loop iterates the code inside for all independent variables in the dataset:

for (i in 2:ncol(data) {

    ...

}

For each independent variable, it initiates an empty entropy value: new_entropy with the following line:

new_entropy = 0

Then, for each sub-branch that can be created by splitting the data on i'th independent variable, it fills the new_entropy by adding the weighted entropy value of that sub-branch. The function does that with the following loop:

for (j in 1:length(levels)) {

    level = data[which(data[,i] == levels[j]), 1]

    new_entropy = new_entropy + entropy(level) * length(level) / nrow(data)

    ...

}

If the new_entropy is smaller than min_entropy, the function sets min_entropy equal to new_entropy. It also stores the index of the variable that produces min_entropy (i.e., the smallest entropy thus far) into ii, with the following lines:

if (new_entropy < min_entropy) { 

    min_entropy = new_entropy

    ii = i

}

In the end, the function returns the index of the variable that produces the lowest entropy. In other words, it returns the index of the variable that produces the highest information gain. If we decide to split this branch into sub-branches, we must split the dataset on that variable.

Now let's try the function.

In [7]:
find_branch_to_split(data)
2
It says we will get the highest information gain if we split the data into new branches using the variable in the second column (i.e., passenger class, Pclass).

Since we know how to find the variable to split a branch into sub-branches, now we can write the function that actually creates new branches. It is below.

In [8]:
## -------------------------------------------------------
# This function creates new branches from a parent branch.
# input:
#    parent (list): parent branch
# returns:
#    children (list): a list of new children (i.e., branches)
create_branches = function(parent) {

    data = parent$data
    ii = find_branch_to_split(data)
    if (ii > 0) {
        levels = unique(data[,ii])
        if (length(levels) > 0) {
            children = vector("list", length(levels))
            for (i in 1:length(levels)) {
                child = data.frame(data[which(data[,ii] == levels[i]), -ii])
                if (ncol(child) == 1) names(child) = names(data)[1]
                if (length(parent[[1]]) > 0) {
                    value = rbind(parent$vars, c(names(data)[ii], levels[i]))
                } else {
                    value = matrix(c(names(data)[ii], levels[i]), nrow=1)
                }
                children[[i]] = list(vars = value, data = child)
            }
            return(children)
        } else {
            return(list())
        }
    } else {
        return(list())
    }

}
The create_branches function gets a branch as the input parameter. Remember from the plant function above that a branch is a list of two items. The first item of this list is a vector and the second item is the data (a dataframe or a matrix). The line of the create_branches function stores the second item of that list into a local variable, data.

data = parent$data

The second line finds the index (ii) of the variable that will produce the highest information gain (i.e., the lowest entropy) if we split the branch into sub-branches using that variable.

ii = find_branch_to_split(data)

If there is such a variable, the following line gets the unique values of that variable into a local variable, levels.

levels = unique(data[,ii])

Then, if there is any variation in the split variable (i.e., the ii'th variable), the following line initializes children branches.

children = vector("list", length(levels))

Then, the following loop creates each child branch and puts that child in children.

for (i in 1:length(levels)) {

    ...

}

The following lines in the loop creates the data item of the child branch.

child = data.frame(data[which(data[,ii] == levels[i]), -ii])

if (ncol(child) == 1) names(child) = names(data)[1]

And the following lines create the vars item of the child branch.

if (length(parent[[1]]) > 0) {

    value = rbind(parent[[1]], c(names(data)[ii], levels[i]))

} else {

    value = matrix(c(names(data)[ii], levels[i]), nrow=1)

}

Remember that there are two items in each branch. The first one is vars. This is the item that we create here. So, what do we put in vars? We put the name of the variable that we used to split the parent branch into sub-branches and also the value of that variable for this sub-branch. For example, "passenger class" "1". Not only that. We also inherit the vars item from the parent branch. I will show an example of this below.

Lastly, the following line adds the new child into children, and then the function returns children.

children[[i]] = list(vars = value, data = child)    
Since we have a function to plant the tree and a function to create new branches from parent branches, now we can write the function that grows the tree. But, before doing that, let's try these two functions.

In [9]:
tree = plant(data)
length(tree)
3
Our tree has three branches for now. That is the reason why its length is 3. Now let us see what is in these three branches.

In [10]:
tree
$vars
A matrix: 1 × 2 of type chr
Pclass	3
$data
A data.frame: 491 × 2
Survived	Sex
<int>	<fct>
1	0	male
3	1	female
5	0	male
6	0	male
8	0	male
9	1	female
11	1	female
13	0	male
14	0	male
15	0	female
17	0	male
19	0	female
20	1	female
23	1	female
25	0	female
26	1	female
27	0	male
29	1	female
30	0	male
33	1	female
37	1	male
38	0	male
39	0	female
40	1	female
41	0	female
43	0	male
45	1	female
46	0	male
47	0	male
48	1	female
⋮	⋮	⋮
838	0	male
839	1	male
841	0	male
844	0	male
845	0	male
846	0	male
847	0	male
848	0	male
851	0	male
852	0	male
853	0	female
856	1	female
859	1	female
860	0	male
861	0	male
864	0	female
869	0	male
870	1	male
871	0	male
874	0	male
876	1	female
877	0	male
878	0	male
879	0	male
882	0	male
883	0	female
885	0	male
886	0	female
889	0	female
891	0	male
$vars
A matrix: 1 × 2 of type chr
Pclass	1
$data
A data.frame: 216 × 2
Survived	Sex
<int>	<fct>
2	1	female
4	1	female
7	0	male
12	1	female
24	1	male
28	0	male
31	0	male
32	1	female
35	0	male
36	0	male
53	1	female
55	0	male
56	1	male
62	1	female
63	0	male
65	0	male
84	0	male
89	1	female
93	0	male
97	0	male
98	1	male
103	0	male
111	0	male
119	0	male
125	0	male
137	1	female
138	0	male
140	0	male
152	1	female
156	0	male
⋮	⋮	⋮
764	1	female
766	1	female
767	0	male
780	1	female
782	1	female
783	0	male
790	0	male
794	0	male
797	1	female
803	1	male
807	0	male
810	1	female
816	0	male
821	1	female
823	0	male
830	1	female
836	1	female
840	1	male
843	1	female
850	1	female
854	1	female
857	1	female
858	1	male
863	1	female
868	0	male
872	1	female
873	0	male
880	1	female
888	1	female
890	1	male
$vars
A matrix: 1 × 2 of type chr
Pclass	2
$data
A data.frame: 184 × 2
Survived	Sex
<int>	<fct>
10	1	female
16	1	female
18	1	male
21	0	male
22	1	male
34	0	male
42	0	female
44	1	female
54	1	female
57	1	female
59	1	female
67	1	female
71	0	male
73	0	male
79	1	male
85	1	female
99	1	female
100	0	male
118	0	male
121	0	male
123	0	male
124	1	female
134	1	female
135	0	male
136	0	male
145	0	male
146	0	male
149	0	male
150	0	male
151	0	male
⋮	⋮	⋮
733	0	male
734	0	male
735	0	male
748	1	female
751	1	female
755	1	female
756	1	male
758	0	male
773	0	female
775	1	female
792	0	male
796	0	male
801	0	male
802	1	female
809	0	male
813	0	male
818	0	male
828	1	male
832	1	male
842	0	male
849	0	male
855	0	female
862	0	male
865	0	male
866	1	female
867	1	female
875	1	female
881	1	female
884	0	male
887	0	male
You need to scroll down to see all three of the branches. Each branch has two items. The first one is a matrix and it shows the variable that we used to split the data to create the current branch. It also shows the value of that variable for this branch. That matrix in the first branch is "Pclass 3". This means, passenger class is 3 for every case in this branch. The second item of the branch is the actual data. Now there are two columns: Survived and Sex. We do not see the passenger class (i.e., Pclass) variable any more, because the value of that variable for all cases in this branch is 3. It is not a variable any more, it is a constant.

If you scroll down and see the other branches, you will see that they all have the same structure. The first item is a matrix and the second item is a data frame that holds actual data. The first item in the second branch is "Pclass 1," becasue all cases in this branch have 1 as the passenger class value. The first item in the third branch is "Pclass 2". Again, it is because all cases in that branch have 2 as their passenger class value.

Now let's see the function that grows the tree. What does that function do?

In [11]:
## --------------------------------------------
# This function grows a decision tree.
# input:
#    tree (list): a decision tree
#    min_split (int): minimum number of cases necessary
#                    to split a branch
# returns:
#    tree (list): a decision tree
grow_tree = function(tree, min_split, min_leaf) {

    # grow
    i = 1
    while (i <= length(tree)) {
        if (nrow(tree[[i]]$data) > min_split & ncol(tree[[i]]$data) > 1) {
            branches = create_branches(tree[[i]])
            if (length(branches) > 0) {
                tree = unlist(list(tree[-i], branches), recursive=F)
            } else {
                i = i + 1
            }
        } else {
            i = i + 1
        }
    }
    
    # trim
    i = 1
    while (i <= length(tree)) {
        if (nrow(tree[[i]]$data) < min_leaf) {
            tree = tree[-i]
        } else {
            i = i + 1
        }
    }
    
    return(tree)

}
This function takes three input parameters. The first one is a tree to grow. The second parameter is a value that will help us decide where to stop growing the tree. We will use the last parameter to trim the tree. The function has two sections. The first section grows the tree and the second one trims the tree. Let us see what happens in this function.

The min_split parameter tells us the minimum number of cases in a branch for us to be able to split that branch. For example, if min_split is 20, we cannot split branches that have less than 20 cases into new branches. That is the point where we stop growing the tree.

The min_leaf variable tells us the minimum number of cases that we need in a leaf. We will use that variable to prune the tree. We will delete leaves that have less cases than min_leaf in the second section of the function.

Let's see what happens in the first half of the function.

The while loop starts with the first branch of the three and does not stop until its index (i.e., i) passes the last branch.

i = 1

while (i < length(tree)) {

    ...

}

The code inside the while loop takes the i'th branch and checks for two things. First, are there more cases in the branch data than min_split. Second, are there any variables left in the dataset that we can use to create new branches. If there are more cases than min_split and there are variables that we can use to split the data into new branches, then we can create new branches. Note that every time we split a parent branch into child branches, we remove the variable that we use to split the data from the dataset. When we created the first branches of the three in the plant function, we split the data using Pclass variable and then we removed that variable from child branches. We do not need that variable in any of the child branches any more, because it is not a variable any more, it is constant. This is the reason why we check whether there are any variables left that we can use to create new branches.

Let me show you what I mean here by running the code inside. Let us set min_split as 50 for now.

In [12]:
min_split = 50

# Let's set i first

    i = 1
#    while (i <= length(tree)) {

# Then, see whether the if condition holds

    nrow(tree[[i]]$data) > min_split
    ncol(tree[[i]]$data) > 1
TRUE
TRUE
In [13]:
# Since both parts of the if condition hold, we can the code inside the if clause

#    i = 1
#    while (i <= length(tree)) {
        if (nrow(tree[[i]]$data) > min_split & ncol(tree[[i]]$data) > 1) {
            branches = create_branches(tree[[i]])
            if (length(branches) > 0) {
                tree = unlist(list(tree[-i], branches), recursive=F)
            } else {
                i = i + 1
            }
        } else {
            i = i + 1
        }
#    }

tree
$vars
A matrix: 1 × 2 of type chr
Pclass	1
$data
A data.frame: 216 × 2
Survived	Sex
<int>	<fct>
2	1	female
4	1	female
7	0	male
12	1	female
24	1	male
28	0	male
31	0	male
32	1	female
35	0	male
36	0	male
53	1	female
55	0	male
56	1	male
62	1	female
63	0	male
65	0	male
84	0	male
89	1	female
93	0	male
97	0	male
98	1	male
103	0	male
111	0	male
119	0	male
125	0	male
137	1	female
138	0	male
140	0	male
152	1	female
156	0	male
⋮	⋮	⋮
764	1	female
766	1	female
767	0	male
780	1	female
782	1	female
783	0	male
790	0	male
794	0	male
797	1	female
803	1	male
807	0	male
810	1	female
816	0	male
821	1	female
823	0	male
830	1	female
836	1	female
840	1	male
843	1	female
850	1	female
854	1	female
857	1	female
858	1	male
863	1	female
868	0	male
872	1	female
873	0	male
880	1	female
888	1	female
890	1	male
$vars
A matrix: 1 × 2 of type chr
Pclass	2
$data
A data.frame: 184 × 2
Survived	Sex
<int>	<fct>
10	1	female
16	1	female
18	1	male
21	0	male
22	1	male
34	0	male
42	0	female
44	1	female
54	1	female
57	1	female
59	1	female
67	1	female
71	0	male
73	0	male
79	1	male
85	1	female
99	1	female
100	0	male
118	0	male
121	0	male
123	0	male
124	1	female
134	1	female
135	0	male
136	0	male
145	0	male
146	0	male
149	0	male
150	0	male
151	0	male
⋮	⋮	⋮
733	0	male
734	0	male
735	0	male
748	1	female
751	1	female
755	1	female
756	1	male
758	0	male
773	0	female
775	1	female
792	0	male
796	0	male
801	0	male
802	1	female
809	0	male
813	0	male
818	0	male
828	1	male
832	1	male
842	0	male
849	0	male
855	0	female
862	0	male
865	0	male
866	1	female
867	1	female
875	1	female
881	1	female
884	0	male
887	0	male
$vars
A matrix: 2 × 2 of type chr
Pclass	3
Sex	2
$data
A data.frame: 347 × 1
Survived
<int>
0
0
0
0
0
0
0
0
0
1
0
0
0
0
0
0
0
0
0
0
0
1
0
0
0
1
0
0
0
0
⋮
0
0
0
1
0
0
0
0
0
1
0
0
0
0
0
0
0
0
0
0
0
1
0
0
0
0
0
0
0
0
$vars
A matrix: 2 × 2 of type chr
Pclass	3
Sex	1
$data
A data.frame: 144 × 1
Survived
<int>
1
1
1
0
0
1
1
0
1
1
1
0
1
0
1
1
0
1
0
1
1
1
0
1
1
0
0
0
0
1
⋮
0
1
0
0
1
1
0
1
0
0
0
1
1
1
0
1
0
0
0
0
1
1
0
1
1
0
1
0
0
0
This is what happened above. The code took the first branch and checked whether the conditions of the if statement holded. Then, it split the first branch (i.e., i'th) branch into two new branches, and added these two new branches to the end of the tree. Then, it deleted the i'th branch. If you scroll down the tree, you will see that branches 3 and 4 are children of the previous i'th branch. The vars items of these two new branches now have two rows. The first row says "Pclass 3" which indicates that both branch 3 and branch 4 are children of a branch that had "Pclass 3" as the vars item. Branches 3 and 4 have a second line in their vars item. This row is "Sex 2" for branch 3 and "Sex 1" for branch 4. These rows show that the previous branch was split into two new branches and the cases with Sex = 2 are put in branch 3 and the cases with Sex = 1 are put in branch 4. Also note that the data items of these two new branches now have only one column, Survived (i.e., the dependent variable). This means, we cannot split these branches any more.

If we run the whole while loop, it will split all branches that could be split into sub branches. If we run the second while loop, it will delete all branches that have less than min_leaf cases.

Let's just run the grow_tree function and see what happens. But before that, let's start over for a clean tree.

In [14]:
tree = plant(data)
tree = grow_tree(tree, 50, 20)
tree
$vars
A matrix: 2 × 2 of type chr
Pclass	3
Sex	2
$data
A data.frame: 347 × 1
Survived
<int>
0
0
0
0
0
0
0
0
0
1
0
0
0
0
0
0
0
0
0
0
0
1
0
0
0
1
0
0
0
0
⋮
0
0
0
1
0
0
0
0
0
1
0
0
0
0
0
0
0
0
0
0
0
1
0
0
0
0
0
0
0
0
$vars
A matrix: 2 × 2 of type chr
Pclass	3
Sex	1
$data
A data.frame: 144 × 1
Survived
<int>
1
1
1
0
0
1
1
0
1
1
1
0
1
0
1
1
0
1
0
1
1
1
0
1
1
0
0
0
0
1
⋮
0
1
0
0
1
1
0
1
0
0
0
1
1
1
0
1
0
0
0
0
1
1
0
1
1
0
1
0
0
0
$vars
A matrix: 2 × 2 of type chr
Pclass	1
Sex	1
$data
A data.frame: 94 × 1
Survived
<int>
1
1
1
1
1
1
1
1
1
1
0
1
1
1
1
1
1
1
1
1
1
1
1
1
0
1
1
1
1
1
⋮
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
$vars
A matrix: 2 × 2 of type chr
Pclass	1
Sex	2
$data
A data.frame: 122 × 1
Survived
<int>
0
1
0
0
0
0
0
1
0
0
0
0
0
1
0
0
0
0
0
0
0
0
0
0
0
1
1
1
0
1
⋮
0
0
1
1
1
0
0
1
1
0
1
1
1
1
0
0
0
0
0
0
0
1
0
0
0
1
1
0
0
1
$vars
A matrix: 2 × 2 of type chr
Pclass	2
Sex	1
$data
A data.frame: 76 × 1
Survived
<int>
1
1
0
1
1
1
1
1
1
1
1
1
1
1
0
1
1
1
1
1
1
0
1
1
1
1
1
1
0
1
⋮
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
0
1
1
0
1
1
1
1
$vars
A matrix: 2 × 2 of type chr
Pclass	2
Sex	2
$data
A data.frame: 108 × 1
Survived
<int>
1
0
1
0
0
0
1
0
0
0
0
0
0
0
0
0
0
0
0
0
1
0
1
0
0
0
0
1
0
0
⋮
0
1
0
0
0
0
0
0
0
0
0
0
0
0
1
0
0
0
0
0
0
0
1
1
0
0
0
0
0
0
This is our tree with all the information that we need to make predictions. There are six branches and we know what exactly we have in each branch. Since these are terminal branches, they are actually leaves of the tree. It is time to use that information and create decision rules. We do that using the following function.

In [15]:
## ----------------------------------------
# This function shows the parameters of a decision tree.
# input:
#    tree (list): a decision tree
# returns:
#    ret (data frame): a data frame that has params of 
#    the decision tree
show_tree = function(tree) {

    map = data.frame(id = c(1:length(tree)), p = NA, n = NA)
    for (i in 1:length(tree)) {
        map[i,]$p = round(mean(tree[[i]]$data[,1]), 2)
        map[i,]$n = nrow(tree[[i]]$data)
        for (j in 1:nrow(tree[[i]]$vars)) {
            if (tree[[i]]$vars[j,1] %in% names(map)) {
                map[i,tree[[i]]$vars[j,1]] = tree[[i]]$vars[j,2]
            } else {
                map = cbind(map, data.frame(dum = NA))
                names(map)[ncol(map)] = tree[[i]]$vars[j,1]
                map[i,ncol(map)] = tree[[i]]$vars[j,2]
            }
        }
    }
    map = within(map, rm(id))
    return(map)

}
This function first creates a data frame to store all decision rules in it with the following line.

map = data.frame(id = c(1:length(tree)), p = NA, n = NA)

The number of rows in this data frame equals the length of the tree. We will store the likelihood of survival for each leaf in p, and store the number of cases in each leaf in n.

The for loop iterates for all leaves of the tree and the code inside the loop does two things. First, it stores the p and n values in the respective columns. Second, if the column that we used to create the leaf already exists, it stores the value of that variable of the leaf in that column. Otherwise, it creates a new column for that variable and then stores the value of the variable of the leaf in that new column.

Let's run the function and see what it produces.

In [16]:
show_tree(tree)
A data.frame: 6 × 4
p	n	Pclass	Sex
<dbl>	<int>	<chr>	<chr>
0.14	347	3	2
0.50	144	3	1
0.97	94	1	1
0.37	122	1	2
0.92	76	2	1
0.16	108	2	2
These are the same decision rules that we created manually above. Note that Sex is coded as 1 = female, 2 = male.

Since we now have all we need to create a decision tree, we can write our function to grow a forest. But, let's talk more about random forest. A random forest is a bunch of trees that are created from the same dataset. However, each tree is created using a random sample from that dataset, hence the name random forest. Thus we need two additional parameters to grow a forest. First, the number of trees in the forest. Second, the number of cases that we will use to grow each tree of the forest. We also need a function to select those cases. There is a built-in sample function in R, but I like using my own functions. Therefore we will write a new sample function. Let's first write the sample function.

Here it is:

In [17]:
## -------------------------------------------
# This function selects a random sample from data
# input:
#    data (matrix/data frame): data
#    n (int): the sample size
# returns:
#    data (matrix/data frame): data
sample = function(data, n) {

    if (n < nrow(data)) {
        data$r = runif(nrow(data))
        data = data[order(data$r), -ncol(data)]
        return(data[1:n,])
    } else {
        return(data)
    }

}
This is a very simple function. It get two inputs; data and n (sample size). If n is smaller than the data length, then it selects a sample as follows. It first creates a random variable with the following line:

data$r = runif(nrow(data))

Then, it sorts the data on this new variable but discards the new variable with the following line:

data = data[order(data$r), -ncol(data)]

Lastly, it returns the first n rows from the sorted dataset.

If n is equal to or larger than the data length, then the function returns the entire dataset.

Now I can present the function that grows the forest. It is below.

In [18]:
## -------------------------------------------
# This function grows a forest from data.
# input:
#    data (matrix/data frame): data
#    size (int): number of cases in each tree
#    n (int): number of trees to grow
#    min_split (int): minimum number of cases to split a branch
#    min_leaf (int): minimum number of cases necessary in a branch
# returns:
#    forest (list): a list of decision trees
grow_forest = function(data, tree_size, n_tree, min_split, min_leaf) {

    forest = vector("list", n_tree)
    for (i in 1:n_tree) {
        tree = plant(sample(data, tree_size))
        tree = grow_tree(tree, min_split, min_leaf)
        forest[[i]] = show_tree(tree)
    }
    return(forest)

}
This is a very simple function. It gets five input parameters. Three of them are the parameters that we need to create a decision tree; data, min_split, and min_leaf. The other two parameters are the number of trees in the forest and the number of cases that we will use to create each three.

The following line initializes the forest:

forest = vector("list", n_tree)

The for loop plants and grows decision trees and puts them in the forest. The following two lines plant and grow decision trees.

tree = plant(sample(data, tree_size))

tree = grow_tree(tree, min_split, min_leaf)

The function does not store all the data of each three in the forest. Instead, it only stores the decision rules. That is we all need from the decision trees afterall. The function does that with the following line:

forest[[i]] = show_tree(tree)

There is one last thing that we need to do to complete this exercise. We need a function that gets a case and predicts the outcome using the random forest. That function is below.

In [19]:
## ------------------------------------------
# This function evaluates the likelihood of a
# case given a decision tree.
# input:
#    forest (list): a list of decision trees
#    case (vector): a single case (i.e., a vector of values) to evaluate.
# returns:
#    vals (matrix): an (n by 2) matrix. 
eval_case = function(forest, case) {

    ret = data.frame(p = NA, n = NA)
    for (i in 1:length(forest)) {
        tree = forest[[i]]
        j = 2
        while (nrow(tree) > 1) {
            j = j + 1
            tree = as.data.frame(tree[which(tree[,j] == case[,names(tree)[j]]), ])
        }
        ret = rbind(ret, tree[,1:2])
    }
    return(ret[-1,])

}
This function gets a forest and a case as input values. It returns a dataframe that has all the votes cast by each tree in the forest. The function initializes a dataframe to store the votes with the following line:

ret = data.frame(p = NA, n = NA)

Then, it requests each tree to vote using the code in the for loop. The the lines while loop searches the leaf that represents the case in the tree. If it finds the leaf, it adds the vote of the tree in the dataframe that was initialized with the above code.

Now we have all the code necessary to build a random forest. Let's use all the data available in the Titanic dataset and make a prediction. Because it takes a lot of time to clean the data, I will only use the variables that do not need much cleaning.

First, let's import the data again.

In [20]:
data = read.csv("../input/titanictrainset/train.csv")
head(data)
A data.frame: 6 × 12
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
<int>	<int>	<int>	<fct>	<fct>	<dbl>	<int>	<int>	<fct>	<dbl>	<fct>	<fct>
1	1	0	3	Braund, Mr. Owen Harris	male	22	1	0	A/5 21171	7.2500		S
2	2	1	1	Cumings, Mrs. John Bradley (Florence Briggs Thayer)	female	38	1	0	PC 17599	71.2833	C85	C
3	3	1	3	Heikkinen, Miss. Laina	female	26	0	0	STON/O2. 3101282	7.9250		S
4	4	1	1	Futrelle, Mrs. Jacques Heath (Lily May Peel)	female	35	1	0	113803	53.1000	C123	S
5	5	0	3	Allen, Mr. William Henry	male	35	0	0	373450	8.0500		S
6	6	0	3	Moran, Mr. James	male	NA	0	0	330877	8.4583		Q
But, this time, I will keep the following variables: Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Cabin, and Embarked.

In [21]:
data = within(data, rm(PassengerId, Name, Ticket))
head(data)
A data.frame: 6 × 9
Survived	Pclass	Sex	Age	SibSp	Parch	Fare	Cabin	Embarked
<int>	<int>	<fct>	<dbl>	<int>	<int>	<dbl>	<fct>	<fct>
1	0	3	male	22	1	0	7.2500		S
2	1	1	female	38	1	0	71.2833	C85	C
3	1	3	female	26	0	0	7.9250		S
4	1	1	female	35	1	0	53.1000	C123	S
5	0	3	male	35	0	0	8.0500		S
6	0	3	male	NA	0	0	8.4583		Q
Let's do some cleaning and encoding. First, let us keep the first letter of the Cabin variable.

In [22]:
data$Cabin = substr(data$Cabin, 1, 1)
head(data)
A data.frame: 6 × 9
Survived	Pclass	Sex	Age	SibSp	Parch	Fare	Cabin	Embarked
<int>	<int>	<fct>	<dbl>	<int>	<int>	<dbl>	<chr>	<fct>
1	0	3	male	22	1	0	7.2500		S
2	1	1	female	38	1	0	71.2833	C	C
3	1	3	female	26	0	0	7.9250		S
4	1	1	female	35	1	0	53.1000	C	S
5	0	3	male	35	0	0	8.0500		S
6	0	3	male	NA	0	0	8.4583		Q
Then, let us create a rich/poor variable by looking at the ticket fares. Those who paid more than the mean fare should be rich.

In [23]:
meanFare = mean(data$Fare)
data$rich = 0
data[which(data$Fare > meanFare), ]$rich = 1
data = within(data, rm(Fare))
head(data)
cat(round(mean(data$rich) * 100), "% of the passengers are rich")   # let us see percentage of rich passengers
A data.frame: 6 × 9
Survived	Pclass	Sex	Age	SibSp	Parch	Cabin	Embarked	rich
<int>	<int>	<fct>	<dbl>	<int>	<int>	<chr>	<fct>	<dbl>
1	0	3	male	22	1	0		S	0
2	1	1	female	38	1	0	C	C	1
3	1	3	female	26	0	0		S	0
4	1	1	female	35	1	0	C	S	1
5	0	3	male	35	0	0		S	0
6	0	3	male	NA	0	0		Q	0
24 % of the passengers are rich
Let's also encode age so that we can use it. Note that age is missing for 177 of the passengers. I encode them as 6 in the new age variable.

In [24]:
data$age = NA
data[which(data$Age < 12 & is.na(data$age)), ]$age = 1
data[which(data$Age < 19 & is.na(data$age)), ]$age = 2
data[which(data$Age < 33 & is.na(data$age)), ]$age = 3
data[which(data$Age < 45 & is.na(data$age)), ]$age = 4
data[which(data$Age > 44 & is.na(data$age)), ]$age = 5
data[which(is.na(data$Age)), ]$age = 6

aggregate(cbind(n = Survived) ~ age, data = data, function(x){NROW(x)})
A data.frame: 6 × 2
age	n
<dbl>	<int>
1	68
2	71
3	309
4	151
5	115
6	177
We don't need the raw age values any more. Let's drop them.

In [25]:
data = within(data, rm(Age))
head(data)
A data.frame: 6 × 9
Survived	Pclass	Sex	SibSp	Parch	Cabin	Embarked	rich	age
<int>	<int>	<fct>	<int>	<int>	<chr>	<fct>	<dbl>	<dbl>
1	0	3	male	1	0		S	0	3
2	1	1	female	1	0	C	C	1	4
3	1	3	female	0	0		S	0	3
4	1	1	female	1	0	C	S	1	4
5	0	3	male	0	0		S	0	4
6	0	3	male	0	0		Q	0	6
Now, we are ready to grow a forest. To make it simple, I will put only three trees in teh forest.

In [26]:
forest = grow_forest(data, 400, 3, 30, 5)
forest
A data.frame: 22 × 10
p	n	Cabin	age	Embarked	Pclass	Sex	Parch	rich	SibSp
<dbl>	<int>	<chr>	<chr>	<chr>	<chr>	<chr>	<chr>	<chr>	<chr>
0.92	13	D	NA	NA	NA	NA	NA	NA	NA
0.88	17	E	NA	NA	NA	NA	NA	NA	NA
0.60	30	C	NA	NA	NA	NA	NA	NA	NA
0.71	17	B	NA	NA	NA	NA	NA	NA	NA
0.43	7	A	NA	NA	NA	NA	NA	NA	NA
0.48	23		1	NA	NA	NA	NA	NA	NA
0.39	23		2	NA	NA	NA	NA	NA	NA
0.33	24		5	NA	NA	NA	NA	NA	NA
0.14	14		6	2	NA	NA	NA	NA	NA
0.50	20		6	3	NA	NA	NA	NA	NA
0.39	18		4	NA	2	NA	NA	NA	NA
0.60	5		4	NA	1	NA	NA	NA	NA
0.10	21		3	NA	2	2	NA	NA	NA
0.93	14		3	NA	2	1	NA	NA	NA
0.42	19		3	NA	3	1	NA	NA	NA
0.07	27		6	4	3	NA	NA	NA	NA
0.05	19		4	NA	3	NA	0	NA	NA
0.20	5		4	NA	3	NA	5	NA	NA
0.17	6		4	NA	3	NA	1	NA	NA
0.25	8		3	2	3	2	NA	NA	NA
0.00	6		3	4	3	2	NA	0	1
0.17	41		3	4	3	2	0	0	0
A data.frame: 24 × 10
p	n	Cabin	age	Sex	SibSp	Embarked	Parch	Pclass	rich
<dbl>	<int>	<chr>	<chr>	<chr>	<chr>	<chr>	<chr>	<chr>	<chr>
0.85	13	D	NA	NA	NA	NA	NA	NA	NA
0.64	28	C	NA	NA	NA	NA	NA	NA	NA
0.33	9	A	NA	NA	NA	NA	NA	NA	NA
0.87	15	B	NA	NA	NA	NA	NA	NA	NA
0.40	5	F	NA	NA	NA	NA	NA	NA	NA
0.62	13	E	NA	NA	NA	NA	NA	NA	NA
0.64	22		1	NA	NA	NA	NA	NA	NA
0.21	19		5	NA	NA	NA	NA	NA	NA
0.62	26		3	1	NA	NA	NA	NA	NA
0.50	12		6	NA	1	NA	NA	NA	NA
0.12	17		2	2	NA	NA	NA	NA	NA
0.53	15		2	1	NA	NA	NA	NA	NA
0.20	15		3	2	NA	2	NA	NA	NA
0.69	13		6	1	0	NA	NA	NA	NA
0.05	19		4	NA	NA	NA	0	3	NA
0.45	11		4	NA	NA	NA	0	2	NA
0.80	5		4	NA	NA	NA	0	1	NA
0.06	18		3	2	NA	4	NA	2	NA
0.33	6		6	2	0	NA	NA	2	NA
0.14	7		3	2	1	4	NA	3	NA
0.11	44		3	2	0	4	0	3	0
0.00	23		6	2	0	4	0	3	0
0.00	7		6	2	0	2	0	3	0
0.00	13		6	2	0	3	0	3	0
A data.frame: 22 × 10
p	n	age	Embarked	Pclass	Cabin	Sex	SibSp	rich	Parch
<dbl>	<int>	<chr>	<chr>	<chr>	<chr>	<chr>	<chr>	<chr>	<chr>
0.57	30	1	NA	NA	NA	NA	NA	NA	NA
0.83	6	4	2	NA	NA	NA	NA	NA	NA
0.67	27	3	NA	1	NA	NA	NA	NA	NA
0.40	30	3	NA	2	NA	NA	NA	NA	NA
0.67	6	6	NA	NA	C	NA	NA	NA	NA
0.53	15	5	2	NA	NA	NA	NA	NA	NA
0.00	16	2	NA	NA	NA	2	NA	NA	NA
0.67	18	2	NA	NA	NA	1	NA	NA	NA
0.41	17	3	NA	3	NA	1	NA	NA	NA
0.60	10	6	NA	NA		NA	1	NA	NA
0.12	8	5	4	3	NA	NA	NA	NA	NA
0.33	12	5	4	2	NA	NA	NA	NA	NA
0.37	19	5	4	1	NA	NA	NA	NA	NA
0.56	9	4	4	NA		1	NA	NA	NA
0.12	26	6	4	NA		NA	0	NA	NA
0.25	12	6	2	NA		NA	0	NA	NA
0.53	17	6	3	NA		NA	0	NA	NA
0.00	7	4	4	NA		2	1	NA	NA
0.08	26	4	4	NA		2	0	NA	NA
0.40	5	3	2	3	NA	2	NA	0	NA
0.17	6	3	4	3	NA	2	1	0	NA
0.11	38	3	4	3		2	0	0	0
There are 3 trees in our forest. Each three is represented as a dataframe, and each line in these dataframes are leaves of a tree (or, decision rules). For exmaple, the first line of the first data frame (i.e., tree) is:

In [27]:
forest[[1]][1,]
A data.frame: 1 × 10
p	n	Cabin	age	Embarked	Pclass	Sex	Parch	rich	SibSp
<dbl>	<int>	<chr>	<chr>	<chr>	<chr>	<chr>	<chr>	<chr>	<chr>
1	0.92	13	D	NA	NA	NA	NA	NA	NA	NA
This decision rule says that if the passenger is a child (i.e., age < 12) s/he survived, because the probability of survival is 61%.

The second leaf (i.e., decision rule) of the first tree is:

In [28]:
forest[[1]][2,]
A data.frame: 1 × 10
p	n	Cabin	age	Embarked	Pclass	Sex	Parch	rich	SibSp
<dbl>	<int>	<chr>	<chr>	<chr>	<chr>	<chr>	<chr>	<chr>	<chr>
2	0.88	17	E	NA	NA	NA	NA	NA	NA	NA
This decision rule says that if the passenger's age is missing and passenger class is 2, then s/he died, because the probability of survival is 43%.

Now, lets predict a case from the testing set using this forest. We need to import the test set first.

In [29]:
library(tidyverse)
data = read.csv("../input/titanictestset/test.csv")
head(data)
A data.frame: 6 × 11
PassengerId	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
<int>	<int>	<fct>	<fct>	<dbl>	<int>	<int>	<fct>	<dbl>	<fct>	<fct>
1	892	3	Kelly, Mr. James	male	34.5	0	0	330911	7.8292		Q
2	893	3	Wilkes, Mrs. James (Ellen Needs)	female	47.0	1	0	363272	7.0000		S
3	894	2	Myles, Mr. Thomas Francis	male	62.0	0	0	240276	9.6875		Q
4	895	3	Wirz, Mr. Albert	male	27.0	0	0	315154	8.6625		S
5	896	3	Hirvonen, Mrs. Alexander (Helga E Lindqvist)	female	22.0	1	1	3101298	12.2875		S
6	897	3	Svensson, Mr. Johan Cervin	male	14.0	0	0	7538	9.2250		S
Then, we need to do all the same data cleanings.

In [30]:
data$Cabin = substr(data$Cabin, 1, 1)

meanFare = mean(data[which(!is.na(data$Fare)),]$Fare)
data$rich = 0
data[which(data$Fare > meanFare), ]$rich = 1
data = within(data, rm(Fare))

data$age = NA
data[which(data$Age < 12 & is.na(data$age)), ]$age = 1
data[which(data$Age < 19 & is.na(data$age)), ]$age = 2
data[which(data$Age < 33 & is.na(data$age)), ]$age = 3
data[which(data$Age < 45 & is.na(data$age)), ]$age = 4
data[which(data$Age > 44 & is.na(data$age)), ]$age = 5
data[which(is.na(data$Age)), ]$age = 6
data = within(data, rm(Age))
In [31]:
data = within(data, rm(PassengerId, Name, Ticket))
head(data)
A data.frame: 6 × 8
Pclass	Sex	SibSp	Parch	Cabin	Embarked	rich	age
<int>	<fct>	<int>	<int>	<chr>	<fct>	<dbl>	<dbl>
1	3	male	0	0		Q	0	4
2	3	female	1	0		S	0	5
3	2	male	0	0		Q	0	5
4	3	male	0	0		S	0	3
5	3	female	1	1		S	0	3
6	3	male	0	0		S	0	2
Now we can make our prediction.

In [34]:
eval_case(forest, data[2,])
A data.frame: 2 × 2
p	n
<dbl>	<int>
8	0.33	24
81	0.21	19
This passenger definitely died.

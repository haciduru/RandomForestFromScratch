# RandomForestFromScratch

Random-Forest-From-Scratch
I appreciate all the excellent work and effort of people who have created machine learning libraries such as SciKit Learn, PyTorch, et cetera. However, I still don't like to use other people's codes until I know what is going on in the background. Therefore, I usually implement my own tools as a learning experience and also to understand the logic and the mechanism of machine learning. Here, I will present my own random forest and decision tree models that I used to predict survival in Titanic data.

I assume everybody here has a basic understanding of random forests and decision trees. Both random forests and decision trees are analytical tools that we use to solve classification problems. A forest is an aggregate of decision trees. Each tree in the forest makes a prediction and casts a vote based on that prediction. The forest aggregates the votes and makes the final decision. Thus, the real job is done by the decision tree. But, what is a decision tree?

A decision tree is a map of the training dataset that looks just like a tree; it has branches and leaves. Leaves hang on branches. Actually, we can call leaves terminal branches. Leaves are also sets of decision rules. We make predictions about the testing dataset using those rules. Then, growing a decision tree is essentially creating sets of rules that can be used to make predictions. In this text, I will explain how you can grow a decision tree using the Titanic Disaster Dataset. I will upload a copy of Titanic Disaster data to this repository if I can.

In the Titanic case, we want to predict who among the passengers would survive and who would die. Thus, survival is our outcome (or dependent) variable. We make that prediction using the passengers' characteristics, such as their sex, age, passenger class, fare, et cetera. These characteristics are called the predictor (or independent) variables.

# Growing a decision tree

While growing a decision tree, we, iteratively, split the data into subsets based on the independent variables. The collection of subsets after the final iteration is the leaves of the tree. We use those leaves (i.e., decision rules) to make our predictions. Here is how it happens. Let's assume that we only have information on two of the independent variables: (1) passenger class and (2) sex. We can split the data into two subsets using one of these two variables--say, we do that using the passenger class. If we do that, we will have the following three new branches, because there are three passenger classes in Titanic.

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

In our example, there are only two variables (passenger class and sex). There are three unique values in passenger class and two unique values in sex. Thus, the maximum number of leaves that we can create is six. Because six is a small number, we can go until the end and create all six of the leaves. If this is the case, it does not matter whether we start splitting the data using passenger class or sex. What if we had more than two variables? For example, if we had five variables, two of them coded as yes/no, two of them with three unique values each (e.g., high, medium, low), and one of them with five unique values (e.g., age 0-12, 13-18, 19-32, 33-45, 46+). In that case, the maximum number of leaves we could create would be 2 2 3 3 4 = 144.

We usually have more than five variables, and the maximum number of leaves that we can create increases exponentially with each additional variable. If we have ten variables with two unique values each (e.g., yes/no), the maximum number of leaves we can create is 2^10 = 1,024. It is neither practical nor necessary to create that many leaves. And when we don't create all possible leaves, it matters whether we start splitting the data using variable A vs. variable B. Then, the question becomes, how do we decide where to start? Also, how to decide which variable to use to split the data at each step?

There are methods that help us make that decision. But, before talking about that, let's think about the leaves. Remember that each leave is also a decision rule. What kind of leaves do we want to have? Consider the following three leaves:

Leaf 1: Passenger class = 1, sex = female, N = 94, 97% survived

Leaf 2: Passenger class = 3, sex = female, N = 144, 50% survived

Leaf 3: Passenger class = 3, sex = male, N = 347, 13% survived

Which one of these leaves (or decision rules) is the most helpful?

If I see a female passenger in passenger class 1 in the testing set, I can easily say that she probably survived (see Leaf 1). That is, the first leaf above (the first decision rule) is very helpful. Likewise, if I see a male passenger in passenger class 3 in the testing set, I can easily say that he probably died (see Leaf 3). Thus, the third leaf is also very helpful. However, if I see a female passenger in passenger class 3, I cannot say whether she survived or died (see Leaf 2). The second leaf is not helpful because it says that the passengers in this group have a 50% chance of survival. I can make the same prediction using the flip of a fair coin. We want to create leaves like the first and the third ones. We do not want leaves like the second one.

# Entropy

The first and third leaves above have one thing in common. They are more homogenous regarding the dependent variable compared to the second leaf. That is, the passengers within these two leaves are similar to each other. The probability of two random passengers within either of these leaves to be similar to each other is high. On the other hand, the likelihood of two random passengers in the second leaf to be similar to each other is lower. A measure of this homogeneity/heterogeneity is called entropy. Higher entropy means higher heterogeneity. The formula for entropy is:

entropy = –(p log2(p)) –(q log2(q))

Int this formula, for the Titanic case, p is the proportion of passengers who survived and is the proportion of passengers who died (q = 1 - p).

According to this formula, the above three leaves have the following entropy values:

Leaf 1: -(.97 log2(.97)) -(.03 log2(.03)) = .19

Leaf 2: -(.5 log2(.5)) -(.5 log2(.5)) = .1

Leaf 3: -(.13 log2(.13)) -(.87 log2(.87)) = .56

# Information gain

When we split branches into sub-branches or leaves, we try to do that in such a way that the new branches/leaves have low entropy. The goal is to have leaves that have the lowest possible entropy. Another measure that we use to achieve this goal is "information gain." Information gain is the difference between the parent branch's entropy value and the weighted totals of the child branches' entropy values. Below is an example of how we calculate information gain.

In total, 38% of the Titanic passengers in the training set survived. Thus, the entropy value for the entire training set is:

-(.38 log2(.38)) -(.62 log2(.62)) = .96

If we split the data into three sub-branches on the passenger class variable, the sizes and entropy values of the branches will be:

Branch 1: N = 216, 63% survived, entropy = -(.63 log2(.63)) -(.37 log2(.37)) = .95

Branch 2: N = 184, 47% survived, entropy = -(.47 log2(.47)) -(.53 log2(.53)) = .98

Branch 3: N = 491, 24% survided, entropy = -(.24 log2(.24)) -(.76 log2(.76)) = .80

Here, the entire training set is the parent branch and branches 1-3 are children branches.

Information gain = .96 - ( (216/891 .95) + (184/891 .98) + (491/891 * .80)) = .09

Note that we weighted the child branches' entropy values by multiplying them with the proportion of passengers in each branch. For example, we multiplied the entropy value of Branch 1 by 216/891, because 216 of the 891 passengers were in this sub-branch (i.e., passenger class = 1).

If we split the entire training dataset into branches on the passenger class variable, the amount of information that we will gain will be .09. What would the information gain be if we decided to split the training set into branches using the sex variable? You can calculate the information gain as an exercise here.

Remember the two questions that I asked above. First, how many leaves (i.e., decision rules) should we create? Second, does it matter whether we start splitting the data into branches using this variable vs. that variable (e.g., passenger class vs. sex)? It depends. If we are going to split the data until there are no more splits, i.e., we want to create the maximum number of leaves possible, then it does not matter whether we start with this or that variable. However, it is neither practical nor necessary to create the maximum number of leaves possible. Moreover, if we create too many leaves, we will probably overfit the data to the training set, and our tree will not perform well in the testing set. Thus, we do not want to create many leaves. We want to create a few leaves--no more than enough. Therefore, we use the information gain measure to decide which variables we will use to create new branches. We split each branch into sub-branches using the variable that produces the highest information gain.

When we calculated the information gain above, we weighted the entropy values of child branches by their size. Why? The logic is very simple; there are more cases in larger branches. That makes sense, right?

Entropy and information gain are the most fundamental ideas about decision trees. If you know what they are, then you know what a decision tree is. All the rest is a coding practice. In this repository, you will find functions that to grow decision trees and random forests.

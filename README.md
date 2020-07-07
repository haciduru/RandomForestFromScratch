# RandomForestFromScratch

I appreciate all the excellent work and effort of people who have created machine learning libraries such as SciKit Learn, PyTorch, et cetera. However, I still do not like to use other people's codes until I know what is going on in the background. Therefore, I usually implement my own tools as a learning experience and really understand the logic and the mechanism of machine learning. Here, I will present my own random forest and decision tree codes that I used to predict survival in Titanic data.

I assume that everybody here knows what a random forest and what a decision tree is. If not, there are excellent tutorials out there that you can read (I am not sure if I can share their links here). Both random forests and decision trees are analytical tools that we use to solve classification problems. A forest is an aggregate of decision trees. Each tree in the forest makes a prediction and casts a vote based on that prediction. The forest aggregates the votes and makes the final decision. Thus, the real job is done by the decision tree. Here is how it happens.

Let me use the Titanic data to illustrate my points. The problem that we want to solve is to predict who among the passengers survived and who died. We want to classify passengers into two groups: survived vs. died--survival is our dependent (or outcome) variable. We will group passengers into survived vs. dead categories based on their characteristics, such as male vs. female, young vs. old, rich vs. poor, and et cetera. These characteristics are called the independent (or predictor) variables.

A decision tree, iteratively, divides the data into subgroups on the independent variables and calculates the likelihood of survival (in Titanic example) for each sub-group. Let us say we have information on only two of the passengers' characteristics in Titanic, (1) passenger class, and (2) sex. The algorithm divides the sample into subgroups on one of these characteristics first (e.g., passenger class), and then it divides each subgroup into smaller subgroups based on the other feature. Then, it calculates the proportion (i.e., probability) of passengers who survived in each subgroup.

A decision tree is a set of rules. Each subgroup is a decision rule. When we create a decision tree, we create a set of rules from a training dataset. Then, we use those rules to predict the outcome variable in a testing dataset. For example, We can create the following subgroup using Titanic data: passenger class = 1, sex = female. Ninety-seven percent of the passengers in this subgroup survived. The rule for this subgroup is: when you see a passenger from this subgroup in the testing dataset, classify that passenger as "survived." Why, because the likelihood of survival is very high. Or, we can create the following rule which is more precise: Classify 97% of the passengers in this subgroup in the testing dataset as survived and classify 3% of them died.

How many subgroups are there in a decision tree? And does it matter whether we start dividing the data into subgroups using the passenger class variable or sex variable? The answer to both of these questions is: it depends. It depends on how deep you want to go. In our example, we could have stopped when we divided the data into subgroups on the passenger class. Then we would have three subgroups because there are three passenger classes in Titanic. If we had started dividing the data into subgroups on sex, and stopped there, then we would have only two subgroups because there are only two sexes in Titanic.

A "branch" or a "leaf" is better terms and more in line with the decision tree analogy than the word "subgroup." Thus, from now on, I will use "branch(es)" or "leaf(es)" to refer to subgroups. A leaf is a terminal branch.

The number of leaves depends on how deep we want to go. If we want to go until the end, then the number of leaves is the product of the number of unique values of each variable. Let us say we have 5 predictor variables, two of them are coded as yes/no, two of them have three unique values each (e.g., high, medium, low), and one of them has five unique values (e.g., age 0-12, 13-18, 19-32, 33-45, 46+). In this case, we will have a total number of 2 * 2 * 3 * 3 * 4 = 144 leaves if we go until the end. We usually have more than five variables and the number of leaves increases exponentially with each additional vairable.

If we have ten dummy variables (e.g., yes/no) in our dataset, then the total number of leaves that we will have--if we decide to go until the end--is 1024. If we have 20 dummy variables, then we will have 1024 * 1024 = 1,048,576 leaves. It is neither practical nor necessary to have so many leaves in a decision tree. Thus, we usually have fewer leaves than the possible maximum number of leaves. In that case, it matters whether we start dividing the dataset into subgroups (or, branching out leaves from the trunk) using this variable (e.g., passenger class) versus that variable (e.g., sex). So, how do we select the variable to split the tree into branches, or a branch to sub-branches, or finally branches to leaves?

There are methods to help us select the variable to split a tree/branch into sub-branches or leaves. But, before talking about that, let us think about the leaves. Remember that each leave is also a decision rule. What kind of leaves do we want to have? Consider the following three leaves:

Leaf 1: Passenger class = 1, sex = female, survived = 97%
Leaf 2: Passenger class = 3, sex = female, survived = 50%
Leaf 3: Passenger class = 3, sex = male, survived = 13%

Which one of these leaves (or decision rules) is the most helpful?

If I see a female passenger in the first class in the testing dataset, I can easily say that she probably survived (see Leaf 1). That is, the first leaf above (the first decision rule) is very helpful. Likewise, if I see a male passenger in the third class in the testing dataset, I can easily say that he probebly died (see Leaf 3). Thus, the third leaf is also very helpful. However, if I see a female passenger in the third class, I cannot say whether she survived or died (see Leaf 2). The second leaf is not useful because it says that the passengers in this group have a 50% chance of survival. I can make the same prediction using the flip of a fair coin. We want to create leaves like the first and the third ones. We do not want leaves like the second one.

The first and third leaves above have one thing in common. They are more homogenous regarding the dependent variable compared to the second leaf. That is, the passengers within these two leaves are similar to each other. The probability of two random passengers within the leaf to be similar to each other is high. On the other hand, the likelihood of two random passengers in the second leaf to be similar to each other is lower.   A statistical measure of this homogeneity/heterogeneity is called entropy. Higher entropy means higher heterogeneity. The formula for entropy is: 

entropy = –(p * log(p)) –(q * log(q))

According to this formula, the above three leaves have the following values of entropy:

Leaf 1: -(.97 * log(.97)) -(.03 * log(.03)) = .13
Leaf 2: -(.5 * log(.5)) -(.5 * log(.5)) = .69
Leaf 3: -(.13 * log(.13)) -(.87 * log(.87)) = .39

When we split branches into sub-branches or leaves, we try to do that in such a way that the new branches/leaves have low entropy. The goal is to have leaves that have the lowest possible entropy. Another measure that we use to achieve this goal is "information gain." Information gain is the difference between the parent branch's entropy and the weighed totals of the child branches' entropies. Below is an example of how we calculate information gain. 




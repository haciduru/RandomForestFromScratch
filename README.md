# RandomForestFromScratch

I appreciate all the excellent work and effort of people who have created machine learning libraries such as SciKit Learn, PyTorch, et cetera.
However, I still do not like to use other people's codes until I know what is going on in the background. Therefore, I usually implement my
own tools as a learning experience and really understand the logic and the mechanism of machine learning. Here, I will present my own random
forest and decision tree codes that I used to predict survival in Titanic data.

I assume that everybody here knows what a random forest and what a decision tree is. If not, there are excellent tutorials out there that you 
can read (I am not sure if I can share their links here). Both random forests and decision trees are analytical tools that we use to solve 
classification problems. A forest is an aggregate of decision trees. Each tree in the forest makes a prediction and casts a vote based on that
prediction. The forest aggregates the votes and makes the final decision. Thus, the real job is done by the decision tree. Here is how it happens.

Let me use the Titanic data to illustrate my points. The problem that we want to solve is to predict who among the passengers survived and who died.
We want to classify passengers into two groups: survived vs. died--survival is our dependent (or outcome) variable. We will group passengers into 
survived vs. dead categories based on their characteristics, such as male vs. female, young vs. old, rich vs. poor, and et cetera. These
characteristics are called the independent (or predictor) variables.

A decision tree, iteratively, divides the data into subgroups on the independent variables and calculates the likelihood of survival (in Titanic 
example) for each sub-group. Let us say we have information on only two of the passengers' characteristics in Titanic, (1) passenger class, and (2) sex.
The algorithm divides the sample into subgroups on one of these characteristics first (e.g., passenger class), and then it divides each subgroup
into smaller subgroups based on the other feature. Then, it calculates the proportion (i.e., probability) of passengers who survived in each subgroup.


A decision tree is a set of rules. Each subgroup is a decision rule. When we create a decision tree, we create a set of rules from a training dataset.
Then, we use those rules to predict the outcome variable in a testing dataset. For example, We can create the following subgroup using Titanic data:
passenger class = 1, sex = female. Ninety-seven percent of the passengers in this subgroup survived. The rule for this subgroup is: when you see a
passenger from this subgroup in the testing dataset, classify that passenger as "survived." Why, because the likelihood of survival is very high. Or,
we can create the following rule which is more precise: Classify 97% of the passengers in this subgroup in the testing dataset as survived and classify
3% of them died.

How many subgroups are there in a decision tree? And does it matter whether we start dividing the data into subgroups using the passenger class variable
or sex variable? The answer to both of these questions is: it depends. It depends on how deep you want to go. In our example, we could have stopped when
we divided the data into subgroups on the passenger class. Then we would have three subgroups because there are three passenger classes in Titanic. If we had
started dividing the data into subgroups on sex, and stopped there, then we would have only two subgroups because there are only two sexes in Titanic.

A "leaf" is a better term and more in line with the decision tree analogy than the word "subgroup." Thus, I will use leaf and leaves to refer 
to subgroups from now on.

The number of leaves depends on how deep we want to go. If we want to go until the end, then the number of leaves is the product of the number of unique 
values of each variable. Let us say we have 5 predictor variables, two of them are coded as yes/no, two of them have three unique values each (e.g., high,
medium, low), and one of them has five unique values (e.g., age 0-12, 13-18, 19-32, 33-45, 46+). In this case, we will have a total number of 2 * 2 * 3 * 
3 * 4 = 144 leaves if we go until the end. We usually have more than five variables and the number of leaves increases exponentially with each additional
vairable.

If we have ten dummy variables (e.g., yes/no) in our dataset, then the total number of leaves that we will have--if we decide to go until the end--is 1024.
If we have 20 dummy variables, then we will have 1024 * 1204 leaves. It is neither practical nor necessary to have so many leaves in a decision tree. Thus, 
we usually have fewer leaves than the possible maximum number of leaves. In that case, it matters whether we start dividing the dataset into subgroups (or, branching out leaves from the trunk) using this variable (e.g., passenger class) versus that variable (e.g., sex). So, how do we select the variable to split 
the tree into branches, or a branch to sub-branches, or finally branches to leaves?






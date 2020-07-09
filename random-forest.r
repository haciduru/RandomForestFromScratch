## -----------------------------------------------------------------------------
# This function calculates information entropy.
# input: 
#	var (vector): of dummy values.
# returns:
#	entropy (float): entropy
entropy = function(var) {

	p = mean(var)
	if (p == 1 | p == 0) {
		return(0)
	} else {
	q = 1 - p
		return(
			-p * log(p, 2) 		# -p(1) * log2(p(1)) 
			-q * log(q, 2)		# -p(2) * log2(p(2))
			)
	}

}

## -----------------------------------------------------------------------------
# unique gets the levels of a categorical variable
# unique(vector)

## -----------------------------------------------------------------------------
# This function initializes a decision tree.
# input:
#	dta (matrix/data frame): data
# returns:
#	tree (list): a list of tree branches.
plant = function(data) {
	
	tree = list(vars = c(), data = data)
	tree = create_branches(tree)
	return(tree)
	
}

## -----------------------------------------------------------------------------
# This function finds the variable to with smallest entropy.
# input:
# 	data (matrix/data frame): data
# returns:
#	ii (int): index of the variable with smallest entropy.
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
					if (new_entropy < min_entropy) { 
						min_entropy = new_entropy
						ii = i
					}
				}
			}
		}
	}
	return(ii)	
	
}

## -----------------------------------------------------------------------------
# This function creates new branches from a parent branch.
# input:
#	parent (list): parent branch
# returns:
#	children (list): a list of new children (i.e., branches)
create_branches = function(parent) {
	
	data = parent[[2]]
	ii = find_branch_to_split(data)
	if (ii > 0) {
		levels = unique(data[,ii])
		children = vector("list", length(levels))
		if (length(levels) > 0) {
			for (i in 1:length(levels)) {
				child = data.frame(data[which(data[,ii] == levels[i]), -ii])
				if (ncol(child) == 1) names(child) = names(data)[1]
				if (length(parent[[1]]) > 0) {
					value = rbind(parent[[1]], c(names(data)[ii], levels[i]))
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

## -----------------------------------------------------------------------------
# This function grows a decision tree.
# input:
#	tree (list): a decision tree
#	min_split (int): minimum number of cases to split a branch
# returns:
#	tree (list): a decision tree
grow_tree = function(tree, min_split, min_leaf) {
	
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

## -----------------------------------------------------------------------------
# This function shows the parameters of a decision tree.
# input:
#	tree (list): a decision tree
# returns:
#	ret (data frame): a data frame that has params of the decision tree
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

## -----------------------------------------------------------------------------
# This function grows a forest from data.
# input:
#	data (matrix/data frame): data
#	size (int): number of cases in each tree
#	n (int): number of trees to grow
#	min_split (int): minimum number of cases to split a branch
# returns:
#	forest (list): a list of decision trees
grow_forest = function(data, tree_size, n_tree, min_split, min_leaf) {
	
	forest = vector("list", n_tree)
	for (i in 1:n_tree) {
		tree = plant(sample(data, tree_size))
		tree = grow_tree(tree, min_split, min_leaf)
		forest[[i]] = show_tree(tree)
	}
	return(forest)
	
}

## -----------------------------------------------------------------------------
# This function evaluates the likelihood of a case given a decision tree.
# input:
#	forest (list): a list of decision trees
#	case (vector): a single case (i.e., a vector of values) to evaluate.
# returns:
#	vals (matrix): an (n by 2) matrix. 
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

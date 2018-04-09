import tensorflow as tf
import numpy as np
def placeholder():
    """
    最简单的一个tensorflow实例
    """
    input_x = tf.placeholder(tf.float32, [None, 2])
    x = np.random.rand(12, 2)
    with tf.Session() as sess:
        print(sess.run(input_x, feed_dict={input_x : x}))
placeholder()
from sklearn.tree import DecisionTreeClassifier as DTC
import numpy as np
X = [[0],[1],[2]] # 3 simple training examples
Y = [ 1,  2,  1 ] # class labels

dtc = DTC(max_depth=1)
dtc.fit(X, Y, sample_weight=[1,2,3])
print(dtc.predict_proba([0.2]))
print(dtc.score(X, Y, sample_weight=[1,2,3]))
n_nodes = dtc.tree_.node_count
children_left = dtc.tree_.children_left
children_right = dtc.tree_.children_right
feature = dtc.tree_.feature
threshold = dtc.tree_.threshold
# print(help(dtc.tree_))
print(dtc.tree_.value)


# The tree structure can be traversed to compute various properties such
# as the depth of each node and whether or not it is a leaf.
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True

print("The binary tree structure has %s nodes and has "
      "the following tree structure:"
      % n_nodes)
for i in range(n_nodes):
    if is_leaves[i]:
        print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
    else:
        print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
              "node %s."
              % (node_depth[i] * "\t",
                 i,
                 children_left[i],
                 feature[i],
                 threshold[i],
                 children_right[i],
                 ))
print()

# [0.44444444, 0, 0.5]
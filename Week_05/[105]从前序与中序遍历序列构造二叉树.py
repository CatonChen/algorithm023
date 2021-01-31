# 根据一棵树的前序遍历与中序遍历构造二叉树。 
# 
#  注意: 
# 你可以假设树中没有重复的元素。 
# 
#  例如，给出 
# 
#  前序遍历 preorder = [3,9,20,15,7]
# 中序遍历 inorder = [9,3,15,20,7] 
# 
#  返回如下的二叉树： 
# 
#      3
#    / \
#   9  20
#     /  \
#    15   7 
#  Related Topics 树 深度优先搜索 数组 
#  👍 847 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        def myBuildTree(pre_left, pre_right, in_left, in_right):
            # 终止条件
            if pre_left > pre_right:
                return None
            # 前序根下标
            pre_root = preorder[pre_left]
            # 中序根下标
            in_root = index[pre_root]

            root = TreeNode(pre_root)

            len_left = in_root - in_left

            root.left = myBuildTree(pre_left + 1, pre_left + len_left, in_left, in_root - 1)
            root.right = myBuildTree(pre_left + len_left + 1, pre_right, in_root + 1, in_right)
            return root

        n = len(preorder)
        index = {ele: i for i, ele in enumerate(inorder)}
        return myBuildTree(0, n - 1, 0, n - 1)

# leetcode submit region end(Prohibit modification and deletion)

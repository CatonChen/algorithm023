# 给你一个二叉树，请你返回其按 层序遍历 得到的节点值。 （即逐层地，从左到右访问所有节点）。 
# 
#  
# 
#  示例： 
# 二叉树：[3,9,20,null,null,15,7], 
# 
#  
#     3
#    / \
#   9  20
#     /  \
#    15   7
#  
# 
#  返回其层序遍历结果： 
# 
#  
# [
#   [3],
#   [9,20],
#   [15,7]
# ]
#  
#  Related Topics 树 广度优先搜索 
#  👍 801 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    # 递归
    def dfs(self, root, level, res):
        # 终止
        if not root:
            return
            # 遇到新层级时，在结果创建一个数组
        if len(res) == level:
            res.append([])
        # 节点加入level的数组中
        res[level].append(root.val)
        # 递归调用左右子节点
        self.dfs(root.left, level + 1, res)
        self.dfs(root.right, level + 1, res)

    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        res = []
        self.dfs(root, 0, res)
        return res
# leetcode submit region end(Prohibit modification and deletion)

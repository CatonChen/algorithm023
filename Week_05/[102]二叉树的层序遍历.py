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
#  👍 760 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    # 递归
    def dfs(self, root, level, res):
        # 终止条件
        if root is None:
            return
        # 当到新的一层时，要创建一个新数组保存结果
        if len(res) == level:
            res.append([])
        res[level].append(root.val)  # 把当层结果加入当层的数组里
        # 递归工作
        self.dfs(root.left, level + 1, res)
        self.dfs(root.right, level + 1, res)

    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        res = []
        self.dfs(root, 0, res)
        return res

# leetcode submit region end(Prohibit modification and deletion)

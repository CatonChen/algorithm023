#
# @lc app=leetcode.cn id=98 lang=python3
#
# [98] 验证二叉搜索树
#

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        stack =[]
        inorder = float('-inf')

        while stack or root:
            while root:
                stack.append(root)
                root=root.left
            node = stack.pop()
            if node.val<=inorder:
                return False
            inorder=node.val
            root=node.right
        
        return True

# @lc code=end


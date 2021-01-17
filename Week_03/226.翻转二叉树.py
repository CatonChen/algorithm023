#
# @lc app=leetcode.cn id=226 lang=python3
#
# [226] 翻转二叉树
#

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        from collections import deque
        queue = deque()
        if root:
            queue.append(root)
        else:
            return root
        while queue:
            cur_node = queue.popleft()
            cur_node.left ,cur_node.right = cur_node.right, cur_node.left
            if cur_node.left:
                queue.append(cur_node.left)
            if cur_node.right:
                queue.append(cur_node.right)
        return root

# @lc code=end


#
# @lc app=leetcode.cn id=104 lang=python3
#
# [104] 二叉树的最大深度
#

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        from collections import deque
        if root is None:
            return 0
        queue = deque([root])
        level = 0 
        while queue:
            n = len(queue)
            print(n)
            for i in range(n):
                node = queue.popleft()
                print(node)
                if node:	
                    queue.extend([node.left,node.right])
            level+=1
        return level -1



# @lc code=end


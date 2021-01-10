#
# @lc app=leetcode.cn id=429 lang=python3
#
# [429] N 叉树的层序遍历
#

# @lc code=start
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        if root is None:
            return []
        res = []
        queue = collections.deque()
        queue.append(root)
        while queue:
            level=[]
            for i in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                queue.extend(node.children)
            res.append(level)
        return res
        
# @lc code=end


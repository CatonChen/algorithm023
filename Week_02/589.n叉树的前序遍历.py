#
# @lc app=leetcode.cn id=589 lang=python3
#
# [589] N叉树的前序遍历
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
    def preorder(self, root: 'Node') -> List[int]:
        if not root: return []
        res=[]
        st = [root]
        while st:
            node = st.pop()
            res.append(node.val)
            st.extend(node.children[::-1])
        return res


# @lc code=end


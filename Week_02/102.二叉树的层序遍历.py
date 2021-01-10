#
# @lc app=leetcode.cn id=102 lang=python3
#
# [102] 二叉树的层序遍历
#

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def helper(self,root,level,res):
        if root is None: return
        if len(res)==level:
            res.append([])
        res[level].append(root.val)
        if root.left:
            self.helper(root.left,level+1,res)
        if root.right:
            self.helper(root.right,level+1,res)
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        res=[]
        self.helper(root,0,res)
        return res

# @lc code=end


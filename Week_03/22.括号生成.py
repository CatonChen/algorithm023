#
# @lc app=leetcode.cn id=22 lang=python3
#
# [22] 括号生成
#

# @lc code=start
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res=[]
        cur_str=''
        def dfs(cur_str,left,right):
            if left==0 and right==0:    #left,right==0时，添加结果
                return res.append(cur_str)
            if right<left:  #right<left，不符合返回空
                return
            if left>0:
                dfs(cur_str+'(',left-1,right)   #添加（，左-1
            if right>0:
                dfs(cur_str+')',left,right-1)   #添加），右-1
        
        dfs(cur_str,n,n)
        return res

# @lc code=end


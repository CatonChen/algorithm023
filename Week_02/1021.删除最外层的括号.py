#
# @lc app=leetcode.cn id=1021 lang=python3
#
# [1021] 删除最外层的括号
#

# @lc code=start
class Solution:
    def removeOuterParentheses(self, S: str) -> str:
        #双指针
        res = []
        left , counter = 0,0
        for i in range(len(S)):
            if S[i]=='(':
                counter+=1
            elif S[i]==')':
                counter-=1
            if counter==0:
                res.append([left,i])
                left=i+1
        # print(res)
        return "".join(S[m+1:n] for m,n in res)        
# @lc code=end


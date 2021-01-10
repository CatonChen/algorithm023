#
# @lc app=leetcode.cn id=830 lang=python3
#
# [830] 较大分组的位置
#

# @lc code=start
class Solution:
    def largeGroupPositions(self, s: str) -> List[List[int]]:
        res=[]
        left = 0
        s+='1'  #添加非字符值，避免越界
        for index ,v in enumerate(s):
            # print(index)
            # print(v)
            if v!=s[left]:
                if index - left >=3 :
                    res.append([left,index-1])
                left = index
        return res

# @lc code=end


#
# @lc app=leetcode.cn id=66 lang=python3
#
# [66] 加一
#

# @lc code=start
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        r = list(map(int,str(int("".join(map(str, digits)))+1)))
        # print(r)
        return [0]*(len(digits)-len(r)) + r

# @lc code=end


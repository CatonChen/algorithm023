#
# @lc app=leetcode.cn id=258 lang=python3
#
# [258] 各位相加
#

# @lc code=start
class Solution:
    def addDigits(self, num: int) -> int:
        while num >=10:
            digits = list(map(int,str(num)))
            num=0
            for i in range(len(digits)):
                num+=digits[i]
        return num
# @lc code=end


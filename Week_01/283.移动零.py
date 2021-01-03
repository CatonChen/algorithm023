#
# @lc app=leetcode.cn id=283 lang=python3
#
# [283] 移动零
#

# @lc code=start
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n=len(nums)
        y=0
        for i in range(n):
            if nums[i]!=0:
                nums[y],nums[i]=nums[i],nums[y]
                y+=1
                
# @lc code=end


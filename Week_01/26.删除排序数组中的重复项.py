#
# @lc app=leetcode.cn id=26 lang=python3
#
# [26] 删除排序数组中的重复项
#

# @lc code=start
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        l = 0 
        if len(nums)==0:
            return 0
        for i in range(1,len(nums)): #指针位移
            if nums[l]!=nums[i]:    #两者不等
                l+=1                #移动第一个指针
                nums[l]=nums[i]     #赋值
        return l+1                  #返回长度


# @lc code=end


#
# @lc app=leetcode.cn id=15 lang=python3
#
# [15] 三数之和
#

# @lc code=start
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        nums.sort()
        res=[]

        for k in range(n-2):  
            if nums[k]>0:
                return res  
            if k>0 and nums[k]==nums[k-1]:  #跳过重复元素
                continue
            i , j = k+1, n-1
            while (i<j):
                if nums[k]+nums[i]+nums[j]==0:  #符合条件，添加结果
                    res.append([nums[k],nums[i],nums[j]])
                    i+=1
                    j-=1
                    while i<j and nums[i]==nums[i-1]:   #跳过重复元素
                        i+=1
                    while i<j and nums[j]==nums[j+1]:   #跳过重复元素
                        j-=1
                elif nums[k]+nums[i]+nums[j]<0: #i元素太小，移动i
                    i+=1
                    while i<j and nums[i]==nums[i-1]:
                        i+=1
                else: 
                    j-=1    #j元素太大，移动j
                    while i<j and nums[j]==nums[j+1]:
                        j-=1
        
        return res

# @lc code=end


#
# @lc app=leetcode.cn id=42 lang=python3
#
# [42] 接雨水
#

# @lc code=start
class Solution:
    def trap(self, height: List[int]) -> int:
        res = 0
        l ,r =0,len(height)-1   #双指针
        temp_h=0    
        while  l <=r :
            min_l_h=min(height[l],height[r])    #取左右指针中的较小者
            if min_l_h>=temp_h: #较小者>=上次较小者
                res += (min_l_h - temp_h) * (r - l +1)  #计算高度差*当前左右指针的范围
                temp_h=min_l_h  #保留较小者
            while l<=r and height[l]<=min_l_h:  
                l+=1    #当做指针高度小于等于较小者，做指针移动
            while l<=r and height[r]<=min_l_h:
                r-=1    #当右指针高度小于等于较小者，有指针移动
        res = res - sum(height) #总面积-所有柱子面积
        return res
# @lc code=end


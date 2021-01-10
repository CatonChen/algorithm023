#
# @lc app=leetcode.cn id=350 lang=python3
#
# [350] 两个数组的交集 II
#

# @lc code=start
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        #双指针
        # nums1.sort()
        # nums2.sort()
        # res=[]
        # i=j=0
        # while i<len(nums1) and j<len(nums2):
        #     if nums1[i]==nums2[j]:
        #         res.append(nums1[i])
        #         i+=1
        #         j+=1
        #     elif nums1[i]<nums2[j]:
        #         i+=1
        #     else:
        #         j+=1
        # return res

        #利用counter
        from collections import Counter
        return [*(Counter(nums1) & Counter(nums2)).elements()]

# @lc code=end


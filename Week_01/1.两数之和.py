#
# @lc app=leetcode.cn id=1 lang=python3
#
# [1] 两数之和
#

# @lc code=start
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashmap=dict()
        for i , v in enumerate(nums):
            if hashmap.get(target - v) is not None:
                return [hashmap.get(target - v),i]
            hashmap[v]=i
# @lc code=end


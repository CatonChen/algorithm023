#
# @lc app=leetcode.cn id=1 lang=python3
#
# [1] 两数之和
#

# @lc code=start
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashmap=dict()
        res=[]
        for i, val in enumerate(nums):
            if hashmap.get(target - val) is not None:
                res.append(hashmap.get(target - val))
                res.append(i)
            hashmap[val]=i
        return res

# @lc code=end


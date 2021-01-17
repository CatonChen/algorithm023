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
        for i ,v in enumerate(nums):
            if hashmap.get((target - v)) is not None:
                res.append(hashmap.get(target - v))
                res.append(i)
            hashmap[v]=i
        return res

# @lc code=end


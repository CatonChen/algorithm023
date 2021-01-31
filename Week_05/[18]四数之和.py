# 给定一个包含 n 个整数的数组 nums 和一个目标值 target，判断 nums 中是否存在四个元素 a，b，c 和 d ，使得 a + b + c +
#  d 的值与 target 相等？找出所有满足条件且不重复的四元组。 
# 
#  注意： 
# 
#  答案中不可以包含重复的四元组。 
# 
#  示例： 
# 
#  给定数组 nums = [1, 0, -1, 0, -2, 2]，和 target = 0。
# 
# 满足要求的四元组集合为：
# [
#   [-1,  0, 0, 1],
#   [-2, -1, 1, 2],
#   [-2,  0, 0, 2]
# ]
#  
#  Related Topics 数组 哈希表 双指针 
#  👍 728 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def nSum(self, nums, n, target):
        if len(nums) < n:
            return []

        res = []
        # 分解到n为2时，求2数之和等于目标
        if n == 2:
            i, j = 0, len(nums) - 1
            while i < j:
                s = nums[i] + nums[j]
                if s == target:
                    res.append([nums[i], nums[j]])
                    while i < j and nums[i] == nums[i + 1]:
                        i += 1
                    while i < j and nums[j] == nums[j - 1]:
                        j -= 1
                    i += 1
                    j -= 1
                elif s < target:
                    i += 1
                else:
                    j -= 1
            return res
        else:  # 否则继续分解
            for i in range(len(nums)):
                if i > 0 and nums[i] == nums[i - 1]:
                    continue
                subres = self.nSum(nums[i + 1:], n - 1, target - nums[i])
                for j in range(len(subres)):
                    res.append([nums[i]] + subres[j])
            return res

    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        if len(nums) < 4:
            return []
        nums.sort()
        return self.nSum(nums, 4, target)
# leetcode submit region end(Prohibit modification and deletion)

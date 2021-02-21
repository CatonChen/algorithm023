# 给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重
# 复的三元组。 
# 
#  注意：答案中不可以包含重复的三元组。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：nums = [-1,0,1,2,-1,-4]
# 输出：[[-1,-1,2],[-1,0,1]]
#  
# 
#  示例 2： 
# 
#  
# 输入：nums = []
# 输出：[]
#  
# 
#  示例 3： 
# 
#  
# 输入：nums = [0]
# 输出：[]
#  
# 
#  
# 
#  提示： 
# 
#  
#  0 <= nums.length <= 3000 
#  -105 <= nums[i] <= 105 
#  
#  Related Topics 数组 双指针 
#  👍 2975 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    # 递归nSum
    def nSum(self, nums, n, target):
        res = []
        # 递归终止条件
        if len(nums) < n:
            return res
        # n=2
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
        # n<>2
        else:
            for k in range(len(nums)):
                if k > 0 and nums[k] == nums[k - 1]:
                    continue
                else:
                    subres = self.nSum(nums[k + 1:], n - 1, target - nums[k])
                    for p in range(len(subres)):
                        res.append([nums[k]] + subres[p])
            return res

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        return self.nSum(nums, 3, 0)

# leetcode submit region end(Prohibit modification and deletion)

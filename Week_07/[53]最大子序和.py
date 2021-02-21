# 给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
# 输出：6
# 解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
#  
# 
#  示例 2： 
# 
#  
# 输入：nums = [1]
# 输出：1
#  
# 
#  示例 3： 
# 
#  
# 输入：nums = [0]
# 输出：0
#  
# 
#  示例 4： 
# 
#  
# 输入：nums = [-1]
# 输出：-1
#  
# 
#  示例 5： 
# 
#  
# 输入：nums = [-100000]
# 输出：-100000
#  
# 
#  
# 
#  提示： 
# 
#  
#  1 <= nums.length <= 3 * 104 
#  -105 <= nums[i] <= 105 
#  
# 
#  
# 
#  进阶：如果你已经实现复杂度为 O(n) 的解法，尝试使用更为精妙的 分治法 求解。 
#  Related Topics 数组 分治算法 动态规划 
#  👍 2891 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # 分治
        n = len(nums)
        # 递归终止条件
        if n == 1:
            return nums[0]
        else:
            # 递归左半部分
            max_left = self.maxSubArray(nums[0:n // 2])
            # 递归右半部分
            max_right = self.maxSubArray(nums[n // 2:n])

        # 计数中间部分
        # 中间部分的左半部分
        max_l = nums[n // 2 - 1]
        tmp = 0
        for i in range(n // 2 - 1, -1, -1):
            tmp += nums[i]
            max_l = max(max_l, tmp)
        # 中间部分的右半部分
        max_r = nums[n // 2]
        tmp = 0
        for i in range(n // 2, n):
            tmp += nums[i]
            max_r = max(max_r, tmp)

        # 返回最终结果
        return max(max_left, max_right, max_l + max_r)
# leetcode submit region end(Prohibit modification and deletion)

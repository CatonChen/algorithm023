# 给你一个整型数组 nums ，在数组中找出由三个数组成的最大乘积，并输出这个乘积。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：nums = [1,2,3]
# 输出：6
#  
# 
#  示例 2： 
# 
#  
# 输入：nums = [1,2,3,4]
# 输出：24
#  
# 
#  示例 3： 
# 
#  
# 输入：nums = [-1,-2,-3]
# 输出：-6
#  
# 
#  
# 
#  提示： 
# 
#  
#  3 <= nums.length <= 104 
#  -1000 <= nums[i] <= 1000 
#  
#  Related Topics 数组 数学 
#  👍 274 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def maximumProduct(self, nums: List[int]) -> int:
        min1, min2 = float('inf'), float('inf')
        max1, max2, max3 = float('-inf'), float('-inf'), float('-inf')
        for num in nums:
            min1, min2, _ = sorted([min1, min2, num])
            _, max1, max2, max3 = sorted([max1, max2, max3, num])
        return max(min1 * min2 * max3, max1 * max2 * max3)

# leetcode submit region end(Prohibit modification and deletion)

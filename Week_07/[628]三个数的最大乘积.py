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
        # 排序
        nums.sort()
        a = nums[-3] * nums[-2] * nums[-1]
        b = nums[0] * nums[1] * nums[-1]
        return max(a, b)

# leetcode submit region end(Prohibit modification and deletion)

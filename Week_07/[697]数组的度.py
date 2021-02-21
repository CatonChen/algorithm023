# 给定一个非空且只包含非负数的整数数组 nums，数组的度的定义是指数组里任一元素出现频数的最大值。 
# 
#  你的任务是在 nums 中找到与 nums 拥有相同大小的度的最短连续子数组，返回其长度。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：[1, 2, 2, 3, 1]
# 输出：2
# 解释：
# 输入数组的度是2，因为元素1和2的出现频数最大，均为2.
# 连续子数组里面拥有相同度的有如下所示:
# [1, 2, 2, 3, 1], [1, 2, 2, 3], [2, 2, 3, 1], [1, 2, 2], [2, 2, 3], [2, 2]
# 最短连续子数组[2, 2]的长度为2，所以返回2.
#  
# 
#  示例 2： 
# 
#  
# 输入：[1,2,2,3,1,4,2]
# 输出：6
#  
# 
#  
# 
#  提示： 
# 
#  
#  nums.length 在1到 50,000 区间范围内。 
#  nums[i] 是一个在 0 到 49,999 范围内的整数。 
#  
#  Related Topics 数组 
#  👍 297 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def findShortestSubArray(self, nums: List[int]) -> int:
        # 初始化哈希字典
        dict = {}
        # 记录相同元素出现的次数，第一次位置和最后一次位置
        for i, v in enumerate(nums):
            if v in dict:
                dict[v][0] += 1
                dict[v][2] = i
            else:
                dict[v] = [1, i, i]
        # print(dict)
        # print(dict.values())
        # print(dict.items())
        # 设定度，最短长度
        maxcount = minlen = 0
        for count, left, right in dict.values():
            if maxcount < count:  # 度不是最大时，不断更新度和最短长度
                maxcount = count
                minlen = right - left + 1
            elif maxcount == count:  # 度一样时找最短长度
                minlen = min(minlen, right - left + 1)
        return minlen
# leetcode submit region end(Prohibit modification and deletion)

# 给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位
# 。 
# 
#  返回滑动窗口中的最大值。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
# 输出：[3,3,5,5,6,7]
# 解释：
# 滑动窗口的位置                最大值
# ---------------               -----
# [1  3  -1] -3  5  3  6  7       3
#  1 [3  -1  -3] 5  3  6  7       3
#  1  3 [-1  -3  5] 3  6  7       5
#  1  3  -1 [-3  5  3] 6  7       5
#  1  3  -1  -3 [5  3  6] 7       6
#  1  3  -1  -3  5 [3  6  7]      7
#  
# 
#  示例 2： 
# 
#  
# 输入：nums = [1], k = 1
# 输出：[1]
#  
# 
#  示例 3： 
# 
#  
# 输入：nums = [1,-1], k = 1
# 输出：[1,-1]
#  
# 
#  示例 4： 
# 
#  
# 输入：nums = [9,11], k = 2
# 输出：[11]
#  
# 
#  示例 5： 
# 
#  
# 输入：nums = [4,-2], k = 2
# 输出：[4] 
# 
#  
# 
#  提示： 
# 
#  
#  1 <= nums.length <= 105 
#  -104 <= nums[i] <= 104 
#  1 <= k <= nums.length 
#  
#  Related Topics 堆 Sliding Window 
#  👍 887 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        # 初始两个队列
        win = []
        res = []
        # 遍历一次数组
        for i in range(len(nums)):
            # 当形成窗口后，且win[0]移出了窗口范围，win[0]删除
            if i >= k and win[0] <= i - k:
                win.pop(0)
            # 当win不为空，清空win中所有<i代表的元素的下标
            while win and nums[win[-1]] < nums[i]:
                win.pop()
            # 不满足以上，将i加入win
            win.append(i)
            # 当有窗口后，每次移动时将窗口的最大值加入res
            if i >= k - 1:
                res.append(nums[win[0]])
        # 返回结果
        return res
# leetcode submit region end(Prohibit modification and deletion)

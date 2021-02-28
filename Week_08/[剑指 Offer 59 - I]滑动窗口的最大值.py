# 给定一个数组 nums 和滑动窗口的大小 k，请找出所有滑动窗口里的最大值。 
# 
#  示例: 
# 
#  输入: nums = [1,3,-1,-3,5,3,6,7], 和 k = 3
# 输出: [3,3,5,5,6,7] 
# 解释: 
# 
#   滑动窗口的位置                最大值
# ---------------               -----
# [1  3  -1] -3  5  3  6  7       3
#  1 [3  -1  -3] 5  3  6  7       3
#  1  3 [-1  -3  5] 3  6  7       5
#  1  3  -1 [-3  5  3] 6  7       5
#  1  3  -1  -3 [5  3  6] 7       6
#  1  3  -1  -3  5 [3  6  7]      7 
# 
#  
# 
#  提示： 
# 
#  你可以假设 k 总是有效的，在输入数组不为空的情况下，1 ≤ k ≤ 输入数组的大小。 
# 
#  注意：本题与主站 239 题相同：https://leetcode-cn.com/problems/sliding-window-maximum/ 
#  Related Topics 队列 Sliding Window 
#  👍 190 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        res = []
        win = []  # 保存nums下标i
        # 遍历数组nums
        for i in range(len(nums)):
            # 当win的首端元素不在滑动窗口k的范围内后，要删除
            if i >= k and win[0] <= i - k:
                win.pop(0)
            # 当win不为空，且win末端代表的元素小于i代表的元素时，需要删除
            while win and nums[win[-1]] < nums[i]:
                win.pop()
            # 不满足上述的i都加入win中
            win.append(i)
            # 当窗口形成后，开始将每次移动后的窗口最大值加入结果
            if i >= k - 1:
                res.append(nums[win[0]])
        # 返回结果
        return res
# leetcode submit region end(Prohibit modification and deletion)

# 给定一个非负整数数组，你最初位于数组的第一个位置。 
# 
#  数组中的每个元素代表你在该位置可以跳跃的最大长度。 
# 
#  你的目标是使用最少的跳跃次数到达数组的最后一个位置。 
# 
#  示例: 
# 
#  输入: [2,3,1,1,4]
# 输出: 2
# 解释: 跳到最后一个位置的最小跳跃数是 2。
#      从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。
#  
# 
#  说明: 
# 
#  假设你总是可以到达数组的最后一个位置。 
#  Related Topics 贪心算法 数组 
#  👍 809 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        #特例
        if n == 1:
            return 0
        cur = nex = step = 0  # cur当前跳的最大范围，nex下一跳最远距离
        for i, v in enumerate(nums):
            # print(i,v,cur)
            if i > cur:
                cur = nex
                # print(i,cur)
                step += 1
            nex = max(nex, i + v)
            # print(nex)
            if nex >= n - 1:
                return step + 1
# leetcode submit region end(Prohibit modification and deletion)

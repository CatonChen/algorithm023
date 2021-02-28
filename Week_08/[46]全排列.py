# 给定一个 没有重复 数字的序列，返回其所有可能的全排列。 
# 
#  示例: 
# 
#  输入: [1,2,3]
# 输出:
# [
#   [1,2,3],
#   [1,3,2],
#   [2,1,3],
#   [2,3,1],
#   [3,1,2],
#   [3,2,1]
# ] 
#  Related Topics 回溯算法 
#  👍 1089 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        # 终止条件
        if not nums:
            return [[]]
        # 遍历nums
        for i in range(len(nums)):
            subres = self.permute(nums[:i] + nums[i + 1:])
            for p in subres:
                res.append([nums[i]] + p)
        return res
# leetcode submit region end(Prohibit modification and deletion)

# 给定两个整数 n 和 k，返回 1 ... n 中所有可能的 k 个数的组合。 
# 
#  示例: 
# 
#  输入: n = 4, k = 2
# 输出:
# [
#   [2,4],
#   [3,4],
#   [2,3],
#   [1,2],
#   [1,3],
#   [1,4],
# ] 
#  Related Topics 回溯算法 
#  👍 471 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        # 定义dfs
        def backtrace(cur, idx):
            # 终止条件
            if len(cur) == k:
                res.append(cur[:])
                return
                # 逻辑处理与下钻
            for i in range(idx, n + 1):
                cur.append(i)
                backtrace(cur, i + 1)  # 下标+1
                cur.pop()  # 回溯

        res = []
        if k == 0:
            return res
        backtrace([], 1)
        return res
# leetcode submit region end(Prohibit modification and deletion)

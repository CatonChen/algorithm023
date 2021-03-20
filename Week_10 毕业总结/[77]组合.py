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
        # 回溯
        def dfs(tmp, idx):
            # 递归终止
            if len(tmp) == k:
                res.append(tmp[:])
                return
                # 处理当前值
            for i in range(idx, n + 1):
                tmp.append(i)
                # print(['a']+tmp)
                dfs(tmp, i + 1)
                tmp.pop()  # 回溯
                # print(['b']+tmp)

        # 主逻辑
        res = []
        if k == 0:
            return res
        dfs([], 1)
        return res

# leetcode submit region end(Prohibit modification and deletion)

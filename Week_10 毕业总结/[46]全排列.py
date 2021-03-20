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
        n = len(nums)
        # uesd数组，记录节点状态
        used = [False] * n
        res = []

        # 回溯
        def dfs(depth, tmp, used):
            # 递归终止：当depth==n时，表示排列完成
            if depth == n:
                res.append(tmp.copy())
                return
                # 遍历节点
            for i in range(n):
                # 未使用的节点处理
                if not used[i]:
                    used[i] = True
                    tmp.append(nums[i])
                    # 递归工作
                    dfs(depth + 1, tmp, used)
                    # 状态重置
                    used[i] = False
                    tmp.pop()

        # 从第一层开始递归
        dfs(0, [], used)
        return res

# leetcode submit region end(Prohibit modification and deletion)

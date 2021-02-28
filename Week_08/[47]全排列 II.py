# 给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：nums = [1,1,2]
# 输出：
# [[1,1,2],
#  [1,2,1],
#  [2,1,1]]
#  
# 
#  示例 2： 
# 
#  
# 输入：nums = [1,2,3]
# 输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
#  
# 
#  
# 
#  提示： 
# 
#  
#  1 <= nums.length <= 8 
#  -10 <= nums[i] <= 10 
#  
#  Related Topics 回溯算法 
#  👍 566 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        nums.sort()
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
                    # 剪枝
                    if i > 0 and nums[i] == nums[i - 1] and used[i - 1] is True:
                        continue
                    # 正常处理
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

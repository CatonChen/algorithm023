# n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。 
# 
#  给你一个整数 n ，返回 n 皇后问题 不同的解决方案的数量。 
# 
#  
# 
#  
#  
#  示例 1： 
# 
#  
# 输入：n = 4
# 输出：2
# 解释：如上图所示，4 皇后问题存在两个不同的解法。
#  
# 
#  示例 2： 
# 
#  
# 输入：n = 1
# 输出：1
#  
# 
#  
# 
#  提示： 
# 
#  
#  1 <= n <= 9 
#  皇后彼此不能相互攻击，也就是说：任何两个皇后都不能处于同一条横行、纵行或斜线上。 
#  
#  
#  
#  Related Topics 回溯算法 
#  👍 242 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def totalNQueens(self, n: int) -> int:
        # 回溯
        def dfs(queens, xy_diff, xy_sum):
            nonlocal res
            p = len(queens)  # 算行数，行列相等
            if p == n:
                res += 1
                return
            # 遍历每列
            for q in range(n):
                # 剪枝 ，逆向思维，能放Q的位置必须同时满足不在列、撇、捺里
                if q not in queens and p - q not in xy_diff and p + q not in xy_sum:
                    # 递归操作
                    dfs(queens + [q], xy_diff + [p - q], xy_sum + [p + q])

        res = 0
        dfs([], [], [])
        return res

# leetcode submit region end(Prohibit modification and deletion)

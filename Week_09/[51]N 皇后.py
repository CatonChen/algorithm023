# n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。 
# 
#  给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案。 
# 
#  
#  
#  每一种解法包含一个不同的 n 皇后问题 的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：n = 4
# 输出：[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
# 解释：如上图所示，4 皇后问题存在两个不同的解法。
#  
# 
#  示例 2： 
# 
#  
# 输入：n = 1
# 输出：[["Q"]]
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
#  👍 766 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        # 回溯
        def dfs(queens, xy_diff, xy_sum):
            p = len(queens)  # 算行数，行列相等
            if p == n:
                res.append(queens)
                return
            # 遍历每列
            for q in range(n):
                # 剪枝 ，逆向思维，能放Q的位置必须同时满足不在列、撇、捺里
                if q not in queens and p - q not in xy_diff and p + q not in xy_sum:
                    # 递归操作
                    dfs(queens + [q], xy_diff + [p - q], xy_sum + [p + q])

        res = []
        dfs([], [], [])
        return [['.' * i + 'Q' + '.' * (n - i - 1) for i in sol] for sol in res]
# leetcode submit region end(Prohibit modification and deletion)

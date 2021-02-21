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
#  👍 748 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        # 递归
        def dfs(r):
            if r == n:  # 递归终止条件
                res.append([''.join(row) for row in b])
                return
            else:
                for c in range(n):
                    if isValid(r, c):
                        b[r][c] = 'Q'
                        dfs(r + 1)
                        # 回溯
                        b[r][c] = '.'

        # 剪枝
        def isValid(r, c):
            for i in range(r):
                for j in range(n):  # 这里不是c，是n，表示每列都要检查
                    if b[i][j] == 'Q' and (c == j or r + c == i + j or r - c == i - j):
                        return False
            return True

        b = [['.'] * n for _ in range(n)]
        res = []
        dfs(0)  # 从第一行开始，第一行下标为0
        return res

# leetcode submit region end(Prohibit modification and deletion)

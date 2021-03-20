# 给你一个 m x n 的矩阵 board ，由若干字符 'X' 和 'O' ，找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充
# 。
#  
#  
#  
# 
#  示例 1： 
# 
#  
# 输入：board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X"
# ,"X"]]
# 输出：[["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]
# 解释：被围绕的区间不会存在于边界上，换句话说，任何边界上的 'O' 都不会被填充为 'X'。 任何不在边界上，或不与边界上的 'O' 相连的 'O' 最终都
# 会被填充为 'X'。如果两个元素在水平或垂直方向相邻，则称它们是“相连”的。
#  
# 
#  示例 2： 
# 
#  
# 输入：board = [["X"]]
# 输出：[["X"]]
#  
# 
#  
# 
#  提示： 
# 
#  
#  m == board.length 
#  n == board[i].length 
#  1 <= m, n <= 200 
#  board[i][j] 为 'X' 或 'O' 
#  
#  
#  
#  Related Topics 深度优先搜索 广度优先搜索 并查集 
#  👍 495 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
# unionfind
class UnionFind:
    def __init__(self):
        self.father = {}

    def add(self, x):
        if x not in self.father:
            self.father[x] = None

    def find(self, x):
        root = x
        self.add(root)
        while self.father[root] is not None:
            root = self.father[root]
        # 路径压缩
        while x != root:
            o_father = self.father[x]
            self.father[x] = root
            x = o_father
        return root

    def merge(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x != root_y:
            self.father[root_x] = root_y

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)


class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        if not board:
            return None
        m, n = len(board), len(board[0])

        # 定义节点，用于将xy坐标转为一个节点
        def node(x, y):
            return x * n + y

        # dummy节点
        uf = UnionFind()
        dummy = m * n
        uf.add(dummy)
        # 遍历矩阵中的O
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O':
                    # 边界O
                    if i == 0 or j == 0 or i == m - 1 or j == n - 1:
                        uf.merge(node(i, j), dummy)
                    else:  # 非边界O
                        for mx, my in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                            if 0 <= mx < m and 0 <= my < n and board[mx][my] == 'O':
                                uf.merge(node(i, j), node(mx, my))
        # 遍历矩阵的O
        # 判断与dummy的连通性
        for i in range(m):
            for j in range(n):
                if uf.is_connected(dummy, node(i, j)):
                    board[i][j] = 'O'
                else:
                    board[i][j] = 'X'

# leetcode submit region end(Prohibit modification and deletion)

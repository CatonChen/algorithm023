# 给定一个二维的矩阵，包含 'X' 和 'O'（字母 O）。 
# 
#  找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。 
# 
#  示例: 
# 
#  X X X X
# X O O X
# X X O X
# X O X X
#  
# 
#  运行你的函数后，矩阵变为： 
# 
#  X X X X
# X X X X
# X X X X
# X O X X
#  
# 
#  解释: 
# 
#  被围绕的区间不会存在于边界上，换句话说，任何边界上的 'O' 都不会被填充为 'X'。 任何不在边界上，或不与边界上的 'O' 相连的 'O' 最终都会被
# 填充为 'X'。如果两个元素在水平或垂直方向相邻，则称它们是“相连”的。 
#  Related Topics 深度优先搜索 广度优先搜索 并查集 
#  👍 471 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
# 并查集模板
class UnionFind:
    def __init__(self):
        """
        记录每个节点的父节点
        """
        self.father = {}

    def find(self, x):
        """
        查找根节点
        路径压缩
        """
        root = x

        while self.father[root] is not None:
            root = self.father[root]

        # 路径压缩
        while x != root:
            original_father = self.father[x]
            self.father[x] = root
            x = original_father

        return root

    def merge(self, x, y):
        """
        合并两个节点
        """
        root_x, root_y = self.find(x), self.find(y)

        if root_x != root_y:
            self.father[root_x] = root_y

    def is_connected(self, x, y):
        """
        判断两节点是否相连
        """
        return self.find(x) == self.find(y)

    def add(self, x):
        """
        添加新节点
        """
        if x not in self.father:
            self.father[x] = None


class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        if not board:
            return None

        # 初始化矩阵行列
        m, n = len(board), len(board[0])

        # 定义node
        def node(x, y):
            return x * n + y

        # 初始化uf
        uf = UnionFind()
        # 初始化dummy
        dummy = m * n
        uf.add(dummy)
        # print(uf.father)

        # 遍历矩阵检查字母O
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O':
                    # 属于矩阵边界的O
                    if i == 0 or j == 0 or j == n - 1 or i == m - 1:
                        # print(node(i,j))
                        uf.add(node(i, j))
                        # 将边界O与dummy连通
                        uf.merge(node(i, j), dummy)
                    else:
                        # 非边界的O
                        for mx, my in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                            if 0 <= mx < m and 0 <= my < n and board[mx][my] == 'O':
                                uf.add(node(i, j))
                                uf.add(node(mx, my))
                                # 将相互连通的O进行合并
                                # 与边界O连通的O，最终会连通至dummy
                                uf.merge(node(mx, my), node(i, j))
        #
        # print(uf.father)
        # 遍历矩阵
        for i in range(m):
            for j in range(n):
                uf.add(node(i, j))
                # 与dummy是同一个祖先
                if uf.is_connected(dummy, node(i, j)):
                    board[i][j] = 'O'
                else:
                    board[i][j] = 'X'
        # print(uf.father)
# leetcode submit region end(Prohibit modification and deletion)

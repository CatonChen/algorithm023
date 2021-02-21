# 
#  
#  有 n 个城市，其中一些彼此相连，另一些没有相连。如果城市 a 与城市 b 直接相连，且城市 b 与城市 c 直接相连，那么城市 a 与城市 c 间接相连
# 。 
# 
#  省份 是一组直接或间接相连的城市，组内不含其他没有相连的城市。 
# 
#  给你一个 n x n 的矩阵 isConnected ，其中 isConnected[i][j] = 1 表示第 i 个城市和第 j 个城市直接相连，而 
# isConnected[i][j] = 0 表示二者不直接相连。 
# 
#  返回矩阵中 省份 的数量。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：isConnected = [[1,1,0],[1,1,0],[0,0,1]]
# 输出：2
#  
# 
#  示例 2： 
# 
#  
# 输入：isConnected = [[1,0,0],[0,1,0],[0,0,1]]
# 输出：3
#  
# 
#  
# 
#  提示： 
# 
#  
#  1 <= n <= 200 
#  n == isConnected.length 
#  n == isConnected[i].length 
#  isConnected[i][j] 为 1 或 0 
#  isConnected[i][i] == 1 
#  isConnected[i][j] == isConnected[j][i] 
#  
#  
#  
#  Related Topics 深度优先搜索 并查集 
#  👍 491 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class unionFind:
    def __init__(self):
        """
        记录每个节点的父节点
        """
        self.father = {}
        # 记录集合数量
        self.num_of_set = 0

    def add(self, x):
        """
        添加新节点
        """
        if x not in self.father:
            self.father[x] = None
            # 集合数量+1
            self.num_of_set += 1

    def find(self, x):
        """
        查找节点
        """
        root = x
        while self.father[root] != None:
            root = self.father[root]

        """
        路径压缩
        """
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
            # 集合数量-1
            self.num_of_set -= 1

    # def isconnected(self,x,y):
    #     """
    #     判断两个节点是否相连
    #     """
    #     return self.find(x)==self.find(y)


class Solution:
    def findCircleNum(self, M: List[List[int]]) -> int:
        # 初始化并查集
        uf = unionFind()
        for i in range(len(M)):
            # print('i: ' + str(i))
            uf.add(i)  # 添加节点
            for j in range(i):
                # print('j: ' + str(j))
                # print('i: ' + str(i) + ' j: ' + str(j) + ' M[i][j]: ' + str(M[i][j]))
                if M[i][j] == 1:  # M[i][j]有效
                    uf.merge(i, j)

        return uf.num_of_set

# leetcode submit region end(Prohibit modification and deletion)
